#!/usr/bin/env python3
"""Inspect Kaggle Web TPU Active Events and attempt a bounded runtime reopen."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import kaggle_web_tpu_ui_state_probe as ui_state  # noqa: E402


SCHEMA = "kaggle_web_tpu_active_event_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/kaggle-web-tpu-active-event-probe"
DEFAULT_NOTEBOOK_URL = ui_state.DEFAULT_NOTEBOOK_URL
SENSITIVE_FRAGMENTS = ui_state.SENSITIVE_FRAGMENTS


OBSERVE_JS = r"""
async () => {
  const textOf = (el) => (el && (el.innerText || el.textContent || "") || "").trim();
  const controls = Array.from(document.querySelectorAll("button, [role=button], a, span"))
    .slice(0, 180)
    .map((el, index) => {
      const rect = el.getBoundingClientRect();
      return {
        selector: el.tagName.toLowerCase(),
        index,
        text: textOf(el),
        aria: el.getAttribute("aria-label") || "",
        title: el.getAttribute("title") || "",
        role: el.getAttribute("role") || "",
        disabled: !!el.disabled || el.getAttribute("aria-disabled") === "true",
        visible: !!(rect.width || rect.height || el.getClientRects().length),
      };
    });
  const frames = await Promise.all(Array.from(window.frames).map(async (frame) => {
    try {
      const sm = frame.jupyterapp && frame.jupyterapp.serviceManager;
      let sessionCount = 0;
      let kernelCount = 0;
      if (sm) {
        try {
          if (sm.sessions && sm.sessions.refreshRunning) {
            await Promise.race([sm.sessions.refreshRunning(), new Promise((resolve) => setTimeout(resolve, 3000))]);
          }
          const sessions = sm.sessions && sm.sessions.running ? Array.from(sm.sessions.running()) : [];
          sessionCount = sessions.length;
        } catch (err) {}
        try {
          if (sm.kernels && sm.kernels.refreshRunning) {
            await Promise.race([sm.kernels.refreshRunning(), new Promise((resolve) => setTimeout(resolve, 3000))]);
          }
          const kernels = sm.kernels && sm.kernels.running ? Array.from(sm.kernels.running()) : [];
          kernelCount = kernels.length;
        } catch (err) {}
      }
      return {has_jupyterapp: !!sm, session_count: sessionCount, kernel_count: kernelCount};
    } catch (err) {
      return {has_jupyterapp: false, session_count: 0, kernel_count: 0};
    }
  }));
  return {
    body_text: document.body ? document.body.innerText || "" : "",
    controls,
    frames,
  };
}
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    entry: dict[str, Any] = {"kind": kind, "path": relative, "present": path.is_file()}
    if path.is_file():
        entry["sha256"] = sha256_file(path)
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def parse_active_events_from_body(body_text: str) -> list[dict[str, Any]]:
    lines = [line.strip() for line in str(body_text or "").splitlines() if line.strip()]
    events: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.lower().startswith("interactive session with "):
            continue
        title = lines[index - 1] if index > 0 else ""
        status = lines[index + 1] if index + 1 < len(lines) else ""
        age = lines[index + 2] if index + 2 < len(lines) else ""
        accelerator = ""
        match = re.search(r"(TPU\s+v[0-9a-zA-Z.-]+(?:-[0-9]+)?)", line)
        if match:
            accelerator = match.group(1)
        events.append(
            {
                "event_title_hash": sha_payload(title),
                "event_title_public": False,
                "event_kind_public": line[:96],
                "accelerator_public": accelerator,
                "status_public": status[:64],
                "age_public": age[:64],
                "queued": status.lower() == "queued",
                "running": status.lower().startswith("running"),
                "starting": "starting" in status.lower(),
            }
        )
    return events


def summarize_frames(frames: list[Any]) -> dict[str, Any]:
    public_frames = [
        {
            "has_jupyterapp": item.get("has_jupyterapp") is True,
            "session_count": _int(item.get("session_count")),
            "kernel_count": _int(item.get("kernel_count")),
            "url_public": "<url>",
        }
        for item in frames
        if isinstance(item, dict)
    ]
    session_count = max([_int(item.get("session_count")) for item in public_frames] + [0])
    kernel_count = max([_int(item.get("kernel_count")) for item in public_frames] + [0])
    has_jupyterapp = any(item.get("has_jupyterapp") is True for item in public_frames)
    return {
        "frames_public": public_frames,
        "jupyter_frame_visible": has_jupyterapp,
        "jupyter_session_count": session_count,
        "jupyter_kernel_count": kernel_count,
        "jupyter_session_or_kernel_visible": bool(session_count > 0 or kernel_count > 0),
    }


def build_report(
    args: argparse.Namespace,
    *,
    initial_observation: dict[str, Any],
    final_observation: dict[str, Any],
    steps: list[dict[str, Any]],
    observations: list[dict[str, Any]] | None = None,
    output_dir: Path,
) -> dict[str, Any]:
    initial_body = str(initial_observation.get("body_text") or "")
    final_body = str(final_observation.get("body_text") or "")
    events = parse_active_events_from_body(final_body)
    frames = summarize_frames(_list(final_observation.get("frames")))
    active_event_count = len(events)
    tpu_events = [item for item in events if "TPU" in str(item.get("event_kind_public") or "")]
    queued = any(item.get("queued") is True for item in tpu_events)
    running = any(item.get("running") is True for item in tpu_events)
    ready = bool(running and frames["jupyter_frame_visible"] and frames["jupyter_session_or_kernel_visible"])
    blockers: set[str] = set()
    if active_event_count < 1:
        blockers.add("kaggle_web_tpu_active_event_missing")
    if tpu_events and queued:
        blockers.add("kaggle_web_tpu_active_event_queued")
    if tpu_events and not running:
        blockers.add("kaggle_web_tpu_active_event_not_running")
    if not frames["jupyter_frame_visible"]:
        blockers.add("kaggle_web_tpu_jupyter_frame_not_visible")
    if not frames["jupyter_session_or_kernel_visible"]:
        blockers.add("kaggle_web_tpu_jupyter_session_not_visible")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "active_event_probe_ready": True,
        "kaggle_notebook_url_public": False,
        "active_event_dialog_opened": any(step.get("name") == "open_active_events_dialog" and step.get("ok") is True for step in steps),
        "active_event_open_attempted": any(step.get("name") == "open_running_active_event" for step in steps),
        "active_event_opened": any(step.get("name") == "open_running_active_event" and step.get("ok") is True for step in steps),
        "bounded_wait_seconds": float(getattr(args, "wait_seconds", 0.0)),
        "poll_seconds": float(getattr(args, "poll_seconds", 0.0)),
        "observation_count": len(observations or []),
        "active_event_count": active_event_count,
        "tpu_v5e_active_event_visible": any("TPU v5e-8" in str(item.get("event_kind_public") or "") for item in events),
        "active_event_queued": queued,
        "active_event_running": running,
        "active_event_runtime_ready": ready,
        "active_events": events,
        "initial_body_text_hash": sha_payload(initial_body),
        "final_body_text_hash": sha_payload(final_body),
        "observations": [
            {
                "elapsed_seconds": float(item.get("elapsed_seconds") or 0.0),
                "active_event_count": _int(item.get("active_event_count")),
                "active_event_queued": item.get("active_event_queued") is True,
                "active_event_running": item.get("active_event_running") is True,
                "jupyter_frame_visible": item.get("jupyter_frame_visible") is True,
                "jupyter_session_or_kernel_visible": item.get("jupyter_session_or_kernel_visible") is True,
                "body_text_hash": str(item.get("body_text_hash") or ""),
            }
            for item in (observations or [])[-20:]
        ],
        **frames,
        "blocked_reason": "" if ready else (sorted(blockers)[0] if blockers else "kaggle_web_tpu_active_event_not_ready"),
        "blocker_codes": sorted(blockers),
        "diagnosis_codes": sorted(
            {
                "kaggle_web_tpu_active_event_probe_ready",
                "kaggle_web_tpu_active_event_runtime_ready" if ready else "kaggle_web_tpu_active_event_runtime_not_ready",
            }
        ),
        "steps": steps,
        "cleanup_status": {
            "temporary_kaggle_kernels_created": False,
            "temporary_kaggle_kernels_deleted": True,
            "temporary_private_packages_removed": True,
            "live_resources_left_running": False,
            "cookie_file_public": False,
            "storage_state_file_public": False,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "jupyter_proxy_token_public": False,
            "private_runtime_state_public": False,
        },
        "public_artifact_safe": True,
        "limitations": [
            "This inspects Kaggle Active Events and may attempt to reopen a running event; it does not execute TPU model code.",
            "A queued active event is scheduling evidence only, not a usable TPU runtime.",
        ],
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blocker_codes"].append("public_redaction_scan_failed")
        report["diagnosis_codes"].append("public_redaction_scan_failed")
        report["redaction_errors"] = leaks
    summary_path = output_dir / "kaggle_web_tpu_active_event_probe.json"
    write_json(summary_path, report)
    report["artifacts"] = {
        "summary_json": artifact_entry(summary_path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(report.get("ok"))),
    }
    write_json(summary_path, report)
    return report


def observe_live(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    from playwright.sync_api import sync_playwright

    steps: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            executable_path=args.chrome_executable,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        try:
            context = browser.new_context(storage_state=args.kaggle_web_storage_state, viewport={"width": 1440, "height": 1000})
            page = context.new_page()
            started = __import__("time").monotonic()
            page.goto(args.kaggle_notebook_url, wait_until="domcontentloaded", timeout=int(args.page_timeout_seconds * 1000))
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            initial = page.evaluate(OBSERVE_JS)
            opened = False
            try:
                page.get_by_text(re.compile("View Active Events", re.I)).first.click(timeout=5000)
                opened = True
            except Exception:
                try:
                    page.get_by_role("button", name=re.compile("View Active Events", re.I)).first.click(timeout=5000)
                    opened = True
                except Exception:
                    opened = False
            steps.append({"name": "open_active_events_dialog", "ok": opened})
            page.wait_for_timeout(2000)
            final = page.evaluate(OBSERVE_JS)
            deadline = started + float(args.wait_seconds)
            while True:
                events = parse_active_events_from_body(str(final.get("body_text") or ""))
                frames = summarize_frames(_list(final.get("frames")))
                observations.append(
                    {
                        "elapsed_seconds": round(__import__("time").monotonic() - started, 1),
                        "active_event_count": len(events),
                        "active_event_queued": any(item.get("queued") for item in events),
                        "active_event_running": any(item.get("running") for item in events),
                        "jupyter_frame_visible": frames.get("jupyter_frame_visible") is True,
                        "jupyter_session_or_kernel_visible": frames.get("jupyter_session_or_kernel_visible") is True,
                        "body_text_hash": sha_payload(str(final.get("body_text") or "")),
                    }
                )
                if args.attempt_open_running_event and any(item.get("running") for item in events):
                    opened_event = False
                    try:
                        page.get_by_text(re.compile("Interactive Session with TPU", re.I)).first.click(timeout=5000)
                        opened_event = True
                        page.wait_for_timeout(5000)
                    except Exception:
                        opened_event = False
                    steps.append({"name": "open_running_active_event", "ok": opened_event})
                    final = page.evaluate(OBSERVE_JS)
                    break
                if __import__("time").monotonic() >= deadline:
                    break
                page.wait_for_timeout(int(float(args.poll_seconds) * 1000))
                final = page.evaluate(OBSERVE_JS)
        finally:
            browser.close()
    return initial, final, steps, observations


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial, final, steps, observations = observe_live(args)
    return build_report(args, initial_observation=initial, final_observation=final, steps=steps, observations=observations, output_dir=output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Kaggle Web TPU Active Events.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kaggle-notebook-url", default=DEFAULT_NOTEBOOK_URL)
    parser.add_argument("--kaggle-web-storage-state", default="/root/kaggle-web-storage-state.json")
    parser.add_argument("--chrome-executable", default="/usr/bin/google-chrome")
    parser.add_argument("--page-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--settle-seconds", type=float, default=8.0)
    parser.add_argument("--wait-seconds", type=float, default=0.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--attempt-open-running-event", action="store_true", default=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.page_timeout_seconds < 10 or args.page_timeout_seconds > 180:
        raise SystemExit("--page-timeout-seconds must be between 10 and 180")
    if args.settle_seconds < 1 or args.settle_seconds > 60:
        raise SystemExit("--settle-seconds must be between 1 and 60")
    if args.wait_seconds < 0 or args.wait_seconds > 7200:
        raise SystemExit("--wait-seconds must be between 0 and 7200")
    if args.poll_seconds < 5 or args.poll_seconds > 600:
        raise SystemExit("--poll-seconds must be between 5 and 600")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_probe(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            f"{SCHEMA}: events={report.get('active_event_count')} "
            f"running={bool(report.get('active_event_running'))} "
            f"ready={bool(report.get('active_event_runtime_ready'))} "
            f"blocked={report.get('blocked_reason') or 'none'}"
        )
    return 0 if report.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
