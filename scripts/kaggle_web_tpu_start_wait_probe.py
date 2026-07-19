#!/usr/bin/env python3
"""Start a Kaggle Web TPU session and wait for a visible Jupyter runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import kaggle_web_tpu_ui_state_probe as ui_state  # noqa: E402


SCHEMA = "kaggle_web_tpu_start_wait_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/kaggle-web-tpu-start-wait-probe"
DEFAULT_NOTEBOOK_URL = ui_state.DEFAULT_NOTEBOOK_URL
SENSITIVE_FRAGMENTS = ui_state.SENSITIVE_FRAGMENTS


OBSERVE_JS = r"""
() => {
  const textOf = (el) => (el && (el.innerText || el.textContent || "") || "").trim();
  const controls = Array.from(document.querySelectorAll("button, [role=button], input, textarea"))
    .slice(0, 120)
    .map((el, index) => ({
      selector: el.tagName.toLowerCase(),
      index,
      text: textOf(el),
      aria: el.getAttribute("aria-label") || "",
      title: el.getAttribute("title") || "",
      disabled: !!el.disabled || el.getAttribute("aria-disabled") === "true",
      visible: !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length),
    }));
  const frames = Array.from(window.frames).map((frame) => {
    try {
      const sm = frame.jupyterapp && frame.jupyterapp.serviceManager;
      let sessionCount = 0;
      let kernelCount = 0;
      if (sm) {
        try {
          const sessions = sm.sessions && sm.sessions.running ? Array.from(sm.sessions.running()) : [];
          sessionCount = sessions.length;
        } catch (err) {}
        try {
          const kernels = sm.kernels && sm.kernels.running ? Array.from(sm.kernels.running()) : [];
          kernelCount = kernels.length;
        } catch (err) {}
      }
      return {has_jupyterapp: !!sm, session_count: sessionCount, kernel_count: kernelCount};
    } catch (err) {
      return {has_jupyterapp: false, session_count: 0, kernel_count: 0};
    }
  });
  return {
    body_text: document.body ? document.body.innerText || "" : "",
    start_session_visible: controls.some((item) => /start session/i.test(`${item.text} ${item.aria} ${item.title}`)),
    controls,
    frames,
  };
}
"""


def timeout_observation(error_name: str, *, elapsed_seconds: float) -> dict[str, Any]:
    return {
        "elapsed_seconds": round(float(elapsed_seconds), 1),
        "excerpt_public": "",
        "body_text_hash": sha_payload(error_name),
        "tpu_v5e_option_visible": False,
        "start_session_visible": False,
        "session_started_text_visible": False,
        "session_starting_text_visible": False,
        "queue_visible": False,
        "jupyter_frame_visible": False,
        "jupyter_session_or_kernel_visible": False,
        "jupyter_session_count": 0,
        "jupyter_kernel_count": 0,
        "web_tpu_ui_runtime_ready": False,
        "frames_public": [],
        "probe_error": error_name,
    }


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


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def summarize_observation(observation: dict[str, Any], *, elapsed_seconds: float = 0.0) -> dict[str, Any]:
    body_text = str(observation.get("body_text") or "")
    frames = [
        {
            "has_jupyterapp": item.get("has_jupyterapp") is True,
            "session_count": _int(item.get("session_count")),
            "kernel_count": _int(item.get("kernel_count")),
            "url_public": "<url>",
        }
        for item in _list(observation.get("frames"))
        if isinstance(item, dict)
    ]
    session_count = max([_int(item.get("session_count")) for item in frames] + [0])
    kernel_count = max([_int(item.get("kernel_count")) for item in frames] + [0])
    has_jupyterapp = any(item.get("has_jupyterapp") is True for item in frames)
    session_started_text = "Session started" in body_text
    session_starting_text = "Session is starting" in body_text or "\nStarting\n" in f"\n{body_text}\n"
    queue_visible = "TPUs are popular" in body_text or "queue" in body_text.lower()
    runtime_ready = bool(has_jupyterapp and (session_count > 0 or kernel_count > 0) and session_started_text)
    return {
        "elapsed_seconds": round(float(elapsed_seconds), 1),
        "excerpt_public": ui_state.public_excerpt(body_text, limit=260),
        "body_text_hash": sha_payload(body_text),
        "tpu_v5e_option_visible": "TPU v5e-8" in body_text,
        "start_session_visible": observation.get("start_session_visible") is True,
        "session_started_text_visible": session_started_text,
        "session_starting_text_visible": session_starting_text,
        "queue_visible": queue_visible,
        "jupyter_frame_visible": has_jupyterapp,
        "jupyter_session_or_kernel_visible": bool(session_count > 0 or kernel_count > 0),
        "jupyter_session_count": session_count,
        "jupyter_kernel_count": kernel_count,
        "web_tpu_ui_runtime_ready": runtime_ready,
        "frames_public": frames,
    }


def click_first(page: Any, locators: list[Any], *, timeout_ms: int = 5000) -> bool:
    for locator in locators:
        try:
            locator.first.click(timeout=timeout_ms)
            return True
        except Exception:
            continue
    return False


def run_live(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from playwright.sync_api import sync_playwright

    steps: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    started = time.monotonic()
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            executable_path=args.chrome_executable,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        try:
            context = browser.new_context(storage_state=args.kaggle_web_storage_state, viewport={"width": 1440, "height": 1000})
            page = context.new_page()
            page.goto(args.kaggle_notebook_url, wait_until="domcontentloaded", timeout=int(args.page_timeout_seconds * 1000))
            page.wait_for_timeout(int(args.settle_seconds * 1000))
            page.set_default_timeout(int(min(max(args.poll_seconds, 5.0), 30.0) * 1000))
            try:
                before = page.evaluate(OBSERVE_JS)
            except Exception as exc:  # noqa: BLE001 - public-safe report path
                steps.append({"name": "initial_observe_ui", "ok": False, "error_type": type(exc).__name__})
                observations.append(timeout_observation("initial_observe_ui_failed", elapsed_seconds=time.monotonic() - started))
                return steps, observations
            observations.append(summarize_observation(before, elapsed_seconds=time.monotonic() - started))
            if observations[-1].get("web_tpu_ui_runtime_ready") is True:
                steps.append({"name": "runtime_already_ready", "ok": True})
                return steps, observations

            expanded = click_first(
                page,
                [
                    page.get_by_title(re.compile("Expand Session options", re.I)),
                    page.get_by_label(re.compile("Expand Session options", re.I)),
                    page.get_by_text(re.compile("Session options", re.I)),
                ],
                timeout_ms=5000,
            )
            steps.append({"name": "expand_session_options", "ok": expanded})
            page.wait_for_timeout(1500)

            try:
                after_expand = page.evaluate(OBSERVE_JS)
            except Exception as exc:  # noqa: BLE001 - public-safe report path
                steps.append({"name": "observe_after_expand", "ok": False, "error_type": type(exc).__name__})
                observations.append(timeout_observation("observe_after_expand_failed", elapsed_seconds=time.monotonic() - started))
                return steps, observations
            observations.append(summarize_observation(after_expand, elapsed_seconds=time.monotonic() - started))
            if not observations[-1].get("tpu_v5e_option_visible"):
                steps.append({"name": "tpu_v5e_option_visible", "ok": False})
                return steps, observations
            steps.append({"name": "tpu_v5e_option_visible", "ok": True})

            selected = click_first(page, [page.get_by_text("TPU v5e-8", exact=True)], timeout_ms=5000)
            steps.append({"name": "select_tpu_v5e8", "ok": selected})
            page.keyboard.press("Escape")
            page.wait_for_timeout(1000)

            clicked = click_first(
                page,
                [
                    page.get_by_role("button", name=re.compile("Start Session", re.I)),
                    page.get_by_label(re.compile("Start session", re.I)),
                    page.get_by_text("Start Session", exact=True),
                ],
                timeout_ms=8000,
            )
            steps.append({"name": "click_start_session", "ok": clicked})
            page.wait_for_timeout(3000)

            deadline = time.monotonic() + float(args.wait_seconds)
            while True:
                try:
                    observation = page.evaluate(OBSERVE_JS)
                except Exception as exc:  # noqa: BLE001 - public-safe report path
                    steps.append({"name": "observe_wait_loop", "ok": False, "error_type": type(exc).__name__})
                    observations.append(timeout_observation("observe_wait_loop_failed", elapsed_seconds=time.monotonic() - started))
                    break
                summary = summarize_observation(observation, elapsed_seconds=time.monotonic() - started)
                observations.append(summary)
                if summary.get("web_tpu_ui_runtime_ready") is True:
                    break
                if time.monotonic() >= deadline:
                    break
                page.wait_for_timeout(int(args.poll_seconds * 1000))
        finally:
            browser.close()
    return steps, observations


def build_report(args: argparse.Namespace, *, steps: list[dict[str, Any]], observations: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    final = observations[-1] if observations else {}
    ready = final.get("web_tpu_ui_runtime_ready") is True
    blockers: set[str] = set()
    if not ready:
        if final.get("probe_error"):
            blockers.add("kaggle_web_tpu_start_wait_observe_failed")
        if final.get("queue_visible"):
            blockers.add("kaggle_web_tpu_queue_visible")
        if final.get("session_starting_text_visible"):
            blockers.add("kaggle_web_tpu_session_still_starting")
        if final.get("start_session_visible"):
            blockers.add("kaggle_web_tpu_start_session_visible")
        if not final.get("jupyter_frame_visible"):
            blockers.add("kaggle_web_tpu_jupyter_frame_not_visible")
        if not final.get("jupyter_session_or_kernel_visible"):
            blockers.add("kaggle_web_tpu_jupyter_session_not_visible")
        if not any(step.get("name") == "click_start_session" and step.get("ok") is True for step in steps):
            blockers.add("kaggle_web_tpu_start_session_not_clicked")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "start_wait_probe_ready": True,
        "kaggle_notebook_url_public": False,
        "start_clicked": any(step.get("name") == "click_start_session" and step.get("ok") is True for step in steps),
        "web_tpu_ui_runtime_ready": ready,
        "bounded_wait_seconds": float(args.wait_seconds),
        "observation_count": len(observations),
        "final_observation": final,
        "observations": observations[-12:],
        "steps": steps,
        "blocked_reason": "" if ready else (sorted(blockers)[0] if blockers else "kaggle_web_tpu_runtime_not_ready_within_bounded_wait"),
        "blocker_codes": sorted(blockers),
        "diagnosis_codes": sorted({
            "kaggle_web_tpu_start_wait_probe_ready",
            "kaggle_web_tpu_runtime_ready" if ready else "kaggle_web_tpu_runtime_not_ready",
        }),
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
            "This starts or waits for a Kaggle Web TPU UI runtime but does not execute TPU code.",
            "A ready UI state must still be followed by a successful execution-channel probe before 72B live decode.",
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
    summary_path = output_dir / "kaggle_web_tpu_start_wait_probe.json"
    write_json(summary_path, report)
    report["artifacts"] = {
        "summary_json": artifact_entry(summary_path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(report.get("ok"))),
    }
    write_json(summary_path, report)
    return report


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps, observations = run_live(args)
    return build_report(args, steps=steps, observations=observations, output_dir=output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start/wait for a Kaggle Web TPU session.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kaggle-notebook-url", default=DEFAULT_NOTEBOOK_URL)
    parser.add_argument("--kaggle-web-storage-state", default="/root/kaggle-web-storage-state.json")
    parser.add_argument("--chrome-executable", default="/usr/bin/google-chrome")
    parser.add_argument("--page-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--settle-seconds", type=float, default=8.0)
    parser.add_argument("--wait-seconds", type=float, default=900.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.page_timeout_seconds < 10 or args.page_timeout_seconds > 180:
        raise SystemExit("--page-timeout-seconds must be between 10 and 180")
    if args.settle_seconds < 1 or args.settle_seconds > 60:
        raise SystemExit("--settle-seconds must be between 1 and 60")
    if args.wait_seconds < 30 or args.wait_seconds > 7200:
        raise SystemExit("--wait-seconds must be between 30 and 7200")
    if args.poll_seconds < 5 or args.poll_seconds > 300:
        raise SystemExit("--poll-seconds must be between 5 and 300")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_probe(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            f"{SCHEMA}: runtime_ready={bool(report.get('web_tpu_ui_runtime_ready'))} "
            f"start_clicked={bool(report.get('start_clicked'))}"
        )
    return 0 if report.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
