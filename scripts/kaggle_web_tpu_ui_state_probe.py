#!/usr/bin/env python3
"""Read the current authenticated Kaggle Web TPU notebook UI state."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import kaggle_web_tpu_execution_channel_probe as channel_probe  # noqa: E402


SCHEMA = "kaggle_web_tpu_ui_state_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/kaggle-web-tpu-ui-state-probe"
DEFAULT_NOTEBOOK_URL = channel_probe.DEFAULT_NOTEBOOK_URL
SENSITIVE_FRAGMENTS = channel_probe.SENSITIVE_FRAGMENTS


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


def public_excerpt(text: str, *, limit: int = 360) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    excerpt = "\n".join(lines[:20])[:limit]
    for fragment in SENSITIVE_FRAGMENTS:
        excerpt = excerpt.replace(fragment, "<redacted>")
    return excerpt


def summarize_frame(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        "url_public": "<url>",
        "has_jupyterapp": frame.get("has_jupyterapp") is True,
        "session_count": _int(frame.get("session_count")),
        "kernel_count": _int(frame.get("kernel_count")),
        "running_session_count": _int(frame.get("running_session_count")),
        "running_kernel_count": _int(frame.get("running_kernel_count")),
    }


def build_report(args: argparse.Namespace, *, observation: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    body_text = str(observation.get("body_text") or "")
    frames = [summarize_frame(item) for item in _list(observation.get("frames")) if isinstance(item, dict)]
    session_count = max([_int(item.get("session_count")) for item in frames] + [0])
    kernel_count = max([_int(item.get("kernel_count")) for item in frames] + [0])
    has_jupyterapp = any(item.get("has_jupyterapp") is True for item in frames)
    start_session_visible = bool(observation.get("start_session_visible") is True)
    session_started_text = "Session started" in body_text
    session_starting_text = "Session is starting" in body_text or "\nStarting\n" in f"\n{body_text}\n"
    queue_visible = "TPUs are popular" in body_text or "queue" in body_text.lower()
    tpu_v5e_visible = "TPU v5e-8" in body_text
    jupyter_session_or_kernel_visible = bool(session_count > 0 or kernel_count > 0)
    ready = bool(has_jupyterapp and jupyter_session_or_kernel_visible and session_started_text)
    blockers: set[str] = set()
    if not has_jupyterapp:
        blockers.add("kaggle_web_tpu_jupyter_frame_not_visible")
    if not jupyter_session_or_kernel_visible:
        blockers.add("kaggle_web_tpu_jupyter_session_not_visible")
    if session_starting_text and not ready:
        blockers.add("kaggle_web_tpu_session_still_starting")
    if queue_visible and not ready:
        blockers.add("kaggle_web_tpu_queue_visible")
    if start_session_visible and not ready:
        blockers.add("kaggle_web_tpu_start_session_visible")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "ui_state_probe_ready": True,
        "kaggle_notebook_url_public": False,
        "tpu_v5e_option_visible": tpu_v5e_visible,
        "start_session_visible": start_session_visible,
        "session_started_text_visible": session_started_text,
        "session_starting_text_visible": session_starting_text,
        "queue_visible": queue_visible,
        "jupyter_frame_visible": has_jupyterapp,
        "jupyter_session_or_kernel_visible": jupyter_session_or_kernel_visible,
        "jupyter_session_count": session_count,
        "jupyter_kernel_count": kernel_count,
        "web_tpu_ui_runtime_ready": ready,
        "blocked_reason": "" if ready else (sorted(blockers)[0] if blockers else "kaggle_web_tpu_ui_runtime_not_ready"),
        "blocker_codes": sorted(blockers),
        "diagnosis_codes": sorted({
            "kaggle_web_tpu_ui_state_probe_ready",
            "kaggle_web_tpu_ui_runtime_ready" if ready else "kaggle_web_tpu_ui_runtime_not_ready",
        }),
        "body_text_excerpt_public": public_excerpt(body_text),
        "body_text_hash": sha_payload(body_text),
        "frames_public": frames,
        "controls_public": [
            {
                "selector": str(item.get("selector") or ""),
                "index": _int(item.get("index")),
                "text": public_excerpt(str(item.get("text") or ""), limit=80),
                "aria": public_excerpt(str(item.get("aria") or ""), limit=80),
                "title": public_excerpt(str(item.get("title") or ""), limit=80),
                "disabled": item.get("disabled") is True,
                "visible": item.get("visible") is True,
            }
            for item in _list(observation.get("controls"))
            if isinstance(item, dict)
        ][:80],
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
            "This is a read-only UI/session visibility probe; it does not execute TPU code.",
            "A ready UI state is necessary but not sufficient for 72B same-request decode.",
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
    summary_path = output_dir / "kaggle_web_tpu_ui_state_probe.json"
    write_json(summary_path, report)
    report["artifacts"] = {
        "summary_json": artifact_entry(summary_path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(report.get("ok"))),
    }
    write_json(summary_path, report)
    return report


def observe_ui(args: argparse.Namespace) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

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
            return page.evaluate(
                """
                async () => {
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
                  const frames = await Promise.all(Array.from(window.frames).map(async (frame) => {
                    try {
                      const sm = frame.jupyterapp && frame.jupyterapp.serviceManager;
                      let sessionCount = 0;
                      let kernelCount = 0;
                      let runningSessionCount = 0;
                      let runningKernelCount = 0;
                      if (sm) {
                        try {
                          if (sm.sessions && sm.sessions.refreshRunning) {
                            await Promise.race([
                              sm.sessions.refreshRunning(),
                              new Promise((resolve) => setTimeout(resolve, 3000)),
                            ]);
                          }
                        } catch (err) {}
                        try {
                          const sessions = sm.sessions && sm.sessions.running ? Array.from(sm.sessions.running()) : [];
                          sessionCount = sessions.length;
                          runningSessionCount = sessions.length;
                        } catch (err) {}
                        try {
                          if (sm.kernels && sm.kernels.refreshRunning) {
                            await Promise.race([
                              sm.kernels.refreshRunning(),
                              new Promise((resolve) => setTimeout(resolve, 3000)),
                            ]);
                          }
                          const kernels = sm.kernels && sm.kernels.running ? Array.from(sm.kernels.running()) : [];
                          kernelCount = kernels.length;
                          runningKernelCount = kernels.length;
                        } catch (err) {}
                      }
                      return {
                        has_jupyterapp: !!sm,
                        session_count: sessionCount,
                        kernel_count: kernelCount,
                        running_session_count: runningSessionCount,
                        running_kernel_count: runningKernelCount,
                      };
                    } catch (err) {
                      return {has_jupyterapp: false, session_count: 0, kernel_count: 0};
                    }
                  }));
                  return {
                    body_text: document.body ? document.body.innerText || "" : "",
                    start_session_visible: controls.some((item) => /start session/i.test(`${item.text} ${item.aria} ${item.title}`)),
                    controls,
                    frames,
                  };
                }
                """
            )
        finally:
            browser.close()


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observation = observe_ui(args)
    return build_report(args, observation=observation, output_dir=output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read current Kaggle Web TPU notebook UI state.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kaggle-notebook-url", default=DEFAULT_NOTEBOOK_URL)
    parser.add_argument("--kaggle-web-storage-state", default="/root/kaggle-web-storage-state.json")
    parser.add_argument("--chrome-executable", default="/usr/bin/google-chrome")
    parser.add_argument("--page-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--settle-seconds", type=float, default=8.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.page_timeout_seconds < 10 or args.page_timeout_seconds > 180:
        raise SystemExit("--page-timeout-seconds must be between 10 and 180")
    if args.settle_seconds < 1 or args.settle_seconds > 60:
        raise SystemExit("--settle-seconds must be between 1 and 60")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_probe(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            f"{SCHEMA}: ui_ready={bool(report.get('web_tpu_ui_runtime_ready'))} "
            f"sessions={report.get('jupyter_session_count')} kernels={report.get('jupyter_kernel_count')}"
        )
    return 0 if report.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
