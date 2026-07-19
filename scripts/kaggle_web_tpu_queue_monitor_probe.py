#!/usr/bin/env python3
"""Monitor Kaggle Web TPU queue/runtime state with public-safe output.

This probe targets the UI path because Kaggle does not expose a stable public
CLI/API surface for interactive TPU allocation. It can optionally click Start
Session for TPU v5e-8, then records queue-position changes, Active Event state,
and Jupyter runtime readiness.
"""

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

from scripts import kaggle_web_tpu_active_event_probe as active_event_probe  # noqa: E402
from scripts import kaggle_web_tpu_ui_state_probe as ui_state  # noqa: E402


SCHEMA = "kaggle_web_tpu_queue_monitor_probe_v1"
LIVE_STATUS_SCHEMA = "kaggle_web_tpu_queue_monitor_live_status_v1"
DEFAULT_OUTPUT_DIR = "dist/kaggle-web-tpu-queue-monitor-probe"
DEFAULT_NOTEBOOK_URL = ui_state.DEFAULT_NOTEBOOK_URL
SENSITIVE_FRAGMENTS = ui_state.SENSITIVE_FRAGMENTS


OBSERVE_JS = r"""
async () => {
  const textOf = (el) => (el && (el.innerText || el.textContent || "") || "").trim();
  const controls = Array.from(document.querySelectorAll("button, [role=button], a, input, textarea, span"))
    .slice(0, 220)
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
        return int(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def queue_prompt_excerpt(text: str, *, limit: int = 320) -> str:
    body = str(text or "")
    match = re.search(r"TPUs are popular.{0,280}(?:accelerator|queue|later)\.?", body, flags=re.IGNORECASE | re.DOTALL)
    excerpt = match.group(0) if match else ui_state.public_excerpt(body, limit=limit)
    excerpt = re.sub(r"\s+", " ", excerpt).strip()[:limit]
    for fragment in SENSITIVE_FRAGMENTS:
        excerpt = excerpt.replace(fragment, "<redacted>")
    return excerpt


def parse_queue_prompt(body_text: str) -> dict[str, Any]:
    body = re.sub(r"\s+", " ", str(body_text or " ")).strip()
    queue_visible = "tpu" in body.lower() and ("queue" in body.lower() or "popular right now" in body.lower())
    patterns = [
        r"TPUs are popular right now\.?\s*You are\s*#\s*([0-9,]+)\s*in the queue",
        r"You are\s*#\s*([0-9,]+)\s*in the queue",
        r"#\s*([0-9,]+)\s*in the queue",
    ]
    position: int | None = None
    for pattern in patterns:
        match = re.search(pattern, body, flags=re.IGNORECASE)
        if match:
            position = _int(match.group(1), default=0) or None
            break
    return {
        "queue_prompt_visible": queue_visible,
        "queue_position": position,
        "queue_position_visible": position is not None,
        "queue_prompt_excerpt_public": queue_prompt_excerpt(body_text) if queue_visible else "",
        "queue_prompt_hash": sha_payload(queue_prompt_excerpt(body_text)) if queue_visible else "",
    }


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


def summarize_observation(raw: dict[str, Any], *, elapsed_seconds: float, label: str) -> dict[str, Any]:
    body_text = str(raw.get("body_text") or "")
    controls = _list(raw.get("controls"))
    frames = summarize_frames(_list(raw.get("frames")))
    queue = parse_queue_prompt(body_text)
    events = active_event_probe.parse_active_events_from_body(body_text)
    tpu_events = [item for item in events if "TPU" in str(item.get("event_kind_public") or "")]
    active_event_queued = any(item.get("queued") is True for item in tpu_events)
    active_event_running = any(item.get("running") is True for item in tpu_events)
    session_started_text = "Session started" in body_text
    session_starting_text = "Session is starting" in body_text or "\nStarting\n" in f"\n{body_text}\n"
    start_session_visible = any(
        "start session" in f"{item.get('text') or ''} {item.get('aria') or ''} {item.get('title') or ''}".lower()
        for item in controls
        if isinstance(item, dict) and item.get("visible") is True
    )
    runtime_ready = bool(
        frames["jupyter_frame_visible"]
        and frames["jupyter_session_or_kernel_visible"]
        and (session_started_text or active_event_running)
    )
    return {
        "label": label,
        "elapsed_seconds": round(float(elapsed_seconds), 1),
        "body_text_hash": sha_payload(body_text),
        "body_text_excerpt_public": ui_state.public_excerpt(body_text, limit=260),
        "tpu_v5e_option_visible": "TPU v5e-8" in body_text,
        "start_session_visible": start_session_visible,
        "session_started_text_visible": session_started_text,
        "session_starting_text_visible": session_starting_text,
        "active_event_count": len(events),
        "tpu_v5e_active_event_visible": any("TPU v5e-8" in str(item.get("event_kind_public") or "") for item in events),
        "active_event_queued": active_event_queued,
        "active_event_running": active_event_running,
        "active_events": events[:5],
        "web_tpu_runtime_ready": runtime_ready,
        **frames,
        **queue,
    }


def queue_progress(observations: list[dict[str, Any]]) -> dict[str, Any]:
    positions = [
        int(item["queue_position"])
        for item in observations
        if isinstance(item.get("queue_position"), int) and item.get("queue_position") is not None
    ]
    if not positions:
        return {
            "queue_position_observed": False,
            "queue_position_changed": False,
            "queue_position_decreased": False,
            "queue_position_increased": False,
            "first_queue_position": None,
            "last_queue_position": None,
            "min_queue_position": None,
            "max_queue_position": None,
            "unique_queue_positions": [],
        }
    return {
        "queue_position_observed": True,
        "queue_position_changed": len(set(positions)) > 1,
        "queue_position_decreased": any(next_pos < pos for pos, next_pos in zip(positions, positions[1:])),
        "queue_position_increased": any(next_pos > pos for pos, next_pos in zip(positions, positions[1:])),
        "first_queue_position": positions[0],
        "last_queue_position": positions[-1],
        "min_queue_position": min(positions),
        "max_queue_position": max(positions),
        "unique_queue_positions": sorted(set(positions)),
    }


def click_first(page: Any, locators: list[Any], *, timeout_ms: int = 5000) -> bool:
    for locator in locators:
        try:
            locator.first.click(timeout=timeout_ms)
            return True
        except Exception:
            continue
    return False


def observe_page(page: Any, started: float, label: str) -> dict[str, Any]:
    raw = page.evaluate(OBSERVE_JS)
    return summarize_observation(raw, elapsed_seconds=time.monotonic() - started, label=label)


def session_started_without_queue(observation: dict[str, Any]) -> bool:
    return bool(
        observation.get("session_started_text_visible") is True
        and observation.get("queue_prompt_visible") is not True
        and observation.get("session_starting_text_visible") is not True
    )


def session_started_handoff_candidate(observation: dict[str, Any]) -> bool:
    """Return true only when Session started text has a runtime signal behind it."""
    return bool(
        session_started_without_queue(observation)
        and (
            observation.get("web_tpu_runtime_ready") is True
            or observation.get("active_event_running") is True
            or observation.get("jupyter_frame_visible") is True
            or observation.get("jupyter_session_or_kernel_visible") is True
        )
    )


def write_live_status(
    output_dir: Path | None,
    args: argparse.Namespace,
    *,
    steps: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> None:
    if output_dir is None or not observations:
        return
    latest = observations[-1]
    progress = queue_progress(observations)
    status: dict[str, Any] = {
        "schema": LIVE_STATUS_SCHEMA,
        "generated_at": utc_now(),
        "queue_monitor_live_status_ready": True,
        "kaggle_notebook_url_public": False,
        "read_only": bool(args.read_only),
        "observation_count": len(observations),
        "latest_observation_label": str(latest.get("label") or ""),
        "latest_elapsed_seconds": float(latest.get("elapsed_seconds") or 0.0),
        "web_tpu_runtime_ready": latest.get("web_tpu_runtime_ready") is True,
        "queue_prompt_visible": latest.get("queue_prompt_visible") is True,
        "queue_position_visible": latest.get("queue_position_visible") is True,
        "queue_position": latest.get("queue_position") if isinstance(latest.get("queue_position"), int) else None,
        "queue_progress": progress,
        "session_started_text_visible": latest.get("session_started_text_visible") is True,
        "session_started_handoff_candidate": session_started_handoff_candidate(latest),
        "session_starting_text_visible": latest.get("session_starting_text_visible") is True,
        "active_event_queued": latest.get("active_event_queued") is True,
        "active_event_running": latest.get("active_event_running") is True,
        "active_event_count": _int(latest.get("active_event_count")),
        "jupyter_frame_visible": latest.get("jupyter_frame_visible") is True,
        "jupyter_session_or_kernel_visible": latest.get("jupyter_session_or_kernel_visible") is True,
        "start_clicked": any(step.get("name") == "click_start_session" and step.get("ok") is True for step in steps),
        "public_artifact_safe": True,
        "safety": {
            "credentials_public": False,
            "cookies_public": False,
            "jupyter_proxy_token_public": False,
            "private_runtime_state_public": False,
            "public_artifact_safe": True,
        },
    }
    leaks = public_redaction_errors(status)
    if leaks:
        status["public_artifact_safe"] = False
        status["safety"]["public_artifact_safe"] = False
        status["redaction_errors"] = leaks
    write_json(output_dir / "kaggle_web_tpu_queue_monitor_live_status.json", status)


def maybe_open_active_events(page: Any, steps: list[dict[str, Any]], *, timeout_ms: int = 5000) -> bool:
    opened = click_first(
        page,
        [
            page.get_by_text(re.compile("View Active Events", re.I)),
            page.get_by_role("button", name=re.compile("View Active Events", re.I)),
        ],
        timeout_ms=timeout_ms,
    )
    steps.append({"name": "open_active_events_dialog", "ok": opened})
    if opened:
        page.wait_for_timeout(1500)
    return opened


def maybe_start_tpu(page: Any, steps: list[dict[str, Any]], *, timeout_ms: int = 5000) -> None:
    expanded = click_first(
        page,
        [
            page.get_by_title(re.compile("Expand Session options", re.I)),
            page.get_by_label(re.compile("Expand Session options", re.I)),
            page.get_by_text(re.compile("Session options", re.I)),
        ],
        timeout_ms=timeout_ms,
    )
    steps.append({"name": "expand_session_options", "ok": expanded})
    page.wait_for_timeout(1500)
    selected = click_first(page, [page.get_by_text("TPU v5e-8", exact=True)], timeout_ms=timeout_ms)
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
        timeout_ms=max(timeout_ms, 8000),
    )
    steps.append({"name": "click_start_session", "ok": clicked})
    page.wait_for_timeout(3000)


def run_live(args: argparse.Namespace, *, output_dir: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from playwright.sync_api import sync_playwright

    steps: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    started = time.monotonic()

    def record(page: Any, label: str) -> dict[str, Any]:
        observation = observe_page(page, started, label)
        observations.append(observation)
        write_live_status(output_dir, args, steps=steps, observations=observations)
        return observation

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
            record(page, "initial")

            if args.observe_active_events:
                maybe_open_active_events(page, steps)
                record(page, "active_events_initial")
                page.keyboard.press("Escape")
                page.wait_for_timeout(1000)

            if not args.read_only and not observations[-1].get("web_tpu_runtime_ready"):
                already_waiting = any(
                    item.get("queue_prompt_visible")
                    or item.get("active_event_queued")
                    or item.get("active_event_running")
                    or item.get("session_starting_text_visible")
                    for item in observations
                )
                if already_waiting and not args.force_start_click:
                    steps.append({"name": "skip_start_existing_waiting_state", "ok": True})
                else:
                    maybe_start_tpu(page, steps)
                    record(page, "after_start_attempt")

            deadline = time.monotonic() + float(args.wait_seconds)
            poll_index = 0
            session_started_streak = 0
            while True:
                poll_index += 1
                record(page, f"poll_{poll_index}")
                if observations[-1].get("web_tpu_runtime_ready") is True:
                    break
                if session_started_handoff_candidate(observations[-1]):
                    session_started_streak += 1
                else:
                    session_started_streak = 0
                if args.stop_after_session_started_polls > 0 and session_started_streak >= args.stop_after_session_started_polls:
                    steps.append(
                        {
                            "name": "stop_after_session_started_stable",
                            "ok": True,
                            "polls": session_started_streak,
                        }
                    )
                    break
                if time.monotonic() >= deadline:
                    break
                page.wait_for_timeout(int(args.poll_seconds * 1000))
                if args.observe_active_events_each_poll:
                    maybe_open_active_events(page, steps, timeout_ms=2500)
                    record(page, f"active_events_poll_{poll_index}")
                    if observations[-1].get("active_event_running") is True and args.attempt_open_running_event:
                        opened = click_first(page, [page.get_by_text(re.compile("Interactive Session with TPU", re.I))], timeout_ms=5000)
                        steps.append({"name": "open_running_active_event", "ok": opened})
                        if opened:
                            page.wait_for_timeout(5000)
                    page.keyboard.press("Escape")
                    page.wait_for_timeout(1000)
        finally:
            browser.close()
    return steps, observations


def build_report(args: argparse.Namespace, *, steps: list[dict[str, Any]], observations: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    final = observations[-1] if observations else {}
    ready = final.get("web_tpu_runtime_ready") is True
    progress = queue_progress(observations)
    start_clicked = any(step.get("name") == "click_start_session" and step.get("ok") is True for step in steps)
    blockers: set[str] = set()
    if not ready:
        final_running = final.get("active_event_running") is True
        if final.get("queue_prompt_visible") is True and not final_running:
            blockers.add("kaggle_web_tpu_queue_prompt_visible")
        if progress["queue_position_observed"] and not progress["queue_position_changed"]:
            blockers.add("kaggle_web_tpu_queue_position_static")
        if final.get("active_event_queued") is True and not final_running:
            blockers.add("kaggle_web_tpu_active_event_queued")
        if final.get("session_starting_text_visible") is True and not final_running:
            blockers.add("kaggle_web_tpu_session_still_starting")
        if any(session_started_without_queue(item) for item in observations) and not any(
            item.get("web_tpu_runtime_ready")
            or item.get("active_event_running")
            or item.get("jupyter_frame_visible")
            or item.get("jupyter_session_or_kernel_visible")
            for item in observations
        ):
            blockers.add("kaggle_web_tpu_session_started_text_without_runtime")
        if not final.get("jupyter_frame_visible"):
            blockers.add("kaggle_web_tpu_jupyter_frame_not_visible")
        if not final.get("jupyter_session_or_kernel_visible"):
            blockers.add("kaggle_web_tpu_jupyter_session_not_visible")
        if not args.read_only and not start_clicked and not any(
            item.get("queue_prompt_visible") or item.get("active_event_queued") or item.get("session_starting_text_visible")
            for item in observations
        ):
            blockers.add("kaggle_web_tpu_start_session_not_clicked")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "queue_monitor_probe_ready": True,
        "kaggle_notebook_url_public": False,
        "read_only": bool(args.read_only),
        "start_clicked": start_clicked,
        "force_start_click": bool(args.force_start_click),
        "stop_after_session_started_polls": int(getattr(args, "stop_after_session_started_polls", 0)),
        "session_started_early_stop_triggered": any(
            step.get("name") == "stop_after_session_started_stable" and step.get("ok") is True for step in steps
        ),
        "bounded_wait_seconds": float(args.wait_seconds),
        "poll_seconds": float(args.poll_seconds),
        "observation_count": len(observations),
        "web_tpu_runtime_ready": ready,
        "active_event_running": final.get("active_event_running") is True,
        "active_event_queued": final.get("active_event_queued") is True,
        "session_started_text_visible": final.get("session_started_text_visible") is True,
        "jupyter_frame_visible": final.get("jupyter_frame_visible") is True,
        "jupyter_session_or_kernel_visible": final.get("jupyter_session_or_kernel_visible") is True,
        "queue_progress": progress,
        "final_observation": final,
        "observations": observations[-40:],
        "steps": steps,
        "blocked_reason": "" if ready else (sorted(blockers)[0] if blockers else "kaggle_web_tpu_runtime_not_ready_within_bounded_wait"),
        "blocker_codes": sorted(blockers),
        "diagnosis_codes": sorted(
            {
                "kaggle_web_tpu_queue_monitor_probe_ready",
                "kaggle_web_tpu_runtime_ready" if ready else "kaggle_web_tpu_runtime_not_ready",
                "kaggle_web_tpu_queue_position_observed" if progress["queue_position_observed"] else "kaggle_web_tpu_queue_position_not_observed",
                "kaggle_web_tpu_queue_position_changed" if progress["queue_position_changed"] else "kaggle_web_tpu_queue_position_not_changed",
            }
        ),
        "cleanup_status": {
            "temporary_kaggle_kernels_created": False,
            "temporary_kaggle_kernels_deleted": True,
            "temporary_private_packages_removed": True,
            "live_resources_left_running": bool(
                not ready
                and (
                    start_clicked
                    or final.get("active_event_queued") is True
                    or final.get("active_event_running") is True
                )
            ),
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
            "This observes or starts a Kaggle Web TPU UI allocation; it does not execute TPU code.",
            "Queue position is parsed from visible Kaggle UI text and may be absent if Kaggle changes the prompt or shows only Active Events.",
            "A ready UI state must still be followed by the execution-channel probe before model inference.",
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
    summary_path = output_dir / "kaggle_web_tpu_queue_monitor_probe.json"
    write_json(summary_path, report)
    report["artifacts"] = {
        "summary_json": artifact_entry(summary_path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(report.get("ok"))),
        "live_status_json": artifact_entry(
            output_dir / "kaggle_web_tpu_queue_monitor_live_status.json",
            output_dir,
            kind="live_status_json",
            schema=LIVE_STATUS_SCHEMA,
            ok=True,
        ),
    }
    write_json(summary_path, report)
    return report


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps, observations = run_live(args, output_dir=output_dir)
    return build_report(args, steps=steps, observations=observations, output_dir=output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Kaggle Web TPU queue/runtime state.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kaggle-notebook-url", default=DEFAULT_NOTEBOOK_URL)
    parser.add_argument("--kaggle-web-storage-state", default="/root/kaggle-web-storage-state.json")
    parser.add_argument("--chrome-executable", default="/usr/bin/google-chrome")
    parser.add_argument("--page-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--settle-seconds", type=float, default=8.0)
    parser.add_argument("--wait-seconds", type=float, default=900.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--force-start-click", action="store_true")
    parser.add_argument("--observe-active-events", action="store_true", default=True)
    parser.add_argument("--observe-active-events-each-poll", action="store_true", default=False)
    parser.add_argument("--attempt-open-running-event", action="store_true", default=True)
    parser.add_argument(
        "--stop-after-session-started-polls",
        type=int,
        default=0,
        help=(
            "If >0, stop after this many consecutive polls show Session started without a queue prompt, "
            "so the execution-channel probe can run promptly."
        ),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.page_timeout_seconds < 10 or args.page_timeout_seconds > 180:
        raise SystemExit("--page-timeout-seconds must be between 10 and 180")
    if args.settle_seconds < 1 or args.settle_seconds > 60:
        raise SystemExit("--settle-seconds must be between 1 and 60")
    if args.wait_seconds < 0 or args.wait_seconds > 21600:
        raise SystemExit("--wait-seconds must be between 0 and 21600")
    if args.poll_seconds < 5 or args.poll_seconds > 600:
        raise SystemExit("--poll-seconds must be between 5 and 600")
    if args.stop_after_session_started_polls < 0 or args.stop_after_session_started_polls > 100:
        raise SystemExit("--stop-after-session-started-polls must be between 0 and 100")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_probe(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        progress = report.get("queue_progress") if isinstance(report.get("queue_progress"), dict) else {}
        print(
            f"{SCHEMA}: ready={bool(report.get('web_tpu_runtime_ready'))} "
            f"start_clicked={bool(report.get('start_clicked'))} "
            f"queue={progress.get('last_queue_position')} "
            f"changed={bool(progress.get('queue_position_changed'))} "
            f"blocked={report.get('blocked_reason') or 'none'}"
        )
    return 0 if report.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
