#!/usr/bin/env python3
"""Boundedly retry Colab TPU session allocation and emit public-safe evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


SCHEMA = "colab_tpu_reacquire_retry_probe_v1"
SESSION_PROBE = Path(__file__).with_name("colab_tpu_session_probe.py")


def sha256_short(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def summarize_attempt(index: int, accelerator: str, report_path: Path, report: dict[str, Any], returncode: int) -> dict[str, Any]:
    ok = bool(report.get("ok") is True and report.get("colab_tpu_session_allocated") is True)
    blockers = []
    if not ok:
        blockers.append("colab_tpu_session_not_allocated")
    status = report.get("http_status")
    if status:
        blockers.append(f"colab_assignment_http_{status}")
    blockers.extend(str(item) for item in _list(report.get("diagnosis_codes")) if item)
    error_type = str(report.get("error_type") or "")
    if error_type:
        blockers.append("colab_assignment_error_" + error_type.lower())
    return {
        "attempt_index": index,
        "accelerator_requested": accelerator,
        "authuser": str(report.get("authuser") or ""),
        "ok": ok,
        "returncode": int(returncode),
        "report_path": str(report_path),
        "schema": report.get("schema"),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "oauth_token_public": report.get("oauth_token_public") is True,
        "runtime_proxy_token_public": report.get("runtime_proxy_token_public") is True,
        "runtime_proxy_url_public": report.get("runtime_proxy_url_public") is True,
        "endpoint_public": report.get("endpoint_public") is True,
        "endpoint_hash": report.get("endpoint_hash") or "",
        "runtime_proxy_host_hash": report.get("runtime_proxy_host_hash") or "",
        "accelerator": report.get("accelerator") or "",
        "http_status": status,
        "error_type": error_type,
        "error_digest": report.get("error_digest") or "",
        "diagnosis_codes": _list(report.get("diagnosis_codes")),
        "blockers": sorted(set(blockers)),
        "duration_seconds": report.get("duration_seconds"),
    }


def build_report(
    *,
    output_dir: Path,
    session_name: str,
    accelerators: list[str],
    attempts_requested: int,
    sleep_seconds: float,
    attempt_summaries: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    successful = next((attempt for attempt in attempt_summaries if attempt.get("ok") is True), None)
    blockers = sorted(
        {
            str(blocker)
            for attempt in attempt_summaries
            for blocker in _list(attempt.get("blockers"))
            if blocker
        }
    )
    if successful is None and not blockers:
        blockers = ["colab_tpu_reacquire_not_verified"]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": successful is not None,
        "colab_tpu_reacquire_ready": successful is not None,
        "session_name": session_name,
        "accelerators_attempted": accelerators,
        "attempts_requested": int(attempts_requested),
        "attempts_completed": len(attempt_summaries),
        "sleep_seconds": float(sleep_seconds),
        "attempts": attempt_summaries,
        "successful_attempt_index": int(successful.get("attempt_index")) if successful else 0,
        "successful_report_path": str(successful.get("report_path") or "") if successful else "",
        "accelerator": str(successful.get("accelerator") or successful.get("accelerator_requested") or "") if successful else "",
        "endpoint_hash": str(successful.get("endpoint_hash") or "") if successful else "",
        "runtime_proxy_host_hash": str(successful.get("runtime_proxy_host_hash") or "") if successful else "",
        "blockers": blockers if successful is None else [],
        "public_artifact_safe": True,
        "oauth_token_public": False,
        "runtime_proxy_token_public": False,
        "runtime_proxy_url_public": False,
        "endpoint_public": False,
        "credentials_public": False,
        "private_runtime_state_public": False,
        "duration_seconds": round(time.time() - started, 3),
        "artifact_paths": {
            "summary": str(output_dir / "colab_tpu_reacquire_retry_probe.json"),
            "attempt_dirs": sorted(str(path.parent) for path in [Path(str(a.get("report_path"))) for a in attempt_summaries if a.get("report_path")]),
        },
        "notes": [
            "This retry report is allocation evidence only.",
            "It is not a 72B same-request decode proof.",
        ],
    }
    return report


def parse_accelerators(value: str) -> list[str]:
    accelerators = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in accelerators if item not in {"V5E1", "V6E1"}]
    if invalid:
        raise SystemExit("invalid accelerator(s): " + ",".join(invalid))
    return accelerators or ["V5E1"]


def run_session_probe(args: argparse.Namespace, *, attempt_index: int, accelerator: str, attempt_dir: Path) -> tuple[int, dict[str, Any], Path]:
    command = [
        sys.executable,
        str(SESSION_PROBE),
        "--session-name",
        args.session_name,
        "--accelerator",
        accelerator,
        "--token-cache",
        args.token_cache,
        "--state-path",
        args.state_path,
        "--authuser",
        str(args.authuser_list[(attempt_index - 1) % len(args.authuser_list)]),
        "--output-dir",
        str(attempt_dir),
        "--json",
    ]
    if args.cleanup_other_tpu:
        command.append("--cleanup-other-tpu")
    if args.cleanup_before_tpu:
        command.append("--cleanup-before-tpu")
    completed = subprocess.run(command, text=True, capture_output=True, timeout=args.attempt_timeout_seconds, check=False)
    report_path = attempt_dir / "colab_tpu_session_probe.json"
    if report_path.is_file():
        report = load_json(report_path)
    else:
        digest_source = (completed.stderr or completed.stdout or f"returncode:{completed.returncode}")[:4000]
        report = {
            "schema": "colab_tpu_session_probe_v1",
            "ok": False,
            "colab_tpu_session_allocated": False,
            "public_artifact_safe": True,
            "oauth_token_public": False,
            "runtime_proxy_token_public": False,
            "runtime_proxy_url_public": False,
            "endpoint_public": False,
            "error_type": "session_probe_no_report",
            "error_digest": "sha256:" + hashlib.sha256(digest_source.encode("utf-8")).hexdigest(),
            "diagnosis_codes": [],
        }
        write_json(report_path, report)
    return int(completed.returncode), report, report_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--session-name", default="ct-colab-tpu-v5e1")
    parser.add_argument("--accelerators", default="V5E1")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=60.0)
    parser.add_argument("--attempt-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--token-cache", default=str(Path.home() / ".config/colab-exec/token.json"))
    parser.add_argument("--state-path", default=str(Path.home() / ".config/colab-cli/sessions.json"))
    parser.add_argument("--authusers", default="0")
    parser.add_argument("--cleanup-other-tpu", action="store_true")
    parser.add_argument("--cleanup-before-tpu", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.attempts < 1 or args.attempts > 50:
        raise SystemExit("--attempts must be between 1 and 50")
    if args.sleep_seconds < 0 or args.sleep_seconds > 3600:
        raise SystemExit("--sleep-seconds must be between 0 and 3600")
    if args.attempt_timeout_seconds < 10 or args.attempt_timeout_seconds > 1800:
        raise SystemExit("--attempt-timeout-seconds must be between 10 and 1800")
    args.accelerator_list = parse_accelerators(args.accelerators)
    args.authuser_list = [item.strip() for item in str(args.authusers).split(",") if item.strip()] or ["0"]
    if len(args.authuser_list) > 10:
        raise SystemExit("--authusers may contain at most 10 entries")
    if any(not item.isdigit() for item in args.authuser_list):
        raise SystemExit("--authusers entries must be numeric")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    attempts: list[dict[str, Any]] = []
    for index in range(1, args.attempts + 1):
        accelerator = args.accelerator_list[(index - 1) % len(args.accelerator_list)]
        authuser = args.authuser_list[(index - 1) % len(args.authuser_list)]
        attempt_dir = output_dir / f"attempt-{index:02d}-{accelerator.lower()}-authuser{authuser}"
        returncode, report, report_path = run_session_probe(args, attempt_index=index, accelerator=accelerator, attempt_dir=attempt_dir)
        summary = summarize_attempt(index, accelerator, report_path, report, returncode)
        attempts.append(summary)
        if summary.get("ok") is True:
            break
        if index < args.attempts and args.sleep_seconds:
            time.sleep(args.sleep_seconds)
    report = build_report(
        output_dir=output_dir,
        session_name=args.session_name,
        accelerators=args.accelerator_list,
        attempts_requested=args.attempts,
        sleep_seconds=args.sleep_seconds,
        attempt_summaries=attempts,
        started=started,
    )
    report_path = output_dir / "colab_tpu_reacquire_retry_probe.json"
    write_json(report_path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(report_path)
    if not report.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
