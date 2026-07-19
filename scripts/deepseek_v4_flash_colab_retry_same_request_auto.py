#!/usr/bin/env python3
"""Retry Colab CUDA allocation, then run DeepSeek same-request if ready."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


SCHEMA = "deepseek_v4_flash_colab_retry_same_request_auto_v1"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def run_command(command: list[str], *, timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "command_public": [str(item) for item in command],
        "stdout_tail": (completed.stdout or "")[-2000:],
        "stderr_tail": (completed.stderr or "")[-2000:],
    }


def build_retry_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/colab_cuda_reacquire_retry_probe.py",
        "--output-dir",
        str(output_dir),
        "--accelerators",
        args.colab_accelerators,
        "--authusers",
        args.colab_authusers,
        "--attempts",
        str(args.colab_retry_attempts),
        "--sleep-seconds",
        str(args.colab_retry_sleep_seconds),
        "--attempt-timeout-seconds",
        str(args.colab_retry_attempt_timeout_seconds),
        "--session-name",
        args.colab_session,
        "--token-cache",
        args.colab_token_cache,
        "--state-path",
        args.colab_config,
        "--json",
    ]
    if args.cleanup_before_gpu:
        command.append("--cleanup-before-gpu")
    if args.cleanup_other_gpu:
        command.append("--cleanup-other-gpu")
    return command


def build_same_request_command(args: argparse.Namespace, output_dir: Path, retry: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        "scripts/deepseek_v4_flash_quantized_same_request_probe.py",
        "--mode",
        "kaggle-auto",
        "--output-dir",
        str(output_dir),
        "--source-resolver-report",
        args.source_resolver_report,
        "--kaggle-owner",
        args.kaggle_owner,
        "--runtime-tarball-path",
        args.runtime_tarball_path,
        "--runtime-tarball-sha256",
        args.runtime_tarball_sha256,
        "--colab-accelerators",
        str(retry.get("accelerator") or args.colab_accelerators),
        "--colab-authusers",
        str(retry.get("authuser") or args.colab_authusers),
        "--colab-max-attempts",
        str(args.colab_max_attempts),
        "--colab-background-launch-timeout-seconds",
        str(args.colab_background_launch_timeout_seconds),
        "--colab-background-timeout-seconds",
        str(args.colab_background_timeout_seconds),
        "--colab-background-poll-interval-seconds",
        str(args.colab_background_poll_interval_seconds),
        "--colab-keepalive-seconds",
        str(args.colab_keepalive_seconds),
        "--kernel-timeout-seconds",
        str(args.kernel_timeout_seconds),
        "--kaggle-status-timeout-seconds",
        str(args.kaggle_status_timeout_seconds),
        "--kaggle-status-poll-interval",
        str(args.kaggle_status_poll_interval),
        "--run-timeout-seconds",
        str(args.run_timeout_seconds),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--context-length",
        str(args.context_length),
        "--json",
    ]
    if args.colab_reacquire_before_same_request:
        command.append("--colab-reacquire-before")
    return command


def build_report(
    *,
    args: argparse.Namespace,
    retry_report: dict[str, Any],
    same_request_report: dict[str, Any],
    steps: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    same_success = bool(
        same_request_report.get("same_request_decode_verified") is True
        and int(same_request_report.get("generated_token_count") or 0) >= 1
        and {"kaggle_cuda", "colab_cuda", "cpu"}.issubset(set(same_request_report.get("accepted_providers") or []))
    )
    blockers = set(str(item) for item in retry_report.get("blockers", []) if item)
    blockers.update(str(item) for item in same_request_report.get("blockers", []) if item)
    if not retry_report.get("colab_cuda_reacquire_ready"):
        blockers.add("colab_cuda_reacquire_not_ready")
    if retry_report.get("colab_cuda_reacquire_ready") and not same_success:
        blockers.add("deepseek_v4_flash_quantized_same_request_decode_not_verified")
    return {
        "schema": SCHEMA,
        "ok": same_success,
        "deepseek_v4_flash_colab_retry_same_request_ready": same_success,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": round(time.monotonic() - started, 3),
        "retry_ready": retry_report.get("colab_cuda_reacquire_ready") is True,
        "same_request_started": bool(same_request_report),
        "same_request_decode_verified": same_success,
        "generated_token_count": int(same_request_report.get("generated_token_count") or 0),
        "accepted_providers": [str(item) for item in same_request_report.get("accepted_providers", [])] if isinstance(same_request_report.get("accepted_providers"), list) else [],
        "failure_stage": "" if same_success else (str(same_request_report.get("failure_stage") or "") if same_request_report else "colab_cuda_reacquire_not_ready"),
        "retry_summary": {
            "schema": retry_report.get("schema"),
            "ok": retry_report.get("ok") is True,
            "colab_cuda_reacquire_ready": retry_report.get("colab_cuda_reacquire_ready") is True,
            "attempts_completed": int(retry_report.get("attempts_completed") or 0),
            "accelerator": str(retry_report.get("accelerator") or ""),
            "authuser": str(retry_report.get("authuser") or ""),
            "blockers": [str(item) for item in retry_report.get("blockers", [])] if isinstance(retry_report.get("blockers"), list) else [],
            "public_artifact_safe": retry_report.get("public_artifact_safe") is True,
        },
        "same_request_summary": {
            "schema": same_request_report.get("schema"),
            "ok": same_request_report.get("ok") is True,
            "same_request_decode_verified": same_request_report.get("same_request_decode_verified") is True,
            "generated_token_count": int(same_request_report.get("generated_token_count") or 0),
            "failure_stage": str(same_request_report.get("failure_stage") or ""),
            "blockers": [str(item) for item in same_request_report.get("blockers", [])] if isinstance(same_request_report.get("blockers"), list) else [],
            "public_artifact_safe": same_request_report.get("public_artifact_safe") is True,
        },
        "steps": steps,
        "artifact_paths": {
            "retry_report": str(Path(args.retry_output_dir) / "colab_cuda_reacquire_retry_probe.json"),
            "same_request_report": str(Path(args.same_request_output_dir) / "deepseek_v4_flash_quantized_same_request_probe.json") if same_request_report else "",
            "summary": str(Path(args.output_dir) / "deepseek_v4_flash_colab_retry_same_request_auto.json"),
        },
        "blockers": sorted(blockers),
        "public_artifact_safe": True,
        "credentials_public": False,
        "private_runtime_state_public": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--retry-output-dir", required=True)
    parser.add_argument("--same-request-output-dir", required=True)
    parser.add_argument("--source-resolver-report", required=True)
    parser.add_argument("--kaggle-owner", required=True)
    parser.add_argument("--runtime-tarball-path", required=True)
    parser.add_argument("--runtime-tarball-sha256", required=True)
    parser.add_argument("--colab-accelerators", default="T4")
    parser.add_argument("--colab-authusers", default="0,1")
    parser.add_argument("--colab-session", default="ct-colab-cuda-gpu")
    parser.add_argument("--colab-config", default=str(Path.home() / ".config/colab-cli/sessions.json"))
    parser.add_argument("--colab-token-cache", default=str(Path.home() / ".config/colab-exec/token.json"))
    parser.add_argument("--colab-retry-attempts", type=int, default=6)
    parser.add_argument("--colab-retry-sleep-seconds", type=float, default=30.0)
    parser.add_argument("--colab-retry-attempt-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--cleanup-before-gpu", action="store_true")
    parser.add_argument("--cleanup-other-gpu", action="store_true")
    parser.add_argument("--colab-reacquire-before-same-request", action="store_true")
    parser.add_argument("--colab-max-attempts", type=int, default=1)
    parser.add_argument("--colab-background-launch-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--colab-background-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--colab-background-poll-interval-seconds", type=float, default=30.0)
    parser.add_argument("--colab-keepalive-seconds", type=int, default=7200)
    parser.add_argument("--kernel-timeout-seconds", type=int, default=7200)
    parser.add_argument("--kaggle-status-timeout-seconds", type=float, default=7500.0)
    parser.add_argument("--kaggle-status-poll-interval", type=float, default=60.0)
    parser.add_argument("--run-timeout-seconds", type=int, default=2400)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--retry-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--same-request-timeout-seconds", type=float, default=9000.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    steps: list[dict[str, Any]] = []

    retry_step = run_command(build_retry_command(args, Path(args.retry_output_dir)), timeout=float(args.retry_timeout_seconds))
    steps.append({"name": "colab_cuda_reacquire_retry", **retry_step})
    retry_report = load_json(Path(args.retry_output_dir) / "colab_cuda_reacquire_retry_probe.json")
    same_request_report: dict[str, Any] = {}

    if retry_report.get("colab_cuda_reacquire_ready") is True:
        same_step = run_command(build_same_request_command(args, Path(args.same_request_output_dir), retry_report), timeout=float(args.same_request_timeout_seconds))
        steps.append({"name": "deepseek_same_request", **same_step})
        same_request_report = load_json(Path(args.same_request_output_dir) / "deepseek_v4_flash_quantized_same_request_probe.json")

    report = build_report(args=args, retry_report=retry_report, same_request_report=same_request_report, steps=steps, started=started)
    report_path = output_dir / "deepseek_v4_flash_colab_retry_same_request_auto.json"
    write_json(report_path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(report_path)
    return 0 if report.get("public_artifact_safe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
