#!/usr/bin/env python3
"""CI-safe check for the remote pipeline-sharded inference Beta path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECRET_FRAGMENTS = [
    "lease_token",
    "idempotency_key",
    "sharded_inference_result",
    "activation_results",
    "activation_result",
    "logits",
    "inference_results",
    "inference_result",
    "Bearer ",
    "observer-secret",
    "admin-secret",
]
REQUIRED_CODES = {
    "remote_sharded_inference_ready",
    "sharded_inference_ready",
    "stage_0_accepted",
    "stage_1_accepted",
    "activation_transport_ready",
    "baseline_match",
}


def run_json(command: list[str], *, timeout: float) -> dict:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    for line in reversed([line for line in completed.stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError(f"command emitted no JSON: {' '.join(command)}\nstdout:\n{completed.stdout}")


def validate_report(payload: dict, *, mode: str, require_requeue: bool) -> None:
    if payload.get("schema") != "remote_sharded_inference_beta_v1" or payload.get("ok") is not True:
        raise SystemExit(f"unexpected remote sharded inference Beta report: {json.dumps(payload, sort_keys=True)}")
    if payload.get("mode") != mode:
        raise SystemExit(f"unexpected mode: {payload.get('mode')}")
    codes = set(payload.get("diagnosis_codes") or [])
    missing = sorted(REQUIRED_CODES - codes)
    if missing:
        raise SystemExit(f"missing readiness diagnosis: {missing}")
    expected_mode_code = "remote_sharded_loopback_ready" if mode == "remote-loopback" else "local_sharded_inference_ready"
    if mode in {"local", "remote-loopback"} and expected_mode_code not in codes:
        raise SystemExit(f"missing mode readiness diagnosis: {expected_mode_code}")
    if require_requeue and "stage_requeue_ready" not in codes:
        raise SystemExit("missing stage_requeue_ready")
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    if safety.get("cpu_only_default") is not True or safety.get("activation_payloads_redacted") is not True:
        raise SystemExit(f"unexpected safety summary: {safety}")
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            raise SystemExit(f"remote sharded inference Beta report leaked sensitive fragment: {fragment}")
    if not (payload.get("artifacts") or {}).get("remote_sharded_inference_beta_json", {}).get("present"):
        raise SystemExit("remote sharded inference Beta JSON artifact was not reported present")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the remote sharded inference Beta check.")
    parser.add_argument("--mode", choices=["local", "remote-loopback"], default="remote-loopback")
    parser.add_argument("--base-port", type=int, default=9830)
    parser.add_argument("--request-count", type=int, default=3)
    parser.add_argument("--failure-mode", choices=["none", "kill-stage-after-claim"], default="none")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="crowdtensor_remote_sharded_beta_") as temp:
        output_dir = Path(temp) / "remote-sharded-inference"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "remote_sharded_inference_beta_pack.py"),
            "--mode",
            args.mode,
            "--output-dir",
            str(output_dir),
            "--base-port",
            str(args.base_port),
            "--request-count",
            str(args.request_count),
            "--failure-mode",
            args.failure_mode,
            "--timeout-seconds",
            str(int(args.timeout_seconds)),
            "--json",
        ]
        payload = run_json(command, timeout=args.timeout_seconds)
        validate_report(
            payload,
            mode=args.mode,
            require_requeue=args.failure_mode == "kill-stage-after-claim",
        )
        print(json.dumps({
            "ok": True,
            "schema": "remote_sharded_inference_beta_check_v1",
            "beta_schema": payload.get("schema"),
            "mode": payload.get("mode"),
            "failure_mode": payload.get("failure_mode"),
            "step_count": len(payload.get("steps") or []),
            "diagnosis_codes": payload.get("diagnosis_codes"),
        }, sort_keys=True))


if __name__ == "__main__":
    main()
