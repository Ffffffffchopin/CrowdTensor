#!/usr/bin/env python3
"""CI-safe check for the remote micro-LLM pipeline-sharded inference Beta path."""

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
    "hidden_state",
    "inference_results",
    "inference_result",
    "Bearer ",
    "observer-secret",
    "admin-secret",
]
REQUIRED_CODES = {
    "remote_micro_llm_sharded_ready",
    "micro_llm_sharded_ready",
    "stage_0_accepted",
    "stage_1_accepted",
    "activation_transport_ready",
    "baseline_match",
    "decoded_tokens_match",
}
STAGE_AWARE_CODES = {
    "distinct_stage_miners",
    "stage_assignment_valid",
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


def validate_report(payload: dict, *, mode: str, require_requeue: bool, require_distinct_stage_miners: bool) -> None:
    if payload.get("schema") != "remote_micro_llm_sharded_beta_v1" or payload.get("ok") is not True:
        raise SystemExit(f"unexpected remote micro-LLM sharded Beta report: {json.dumps(payload, sort_keys=True)}")
    if payload.get("mode") != mode:
        raise SystemExit(f"unexpected mode: {payload.get('mode')}")
    codes = set(payload.get("diagnosis_codes") or [])
    missing = sorted(REQUIRED_CODES - codes)
    if missing:
        raise SystemExit(f"missing readiness diagnosis: {missing}")
    expected_mode_code = "remote_micro_llm_sharded_loopback_ready" if mode == "remote-loopback" else "local_micro_llm_sharded_inference_ready"
    if mode in {"local", "remote-loopback"} and expected_mode_code not in codes:
        raise SystemExit(f"missing mode readiness diagnosis: {expected_mode_code}")
    if require_requeue and "stage_requeue_ready" not in codes:
        raise SystemExit("missing stage_requeue_ready")
    if require_distinct_stage_miners:
        missing_stage = sorted(STAGE_AWARE_CODES - codes)
        if missing_stage:
            raise SystemExit(f"missing stage-aware diagnosis: {missing_stage}")
        summaries = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
        inner = next((value for value in summaries.values() if isinstance(value, dict)), {})
        assignment = inner.get("stage_assignment") if isinstance(inner.get("stage_assignment"), dict) else {}
        if assignment.get("distinct_stage_miners") is not True:
            raise SystemExit(f"distinct stage miner proof missing: {assignment}")
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    if safety.get("cpu_only_default") is not True or safety.get("activation_payloads_redacted") is not True:
        raise SystemExit(f"unexpected safety summary: {safety}")
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            raise SystemExit(f"remote micro-LLM sharded Beta report leaked sensitive fragment: {fragment}")
    if not (payload.get("artifacts") or {}).get("remote_micro_llm_sharded_beta_json", {}).get("present"):
        raise SystemExit("remote micro-LLM sharded Beta JSON artifact was not reported present")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the remote micro-LLM sharded inference Beta check.")
    parser.add_argument("--mode", choices=["local", "remote-loopback"], default="remote-loopback")
    parser.add_argument("--base-port", type=int, default=9870)
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument(
        "--failure-mode",
        choices=["none", "kill-stage-after-claim", "kill-stage0-after-claim", "kill-stage1-after-claim"],
        default="none",
    )
    parser.add_argument("--stage-mode", choices=["both", "split"], default="both")
    parser.add_argument("--require-distinct-stage-miners", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="crowdtensor_remote_micro_llm_sharded_beta_") as temp:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "remote_micro_llm_sharded_beta_pack.py"),
            "--mode",
            args.mode,
            "--output-dir",
            temp,
            "--base-port",
            str(args.base_port),
            "--request-count",
            str(args.request_count),
            "--decode-steps",
            str(args.decode_steps),
            "--failure-mode",
            args.failure_mode,
            "--stage-mode",
            args.stage_mode,
            "--timeout-seconds",
            str(int(args.timeout_seconds)),
            "--json",
        ]
        if args.require_distinct_stage_miners:
            command.append("--require-distinct-stage-miners")
        payload = run_json(command, timeout=args.timeout_seconds)
        validate_report(
            payload,
            mode=args.mode,
            require_requeue=args.failure_mode != "none",
            require_distinct_stage_miners=args.require_distinct_stage_miners or args.stage_mode == "split",
        )
        print(json.dumps({
            "schema": "remote_micro_llm_sharded_beta_check_v1",
            "ok": True,
            "mode": args.mode,
            "failure_mode": args.failure_mode,
            "stage_mode": args.stage_mode,
            "require_distinct_stage_miners": bool(args.require_distinct_stage_miners),
            "decode_steps": args.decode_steps,
            "diagnosis_codes": payload.get("diagnosis_codes") or [],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
