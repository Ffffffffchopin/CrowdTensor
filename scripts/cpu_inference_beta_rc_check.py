#!/usr/bin/env python3
"""CI-safe check for the CPU Inference Beta release-candidate path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "cpu_inference_beta_rc_check_v1"
SECRET_FRAGMENTS = (
    "lease_token",
    "idempotency_key",
    "inference_results",
    "external_llm_results",
    "output_text",
    "Bearer ",
    "observer-secret",
    "admin-secret",
    "runtime-secret",
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
)


def run_json(command: list[str], *, timeout: float) -> dict[str, Any]:
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


def assert_no_sensitive_output(payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            raise SystemExit(f"CPU Inference Beta RC check leaked sensitive fragment: {fragment}")


def validate_report(payload: dict[str, Any]) -> None:
    if payload.get("schema") != "cpu_inference_beta_rc_v1" or payload.get("ok") is not True:
        raise SystemExit(f"unexpected CPU Inference Beta RC report: {json.dumps(payload, sort_keys=True)}")
    codes = payload.get("diagnosis_codes") or []
    for code in [
        "cpu_inference_beta_rc_ready",
        "local_cpu_inference_ready",
        "remote_loopback_ready",
        "two_machine_rehearsal_ready",
        "kaggle_remote_miner_artifacts_ready",
        "miner_join_pack_ready",
        "cpu_miner_beta_ready",
    ]:
        if code not in codes:
            raise SystemExit(f"missing readiness diagnosis {code}: {codes}")
    artifacts = payload.get("artifacts") or {}
    for name in [
        "cpu_inference_beta_rc_json",
        "cpu_inference_beta_rc_markdown",
        "local_cpu_inference_beta_json",
        "remote_loopback_cpu_inference_beta_json",
        "kaggle_remote_miner_script",
        "kaggle_remote_miner_runbook",
        "miner_join_script",
        "miner_join_runbook",
        "demo_manifest_json",
    ]:
        artifact = artifacts.get(name) or {}
        if artifact.get("present") is not True:
            raise SystemExit(f"missing CPU Inference Beta RC artifact {name}: {artifact}")
    join = payload.get("miner_join_pack") or {}
    if join.get("schema") != "miner_join_pack_v1" or join.get("ready") is not True:
        raise SystemExit(f"missing ready Miner join pack: {join}")
    safety = payload.get("safety") or {}
    for key in ["cpu_only", "read_only", "not_production", "not_p2p", "not_gpu_tpu_workload"]:
        if safety.get(key) is not True:
            raise SystemExit(f"missing safety flag {key}: {safety}")
    assert_no_sensitive_output(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CPU Inference Beta RC aggregate check.")
    parser.add_argument("--base-port", type=int, default=9070)
    parser.add_argument("--request-count", type=int, default=1)
    parser.add_argument("--external-llm-request-count", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--remote-timeout-seconds", type=float, default=40.0)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--verify-timeout", type=float, default=40.0)
    parser.add_argument("--command-timeout", type=float, default=120.0)
    parser.add_argument("--two-machine-timeout-seconds", type=int, default=240)
    parser.add_argument("--kaggle-timeout-seconds", type=int, default=240)
    parser.add_argument("--kaggle-real-runtime-report", default="")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.base_port < 1:
        raise SystemExit("--base-port must be positive")
    with tempfile.TemporaryDirectory(prefix="crowdtensor_cpu_infer_beta_rc_") as temp:
        output_dir = Path(temp) / "beta-rc"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "cpu_inference_beta_rc_pack.py"),
            "--output-dir",
            str(output_dir),
            "--base-port",
            str(args.base_port),
            "--request-count",
            str(args.request_count),
            "--external-llm-request-count",
            str(args.external_llm_request_count),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--remote-timeout-seconds",
            str(args.remote_timeout_seconds),
            "--startup-timeout",
            str(args.startup_timeout),
            "--verify-timeout",
            str(args.verify_timeout),
            "--command-timeout",
            str(args.command_timeout),
            "--two-machine-timeout-seconds",
            str(args.two_machine_timeout_seconds),
            "--kaggle-timeout-seconds",
            str(args.kaggle_timeout_seconds),
            "--json",
        ]
        if args.kaggle_real_runtime_report:
            command.extend(["--kaggle-real-runtime-report", args.kaggle_real_runtime_report])
        if args.quick:
            command.append("--quick")
        payload = run_json(command, timeout=args.timeout_seconds + args.two_machine_timeout_seconds + args.kaggle_timeout_seconds)
        validate_report(payload)
        print(json.dumps({
            "ok": True,
            "schema": SCHEMA,
            "beta_rc_schema": payload.get("schema"),
            "step_count": len(payload.get("steps") or []),
            "diagnosis_codes": payload.get("diagnosis_codes"),
            "artifact_count": len(payload.get("artifacts") or {}),
        }, sort_keys=True))


if __name__ == "__main__":
    main()
