#!/usr/bin/env python3
"""CI-safe check for the CPU inference Beta aggregate path."""

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
    "inference_results",
    "external_llm_results",
    "output_text",
    "Bearer ",
    "observer-secret",
    "admin-secret",
    "runtime-secret",
]


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


def validate_report(payload: dict, *, mode: str, workload: str) -> None:
    if payload.get("schema") != "cpu_inference_beta_v1" or payload.get("ok") is not True:
        raise SystemExit(f"unexpected CPU inference Beta report: {json.dumps(payload, sort_keys=True)}")
    if payload.get("mode") != mode:
        raise SystemExit(f"unexpected mode: {payload.get('mode')}")
    if mode != "local" and payload.get("workload") != workload:
        raise SystemExit(f"unexpected workload: {payload.get('workload')}")
    if "cpu_inference_beta_ready" not in payload.get("diagnosis_codes", []):
        raise SystemExit(f"missing readiness diagnosis: {payload.get('diagnosis_codes')}")
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            raise SystemExit(f"CPU inference Beta report leaked sensitive fragment: {fragment}")
    if not (payload.get("artifacts") or {}).get("cpu_inference_beta_json", {}).get("present"):
        raise SystemExit("CPU inference Beta JSON artifact was not reported present")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CPU inference Beta aggregate check.")
    parser.add_argument("--mode", choices=["local", "remote-loopback"], default="local")
    parser.add_argument("--workload", choices=["model-bundle", "external-llm", "all"], default="all")
    parser.add_argument("--base-port", type=int, default=8970)
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument("--external-llm-request-count", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--remote-timeout-seconds", type=float, default=40.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="crowdtensor_cpu_infer_beta_") as temp:
        output_dir = Path(temp) / "cpu-infer"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "cpu_inference_beta_pack.py"),
            "--mode",
            args.mode,
            "--workload",
            args.workload,
            "--output-dir",
            str(output_dir),
            "--base-port",
            str(args.base_port),
            "--request-count",
            str(args.request_count),
            "--external-llm-request-count",
            str(args.external_llm_request_count),
            "--timeout-seconds",
            str(int(args.timeout_seconds)),
            "--remote-timeout-seconds",
            str(args.remote_timeout_seconds),
            "--json",
        ]
        payload = run_json(command, timeout=args.timeout_seconds)
        validate_report(payload, mode=args.mode, workload=args.workload)
        print(json.dumps({
            "ok": True,
            "schema": "cpu_inference_beta_check_v1",
            "beta_schema": payload.get("schema"),
            "mode": payload.get("mode"),
            "workload": payload.get("workload"),
            "step_count": len(payload.get("steps") or []),
            "diagnosis_codes": payload.get("diagnosis_codes"),
        }, sort_keys=True))


if __name__ == "__main__":
    main()
