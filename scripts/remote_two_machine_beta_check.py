#!/usr/bin/env python3
"""CI-safe aggregate check for the real two-machine CPU inference Beta path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "remote_two_machine_beta_check_v1"
REMOTE_CHECK_SCHEMA = "remote_home_compute_demo_check_v1"
MODEL_BUNDLE_WORKLOAD = "model-bundle"
EXTERNAL_LLM_WORKLOAD = "external-llm"
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
    "CROWDTENSOR_LLM_RUNTIME_API_KEY=",
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


def workloads_for(value: str) -> list[str]:
    if value == "all":
        return [MODEL_BUNDLE_WORKLOAD, EXTERNAL_LLM_WORKLOAD]
    return [value]


def expected_ready_code(workload: str) -> str:
    if workload == EXTERNAL_LLM_WORKLOAD:
        return "remote_two_machine_external_llm_ready"
    return "remote_two_machine_inference_ready"


def validate_remote_check(payload: dict[str, Any], *, workload: str) -> None:
    if payload.get("schema") != REMOTE_CHECK_SCHEMA or payload.get("ok") is not True:
        raise SystemExit(f"unexpected remote two-machine Beta check payload: {json.dumps(payload, sort_keys=True)}")
    if payload.get("workload") != workload:
        raise SystemExit(f"unexpected workload: {payload.get('workload')}")
    codes = payload.get("diagnosis_codes") or []
    if expected_ready_code(workload) not in codes:
        raise SystemExit(f"missing {expected_ready_code(workload)} diagnosis: {codes}")
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            raise SystemExit(f"remote two-machine Beta check leaked sensitive fragment: {fragment}")


def run_workload(args: argparse.Namespace, *, workload: str, port: int, output_dir: Path) -> dict[str, Any]:
    request_count = args.external_llm_request_count if workload == EXTERNAL_LLM_WORKLOAD else args.request_count
    command = [
        sys.executable,
        str(ROOT / "scripts" / "remote_home_compute_demo_check.py"),
        "--port",
        str(port),
        "--workload",
        workload,
        "--request-count",
        str(request_count),
        "--scenario-id",
        args.scenario_id,
        "--startup-timeout",
        str(args.startup_timeout),
        "--verify-timeout",
        str(args.verify_timeout),
        "--command-timeout",
        str(args.command_timeout),
    ]
    payload = run_json(command, timeout=args.timeout_seconds)
    validate_remote_check(payload, workload=workload)
    return {
        "name": f"remote_two_machine_{workload}",
        "ok": True,
        "workload": workload,
        "port": port,
        "schema": payload.get("schema"),
        "demo_schema": payload.get("demo_schema"),
        "acceptance_schema": payload.get("acceptance_schema"),
        "evidence_schema": payload.get("evidence_schema"),
        "observability_schema": payload.get("observability_schema"),
        "diagnosis_codes": payload.get("diagnosis_codes") or [],
        "output_dir": str(output_dir),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="crowdtensor_remote_two_machine_beta_") as temp:
        output_dir = Path(temp)
        for offset, workload in enumerate(workloads_for(args.workload)):
            steps.append(run_workload(args, workload=workload, port=args.base_port + offset, output_dir=output_dir))
    diagnosis_codes = sorted({code for step in steps for code in step.get("diagnosis_codes", [])})
    ok = bool(steps and all(step.get("ok") for step in steps))
    if ok:
        diagnosis_codes.append("remote_two_machine_beta_ready")
    return {
        "ok": ok,
        "schema": SCHEMA,
        "workload": args.workload,
        "step_count": len(steps),
        "steps": steps,
        "diagnosis_codes": sorted(set(diagnosis_codes)),
        "safety": {
            "local_loopback_standin": True,
            "token_redaction_checked": True,
            "read_only": True,
            "requires_operator_network_for_real_two_machine": True,
            "not_production": True,
            "not_p2p": True,
        },
        "limitations": [
            "CI uses local loopback processes as a stand-in for two physical machines.",
            "Real two-machine use still requires operator-provided TLS, VPN, tunnel, or trusted network path.",
            "This validates task-level remote CPU inference, not model sharding, P2P routing, GPU pooling, or arbitrary prompt serving.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the remote two-machine CPU inference Beta aggregate check.")
    parser.add_argument("--workload", choices=[MODEL_BUNDLE_WORKLOAD, EXTERNAL_LLM_WORKLOAD, "all"], default="all")
    parser.add_argument("--base-port", type=int, default=9050)
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument("--external-llm-request-count", type=int, default=2)
    parser.add_argument("--scenario-id", default="route-baseline")
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--verify-timeout", type=float, default=40.0)
    parser.add_argument("--command-timeout", type=float, default=120.0)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.base_port < 1:
        raise SystemExit("--base-port must be positive")
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    if args.external_llm_request_count < 1:
        raise SystemExit("--external-llm-request-count must be at least 1")
    report = build_report(args)
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
