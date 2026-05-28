#!/usr/bin/env python3
"""CI-safe check for the Kaggle Remote Miner Beta target."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "kaggle_remote_miner_beta_check_v1"
SECRET_FRAGMENTS = (
    "lease_token",
    "idempotency_key",
    "inference_results",
    "external_llm_results",
    "output_text",
    "Bearer ",
    "observer-secret",
    "admin-secret",
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


def parse_private_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        parsed = shlex.split(raw_value)
        values[key] = parsed[0] if parsed else ""
    return values


def assert_no_sensitive_output(payload: dict[str, Any], *, extra_secrets: list[str] | None = None) -> None:
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in [*SECRET_FRAGMENTS, *(extra_secrets or [])]:
        if fragment and fragment in encoded:
            raise SystemExit(f"Kaggle Remote Miner Beta check leaked sensitive fragment: {fragment}")


def validate_prepare(payload: dict[str, Any], output_dir: Path, *, operator_env: dict[str, str], miner_env: dict[str, str]) -> dict[str, Any]:
    if payload.get("schema") != "remote_home_compute_demo_v1" or payload.get("ok") is not True:
        raise SystemExit(f"unexpected prepare payload: {json.dumps(payload, sort_keys=True)}")
    target = payload.get("target_environment") or {}
    if target.get("name") != "kaggle" or target.get("kaggle_remote_miner_beta") is not True:
        raise SystemExit(f"prepare did not mark Kaggle target: {target}")
    if target.get("gpu_tpu_workload_enabled") is not False:
        raise SystemExit(f"Kaggle target must not enable GPU/TPU workload: {target}")
    codes = payload.get("diagnosis_codes") or []
    if "kaggle_remote_miner_prepare_ready" not in codes:
        raise SystemExit(f"prepare missing Kaggle ready code: {codes}")
    artifacts = payload.get("artifacts") or {}
    for artifact_name in ["kaggle_remote_miner_script", "kaggle_remote_miner_runbook", "miner_private_env", "operator_private_env"]:
        artifact = artifacts.get(artifact_name) or {}
        if artifact.get("present") is not True:
            raise SystemExit(f"prepare missing {artifact_name}: {artifact}")
    script_text = (output_dir / "kaggle_remote_miner.py").read_text(encoding="utf-8")
    markdown_text = (output_dir / "kaggle_remote_miner.md").read_text(encoding="utf-8")
    if "CROWDTENSOR_REMOTE_ENVIRONMENT" not in script_text or "kaggle" not in script_text:
        raise SystemExit("Kaggle Miner script does not set CROWDTENSOR_REMOTE_ENVIRONMENT")
    if "operator.private.env" in script_text:
        raise SystemExit("Kaggle Miner script must not reference operator.private.env")
    if "Upload only `miner.private.env`" not in markdown_text:
        raise SystemExit("Kaggle runbook must instruct operators to upload only miner.private.env")
    assert_no_sensitive_output(payload, extra_secrets=list(operator_env.values()) + list(miner_env.values()))
    return {
        "name": "kaggle_prepare",
        "ok": True,
        "schema": payload.get("schema"),
        "diagnosis_codes": codes,
        "artifacts": {
            "kaggle_remote_miner_script": artifacts.get("kaggle_remote_miner_script"),
            "kaggle_remote_miner_runbook": artifacts.get("kaggle_remote_miner_runbook"),
        },
    }


def run_prepare(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    return run_json([
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "remote-demo",
        "prepare",
        "--target",
        "kaggle",
        "--coordinator-url",
        args.coordinator_url,
        "--miner-id",
        args.miner_id,
        "--output-dir",
        str(output_dir),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--replace",
        "--json",
    ], timeout=args.command_timeout)


def run_loopback(args: argparse.Namespace) -> dict[str, Any]:
    return run_json([
        sys.executable,
        str(ROOT / "scripts" / "remote_home_compute_demo_check.py"),
        "--port",
        str(args.port),
        "--workload",
        "model-bundle",
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--startup-timeout",
        str(args.startup_timeout),
        "--verify-timeout",
        str(args.verify_timeout),
        "--command-timeout",
        str(args.command_timeout),
    ], timeout=args.timeout_seconds)


def validate_loopback(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "remote_home_compute_demo_check_v1" or payload.get("ok") is not True:
        raise SystemExit(f"unexpected loopback payload: {json.dumps(payload, sort_keys=True)}")
    codes = payload.get("diagnosis_codes") or []
    if "remote_home_compute_ready" not in codes:
        raise SystemExit(f"loopback missing remote_home_compute_ready: {codes}")
    assert_no_sensitive_output(payload)
    return {
        "name": "kaggle_protocol_loopback",
        "ok": True,
        "schema": payload.get("schema"),
        "diagnosis_codes": codes,
        "evidence_schema": payload.get("evidence_schema"),
        "observability_schema": payload.get("observability_schema"),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="crowdtensor_kaggle_remote_miner_beta_") as temp:
        output_dir = Path(temp) / "remote-home-compute"
        prepare = run_prepare(args, output_dir)
        operator_env = parse_private_env(output_dir / "operator.private.env")
        miner_env = parse_private_env(output_dir / "miner.private.env")
        steps = [
            validate_prepare(prepare, output_dir, operator_env=operator_env, miner_env=miner_env),
            validate_loopback(run_loopback(args)),
        ]
    diagnosis_codes = sorted({code for step in steps for code in step.get("diagnosis_codes", [])})
    diagnosis_codes.append("kaggle_remote_miner_beta_ready")
    return {
        "ok": all(step.get("ok") for step in steps),
        "schema": SCHEMA,
        "target": "kaggle",
        "step_count": len(steps),
        "steps": steps,
        "diagnosis_codes": sorted(set(diagnosis_codes)),
        "safety": {
            "kaggle_outbound_miner_only": True,
            "operator_env_excluded_from_kaggle": True,
            "token_redaction_checked": True,
            "read_only": True,
            "cpu_only_workload": True,
            "gpu_tpu_workload_enabled": False,
            "not_production": True,
            "not_p2p": True,
        },
        "limitations": [
            "CI uses a local loopback Miner as a protocol stand-in for Kaggle.",
            "Real Kaggle use still requires an operator-owned reachable Coordinator URL.",
            "This validates Kaggle as a temporary outbound CPU Miner target, not GPU/TPU workload adapters.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Kaggle Remote Miner Beta check.")
    parser.add_argument("--coordinator-url", default="https://YOUR_COORDINATOR_HOST")
    parser.add_argument("--miner-id", default="kaggle-cpu-1")
    parser.add_argument("--port", type=int, default=9060)
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument("--scenario-id", default="route-baseline")
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--verify-timeout", type=float, default=40.0)
    parser.add_argument("--command-timeout", type=float, default=120.0)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.port < 1:
        raise SystemExit("--port must be positive")
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    report = build_report(args)
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
