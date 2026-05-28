#!/usr/bin/env python3
"""CI-safe check for the real small-LLM live two-node RC path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "real_llm_live_rc_check_v1"

SECRET_FRAGMENTS = [
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "lease_token",
    "idempotency_key",
    "activation_results",
    "activation_result",
    "hidden_state",
    "input_ids",
    "logits",
    "inference_results",
    "inference_result",
    "sharded_inference_result",
    "real_llm_sharded_result",
    "Bearer ",
    "CrowdTensor routes",
    "A miner returns",
]

REQUIRED_CODES = {
    "real_llm_live_rc_ready",
    "local_generated_real_llm_stage_upload_standins_ready",
    "local_generated_real_llm_stage_upload_packages_ready",
    "remote_real_llm_sharded_ready",
    "remote_real_llm_sharded_existing_ready",
    "real_llm_sharded_ready",
    "real_llm_artifact_ready",
    "stage_0_accepted",
    "stage_1_accepted",
    "activation_transport_ready",
    "baseline_match",
    "decoded_tokens_match",
    "distinct_stage_miners",
    "stage_assignment_valid",
}


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError("command emitted no JSON object")


def run_json(command: list[str], *, timeout: float) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json_from_stdout(completed.stdout)


def assert_no_sensitive_output(payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            raise SystemExit(f"real LLM live RC check leaked sensitive fragment: {fragment}")


def validate_local_generated(payload: dict[str, Any]) -> None:
    if payload.get("schema") != "real_llm_live_rc_v1" or payload.get("ok") is not True:
        raise SystemExit(f"unexpected real LLM live RC report: {json.dumps(payload, sort_keys=True)}")
    if payload.get("mode") != "local-generated":
        raise SystemExit(f"unexpected mode: {payload.get('mode')}")
    codes = set(payload.get("diagnosis_codes") or [])
    missing = sorted(REQUIRED_CODES - codes)
    if missing:
        raise SystemExit(f"missing readiness diagnosis: {missing}")
    runtime = payload.get("runtime_classification") if isinstance(payload.get("runtime_classification"), dict) else {}
    if runtime.get("local_generated_stage_upload_standins") is not True:
        raise SystemExit(f"local generated runtime classification missing: {runtime}")
    if runtime.get("external_runtime_verified") is not False:
        raise SystemExit(f"local generated check must not claim external runtime: {runtime}")
    packages = payload.get("stage_packages") if isinstance(payload.get("stage_packages"), list) else []
    if {package.get("stage_role") for package in packages} != {"stage0", "stage1"}:
        raise SystemExit(f"stage packages missing stage0/stage1: {packages}")
    for package in packages:
        for key in [
            "miner_env_present",
            "miner_script_present",
            "operator_env_excluded",
            "launcher_has_stage_role",
            "hf_runtime_enabled",
            "hf_model_id_present",
        ]:
            if package.get(key) is not True:
                raise SystemExit(f"stage package {package.get('stage_role')} failed {key}: {package}")
    processes = payload.get("process_summary") if isinstance(payload.get("process_summary"), list) else []
    miner_names = {process.get("name") for process in processes if str(process.get("name", "")).endswith("_miner")}
    if miner_names != {"stage0_miner", "stage1_miner"}:
        raise SystemExit(f"stage Miner process summary missing: {processes}")
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    for key in [
        "read_only",
        "cpu_only_workload",
        "summary_excludes_plaintext_tokens",
        "raw_activation_redacted",
        "not_production",
        "not_p2p",
        "not_large_model_serving",
    ]:
        if safety.get(key) is not True:
            raise SystemExit(f"safety flag {key} must be true: {safety}")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    for name in [
        "real_llm_live_rc_json",
        "remote_real_llm_sharded_beta_json",
        "remote_existing_real_llm_sharded_evidence_json",
        "kaggle_upload_real_llm_stage0_miner_env",
        "kaggle_upload_real_llm_stage1_miner_env",
    ]:
        if (artifacts.get(name) or {}).get("present") is not True:
            raise SystemExit(f"missing artifact {name}: {artifacts.get(name)}")
    assert_no_sensitive_output(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real small-LLM live two-node RC check.")
    parser.add_argument("--base-port", type=int, default=9184)
    parser.add_argument("--request-count", type=int, default=1)
    parser.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="crowdtensor_real_llm_live_rc_") as temp:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "real_llm_live_rc_pack.py"),
            "--mode",
            "local-generated",
            "--output-dir",
            str(Path(temp) / "real-llm-live-rc"),
            "--port",
            str(args.base_port),
            "--request-count",
            str(args.request_count),
            "--hf-model-id",
            args.hf_model_id,
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--json",
        ]
        if args.hf_cache_dir:
            command.extend(["--hf-cache-dir", args.hf_cache_dir])
        payload = run_json(command, timeout=args.timeout_seconds)
        validate_local_generated(payload)
        print(json.dumps({
            "schema": SCHEMA,
            "ok": True,
            "mode": payload.get("mode"),
            "diagnosis_codes": payload.get("diagnosis_codes") or [],
            "request_count": args.request_count,
            "hf_model_id": args.hf_model_id,
        }, sort_keys=True))


if __name__ == "__main__":
    main()
