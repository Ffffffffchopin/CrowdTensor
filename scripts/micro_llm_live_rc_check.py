#!/usr/bin/env python3
"""CI-safe check for the micro-LLM live two-node RC path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "micro_llm_live_rc_check_v1"

SECRET_FRAGMENTS = [
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "lease_token",
    "idempotency_key",
    "activation_results",
    "activation_result",
    "hidden_state",
    "logits",
    "inference_results",
    "inference_result",
    "Bearer ",
]

REQUIRED_CODES = {
    "micro_llm_live_rc_ready",
    "local_generated_stage_upload_standins_ready",
    "local_generated_stage_upload_packages_ready",
    "kaggle_artifacts_ready",
    "kaggle_micro_llm_stage0_seen",
    "kaggle_micro_llm_stage1_seen",
    "kaggle_micro_llm_stage_assignment_valid",
    "stage_assignment_valid",
    "kaggle_micro_llm_sharded_ready",
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
            raise SystemExit(f"micro-LLM live RC check leaked sensitive fragment: {fragment}")


def validate_local_generated(payload: dict[str, Any]) -> list[str]:
    if payload.get("schema") != "micro_llm_live_rc_v1" or payload.get("ok") is not True:
        raise SystemExit(f"unexpected micro-LLM live RC report: {json.dumps(payload, sort_keys=True)}")
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
        for key in ["miner_env_present", "miner_script_present", "operator_env_excluded", "launcher_has_stage_role"]:
            if package.get(key) is not True:
                raise SystemExit(f"stage package {package.get('stage_role')} failed {key}: {package}")
    processes = payload.get("process_summary") if isinstance(payload.get("process_summary"), list) else []
    miner_names = {process.get("name") for process in processes if str(process.get("name", "")).endswith("_miner")}
    if miner_names != {"stage0_miner", "stage1_miner"}:
        raise SystemExit(f"stage Miner process summary missing: {processes}")
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    for key in ["read_only", "cpu_only_workload", "summary_excludes_plaintext_tokens", "not_production", "not_p2p"]:
        if safety.get(key) is not True:
            raise SystemExit(f"safety flag {key} must be true: {safety}")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    for name in [
        "micro_llm_live_rc_json",
        "kaggle_real_runtime_acceptance_json",
        "kaggle_upload_stage0_miner_env",
        "kaggle_upload_stage1_miner_env",
        "remote_micro_llm_sharded_beta_json",
        "support_bundle_json",
        "micro_llm_live_rc_support_bundle_json",
    ]:
        if (artifacts.get(name) or {}).get("present") is not True:
            raise SystemExit(f"missing artifact {name}: {artifacts.get(name)}")
    scope_errors: list[str] = []
    output_request = payload.get("output_request") if isinstance(payload.get("output_request"), dict) else {}
    if output_request.get("include_output") is not False:
        scope_errors.append("output_request_include_output_mismatch")
    if output_request.get("raw_prompt_public") is not False:
        scope_errors.append("output_request_raw_prompt_public_mismatch")
    if output_request.get("raw_generated_text_public") is not False:
        scope_errors.append("output_request_raw_generated_text_public_mismatch")
    if output_request.get("generated_token_ids_public") is not False:
        scope_errors.append("output_request_generated_token_ids_public_mismatch")
    if output_request.get("raw_activation_public") is not False:
        scope_errors.append("output_request_raw_activation_public_mismatch")
    if output_request.get("public_artifact_safe") is not True:
        scope_errors.append("output_request_public_artifact_safe_mismatch")
    prompt_scope = payload.get("prompt_scope") if isinstance(payload.get("prompt_scope"), dict) else {}
    if prompt_scope.get("source") != "built-in-fixed-scenario":
        scope_errors.append("prompt_scope_source_mismatch")
    if prompt_scope.get("inline_prompt_text") is not False:
        scope_errors.append("prompt_scope_inline_prompt_text_mismatch")
    if prompt_scope.get("raw_prompt_public") is not False:
        scope_errors.append("prompt_scope_raw_prompt_public_mismatch")
    if prompt_scope.get("public_artifact_safe") is not True:
        scope_errors.append("prompt_scope_public_artifact_safe_mismatch")
    answer_scope = payload.get("answer_scope") if isinstance(payload.get("answer_scope"), dict) else {}
    if answer_scope.get("scope_state") != "no-local-answer":
        scope_errors.append("answer_scope_state_mismatch")
    if answer_scope.get("raw_generated_text_public") is not False:
        scope_errors.append("answer_scope_raw_generated_text_public_mismatch")
    if answer_scope.get("generated_token_ids_public") is not False:
        scope_errors.append("answer_scope_generated_token_ids_public_mismatch")
    if answer_scope.get("raw_activation_public") is not False:
        scope_errors.append("answer_scope_raw_activation_public_mismatch")
    if answer_scope.get("public_artifact_safe") is not True:
        scope_errors.append("answer_scope_public_artifact_safe_mismatch")
    shareable = payload.get("shareable_summary") if isinstance(payload.get("shareable_summary"), dict) else {}
    if shareable.get("saved_artifacts_public_safe") is not True:
        scope_errors.append("shareable_saved_artifacts_public_safe_mismatch")
    if shareable.get("raw_prompt_public") is not False:
        scope_errors.append("shareable_raw_prompt_public_mismatch")
    if shareable.get("raw_generated_text_public") is not False:
        scope_errors.append("shareable_raw_generated_text_public_mismatch")
    if shareable.get("generated_token_ids_public") is not False:
        scope_errors.append("shareable_generated_token_ids_public_mismatch")
    if shareable.get("raw_activation_public") is not False:
        scope_errors.append("shareable_raw_activation_public_mismatch")
    if scope_errors:
        raise SystemExit(f"micro-LLM live RC scope contract failed: {scope_errors}")
    guidance_errors: list[str] = []
    user_status = payload.get("user_status") if isinstance(payload.get("user_status"), dict) else {}
    review = payload.get("review_summary") if isinstance(payload.get("review_summary"), dict) else {}
    recommended = payload.get("recommended_next_command") if isinstance(payload.get("recommended_next_command"), dict) else {}
    next_commands = payload.get("next_commands") if isinstance(payload.get("next_commands"), list) else []
    artifact_summary = payload.get("artifact_summary") if isinstance(payload.get("artifact_summary"), dict) else {}
    not_completed = payload.get("not_completed") if isinstance(payload.get("not_completed"), list) else []
    if user_status.get("public_artifact_safe") is not True:
        guidance_errors.append("user_status_public_artifact_safe_mismatch")
    if user_status.get("state") != "ready":
        guidance_errors.append("user_status_ready_state_mismatch")
    if review.get("schema") != "micro_llm_live_rc_review_summary_v1":
        guidance_errors.append("review_summary_schema_mismatch")
    if review.get("state") != "ready":
        guidance_errors.append("review_summary_ready_state_mismatch")
    if not isinstance(recommended.get("command_line"), str) or not recommended.get("command_line"):
        guidance_errors.append("recommended_next_command_missing")
    if recommended.get("public_artifact_safe") is not True:
        guidance_errors.append("recommended_next_command_public_artifact_safe_mismatch")
    if len(next_commands) < 2:
        guidance_errors.append("next_commands_missing")
    if not_completed:
        guidance_errors.append("ready_report_not_completed")
    if artifact_summary.get("public_artifact_safe") is not True:
        guidance_errors.append("artifact_summary_public_artifact_safe_mismatch")
    if artifact_summary.get("present_artifact_count") != artifact_summary.get("artifact_count"):
        guidance_errors.append("artifact_summary_present_count_mismatch")
    if guidance_errors:
        raise SystemExit(f"micro-LLM live RC guidance contract failed: {guidance_errors}")
    assert_no_sensitive_output(payload)
    return guidance_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the micro-LLM live two-node RC check.")
    parser.add_argument("--base-port", type=int, default=9180)
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="crowdtensor_micro_llm_live_rc_") as temp:
        payload = run_json(
            [
                sys.executable,
                str(ROOT / "scripts" / "micro_llm_live_rc_pack.py"),
                "--mode",
                "local-generated",
                "--output-dir",
                str(Path(temp) / "micro-llm-live-rc"),
                "--port",
                str(args.base_port),
                "--request-count",
                str(args.request_count),
                "--decode-steps",
                str(args.decode_steps),
                "--timeout-seconds",
                str(args.timeout_seconds),
                "--json",
            ],
            timeout=args.timeout_seconds,
        )
        guidance_errors = validate_local_generated(payload)
        print(json.dumps({
            "schema": SCHEMA,
            "ok": True,
            "mode": payload.get("mode"),
            "diagnosis_codes": payload.get("diagnosis_codes") or [],
            "request_count": args.request_count,
            "decode_steps": args.decode_steps,
            "guidance_errors": guidance_errors,
        }, sort_keys=True))


if __name__ == "__main__":
    main()
