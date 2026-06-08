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


def validate_report(payload: dict[str, Any]) -> list[str]:
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
    support_artifact = artifacts.get("support_bundle_json") or {}
    if support_artifact.get("present") is not True:
        raise SystemExit(f"missing CPU Inference Beta RC support bundle: {support_artifact}")
    join = payload.get("miner_join_pack") or {}
    if join.get("schema") != "miner_join_pack_v1" or join.get("ready") is not True:
        raise SystemExit(f"missing ready Miner join pack: {join}")
    safety = payload.get("safety") or {}
    for key in ["cpu_only", "read_only", "not_production", "not_p2p", "not_gpu_tpu_workload"]:
        if safety.get(key) is not True:
            raise SystemExit(f"missing safety flag {key}: {safety}")
    scope_errors: list[str] = []
    output_request = payload.get("output_request") if isinstance(payload.get("output_request"), dict) else {}
    if output_request.get("include_output") is not False:
        scope_errors.append("output_request_include_output_mismatch")
    if output_request.get("raw_prompt_public") is not False:
        scope_errors.append("output_request_raw_prompt_public_mismatch")
    if output_request.get("raw_generation_public") is not False:
        scope_errors.append("output_request_raw_generation_public_mismatch")
    if output_request.get("raw_external_llm_output_public") is not False:
        scope_errors.append("output_request_raw_external_llm_output_public_mismatch")
    if output_request.get("public_artifact_safe") is not True:
        scope_errors.append("output_request_public_artifact_safe_mismatch")
    prompt_scope = payload.get("prompt_scope") if isinstance(payload.get("prompt_scope"), dict) else {}
    if prompt_scope.get("source") != "built-in-fixed-scenarios":
        scope_errors.append("prompt_scope_source_mismatch")
    if prompt_scope.get("inline_prompt_text") is not False:
        scope_errors.append("prompt_scope_inline_prompt_text_mismatch")
    if prompt_scope.get("terminal_next_commands_local_private") is not False:
        scope_errors.append("prompt_scope_terminal_next_commands_local_private_mismatch")
    if prompt_scope.get("raw_prompt_public") is not False:
        scope_errors.append("prompt_scope_raw_prompt_public_mismatch")
    if prompt_scope.get("public_artifact_safe") is not True:
        scope_errors.append("prompt_scope_public_artifact_safe_mismatch")
    answer_scope = payload.get("answer_scope") if isinstance(payload.get("answer_scope"), dict) else {}
    if answer_scope.get("scope_state") != "no-local-answer":
        scope_errors.append("answer_scope_state_mismatch")
    if answer_scope.get("raw_generation_public") is not False:
        scope_errors.append("answer_scope_raw_generation_public_mismatch")
    if answer_scope.get("raw_external_llm_output_public") is not False:
        scope_errors.append("answer_scope_raw_external_llm_output_public_mismatch")
    if answer_scope.get("public_artifact_safe") is not True:
        scope_errors.append("answer_scope_public_artifact_safe_mismatch")
    shareable = payload.get("shareable_summary") if isinstance(payload.get("shareable_summary"), dict) else {}
    if shareable.get("saved_artifacts_public_safe") is not True:
        scope_errors.append("shareable_saved_artifacts_public_safe_mismatch")
    if shareable.get("raw_prompt_public") is not False:
        scope_errors.append("shareable_raw_prompt_public_mismatch")
    if shareable.get("raw_generation_public") is not False:
        scope_errors.append("shareable_raw_generation_public_mismatch")
    if shareable.get("raw_external_llm_output_public") is not False:
        scope_errors.append("shareable_raw_external_llm_output_public_mismatch")
    if scope_errors:
        raise SystemExit(f"CPU Inference Beta RC scope contract failed: {scope_errors}")
    guidance_errors: list[str] = []
    user_status = payload.get("user_status") if isinstance(payload.get("user_status"), dict) else {}
    review = payload.get("review_summary") if isinstance(payload.get("review_summary"), dict) else {}
    recommended = payload.get("recommended_next_command") if isinstance(payload.get("recommended_next_command"), dict) else {}
    next_commands = payload.get("next_commands") if isinstance(payload.get("next_commands"), list) else []
    artifact_report = payload.get("artifact_summary") if isinstance(payload.get("artifact_summary"), dict) else {}
    not_completed = payload.get("not_completed") if isinstance(payload.get("not_completed"), list) else []
    if user_status.get("public_artifact_safe") is not True:
        guidance_errors.append("user_status_public_artifact_safe_mismatch")
    if user_status.get("state") != "ready":
        guidance_errors.append("user_status_ready_state_mismatch")
    if review.get("schema") != "cpu_inference_beta_rc_review_summary_v1":
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
    if artifact_report.get("public_artifact_safe") is not True:
        guidance_errors.append("artifact_summary_public_artifact_safe_mismatch")
    if artifact_report.get("present_artifact_count") != artifact_report.get("artifact_count"):
        guidance_errors.append("artifact_summary_present_count_mismatch")
    if guidance_errors:
        raise SystemExit(f"CPU Inference Beta RC guidance contract failed: {guidance_errors}")
    assert_no_sensitive_output(payload)
    return guidance_errors


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
        guidance_errors = validate_report(payload)
        print(json.dumps({
            "ok": True,
            "schema": SCHEMA,
            "beta_rc_schema": payload.get("schema"),
            "step_count": len(payload.get("steps") or []),
            "diagnosis_codes": payload.get("diagnosis_codes"),
            "artifact_count": len(payload.get("artifacts") or {}),
            "guidance_errors": guidance_errors,
        }, sort_keys=True))


if __name__ == "__main__":
    main()
