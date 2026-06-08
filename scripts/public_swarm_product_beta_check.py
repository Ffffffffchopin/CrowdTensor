#!/usr/bin/env python3
"""Acceptance checks for Public Swarm Product Beta."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import public_swarm_product_beta_pack as pack  # noqa: E402


SCHEMA = "public_swarm_product_beta_check_v1"
COMMON_REQUIRED_CODES = {
    "public_swarm_product_beta_ready",
    "public_swarm_product_beta_user_path_ready",
    "support_bundle_ready",
    "cpu_fallback_ready",
    "local_cpu_inference_ready",
    "read_only_workload",
    "not_production",
    "not_p2p",
    "not_large_model_serving",
}
LOCAL_REQUIRED_CODES = COMMON_REQUIRED_CODES | {
    "serve_ready",
    "stage0_join_ready",
    "stage1_join_ready",
    "generate_ready",
    "serve_join_generate_loop_ready",
    "remote_generate_session_ready",
    "public_swarm_generate_ready",
    "decoded_tokens_match",
    "distinct_stage_miners",
    "stage_assignment_valid",
    "private_artifacts_cleaned",
}
PACKAGE_REQUIRED_CODES = COMMON_REQUIRED_CODES | {
    "public_swarm_product_beta_package_ready",
    "miner_join_pack_ready",
    "private_artifacts_local_only",
}
EXTERNAL_REQUIRED_CODES = COMMON_REQUIRED_CODES | {
    "serve_ready",
    "stage0_join_ready",
    "stage1_join_ready",
    "generate_ready",
    "serve_join_generate_loop_ready",
    "remote_generate_session_ready",
    "public_swarm_generate_ready",
    "external_runtime_verified",
    "remote_real_llm_sharded_existing_ready",
}
SECRET_FRAGMENTS = [
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "lease_token",
    "idempotency_key",
    "Bearer ",
    "hidden_state",
    "input_ids",
    "logits",
    "activation_results",
    '"generated_text":',
    '"generated_token_ids":',
    '"prompt_text":',
]


def write(path: Path, payload: str = "{}\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def validate_payload(payload: dict[str, Any], *, mode: str, required_codes: set[str]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if payload.get("ok") is not True:
        errors.append("product_beta_not_ready")
    if payload.get("mode") != mode:
        errors.append("mode_mismatch")
    beta = payload.get("product_beta") if isinstance(payload.get("product_beta"), dict) else {}
    if beta.get("ready") is not True:
        errors.append("product_beta_summary_not_ready")
    codes = set(payload.get("diagnosis_codes") or [])
    for code in sorted(required_codes - codes):
        errors.append(f"missing_code:{code}")
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    for key in ["coordinator_backed_task_execution", "p2p_lite_discovery_only", "not_libp2p", "not_dht", "not_nat_traversal"]:
        if safety.get(key) is not True:
            errors.append(f"safety_missing:{key}")
    if safety.get("read_only_workload") != pack.WORKLOAD_TYPE:
        errors.append("safety_workload_mismatch")
    output_request = payload.get("output_request") if isinstance(payload.get("output_request"), dict) else {}
    if output_request.get("include_output") is not False:
        errors.append("output_request_include_output_mismatch")
    if output_request.get("raw_prompt_public") is not False:
        errors.append("output_request_raw_prompt_public_mismatch")
    if output_request.get("raw_generated_text_public") is not False:
        errors.append("output_request_raw_generated_text_public_mismatch")
    if output_request.get("generated_token_ids_public") is not False:
        errors.append("output_request_generated_token_ids_public_mismatch")
    if output_request.get("public_artifact_safe") is not True:
        errors.append("output_request_public_artifact_safe_mismatch")
    prompt_scope = payload.get("prompt_scope") if isinstance(payload.get("prompt_scope"), dict) else {}
    if prompt_scope.get("source") not in {"prompt-text", "prompt-texts", "prompt-texts-file"}:
        errors.append("prompt_scope_source_mismatch")
    if not isinstance(prompt_scope.get("prompt_count"), int) or prompt_scope.get("prompt_count") < 1:
        errors.append("prompt_scope_count_mismatch")
    expected_inline = prompt_scope.get("source") in {"prompt-text", "prompt-texts"}
    if prompt_scope.get("inline_prompt_text") is not expected_inline:
        errors.append("prompt_scope_inline_prompt_text_mismatch")
    if prompt_scope.get("terminal_next_commands_local_private") is not expected_inline:
        errors.append("prompt_scope_terminal_next_commands_mismatch")
    if prompt_scope.get("terminal_logs_local_private") is not expected_inline:
        errors.append("prompt_scope_terminal_logs_mismatch")
    if prompt_scope.get("saved_artifacts_prompt_placeholders") is not True:
        errors.append("prompt_scope_saved_placeholders_mismatch")
    if prompt_scope.get("saved_artifacts_public_safe") is not True:
        errors.append("prompt_scope_saved_artifacts_public_safe_mismatch")
    if prompt_scope.get("prefer_prompt_file_or_stdin_for_shareable_logs") is not True:
        errors.append("prompt_scope_shareable_log_guidance_mismatch")
    if prompt_scope.get("prompt_file_path_public") is not False:
        errors.append("prompt_scope_file_path_public_mismatch")
    if prompt_scope.get("raw_prompt_public") is not False:
        errors.append("prompt_scope_raw_prompt_public_mismatch")
    if prompt_scope.get("public_artifact_safe") is not True:
        errors.append("prompt_scope_public_artifact_safe_mismatch")
    answer_scope = payload.get("answer_scope") if isinstance(payload.get("answer_scope"), dict) else {}
    if answer_scope.get("scope_state") != "no-local-answer":
        errors.append("answer_scope_state_mismatch")
    if answer_scope.get("visible_in_terminal") is not False:
        errors.append("answer_scope_visible_in_terminal_mismatch")
    if answer_scope.get("terminal_only") is not False:
        errors.append("answer_scope_terminal_only_mismatch")
    if answer_scope.get("saved_json_display") != "hash-only":
        errors.append("answer_scope_saved_json_display_mismatch")
    if answer_scope.get("saved_markdown_display") != "hash-only":
        errors.append("answer_scope_saved_markdown_display_mismatch")
    if answer_scope.get("public_artifact_safe") is not True:
        errors.append("answer_scope_public_artifact_safe_mismatch")
    shareable = payload.get("shareable_summary") if isinstance(payload.get("shareable_summary"), dict) else {}
    if shareable.get("saved_artifacts_public_safe") is not True:
        errors.append("shareable_saved_artifacts_public_safe_mismatch")
    if shareable.get("raw_prompt_public") is not False:
        errors.append("shareable_raw_prompt_public_mismatch")
    if shareable.get("raw_generated_text_public") is not False:
        errors.append("shareable_raw_generated_text_public_mismatch")
    if shareable.get("generated_token_ids_public") is not False:
        errors.append("shareable_generated_token_ids_public_mismatch")
    if shareable.get("answer_scope_state") != "no-local-answer":
        errors.append("shareable_answer_scope_state_mismatch")
    if shareable.get("local_answer_terminal_only") is not False:
        errors.append("shareable_local_answer_terminal_only_mismatch")
    if shareable.get("public_artifact_safe") is not True:
        errors.append("shareable_public_artifact_safe_mismatch")
    provenance = payload.get("runtime_provenance") if isinstance(payload.get("runtime_provenance"), dict) else {}
    if provenance.get("schema") != pack.RUNTIME_PROVENANCE_SCHEMA:
        errors.append("runtime_provenance_schema_mismatch")
    if provenance.get("mode") != mode:
        errors.append("runtime_provenance_mode_mismatch")
    if not provenance.get("proof_level"):
        errors.append("runtime_provenance_proof_level_missing")
    if provenance.get("public_artifact_safe") is not True:
        errors.append("runtime_provenance_public_artifact_safe_mismatch")
    if provenance.get("fresh_kaggle_gpu_attempted") is not False or provenance.get("fresh_kaggle_gpu_verified") is not False:
        errors.append("runtime_provenance_fresh_kaggle_gpu_claim_mismatch")
    if mode == "local-loopback":
        if provenance.get("proof_level") != "local-loopback-cpu-product-path":
            errors.append("runtime_provenance_local_proof_level_mismatch")
        if provenance.get("local_loopback_product_path_ran") is not True:
            errors.append("runtime_provenance_local_loopback_ran_missing")
        if provenance.get("local_loopback_product_path_ready") is not True:
            errors.append("runtime_provenance_local_loopback_ready_missing")
        if provenance.get("external_existing_verified") is not False:
            errors.append("runtime_provenance_local_external_claim_mismatch")
    elif mode == "package":
        if provenance.get("proof_level") != "package-only":
            errors.append("runtime_provenance_package_proof_level_mismatch")
        if provenance.get("package_only") is not True:
            errors.append("runtime_provenance_package_only_missing")
        if provenance.get("kaggle_package_generated") is not (payload.get("target") == "kaggle"):
            errors.append("runtime_provenance_kaggle_package_mismatch")
        if provenance.get("local_loopback_product_path_ran") is not False:
            errors.append("runtime_provenance_package_local_loopback_claim_mismatch")
    elif mode == "external-existing":
        if provenance.get("proof_level") != "external-existing-runtime":
            errors.append("runtime_provenance_external_proof_level_mismatch")
        if provenance.get("external_existing_attempted") is not True:
            errors.append("runtime_provenance_external_attempt_missing")
        if provenance.get("external_existing_verified") is not True:
            errors.append("runtime_provenance_external_verified_missing")
    user_status = payload.get("user_status") if isinstance(payload.get("user_status"), dict) else {}
    if user_status.get("state") not in {"ready", "blocked", "package-blocked"}:
        errors.append("user_status_state_mismatch")
    if not user_status.get("recommended_label"):
        errors.append("user_status_recommended_missing")
    review = payload.get("review_summary") if isinstance(payload.get("review_summary"), dict) else {}
    if review.get("schema") != "public_swarm_product_beta_review_summary_v1":
        errors.append("review_summary_schema_mismatch")
    if not review.get("inspect_first"):
        errors.append("review_summary_inspect_first_missing")
    if not review.get("recommended_label"):
        errors.append("review_summary_recommended_missing")
    recommended = payload.get("recommended_next_command") if isinstance(payload.get("recommended_next_command"), dict) else {}
    if not recommended.get("label") or not recommended.get("command_line"):
        errors.append("recommended_next_command_missing")
    next_commands = payload.get("next_commands") if isinstance(payload.get("next_commands"), list) else []
    if not next_commands:
        errors.append("next_commands_missing")
    if not any(isinstance(item, dict) and item.get("label") == "inspect support bundle" for item in next_commands):
        errors.append("next_commands_support_missing")
    artifact_summary = payload.get("artifact_summary") if isinstance(payload.get("artifact_summary"), dict) else {}
    if artifact_summary.get("public_artifact_safe") is not True:
        errors.append("artifact_summary_public_safe_mismatch")
    if artifact_summary.get("present_artifact_count") != artifact_summary.get("artifact_count"):
        errors.append("artifact_summary_present_count_mismatch")
    if not isinstance(payload.get("not_completed"), list):
        errors.append("not_completed_missing")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    for name in ["public_swarm_product_beta_json", "support_bundle_json", "public_swarm_inference_beta_rc_json"]:
        artifact = artifacts.get(name) if isinstance(artifacts.get(name), dict) else {}
        if artifact.get("present") is not True:
            errors.append(f"missing_artifact:{name}")
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            errors.append(f"sensitive_fragment:{fragment}")
    return errors


def fake_rc_payload(mode: str, output_dir: Path) -> dict[str, Any]:
    write(output_dir / "public_swarm_inference_beta_rc.json")
    codes = [
        "public_swarm_inference_beta_rc_ready",
        "public_swarm_product_beta_ready",
        "p2p_lite_route_ready",
        "p2p_lite_discovery_ready",
        "cpu_fallback_ready",
        "local_cpu_inference_ready",
        "read_only_workload",
        "not_production",
    ]
    if mode == "local-loopback":
        codes.extend([
            "serve_join_generate_loop_ready",
            "remote_generate_session_ready",
            "public_swarm_generate_ready",
            "private_artifacts_cleaned",
        ])
    elif mode == "package":
        codes.extend([
            "public_swarm_beta_rc_package_ready",
            "miner_join_pack_ready",
            "private_artifacts_local_only",
            "kaggle_remote_miner_package_ready",
        ])
    else:
        codes.extend([
            "serve_join_generate_loop_ready",
            "remote_generate_session_ready",
            "public_swarm_generate_ready",
            "external_runtime_verified",
            "remote_real_llm_sharded_existing_ready",
            "decoded_tokens_match",
            "distinct_stage_miners",
            "stage_assignment_valid",
        ])
    return {
        "schema": pack.RC_SCHEMA,
        "ok": True,
        "mode": mode,
        "rc": {
            "ready": True,
            "product_beta_ready": True,
            "p2p_lite_route_ready": True,
            "cpu_fallback_ready": True,
            "mode_ready": True,
            "workload_type": pack.WORKLOAD_TYPE,
            "max_new_tokens": 2,
        },
        "diagnosis_codes": codes,
        "payload_summaries": {
            "serve_join_generate": {
                "diagnosis_codes": ["public_swarm_generate_ready"],
            }
        },
    }


def fake_split_payload(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    write(output_dir / "remote_real_llm_sharded_beta.json")
    payload = {
        "schema": pack.REMOTE_REAL_SCHEMA,
        "ok": True,
        "mode": "remote-loopback",
        "diagnosis_codes": [
            "remote_real_llm_sharded_ready",
            "remote_real_llm_sharded_loopback_ready",
            "real_llm_sharded_ready",
            "activation_transport_ready",
            "decoded_tokens_match",
            "distinct_stage_miners",
            "stage_assignment_valid",
        ],
        "payload_summaries": {
            "remote_real_llm_sharded_beta": {
                "session": {"stage_count": 2, "request_count": 1, "model_id": "sshleifer/tiny-gpt2"},
                "stage_assignment": {
                    "stage0_miner_id": "stage0",
                    "stage1_miner_id": "stage1",
                    "distinct_stage_miners": True,
                    "stage_assignment_valid": True,
                },
            }
        },
    }
    step = {"name": "real_llm_split_validation", "ok": True, "payload_schema": pack.REMOTE_REAL_SCHEMA, "payload_ok": True}
    return step, payload


def build_fake_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="crowdtensor_product_beta_check_"))
    argv = [
        args.mode,
        "--output-dir",
        str(output_dir / "product-beta"),
        "--base-port",
        str(args.base_port),
        "--target",
        args.target,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--json",
    ]
    if args.mode == "external-existing":
        argv.extend([
            "--coordinator-url",
            "http://127.0.0.1:9999",
            "--observer-token",
            "observer-secret",
            "--admin-token",
            "admin-secret",
        ])
    report_args = pack.parse_args(argv)
    original_rc = pack.run_rc_core
    original_split = pack.run_split_validation

    def fake_rc(report_args: argparse.Namespace, *, output_dir: Path, runner: pack.Runner) -> tuple[dict[str, Any], dict[str, Any]]:
        del runner
        payload = fake_rc_payload(report_args.mode, output_dir)
        step = {"name": "public_swarm_beta_rc_core", "ok": True, "payload_schema": pack.RC_SCHEMA, "payload_ok": True}
        return step, payload

    def fake_split(report_args: argparse.Namespace, *, output_dir: Path, runner: pack.Runner) -> tuple[dict[str, Any], dict[str, Any]]:
        del report_args, runner
        return fake_split_payload(output_dir)

    try:
        pack.run_rc_core = fake_rc
        pack.run_split_validation = fake_split
        return pack.build_report(report_args, runner=subprocess.run)
    finally:
        pack.run_rc_core = original_rc
        pack.run_split_validation = original_split


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_fake_report(args)
    required = {
        "local-loopback": LOCAL_REQUIRED_CODES,
        "package": PACKAGE_REQUIRED_CODES | ({"kaggle_remote_miner_package_ready"} if args.target == "kaggle" else set()),
        "external-existing": EXTERNAL_REQUIRED_CODES,
    }[args.mode]
    errors = validate_payload(payload, mode=args.mode, required_codes=required)
    output_dir = Path(args.output_dir) if args.output_dir else Path(payload.get("output_dir", tempfile.mkdtemp(prefix="crowdtensor_product_beta_check_"))).parent
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "mode": args.mode,
        "target": args.target,
        "output_dir": str(output_dir),
        "errors": errors,
        "product_beta_schema": payload.get("schema"),
        "product_beta_ok": payload.get("ok"),
        "diagnosis_codes": ["public_swarm_product_beta_check_ready"] if not errors else ["public_swarm_product_beta_check_blocked"],
        "artifacts": {
            "public_swarm_product_beta_json": str(Path(payload.get("output_dir", "")) / "public_swarm_product_beta.json")
        },
    }
    write(output_dir / "public_swarm_product_beta_check.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Public Swarm Product Beta.")
    parser.add_argument("--mode", choices=["local-loopback", "package", "external-existing"], default="local-loopback")
    parser.add_argument("--target", choices=["local", "kaggle"], default="local")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--base-port", type=int, default=9320)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.base_port < 1:
        raise SystemExit("--base-port must be positive")
    if args.max_new_tokens < 2 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 2 and 32")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    return args


def main() -> None:
    args = parse_args()
    result = run_check(args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"Public Swarm Product Beta check ready: {result.get('ok')}")
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
