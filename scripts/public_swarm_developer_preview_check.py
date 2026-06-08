#!/usr/bin/env python3
"""Acceptance checks for Public Swarm Developer Preview."""

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

import public_swarm_developer_preview_pack as pack  # noqa: E402


SCHEMA = "public_swarm_developer_preview_check_v1"
COMMON_REQUIRED_CODES = {
    "developer_preview_ready",
    "public_swarm_developer_preview_ready",
    "product_beta_ready",
    "support_bundle_ready",
    "cpu_fallback_ready",
    "local_cpu_inference_ready",
    "read_only_workload",
    "not_production",
    "not_p2p",
    "not_large_model_serving",
}
LOCAL_REQUIRED_CODES = COMMON_REQUIRED_CODES | {
    "developer_preview_local_ready",
    "local_two_stage_generation_ready",
    "serve_join_generate_ready",
    "serve_ready",
    "stage0_join_ready",
    "stage1_join_ready",
    "generate_ready",
    "decoded_tokens_match",
    "distinct_stage_miners",
    "stage_assignment_valid",
    "private_artifacts_cleaned",
}
PACKAGE_REQUIRED_CODES = COMMON_REQUIRED_CODES | {
    "developer_preview_package_ready",
    "miner_join_pack_ready",
    "private_artifacts_local_only",
}
EXTERNAL_REQUIRED_CODES = COMMON_REQUIRED_CODES | {
    "developer_preview_external_ready",
    "external_runtime_verified",
    "remote_real_llm_sharded_existing_ready",
}
IMPORT_REQUIRED_CODES = COMMON_REQUIRED_CODES | {
    "developer_preview_evidence_import_ready",
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_payload(payload: dict[str, Any], *, mode: str, required_codes: set[str]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if payload.get("ok") is not True:
        errors.append("developer_preview_not_ready")
    if payload.get("mode") != mode:
        errors.append("mode_mismatch")
    preview = payload.get("developer_preview") if isinstance(payload.get("developer_preview"), dict) else {}
    if preview.get("ready") is not True:
        errors.append("developer_preview_summary_not_ready")
    codes = set(payload.get("diagnosis_codes") or [])
    for code in sorted(required_codes - codes):
        errors.append(f"missing_code:{code}")
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    for key in [
        "coordinator_backed_task_execution",
        "p2p_lite_discovery_only",
        "not_libp2p",
        "not_dht",
        "not_nat_traversal",
        "not_large_model_serving",
    ]:
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
    if prompt_scope.get("source") not in {"prompt-text", "prompt-texts"}:
        errors.append("prompt_scope_source_mismatch")
    if not isinstance(prompt_scope.get("prompt_count"), int) or prompt_scope.get("prompt_count") < 1:
        errors.append("prompt_scope_count_mismatch")
    if prompt_scope.get("inline_prompt_text") is not True:
        errors.append("prompt_scope_inline_prompt_text_mismatch")
    if prompt_scope.get("terminal_next_commands_local_private") is not True:
        errors.append("prompt_scope_terminal_next_commands_mismatch")
    if prompt_scope.get("terminal_logs_local_private") is not True:
        errors.append("prompt_scope_terminal_logs_mismatch")
    if prompt_scope.get("saved_artifacts_prompt_placeholders") is not True:
        errors.append("prompt_scope_saved_placeholders_mismatch")
    if prompt_scope.get("saved_artifacts_public_safe") is not True:
        errors.append("prompt_scope_saved_artifacts_public_safe_mismatch")
    if prompt_scope.get("prefer_prompt_file_or_stdin_for_shareable_logs") is not True:
        errors.append("prompt_scope_shareable_log_guidance_mismatch")
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
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    for name in ["public_swarm_developer_preview_json", "support_bundle_json", "public_swarm_product_beta_json"]:
        artifact = artifacts.get(name) if isinstance(artifacts.get(name), dict) else {}
        if artifact.get("present") is not True:
            errors.append(f"missing_artifact:{name}")
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            errors.append(f"sensitive_fragment:{fragment}")
    return errors


def fake_product_payload(mode: str, output_dir: Path) -> dict[str, Any]:
    write_json(output_dir / "public_swarm_product_beta.json", {"schema": pack.PRODUCT_BETA_SCHEMA, "ok": True})
    codes = [
        "public_swarm_product_beta_ready",
        "public_swarm_product_beta_user_path_ready",
        "support_bundle_ready",
        "p2p_lite_route_ready",
        "p2p_lite_discovery_ready",
        "cpu_fallback_ready",
        "local_cpu_inference_ready",
        "read_only_workload",
        "not_production",
        "not_p2p",
        "not_large_model_serving",
    ]
    if mode == "local":
        product_mode = "local-loopback"
        codes.extend([
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
        ])
    elif mode == "package":
        product_mode = "package"
        codes.extend([
            "public_swarm_product_beta_package_ready",
            "miner_join_pack_ready",
            "private_artifacts_local_only",
        ])
    else:
        product_mode = "external-existing"
        codes.extend([
            "external_runtime_verified",
            "remote_real_llm_sharded_existing_ready",
        ])
    return {
        "schema": pack.PRODUCT_BETA_SCHEMA,
        "ok": True,
        "mode": product_mode,
        "product_beta": {
            "ready": True,
            "mode_ready": True,
            "support_bundle_ready": True,
            "privacy_ready": True,
            "workload_type": pack.WORKLOAD_TYPE,
            "hf_model_id": "sshleifer/tiny-gpt2",
            "max_new_tokens": 2,
        },
        "prompt_scope": {
            "source": "prompt-text",
            "prompt_count": 1,
            "inline_prompt_text": True,
            "terminal_next_commands_local_private": True,
            "terminal_logs_local_private": True,
            "saved_artifacts_prompt_placeholders": True,
            "saved_artifacts_public_safe": True,
            "prefer_prompt_file_or_stdin_for_shareable_logs": True,
            "raw_prompt_public": False,
            "public_artifact_safe": True,
        },
        "diagnosis_codes": codes,
        "payload_summaries": {
            "real_llm_split_validation": {
                "stage_assignment": {
                    "stage0_miner_id": "stage0",
                    "stage1_miner_id": "stage1",
                    "distinct_stage_miners": True,
                    "stage_assignment_valid": True,
                }
            }
        },
    }


def fake_gpu_report(path: Path) -> None:
    write_json(path, {
        "schema": pack.GPU_GENERATION_SCHEMA,
        "ok": True,
        "diagnosis_codes": ["gpu_sharded_generation_ready", "multi_token_generation_ready"],
        "generated_token_count": 16,
        "generated_text_hash": "abc123",
        "raw_generated_text_public": False,
    })


def build_fake_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="crowdtensor_preview_check_"))
    gpu_report = output_dir / "gpu_report.json"
    fake_gpu_report(gpu_report)
    if args.mode == "evidence-import":
        product_report = output_dir / "product_beta_import.json"
        write_json(product_report, fake_product_payload("local", output_dir / "import-product"))
        report_args = pack.parse_args([
            "evidence-import",
            "--output-dir",
            str(output_dir / "preview"),
            "--product-beta-report",
            str(product_report),
            "--gpu-report",
            str(gpu_report),
            "--json",
        ])
        return pack.build_report(report_args, runner=subprocess.run)

    argv = [
        args.mode,
        "--output-dir",
        str(output_dir / "preview"),
        "--base-port",
        str(args.base_port),
        "--target",
        args.target,
        "--gpu-report",
        str(gpu_report),
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
    original = pack.run_product_beta

    def fake_product(report_args: argparse.Namespace, *, output_dir: Path, runner: pack.Runner) -> tuple[dict[str, Any], dict[str, Any]]:
        del runner
        payload = fake_product_payload(report_args.mode, output_dir)
        step = {
            "name": "public_swarm_product_beta",
            "ok": True,
            "payload_schema": pack.PRODUCT_BETA_SCHEMA,
            "payload_ok": True,
        }
        return step, payload

    try:
        pack.run_product_beta = fake_product
        return pack.build_report(report_args, runner=subprocess.run)
    finally:
        pack.run_product_beta = original


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_fake_report(args)
    required = {
        "local": LOCAL_REQUIRED_CODES,
        "package": PACKAGE_REQUIRED_CODES,
        "external-existing": EXTERNAL_REQUIRED_CODES,
        "evidence-import": IMPORT_REQUIRED_CODES,
    }[args.mode]
    errors = validate_payload(payload, mode=args.mode, required_codes=required)
    output_dir = Path(payload.get("output_dir") or args.output_dir or tempfile.mkdtemp(prefix="crowdtensor_preview_check_"))
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "mode": args.mode,
        "target": args.target,
        "output_dir": str(output_dir.parent if output_dir.name == "preview" else output_dir),
        "developer_preview_ok": payload.get("ok"),
        "developer_preview_schema": payload.get("schema"),
        "errors": errors,
        "artifacts": {
            "public_swarm_developer_preview_json": str(output_dir / "public_swarm_developer_preview.json"),
        },
        "diagnosis_codes": ["public_swarm_developer_preview_check_ready"] if not errors else ["public_swarm_developer_preview_check_failed"],
    }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Public Swarm Developer Preview contract.")
    parser.add_argument("--mode", choices=["local", "package", "external-existing", "evidence-import"], default="local")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--base-port", type=int, default=9330)
    parser.add_argument("--target", choices=["local", "kaggle"], default="local")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    result = run_check(parse_args())
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
