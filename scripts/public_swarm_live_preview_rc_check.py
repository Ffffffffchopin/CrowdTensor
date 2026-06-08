#!/usr/bin/env python3
"""CI-safe checks for Public Swarm Live Preview RC."""

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

import public_swarm_live_preview_rc_pack as pack  # noqa: E402


SCHEMA = "public_swarm_live_preview_rc_check_v1"
SECRET_FRAGMENTS = [
    "operator-secret",
    "admin-secret",
    "stage0-secret",
    "stage1-secret",
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "lease_token",
    "idempotency_key",
    "hidden_state",
    "input_ids",
    "logits",
    "activation_results",
    "real_llm_sharded_result",
    '"generated_text":',
    '"generated_token_ids":',
    "Bearer ",
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def completed(payload: dict[str, Any], *, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


def option_value(command: list[str], option: str, default: str = "") -> str:
    if option not in command:
        return default
    index = command.index(option) + 1
    return command[index] if index < len(command) else default


def fake_gpu_report(path: Path) -> None:
    write_json(path, {
        "schema": pack.GPU_GENERATION_SCHEMA,
        "ok": True,
        "diagnosis_codes": ["gpu_sharded_generation_ready", "multi_token_generation_ready"],
        "generated_token_count": 16,
        "generated_text_hash": "abc123",
        "raw_generated_text_public": False,
    })


def fake_developer_preview_payload(mode: str = "local") -> dict[str, Any]:
    codes = [
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
    ]
    if mode == "package":
        codes.extend(["developer_preview_package_ready", "miner_join_pack_ready", "private_artifacts_local_only"])
    else:
        codes.extend(["developer_preview_local_ready", "local_two_stage_generation_ready", "serve_join_generate_ready"])
    return {
        "schema": pack.DEVELOPER_PREVIEW_SCHEMA,
        "ok": True,
        "mode": mode,
        "developer_preview": {
            "ready": True,
            "mode_ready": True,
            "product_beta_ready": True,
            "support_bundle_ready": True,
            "cpu_fallback_ready": True,
            "workload_type": pack.WORKLOAD_TYPE,
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
    }


def fake_developer_preview_check_payload() -> dict[str, Any]:
    return {
        "schema": pack.DEVELOPER_PREVIEW_CHECK_SCHEMA,
        "ok": True,
        "mode": "local",
        "diagnosis_codes": [
            "public_swarm_developer_preview_check_ready",
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
        ],
    }


def fake_alpha_check_payload() -> dict[str, Any]:
    return {
        "schema": pack.ALPHA_CHECK_SCHEMA,
        "ok": True,
        "diagnosis_codes": [
            "public_swarm_inference_alpha_check_ready",
            "public_swarm_inference_alpha_ready",
            "public_swarm_session_ready",
            "local_stage_requeue_ready",
            "stage_requeue_ready",
            "decoded_tokens_match",
            "distinct_stage_miners",
            "stage_assignment_valid",
        ],
    }


def fake_alpha_payload(failure_mode: str) -> dict[str, Any]:
    target = "stage1" if failure_mode == pack.FAILURE_KILL_STAGE1_AFTER_CLAIM else "stage0"
    requeue_enabled = failure_mode != pack.FAILURE_NONE
    codes = [
        "public_swarm_inference_alpha_ready",
        "public_swarm_session_ready",
        "public_swarm_live_kaggle_ready",
        "stage_requeue_ready",
        "local_stage_requeue_ready",
        "external_runtime_verified",
        "kaggle_kernels_deleted",
        "decoded_tokens_match",
        "distinct_stage_miners",
        "stage_assignment_valid",
        "token_rotation_required",
    ]
    if requeue_enabled:
        codes.extend(["public_swarm_live_requeue_ready", "external_stage_requeue_ready", f"live_{target}_requeue_ready"])
    return {
        "schema": pack.ALPHA_SCHEMA,
        "ok": True,
        "mode": "live-kaggle",
        "failure_mode": failure_mode,
        "session": {
            "live_external_runtime_verified": True,
            "decoded_tokens_match": True,
            "distinct_stage_miners": True,
            "stage_assignment_valid": True,
            "local_stage_requeue_verified": True,
            "live_stage_requeue_verified": requeue_enabled,
            "live_kaggle_kernels_deleted": True,
        },
        "artifact_cleanup": {"child_artifacts_pruned": True},
        "diagnosis_codes": codes,
        "safety": {
            "cpu_only": True,
            "read_only_workload": pack.WORKLOAD_TYPE,
            "not_production": True,
            "not_p2p": True,
            "not_large_model_serving": True,
        },
    }


def fake_alpha_rc_payload() -> dict[str, Any]:
    return {
        "schema": pack.ALPHA_RC_SCHEMA,
        "ok": True,
        "mode": "evidence-import",
        "release_candidate": {
            "ready": True,
            "evidence_imported": True,
            "stage0_live_requeue_ready": True,
            "stage1_live_requeue_ready": True,
        },
        "diagnosis_codes": [
            "public_swarm_inference_alpha_rc_ready",
            "public_swarm_alpha_rc_evidence_imported",
            "stage0_live_requeue_evidence_ready",
            "stage1_live_requeue_evidence_ready",
            "public_swarm_live_requeue_evidence_ready",
            "public_swarm_alpha_private_artifacts_absent",
            "external_runtime_verified",
            "kaggle_kernels_deleted",
            "decoded_tokens_match",
            "distinct_stage_miners",
            "stage_assignment_valid",
            "token_rotation_required",
        ],
    }


def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    joined = " ".join(command)
    if "public_swarm_developer_preview_check.py" in joined:
        return completed(fake_developer_preview_check_payload())
    if "public_swarm_inference_alpha_check.py" in joined:
        return completed(fake_alpha_check_payload())
    if "public_swarm_developer_preview_pack.py" in joined:
        output_dir = Path(option_value(command, "--output-dir"))
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = fake_developer_preview_payload("package")
        write_json(output_dir / "public_swarm_developer_preview.json", payload)
        return completed(payload)
    if "public_swarm_inference_alpha_pack.py" in joined:
        output_dir = Path(option_value(command, "--output-dir"))
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_mode = option_value(command, "--failure-mode", pack.FAILURE_KILL_STAGE0_AFTER_CLAIM)
        payload = fake_alpha_payload(failure_mode)
        write_json(output_dir / "public_swarm_inference_alpha.json", payload)
        return completed(payload)
    raise AssertionError(command)


def validate_payload(payload: dict[str, Any], *, mode: str) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if payload.get("ok") is not True:
        errors.append("report_not_ready")
    if payload.get("mode") != mode:
        errors.append("mode_mismatch")
    codes = set(payload.get("diagnosis_codes") or [])
    common = {
        "public_swarm_live_preview_rc_ready",
        "developer_preview_ready",
        "public_swarm_developer_preview_ready",
        "support_bundle_ready",
        "read_only_workload",
        "not_production",
        "not_p2p",
        "not_large_model_serving",
    }
    required = set(common)
    if mode == pack.MODE_LOCAL_SMOKE:
        required.update({"public_swarm_live_preview_local_smoke_ready", "public_swarm_live_preview_contract_ready"})
    elif mode == pack.MODE_PACKAGE:
        required.update({"public_swarm_live_preview_package_ready", "miner_join_pack_ready", "private_artifacts_local_only"})
    elif mode == pack.MODE_LIVE_KAGGLE:
        required.update({
            "public_swarm_live_preview_live_kaggle_ready",
            "public_swarm_live_kaggle_ready",
            "external_runtime_verified",
            "kaggle_kernels_deleted",
            "token_rotation_required",
            "external_stage_requeue_ready",
            "private_artifacts_cleaned",
        })
    else:
        required.update({
            "public_swarm_live_preview_evidence_import_ready",
            "public_swarm_alpha_rc_evidence_imported",
            "public_swarm_live_kaggle_ready",
            "external_runtime_verified",
            "private_artifacts_cleaned",
        })
    for code in sorted(required - codes):
        errors.append(f"missing_code:{code}")
    if mode == pack.MODE_LIVE_KAGGLE and not ({"live_stage0_requeue_ready", "live_stage1_requeue_ready"} & codes):
        errors.append("missing_code:live_stage_requeue_ready")
    preview = payload.get("live_preview") if isinstance(payload.get("live_preview"), dict) else {}
    if preview.get("ready") is not True:
        errors.append("live_preview_summary_not_ready")
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
    if output_request.get("local_output_display_only") is not False:
        errors.append("output_request_local_output_display_only_mismatch")
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
    if answer_scope.get("raw_prompt_public") is not False:
        errors.append("answer_scope_raw_prompt_public_mismatch")
    if answer_scope.get("raw_generated_text_public") is not False:
        errors.append("answer_scope_raw_generated_text_public_mismatch")
    if answer_scope.get("generated_token_ids_public") is not False:
        errors.append("answer_scope_generated_token_ids_public_mismatch")
    if answer_scope.get("public_artifact_safe") is not True:
        errors.append("answer_scope_public_artifact_safe_mismatch")
    shareable = payload.get("shareable_summary") if isinstance(payload.get("shareable_summary"), dict) else {}
    for key in [
        "saved_artifacts_public_safe",
        "public_artifact_safe",
    ]:
        if shareable.get(key) is not True:
            errors.append(f"shareable_{key}_mismatch")
    for key in [
        "raw_prompt_public",
        "raw_generated_text_public",
        "generated_token_ids_public",
        "local_output_display_only",
        "local_answer_terminal_only",
    ]:
        if shareable.get(key) is not False:
            errors.append(f"shareable_{key}_mismatch")
    if shareable.get("answer_scope_state") != "no-local-answer":
        errors.append("shareable_answer_scope_state_mismatch")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    for name in ["public_swarm_live_preview_rc_json", "public_swarm_live_preview_rc_markdown", "support_bundle_json"]:
        artifact = artifacts.get(name) if isinstance(artifacts.get(name), dict) else {}
        if artifact.get("present") is not True:
            errors.append(f"missing_artifact:{name}")
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            errors.append(f"sensitive_fragment:{fragment}")
    return errors


def build_fake_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="crowdtensor_live_preview_check_"))
    gpu_report = output_dir / "gpu_report.json"
    fake_gpu_report(gpu_report)
    if args.mode == pack.MODE_EVIDENCE_IMPORT:
        developer_report = output_dir / "developer_preview.json"
        alpha_rc_report = output_dir / "alpha_rc.json"
        write_json(developer_report, fake_developer_preview_payload("local"))
        write_json(alpha_rc_report, fake_alpha_rc_payload())
        report_args = pack.parse_args([
            args.mode,
            "--output-dir",
            str(output_dir / "live-preview"),
            "--developer-preview-report",
            str(developer_report),
            "--alpha-rc-report",
            str(alpha_rc_report),
            "--gpu-report",
            str(gpu_report),
            "--json",
        ])
        return pack.build_report(report_args, runner=fake_runner)

    argv = [
        args.mode,
        "--output-dir",
        str(output_dir / "live-preview"),
        "--base-port",
        str(args.base_port),
        "--port",
        str(args.port),
        "--public-host",
        "24.199.118.54",
        "--target",
        "kaggle",
        "--gpu-report",
        str(gpu_report),
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.timeout_seconds),
        "--kaggle-owner",
        "xuyuhaosuyi",
        "--failure-mode",
        args.failure_mode,
        "--json",
    ]
    report_args = pack.parse_args(argv)
    return pack.build_report(report_args, runner=fake_runner)


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_fake_report(args)
    errors = validate_payload(payload, mode=args.mode)
    output_dir = Path(payload.get("output_dir") or args.output_dir or tempfile.mkdtemp(prefix="crowdtensor_live_preview_check_"))
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "mode": args.mode,
        "output_dir": str(output_dir.parent if output_dir.name == "live-preview" else output_dir),
        "live_preview_schema": payload.get("schema"),
        "live_preview_ok": payload.get("ok"),
        "errors": errors,
        "artifacts": {
            "public_swarm_live_preview_rc_json": str(output_dir / "public_swarm_live_preview_rc.json"),
        },
        "diagnosis_codes": ["public_swarm_live_preview_rc_check_ready"] if not errors else ["public_swarm_live_preview_rc_check_failed"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Public Swarm Live Preview RC contract.")
    parser.add_argument("--mode", choices=[pack.MODE_LOCAL_SMOKE, pack.MODE_PACKAGE, pack.MODE_LIVE_KAGGLE, pack.MODE_EVIDENCE_IMPORT], default=pack.MODE_LOCAL_SMOKE)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--base-port", type=int, default=9340)
    parser.add_argument("--port", type=int, default=9341)
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--failure-mode", choices=sorted(pack.FAILURE_MODES), default=pack.FAILURE_KILL_STAGE0_AFTER_CLAIM)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    result = run_check(parse_args())
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
