#!/usr/bin/env python3
"""CI-safe checks for Public Swarm Inference Preview v0.4."""

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

import public_swarm_preview_v04_pack as pack  # noqa: E402


SCHEMA = "public_swarm_preview_v04_check_v1"
REQUIRED_CODES = {
    "public_swarm_preview_v04_ready",
    "external_two_stage_generation_ready",
    "multi_token_generation_ready",
    "distinct_stage_miners",
    "stage_assignment_valid",
    "stage_latency_ready",
    "throughput_summary_ready",
    "memory_or_vram_summary_ready",
    "external_stage_requeue_ready",
    "tiny_gpt2_ci_fallback_ready",
    "redacted_evidence_ready",
    "read_only_workload",
    "not_production",
    "not_p2p",
    "not_libp2p",
    "not_dht",
    "not_nat_traversal",
    "not_large_model_serving",
}
PACKAGE_CODES = {"kaggle_or_two_machine_runbook_ready", "miner_join_pack_ready", "private_artifacts_local_only"}
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
    "Bearer ",
    "hidden_state",
    "input_ids",
    "logits",
    "activation_results",
    '"generated_text":',
    '"generated_token_ids":',
    '"prompt_text":',
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
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


def mode_from_command(command: list[str], script_name: str, default: str) -> str:
    for index, part in enumerate(command):
        if part.endswith(script_name) and index + 1 < len(command):
            return command[index + 1]
    return default


def fake_product_mvp_payload(model_id: str = "sshleifer/tiny-gpt2") -> dict[str, Any]:
    return {
        "schema": pack.PRODUCT_MVP_SCHEMA,
        "ok": True,
        "mode": "local-loopback",
        "hf_model_id": model_id,
        "generation": {
            "generated_token_count": 2,
            "generated_text_hash": "sha256:generated",
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        },
        "performance": {
            "stage_latency_ready": True,
            "throughput_summary_ready": True,
            "stage_total_seconds": 1.0,
            "generated_tokens_per_stage_second": 2.0,
            "per_stage": {
                "stage0": {"count": 2, "avg_seconds": 0.2},
                "stage1": {"count": 2, "avg_seconds": 0.3},
            },
        },
        "runtime_resources": {
            "memory_or_vram_summary_ready": True,
            "peak_child_rss_mb": 256.0,
            "cuda_available": False,
            "vram_total_mb": [],
        },
        "diagnosis_codes": [
            "product_swarm_mvp_ready",
            "serve_join_generate_mvp_ready",
            "generated_token_count_ready",
            "distinct_stage_miners",
            "stage_assignment_valid",
            "stage_latency_ready",
            "throughput_summary_ready",
            "memory_or_vram_summary_ready",
        ],
    }


def fake_live_preview_payload(target: str) -> dict[str, Any]:
    code = f"live_{target}_requeue_ready"
    return {
        "schema": pack.LIVE_PREVIEW_SCHEMA,
        "ok": True,
        "mode": "live-kaggle",
        "failure_mode": f"kill-{target}-after-claim",
        "live_preview": {
            "ready": True,
            "external_runtime_verified": True,
            "kaggle_kernels_deleted": True,
            "private_artifacts_cleaned": True,
            "token_rotation_required": True,
        },
        "session": {
            "distinct_stage_miners": True,
            "stage_assignment_valid": True,
        },
        "diagnosis_codes": [
            "public_swarm_live_preview_rc_ready",
            "external_runtime_verified",
            "external_stage_requeue_ready",
            code,
            "kaggle_kernels_deleted",
            "private_artifacts_cleaned",
            "token_rotation_required",
            "multi_token_generation_ready",
            "distinct_stage_miners",
            "stage_assignment_valid",
        ],
    }


def fake_real_llm_evidence_payload() -> dict[str, Any]:
    return {
        "schema": pack.REAL_LLM_EVIDENCE_SCHEMA,
        "ok": True,
        "diagnosis_codes": ["stage_gpu_memory_reduced"],
        "generation": {
            "generated_token_count": 16,
            "generated_text_hash": "sha256:gpu-generated",
            "multi_token_generation_ready": True,
        },
        "stage_summary": {
            "stage_0": {
                "elapsed_ms": 1200.0,
                "stage_gpu_memory_reduced": True,
                "stage_parameter_count": 100,
                "full_model_parameter_count": 200,
            },
            "stage_1": {
                "elapsed_ms": 900.0,
                "stage_gpu_memory_reduced": True,
                "stage_parameter_count": 100,
                "full_model_parameter_count": 200,
            },
        },
    }


def fake_gpu_report(path: Path) -> None:
    public_dir = path.parent / "public-gpu-beta"
    remote_dir = public_dir / "remote-runtime"
    evidence_dir = remote_dir / "evidence"
    write_json(evidence_dir / "real_llm_sharded_evidence.json", fake_real_llm_evidence_payload())
    write_json(remote_dir / "remote_real_llm_sharded_beta.json", {
        "schema": pack.REMOTE_REAL_SCHEMA,
        "ok": True,
        "artifacts": {
            "remote_existing_real_llm_sharded_evidence_json": {
                "path": "evidence/real_llm_sharded_evidence.json",
                "present": True,
                "schema": pack.REAL_LLM_EVIDENCE_SCHEMA,
            }
        },
    })
    write_json(public_dir / "public_swarm_gpu_inference_beta_kaggle_auto.json", {
        "schema": "public_swarm_gpu_inference_beta_v1",
        "ok": True,
        "artifacts": {
            "external_remote_real_llm_sharded_beta_json": {
                "path": "remote-runtime/remote_real_llm_sharded_beta.json",
                "present": True,
                "schema": pack.REMOTE_REAL_SCHEMA,
            }
        },
    })
    write_json(path, {
        "schema": pack.GPU_GENERATION_SCHEMA,
        "ok": True,
        "mode": "kaggle-auto",
        "generation": {
            "generated_token_count": 16,
            "generated_text_hash": "sha256:gpu-generated",
            "multi_token_generation_ready": True,
            "raw_generated_text_public": False,
        },
        "diagnosis_codes": [
            "gpu_sharded_generation_ready",
            "gpu_generation_evidence_import_ready",
            "multi_token_generation_ready",
            "external_gpu_runtime_verified",
            "external_runtime_verified",
            "distinct_stage_miners",
            "stage_assignment_valid",
            "stage_gpu_memory_reduced",
            "read_only_workload",
            "not_production",
            "not_p2p",
        ],
        "artifacts": {
            "public_swarm_gpu_beta_json": {
                "path": "public-gpu-beta/public_swarm_gpu_inference_beta_kaggle_auto.json",
                "present": True,
                "schema": "public_swarm_gpu_inference_beta_v1",
            }
        },
    })


def fake_product_beta_payload() -> dict[str, Any]:
    return {
        "schema": pack.PRODUCT_BETA_SCHEMA,
        "ok": True,
        "mode": "package",
        "diagnosis_codes": [
            "public_swarm_product_beta_ready",
            "public_swarm_product_beta_package_ready",
            "miner_join_pack_ready",
            "private_artifacts_local_only",
            "support_bundle_ready",
            "read_only_workload",
        ],
    }


def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    joined = " ".join(command)
    output_dir = Path(option_value(command, "--output-dir", tempfile.mkdtemp(prefix="crowdtensor_preview_v04_step_")))
    output_dir.mkdir(parents=True, exist_ok=True)
    if "product_swarm_mvp_check.py" in joined:
        model_id = option_value(command, "--hf-model-id", "sshleifer/tiny-gpt2")
        payload = fake_product_mvp_payload(model_id)
        write_json(output_dir / "product_swarm_mvp_check.json", payload)
        return completed(payload)
    if "public_swarm_product_beta_pack.py" in joined:
        payload = fake_product_beta_payload()
        write_json(output_dir / "public_swarm_product_beta.json", payload)
        return completed(payload)
    raise AssertionError(command)


def validate_payload(payload: dict[str, Any], *, mode: str, require_optional: bool) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if payload.get("ok") is not True:
        errors.append("preview_v04_not_ready")
    if payload.get("mode") != mode:
        errors.append("mode_mismatch")
    preview = payload.get("preview") if isinstance(payload.get("preview"), dict) else {}
    for key in [
        "ready",
        "external_two_stage_generation_ready",
        "external_stage_requeue_ready",
        "multi_token_generation_ready",
        "distinct_stage_miners",
        "stage_assignment_valid",
        "stage_latency_ready",
        "throughput_summary_ready",
        "memory_or_vram_summary_ready",
        "tiny_gpt2_ci_fallback_ready",
    ]:
        if preview.get(key) is not True:
            errors.append(f"preview_missing:{key}")
    if require_optional and preview.get("optional_model_ready") is not True:
        errors.append("preview_missing:optional_model_ready")
    codes = set(payload.get("diagnosis_codes") or [])
    required = set(REQUIRED_CODES)
    if mode == pack.MODE_PACKAGE:
        required.update(PACKAGE_CODES)
    if require_optional:
        required.add("optional_distilgpt2_or_gpt2_strict_ready")
    for code in sorted(required - codes):
        errors.append(f"missing_code:{code}")
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    for key in ["coordinator_backed_task_execution", "not_libp2p", "not_dht", "not_nat_traversal", "not_hivemind_or_petals_parity", "not_large_model_serving"]:
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
    if prompt_scope.get("source") != "prompt-text":
        errors.append("prompt_scope_source_mismatch")
    if prompt_scope.get("prompt_count") != 1:
        errors.append("prompt_scope_count_mismatch")
    if prompt_scope.get("inline_prompt_text") is not True:
        errors.append("prompt_scope_inline_prompt_text_mismatch")
    if prompt_scope.get("terminal_next_commands_local_private") is not True:
        errors.append("prompt_scope_terminal_next_commands_local_private_mismatch")
    if prompt_scope.get("terminal_logs_local_private") is not True:
        errors.append("prompt_scope_terminal_logs_local_private_mismatch")
    if prompt_scope.get("saved_artifacts_prompt_placeholders") is not True:
        errors.append("prompt_scope_saved_artifacts_prompt_placeholders_mismatch")
    if prompt_scope.get("saved_artifacts_public_safe") is not True:
        errors.append("prompt_scope_saved_artifacts_public_safe_mismatch")
    if prompt_scope.get("prefer_prompt_file_or_stdin_for_shareable_logs") is not True:
        errors.append("prompt_scope_prefer_prompt_file_or_stdin_mismatch")
    if prompt_scope.get("prompt_file_path_public") is not False:
        errors.append("prompt_scope_prompt_file_path_public_mismatch")
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
    user_status = payload.get("user_status") if isinstance(payload.get("user_status"), dict) else {}
    if user_status.get("state") not in {"ready", "blocked", "package-blocked"}:
        errors.append("user_status_state_mismatch")
    if not user_status.get("headline"):
        errors.append("user_status_headline_missing")
    if not user_status.get("recommended_label"):
        errors.append("user_status_recommended_label_missing")
    if user_status.get("public_artifact_safe") is not True:
        errors.append("user_status_public_artifact_safe_mismatch")
    review = payload.get("review_summary") if isinstance(payload.get("review_summary"), dict) else {}
    if review.get("schema") != "public_swarm_preview_v04_review_summary_v1":
        errors.append("review_schema_mismatch")
    if not review.get("inspect_first"):
        errors.append("review_inspect_first_missing")
    if not review.get("support_bundle"):
        errors.append("review_support_bundle_missing")
    if not review.get("recommended_label"):
        errors.append("review_recommended_label_missing")
    if not review.get("next_command"):
        errors.append("review_next_command_missing")
    if review.get("public_artifact_safe") is not True:
        errors.append("review_public_artifact_safe_mismatch")
    recommended = payload.get("recommended_next_command") if isinstance(payload.get("recommended_next_command"), dict) else {}
    if not recommended.get("label"):
        errors.append("recommended_label_missing")
    if not recommended.get("command_line"):
        errors.append("recommended_command_missing")
    if recommended.get("public_artifact_safe") is not True:
        errors.append("recommended_public_artifact_safe_mismatch")
    next_commands = payload.get("next_commands") if isinstance(payload.get("next_commands"), list) else []
    if not next_commands:
        errors.append("next_commands_missing")
    if not any(isinstance(item, dict) and item.get("label") == "inspect support bundle" for item in next_commands):
        errors.append("next_commands_support_bundle_missing")
    artifact_summary = payload.get("artifact_summary") if isinstance(payload.get("artifact_summary"), dict) else {}
    if artifact_summary.get("public_artifact_safe") is not True:
        errors.append("artifact_summary_public_artifact_safe_mismatch")
    if not artifact_summary.get("support_bundle"):
        errors.append("artifact_summary_support_bundle_missing")
    if artifact_summary.get("present_artifact_count") != artifact_summary.get("artifact_count"):
        errors.append("artifact_summary_present_count_mismatch")
    if not isinstance(payload.get("not_completed"), list):
        errors.append("not_completed_list_missing")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    for name in ["public_swarm_preview_v04_json", "public_swarm_preview_v04_markdown", "support_bundle_json"]:
        artifact = artifacts.get(name) if isinstance(artifacts.get(name), dict) else {}
        if artifact.get("present") is not True:
            errors.append(f"missing_artifact:{name}")
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            errors.append(f"sensitive_fragment:{fragment}")
    return errors


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="crowdtensor_preview_v04_check_"))
    output_dir.mkdir(parents=True, exist_ok=True)
    product_report = output_dir / "product_mvp.json"
    optional_report = output_dir / "optional_mvp.json"
    stage0_report = output_dir / "stage0_live_preview.json"
    stage1_report = output_dir / "stage1_live_preview.json"
    gpu_report = output_dir / "gpu_generation.json"
    write_json(product_report, fake_product_mvp_payload())
    write_json(optional_report, fake_product_mvp_payload(args.optional_model_id))
    write_json(stage0_report, fake_live_preview_payload("stage0"))
    write_json(stage1_report, fake_live_preview_payload("stage1"))
    fake_gpu_report(gpu_report)

    argv = [
        args.mode,
        "--output-dir",
        str(output_dir / "preview-v04"),
        "--backend",
        args.backend,
        "--gpu-report",
        str(gpu_report),
        "--live-stage0-report",
        str(stage0_report),
        "--live-stage1-report",
        str(stage1_report),
        "--product-mvp-report",
        str(product_report),
        "--optional-model-report",
        str(optional_report),
        "--optional-model-id",
        args.optional_model_id,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.timeout_seconds),
        "--cpu-timeout-seconds",
        str(args.timeout_seconds),
        "--miner-timeout",
        str(args.timeout_seconds),
        "--generate-timeout",
        str(args.timeout_seconds),
        "--json",
    ]
    if args.mode == pack.MODE_PACKAGE:
        argv.extend(["--target", "kaggle"])
    if args.mode in {pack.MODE_LOCAL_SMOKE, pack.MODE_PACKAGE} and args.run_optional_model:
        argv.append("--run-optional-model")
    if args.require_optional_model_ready:
        argv.append("--require-optional-model-ready")
    parsed = pack.parse_args(argv)
    report = pack.build_report(parsed, runner=fake_runner)
    errors = validate_payload(report, mode=args.mode, require_optional=args.require_optional_model_ready)
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "mode": args.mode,
        "backend": args.backend,
        "output_dir": str(output_dir),
        "errors": errors,
        "public_swarm_preview_v04_schema": report.get("schema"),
        "public_swarm_preview_v04_ok": report.get("ok"),
        "diagnosis_codes": ["public_swarm_preview_v04_check_ready"] if not errors else ["public_swarm_preview_v04_check_failed"],
        "artifacts": {
            "public_swarm_preview_v04_json": str(output_dir / "preview-v04" / "public_swarm_preview_v04.json"),
            "public_swarm_preview_v04_markdown": str(output_dir / "preview-v04" / "public_swarm_preview_v04.md"),
            "support_bundle_json": str(output_dir / "preview-v04" / "support_bundle.json"),
        },
    }
    write_json(output_dir / "public_swarm_preview_v04_check.json", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Public Swarm Inference Preview v0.4.")
    parser.add_argument("--mode", choices=pack.MODES, default=pack.MODE_EVIDENCE_IMPORT)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--optional-model-id", choices=["distilgpt2", "gpt2"], default=pack.DEFAULT_OPTIONAL_MODEL_ID)
    parser.add_argument("--run-optional-model", action="store_true")
    parser.add_argument("--require-optional-model-ready", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 2 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 2 and 32")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    return args


def main() -> None:
    report = run_check(parse_args())
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
