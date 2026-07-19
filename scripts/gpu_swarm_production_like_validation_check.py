#!/usr/bin/env python3
"""Validate GPU Swarm production-like validation RC evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import gpu_swarm_production_like_validation_pack as pack  # noqa: E402


SCHEMA = "gpu_swarm_production_like_validation_check_v1"


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def validate_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("ok") is not True:
        errors.append("report_not_ok")
    for field in [
        "gpu_swarm_production_like_validation_ready",
        "production_like_workload_ready",
        "larger_model_attempted",
        "multi_token_decode_ready",
        "batch_or_multi_request_ready",
        "two_gpu_stage_route_ready",
        "distinct_stage_miners_ready",
        "stage_requeue_or_failure_recovery_ready",
        "gpu_runtime_readiness_checked",
        "stage_owned_weight_loading_ready",
        "latency_throughput_summary_ready",
        "network_activation_transfer_summary_ready",
        "public_artifact_safe",
        "retained_evidence_imported",
    ]:
        if report.get(field) is not True:
            errors.append(f"{field}_missing")
    for field in [
        "largest_successful_model_tier",
        "largest_attempted_model_tier",
        "larger_model_blocked_reason",
        "execution_mode",
    ]:
        if not str(report.get(field) or "").strip():
            errors.append(f"{field}_missing")
    if report.get("execution_mode") not in pack.EXECUTION_MODES:
        errors.append("execution_mode_invalid")
    if not isinstance(report.get("external_runtime_verified"), bool):
        errors.append("external_runtime_verified_not_boolean")
    if not isinstance(report.get("fresh_gpu_run_performed"), bool):
        errors.append("fresh_gpu_run_performed_not_boolean")

    boundaries = _dict(report.get("boundaries"))
    for name in pack.BOUNDARIES:
        if boundaries.get(name) is not True:
            errors.append(f"boundary_missing:{name}")

    diagnosis = set(report.get("diagnosis_codes") or [])
    for code in [
        "gpu_swarm_production_like_validation_ready",
        "production_like_workload_ready",
        "larger_model_attempted",
        "larger_model_preflight_blocked",
        "multi_token_decode_ready",
        "batch_or_multi_request_ready",
        "two_gpu_stage_route_ready",
        "distinct_stage_miners_ready",
        "stage_requeue_or_failure_recovery_ready",
        "gpu_runtime_readiness_checked",
        "stage_owned_weight_loading_ready",
        "latency_throughput_summary_ready",
        "retained_evidence_imported",
        "fresh_gpu_run_not_performed",
        "gpu_swarm_production_public_artifact_redaction_ready",
    ]:
        if code not in diagnosis:
            errors.append(f"diagnosis_missing:{code}")

    workload = _dict(report.get("production_like_workload"))
    if workload.get("generated_token_count", 0) < 16:
        errors.append("workload_generated_token_count_below_target")
    if workload.get("request_count", 0) < 2:
        errors.append("workload_request_count_below_target")
    if _dict(workload.get("network_activation_transfer")).get("raw_activations_public") is not False:
        errors.append("network_activation_scope_mismatch")

    attempt = _dict(report.get("larger_model_attempt"))
    feasibility = _dict(attempt.get("feasibility"))
    memory = _dict(attempt.get("memory_estimate"))
    hardware = _dict(attempt.get("hardware_profile"))
    if attempt.get("largest_attempted_model_tier") != report.get("largest_attempted_model_tier"):
        errors.append("larger_attempt_tier_mismatch")
    if feasibility.get("feasible_on_current_retained_profile") is not False:
        errors.append("larger_attempt_should_record_current_profile_blocker")
    if not feasibility.get("larger_model_blocked_reason"):
        errors.append("larger_attempt_blocked_reason_missing")
    if memory.get("required_vram_mb_per_stage", 0) <= hardware.get("available_vram_per_gpu_mb", 0):
        errors.append("larger_attempt_memory_blocker_not_proven")
    if feasibility.get("max_fresh_model_attempts", 3) > 2:
        errors.append("fresh_model_attempt_limit_exceeded")
    if feasibility.get("max_requeue_attempts", 2) > 1:
        errors.append("requeue_attempt_limit_exceeded")
    if feasibility.get("single_attempt_timeout_minutes", 61) > 60:
        errors.append("attempt_timeout_limit_exceeded")

    core = _dict(report.get("core_scale_summary"))
    if core.get("largest_successful_retained_model_tier") != "14b":
        errors.append("largest_retained_tier_not_14b")
    if core.get("fourteen_b_dual_kaggle_verified") is not True:
        errors.append("fourteen_b_retained_evidence_missing")
    if core.get("seven_b_multi_token_verified") is not True:
        errors.append("seven_b_multi_token_retained_evidence_missing")

    generation = _dict(report.get("generation_source_summary"))
    if generation.get("external_gpu_runtime_verified") is not True:
        errors.append("retained_gpu_runtime_summary_missing")
    if generation.get("generated_token_count", 0) < 16:
        errors.append("retained_gpu_generation_token_count_missing")

    batch_stream = _dict(report.get("batch_stream_source_summary"))
    if batch_stream.get("batch_or_multi_request_ready") is not True:
        errors.append("batch_stream_source_missing")

    safety = _dict(report.get("safety"))
    if safety.get("public_artifact_safe") is not True:
        errors.append("safety_public_artifact_safe_mismatch")
    if safety.get("report_public_leak_paths"):
        errors.append("report_public_leak_paths_present")
    for field in [
        "raw_prompt_public",
        "raw_generated_text_public",
        "generated_token_ids_public",
        "activation_public",
        "hidden_states_public",
        "logits_public",
        "kv_cache_public",
        "credentials_public",
        "lease_material_public",
        "idempotency_material_public",
        "private_env_written",
        "registry_written",
        "kaggle_payload_written",
        "runtime_private_material_written",
    ]:
        if safety.get(field) is not False:
            errors.append(f"safety_flag_mismatch:{field}")
    leaks = pack.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))

    artifacts = _dict(report.get("artifacts"))
    for name in [
        "summary_json",
        "summary_markdown",
        "support_bundle_json",
        "usability_alpha_json",
        "gpu_generation_json",
        "batch_stream_json",
        "core_status_json",
        "core_handoff_json",
    ]:
        if _dict(artifacts.get(name)).get("present") is not True:
            errors.append(f"artifact_missing:{name}")
    return errors


def build_check(args: argparse.Namespace) -> dict[str, Any]:
    if args.report:
        report = load_json(Path(args.report))
    else:
        pack_args = pack.parse_args([
            "--output-dir",
            args.output_dir,
            "--execution-mode",
            args.execution_mode,
            "--usability-report",
            args.usability_report,
            "--core-handoff-report",
            args.core_handoff_report,
            "--core-status-report",
            args.core_status_report,
            "--control-user-alpha-report",
            args.control_user_alpha_report,
            "--gpu-generation-report",
            args.gpu_generation_report,
            "--gpu-generation-fallback-report",
            args.gpu_generation_fallback_report,
            "--batch-stream-report",
            args.batch_stream_report,
            "--larger-candidate-tier",
            args.larger_candidate_tier,
            "--larger-candidate-model-id",
            args.larger_candidate_model_id,
            "--target-max-new-tokens",
            str(args.target_max_new_tokens),
        ])
        report = pack.build_report(pack_args)
    errors = validate_report(report)
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "report_schema": report.get("schema"),
        "output_dir": report.get("output_dir") or args.output_dir,
        "report_path": args.report or str(Path(args.output_dir) / "gpu_swarm_production_like_validation.json"),
        "gpu_swarm_production_like_validation_ready": report.get("gpu_swarm_production_like_validation_ready") is True,
        "production_like_workload_ready": report.get("production_like_workload_ready") is True,
        "larger_model_attempted": report.get("larger_model_attempted") is True,
        "largest_successful_model_tier": report.get("largest_successful_model_tier"),
        "largest_attempted_model_tier": report.get("largest_attempted_model_tier"),
        "larger_model_blocked_reason": report.get("larger_model_blocked_reason"),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "execution_mode": report.get("execution_mode"),
        "external_runtime_verified": report.get("external_runtime_verified") is True,
        "fresh_gpu_run_performed": report.get("fresh_gpu_run_performed") is True,
        "retained_evidence_imported": report.get("retained_evidence_imported") is True,
        "errors": errors,
        "diagnosis_codes": ["gpu_swarm_production_like_validation_check_ready"] if not errors else ["gpu_swarm_production_like_validation_check_failed"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GPU Swarm production-like validation RC evidence.")
    parser.add_argument("--report", default="")
    parser.add_argument("--output-dir", default=pack.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execution-mode", choices=pack.EXECUTION_MODES, default="evidence-import")
    parser.add_argument("--usability-report", default=pack.DEFAULT_USABILITY_REPORT)
    parser.add_argument("--core-handoff-report", default=pack.DEFAULT_CORE_HANDOFF_REPORT)
    parser.add_argument("--core-status-report", default=pack.DEFAULT_CORE_STATUS_REPORT)
    parser.add_argument("--control-user-alpha-report", default=pack.DEFAULT_CONTROL_USER_ALPHA_REPORT)
    parser.add_argument("--gpu-generation-report", default=pack.DEFAULT_GPU_GENERATION_REPORT)
    parser.add_argument("--gpu-generation-fallback-report", default=pack.DEFAULT_GPU_GENERATION_FALLBACK_REPORT)
    parser.add_argument("--batch-stream-report", default=pack.DEFAULT_BATCH_STREAM_REPORT)
    parser.add_argument("--larger-candidate-model-id", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--larger-candidate-tier", default="32b")
    parser.add_argument("--target-max-new-tokens", type=int, default=16)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.report and not Path(args.report).is_file():
        raise SystemExit("--report must point to an existing JSON file")
    if args.target_max_new_tokens < 1 or args.target_max_new_tokens > 128:
        raise SystemExit("--target-max-new-tokens must be between 1 and 128")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_check(args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"GPU Swarm production-like validation check ready: {result.get('ok')}")
        if result.get("errors"):
            print("errors: " + ", ".join(result.get("errors") or []))
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
