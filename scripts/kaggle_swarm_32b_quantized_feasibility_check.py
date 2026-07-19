#!/usr/bin/env python3
"""Validate Kaggle Swarm 32B quantized feasibility RC evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import kaggle_swarm_32b_quantized_feasibility_pack as pack  # noqa: E402


SCHEMA = "kaggle_swarm_32b_quantized_feasibility_check_v1"


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def validate_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("ok") is not True:
        errors.append("report_not_ok")
    for field in [
        "kaggle_swarm_32b_quantized_feasibility_ready",
        "candidate_32b_model_selected",
        "quantized_runtime_plan_ready",
        "kaggle_multi_kernel_topology_ready",
        "stage_partition_plan_ready",
        "per_stage_memory_estimate_ready",
        "activation_transfer_estimate_ready",
        "kaggle_stage_package_plan_ready",
        "public_artifact_safe",
    ]:
        if report.get(field) is not True:
            errors.append(f"{field}_missing")
    for field in [
        "stage_owned_loading_feasible",
        "one_token_generation_feasible",
        "multi_token_generation_feasible",
        "coordinator_direct_management_feasible",
        "upper_bound_crossing_feasible",
        "batch_or_sequential_request_feasible",
        "stage_requeue_feasible",
        "fresh_kaggle_run_performed",
        "external_runtime_verified",
        "retained_evidence_imported",
    ]:
        if not isinstance(report.get(field), bool):
            errors.append(f"{field}_not_boolean")
    for field in [
        "largest_feasible_model_tier",
        "largest_attempted_model_tier",
        "feasibility_verdict",
        "execution_mode",
    ]:
        if not str(report.get(field) or "").strip():
            errors.append(f"{field}_missing")
    ready_verdicts = {
        "feasible_32b_one_token_cross_kernel_rc",
        "feasible_32b_multitoken_coordinator_rc",
        "feasible_32b_upper_bound_crossing_rc",
    }
    if report.get("feasibility_verdict") not in ready_verdicts and not str(report.get("blocked_reason") or "").strip():
        errors.append("blocked_reason_missing")
    if report.get("execution_mode") not in pack.EXECUTION_MODES:
        errors.append("execution_mode_invalid")
    if report.get("largest_attempted_model_tier") != "32b-quantized":
        errors.append("largest_attempted_model_tier_not_32b_quantized")

    boundaries = _dict(report.get("boundaries"))
    for name in pack.BOUNDARIES:
        if boundaries.get(name) is not True:
            errors.append(f"boundary_missing:{name}")

    runtime = _dict(report.get("quantized_runtime_plan"))
    if runtime.get("candidate_32b_model_selected") is not True:
        errors.append("runtime_candidate_missing")
    if runtime.get("quantized_runtime_plan_ready") is not True:
        errors.append("runtime_plan_not_ready")
    if runtime.get("selected_runtime_adapter") not in pack.RUNTIME_ADAPTERS:
        errors.append("runtime_adapter_invalid")
    if not _dict(runtime.get("quantization")).get("format"):
        errors.append("quantization_format_missing")
    if not runtime.get("runtime_adapter_candidates"):
        errors.append("runtime_adapter_candidates_missing")

    profile = _dict(report.get("kaggle_gpu_profile"))
    if profile.get("assumed_gpu_count", 0) < 1:
        errors.append("kaggle_gpu_profile_missing")
    if not profile.get("quota_boundary") or not profile.get("runtime_boundary"):
        errors.append("kaggle_boundary_notes_missing")

    partition = _dict(report.get("stage_partition_plan"))
    memory = _dict(partition.get("memory_estimate"))
    activation = _dict(partition.get("activation_transfer_estimate"))
    if partition.get("stage_count", 0) < 1:
        errors.append("stage_count_missing")
    if memory.get("required_vram_mb_per_stage", 0) <= 0:
        errors.append("required_vram_missing")
    if memory.get("available_vram_mb_per_gpu", 0) <= 0:
        errors.append("available_vram_missing")
    if "memory_feasible_on_assumed_profile" not in memory:
        errors.append("memory_feasibility_missing")
    if activation.get("raw_activations_public") is not False:
        errors.append("activation_scope_mismatch")
    if activation.get("estimated_stage_handoff_mb_per_request", 0) <= 0:
        errors.append("activation_transfer_estimate_missing")

    topology = _dict(report.get("kaggle_multi_kernel_topology"))
    if topology.get("kaggle_multi_kernel_topology_ready") is not True:
        errors.append("topology_not_ready")
    if topology.get("private_package_payloads_written") is not False:
        errors.append("private_package_payload_written")
    if not topology.get("stage_miners"):
        errors.append("stage_miners_missing")

    package_plan = _dict(report.get("kaggle_stage_package_plan"))
    if package_plan.get("kaggle_stage_package_plan_ready") is not True:
        errors.append("stage_package_plan_not_ready")
    for item in package_plan.get("stage_packages") or []:
        if isinstance(item, dict):
            if item.get("contains_private_payload") is not False:
                errors.append("stage_package_private_payload")
            if item.get("contains_inline_kernel_payload") is not False:
                errors.append("stage_package_inline_payload")
            if not item.get("metadata_hash"):
                errors.append("stage_package_metadata_hash_missing")
    cleanup = _dict(package_plan.get("private_artifact_cleanup_plan"))
    if cleanup.get("delete_private_kaggle_kernels") is not True:
        errors.append("cleanup_plan_missing_kernel_delete")
    if cleanup.get("rotate_runtime_tokens") is not True:
        errors.append("cleanup_plan_missing_token_rotation")

    blockers = _dict(report.get("blocker_details"))
    for name in [
        "runtime_adapter",
        "vram",
        "model_format",
        "kaggle_quota",
        "download_build_time",
        "activation_transfer",
        "stage_partitioning",
        "missing_live_hardware",
        "fresh_32b_generation",
    ]:
        if name not in blockers:
            errors.append(f"blocker_missing:{name}")
    if report.get("feasibility_verdict") == "blocked_current_repo_or_kaggle_profile" and not report.get("blocked_reason"):
        errors.append("blocked_reason_missing_for_blocked_verdict")

    evidence = _dict(report.get("evidence_validation"))
    if evidence.get("fresh_kaggle_run_performed") is not report.get("fresh_kaggle_run_performed"):
        errors.append("fresh_kaggle_truth_mismatch")
    if evidence.get("external_runtime_verified") is not report.get("external_runtime_verified"):
        errors.append("external_runtime_truth_mismatch")

    diagnosis = set(report.get("diagnosis_codes") or [])
    for code in [
        "kaggle_swarm_32b_quantized_feasibility_ready",
        "candidate_32b_model_selected",
        "quantized_runtime_plan_ready",
        "kaggle_multi_kernel_topology_ready",
        "stage_partition_plan_ready",
        "per_stage_memory_estimate_ready",
        "activation_transfer_estimate_ready",
        "kaggle_stage_package_plan_ready",
        "kaggle_swarm_32b_public_artifact_redaction_ready",
    ]:
        if code not in diagnosis:
            errors.append(f"diagnosis_missing:{code}")
    if report.get("external_runtime_verified") is True:
        if "external_runtime_verified" not in diagnosis:
            errors.append("diagnosis_missing:external_runtime_verified")
    elif "external_runtime_not_verified" not in diagnosis:
        errors.append("diagnosis_missing:external_runtime_not_verified")
    if not ({"fresh_kaggle_run_not_performed", "fresh_kaggle_run_performed"} & diagnosis):
        errors.append("diagnosis_missing:fresh_kaggle_run_truth")
    fresh_probe = _dict(report.get("fresh_32b_live_probe_summary"))
    if fresh_probe.get("present"):
        if fresh_probe.get("fresh_kaggle_run_performed") is not True:
            errors.append("fresh_probe_truth_missing")
        if fresh_probe.get("all_kernels_deleted") is not True:
            errors.append("fresh_probe_kernel_cleanup_missing")
        if fresh_probe.get("all_private_packages_removed") is not True:
            errors.append("fresh_probe_private_package_cleanup_missing")
        if fresh_probe.get("gpu_hardware_verified") is not True:
            errors.append("fresh_probe_gpu_hardware_missing")
        if fresh_probe.get("q2k_all_splits_downloaded") is not True:
            errors.append("fresh_probe_download_missing")
        if fresh_probe.get("one_token_generation_verified") is not False:
            errors.append("fresh_probe_generation_truth_mismatch")
    stage_owned_probe = _dict(report.get("fresh_32b_stage_owned_loading_probe_summary"))
    if stage_owned_probe.get("present"):
        if stage_owned_probe.get("stage_owned_quantized_32b_loading_ready") is not True:
            errors.append("stage_owned_probe_not_ready")
        if stage_owned_probe.get("gpu_hardware_verified") is not True:
            errors.append("stage_owned_probe_gpu_hardware_missing")
        if stage_owned_probe.get("stage_owned_download_scope_ready") is not True:
            errors.append("stage_owned_probe_download_scope_missing")
        if stage_owned_probe.get("loads_only_stage_weight_keys_ready") is not True:
            errors.append("stage_owned_probe_key_scope_missing")
        if stage_owned_probe.get("all_kernels_deleted") is not True:
            errors.append("stage_owned_probe_kernel_cleanup_missing")
        if stage_owned_probe.get("all_private_packages_removed") is not True:
            errors.append("stage_owned_probe_private_package_cleanup_missing")
        if stage_owned_probe.get("one_token_generation_verified") is not False:
            errors.append("stage_owned_probe_generation_truth_mismatch")
        for item in stage_owned_probe.get("stage_summaries") or []:
            if isinstance(item, dict):
                if item.get("loads_only_stage_weight_keys") is not True:
                    errors.append("stage_owned_probe_stage_key_scope_missing")
                if item.get("cross_stage_weight_keys_loaded") is not False:
                    errors.append("stage_owned_probe_cross_stage_key_loaded")
                if item.get("materialize_clone_requested") is not True:
                    errors.append("stage_owned_probe_clone_not_requested")
                if item.get("materialized_weight_key_count") != item.get("assigned_weight_key_count"):
                    errors.append("stage_owned_probe_clone_count_mismatch")
    activation_decode_probe = _dict(report.get("fresh_32b_activation_decode_probe_summary"))
    if activation_decode_probe.get("present"):
        if activation_decode_probe.get("cross_kernel_activation_decode_verified") is not True:
            errors.append("activation_decode_probe_not_ready")
        if activation_decode_probe.get("one_token_generation_verified") is not True:
            errors.append("activation_decode_probe_one_token_missing")
        if report.get("multi_token_generation_feasible") is True:
            if activation_decode_probe.get("multi_token_decode_verified") is not True:
                errors.append("activation_decode_probe_multi_token_missing")
            if activation_decode_probe.get("coordinator_direct_management_verified") is not True:
                errors.append("activation_decode_probe_coordinator_missing")
            if activation_decode_probe.get("single_kernel_attempted") is not True:
                errors.append("activation_decode_probe_single_kernel_not_attempted")
            if report.get("upper_bound_crossing_feasible") is True:
                if activation_decode_probe.get("upper_bound_crossing_verified") is not True:
                    errors.append("activation_decode_probe_upper_bound_crossing_missing")
                if activation_decode_probe.get("single_kernel_ok") is True:
                    errors.append("activation_decode_probe_single_kernel_unexpectedly_ready_for_crossing")
            elif activation_decode_probe.get("single_kernel_ok") is not True:
                errors.append("activation_decode_probe_single_kernel_not_ready")
        if activation_decode_probe.get("stage_owned_awq_runtime_verified") is not True:
            errors.append("activation_decode_probe_awq_runtime_missing")
        if activation_decode_probe.get("activation_handoff_verified") is not True:
            errors.append("activation_decode_probe_handoff_missing")
        if activation_decode_probe.get("all_kernels_deleted") is not True:
            errors.append("activation_decode_probe_kernel_cleanup_missing")
        if activation_decode_probe.get("all_private_packages_removed") is not True:
            errors.append("activation_decode_probe_private_package_cleanup_missing")
        if activation_decode_probe.get("private_activation_removed") is not True:
            errors.append("activation_decode_probe_private_activation_cleanup_missing")
        if activation_decode_probe.get("activation_public") is not False:
            errors.append("activation_decode_probe_activation_public")
        if activation_decode_probe.get("hidden_state_public") is not False:
            errors.append("activation_decode_probe_hidden_state_public")
        if activation_decode_probe.get("generated_token_ids_public") is not False:
            errors.append("activation_decode_probe_token_ids_public")
        if report.get("one_token_generation_feasible") is not True:
            errors.append("activation_decode_probe_report_generation_truth_mismatch")
        if report.get("upper_bound_crossing_feasible") is True:
            expected_tier = "32b-quantized-4stage-upper-bound-rc"
        elif report.get("multi_token_generation_feasible") is True:
            expected_tier = "32b-quantized-2token-rc"
        else:
            expected_tier = "32b-quantized-1token"
        if report.get("largest_feasible_model_tier") != expected_tier:
            errors.append("activation_decode_probe_largest_feasible_tier_mismatch")

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
        "model_cache_private_paths_public",
        "kaggle_credentials_public",
        "api_keys_public",
        "coordinator_tokens_public",
        "lease_material_public",
        "idempotency_material_public",
        "private_env_written",
        "registry_written",
        "inline_kaggle_payload_public",
        "runtime_private_state_public",
        "private_package_payloads_written",
    ]:
        if safety.get(field) is not False:
            errors.append(f"safety_flag_mismatch:{field}")
    leaks = pack.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))

    hard_limits = _dict(report.get("hard_limits"))
    if not hard_limits:
        errors.append("hard_limits_missing")
    if hard_limits.get("max_fresh_model_attempts", 3) > 2:
        errors.append("fresh_model_attempt_limit_exceeded")
    if hard_limits.get("max_requeue_attempts", 2) > 1:
        errors.append("requeue_attempt_limit_exceeded")
    if hard_limits.get("single_attempt_timeout_minutes", 61) > 60:
        errors.append("attempt_timeout_limit_exceeded")

    artifacts = _dict(report.get("artifacts"))
    for name in [
        "summary_json",
        "summary_markdown",
        "support_bundle_json",
        "stage_package_plan_json",
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
            "--production-like-report",
            args.production_like_report,
            "--core-status-report",
            args.core_status_report,
            "--large-model-kaggle-report",
            args.large_model_kaggle_report,
            "--fresh-32b-live-probe-report",
            args.fresh_32b_live_probe_report,
            "--fresh-32b-stage-owned-loading-probe-report",
            args.fresh_32b_stage_owned_loading_probe_report,
            "--runtime-adapter",
            args.runtime_adapter,
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
        "report_path": args.report or str(Path(args.output_dir) / "kaggle_swarm_32b_quantized_feasibility.json"),
        "kaggle_swarm_32b_quantized_feasibility_ready": report.get("kaggle_swarm_32b_quantized_feasibility_ready") is True,
        "feasibility_verdict": report.get("feasibility_verdict"),
        "blocked_reason": report.get("blocked_reason"),
        "largest_feasible_model_tier": report.get("largest_feasible_model_tier"),
        "largest_attempted_model_tier": report.get("largest_attempted_model_tier"),
        "execution_mode": report.get("execution_mode"),
        "fresh_kaggle_run_performed": report.get("fresh_kaggle_run_performed") is True,
        "external_runtime_verified": report.get("external_runtime_verified") is True,
        "retained_evidence_imported": report.get("retained_evidence_imported") is True,
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "errors": errors,
        "diagnosis_codes": ["kaggle_swarm_32b_quantized_feasibility_check_ready"] if not errors else ["kaggle_swarm_32b_quantized_feasibility_check_failed"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Kaggle Swarm 32B quantized feasibility RC evidence.")
    parser.add_argument("--report", default="")
    parser.add_argument("--output-dir", default=pack.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execution-mode", choices=pack.EXECUTION_MODES, default="fixture")
    parser.add_argument("--production-like-report", default=pack.DEFAULT_PRODUCTION_LIKE_REPORT)
    parser.add_argument("--core-status-report", default=pack.DEFAULT_CORE_STATUS_REPORT)
    parser.add_argument("--large-model-kaggle-report", default=pack.DEFAULT_LARGE_MODEL_KAGGLE_REPORT)
    parser.add_argument("--fresh-32b-live-probe-report", default=pack.DEFAULT_FRESH_32B_LIVE_PROBE_REPORT)
    parser.add_argument("--fresh-32b-stage-owned-loading-probe-report", default=pack.DEFAULT_FRESH_32B_STAGE_OWNED_LOADING_PROBE_REPORT)
    parser.add_argument("--runtime-adapter", choices=pack.RUNTIME_ADAPTERS, default="gguf-llama-cpp-cuda")
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
        print(f"Kaggle Swarm 32B quantized feasibility check ready: {result.get('ok')}")
        if result.get("errors"):
            print("errors: " + ", ".join(result.get("errors") or []))
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
