#!/usr/bin/env python3
"""Validate GPU+TPU+CPU 32B heterogeneous stage-inference RC evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import gpu_tpu_cpu_32b_heterogeneous_rc_pack as pack  # noqa: E402


SCHEMA = "gpu_tpu_cpu_32b_heterogeneous_rc_check_v1"


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def validate_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("ok") is not True:
        errors.append("report_not_ok")
    for field in [
        "gpu_tpu_cpu_32b_heterogeneous_rc_ready",
        "public_artifact_safe",
    ]:
        if report.get(field) is not True:
            errors.append(f"{field}_missing")
    if report.get("execution_mode") not in pack.EXECUTION_MODES:
        errors.append("execution_mode_invalid")
    if report.get("live_proof_mode") not in pack.LIVE_PROOF_MODES:
        errors.append("live_proof_mode_invalid")

    safety = _dict(report.get("safety"))
    for name, expected in pack.default_safety_flags().items():
        if safety.get(name) is not expected:
            errors.append(f"safety_flag_mismatch:{name}")
    if pack.public_redaction_errors(report):
        errors.append("public_redaction_scan_failed:" + ",".join(pack.public_redaction_errors(report)[:8]))

    alpha = _dict(report.get("alpha_import"))
    if alpha.get("schema") != "gpu_tpu_cpu_32b_rc_alpha_import_v1":
        errors.append("alpha_import_schema_mismatch")
    for field in [
        "alpha_ready",
        "gpu_backend_evidence_ready",
        "tpu_backend_evidence_ready",
        "cpu_backend_evidence_ready",
        "alpha_32b_feasibility_ready",
    ]:
        if alpha.get(field) is not True:
            errors.append(f"alpha_import_{field}_missing")

    matrix = _dict(report.get("stage_runtime_matrix"))
    if matrix.get("schema") != "gpu_tpu_cpu_32b_stage_runtime_matrix_v1":
        errors.append("stage_runtime_matrix_schema_mismatch")
    cuda_stage = _dict(matrix.get("cuda_gpu_stage"))
    tpu_stage = _dict(matrix.get("jax_tpu_stage"))
    cpu_stage = _dict(matrix.get("cpu_tail_or_verifier_stage"))
    if cuda_stage.get("qwen_llama_like_stage_loading_ready") is not True:
        errors.append("cuda_stage_loading_missing")
    if cpu_stage.get("tail_or_verifier_ready") is not True:
        errors.append("cpu_tail_or_verifier_missing")
    if tpu_stage.get("real_model_tpu_runtime_ready") is not True:
        errors.append("tpu_real_model_runtime_missing")
    if report.get("tpu_stage_adapter_plan_ready") is True:
        if tpu_stage.get("checkpoint_bridge_plan_ready") is not True:
            errors.append("tpu_stage_matrix_checkpoint_bridge_plan_missing")
        if tpu_stage.get("stage_owned_tpu_loader_plan_ready") is not True:
            errors.append("tpu_stage_matrix_loader_plan_missing")
        if _int(tpu_stage.get("adapter_assigned_key_count")) < 1:
            errors.append("tpu_stage_matrix_assigned_keys_missing")
        if tpu_stage.get("adapter_all_assigned_keys_mapped") is not True:
            errors.append("tpu_stage_matrix_key_mapping_missing")
        if "safetensors_or_maxtext_checkpoint_bridge" in set(tpu_stage.get("missing_items") or []):
            errors.append("tpu_stage_matrix_checkpoint_bridge_still_marked_missing")
    if report.get("tpu_qwen_like_stage_runtime_probe_ready") is True:
        if tpu_stage.get("qwen_like_stage_runtime_probe_ready") is not True:
            errors.append("tpu_stage_matrix_runtime_probe_missing")
        if not _dict(tpu_stage.get("stage_runtime_probe_shape_metadata")):
            errors.append("tpu_stage_matrix_runtime_probe_shape_missing")
        if "jax_tpu_llama_like_stage_runtime" in set(tpu_stage.get("missing_items") or []):
            errors.append("tpu_stage_matrix_runtime_still_marked_missing")
    if tpu_stage.get("stage_owned_32b_partial_tensor_to_tpu_verified") is True:
        if not str(tpu_stage.get("stage_owned_32b_loader_selected_tensor_hash") or "").startswith("sha256:"):
            errors.append("tpu_stage_matrix_partial_loader_hash_missing")
        if tpu_stage.get("stage_owned_32b_full_loader_ready") is not True and tpu_stage.get("tpu_32b_runtime_adapter_ready") is True:
            errors.append("tpu_stage_matrix_partial_loader_overclaims_adapter_ready")

    protocol = _dict(report.get("activation_protocol"))
    if protocol.get("schema") != pack.ACTIVATION_PROTOCOL_SCHEMA:
        errors.append("activation_protocol_schema_mismatch")
    if protocol.get("protocol_ready") is not True:
        errors.append("activation_protocol_not_ready")
    hops = _list(protocol.get("hops"))
    hop_pairs = {(str(item.get("from_backend") or ""), str(item.get("to_backend") or "")) for item in hops if isinstance(item, dict)}
    if ("cuda", "jax_tpu") not in hop_pairs or ("jax_tpu", "cpu") not in hop_pairs:
        errors.append("activation_protocol_required_hops_missing")
    for item in hops:
        if isinstance(item, dict):
            for flag in ["shape_metadata_required", "dtype_metadata_required", "layout_metadata_required", "activation_hash_required"]:
                if item.get(flag) is not True:
                    errors.append(f"activation_protocol_{flag}_missing")
            if item.get("activation_payload_public") is not False:
                errors.append("activation_protocol_payload_public")

    live = _dict(report.get("live_same_request_summary"))
    if live.get("schema") != "gpu_tpu_cpu_32b_same_request_live_summary_v1":
        errors.append("live_summary_schema_mismatch")
    rc_success = report.get("gpu_tpu_cpu_32b_bounded_rc_success") is True
    same_request = report.get("gpu_tpu_cpu_32b_same_request_verified") is True
    if rc_success != same_request:
        errors.append("rc_success_same_request_mismatch")
    if rc_success:
        if live.get("same_request_verified") is not True:
            errors.append("success_without_live_summary_success")
        if live.get("is_32b_class") is not True:
            errors.append("success_without_32b_class_model")
        if report.get("fallback_model_used") is not False:
            errors.append("success_with_fallback_model")
        if report.get("live_tpu_stage_miner_integrated") is not True:
            errors.append("success_without_live_tpu_stage")
        if report.get("tpu_32b_runtime_adapter_ready") is not True:
            errors.append("success_without_tpu_32b_adapter")
        if _int(live.get("generated_token_count")) < _int(report.get("target_generated_token_count"), 1):
            errors.append("success_generated_token_count_too_low")
        accepted = set(live.get("accepted_stage_backends") or [])
        if not {"cuda", "jax_tpu", "cpu"}.issubset(accepted):
            errors.append("success_missing_required_stage_backend")
        counts = _dict(live.get("stage_task_counts"))
        for backend in ["cuda", "jax_tpu", "cpu"]:
            if _int(counts.get(backend)) < 1:
                errors.append(f"success_missing_stage_task_count:{backend}")
        if _int(live.get("activation_handoff_count")) < 2:
            errors.append("success_activation_handoff_count_too_low")
        for item in _list(live.get("activation_handoff_hashes")):
            if isinstance(item, dict):
                if not str(item.get("activation_hash") or "").startswith("sha256:"):
                    errors.append("success_activation_hash_missing")
                if item.get("activation_payload_public") is not False:
                    errors.append("success_activation_payload_public")
    else:
        if report.get("blocked_reason") in {"", None}:
            errors.append("blocked_report_missing_blocked_reason")
        if live.get("same_request_verified") is True:
            errors.append("report_not_success_but_live_summary_success")

    tpu_attempt = _dict(report.get("tpu_allocation_attempt_summary"))
    if tpu_attempt.get("schema") != pack.TPU_ALLOCATION_SUMMARY_SCHEMA:
        errors.append("tpu_allocation_attempt_schema_mismatch")
    if tpu_attempt.get("bounded_tpu_allocation_attempted") is True:
        if tpu_attempt.get("bounded_tpu_allocation_attempted") is not True:
            errors.append("tpu_allocation_attempt_flag_mismatch")
        if tpu_attempt.get("source", {}).get("present") is not True:
            errors.append("tpu_allocation_attempt_source_missing")
        if tpu_attempt.get("tpu_runtime_ready") is not True:
            if tpu_attempt.get("blocked_reason") in {"", None}:
                errors.append("tpu_allocation_attempt_blocked_reason_missing")
            if not tpu_attempt.get("blockers"):
                errors.append("tpu_allocation_attempt_blockers_missing")
            if tpu_attempt.get("kernels_deleted") is not True:
                errors.append("tpu_allocation_attempt_kernel_cleanup_missing")
            if tpu_attempt.get("private_packages_removed") is not True:
                errors.append("tpu_allocation_attempt_private_cleanup_missing")

    web_event = _dict(report.get("tpu_web_active_event_summary"))
    if web_event.get("schema") != pack.TPU_WEB_ACTIVE_EVENT_SUMMARY_SCHEMA:
        errors.append("tpu_web_active_event_schema_mismatch")
    if web_event.get("web_active_event_attempted") is True:
        if web_event.get("source", {}).get("present") is not True:
            errors.append("tpu_web_active_event_source_missing")
        if web_event.get("public_artifact_safe") is not True:
            errors.append("tpu_web_active_event_public_artifact_unsafe")
        if web_event.get("cleanup_not_required") is not True:
            errors.append("tpu_web_active_event_cleanup_contract_missing")
        if web_event.get("running") is not True:
            if web_event.get("blocked_reason") in {"", None}:
                errors.append("tpu_web_active_event_blocked_reason_missing")
            if not web_event.get("blockers"):
                errors.append("tpu_web_active_event_blockers_missing")
            blockers = {str(item) for item in web_event.get("blockers") or []}
            detached_or_unattached = bool(
                blockers.intersection(
                    {
                        "kaggle_web_tpu_jupyter_proxy_not_visible",
                        "kaggle_web_tpu_runtime_not_currently_ready",
                        "kaggle_web_tpu_runtime_not_currently_attached",
                        "kaggle_web_tpu_session_still_starting",
                    }
                )
            )
            if web_event.get("queue_seen") is not True and not detached_or_unattached:
                errors.append("tpu_web_active_event_queue_missing")
    if report.get("tpu_runtime_allocation_attempted") is True:
        if (
            tpu_attempt.get("bounded_tpu_allocation_attempted") is not True
            and web_event.get("web_active_event_attempted") is not True
        ):
            errors.append("tpu_runtime_allocation_attempt_source_missing")

    runtime_bridge = _dict(report.get("runtime_bridge_summary"))
    if runtime_bridge.get("schema") != pack.RUNTIME_BRIDGE_SUMMARY_SCHEMA:
        errors.append("runtime_bridge_schema_mismatch")
    if runtime_bridge.get("runtime_bridge_present") is True:
        if runtime_bridge.get("source", {}).get("present") is not True:
            errors.append("runtime_bridge_source_missing")
        if runtime_bridge.get("public_artifact_safe") is not True:
            errors.append("runtime_bridge_public_artifact_unsafe")
        bridge_32b_allowed = bool(
            rc_success
            and live.get("same_request_verified") is True
            and runtime_bridge.get("same_request_runtime_bridge_verified") is True
        )
        if runtime_bridge.get("gpu_tpu_cpu_32b_same_request_verified") is True and not bridge_32b_allowed:
            errors.append("runtime_bridge_overclaims_32b_same_request")
        if runtime_bridge.get("same_request_32b_model_verified") is True and not bridge_32b_allowed:
            errors.append("runtime_bridge_overclaims_32b_model")
        if runtime_bridge.get("not_32b_weight_success") is not True and not bridge_32b_allowed:
            errors.append("runtime_bridge_boundary_missing")
        if runtime_bridge.get("same_request_runtime_bridge_verified") is not True:
            if runtime_bridge.get("blocked_reason") in {"", None}:
                errors.append("runtime_bridge_blocked_reason_missing")
            if not runtime_bridge.get("blockers"):
                errors.append("runtime_bridge_blockers_missing")

    adapter = _dict(report.get("tpu_stage_adapter_plan_summary"))
    if adapter.get("schema") != pack.TPU_STAGE_ADAPTER_PLAN_SUMMARY_SCHEMA:
        errors.append("tpu_stage_adapter_plan_schema_mismatch")
    if report.get("tpu_stage_adapter_plan_ready") is True:
        if adapter.get("source", {}).get("present") is not True:
            errors.append("tpu_stage_adapter_plan_source_missing")
        if adapter.get("checkpoint_bridge_plan_ready") is not True:
            errors.append("tpu_stage_adapter_checkpoint_bridge_plan_missing")
        if adapter.get("stage_owned_tpu_loader_plan_ready") is not True:
            errors.append("tpu_stage_adapter_loader_plan_missing")
        if adapter.get("all_assigned_keys_mapped") is not True:
            errors.append("tpu_stage_adapter_key_mapping_missing")
        if _int(adapter.get("assigned_key_count")) < 1:
            errors.append("tpu_stage_adapter_assigned_keys_missing")
        if _int(adapter.get("unsupported_key_count")) != 0:
            errors.append("tpu_stage_adapter_unsupported_keys_present")
        if adapter.get("public_artifact_safe") is not True:
            errors.append("tpu_stage_adapter_public_artifact_unsafe")
        activation_metadata = _dict(adapter.get("activation_metadata"))
        if not activation_metadata.get("shape") or not activation_metadata.get("dtype") or not activation_metadata.get("layout"):
            errors.append("tpu_stage_adapter_activation_metadata_missing")
        kv_metadata = _dict(adapter.get("stage_local_kv_cache_metadata"))
        if kv_metadata.get("planned") is not True:
            errors.append("tpu_stage_adapter_kv_metadata_missing")
        if kv_metadata.get("kv_payload_public") is not False:
            errors.append("tpu_stage_adapter_kv_payload_public")
        if adapter.get("tpu_32b_runtime_adapter_ready") is True and adapter.get("jax_tpu_runtime_execution_ready") is not True:
            errors.append("tpu_stage_adapter_runtime_ready_without_execution")

    runtime_probe = _dict(report.get("tpu_stage_runtime_probe_summary"))
    if runtime_probe.get("schema") != pack.TPU_STAGE_RUNTIME_PROBE_SUMMARY_SCHEMA:
        errors.append("tpu_stage_runtime_probe_schema_mismatch")
    if report.get("tpu_qwen_like_stage_runtime_probe_ready") is True:
        if runtime_probe.get("source", {}).get("present") is not True:
            errors.append("tpu_stage_runtime_probe_source_missing")
        if runtime_probe.get("qwen_like_stage_runtime_ready") is not True:
            errors.append("tpu_stage_runtime_probe_ready_flag_mismatch")
        if runtime_probe.get("tpu_32b_runtime_adapter_ready") is True:
            errors.append("tpu_stage_runtime_probe_overclaims_32b_adapter")
        if runtime_probe.get("stage_local_kv_cache_verified") is not True:
            errors.append("tpu_stage_runtime_probe_kv_missing")
        if not str(runtime_probe.get("stage_output_hash") or "").startswith("sha256:"):
            errors.append("tpu_stage_runtime_probe_output_hash_missing")
        if runtime_probe.get("public_artifact_safe") is not True:
            errors.append("tpu_stage_runtime_probe_public_artifact_unsafe")
    elif runtime_probe.get("runtime_probe_present") is True:
        if runtime_probe.get("bounded_probe_blocked") is not True:
            errors.append("tpu_stage_runtime_probe_blocked_flag_missing")
        if runtime_probe.get("kernels_deleted") is not True:
            errors.append("tpu_stage_runtime_probe_kernel_cleanup_missing")
        if runtime_probe.get("private_packages_removed") is not True:
            errors.append("tpu_stage_runtime_probe_private_cleanup_missing")

    loader_probe = _dict(report.get("tpu_stage_loader_probe_summary"))
    if loader_probe.get("loader_probe_present") is True:
        if loader_probe.get("schema") != pack.TPU_STAGE_32B_LOADER_PROBE_SUMMARY_SCHEMA:
            errors.append("tpu_stage_loader_probe_schema_mismatch")
        if loader_probe.get("source", {}).get("present") is not True:
            errors.append("tpu_stage_loader_probe_source_missing")
        if loader_probe.get("public_artifact_safe") is not True:
            errors.append("tpu_stage_loader_probe_public_artifact_unsafe")
        if loader_probe.get("partial_tensor_to_tpu_verified") is True:
            if loader_probe.get("stage_owned_header_verified") is not True:
                errors.append("tpu_stage_loader_partial_without_headers")
            if not str(loader_probe.get("selected_tensor_tpu_summary_hash") or "").startswith("sha256:"):
                errors.append("tpu_stage_loader_selected_tensor_hash_missing")
            if _int(loader_probe.get("assigned_weight_key_count")) < 1:
                errors.append("tpu_stage_loader_assigned_keys_missing")
        if loader_probe.get("tpu_32b_runtime_adapter_ready") is True and loader_probe.get("full_stage_owned_tpu_loader_ready") is not True:
            errors.append("tpu_stage_loader_overclaims_adapter_ready")
        if loader_probe.get("full_stage_owned_tpu_loader_ready") is not True and loader_probe.get("bounded_probe_blocked") is not True:
            errors.append("tpu_stage_loader_blocked_flag_missing")

    blocker = _dict(report.get("blocker_report"))
    if blocker.get("schema") != "gpu_tpu_cpu_32b_rc_blocker_report_v1":
        errors.append("blocker_report_schema_mismatch")
    if not rc_success:
        if blocker.get("blocked") is not True:
            errors.append("blocker_report_not_blocked")
        if not blocker.get("blockers"):
            errors.append("blocker_report_blockers_missing")
        if not blocker.get("minimum_next_fix"):
            errors.append("blocker_report_minimum_next_fix_missing")
    else:
        if blocker.get("blocked") is not False:
            errors.append("success_blocker_report_marked_blocked")
        stale_success_blockers = {
            "jax_tpu_llama_like_stage_runtime",
            "jax_tpu_runtime_execution_not_performed",
            "full_32b_tpu_stage_owned_runtime_not_verified",
            "full_stage_owned_tpu_loader_not_executed",
            "tpu_32b_runtime_adapter_missing",
            "same_request_live_gpu_tpu_cpu_32b_not_verified",
            "same_request_live_proof_missing",
        }
        for item in blocker.get("blockers") or []:
            if str(item) in stale_success_blockers:
                errors.append(f"success_blocker_report_stale_blocker:{item}")

    for field in [
        "gpu_tpu_cpu_32b_same_request_verified",
        "live_tpu_stage_miner_integrated",
        "fallback_model_used",
        "tpu_32b_runtime_adapter_ready",
        "tpu_stage_adapter_plan_ready",
        "tpu_checkpoint_bridge_plan_ready",
        "tpu_stage_owned_loader_plan_ready",
        "tpu_qwen_like_stage_runtime_probe_ready",
        "tpu_qwen32b_single_layer_runtime_probe_ready",
        "stage_local_kv_cache_verified",
        "public_artifact_safe",
    ]:
        if field not in report:
            errors.append(f"required_top_level_field_missing:{field}")

    artifacts = _dict(report.get("artifacts"))
    required_artifacts = [
        "summary_json",
        "summary_markdown",
        "support_bundle_json",
        "stage_runtime_matrix_json",
        "activation_protocol_json",
        "live_same_request_summary_json",
        "tpu_allocation_attempt_summary_json",
        "tpu_web_active_event_summary_json",
        "runtime_bridge_summary_json",
        "tpu_stage_adapter_plan_summary_json",
        "tpu_stage_runtime_probe_summary_json",
        "blocker_report_json",
    ]
    if _dict(report.get("tpu_stage_loader_probe_summary")).get("loader_probe_present") is True:
        required_artifacts.append("tpu_stage_loader_probe_summary_json")
    for name in required_artifacts:
        if not isinstance(artifacts.get(name), dict) or artifacts[name].get("present") is not True:
            errors.append(f"artifact_missing:{name}")

    diagnosis = set(report.get("diagnosis_codes") or [])
    if "gpu_tpu_cpu_32b_heterogeneous_rc_report_ready" not in diagnosis:
        errors.append("diagnosis_report_ready_missing")
    if rc_success and "gpu_tpu_cpu_32b_bounded_rc_success" not in diagnosis:
        errors.append("diagnosis_success_missing")
    if not rc_success and "gpu_tpu_cpu_32b_bounded_rc_not_success" not in diagnosis:
        errors.append("diagnosis_not_success_missing")
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GPU+TPU+CPU 32B heterogeneous RC evidence.")
    parser.add_argument("--report", default="")
    parser.add_argument("--output-dir", default=pack.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execution-mode", choices=pack.EXECUTION_MODES, default="evidence-import")
    parser.add_argument("--alpha-report", default=pack.DEFAULT_ALPHA_REPORT)
    parser.add_argument("--live-proof-mode", choices=pack.LIVE_PROOF_MODES, default="none")
    parser.add_argument("--live-same-request-report", default="")
    parser.add_argument("--tpu-allocation-attempt-report", default="")
    parser.add_argument("--tpu-web-active-event-report", default="")
    parser.add_argument("--runtime-bridge-report", default="")
    parser.add_argument("--tpu-stage-adapter-plan-report", default="")
    parser.add_argument("--tpu-stage-runtime-probe-report", default="")
    parser.add_argument("--tpu-stage-loader-probe-report", default="")
    parser.add_argument("--target-32b-model-id", default=pack.TARGET_32B_MODEL_ID)
    parser.add_argument("--target-max-new-tokens", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def build_check(args: argparse.Namespace) -> dict[str, Any]:
    if args.report:
        report_path = Path(args.report)
        report = load_json(report_path)
    else:
        pack_args = pack.parse_args([
            "--output-dir",
            args.output_dir,
            "--execution-mode",
            args.execution_mode,
            "--alpha-report",
            args.alpha_report,
            "--live-proof-mode",
            args.live_proof_mode,
            "--target-32b-model-id",
            args.target_32b_model_id,
            "--target-max-new-tokens",
            str(args.target_max_new_tokens),
            "--context-length",
            str(args.context_length),
            *(
                ["--live-same-request-report", args.live_same_request_report]
                if args.live_same_request_report
                else []
            ),
            *(
                ["--tpu-allocation-attempt-report", args.tpu_allocation_attempt_report]
                if args.tpu_allocation_attempt_report
                else []
            ),
            *(
                ["--tpu-web-active-event-report", args.tpu_web_active_event_report]
                if args.tpu_web_active_event_report
                else []
            ),
            *(
                ["--runtime-bridge-report", args.runtime_bridge_report]
                if args.runtime_bridge_report
                else []
            ),
            *(
                ["--tpu-stage-adapter-plan-report", args.tpu_stage_adapter_plan_report]
                if args.tpu_stage_adapter_plan_report
                else []
            ),
            *(
                ["--tpu-stage-runtime-probe-report", args.tpu_stage_runtime_probe_report]
                if args.tpu_stage_runtime_probe_report
                else []
            ),
            *(
                ["--tpu-stage-loader-probe-report", args.tpu_stage_loader_probe_report]
                if args.tpu_stage_loader_probe_report
                else []
            ),
        ])
        report = pack.build_report(pack_args)
        report_path = Path(args.output_dir) / "gpu_tpu_cpu_32b_heterogeneous_rc.json"
    errors = validate_report(report)
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "report_schema": report.get("schema"),
        "report_path": str(report_path),
        "output_dir": str(report.get("output_dir") or ""),
        "gpu_tpu_cpu_32b_heterogeneous_rc_ready": report.get("gpu_tpu_cpu_32b_heterogeneous_rc_ready") is True,
        "gpu_tpu_cpu_32b_bounded_rc_success": report.get("gpu_tpu_cpu_32b_bounded_rc_success") is True,
        "gpu_tpu_cpu_32b_same_request_verified": report.get("gpu_tpu_cpu_32b_same_request_verified") is True,
        "live_tpu_stage_miner_integrated": report.get("live_tpu_stage_miner_integrated") is True,
        "tpu_runtime_allocation_attempted": report.get("tpu_runtime_allocation_attempted") is True,
        "tpu_runtime_allocation_blocked": report.get("tpu_runtime_allocation_blocked") is True,
        "tpu_stage_adapter_plan_ready": report.get("tpu_stage_adapter_plan_ready") is True,
        "tpu_qwen_like_stage_runtime_probe_ready": report.get("tpu_qwen_like_stage_runtime_probe_ready") is True,
        "fallback_model_used": report.get("fallback_model_used") is True,
        "blocked_reason": str(report.get("blocked_reason") or ""),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
        "errors": errors,
        "diagnosis_codes": ["gpu_tpu_cpu_32b_heterogeneous_rc_check_ready"] if not errors else ["gpu_tpu_cpu_32b_heterogeneous_rc_check_failed"],
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_check(args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"GPU+TPU+CPU 32B heterogeneous RC check ok: {result.get('ok')}")
        print(f"report: {result.get('report_path')}")
        if result.get("errors"):
            print("errors:")
            for error in result.get("errors") or []:
                print(f"- {error}")
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
