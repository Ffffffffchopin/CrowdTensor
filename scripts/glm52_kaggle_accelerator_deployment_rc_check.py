#!/usr/bin/env python3
"""Validate GLM 5.2 Kaggle accelerator deployment RC evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_kaggle_accelerator_deployment_rc_pack as pack  # noqa: E402


SCHEMA = "glm52_kaggle_accelerator_deployment_rc_check_v1"
REQUIRED_PROVIDERS = set(pack.REQUIRED_PROVIDERS)


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
    if report.get("glm52_kaggle_accelerator_deployment_rc_ready") is not True:
        errors.append("rc_ready_missing")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = pack.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))

    model = _dict(report.get("model"))
    if model.get("model_id") != pack.MODEL_ID:
        errors.append("model_id_mismatch")
    if model.get("fallback_model_allowed_for_success") is not False:
        errors.append("fallback_model_success_boundary_missing")

    source = _dict(report.get("source"))
    if source.get("present") is not True:
        errors.append("source_report_missing")
    if source.get("resolver_ready") is not True:
        errors.append("source_resolver_not_ready")
    if source.get("compatible_with_glm52") is not True:
        errors.append("source_not_glm52")
    if source.get("model_id") != pack.MODEL_ID:
        errors.append("source_model_id_mismatch")
    if _int(source.get("official_weight_key_count")) <= 0:
        errors.append("source_weight_keys_missing")
    if not source.get("recommended_repo"):
        errors.append("source_recommended_repo_missing")
    if source.get("public_artifact_safe") is not True:
        errors.append("source_public_artifact_unsafe")

    tpu = _dict(report.get("tpu_request"))
    if tpu.get("present") is not True:
        errors.append("tpu_watch_missing")
    if tpu.get("public_artifact_safe") is not True:
        errors.append("tpu_public_artifact_unsafe")
    if tpu.get("queued") is True and tpu.get("tpu_runtime_ready") is True:
        errors.append("tpu_queued_and_ready_conflict")

    awq = _dict(report.get("awq_stage_header"))
    if awq:
        if awq.get("present") is True:
            if awq.get("public_artifact_safe") is not True:
                errors.append("awq_stage_header_public_artifact_unsafe")
            if awq.get("base_model_id") != pack.MODEL_ID:
                errors.append("awq_stage_header_base_model_mismatch")
            if awq.get("model_repo") and "GLM-5.2" not in str(awq.get("model_repo")):
                errors.append("awq_stage_header_model_repo_mismatch")
            if awq.get("stage_header_ready") is True:
                if _int(awq.get("assigned_weight_key_count")) <= 0:
                    errors.append("awq_stage_header_assigned_keys_missing")
                if _int(awq.get("present_stage_key_count")) != _int(awq.get("assigned_weight_key_count")):
                    errors.append("awq_stage_header_present_key_mismatch")
                if _int(awq.get("missing_stage_key_count")) != 0:
                    errors.append("awq_stage_header_missing_keys_present")
                if not _dict(awq.get("dtype_counts")):
                    errors.append("awq_stage_header_dtype_counts_missing")
                if _dict(awq.get("stage_family_hits")).get("awq_quantized_tensors") is not True:
                    errors.append("awq_stage_header_quantized_family_missing")
            if awq.get("weight_tensor_values_loaded") is not False:
                errors.append("awq_stage_header_weight_value_overclaim")
            if awq.get("weight_tensor_values_public") is not False:
                errors.append("awq_stage_header_weight_values_public_unsafe")
            if awq.get("safetensors_header_payload_public") is not False:
                errors.append("awq_stage_header_payload_public_unsafe")

    value_probe = _dict(report.get("awq_stage_value_probe"))
    if value_probe and value_probe.get("present") is True:
        if value_probe.get("public_artifact_safe") is not True:
            errors.append("awq_stage_value_public_artifact_unsafe")
        if value_probe.get("stage_value_probe_ready") is True:
            if value_probe.get("weight_tensor_values_loaded") is not True:
                errors.append("awq_stage_value_loaded_missing")
            if _int(value_probe.get("stage_value_probe_ready_count")) != _int(value_probe.get("stage_value_probe_count")):
                errors.append("awq_stage_value_ready_count_mismatch")
        probes = [item for item in _list(value_probe.get("probes")) if isinstance(item, dict)]
        if not probes:
            errors.append("awq_stage_value_probe_entries_missing")
        for item in probes:
            provider = str(item.get("provider") or "")
            label = provider or f"stage{_int(item.get('stage_id'), -1)}"
            if item.get("stage_value_probe_ready") is True:
                if item.get("weight_tensor_values_loaded") is not True:
                    errors.append(f"awq_stage_value_loaded_missing:{label}")
                if _int(item.get("weight_value_byte_count")) <= 0:
                    errors.append(f"awq_stage_value_byte_count_missing:{label}")
                if item.get("weight_value_hash_present") is not True:
                    errors.append(f"awq_stage_value_hash_missing:{label}")
                if _int(item.get("assigned_weight_key_count")) <= 0:
                    errors.append(f"awq_stage_value_assigned_keys_missing:{label}")
            if item.get("public_artifact_safe") is not True:
                errors.append(f"awq_stage_value_probe_public_artifact_unsafe:{label}")
            if item.get("weight_tensor_values_public") is not False:
                errors.append(f"awq_stage_value_weight_values_public_unsafe:{label}")
            if item.get("safetensors_header_payload_public") is not False:
                errors.append(f"awq_stage_value_header_payload_public_unsafe:{label}")
            if item.get("stage_runtime_adapter_verified") is True:
                errors.append(f"awq_stage_value_runtime_overclaim:{label}")
            if item.get("same_request_route_verified") is True:
                errors.append(f"awq_stage_value_same_request_route_overclaim:{label}")
            if item.get("same_request_decode_verified") is True:
                errors.append(f"awq_stage_value_same_request_decode_overclaim:{label}")
        if value_probe.get("weight_tensor_values_public") is not False:
            errors.append("awq_stage_value_weight_values_public_unsafe")
        if value_probe.get("safetensors_header_payload_public") is not False:
            errors.append("awq_stage_value_header_payload_public_unsafe")
        if value_probe.get("stage_runtime_adapter_verified") is True:
            errors.append("awq_stage_value_runtime_overclaim")
        if value_probe.get("same_request_route_verified") is True:
            errors.append("awq_stage_value_same_request_route_overclaim")
        if value_probe.get("same_request_decode_verified") is True:
            errors.append("awq_stage_value_same_request_decode_overclaim")

    smoke = _dict(report.get("tpu_stage_smoke"))
    if smoke:
        if smoke.get("present") is True:
            queued_watch = smoke.get("queued_watch") is True
            if smoke.get("public_artifact_safe") is not True:
                errors.append("tpu_stage_smoke_public_artifact_unsafe")
            if not queued_watch and smoke.get("base_model_id") != pack.MODEL_ID:
                errors.append("tpu_stage_smoke_base_model_mismatch")
            if not queued_watch and smoke.get("model_repo") and "GLM-5.2" not in str(smoke.get("model_repo")):
                errors.append("tpu_stage_smoke_model_repo_mismatch")
            if smoke.get("stage_runtime_adapter_smoke_ready") is True:
                if _int(smoke.get("jax_tpu_device_count")) <= 0:
                    errors.append("tpu_stage_smoke_device_count_missing")
                if smoke.get("stage_header_ready") is not True:
                    errors.append("tpu_stage_smoke_stage_header_not_ready")
                if smoke.get("jax_shape_smoke_ready") is not True:
                    errors.append("tpu_stage_smoke_jax_shape_not_ready")
                if _int(smoke.get("assigned_weight_key_count")) <= 0:
                    errors.append("tpu_stage_smoke_assigned_keys_missing")
                if _int(smoke.get("present_stage_key_count")) != _int(smoke.get("assigned_weight_key_count")):
                    errors.append("tpu_stage_smoke_present_key_mismatch")
                if _int(smoke.get("missing_stage_key_count")) != 0:
                    errors.append("tpu_stage_smoke_missing_keys_present")
            if smoke.get("same_request_decode_verified") is True:
                errors.append("tpu_stage_smoke_same_request_overclaim")
            if smoke.get("weight_tensor_values_loaded") is not False:
                errors.append("tpu_stage_smoke_weight_value_overclaim")
            if smoke.get("weight_tensor_values_public") is not False:
                errors.append("tpu_stage_smoke_weight_values_public_unsafe")
            if smoke.get("safetensors_header_payload_public") is not False:
                errors.append("tpu_stage_smoke_header_payload_public_unsafe")

    kaggle_source = _dict(report.get("kaggle_source_search"))
    if kaggle_source and kaggle_source.get("present") is True:
        if kaggle_source.get("public_artifact_safe") is not True:
            errors.append("kaggle_source_search_public_artifact_unsafe")
        if kaggle_source.get("search_ready") is not True:
            errors.append("kaggle_source_search_not_ready")
        attach_verified = kaggle_source.get("kaggle_attach_source_verified") is True
        if attach_verified:
            compatible_count = _int(kaggle_source.get("compatible_model_source_count")) + _int(
                kaggle_source.get("compatible_dataset_source_count")
            )
            if compatible_count <= 0:
                errors.append("kaggle_source_verified_without_compatible_count")
            if (
                _int(kaggle_source.get("compatible_dataset_source_count")) <= 0
                and not _list(kaggle_source.get("recommended_kaggle_kernel_model_sources"))
            ):
                errors.append("kaggle_source_verified_without_attach_ref")
        else:
            blockers = set(str(item) for item in _list(kaggle_source.get("blockers")))
            if "glm52_kaggle_attach_source_not_found" not in blockers:
                errors.append("kaggle_source_missing_not_found_blocker")

    stage_plan = _dict(report.get("stage_runtime_plan"))
    if stage_plan and stage_plan.get("present") is True:
        if stage_plan.get("public_artifact_safe") is not True:
            errors.append("stage_runtime_plan_public_artifact_unsafe")
        if stage_plan.get("plan_ready") is not True:
            errors.append("stage_runtime_plan_not_ready")
        providers = {
            str(spec.get("provider") or "")
            for spec in _list(stage_plan.get("provider_specs"))
            if isinstance(spec, dict)
        }
        if not REQUIRED_PROVIDERS.issubset(providers):
            errors.append("stage_runtime_plan_required_providers_missing")
        for spec in _list(stage_plan.get("provider_specs")):
            if not isinstance(spec, dict):
                continue
            provider = str(spec.get("provider") or "")
            if spec.get("expected_stage_report_schema") != "glm52_kaggle_stage_runtime_report_v1":
                errors.append(f"stage_runtime_plan_report_schema_missing:{provider or 'missing'}")
            layer_range = _list(spec.get("stage_layer_range"))
            if len(layer_range) != 2 or _int(layer_range[1]) <= _int(layer_range[0]):
                errors.append(f"stage_runtime_plan_layer_range_invalid:{provider or 'missing'}")
        if stage_plan.get("stage_runtime_adapter_verified") is True and stage_plan.get("blockers"):
            errors.append("stage_runtime_plan_verified_with_blockers")

    stage_package = _dict(report.get("stage_worker_package"))
    if stage_package and stage_package.get("present") is True:
        if stage_package.get("public_artifact_safe") is not True:
            errors.append("stage_worker_package_public_artifact_unsafe")
        if stage_package.get("package_ready") is not True:
            errors.append("stage_worker_package_not_ready")
        runtime_kind = str(stage_package.get("stage_runtime_package_kind") or "")
        if runtime_kind and runtime_kind not in {"value_op", "full_prefix_stage_decode"}:
            errors.append("stage_worker_package_runtime_kind_invalid")
        providers = {
            str(pkg.get("provider") or "")
            for pkg in _list(stage_package.get("provider_packages"))
            if isinstance(pkg, dict)
        }
        if not REQUIRED_PROVIDERS.issubset(providers):
            errors.append("stage_worker_package_required_providers_missing")
        for pkg in _list(stage_package.get("provider_packages")):
            if not isinstance(pkg, dict):
                continue
            provider = str(pkg.get("provider") or "")
            if pkg.get("expected_stage_report_schema") != "glm52_kaggle_stage_runtime_report_v1":
                errors.append(f"stage_worker_package_report_schema_missing:{provider or 'missing'}")
            if pkg.get("private_kernel") is not True:
                errors.append(f"stage_worker_package_not_private:{provider or 'missing'}")
            if pkg.get("pushed_to_kaggle") is True and pkg.get("live_run_performed") is not True:
                errors.append(f"stage_worker_package_push_without_live_run:{provider or 'missing'}")
            package_runtime_kind = str(pkg.get("stage_runtime_package_kind") or "")
            if runtime_kind and package_runtime_kind and package_runtime_kind != runtime_kind:
                errors.append(f"stage_worker_package_runtime_kind_mismatch:{provider or 'missing'}")
            if runtime_kind == "full_prefix_stage_decode" and pkg.get("full_prefix_runtime_bundle_present") is not True:
                errors.append(f"stage_worker_package_full_prefix_bundle_missing:{provider or 'missing'}")
            if runtime_kind == "full_prefix_stage_decode" and pkg.get("embedded_runtime_bundle_present") is not True:
                errors.append(f"stage_worker_package_embedded_bundle_missing:{provider or 'missing'}")
            if runtime_kind == "full_prefix_stage_decode":
                stage_range = _list(pkg.get("stage_layer_range"))
                probe_range = _list(pkg.get("full_prefix_probe_layer_range"))
                if (
                    len(stage_range) == 2
                    and (
                        len(probe_range) != 2
                        or _int(probe_range[0]) < _int(stage_range[0])
                        or _int(probe_range[1]) > _int(stage_range[1])
                        or _int(probe_range[1]) <= _int(probe_range[0])
                    )
                ):
                    errors.append(f"stage_worker_package_full_prefix_probe_range_invalid:{provider or 'missing'}")
                covers_full_stage = len(stage_range) == 2 and probe_range == stage_range
                if pkg.get("full_prefix_probe_covers_full_stage") is not covers_full_stage:
                    errors.append(f"stage_worker_package_full_prefix_full_stage_flag_mismatch:{provider or 'missing'}")
                if stage_package.get("full_prefix_probe_full_stage_requested") is True and not covers_full_stage:
                    errors.append(f"stage_worker_package_full_stage_requested_but_not_covered:{provider or 'missing'}")
        if stage_package.get("stage_runtime_adapter_verified") is True and stage_package.get("blockers"):
            errors.append("stage_worker_package_verified_with_blockers")

    stage_push = _dict(report.get("stage_worker_push_probe"))
    if stage_push and stage_push.get("present") is True:
        if stage_push.get("public_artifact_safe") is not True:
            errors.append("stage_worker_push_public_artifact_unsafe")
        if stage_push.get("push_probe_ready") is not True:
            errors.append("stage_worker_push_not_ready")
        providers = {
            str(push.get("provider") or "")
            for push in _list(stage_push.get("provider_pushes"))
            if isinstance(push, dict)
        }
        if stage_push.get("required_stage_runtime_reports_verified") is True and not REQUIRED_PROVIDERS.issubset(providers):
            errors.append("stage_worker_push_required_providers_missing")
        if (
            stage_push.get("required_stage_runtime_reports_verified") is True
            and stage_push.get("all_planned_stage_runtime_reports_verified") is False
        ):
            errors.append("stage_worker_push_required_verified_but_planned_stages_missing")
        for push in _list(stage_push.get("provider_pushes")):
            if not isinstance(push, dict):
                continue
            provider = str(push.get("provider") or "")
            if stage_push.get("mode") == "preflight":
                if push.get("pushed") is True:
                    errors.append(f"stage_worker_push_preflight_overclaim:{provider or 'missing'}")
                if push.get("output_collected") is True:
                    errors.append(f"stage_worker_push_preflight_output_overclaim:{provider or 'missing'}")
                if push.get("stage_report_present") is True:
                    errors.append(f"stage_worker_push_preflight_stage_report_overclaim:{provider or 'missing'}")
                if push.get("stage_runtime_verified") is True:
                    errors.append(f"stage_worker_push_preflight_stage_runtime_overclaim:{provider or 'missing'}")
            retained_tpu_queue = bool(
                provider == "kaggle_jax_tpu"
                and "kaggle_tpu_kernel_retained_for_queue" in _list(stage_push.get("blockers"))
                and str(push.get("terminal_status") or "").upper() in {"QUEUED", "RUNNING", "PREPARING"}
            )
            retained_gpu_queue = bool(
                provider == "kaggle_cuda"
                and "kaggle_gpu_kernel_retained_for_queue_or_run" in _list(stage_push.get("blockers"))
                and str(push.get("terminal_status") or "").upper() in {"QUEUED", "RUNNING", "PREPARING", "PENDING", "UNKNOWN"}
            )
            if (
                stage_push.get("mode") == "live"
                and push.get("pushed") is True
                and push.get("cleanup_performed") is not True
                and not retained_tpu_queue
                and not retained_gpu_queue
            ):
                errors.append(f"stage_worker_push_cleanup_missing:{provider or 'missing'}")
        if (
            stage_push.get("mode") == "live"
            and _int(stage_push.get("stage_runtime_reports_collected")) >= len(REQUIRED_PROVIDERS)
            and _int(stage_push.get("stage_runtime_reports_verified")) < len(REQUIRED_PROVIDERS)
        ):
            errors.append("stage_worker_push_reports_collected_but_not_verified")
        if stage_push.get("stage_runtime_adapter_verified") is True or stage_push.get("same_request_route_verified") is True:
            errors.append("stage_worker_push_must_not_claim_runtime_success")

    transformers_preflight = _dict(report.get("transformers_decode_preflight"))
    if transformers_preflight and transformers_preflight.get("present") is True:
        if transformers_preflight.get("public_artifact_safe") is not True:
            errors.append("transformers_decode_preflight_public_artifact_unsafe")
        if transformers_preflight.get("source_ok") is not True:
            errors.append("transformers_decode_preflight_source_not_ok")
        if transformers_preflight.get("model_id") != pack.MODEL_ID:
            errors.append("transformers_decode_preflight_model_id_mismatch")
        if transformers_preflight.get("model_type") != "glm_moe_dsa":
            errors.append("transformers_decode_preflight_model_type_mismatch")
        if transformers_preflight.get("adapter_foundation_ready") is not True:
            errors.append("transformers_decode_preflight_foundation_not_ready")
        if transformers_preflight.get("awq_config_normalized_ready") is not True:
            errors.append("transformers_decode_preflight_config_not_ready")
        if transformers_preflight.get("tiny_forward_ready") is not True:
            errors.append("transformers_decode_preflight_tiny_forward_not_ready")
        if transformers_preflight.get("stage_weight_mapping_ready") is not True:
            errors.append("transformers_decode_preflight_mapping_not_ready")
        if _int(transformers_preflight.get("missing_required_key_count")) != 0:
            errors.append("transformers_decode_preflight_missing_keys_present")
        if _int(transformers_preflight.get("pack_required_key_count")) <= 0:
            errors.append("transformers_decode_preflight_pack_keys_missing")
        if transformers_preflight.get("decode_adapter_ready") is True:
            if transformers_preflight.get("pack_quantized_runtime_ready") is not True:
                errors.append("transformers_decode_preflight_decode_ready_without_pack_runtime")
            if transformers_preflight.get("blockers"):
                errors.append("transformers_decode_preflight_decode_ready_with_blockers")
        else:
            if not transformers_preflight.get("blockers"):
                errors.append("transformers_decode_preflight_not_ready_missing_blockers")
            if "glm52_full_decode_adapter_not_ready" not in set(
                str(item) for item in _list(transformers_preflight.get("blockers"))
            ):
                errors.append("transformers_decode_preflight_missing_full_decode_blocker")

    attention_projection = _dict(report.get("attention_projection"))
    if attention_projection and attention_projection.get("present") is True:
        if attention_projection.get("public_artifact_safe") is not True:
            errors.append("attention_projection_public_artifact_unsafe")
        if attention_projection.get("source_ok") is not True:
            errors.append("attention_projection_source_not_ok")
        if attention_projection.get("model_id") != pack.MODEL_ID:
            errors.append("attention_projection_model_id_mismatch")
        if attention_projection.get("model_type") != "glm_moe_dsa":
            errors.append("attention_projection_model_type_mismatch")
        for key in [
            "input_layernorm_verified",
            "q_lora_projection_verified",
            "kv_lora_projection_verified",
            "attention_projection_verified",
        ]:
            if attention_projection.get(key) is not True:
                errors.append(f"attention_projection_{key}_missing")
        if attention_projection.get("rope_applied") is True:
            errors.append("attention_projection_rope_overclaim")
        if attention_projection.get("attention_scores_verified") is True:
            errors.append("attention_projection_scores_overclaim")
        if attention_projection.get("o_proj_verified") is True:
            errors.append("attention_projection_o_proj_overclaim")
        if attention_projection.get("stage_decode_verified") is True:
            errors.append("attention_projection_stage_decode_overclaim")
        heads = _int(attention_projection.get("num_attention_heads"))
        q_rank = _int(attention_projection.get("q_lora_rank"))
        kv_rank = _int(attention_projection.get("kv_lora_rank"))
        qk = _int(attention_projection.get("qk_head_dim"))
        nope = _int(attention_projection.get("qk_nope_head_dim"))
        rope = _int(attention_projection.get("qk_rope_head_dim"))
        value = _int(attention_projection.get("v_head_dim"))
        expected_shapes = {
            "q_a_output_shape": [q_rank],
            "q_b_output_shape": [heads * qk],
            "query_shape": [heads, qk],
            "q_nope_shape": [heads, nope],
            "q_pe_shape": [heads, rope],
            "kv_a_output_shape": [kv_rank + rope],
            "kv_b_output_shape": [heads * (nope + value)],
            "k_nope_shape": [heads, nope],
            "value_shape": [heads, value],
        }
        for key, expected in expected_shapes.items():
            if _list(attention_projection.get(key)) != expected:
                errors.append(f"attention_projection_shape_mismatch:{key}")
        for key in [
            "input_norm_hash_present",
            "q_a_output_hash_present",
            "q_b_output_hash_present",
            "kv_a_output_hash_present",
            "kv_b_output_hash_present",
            "k_nope_hash_present",
            "value_hash_present",
        ]:
            if attention_projection.get(key) is not True:
                errors.append(f"attention_projection_hash_missing:{key}")
        blockers = set(str(item) for item in _list(attention_projection.get("blockers")))
        for blocker in [
            "glm52_attention_projection_is_not_rope_attention",
            "glm52_attention_projection_is_not_o_proj",
            "glm52_attention_projection_is_not_stage_decode",
            "glm52_attention_projection_missing_attention_scores",
            "glm52_attention_projection_missing_kv_cache_update",
        ]:
            if blocker not in blockers:
                errors.append(f"attention_projection_missing_boundary:{blocker}")

    attention_single = _dict(report.get("attention_single_token"))
    if attention_single and attention_single.get("present") is True:
        if attention_single.get("public_artifact_safe") is not True:
            errors.append("attention_single_token_public_artifact_unsafe")
        if attention_single.get("source_ok") is not True:
            errors.append("attention_single_token_source_not_ok")
        if attention_single.get("model_id") != pack.MODEL_ID:
            errors.append("attention_single_token_model_id_mismatch")
        if attention_single.get("model_type") != "glm_moe_dsa":
            errors.append("attention_single_token_model_type_mismatch")
        for key in [
            "rope_applied",
            "attention_scores_verified",
            "attention_weights_verified",
            "o_proj_verified",
            "single_token_attention_verified",
        ]:
            if attention_single.get(key) is not True:
                errors.append(f"attention_single_token_{key}_missing")
        if attention_single.get("kv_cache_updated") is True:
            errors.append("attention_single_token_kv_cache_overclaim")
        if attention_single.get("dsa_indexer_verified") is True:
            errors.append("attention_single_token_dsa_indexer_overclaim")
        if attention_single.get("stage_decode_verified") is True:
            errors.append("attention_single_token_stage_decode_overclaim")
        hidden = _int(attention_single.get("hidden_size"))
        heads = _int(attention_single.get("num_attention_heads"))
        qk = _int(attention_single.get("qk_head_dim"))
        value = _int(attention_single.get("v_head_dim"))
        expected_shapes = {
            "query_states_shape": [heads, qk],
            "key_states_shape": [heads, qk],
            "value_states_shape": [heads, value],
            "attention_scores_shape": [heads, 1],
            "attention_weights_shape": [heads, 1],
            "head_output_shape": [heads, value],
            "attention_flattened_shape": [heads * value],
            "o_proj_output_shape": [hidden],
        }
        for key, expected in expected_shapes.items():
            if _list(attention_single.get(key)) != expected:
                errors.append(f"attention_single_token_shape_mismatch:{key}")
        if len(_list(attention_single.get("o_proj_weight_shape"))) != 2:
            errors.append("attention_single_token_o_proj_weight_shape_invalid")
        for key in [
            "query_states_hash_present",
            "key_states_hash_present",
            "value_states_hash_present",
            "attention_scores_hash_present",
            "attention_weights_hash_present",
            "head_output_hash_present",
            "o_proj_output_hash_present",
        ]:
            if attention_single.get(key) is not True:
                errors.append(f"attention_single_token_hash_missing:{key}")
        blockers = set(str(item) for item in _list(attention_single.get("blockers")))
        for blocker in [
            "glm52_attention_single_token_is_not_multi_token_prefill",
            "glm52_attention_single_token_is_not_dsa_indexer",
            "glm52_attention_single_token_is_not_kv_cache_decode",
            "glm52_attention_single_token_is_not_transformer_block",
            "glm52_attention_single_token_is_not_stage_decode",
        ]:
            if blocker not in blockers:
                errors.append(f"attention_single_token_missing_boundary:{blocker}")

    kv_decode = _dict(report.get("kv_cache_decode"))
    if kv_decode and kv_decode.get("present") is True:
        if kv_decode.get("public_artifact_safe") is not True:
            errors.append("kv_cache_decode_public_artifact_unsafe")
        if kv_decode.get("source_ok") is not True:
            errors.append("kv_cache_decode_source_not_ok")
        if kv_decode.get("model_id") != pack.MODEL_ID:
            errors.append("kv_cache_decode_model_id_mismatch")
        if kv_decode.get("model_type") != "glm_moe_dsa":
            errors.append("kv_cache_decode_model_type_mismatch")
        for key in [
            "kv_cache_prefill_verified",
            "kv_cache_update_verified",
            "kv_cache_decode_attention_verified",
            "o_proj_verified",
        ]:
            if kv_decode.get(key) is not True:
                errors.append(f"kv_cache_decode_{key}_missing")
        if kv_decode.get("stage_decode_verified") is True:
            errors.append("kv_cache_decode_stage_decode_overclaim")
        if kv_decode.get("generated_token_verified") is True:
            errors.append("kv_cache_decode_generated_token_overclaim")
        hidden = _int(kv_decode.get("hidden_size"))
        heads = _int(kv_decode.get("num_attention_heads"))
        qk = _int(kv_decode.get("qk_head_dim"))
        value = _int(kv_decode.get("v_head_dim"))
        prefill = _int(kv_decode.get("prefill_length"))
        updated = _int(kv_decode.get("updated_cache_length"))
        expected_shapes = {
            "prefill_key_cache_shape": [prefill, heads, qk],
            "prefill_value_cache_shape": [prefill, heads, value],
            "updated_key_cache_shape": [updated, heads, qk],
            "updated_value_cache_shape": [updated, heads, value],
            "decode_query_shape": [heads, qk],
            "attention_scores_shape": [heads, updated],
            "attention_weights_shape": [heads, updated],
            "head_output_shape": [heads, value],
            "attention_flattened_shape": [heads * value],
            "o_proj_output_shape": [hidden],
        }
        for key, expected in expected_shapes.items():
            if _list(kv_decode.get(key)) != expected:
                errors.append(f"kv_cache_decode_shape_mismatch:{key}")
        if len(_list(kv_decode.get("o_proj_weight_shape"))) != 2:
            errors.append("kv_cache_decode_o_proj_weight_shape_invalid")
        for key in [
            "prefill_key_cache_hash_present",
            "prefill_value_cache_hash_present",
            "updated_key_cache_hash_present",
            "updated_value_cache_hash_present",
            "decode_query_hash_present",
            "attention_scores_hash_present",
            "attention_weights_hash_present",
            "head_output_hash_present",
            "o_proj_output_hash_present",
        ]:
            if kv_decode.get(key) is not True:
                errors.append(f"kv_cache_decode_hash_missing:{key}")
        blockers = set(str(item) for item in _list(kv_decode.get("blockers")))
        for blocker in [
            "glm52_kv_cache_decode_is_not_dsa_masked_attention",
            "glm52_kv_cache_decode_is_not_transformer_block",
            "glm52_kv_cache_decode_is_not_stage_decode",
            "glm52_kv_cache_decode_missing_mlp_residual",
            "glm52_kv_cache_decode_missing_lm_head",
        ]:
            if blocker not in blockers:
                errors.append(f"kv_cache_decode_missing_boundary:{blocker}")

    layer_decode = _dict(report.get("layer_decode"))
    if layer_decode and layer_decode.get("present") is True:
        if layer_decode.get("public_artifact_safe") is not True:
            errors.append("layer_decode_public_artifact_unsafe")
        if layer_decode.get("source_ok") is not True:
            errors.append("layer_decode_source_not_ok")
        if layer_decode.get("model_id") != pack.MODEL_ID:
            errors.append("layer_decode_model_id_mismatch")
        if layer_decode.get("model_type") != "glm_moe_dsa":
            errors.append("layer_decode_model_type_mismatch")
        for key in [
            "kv_cache_prefill_verified",
            "kv_cache_update_verified",
            "attention_decode_verified",
            "attention_residual_verified",
            "post_attention_norm_verified",
            "router_topk_verified",
            "routed_expert_gather_verified",
            "shared_experts_mlp_verified",
            "full_moe_mlp_verified",
            "layer_decode_verified",
        ]:
            if layer_decode.get(key) is not True:
                errors.append(f"layer_decode_{key}_missing")
        for key in [
            "dsa_masked_attention_integrated",
            "multi_layer_stage_runtime_verified",
            "lm_head_verified",
            "generated_token_verified",
            "stage_decode_verified",
            "same_request_decode_verified",
        ]:
            if layer_decode.get(key) is True:
                errors.append(f"layer_decode_{key}_overclaim")
        hidden = _int(layer_decode.get("hidden_size"))
        expected_shapes = {
            "attention_output_shape": [hidden],
            "attention_residual_shape": [hidden],
            "post_attention_norm_shape": [hidden],
            "routed_output_shape": [hidden],
            "shared_output_shape": [hidden],
            "full_moe_output_shape": [hidden],
            "layer_output_shape": [hidden],
        }
        for key, expected in expected_shapes.items():
            if _list(layer_decode.get(key)) != expected:
                errors.append(f"layer_decode_shape_mismatch:{key}")
        for key in [
            "attention_output_hash_present",
            "attention_residual_hash_present",
            "post_attention_norm_hash_present",
            "full_moe_output_hash_present",
            "layer_output_hash_present",
        ]:
            if layer_decode.get(key) is not True:
                errors.append(f"layer_decode_hash_missing:{key}")
        if _int(layer_decode.get("executed_expert_count")) != _int(layer_decode.get("num_experts_per_tok")):
            errors.append("layer_decode_executed_expert_count_mismatch")
        blockers = set(str(item) for item in _list(layer_decode.get("blockers")))
        for blocker in [
            "glm52_layer_decode_is_single_layer_only",
            "glm52_layer_decode_uses_basic_attention_not_dsa_masked_attention",
            "glm52_layer_decode_missing_lm_head",
            "glm52_layer_decode_is_not_stage_decode",
            "glm52_layer_decode_is_not_same_request",
        ]:
            if blocker not in blockers:
                errors.append(f"layer_decode_missing_boundary:{blocker}")

    lm_head = _dict(report.get("lm_head_token"))
    if lm_head and lm_head.get("present") is True:
        if lm_head.get("public_artifact_safe") is not True:
            errors.append("lm_head_token_public_artifact_unsafe")
        if lm_head.get("source_ok") is not True:
            errors.append("lm_head_token_source_not_ok")
        if lm_head.get("model_id") != pack.MODEL_ID:
            errors.append("lm_head_token_model_id_mismatch")
        if lm_head.get("model_type") != "glm_moe_dsa":
            errors.append("lm_head_token_model_type_mismatch")
        for key in [
            "final_norm_verified",
            "lm_head_streamed_full_vocab",
            "lm_head_logits_token_selection_verified",
            "selected_token_hash_verified",
        ]:
            if lm_head.get(key) is not True:
                errors.append(f"lm_head_token_{key}_missing")
        for key in [
            "full_model_hidden_verified",
            "generated_token_verified",
            "stage_decode_verified",
            "same_request_decode_verified",
        ]:
            if lm_head.get(key) is True:
                errors.append(f"lm_head_token_{key}_overclaim")
        hidden = _int(lm_head.get("hidden_size"))
        vocab = _int(lm_head.get("vocab_size"))
        expected_shapes = {
            "norm_weight_shape": [hidden],
            "hidden_shape": [hidden],
            "normalized_hidden_shape": [hidden],
            "lm_head_shape": [vocab, hidden],
        }
        for key, expected in expected_shapes.items():
            if _list(lm_head.get(key)) != expected:
                errors.append(f"lm_head_token_shape_mismatch:{key}")
        if lm_head.get("lm_head_dtype") != "BF16":
            errors.append("lm_head_token_dtype_invalid")
        if _int(lm_head.get("lm_head_rows_scanned")) != vocab:
            errors.append("lm_head_token_rows_scanned_mismatch")
        if _int(lm_head.get("lm_head_nbytes")) != vocab * hidden * 2:
            errors.append("lm_head_token_nbytes_mismatch")
        if _int(lm_head.get("lm_head_file_count")) != 1:
            errors.append("lm_head_token_file_count_mismatch")
        if _int(lm_head.get("lm_head_block_count")) <= 0:
            errors.append("lm_head_token_block_count_missing")
        if _int(lm_head.get("top_k_count")) != _int(lm_head.get("top_k")):
            errors.append("lm_head_token_top_k_count_mismatch")
        if lm_head.get("hidden_source") != "deterministic_probe_vector":
            errors.append("lm_head_token_hidden_source_invalid")
        for key in [
            "hidden_hash_present",
            "normalized_hidden_hash_present",
            "selected_token_id_hash_present",
            "selected_logit_hash_present",
            "top_token_ids_hash_present",
            "top_logits_hash_present",
        ]:
            if lm_head.get(key) is not True:
                errors.append(f"lm_head_token_hash_missing:{key}")
        blockers = set(str(item) for item in _list(lm_head.get("blockers")))
        for blocker in [
            "glm52_lm_head_token_selection_uses_probe_hidden_not_full_model_hidden",
            "glm52_lm_head_token_selection_is_not_stage_decode",
            "glm52_lm_head_token_selection_is_not_same_request",
        ]:
            if blocker not in blockers:
                errors.append(f"lm_head_token_missing_boundary:{blocker}")

    dsa_masked = _dict(report.get("dsa_masked_layer_decode"))
    if dsa_masked and dsa_masked.get("present") is True:
        if dsa_masked.get("public_artifact_safe") is not True:
            errors.append("dsa_masked_layer_decode_public_artifact_unsafe")
        if dsa_masked.get("source_ok") is not True:
            errors.append("dsa_masked_layer_decode_source_not_ok")
        if dsa_masked.get("model_id") != pack.MODEL_ID:
            errors.append("dsa_masked_layer_decode_model_id_mismatch")
        if dsa_masked.get("model_type") != "glm_moe_dsa":
            errors.append("dsa_masked_layer_decode_model_type_mismatch")
        if dsa_masked.get("dsa_indexer_type") != "full":
            errors.append("dsa_masked_layer_decode_indexer_not_full")
        for key in [
            "dsa_indexer_verified",
            "dsa_mask_verified",
            "dsa_mask_pruned_positions_verified",
            "kv_cache_prefill_verified",
            "kv_cache_update_verified",
            "attention_decode_verified",
            "dsa_masked_attention_integrated",
            "attention_residual_verified",
            "post_attention_norm_verified",
            "full_moe_mlp_verified",
            "layer_decode_verified",
        ]:
            if dsa_masked.get(key) is not True:
                errors.append(f"dsa_masked_layer_decode_{key}_missing")
        for key in [
            "full_dsa_topk_scale_verified",
            "lm_head_verified",
            "generated_token_verified",
            "stage_decode_verified",
            "same_request_decode_verified",
        ]:
            if dsa_masked.get(key) is True:
                errors.append(f"dsa_masked_layer_decode_{key}_overclaim")
        hidden = _int(dsa_masked.get("hidden_size"))
        heads = _int(dsa_masked.get("num_attention_heads"))
        qk = _int(dsa_masked.get("qk_head_dim"))
        updated = _int(dsa_masked.get("updated_cache_length"))
        dsa_heads = _int(dsa_masked.get("dsa_index_n_heads"))
        dsa_head_dim = _int(dsa_masked.get("dsa_index_head_dim"))
        expected_shapes = {
            "dsa_index_score_shape": [updated, updated],
            "dsa_attention_mask_shape": [updated],
            "attention_scores_shape": [heads, updated],
            "attention_output_shape": [hidden],
            "attention_residual_shape": [hidden],
            "post_attention_norm_shape": [hidden],
            "full_moe_output_shape": [hidden],
            "layer_output_shape": [hidden],
        }
        for key, expected in expected_shapes.items():
            if _list(dsa_masked.get(key)) != expected:
                errors.append(f"dsa_masked_layer_decode_shape_mismatch:{key}")
        if _int(dsa_masked.get("dsa_mask_topk_count")) <= 0:
            errors.append("dsa_masked_layer_decode_topk_missing")
        if _int(dsa_masked.get("dsa_mask_pruned_position_count")) <= 0:
            errors.append("dsa_masked_layer_decode_pruned_missing")
        if _list(dsa_masked.get("dsa_index_score_shape")) != [updated, updated]:
            errors.append("dsa_masked_layer_decode_index_score_shape_invalid")
        if dsa_heads <= 0 or dsa_head_dim <= 0 or qk <= 0:
            errors.append("dsa_masked_layer_decode_attention_config_missing")
        for key in [
            "dsa_index_score_hash_present",
            "dsa_topk_indices_hash_present",
            "dsa_attention_mask_hash_present",
            "attention_scores_hash_present",
            "attention_output_hash_present",
            "layer_output_hash_present",
        ]:
            if dsa_masked.get(key) is not True:
                errors.append(f"dsa_masked_layer_decode_hash_missing:{key}")
        if _int(dsa_masked.get("executed_expert_count")) != _int(dsa_masked.get("num_experts_per_tok")):
            errors.append("dsa_masked_layer_decode_executed_expert_count_mismatch")
        blockers = set(str(item) for item in _list(dsa_masked.get("blockers")))
        for blocker in [
            "glm52_dsa_masked_layer_decode_is_single_layer_only",
            "glm52_dsa_masked_layer_decode_uses_small_sequence_topk_cap",
            "glm52_dsa_masked_layer_decode_missing_lm_head",
            "glm52_dsa_masked_layer_decode_is_not_stage_decode",
            "glm52_dsa_masked_layer_decode_is_not_same_request",
        ]:
            if blocker not in blockers:
                errors.append(f"dsa_masked_layer_decode_missing_boundary:{blocker}")

    stage_lm = _dict(report.get("stage_hidden_lm_head"))
    if stage_lm and stage_lm.get("present") is True:
        if stage_lm.get("public_artifact_safe") is not True:
            errors.append("stage_hidden_lm_head_public_artifact_unsafe")
        if stage_lm.get("source_ok") is not True:
            errors.append("stage_hidden_lm_head_source_not_ok")
        if stage_lm.get("model_id") != pack.MODEL_ID:
            errors.append("stage_hidden_lm_head_model_id_mismatch")
        if stage_lm.get("model_type") != "glm_moe_dsa":
            errors.append("stage_hidden_lm_head_model_type_mismatch")
        for key in [
            "stage_dsa_masked_attention_integrated",
            "stage_layer_decode_verified",
            "stage_hidden_to_lm_head_verified",
            "lm_head_streamed_full_vocab",
            "stage_hidden_lm_head_token_selection_verified",
            "partial_layer_token_hash_verified",
        ]:
            if stage_lm.get(key) is not True:
                errors.append(f"stage_hidden_lm_head_{key}_missing")
        for key in [
            "full_model_hidden_verified",
            "generated_token_verified",
            "stage_decode_verified",
            "same_request_decode_verified",
        ]:
            if stage_lm.get(key) is True:
                errors.append(f"stage_hidden_lm_head_{key}_overclaim")
        hidden = _int(stage_lm.get("hidden_size"))
        vocab = _int(stage_lm.get("vocab_size"))
        expected_shapes = {
            "stage_hidden_shape": [hidden],
            "normalized_stage_hidden_shape": [hidden],
            "lm_head_shape": [vocab, hidden],
        }
        for key, expected in expected_shapes.items():
            if _list(stage_lm.get(key)) != expected:
                errors.append(f"stage_hidden_lm_head_shape_mismatch:{key}")
        if stage_lm.get("lm_head_dtype") != "BF16":
            errors.append("stage_hidden_lm_head_dtype_invalid")
        if _int(stage_lm.get("lm_head_rows_scanned")) != vocab:
            errors.append("stage_hidden_lm_head_rows_scanned_mismatch")
        if _int(stage_lm.get("lm_head_block_count")) <= 0:
            errors.append("stage_hidden_lm_head_block_count_missing")
        if _int(stage_lm.get("top_k_count")) != _int(stage_lm.get("top_k")):
            errors.append("stage_hidden_lm_head_top_k_count_mismatch")
        if _int(stage_lm.get("stage_dsa_mask_topk_count")) <= 0:
            errors.append("stage_hidden_lm_head_dsa_topk_missing")
        if _int(stage_lm.get("stage_dsa_mask_pruned_position_count")) <= 0:
            errors.append("stage_hidden_lm_head_dsa_pruned_missing")
        for key in [
            "stage_hidden_hash_present",
            "normalized_stage_hidden_hash_present",
            "selected_token_id_hash_present",
            "selected_logit_hash_present",
            "top_token_ids_hash_present",
            "top_logits_hash_present",
        ]:
            if stage_lm.get(key) is not True:
                errors.append(f"stage_hidden_lm_head_hash_missing:{key}")
        blockers = set(str(item) for item in _list(stage_lm.get("blockers")))
        for blocker in [
            "glm52_stage_hidden_lm_head_is_single_layer_only",
            "glm52_stage_hidden_lm_head_uses_small_sequence_topk_cap",
            "glm52_stage_hidden_lm_head_is_not_full_model_hidden",
            "glm52_stage_hidden_lm_head_is_not_stage_decode",
            "glm52_stage_hidden_lm_head_is_not_same_request",
        ]:
            if blocker not in blockers:
                errors.append(f"stage_hidden_lm_head_missing_boundary:{blocker}")

    multi_layer = _dict(report.get("multi_layer_stage_decode"))
    if multi_layer and multi_layer.get("present") is True:
        if multi_layer.get("public_artifact_safe") is not True:
            errors.append("multi_layer_stage_decode_public_artifact_unsafe")
        if multi_layer.get("source_ok") is not True:
            errors.append("multi_layer_stage_decode_source_not_ok")
        if multi_layer.get("model_id") != pack.MODEL_ID:
            errors.append("multi_layer_stage_decode_model_id_mismatch")
        if multi_layer.get("model_type") != "glm_moe_dsa":
            errors.append("multi_layer_stage_decode_model_type_mismatch")
        for key in [
            "multi_layer_stage_hidden_verified",
            "multi_layer_decode_token_chain_verified",
            "all_layers_dsa_masked_attention_integrated",
            "all_layers_moe_mlp_verified",
            "all_layer_outputs_chained",
            "stage_hidden_to_lm_head_verified",
            "lm_head_streamed_full_vocab",
            "stage_hidden_lm_head_token_selection_verified",
            "partial_multi_layer_token_hash_verified",
        ]:
            if multi_layer.get(key) is not True:
                errors.append(f"multi_layer_stage_decode_{key}_missing")
        for key in [
            "full_prefill_stage_hidden_verified",
            "full_model_hidden_verified",
            "generated_token_verified",
            "stage_decode_verified",
            "same_request_decode_verified",
            "live_kaggle_runtime_verified",
        ]:
            if multi_layer.get(key) is True:
                errors.append(f"multi_layer_stage_decode_{key}_overclaim")
        hidden = _int(multi_layer.get("hidden_size"))
        vocab = _int(multi_layer.get("vocab_size"))
        layer_range = _list(multi_layer.get("stage_layer_range"))
        if len(layer_range) != 2 or _int(layer_range[1]) <= _int(layer_range[0]):
            errors.append("multi_layer_stage_decode_range_invalid")
        elif _int(layer_range[1]) - _int(layer_range[0]) != _int(multi_layer.get("stage_layer_count")):
            errors.append("multi_layer_stage_decode_count_mismatch")
        if _int(multi_layer.get("stage_layer_count")) < 2:
            errors.append("multi_layer_stage_decode_count_too_small")
        if _int(multi_layer.get("executed_layer_count")) != _int(multi_layer.get("stage_layer_count")):
            errors.append("multi_layer_stage_decode_executed_count_mismatch")
        if multi_layer.get("decode_token_chain_only") is not True:
            errors.append("multi_layer_stage_decode_chain_flag_missing")
        expected_shapes = {
            "stage_hidden_shape": [hidden],
            "normalized_stage_hidden_shape": [hidden],
            "lm_head_shape": [vocab, hidden],
        }
        for key, expected in expected_shapes.items():
            if _list(multi_layer.get(key)) != expected:
                errors.append(f"multi_layer_stage_decode_shape_mismatch:{key}")
        if multi_layer.get("lm_head_dtype") != "BF16":
            errors.append("multi_layer_stage_decode_dtype_invalid")
        if _int(multi_layer.get("lm_head_rows_scanned")) != vocab:
            errors.append("multi_layer_stage_decode_rows_scanned_mismatch")
        if _int(multi_layer.get("lm_head_block_count")) <= 0:
            errors.append("multi_layer_stage_decode_block_count_missing")
        if _int(multi_layer.get("top_k_count")) != _int(multi_layer.get("top_k")):
            errors.append("multi_layer_stage_decode_top_k_count_mismatch")
        if _int(multi_layer.get("layer_summary_count")) != _int(multi_layer.get("stage_layer_count")):
            errors.append("multi_layer_stage_decode_layer_summary_count_mismatch")
        for key in [
            "initial_decode_hidden_hash_present",
            "stage_hidden_hash_present",
            "normalized_stage_hidden_hash_present",
            "selected_token_id_hash_present",
            "selected_logit_hash_present",
            "top_token_ids_hash_present",
            "top_logits_hash_present",
        ]:
            if multi_layer.get(key) is not True:
                errors.append(f"multi_layer_stage_decode_hash_missing:{key}")
        blockers = set(str(item) for item in _list(multi_layer.get("blockers")))
        for blocker in [
            "glm52_multi_layer_stage_decode_uses_decode_token_chain_only",
            "glm52_multi_layer_stage_decode_prefill_carrier_not_full_layer_outputs",
            "glm52_multi_layer_stage_decode_is_not_full_model_hidden",
            "glm52_multi_layer_stage_decode_is_not_kaggle_runtime",
            "glm52_multi_layer_stage_decode_is_not_same_request",
        ]:
            if blocker not in blockers:
                errors.append(f"multi_layer_stage_decode_missing_boundary:{blocker}")

    full_prefix = _dict(report.get("full_prefix_stage_decode"))
    if full_prefix and full_prefix.get("present") is True:
        if full_prefix.get("public_artifact_safe") is not True:
            errors.append("full_prefix_stage_decode_public_artifact_unsafe")
        if full_prefix.get("source_ok") is not True:
            errors.append("full_prefix_stage_decode_source_not_ok")
        if full_prefix.get("model_id") != pack.MODEL_ID:
            errors.append("full_prefix_stage_decode_model_id_mismatch")
        if full_prefix.get("model_type") != "glm_moe_dsa":
            errors.append("full_prefix_stage_decode_model_type_mismatch")
        for key in [
            "full_prefix_stage_hidden_verified",
            "multi_layer_stage_hidden_verified",
            "full_prefix_token_carrier_verified",
            "all_layers_full_prefix_verified",
            "all_layer_outputs_chained",
            "stage_hidden_to_lm_head_verified",
            "lm_head_streamed_full_vocab",
            "stage_hidden_lm_head_token_selection_verified",
            "partial_full_prefix_token_hash_verified",
        ]:
            if full_prefix.get(key) is not True:
                errors.append(f"full_prefix_stage_decode_{key}_missing")
        for key in [
            "full_model_hidden_verified",
            "generated_token_verified",
            "stage_decode_verified",
            "same_request_decode_verified",
            "live_kaggle_runtime_verified",
        ]:
            if full_prefix.get(key) is True:
                errors.append(f"full_prefix_stage_decode_{key}_overclaim")
        hidden = _int(full_prefix.get("hidden_size"))
        vocab = _int(full_prefix.get("vocab_size"))
        seq_len = _int(full_prefix.get("stage_sequence_length"))
        layer_range = _list(full_prefix.get("stage_layer_range"))
        if seq_len != _int(full_prefix.get("stage_prefill_length")) + 1 or seq_len < 2:
            errors.append("full_prefix_stage_decode_sequence_length_invalid")
        if len(layer_range) != 2 or _int(layer_range[1]) <= _int(layer_range[0]):
            errors.append("full_prefix_stage_decode_range_invalid")
        elif _int(layer_range[1]) - _int(layer_range[0]) != _int(full_prefix.get("stage_layer_count")):
            errors.append("full_prefix_stage_decode_count_mismatch")
        if _int(full_prefix.get("stage_layer_count")) < 2:
            errors.append("full_prefix_stage_decode_count_too_small")
        if _int(full_prefix.get("executed_layer_count")) != _int(full_prefix.get("stage_layer_count")):
            errors.append("full_prefix_stage_decode_executed_count_mismatch")
        if full_prefix.get("small_sequence_probe") is not True:
            errors.append("full_prefix_stage_decode_small_sequence_flag_missing")
        expected_shapes = {
            "stage_hidden_sequence_shape": [seq_len, hidden],
            "stage_hidden_shape": [hidden],
            "normalized_stage_hidden_shape": [hidden],
            "lm_head_shape": [vocab, hidden],
        }
        for key, expected in expected_shapes.items():
            if _list(full_prefix.get(key)) != expected:
                errors.append(f"full_prefix_stage_decode_shape_mismatch:{key}")
        if full_prefix.get("lm_head_dtype") != "BF16":
            errors.append("full_prefix_stage_decode_dtype_invalid")
        if _int(full_prefix.get("lm_head_rows_scanned")) != vocab:
            errors.append("full_prefix_stage_decode_rows_scanned_mismatch")
        if _int(full_prefix.get("lm_head_block_count")) <= 0:
            errors.append("full_prefix_stage_decode_block_count_missing")
        if _int(full_prefix.get("top_k_count")) != _int(full_prefix.get("top_k")):
            errors.append("full_prefix_stage_decode_top_k_count_mismatch")
        if _int(full_prefix.get("layer_summary_count")) != _int(full_prefix.get("stage_layer_count")):
            errors.append("full_prefix_stage_decode_layer_summary_count_mismatch")
        for key in [
            "stage_hidden_sequence_hash_present",
            "stage_hidden_hash_present",
            "normalized_stage_hidden_hash_present",
            "selected_token_id_hash_present",
            "selected_logit_hash_present",
            "top_token_ids_hash_present",
            "top_logits_hash_present",
        ]:
            if full_prefix.get(key) is not True:
                errors.append(f"full_prefix_stage_decode_hash_missing:{key}")
        blockers = set(str(item) for item in _list(full_prefix.get("blockers")))
        for blocker in [
            "glm52_full_prefix_stage_decode_uses_small_sequence_probe",
            "glm52_full_prefix_stage_decode_is_not_kaggle_runtime",
            "glm52_full_prefix_stage_decode_is_not_same_request",
        ]:
            if blocker not in blockers:
                errors.append(f"full_prefix_stage_decode_missing_boundary:{blocker}")

    dsa_indexer = _dict(report.get("dsa_indexer"))
    if dsa_indexer and dsa_indexer.get("present") is True:
        if dsa_indexer.get("public_artifact_safe") is not True:
            errors.append("dsa_indexer_public_artifact_unsafe")
        if dsa_indexer.get("source_ok") is not True:
            errors.append("dsa_indexer_source_not_ok")
        if dsa_indexer.get("model_id") != pack.MODEL_ID:
            errors.append("dsa_indexer_model_id_mismatch")
        if dsa_indexer.get("model_type") != "glm_moe_dsa":
            errors.append("dsa_indexer_model_type_mismatch")
        if dsa_indexer.get("layer_indexer_type") != "full":
            errors.append("dsa_indexer_layer_not_full")
        if dsa_indexer.get("dsa_indexer_verified") is not True:
            errors.append("dsa_indexer_not_verified")
        if dsa_indexer.get("dsa_topk_verified") is not True:
            errors.append("dsa_indexer_topk_not_verified")
        if dsa_indexer.get("indexer_cache_updated") is True:
            errors.append("dsa_indexer_cache_overclaim")
        if dsa_indexer.get("attention_output_verified") is True:
            errors.append("dsa_indexer_attention_output_overclaim")
        if dsa_indexer.get("stage_decode_verified") is True:
            errors.append("dsa_indexer_stage_decode_overclaim")
        seq_len = _int(dsa_indexer.get("sequence_length"))
        heads = _int(dsa_indexer.get("index_n_heads"))
        head_dim = _int(dsa_indexer.get("index_head_dim"))
        expected_shapes = {
            "hidden_norm_shape": [seq_len, _int(dsa_indexer.get("hidden_size"))],
            "q_resid_shape": [seq_len, _int(dsa_indexer.get("q_lora_rank"))],
            "indexer_query_shape": [seq_len, heads, head_dim],
            "indexer_key_shape": [seq_len, head_dim],
            "head_weights_shape": [seq_len, heads],
            "index_score_shape": [seq_len, seq_len],
            "topk_indices_shape": [seq_len, _int(dsa_indexer.get("effective_topk"))],
        }
        for key, expected in expected_shapes.items():
            if _list(dsa_indexer.get(key)) != expected:
                errors.append(f"dsa_indexer_shape_mismatch:{key}")
        for key in [
            "hidden_norm_hash_present",
            "q_resid_hash_present",
            "indexer_query_hash_present",
            "indexer_key_hash_present",
            "head_weights_hash_present",
            "index_score_hash_present",
            "topk_indices_hash_present",
        ]:
            if dsa_indexer.get(key) is not True:
                errors.append(f"dsa_indexer_hash_missing:{key}")
        blockers = set(str(item) for item in _list(dsa_indexer.get("blockers")))
        for blocker in [
            "glm52_dsa_indexer_small_sequence_is_not_full_prefill",
            "glm52_dsa_indexer_is_not_kv_cache_decode",
            "glm52_dsa_indexer_is_not_attention_output",
            "glm52_dsa_indexer_is_not_stage_decode",
        ]:
            if blocker not in blockers:
                errors.append(f"dsa_indexer_missing_boundary:{blocker}")

    pack_dequant = _dict(report.get("pack_quantized_dequant"))
    if pack_dequant and pack_dequant.get("present") is True:
        if pack_dequant.get("public_artifact_safe") is not True:
            errors.append("pack_quantized_dequant_public_artifact_unsafe")
        if pack_dequant.get("source_ok") is not True:
            errors.append("pack_quantized_dequant_source_not_ok")
        if pack_dequant.get("model_id") != pack.MODEL_ID:
            errors.append("pack_quantized_dequant_model_id_mismatch")
        if pack_dequant.get("model_type") != "glm_moe_dsa":
            errors.append("pack_quantized_dequant_model_type_mismatch")
        if "quant" not in str(pack_dequant.get("quantization_format") or "").lower():
            errors.append("pack_quantized_dequant_format_mismatch")
        if pack_dequant.get("pack_quantized_group_loaded") is not True:
            errors.append("pack_quantized_dequant_group_not_loaded")
        if pack_dequant.get("pack_quantized_dequant_verified") is True:
            if pack_dequant.get("q_unpacked_hash_present") is not True:
                errors.append("pack_quantized_dequant_q_hash_missing")
            if pack_dequant.get("zero_point_unpacked_hash_present") is not True:
                errors.append("pack_quantized_dequant_zp_hash_missing")
            if pack_dequant.get("dequant_slice_hash_present") is not True:
                errors.append("pack_quantized_dequant_hash_missing")
            if len(_list(pack_dequant.get("dequant_slice_shape"))) != 2:
                errors.append("pack_quantized_dequant_shape_invalid")
        if pack_dequant.get("pack_quantized_linear_slice_verified") is True:
            if pack_dequant.get("linear_slice_hash_present") is not True:
                errors.append("pack_quantized_dequant_linear_hash_missing")
            if len(_list(pack_dequant.get("linear_slice_shape"))) != 1:
                errors.append("pack_quantized_dequant_linear_shape_invalid")
        if pack_dequant.get("stage_decode_verified") is True:
            errors.append("pack_quantized_dequant_stage_decode_overclaim")
        blockers = set(str(item) for item in _list(pack_dequant.get("blockers")))
        if pack_dequant.get("pack_quantized_dequant_verified") is True and "glm52_pack_quantized_dequant_slice_is_not_full_layer" not in blockers:
            errors.append("pack_quantized_dequant_missing_slice_boundary")
        if pack_dequant.get("pack_quantized_linear_slice_verified") is True and "glm52_pack_quantized_linear_slice_is_not_stage_decode" not in blockers:
            errors.append("pack_quantized_dequant_missing_linear_boundary")

    expert_mlp = _dict(report.get("pack_quantized_expert_mlp"))
    if expert_mlp and expert_mlp.get("present") is True:
        if expert_mlp.get("public_artifact_safe") is not True:
            errors.append("pack_quantized_expert_mlp_public_artifact_unsafe")
        if expert_mlp.get("source_ok") is not True:
            errors.append("pack_quantized_expert_mlp_source_not_ok")
        if expert_mlp.get("model_id") != pack.MODEL_ID:
            errors.append("pack_quantized_expert_mlp_model_id_mismatch")
        if expert_mlp.get("model_type") != "glm_moe_dsa":
            errors.append("pack_quantized_expert_mlp_model_type_mismatch")
        if expert_mlp.get("pack_quantized_expert_mlp_verified") is not True:
            errors.append("pack_quantized_expert_mlp_not_verified")
        if expert_mlp.get("single_expert_mlp_verified") is not True:
            errors.append("pack_quantized_expert_mlp_single_expert_not_verified")
        projections = [item for item in _list(expert_mlp.get("projection_summaries")) if isinstance(item, dict)]
        if [str(item.get("projection") or "") for item in projections] != ["gate_proj", "up_proj", "down_proj"]:
            errors.append("pack_quantized_expert_mlp_projection_order_invalid")
        for item in projections:
            projection = str(item.get("projection") or "missing")
            if item.get("pack_quantized_group_loaded") is not True:
                errors.append(f"pack_quantized_expert_mlp_group_not_loaded:{projection}")
            if item.get("output_hash_present") is not True:
                errors.append(f"pack_quantized_expert_mlp_output_hash_missing:{projection}")
            if len(_list(item.get("weight_shape"))) != 2:
                errors.append(f"pack_quantized_expert_mlp_weight_shape_invalid:{projection}")
            if len(_list(item.get("output_shape"))) != 1:
                errors.append(f"pack_quantized_expert_mlp_output_shape_invalid:{projection}")
        if _list(expert_mlp.get("final_output_shape")) != [_int(expert_mlp.get("hidden_size"))]:
            errors.append("pack_quantized_expert_mlp_final_shape_mismatch")
        if expert_mlp.get("final_output_hash_present") is not True:
            errors.append("pack_quantized_expert_mlp_final_hash_missing")
        if expert_mlp.get("stage_decode_verified") is True:
            errors.append("pack_quantized_expert_mlp_stage_decode_overclaim")
        blockers = set(str(item) for item in _list(expert_mlp.get("blockers")))
        for blocker in [
            "glm52_pack_quantized_expert_mlp_is_single_expert_only",
            "glm52_pack_quantized_expert_mlp_is_not_attention",
            "glm52_pack_quantized_expert_mlp_is_not_topk_router",
            "glm52_pack_quantized_expert_mlp_is_not_stage_decode",
        ]:
            if blocker not in blockers:
                errors.append(f"pack_quantized_expert_mlp_missing_boundary:{blocker}")

    router_gather = _dict(report.get("pack_quantized_router_gather"))
    if router_gather and router_gather.get("present") is True:
        if router_gather.get("public_artifact_safe") is not True:
            errors.append("pack_quantized_router_gather_public_artifact_unsafe")
        if router_gather.get("source_ok") is not True:
            errors.append("pack_quantized_router_gather_source_not_ok")
        if router_gather.get("model_id") != pack.MODEL_ID:
            errors.append("pack_quantized_router_gather_model_id_mismatch")
        if router_gather.get("model_type") != "glm_moe_dsa":
            errors.append("pack_quantized_router_gather_model_type_mismatch")
        if router_gather.get("router_topk_verified") is not True:
            errors.append("pack_quantized_router_gather_topk_not_verified")
        if router_gather.get("routed_expert_subset_verified") is not True:
            errors.append("pack_quantized_router_gather_subset_not_verified")
        if _int(router_gather.get("router_topk_count")) != _int(router_gather.get("num_experts_per_tok")):
            errors.append("pack_quantized_router_gather_topk_count_mismatch")
        if router_gather.get("router_topk_indices_hash_present") is not True:
            errors.append("pack_quantized_router_gather_indices_hash_missing")
        if router_gather.get("router_topk_weights_hash_present") is not True:
            errors.append("pack_quantized_router_gather_weights_hash_missing")
        if _int(router_gather.get("executed_expert_count")) <= 0:
            errors.append("pack_quantized_router_gather_executed_count_missing")
        executed = [item for item in _list(router_gather.get("executed_experts")) if isinstance(item, dict)]
        if len(executed) != _int(router_gather.get("executed_expert_count")):
            errors.append("pack_quantized_router_gather_executed_summary_count_mismatch")
        for item in executed:
            if _int(item.get("expert_id"), -1) < 0:
                errors.append("pack_quantized_router_gather_expert_id_invalid")
            if item.get("expert_weight_hash_present") is not True:
                errors.append("pack_quantized_router_gather_expert_weight_hash_missing")
            if item.get("expert_output_hash_present") is not True:
                errors.append("pack_quantized_router_gather_expert_output_hash_missing")
            if _list(item.get("expert_output_shape")) != [_int(router_gather.get("hidden_size"))]:
                errors.append("pack_quantized_router_gather_expert_output_shape_mismatch")
        if _list(router_gather.get("routed_subset_output_shape")) != [_int(router_gather.get("hidden_size"))]:
            errors.append("pack_quantized_router_gather_output_shape_mismatch")
        if router_gather.get("routed_subset_output_hash_present") is not True:
            errors.append("pack_quantized_router_gather_output_hash_missing")
        if router_gather.get("stage_decode_verified") is True:
            errors.append("pack_quantized_router_gather_stage_decode_overclaim")
        blockers = set(str(item) for item in _list(router_gather.get("blockers")))
        for blocker in [
            "glm52_pack_quantized_router_gather_is_subset_only",
            "glm52_pack_quantized_router_gather_missing_shared_experts",
            "glm52_pack_quantized_router_gather_is_not_attention",
            "glm52_pack_quantized_router_gather_is_not_stage_decode",
        ]:
            if blocker not in blockers:
                errors.append(f"pack_quantized_router_gather_missing_boundary:{blocker}")

    moe_mlp = _dict(report.get("pack_quantized_moe_mlp"))
    if moe_mlp and moe_mlp.get("present") is True:
        if moe_mlp.get("public_artifact_safe") is not True:
            errors.append("pack_quantized_moe_mlp_public_artifact_unsafe")
        if moe_mlp.get("source_ok") is not True:
            errors.append("pack_quantized_moe_mlp_source_not_ok")
        if moe_mlp.get("model_id") != pack.MODEL_ID:
            errors.append("pack_quantized_moe_mlp_model_id_mismatch")
        if moe_mlp.get("model_type") != "glm_moe_dsa":
            errors.append("pack_quantized_moe_mlp_model_type_mismatch")
        for key in [
            "router_topk_verified",
            "routed_expert_gather_verified",
            "shared_experts_mlp_verified",
            "pack_quantized_moe_mlp_verified",
            "full_moe_mlp_verified",
        ]:
            if moe_mlp.get(key) is not True:
                errors.append(f"pack_quantized_moe_mlp_{key}_missing")
        if _int(moe_mlp.get("router_topk_count")) != _int(moe_mlp.get("num_experts_per_tok")):
            errors.append("pack_quantized_moe_mlp_topk_count_mismatch")
        if _int(moe_mlp.get("executed_expert_count")) != _int(moe_mlp.get("num_experts_per_tok")):
            errors.append("pack_quantized_moe_mlp_executed_count_not_full_topk")
        if moe_mlp.get("router_topk_indices_hash_present") is not True:
            errors.append("pack_quantized_moe_mlp_indices_hash_missing")
        if moe_mlp.get("router_topk_weights_hash_present") is not True:
            errors.append("pack_quantized_moe_mlp_weights_hash_missing")
        executed = [item for item in _list(moe_mlp.get("executed_experts")) if isinstance(item, dict)]
        if len(executed) != _int(moe_mlp.get("executed_expert_count")):
            errors.append("pack_quantized_moe_mlp_executed_summary_count_mismatch")
        for item in executed:
            if _int(item.get("expert_id"), -1) < 0:
                errors.append("pack_quantized_moe_mlp_expert_id_invalid")
            if item.get("expert_weight_hash_present") is not True:
                errors.append("pack_quantized_moe_mlp_expert_weight_hash_missing")
            if item.get("expert_output_hash_present") is not True:
                errors.append("pack_quantized_moe_mlp_expert_output_hash_missing")
            if _list(item.get("expert_output_shape")) != [_int(moe_mlp.get("hidden_size"))]:
                errors.append("pack_quantized_moe_mlp_expert_output_shape_mismatch")
        shared = [item for item in _list(moe_mlp.get("shared_projection_summaries")) if isinstance(item, dict)]
        if [str(item.get("projection") or "") for item in shared] != ["gate_proj", "up_proj", "down_proj"]:
            errors.append("pack_quantized_moe_mlp_shared_projection_order_invalid")
        for item in shared:
            projection = str(item.get("projection") or "missing")
            if str(item.get("weight_dtype") or "") not in {"bfloat16", "float16", "float32"}:
                errors.append(f"pack_quantized_moe_mlp_shared_dtype_invalid:{projection}")
            if len(_list(item.get("weight_shape"))) != 2:
                errors.append(f"pack_quantized_moe_mlp_shared_weight_shape_invalid:{projection}")
            if len(_list(item.get("output_shape"))) != 1:
                errors.append(f"pack_quantized_moe_mlp_shared_output_shape_invalid:{projection}")
            if item.get("output_hash_present") is not True:
                errors.append(f"pack_quantized_moe_mlp_shared_output_hash_missing:{projection}")
        for key in ["routed_output_shape", "shared_output_shape", "full_moe_output_shape"]:
            if _list(moe_mlp.get(key)) != [_int(moe_mlp.get("hidden_size"))]:
                errors.append(f"pack_quantized_moe_mlp_shape_mismatch:{key}")
        for key in ["routed_output_hash_present", "shared_output_hash_present", "full_moe_output_hash_present"]:
            if moe_mlp.get(key) is not True:
                errors.append(f"pack_quantized_moe_mlp_hash_missing:{key}")
        if moe_mlp.get("stage_decode_verified") is True:
            errors.append("pack_quantized_moe_mlp_stage_decode_overclaim")
        blockers = set(str(item) for item in _list(moe_mlp.get("blockers")))
        for blocker in [
            "glm52_pack_quantized_moe_mlp_is_not_attention",
            "glm52_pack_quantized_moe_mlp_is_not_transformer_block",
            "glm52_pack_quantized_moe_mlp_is_not_stage_decode",
            "glm52_pack_quantized_moe_mlp_missing_kv_cache",
            "glm52_pack_quantized_moe_mlp_missing_lm_head",
        ]:
            if blocker not in blockers:
                errors.append(f"pack_quantized_moe_mlp_missing_boundary:{blocker}")

    decode_gap = _dict(report.get("decode_adapter_gap"))
    if decode_gap and decode_gap.get("present") is True:
        if decode_gap.get("public_artifact_safe") is not True:
            errors.append("decode_adapter_gap_public_artifact_unsafe")
        if decode_gap.get("source_ok") is not True:
            errors.append("decode_adapter_gap_source_not_ok")
        if decode_gap.get("model_id") != pack.MODEL_ID:
            errors.append("decode_adapter_gap_model_id_mismatch")
        if decode_gap.get("model_type") != "glm_moe_dsa":
            errors.append("decode_adapter_gap_model_type_mismatch")
        if _int(decode_gap.get("num_hidden_layers")) <= 0:
            errors.append("decode_adapter_gap_layer_count_missing")
        if _int(decode_gap.get("n_routed_experts")) <= 0:
            errors.append("decode_adapter_gap_moe_experts_missing")
        if _int(decode_gap.get("required_capability_count")) <= 0:
            errors.append("decode_adapter_gap_capabilities_missing")
        if decode_gap.get("decode_adapter_ready") is True:
            if decode_gap.get("same_request_decode_ready") is not True:
                errors.append("decode_adapter_gap_ready_without_same_request")
            if decode_gap.get("missing_capabilities"):
                errors.append("decode_adapter_gap_ready_with_missing_capabilities")
            if not REQUIRED_PROVIDERS.issubset(set(str(item) for item in _list(decode_gap.get("stage_decode_provider_coverage")))):
                errors.append("decode_adapter_gap_ready_without_provider_decode")
        else:
            if not decode_gap.get("blockers"):
                errors.append("decode_adapter_gap_not_ready_missing_blockers")
            if "glm52_full_decode_adapter_not_ready" not in set(str(item) for item in _list(decode_gap.get("blockers"))):
                errors.append("decode_adapter_gap_missing_not_ready_blocker")

    same = _dict(report.get("same_request"))
    success = _dict(report.get("success"))
    goal_achieved = report.get("goal_achieved") is True
    success_verified = success.get("same_request_decode_verified") is True
    same_verified = same.get("same_request_decode_verified") is True
    accepted = set(str(item) for item in _list(same.get("accepted_providers") or success.get("accepted_providers")))
    generated = _int(same.get("generated_token_count") or success.get("generated_token_count"))

    if goal_achieved != success_verified:
        errors.append("goal_achieved_success_mismatch")
    if goal_achieved:
        if source.get("resolver_ready") is not True:
            errors.append("success_without_source_resolver")
        if source.get("compatible_with_glm52") is not True:
            errors.append("success_without_glm52_source")
        if tpu.get("tpu_runtime_ready") is not True:
            errors.append("success_without_tpu_runtime_ready")
        if same_verified is not True:
            errors.append("success_without_same_request_verified")
        if same.get("model_id") != pack.MODEL_ID:
            errors.append("success_model_id_not_glm52")
        if generated < 1:
            errors.append("success_generated_token_missing")
        if same.get("generated_token_hash_present") is not True:
            errors.append("success_generated_token_hash_missing")
        if same.get("live_run_performed") is not True:
            errors.append("success_without_live_run")
        if same.get("coordinator_request_verified") is not True:
            errors.append("success_without_coordinator_request")
        if same.get("stage_provider_coverage_verified") is not True:
            errors.append("success_without_stage_provider_coverage")
        if same.get("cleanup_verified") is not True:
            errors.append("success_without_cleanup")
        if not REQUIRED_PROVIDERS.issubset(accepted):
            errors.append("success_required_providers_missing")
        if report.get("blockers"):
            errors.append("success_with_blockers")
    else:
        if success_verified:
            errors.append("success_verified_true_when_goal_false")
        if not report.get("blockers"):
            errors.append("blocked_report_missing_blockers")
        if tpu.get("queued") is True and success.get("same_request_decode_verified") is True:
            errors.append("queued_tpu_must_not_count_as_success")

    boundary = _dict(report.get("completion_boundary"))
    for key in [
        "queue_evidence_is_not_success",
        "metadata_only_source_is_not_success",
        "kaggle_source_search_is_not_success",
        "stage_header_evidence_is_not_success",
        "stage_value_probe_evidence_is_not_success",
        "stage_smoke_evidence_is_not_success",
        "transformers_decode_preflight_is_not_success",
        "attention_projection_is_not_success",
        "attention_single_token_is_not_success",
        "kv_cache_decode_is_not_success",
        "layer_decode_is_not_success",
        "lm_head_token_selection_is_not_success",
        "dsa_masked_layer_decode_is_not_success",
        "stage_hidden_lm_head_is_not_success",
        "multi_layer_stage_decode_is_not_success",
        "full_prefix_stage_decode_is_not_success",
        "dsa_indexer_is_not_success",
        "pack_quantized_dequant_slice_is_not_success",
        "pack_quantized_expert_mlp_is_not_success",
        "pack_quantized_router_gather_subset_is_not_success",
        "pack_quantized_moe_mlp_is_not_success",
        "decode_adapter_gap_evidence_is_not_success",
        "single_backend_inference_is_not_success",
        "fallback_model_is_not_success",
        "requires_real_kaggle_cpu_gpu_tpu_same_request",
    ]:
        if boundary.get(key) is not True:
            errors.append(f"completion_boundary_missing:{key}")

    safety = _dict(report.get("safety"))
    if safety.get("public_artifact_safe") is not True:
        errors.append("safety_public_artifact_safe_missing")
    for key in [
        "credentials_public",
        "cookies_public",
        "signed_url_public",
        "raw_prompt_public",
        "raw_generated_text_public",
        "generated_token_ids_public",
        "activation_public",
        "hidden_state_public",
        "logits_public",
        "kv_cache_public",
        "weight_tensor_values_public",
    ]:
        if safety.get(key) is not False:
            errors.append(f"safety_flag_not_false:{key}")
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = load_json(Path(args.report))
    errors = validate_report(report)
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors,
        "goal_achieved": report.get("goal_achieved") is True,
        "same_request_decode_verified": _dict(report.get("success")).get("same_request_decode_verified") is True,
        "failure_stage": report.get("failure_stage"),
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_kaggle_accelerator_deployment_rc_check: ok={result['ok']} "
            f"errors={len(errors)} achieved={result['goal_achieved']} failure_stage={result['failure_stage']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
