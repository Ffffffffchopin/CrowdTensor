from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts import glm52_kaggle_accelerator_deployment_rc_check as check
from scripts import glm52_kaggle_accelerator_deployment_rc_pack as pack


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ct_glm52_rc_"))


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _source_report(*, stage_runtime: bool = False, route: bool = False) -> dict:
    return {
        "schema": "glm52_model_source_resolver_v1",
        "ok": True,
        "glm52_source_resolver_ready": True,
        "public_artifact_safe": True,
        "model": {
            "model_id": pack.MODEL_ID,
            "architecture_class": "moe",
            "model_type": "glm_moe_dsa",
            "num_hidden_layers": 78,
            "official_weight_key_count": 59585,
            "official_weight_total_size_gb": 1506.659,
        },
        "candidate_count": 2,
        "ready_candidate_count": 2,
        "recommended_deployment_candidate": {
            "candidate_id": "awq-int4-safetensors",
            "repo": "cyankiwi/GLM-5.2-AWQ-INT4",
            "format": "safetensors",
            "quantization": "AWQ-INT4",
            "known_total_size_gb": 440,
            "weight_file_count": 83,
            "blockers": ["candidate_exceeds_runtime_disk_budget"],
        },
        "stage_adapter_plan": {
            "schema": "glm52_kaggle_stage_adapter_plan_v1",
            "metadata_only": True,
            "assigned_key_count_total": 59585,
            "stage_runtime_adapter_verified": stage_runtime,
            "same_request_route_verified": route,
        },
        "kaggle_attach_plan": {
            "kaggle_models_source_verified": False,
            "hf_source_verified": True,
            "full_runtime_download_supported": False,
        },
        "blockers": [] if stage_runtime and route else ["glm52_stage_runtime_adapter_not_verified"],
    }


def _tpu_report(*, ready: bool = False, queued: bool = True) -> dict:
    status = "COMPLETE" if ready else ("KernelWorkerStatus.QUEUED" if queued else "FAILED")
    return {
        "schema": "glm52_kaggle_tpu_retained_request_watch_v1",
        "ref": "tpuowner/ct-mcp-tpu-probe-0704-r2",
        "last_status": status,
        "tpu_runtime_ready": ready,
        "observations": [{"attempt": 1, "status": status, "ok": True}],
        "public_artifact_safe": True,
    }


def _mcp_tpu_stage_runtime_watch(*, ready: bool = False) -> dict:
    status = "KernelWorkerStatus.COMPLETE" if ready else "KernelWorkerStatus.QUEUED"
    return {
        "schema": "glm52_mcp_tpu_stage_runtime_watch_v1",
        "ref": "tpuowner/ct-glm52-tpu-value-op-r1",
        "last_status": status,
        "last_status_class": "complete" if ready else "queued",
        "stage_runtime_report_verified": ready,
        "tpu_stage_runtime_ready": ready,
        "same_request_decode_verified": False,
        "observations": [{"attempt": 1, "status": status, "ok": True}],
        "blockers": [] if ready else [
            "glm52_mcp_tpu_stage_runtime_not_ready",
            "glm52_mcp_tpu_stage_runtime_scheduler_queued",
        ],
        "public_artifact_safe": True,
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
    }


def _same_request(*, model_id: str = pack.MODEL_ID, verified: bool = True) -> dict:
    request_hash = "sha256:" + "b" * 64
    stages = [
        {
            "schema": "glm52_kaggle_stage_runtime_report_v1",
            "provider": provider,
            "stage_id": index,
            "model_id": model_id,
            "coordinator_request_id_hash": request_hash,
            "stage_execution_verified": True,
            "stage_decode_verified": True,
            "stage_output_hash": "sha256:" + str(index) * 64,
            "weight_tensor_values_loaded": True,
            "weight_value_byte_count": 16,
            "weight_value_sha256": "sha256:" + "w" * 64,
            "weight_tensor_values_public": False,
            "stage_layer_range": [index * 2, index * 2 + 1],
            "live_run_performed": True,
            "public_artifact_safe": True,
        }
        for index, provider in enumerate(pack.REQUIRED_PROVIDERS)
    ]
    return {
        "schema": "glm52_kaggle_same_request_probe_v1",
        "ok": verified,
        "glm52_kaggle_same_request_verified": verified,
        "same_request_decode_verified": verified,
        "live_run_performed": verified,
        "public_artifact_safe": True,
        "model": {"model_id": model_id},
        "success": {
            "same_request_decode_verified": verified,
            "generated_token_count": 1,
            "generated_token_hash": "sha256:" + "a" * 64,
            "accepted_providers": pack.REQUIRED_PROVIDERS,
        },
        "same_request": {
            "coordinator_request_verified": True,
            "coordinator_request_id_hash": request_hash,
            "model_id": model_id,
        },
        "stage_reports": stages,
        "cleanup": {
            "temporary_kaggle_kernels_deleted": True,
            "temporary_private_packages_removed": True,
            "live_resources_left_running": False,
            "public_artifact_safe": True,
        },
    }


def _awq_stage_header(*, ready: bool = True) -> dict:
    return {
        "schema": "glm52_awq_stage_header_probe_v1",
        "ok": ready,
        "glm52_awq_stage_header_ready": ready,
        "public_artifact_safe": True,
        "model_repo": "cyankiwi/GLM-5.2-AWQ-INT4",
        "base_model_id": pack.MODEL_ID,
        "quantization": "AWQ-INT4",
        "stage_id": 4,
        "stage_count": 12,
        "stage_layer_range": [28, 35],
        "assigned_weight_key_count": 21675,
        "assigned_weight_file_count": 8,
        "header_file_count": 8,
        "present_stage_key_count": 21675,
        "missing_stage_key_count": 0,
        "dtype_counts": {"BF16": 5477, "I32": 10794, "I64": 5397},
        "stage_family_hits": {"awq_quantized_tensors": True, "attention": True, "mlp_or_moe": True},
        "total_selected_tensor_storage_gb": 40.524259,
        "weight_tensor_values_loaded": False,
        "weight_tensor_values_public": False,
        "safetensors_header_payload_public": False,
        "stage_runtime_adapter_verified": False,
        "same_request_route_verified": False,
        "blockers": [] if ready else ["glm52_awq_stage_header_missing_keys"],
    }


def _awq_stage_value_probe(*, same_request_overclaim: bool = False) -> dict:
    return _awq_stage_value_probe_for_stage(4, [28, 35], same_request_overclaim=same_request_overclaim)


def _awq_stage_value_probe_for_stage(stage_id: int, layer_range: list[int], *, same_request_overclaim: bool = False) -> dict:
    return {
        "schema": "glm52_awq_stage_value_probe_v1",
        "ok": True,
        "glm52_awq_stage_value_probe_ready": True,
        "public_artifact_safe": True,
        "model_repo": "cyankiwi/GLM-5.2-AWQ-INT4",
        "base_model_id": pack.MODEL_ID,
        "quantization": "AWQ-INT4",
        "stage_id": stage_id,
        "stage_count": 3 if stage_id in {0, 1, 2} else 12,
        "stage_layer_range": layer_range,
        "assigned_weight_key_count": 21675,
        "assigned_weight_file_count": 8,
        "header_file_count": 8,
        "selected_tensor": {
            "key_digest": "sha256:" + "a" * 64,
            "filename": "model-00029-of-00083.safetensors",
            "dtype": "I64",
            "rank": 1,
            "tensor_nbytes": 16,
        },
        "weight_value_byte_count": 16,
        "weight_value_sha256": "sha256:" + "b" * 64,
        "weight_tensor_values_loaded": True,
        "weight_tensor_values_public": False,
        "safetensors_header_payload_public": False,
        "stage_runtime_adapter_verified": False,
        "same_request_route_verified": False,
        "same_request_decode_verified": same_request_overclaim,
        "stage_smoke_only": True,
        "blockers": [],
    }


def _tpu_stage_smoke(*, same_request_overclaim: bool = False) -> dict:
    return {
        "schema": "glm52_awq_tpu_stage_smoke_v1",
        "ok": True,
        "public_artifact_safe": True,
        "model_repo": "cyankiwi/GLM-5.2-AWQ-INT4",
        "base_model_id": pack.MODEL_ID,
        "quantization": "AWQ-INT4",
        "tpu_runtime_ready": True,
        "jax_tpu_device_count": 8,
        "jax_shape_smoke_ready": True,
        "glm52_awq_stage_header_ready": True,
        "stage_id": 4,
        "stage_count": 12,
        "stage_layer_range": [28, 35],
        "assigned_weight_key_count": 21675,
        "assigned_weight_file_count": 8,
        "header_file_count": 8,
        "present_stage_key_count": 21675,
        "missing_stage_key_count": 0,
        "dtype_counts": {"BF16": 5477, "I32": 10794, "I64": 5397},
        "stage_family_hits": {"awq_quantized_tensors": True, "attention": True, "mlp_or_moe": True},
        "total_selected_tensor_storage_gb": 40.524259,
        "weight_tensor_values_loaded": False,
        "weight_tensor_values_public": False,
        "safetensors_header_payload_public": False,
        "stage_runtime_adapter_verified": False,
        "same_request_route_verified": False,
        "same_request_decode_verified": same_request_overclaim,
        "blockers": [],
    }


def _tpu_stage_smoke_watch() -> dict:
    return {
        "schema": "glm52_kaggle_tpu_awq_stage_smoke_watch_v1",
        "ref": "tpuowner/ct-glm52-awq-tpu-stage-smoke-0704-r1",
        "last_status": "KernelWorkerStatus.QUEUED",
        "observations": [{"attempt": 1, "status": "KernelWorkerStatus.QUEUED", "ok": True}],
        "tpu_runtime_ready": False,
        "stage_runtime_adapter_smoke_ready": False,
        "notebook_output_verified": False,
        "public_artifact_safe": True,
        "credentials_public": False,
        "signed_output_url_public": False,
    }


def _kaggle_source_search(*, verified: bool = False) -> dict:
    return {
        "schema": "glm52_kaggle_public_source_search_v1",
        "ok": True,
        "glm52_kaggle_public_source_search_ready": True,
        "public_artifact_safe": True,
        "query_count": 6,
        "model_result_count": 10,
        "dataset_result_count": 24,
        "compatible_model_source_count": 1 if verified else 0,
        "compatible_dataset_source_count": 0,
        "kaggle_models_glm52_source_verified": verified,
        "kaggle_datasets_glm52_source_verified": False,
        "kaggle_attach_source_verified": verified,
        "recommended_kaggle_kernel_model_sources": ["zai-org/glm-5-2/PyTorch/default/1"] if verified else [],
        "model_results": [],
        "dataset_results": [],
        "compatible_model_candidates": [{"ref": "zai-org/glm-5-2"}] if verified else [],
        "compatible_dataset_candidates": [],
        "blockers": [] if verified else [
            "kaggle_models_glm52_weight_source_not_found",
            "kaggle_datasets_glm52_weight_source_not_found",
        ],
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
        },
    }


def _stage_runtime_plan() -> dict:
    return {
        "schema": "glm52_kaggle_stage_runtime_plan_v1",
        "ok": True,
        "glm52_stage_runtime_plan_ready": True,
        "stage_runtime_adapter_verified": False,
        "same_request_route_verified": False,
        "public_artifact_safe": True,
        "model": {
            "model_id": pack.MODEL_ID,
            "compatible_weight_repo": "cyankiwi/GLM-5.2-AWQ-INT4",
            "fallback_model_allowed_for_success": False,
        },
        "stage_specs": [
            {
                "provider": provider,
                "stage_id": index,
                "stage_layer_range": [index * 26, (index + 1) * 26],
                "expected_stage_report_schema": "glm52_kaggle_stage_runtime_report_v1",
                "stage_runtime_adapter_verified": False,
                "same_request_route_verified": False,
                "public_artifact_safe": True,
            }
            for index, provider in enumerate(pack.REQUIRED_PROVIDERS)
        ],
        "launcher_contract": {
            "expected_stage_report_schema": "glm52_kaggle_stage_runtime_report_v1",
            "private_kernel_required": True,
            "public_artifact_safe": True,
        },
        "completion_boundary": {
            "plan_is_not_runtime_success": True,
            "stage_runtime_report_required": True,
            "same_request_probe_required": True,
            "queue_or_stage_smoke_is_not_success": True,
        },
        "blockers": ["glm52_stage_runtime_live_reports_missing"],
    }


def _stage_worker_package() -> dict:
    return {
        "schema": "glm52_kaggle_stage_worker_package_v1",
        "ok": True,
        "glm52_stage_worker_package_ready": True,
        "stage_runtime_adapter_verified": False,
        "same_request_route_verified": False,
        "live_run_performed": False,
        "public_artifact_safe": True,
        "model": {"model_id": pack.MODEL_ID, "fallback_model_allowed_for_success": False},
        "packages": [
            {
                "provider": provider,
                "stage_id": index,
                "stage_layer_range": [index * 26, (index + 1) * 26],
                "expected_stage_report_schema": "glm52_kaggle_stage_runtime_report_v1",
                "private_kernel": True,
                "pushed_to_kaggle": False,
                "live_run_performed": False,
                "public_artifact_safe": True,
            }
            for index, provider in enumerate(pack.REQUIRED_PROVIDERS)
        ],
        "completion_boundary": {
            "package_is_not_runtime_success": True,
            "kaggle_push_required": True,
            "live_stage_report_required": True,
            "same_request_probe_required": True,
        },
        "blockers": ["glm52_stage_worker_package_is_not_runtime_success"],
    }


def _stage_worker_push_probe() -> dict:
    return {
        "schema": "glm52_kaggle_stage_worker_push_probe_v1",
        "ok": True,
        "glm52_stage_worker_push_probe_ready": True,
        "mode": "preflight",
        "live_run_performed": False,
        "stage_runtime_reports_collected": 0,
        "stage_runtime_reports_verified": 0,
        "stage_runtime_adapter_verified": False,
        "same_request_route_verified": False,
        "public_artifact_safe": True,
        "pushes": [
            {
                "provider": provider,
                "stage_id": index,
                "pushed": False,
                "terminal_status": "",
                "output_collected": False,
                "stage_report_present": False,
                "stage_runtime_verified": False,
                "stage_report_check": {"ok": False, "stage_runtime_verified": False},
                "cleanup_performed": False,
                "public_artifact_safe": True,
            }
            for index, provider in enumerate(pack.REQUIRED_PROVIDERS)
        ],
        "completion_boundary": {
            "preflight_is_not_runtime_success": True,
            "push_required": True,
            "terminal_kernel_output_required": True,
            "stage_runtime_check_required": True,
            "same_request_probe_required": True,
        },
        "blockers": ["glm52_stage_worker_push_not_started"],
    }


def test_summarize_gpu_token_quota_records_session_limit_blocker() -> None:
    summary = pack.summarize_gpu_token_quota({
        "schema": "kaggle_gpu_token_weekly_quota_probe_v1",
        "requested_accelerator": "NvidiaTeslaT4",
        "public_artifact_safe": True,
        "token_values_public": False,
        "summary": {
            "account_count": 3,
            "auth_ok_count": 3,
            "auth_failed_count": 0,
            "gpu_submission_accepted_count": 0,
            "gpu_session_limit_rejected_count": 3,
            "weekly_gpu_quota_exhausted_count": 1,
            "weekly_gpu_quota_exhausted_by_api_count": 1,
            "gpu_reserved_exceeds_remaining_by_api_count": 2,
            "quota_classes": {
                "tpuowner": "gpu_session_limit_rejected",
                "primary Kaggle account": "gpu_session_limit_rejected",
                "cpuowner": "gpu_session_limit_rejected",
            },
        },
        "accounts": [
            {
                "label": "tpuowner",
                "owner": "tpuowner",
                "auth_ok": True,
                "push_accepted": False,
                "quota_class": "gpu_session_limit_rejected",
                "weekly_gpu_quota_exhausted": False,
                "weekly_gpu_quota_exhausted_by_api": False,
                "gpu_reserved_exceeds_remaining_by_api": True,
                "weekly_gpu_quota_available_inferred": False,
                "accelerator_quota": {
                    "quota_refresh_time": "2026-07-11T00:00:00",
                    "gpu_quota": {
                        "present": True,
                        "time_used_seconds": 90,
                        "time_reserved_seconds": 20,
                        "total_time_allowed_seconds": 100,
                        "remaining_seconds": 10,
                        "effective_remaining_after_reserved_seconds": 0,
                        "quota_exhausted_by_used": False,
                        "reserved_exceeds_remaining": True,
                    },
                    "tpu_quota": {"present": True, "remaining_seconds": 100},
                },
            }
        ],
    })

    assert summary["present"] is True
    assert summary["gpu_slot_available"] is False
    assert summary["gpu_session_limit_rejected_count"] == 3
    assert summary["weekly_gpu_quota_exhausted_by_api_count"] == 1
    assert summary["gpu_reserved_exceeds_remaining_by_api_count"] == 2
    assert "kaggle_gpu_batch_session_limit_reached" in summary["blockers"]
    assert "kaggle_gpu_all_accounts_session_limited" in summary["blockers"]
    assert "kaggle_weekly_gpu_quota_exhausted_by_api" in summary["blockers"]
    assert "kaggle_gpu_reserved_time_exceeds_remaining_quota" in summary["blockers"]
    assert summary["accounts"][0]["accelerator_quota"]["gpu_quota"]["effective_remaining_after_reserved_seconds"] == 0
    assert summary["token_values_public"] is False


def _decode_adapter_gap(*, ready: bool = False) -> dict:
    return {
        "schema": "glm52_decode_adapter_gap_probe_v1",
        "ok": True,
        "glm52_decode_adapter_gap_probe_ready": True,
        "decode_adapter_ready": ready,
        "same_request_decode_ready": ready,
        "public_artifact_safe": True,
        "model": {
            "model_id": pack.MODEL_ID,
            "model_repo": "cyankiwi/GLM-5.2-AWQ-INT4",
            "config_ready": True,
            "index_ready": True,
            "model_type": "glm_moe_dsa",
            "num_hidden_layers": 78,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "weight_key_count": 232269,
        },
        "required_capabilities": [
            {
                "capability": "awq_int4_dequant_linear_runtime",
                "required": True,
                "verified": ready,
            },
            {
                "capability": "glm_moe_dsa_topk_router_and_expert_gather",
                "required": True,
                "verified": ready,
            },
        ],
        "stage_runtime_provider_coverage": pack.REQUIRED_PROVIDERS,
        "stage_decode_provider_coverage": pack.REQUIRED_PROVIDERS if ready else [],
        "blockers": [] if ready else [
            "glm52_full_decode_adapter_not_ready",
            "glm52_decode_capability_missing:awq_int4_dequant_linear_runtime",
        ],
        "completion_boundary": {
            "stage_runtime_value_op_is_not_decode": True,
            "requires_transformer_block_semantics": True,
            "requires_awq_dequant_linear_runtime": True,
            "requires_moe_router_and_expert_runtime": True,
            "requires_generated_token_hash": True,
            "requires_cpu_gpu_tpu_same_request": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
    }


def _transformers_decode_preflight(*, ready: bool = False) -> dict:
    return {
        "schema": "glm52_transformers_decode_adapter_preflight_v1",
        "ok": True,
        "glm52_transformers_decode_adapter_preflight_ready": True,
        "decode_adapter_ready": ready,
        "adapter_foundation_ready": True,
        "public_artifact_safe": True,
        "model": {
            "model_id": pack.MODEL_ID,
            "model_type": "glm_moe_dsa",
            "normalization_action": "removed_invalid_layer_types",
            "quantization_format": "pack-quantized",
            "quantization_weight_bits": [4],
        },
        "transformers_runtime": {
            "transformers_version": "5.9.0",
            "tiny_forward_ready": True,
            "awq_config_normalized_ready": True,
        },
        "pack_quantized_runtime": {"ready": ready},
        "stage_weight_mapping": {
            "stage_weight_mapping_ready": True,
            "selected_layer_count": 78,
            "dense_layer_count": 3,
            "sparse_layer_count": 75,
            "full_indexer_layer_count": 21,
            "shared_indexer_layer_count": 57,
            "required_key_count": 232266,
            "pack_required_key_count": 231300,
            "missing_required_key_count": 0,
        },
        "blockers": [] if ready else [
            "glm52_full_decode_adapter_not_ready",
            "glm52_pack_quantized_runtime_dependency_missing",
            "glm52_transformers_preflight_is_not_full_decode",
        ],
        "completion_boundary": {
            "preflight_is_not_decode_success": True,
            "tiny_random_forward_is_not_glm52_inference": True,
            "weight_mapping_is_not_weight_loading": True,
            "requires_pack_quantized_dequant_runtime": True,
            "requires_stage_decode_verified": True,
            "requires_same_request_generated_token_hash": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
    }


def _attention_projection(*, stage_decode_overclaim: bool = False) -> dict:
    return {
        "schema": "glm52_attention_projection_probe_v1",
        "ok": True,
        "glm52_attention_projection_probe_ready": True,
        "public_artifact_safe": True,
        "model_id": pack.MODEL_ID,
        "model_repo": "cyankiwi/GLM-5.2-AWQ-INT4",
        "model_type": "glm_moe_dsa",
        "layer_id": 3,
        "hidden_size": 6144,
        "num_attention_heads": 64,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_head_dim": 256,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "input_norm_shape": [6144],
        "q_a_output_shape": [2048],
        "q_a_norm_shape": [2048],
        "q_b_output_shape": [16384],
        "query_shape": [64, 256],
        "q_nope_shape": [64, 192],
        "q_pe_shape": [64, 64],
        "kv_a_output_shape": [576],
        "k_compressed_shape": [512],
        "k_pe_shape": [64],
        "k_compressed_norm_shape": [512],
        "kv_b_output_shape": [28672],
        "k_nope_shape": [64, 192],
        "value_shape": [64, 256],
        "input_layernorm_verified": True,
        "q_lora_projection_verified": True,
        "kv_lora_projection_verified": True,
        "attention_projection_verified": True,
        "rope_applied": False,
        "attention_scores_verified": False,
        "o_proj_verified": False,
        "stage_decode_verified": stage_decode_overclaim,
        "input_norm_hash": "sha256:" + "a" * 64,
        "q_a_output_hash": "sha256:" + "b" * 64,
        "q_a_norm_hash": "sha256:" + "c" * 64,
        "q_b_output_hash": "sha256:" + "d" * 64,
        "q_nope_hash": "sha256:" + "e" * 64,
        "q_pe_hash": "sha256:" + "f" * 64,
        "kv_a_output_hash": "sha256:" + "1" * 64,
        "k_compressed_norm_hash": "sha256:" + "2" * 64,
        "kv_b_output_hash": "sha256:" + "3" * 64,
        "k_nope_hash": "sha256:" + "4" * 64,
        "value_hash": "sha256:" + "5" * 64,
        "blockers": [
            "glm52_attention_projection_is_not_rope_attention",
            "glm52_attention_projection_is_not_o_proj",
            "glm52_attention_projection_is_not_stage_decode",
            "glm52_attention_projection_missing_attention_scores",
            "glm52_attention_projection_missing_kv_cache_update",
            "glm52_stage_decode_not_verified",
        ],
        "completion_boundary": {
            "attention_projection_is_not_full_attention": True,
            "rope_not_applied": True,
            "attention_scores_not_computed": True,
            "o_proj_not_computed": True,
            "kv_cache_not_updated": True,
            "requires_stage_decode_verified": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
    }


def _attention_single_token(*, stage_decode_overclaim: bool = False) -> dict:
    return {
        "schema": "glm52_attention_single_token_probe_v1",
        "ok": True,
        "glm52_attention_single_token_probe_ready": True,
        "public_artifact_safe": True,
        "model_id": pack.MODEL_ID,
        "model_repo": "cyankiwi/GLM-5.2-AWQ-INT4",
        "model_type": "glm_moe_dsa",
        "layer_id": 3,
        "hidden_size": 6144,
        "num_attention_heads": 64,
        "qk_head_dim": 256,
        "v_head_dim": 256,
        "position_id": 7,
        "query_states_shape": [64, 256],
        "key_states_shape": [64, 256],
        "value_states_shape": [64, 256],
        "attention_scores_shape": [64, 1],
        "attention_weights_shape": [64, 1],
        "head_output_shape": [64, 256],
        "attention_flattened_shape": [16384],
        "o_proj_weight_shape": [6144, 16384],
        "o_proj_output_shape": [6144],
        "query_states_hash": "sha256:" + "a" * 64,
        "key_states_hash": "sha256:" + "b" * 64,
        "value_states_hash": "sha256:" + "c" * 64,
        "attention_scores_hash": "sha256:" + "d" * 64,
        "attention_weights_hash": "sha256:" + "e" * 64,
        "head_output_hash": "sha256:" + "f" * 64,
        "o_proj_output_hash": "sha256:" + "1" * 64,
        "rope_applied": True,
        "attention_scores_verified": True,
        "attention_weights_verified": True,
        "o_proj_verified": True,
        "single_token_attention_verified": True,
        "kv_cache_updated": False,
        "dsa_indexer_verified": False,
        "stage_decode_verified": stage_decode_overclaim,
        "blockers": [
            "glm52_attention_single_token_is_not_multi_token_prefill",
            "glm52_attention_single_token_is_not_dsa_indexer",
            "glm52_attention_single_token_is_not_kv_cache_decode",
            "glm52_attention_single_token_is_not_transformer_block",
            "glm52_attention_single_token_is_not_stage_decode",
            "glm52_stage_decode_not_verified",
        ],
        "completion_boundary": {
            "single_token_attention_is_not_multi_token_prefill": True,
            "single_token_attention_is_not_dsa_indexer": True,
            "single_token_attention_is_not_kv_cache_decode": True,
            "single_token_attention_is_not_transformer_block": True,
            "single_token_attention_is_not_stage_decode": True,
            "requires_stage_decode_verified": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
    }


def _kv_cache_decode(*, stage_decode_overclaim: bool = False, generated_token_overclaim: bool = False) -> dict:
    return {
        "schema": "glm52_kv_cache_decode_probe_v1",
        "ok": True,
        "glm52_kv_cache_decode_probe_ready": True,
        "public_artifact_safe": True,
        "model_id": pack.MODEL_ID,
        "model_repo": "cyankiwi/GLM-5.2-AWQ-INT4",
        "model_type": "glm_moe_dsa",
        "layer_id": 3,
        "hidden_size": 6144,
        "num_attention_heads": 64,
        "qk_head_dim": 256,
        "v_head_dim": 256,
        "prefill_length": 4,
        "decode_length": 1,
        "updated_cache_length": 5,
        "prefill_key_cache_shape": [4, 64, 256],
        "prefill_value_cache_shape": [4, 64, 256],
        "updated_key_cache_shape": [5, 64, 256],
        "updated_value_cache_shape": [5, 64, 256],
        "decode_query_shape": [64, 256],
        "attention_scores_shape": [64, 5],
        "attention_weights_shape": [64, 5],
        "head_output_shape": [64, 256],
        "attention_flattened_shape": [16384],
        "o_proj_weight_shape": [6144, 16384],
        "o_proj_output_shape": [6144],
        "prefill_key_cache_hash": "sha256:" + "a" * 64,
        "prefill_value_cache_hash": "sha256:" + "b" * 64,
        "updated_key_cache_hash": "sha256:" + "c" * 64,
        "updated_value_cache_hash": "sha256:" + "d" * 64,
        "decode_query_hash": "sha256:" + "e" * 64,
        "attention_scores_hash": "sha256:" + "f" * 64,
        "attention_weights_hash": "sha256:" + "1" * 64,
        "head_output_hash": "sha256:" + "2" * 64,
        "o_proj_output_hash": "sha256:" + "3" * 64,
        "kv_cache_prefill_verified": True,
        "kv_cache_update_verified": True,
        "kv_cache_decode_attention_verified": True,
        "o_proj_verified": True,
        "stage_decode_verified": stage_decode_overclaim,
        "generated_token_verified": generated_token_overclaim,
        "blockers": [
            "glm52_kv_cache_decode_is_not_dsa_masked_attention",
            "glm52_kv_cache_decode_is_not_transformer_block",
            "glm52_kv_cache_decode_is_not_stage_decode",
            "glm52_kv_cache_decode_missing_mlp_residual",
            "glm52_kv_cache_decode_missing_lm_head",
            "glm52_stage_decode_not_verified",
        ],
        "completion_boundary": {
            "kv_cache_decode_is_not_dsa_masked_attention": True,
            "kv_cache_decode_is_not_transformer_block": True,
            "kv_cache_decode_is_not_stage_decode": True,
            "requires_mlp_residual_runtime": True,
            "requires_lm_head_token_selection": True,
            "requires_stage_decode_verified": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
    }


def _layer_decode(*, stage_decode_overclaim: bool = False, same_request_overclaim: bool = False) -> dict:
    return {
        "schema": "glm52_layer_decode_probe_v1",
        "ok": True,
        "glm52_layer_decode_probe_ready": True,
        "public_artifact_safe": True,
        "model_id": pack.MODEL_ID,
        "model_repo": "cyankiwi/GLM-5.2-AWQ-INT4",
        "model_type": "glm_moe_dsa",
        "layer_id": 3,
        "hidden_size": 6144,
        "num_attention_heads": 64,
        "qk_head_dim": 256,
        "v_head_dim": 256,
        "num_experts_per_tok": 8,
        "prefill_length": 4,
        "updated_cache_length": 5,
        "attention_output_shape": [6144],
        "attention_residual_shape": [6144],
        "post_attention_norm_shape": [6144],
        "routed_output_shape": [6144],
        "shared_output_shape": [6144],
        "full_moe_output_shape": [6144],
        "layer_output_shape": [6144],
        "attention_output_hash": "sha256:" + "a" * 64,
        "attention_residual_hash": "sha256:" + "b" * 64,
        "post_attention_norm_hash": "sha256:" + "c" * 64,
        "full_moe_output_hash": "sha256:" + "d" * 64,
        "layer_output_hash": "sha256:" + "e" * 64,
        "executed_expert_count": 8,
        "kv_cache_prefill_verified": True,
        "kv_cache_update_verified": True,
        "attention_decode_verified": True,
        "attention_residual_verified": True,
        "post_attention_norm_verified": True,
        "router_topk_verified": True,
        "routed_expert_gather_verified": True,
        "shared_experts_mlp_verified": True,
        "full_moe_mlp_verified": True,
        "layer_decode_verified": True,
        "dsa_masked_attention_integrated": False,
        "multi_layer_stage_runtime_verified": False,
        "lm_head_verified": False,
        "generated_token_verified": False,
        "stage_decode_verified": stage_decode_overclaim,
        "same_request_decode_verified": same_request_overclaim,
        "blockers": [
            "glm52_layer_decode_is_single_layer_only",
            "glm52_layer_decode_uses_basic_attention_not_dsa_masked_attention",
            "glm52_layer_decode_missing_lm_head",
            "glm52_layer_decode_is_not_stage_decode",
            "glm52_layer_decode_is_not_same_request",
            "glm52_stage_decode_not_verified",
            "glm52_same_request_decode_not_verified",
        ],
        "completion_boundary": {
            "layer_decode_is_single_layer_only": True,
            "layer_decode_uses_basic_attention_not_dsa_masked_attention": True,
            "layer_decode_is_not_stage_decode": True,
            "layer_decode_is_not_same_request": True,
            "requires_multi_layer_stage_runtime": True,
            "requires_dsa_masked_attention_integration": True,
            "requires_lm_head_token_selection": True,
            "requires_kaggle_cpu_gpu_tpu_same_request": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
    }


def _lm_head_token(
    *,
    full_model_hidden_overclaim: bool = False,
    generated_token_overclaim: bool = False,
    stage_decode_overclaim: bool = False,
    same_request_overclaim: bool = False,
) -> dict:
    hidden = 6144
    vocab = 154880
    return {
        "schema": "glm52_lm_head_token_probe_v1",
        "ok": True,
        "glm52_lm_head_token_probe_ready": True,
        "public_artifact_safe": True,
        "model_id": pack.MODEL_ID,
        "model_repo": "cyankiwi/GLM-5.2-AWQ-INT4",
        "model_type": "glm_moe_dsa",
        "hidden_size": hidden,
        "vocab_size": vocab,
        "tie_word_embeddings": False,
        "norm_weight_shape": [hidden],
        "hidden_source": "deterministic_probe_vector",
        "hidden_shape": [hidden],
        "normalized_hidden_shape": [hidden],
        "hidden_hash": "sha256:" + "a" * 64,
        "normalized_hidden_hash": "sha256:" + "b" * 64,
        "lm_head_shape": [vocab, hidden],
        "lm_head_dtype": "BF16",
        "lm_head_nbytes": vocab * hidden * 2,
        "lm_head_file_count": 1,
        "lm_head_rows_scanned": vocab,
        "lm_head_block_count": 76,
        "lm_head_row_block_size": 2048,
        "top_k": 5,
        "top_k_count": 5,
        "selected_token_id_hash": "sha256:" + "c" * 64,
        "selected_logit_hash": "sha256:" + "d" * 64,
        "top_token_ids_hash": "sha256:" + "e" * 64,
        "top_logits_hash": "sha256:" + "f" * 64,
        "final_norm_verified": True,
        "lm_head_streamed_full_vocab": True,
        "lm_head_logits_token_selection_verified": True,
        "selected_token_hash_verified": True,
        "full_model_hidden_verified": full_model_hidden_overclaim,
        "generated_token_verified": generated_token_overclaim,
        "stage_decode_verified": stage_decode_overclaim,
        "same_request_decode_verified": same_request_overclaim,
        "blockers": [
            "glm52_lm_head_token_selection_uses_probe_hidden_not_full_model_hidden",
            "glm52_lm_head_token_selection_is_not_stage_decode",
            "glm52_lm_head_token_selection_is_not_same_request",
            "glm52_stage_decode_not_verified",
            "glm52_same_request_decode_not_verified",
        ],
        "completion_boundary": {
            "lm_head_token_selection_uses_probe_hidden_not_full_model_hidden": True,
            "lm_head_token_selection_is_not_stage_decode": True,
            "lm_head_token_selection_is_not_same_request": True,
            "requires_full_model_or_stage_hidden": True,
            "requires_stage_decode_verified": True,
            "requires_kaggle_cpu_gpu_tpu_same_request": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
    }


def _dsa_masked_layer_decode(*, stage_decode_overclaim: bool = False, same_request_overclaim: bool = False) -> dict:
    hidden = 6144
    updated = 9
    return {
        "schema": "glm52_dsa_masked_layer_decode_probe_v1",
        "ok": True,
        "glm52_dsa_masked_layer_decode_probe_ready": True,
        "public_artifact_safe": True,
        "model_id": pack.MODEL_ID,
        "model_repo": "cyankiwi/GLM-5.2-AWQ-INT4",
        "model_type": "glm_moe_dsa",
        "layer_id": 6,
        "hidden_size": hidden,
        "num_attention_heads": 64,
        "qk_head_dim": 256,
        "v_head_dim": 256,
        "num_experts_per_tok": 8,
        "prefill_length": 8,
        "updated_cache_length": updated,
        "dsa_indexer_type": "full",
        "dsa_index_n_heads": 32,
        "dsa_index_head_dim": 128,
        "dsa_index_topk_config": 2048,
        "dsa_mask_topk_count": 4,
        "dsa_mask_pruned_position_count": 5,
        "dsa_index_score_shape": [updated, updated],
        "dsa_attention_mask_shape": [updated],
        "attention_scores_shape": [64, updated],
        "attention_output_shape": [hidden],
        "attention_residual_shape": [hidden],
        "post_attention_norm_shape": [hidden],
        "full_moe_output_shape": [hidden],
        "layer_output_shape": [hidden],
        "dsa_index_score_hash": "sha256:" + "a" * 64,
        "dsa_topk_indices_hash": "sha256:" + "b" * 64,
        "dsa_attention_mask_hash": "sha256:" + "c" * 64,
        "attention_scores_hash": "sha256:" + "d" * 64,
        "attention_output_hash": "sha256:" + "e" * 64,
        "layer_output_hash": "sha256:" + "f" * 64,
        "executed_expert_count": 8,
        "dsa_indexer_verified": True,
        "dsa_mask_verified": True,
        "dsa_mask_pruned_positions_verified": True,
        "kv_cache_prefill_verified": True,
        "kv_cache_update_verified": True,
        "attention_decode_verified": True,
        "dsa_masked_attention_integrated": True,
        "attention_residual_verified": True,
        "post_attention_norm_verified": True,
        "full_moe_mlp_verified": True,
        "layer_decode_verified": True,
        "full_dsa_topk_scale_verified": False,
        "lm_head_verified": False,
        "generated_token_verified": False,
        "stage_decode_verified": stage_decode_overclaim,
        "same_request_decode_verified": same_request_overclaim,
        "blockers": [
            "glm52_dsa_masked_layer_decode_is_single_layer_only",
            "glm52_dsa_masked_layer_decode_uses_small_sequence_topk_cap",
            "glm52_dsa_masked_layer_decode_missing_lm_head",
            "glm52_dsa_masked_layer_decode_is_not_stage_decode",
            "glm52_dsa_masked_layer_decode_is_not_same_request",
            "glm52_stage_decode_not_verified",
            "glm52_same_request_decode_not_verified",
        ],
        "completion_boundary": {
            "dsa_masked_layer_decode_is_single_layer_only": True,
            "dsa_masked_layer_decode_uses_small_sequence_topk_cap": True,
            "dsa_masked_layer_decode_is_not_stage_decode": True,
            "dsa_masked_layer_decode_is_not_same_request": True,
            "requires_full_dsa_topk_scale_or_real_sequence": True,
            "requires_multi_layer_stage_runtime": True,
            "requires_lm_head_token_selection_from_stage_hidden": True,
            "requires_kaggle_cpu_gpu_tpu_same_request": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
    }


def _stage_hidden_lm_head(*, generated_token_overclaim: bool = False, stage_decode_overclaim: bool = False) -> dict:
    hidden = 6144
    vocab = 154880
    return {
        "schema": "glm52_stage_hidden_lm_head_probe_v1",
        "ok": True,
        "glm52_stage_hidden_lm_head_probe_ready": True,
        "public_artifact_safe": True,
        "model_id": pack.MODEL_ID,
        "model_repo": "cyankiwi/GLM-5.2-AWQ-INT4",
        "model_type": "glm_moe_dsa",
        "hidden_size": hidden,
        "vocab_size": vocab,
        "stage_hidden_source": "dsa_masked_single_layer_output",
        "stage_layer_id": 6,
        "stage_prefill_length": 8,
        "stage_updated_cache_length": 9,
        "stage_dsa_indexer_type": "full",
        "stage_dsa_mask_topk_count": 4,
        "stage_dsa_mask_pruned_position_count": 5,
        "stage_dsa_masked_attention_integrated": True,
        "stage_layer_decode_verified": True,
        "stage_hidden_shape": [hidden],
        "stage_hidden_hash": "sha256:" + "a" * 64,
        "normalized_stage_hidden_shape": [hidden],
        "normalized_stage_hidden_hash": "sha256:" + "b" * 64,
        "lm_head_shape": [vocab, hidden],
        "lm_head_dtype": "BF16",
        "lm_head_rows_scanned": vocab,
        "lm_head_block_count": 76,
        "top_k": 5,
        "top_k_count": 5,
        "selected_token_id_hash": "sha256:" + "c" * 64,
        "selected_logit_hash": "sha256:" + "d" * 64,
        "top_token_ids_hash": "sha256:" + "e" * 64,
        "top_logits_hash": "sha256:" + "f" * 64,
        "stage_hidden_to_lm_head_verified": True,
        "lm_head_streamed_full_vocab": True,
        "stage_hidden_lm_head_token_selection_verified": True,
        "partial_layer_token_hash_verified": True,
        "full_model_hidden_verified": False,
        "generated_token_verified": generated_token_overclaim,
        "stage_decode_verified": stage_decode_overclaim,
        "same_request_decode_verified": False,
        "blockers": [
            "glm52_stage_hidden_lm_head_is_single_layer_only",
            "glm52_stage_hidden_lm_head_uses_small_sequence_topk_cap",
            "glm52_stage_hidden_lm_head_is_not_full_model_hidden",
            "glm52_stage_hidden_lm_head_is_not_stage_decode",
            "glm52_stage_hidden_lm_head_is_not_same_request",
            "glm52_stage_decode_not_verified",
            "glm52_same_request_decode_not_verified",
        ],
        "completion_boundary": {
            "stage_hidden_lm_head_is_single_layer_only": True,
            "stage_hidden_lm_head_uses_small_sequence_topk_cap": True,
            "stage_hidden_lm_head_is_not_full_model_hidden": True,
            "stage_hidden_lm_head_is_not_stage_decode": True,
            "stage_hidden_lm_head_is_not_same_request": True,
            "requires_multi_layer_stage_runtime": True,
            "requires_full_model_or_stage_hidden": True,
            "requires_kaggle_cpu_gpu_tpu_same_request": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
    }


def _multi_layer_stage_decode(
    *,
    generated_token_overclaim: bool = False,
    stage_decode_overclaim: bool = False,
    same_request_overclaim: bool = False,
    live_kaggle_overclaim: bool = False,
) -> dict:
    hidden = 6144
    vocab = 154880
    return {
        "schema": "glm52_multi_layer_stage_decode_probe_v1",
        "ok": True,
        "glm52_multi_layer_stage_decode_probe_ready": True,
        "public_artifact_safe": True,
        "model_id": pack.MODEL_ID,
        "model_repo": "cyankiwi/GLM-5.2-AWQ-INT4",
        "model_type": "glm_moe_dsa",
        "hidden_size": hidden,
        "vocab_size": vocab,
        "stage_hidden_source": "dsa_masked_multi_layer_decode_token_chain",
        "stage_layer_range": [6, 8],
        "stage_layer_count": 2,
        "executed_layer_count": 2,
        "stage_prefill_length": 8,
        "stage_updated_cache_length": 9,
        "decode_token_chain_only": True,
        "prefill_hidden_carrier_full_layer_outputs_verified": False,
        "initial_decode_hidden_hash": "sha256:" + "0" * 64,
        "all_layers_dsa_masked_attention_integrated": True,
        "all_layers_moe_mlp_verified": True,
        "all_layer_outputs_chained": True,
        "layer_summaries": [
            {"layer_id": 6, "layer_decode_token_verified": True},
            {"layer_id": 7, "layer_decode_token_verified": True},
        ],
        "stage_hidden_shape": [hidden],
        "stage_hidden_hash": "sha256:" + "a" * 64,
        "normalized_stage_hidden_shape": [hidden],
        "normalized_stage_hidden_hash": "sha256:" + "b" * 64,
        "lm_head_shape": [vocab, hidden],
        "lm_head_dtype": "BF16",
        "lm_head_rows_scanned": vocab,
        "lm_head_block_count": 76,
        "top_k": 5,
        "top_k_count": 5,
        "selected_token_id_hash": "sha256:" + "c" * 64,
        "selected_logit_hash": "sha256:" + "d" * 64,
        "top_token_ids_hash": "sha256:" + "e" * 64,
        "top_logits_hash": "sha256:" + "f" * 64,
        "multi_layer_stage_hidden_verified": True,
        "multi_layer_decode_token_chain_verified": True,
        "stage_hidden_to_lm_head_verified": True,
        "lm_head_streamed_full_vocab": True,
        "stage_hidden_lm_head_token_selection_verified": True,
        "partial_multi_layer_token_hash_verified": True,
        "full_prefill_stage_hidden_verified": False,
        "full_model_hidden_verified": False,
        "generated_token_verified": generated_token_overclaim,
        "stage_decode_verified": stage_decode_overclaim,
        "same_request_decode_verified": same_request_overclaim,
        "live_kaggle_runtime_verified": live_kaggle_overclaim,
        "blockers": [
            "glm52_multi_layer_stage_decode_uses_decode_token_chain_only",
            "glm52_multi_layer_stage_decode_prefill_carrier_not_full_layer_outputs",
            "glm52_multi_layer_stage_decode_is_not_full_model_hidden",
            "glm52_multi_layer_stage_decode_is_not_kaggle_runtime",
            "glm52_multi_layer_stage_decode_is_not_same_request",
            "glm52_stage_decode_not_verified",
            "glm52_same_request_decode_not_verified",
        ],
        "completion_boundary": {
            "multi_layer_stage_decode_uses_decode_token_chain_only": True,
            "multi_layer_stage_decode_prefill_carrier_not_full_layer_outputs": True,
            "multi_layer_stage_decode_is_not_full_model_hidden": True,
            "multi_layer_stage_decode_is_not_kaggle_runtime": True,
            "multi_layer_stage_decode_is_not_same_request": True,
            "requires_full_prefill_layer_outputs": True,
            "requires_kaggle_stage_runtime": True,
            "requires_kaggle_cpu_gpu_tpu_same_request": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
    }


def _full_prefix_stage_decode(
    *,
    generated_token_overclaim: bool = False,
    stage_decode_overclaim: bool = False,
    same_request_overclaim: bool = False,
    live_kaggle_overclaim: bool = False,
) -> dict:
    hidden = 6144
    vocab = 154880
    seq_len = 3
    return {
        "schema": "glm52_full_prefix_stage_decode_probe_v1",
        "ok": True,
        "glm52_full_prefix_stage_decode_probe_ready": True,
        "public_artifact_safe": True,
        "model_id": pack.MODEL_ID,
        "model_repo": "cyankiwi/GLM-5.2-AWQ-INT4",
        "model_type": "glm_moe_dsa",
        "hidden_size": hidden,
        "vocab_size": vocab,
        "stage_hidden_source": "dsa_masked_full_prefix_multi_layer_stage_hidden",
        "stage_layer_range": [6, 8],
        "stage_layer_count": 2,
        "executed_layer_count": 2,
        "stage_prefill_length": 2,
        "stage_sequence_length": seq_len,
        "full_prefix_token_carrier_verified": True,
        "small_sequence_probe": True,
        "all_layers_full_prefix_verified": True,
        "all_layer_outputs_chained": True,
        "layer_summaries": [
            {"layer_id": 6, "layer_full_prefix_verified": True},
            {"layer_id": 7, "layer_full_prefix_verified": True},
        ],
        "stage_hidden_sequence_shape": [seq_len, hidden],
        "stage_hidden_sequence_hash": "sha256:" + "0" * 64,
        "stage_hidden_shape": [hidden],
        "stage_hidden_hash": "sha256:" + "a" * 64,
        "normalized_stage_hidden_shape": [hidden],
        "normalized_stage_hidden_hash": "sha256:" + "b" * 64,
        "lm_head_shape": [vocab, hidden],
        "lm_head_dtype": "BF16",
        "lm_head_rows_scanned": vocab,
        "lm_head_block_count": 76,
        "top_k": 5,
        "top_k_count": 5,
        "selected_token_id_hash": "sha256:" + "c" * 64,
        "selected_logit_hash": "sha256:" + "d" * 64,
        "top_token_ids_hash": "sha256:" + "e" * 64,
        "top_logits_hash": "sha256:" + "f" * 64,
        "full_prefix_stage_hidden_verified": True,
        "multi_layer_stage_hidden_verified": True,
        "stage_hidden_to_lm_head_verified": True,
        "lm_head_streamed_full_vocab": True,
        "stage_hidden_lm_head_token_selection_verified": True,
        "partial_full_prefix_token_hash_verified": True,
        "full_model_hidden_verified": False,
        "generated_token_verified": generated_token_overclaim,
        "stage_decode_verified": stage_decode_overclaim,
        "same_request_decode_verified": same_request_overclaim,
        "live_kaggle_runtime_verified": live_kaggle_overclaim,
        "blockers": [
            "glm52_full_prefix_stage_decode_uses_small_sequence_probe",
            "glm52_full_prefix_stage_decode_is_not_kaggle_runtime",
            "glm52_full_prefix_stage_decode_is_not_same_request",
            "glm52_stage_decode_not_verified",
            "glm52_same_request_decode_not_verified",
        ],
        "completion_boundary": {
            "full_prefix_stage_decode_uses_small_sequence_probe": True,
            "full_prefix_stage_decode_is_not_kaggle_runtime": True,
            "full_prefix_stage_decode_is_not_same_request": True,
            "requires_kaggle_stage_runtime": True,
            "requires_full_model_or_stage_partition": True,
            "requires_kaggle_cpu_gpu_tpu_same_request": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
    }


def _dsa_indexer(*, stage_decode_overclaim: bool = False) -> dict:
    return {
        "schema": "glm52_dsa_indexer_probe_v1",
        "ok": True,
        "glm52_dsa_indexer_probe_ready": True,
        "public_artifact_safe": True,
        "model_id": pack.MODEL_ID,
        "model_repo": "cyankiwi/GLM-5.2-AWQ-INT4",
        "model_type": "glm_moe_dsa",
        "layer_id": 2,
        "layer_indexer_type": "full",
        "hidden_size": 6144,
        "sequence_length": 8,
        "q_lora_rank": 2048,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "qk_rope_head_dim": 64,
        "index_topk_config": 2048,
        "effective_topk": 8,
        "hidden_norm_shape": [8, 6144],
        "q_resid_shape": [8, 2048],
        "indexer_query_shape": [8, 32, 128],
        "indexer_key_shape": [8, 128],
        "head_weights_shape": [8, 32],
        "index_score_shape": [8, 8],
        "topk_indices_shape": [8, 8],
        "hidden_norm_hash": "sha256:" + "a" * 64,
        "q_resid_hash": "sha256:" + "b" * 64,
        "indexer_query_hash": "sha256:" + "c" * 64,
        "indexer_key_hash": "sha256:" + "d" * 64,
        "head_weights_hash": "sha256:" + "e" * 64,
        "index_score_hash": "sha256:" + "f" * 64,
        "topk_indices_hash": "sha256:" + "1" * 64,
        "dsa_indexer_verified": True,
        "dsa_topk_verified": True,
        "indexer_cache_updated": False,
        "attention_output_verified": False,
        "stage_decode_verified": stage_decode_overclaim,
        "blockers": [
            "glm52_dsa_indexer_small_sequence_is_not_full_prefill",
            "glm52_dsa_indexer_is_not_kv_cache_decode",
            "glm52_dsa_indexer_is_not_attention_output",
            "glm52_dsa_indexer_is_not_stage_decode",
            "glm52_stage_decode_not_verified",
        ],
        "completion_boundary": {
            "dsa_indexer_small_sequence_is_not_full_prefill": True,
            "dsa_indexer_is_not_kv_cache_decode": True,
            "dsa_indexer_is_not_attention_output": True,
            "dsa_indexer_is_not_transformer_block": True,
            "dsa_indexer_is_not_stage_decode": True,
            "requires_stage_decode_verified": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
    }


def _pack_quantized_dequant(*, stage_decode_overclaim: bool = False) -> dict:
    return {
        "schema": "glm52_pack_quantized_dequant_probe_v1",
        "ok": True,
        "glm52_pack_quantized_dequant_probe_ready": True,
        "public_artifact_safe": True,
        "model_id": pack.MODEL_ID,
        "model_repo": "cyankiwi/GLM-5.2-AWQ-INT4",
        "model_type": "glm_moe_dsa",
        "quantization_format": "pack-quantized",
        "layer_id": 3,
        "expert_id": 0,
        "projection": "gate_proj",
        "row_count": 4,
        "group_count": 2,
        "pack_quantized_group_loaded": True,
        "pack_quantized_dequant_verified": True,
        "pack_quantized_linear_slice_verified": True,
        "stage_decode_verified": stage_decode_overclaim,
        "q_unpacked_hash": "sha256:" + "a" * 64,
        "zero_point_unpacked_hash": "sha256:" + "b" * 64,
        "dequant_slice_shape": [4, 64],
        "dequant_slice_hash": "sha256:" + "c" * 64,
        "linear_slice_shape": [4],
        "linear_slice_hash": "sha256:" + "d" * 64,
        "blockers": [
            "glm52_pack_quantized_dequant_slice_is_not_full_layer",
            "glm52_pack_quantized_linear_slice_is_not_stage_decode",
            "glm52_stage_decode_not_verified",
        ],
        "completion_boundary": {
            "dequant_slice_is_not_full_layer": True,
            "linear_slice_is_not_stage_decode": True,
            "weight_values_not_public": True,
            "requires_full_projection_runtime": True,
            "requires_transformer_block_runtime": True,
            "requires_stage_decode_verified": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
    }


def _pack_quantized_expert_mlp(*, stage_decode_overclaim: bool = False) -> dict:
    return {
        "schema": "glm52_pack_quantized_expert_mlp_probe_v1",
        "ok": True,
        "glm52_pack_quantized_expert_mlp_probe_ready": True,
        "public_artifact_safe": True,
        "model_id": pack.MODEL_ID,
        "model_repo": "cyankiwi/GLM-5.2-AWQ-INT4",
        "model_type": "glm_moe_dsa",
        "layer_id": 3,
        "expert_id": 0,
        "hidden_size": 6144,
        "pack_quantized_expert_mlp_verified": True,
        "single_expert_mlp_verified": True,
        "stage_decode_verified": stage_decode_overclaim,
        "projection_summaries": [
            {
                "projection": "gate_proj",
                "weight_shape": [2048, 6144],
                "output_shape": [2048],
                "output_hash": "sha256:" + "a" * 64,
                "pack_quantized_group_loaded": True,
            },
            {
                "projection": "up_proj",
                "weight_shape": [2048, 6144],
                "output_shape": [2048],
                "output_hash": "sha256:" + "b" * 64,
                "pack_quantized_group_loaded": True,
            },
            {
                "projection": "down_proj",
                "weight_shape": [6144, 2048],
                "output_shape": [6144],
                "output_hash": "sha256:" + "c" * 64,
                "pack_quantized_group_loaded": True,
            },
        ],
        "final_output_shape": [6144],
        "final_output_hash": "sha256:" + "d" * 64,
        "blockers": [
            "glm52_pack_quantized_expert_mlp_is_single_expert_only",
            "glm52_pack_quantized_expert_mlp_is_not_attention",
            "glm52_pack_quantized_expert_mlp_is_not_topk_router",
            "glm52_pack_quantized_expert_mlp_is_not_stage_decode",
            "glm52_stage_decode_not_verified",
        ],
        "completion_boundary": {
            "single_expert_mlp_is_not_full_moe_layer": True,
            "single_expert_mlp_is_not_attention": True,
            "single_expert_mlp_is_not_topk_router": True,
            "single_expert_mlp_is_not_stage_decode": True,
            "requires_transformer_block_runtime": True,
            "requires_stage_decode_verified": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
    }


def _pack_quantized_router_gather(*, stage_decode_overclaim: bool = False) -> dict:
    return {
        "schema": "glm52_pack_quantized_router_gather_probe_v1",
        "ok": True,
        "glm52_pack_quantized_router_gather_probe_ready": True,
        "public_artifact_safe": True,
        "model_id": pack.MODEL_ID,
        "model_repo": "cyankiwi/GLM-5.2-AWQ-INT4",
        "model_type": "glm_moe_dsa",
        "layer_id": 3,
        "hidden_size": 6144,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "router_topk_count": 8,
        "router_topk_verified": True,
        "router_topk_indices_hash": "sha256:" + "a" * 64,
        "router_topk_weights_hash": "sha256:" + "b" * 64,
        "executed_expert_count": 2,
        "requested_executed_expert_count": 2,
        "executed_experts": [
            {
                "topk_position": 0,
                "expert_id": 26,
                "expert_weight_hash": "sha256:" + "c" * 64,
                "expert_output_shape": [6144],
                "expert_output_hash": "sha256:" + "d" * 64,
            },
            {
                "topk_position": 1,
                "expert_id": 174,
                "expert_weight_hash": "sha256:" + "e" * 64,
                "expert_output_shape": [6144],
                "expert_output_hash": "sha256:" + "f" * 64,
            },
        ],
        "routed_expert_subset_verified": True,
        "routed_subset_output_shape": [6144],
        "routed_subset_output_hash": "sha256:" + "1" * 64,
        "stage_decode_verified": stage_decode_overclaim,
        "blockers": [
            "glm52_pack_quantized_router_gather_is_subset_only",
            "glm52_pack_quantized_router_gather_missing_shared_experts",
            "glm52_pack_quantized_router_gather_is_not_attention",
            "glm52_pack_quantized_router_gather_is_not_stage_decode",
            "glm52_stage_decode_not_verified",
        ],
        "completion_boundary": {
            "routed_subset_is_not_full_moe_layer": True,
            "shared_experts_not_included": True,
            "attention_not_included": True,
            "stage_decode_not_included": True,
            "requires_transformer_block_runtime": True,
            "requires_stage_decode_verified": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
    }


def _pack_quantized_moe_mlp(*, stage_decode_overclaim: bool = False) -> dict:
    return {
        "schema": "glm52_pack_quantized_moe_mlp_probe_v1",
        "ok": True,
        "glm52_pack_quantized_moe_mlp_probe_ready": True,
        "public_artifact_safe": True,
        "model_id": pack.MODEL_ID,
        "model_repo": "cyankiwi/GLM-5.2-AWQ-INT4",
        "model_type": "glm_moe_dsa",
        "layer_id": 3,
        "hidden_size": 6144,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "moe_intermediate_size": 2048,
        "router_topk_count": 8,
        "router_topk_verified": True,
        "router_topk_indices_hash": "sha256:" + "a" * 64,
        "router_topk_weights_hash": "sha256:" + "b" * 64,
        "executed_expert_count": 8,
        "requested_executed_expert_count": 8,
        "executed_experts": [
            {
                "topk_position": index,
                "expert_id": expert_id,
                "expert_weight_hash": "sha256:" + str(index) * 64,
                "expert_output_shape": [6144],
                "expert_output_hash": "sha256:" + format(index + 1, "x") * 64,
            }
            for index, expert_id in enumerate([26, 174, 3, 206, 233, 41, 161, 166])
        ],
        "routed_output_shape": [6144],
        "routed_output_hash": "sha256:" + "1" * 64,
        "shared_projection_summaries": [
            {
                "projection": "gate_proj",
                "weight_dtype": "bfloat16",
                "weight_shape": [2048, 6144],
                "output_shape": [2048],
                "output_hash": "sha256:" + "2" * 64,
            },
            {
                "projection": "up_proj",
                "weight_dtype": "bfloat16",
                "weight_shape": [2048, 6144],
                "output_shape": [2048],
                "output_hash": "sha256:" + "3" * 64,
            },
            {
                "projection": "down_proj",
                "weight_dtype": "bfloat16",
                "weight_shape": [6144, 2048],
                "output_shape": [6144],
                "output_hash": "sha256:" + "4" * 64,
            },
        ],
        "shared_output_shape": [6144],
        "shared_output_hash": "sha256:" + "5" * 64,
        "full_moe_output_shape": [6144],
        "full_moe_output_hash": "sha256:" + "6" * 64,
        "routed_expert_gather_verified": True,
        "shared_experts_mlp_verified": True,
        "pack_quantized_moe_mlp_verified": True,
        "full_moe_mlp_verified": True,
        "stage_decode_verified": stage_decode_overclaim,
        "blockers": [
            "glm52_pack_quantized_moe_mlp_is_not_attention",
            "glm52_pack_quantized_moe_mlp_is_not_transformer_block",
            "glm52_pack_quantized_moe_mlp_is_not_stage_decode",
            "glm52_pack_quantized_moe_mlp_missing_kv_cache",
            "glm52_pack_quantized_moe_mlp_missing_lm_head",
            "glm52_stage_decode_not_verified",
        ],
        "completion_boundary": {
            "full_moe_mlp_is_not_attention": True,
            "full_moe_mlp_is_not_transformer_block": True,
            "full_moe_mlp_is_not_stage_decode": True,
            "requires_attention_runtime": True,
            "requires_residual_norm_runtime": True,
            "requires_stage_local_kv_cache": True,
            "requires_lm_head_token_selection": True,
            "requires_stage_decode_verified": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
    }


def test_checker_accepts_queued_blocker_rc_without_marking_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _tpu_report())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["success"]["same_request_decode_verified"] is False
    assert "kaggle_tpu_runtime_not_ready" in report["blockers"]
    assert check.validate_report(report) == []


def test_rc_imports_awq_stage_header_without_claiming_runtime_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _tpu_report())
    awq = _write(base / "awq.json", _awq_stage_header())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--awq-stage-header-report",
            str(awq),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["awq_stage_header"]["stage_header_ready"] is True
    assert report["awq_stage_header"]["present_stage_key_count"] == 21675
    assert "glm52_awq_stage_runtime_adapter_not_verified" in report["blockers"]
    assert check.validate_report(report) == []


def test_rc_imports_awq_stage_value_probe_without_claiming_runtime_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _tpu_report(ready=True, queued=False))
    value = _write(base / "value.json", _awq_stage_value_probe())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--awq-stage-value-report",
            str(value),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["awq_stage_value_probe"]["stage_value_probe_ready"] is True
    assert report["awq_stage_value_probe"]["weight_tensor_values_loaded"] is True
    assert report["awq_stage_value_probe"]["stage_value_probe_ready_count"] == 1
    assert "glm52_awq_stage_value_probe_is_not_runtime_success" in report["blockers"]
    assert check.validate_report(report) == []


def test_checker_rejects_awq_stage_value_same_request_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _tpu_report(ready=True, queued=False))
    value = _write(base / "value.json", _awq_stage_value_probe(same_request_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--awq-stage-value-report",
            str(value),
        ])
    )

    assert "awq_stage_value_same_request_decode_overclaim:stage4" in check.validate_report(report)


def test_rc_summarizes_provider_aligned_stage_value_probe_coverage() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _tpu_report(ready=True, queued=False))
    values = [
        _write(base / "value0.json", _awq_stage_value_probe_for_stage(0, [0, 26])),
        _write(base / "value1.json", _awq_stage_value_probe_for_stage(1, [26, 52])),
        _write(base / "value2.json", _awq_stage_value_probe_for_stage(2, [52, 78])),
    ]
    argv = [
        "--output-dir",
        str(base / "rc"),
        "--source-report",
        str(source),
        "--tpu-watch-report",
        str(tpu),
    ]
    for value in values:
        argv.extend(["--awq-stage-value-report", str(value)])

    report = pack.build_report(pack.parse_args(argv))

    assert report["goal_achieved"] is False
    assert report["awq_stage_value_probe"]["provider_aligned_stage_value_probe_ready"] is True
    assert set(report["awq_stage_value_probe"]["provider_coverage"]) == set(pack.REQUIRED_PROVIDERS)
    assert report["awq_stage_value_probe"]["stage_value_probe_ready_count"] == 3
    assert "glm52_awq_stage_value_probe_provider_coverage_incomplete" not in report["blockers"]
    assert check.validate_report(report) == []


def test_rc_imports_tpu_stage_smoke_queue_watch_as_blocker() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _tpu_report(ready=True, queued=False))
    smoke_watch = _write(base / "smoke-watch.json", _tpu_stage_smoke_watch())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--tpu-stage-smoke-report",
            str(smoke_watch),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["tpu_stage_smoke"]["queued_watch"] is True
    assert "glm52_awq_tpu_stage_smoke_scheduler_queued" in report["blockers"]
    assert report["failure_stage"] == "glm52_awq_tpu_stage_smoke_not_ready"
    assert check.validate_report(report) == []


def test_rc_imports_kaggle_source_search_not_found_blocker() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _tpu_report(ready=True, queued=False))
    kaggle_source = _write(base / "kaggle-source.json", _kaggle_source_search())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--kaggle-source-search-report",
            str(kaggle_source),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["kaggle_source_search"]["present"] is True
    assert report["kaggle_source_search"]["kaggle_attach_source_verified"] is False
    assert "glm52_kaggle_attach_source_not_found" in report["blockers"]
    assert check.validate_report(report) == []


def test_rc_imports_stage_runtime_plan_as_contract_not_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _tpu_report(ready=True, queued=False))
    stage_plan = _write(base / "stage-plan.json", _stage_runtime_plan())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--stage-runtime-plan-report",
            str(stage_plan),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["stage_runtime_plan"]["plan_ready"] is True
    assert report["stage_runtime_plan"]["stage_runtime_adapter_verified"] is False
    assert "glm52_stage_runtime_live_reports_missing" in report["blockers"]
    assert check.validate_report(report) == []


def test_rc_imports_stage_worker_package_as_not_live_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _tpu_report(ready=True, queued=False))
    package_report = _write(base / "package.json", _stage_worker_package())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--stage-worker-package-report",
            str(package_report),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["stage_worker_package"]["package_ready"] is True
    assert report["stage_worker_package"]["live_run_performed"] is False
    assert "glm52_stage_worker_package_is_not_runtime_success" in report["blockers"]
    assert check.validate_report(report) == []


def test_rc_imports_stage_worker_push_preflight_as_not_live_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _tpu_report(ready=True, queued=False))
    push_probe = _write(base / "push.json", _stage_worker_push_probe())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--stage-worker-push-probe-report",
            str(push_probe),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["stage_worker_push_probe"]["push_probe_ready"] is True
    assert report["stage_worker_push_probe"]["live_run_performed"] is False
    assert "glm52_stage_worker_push_not_started" in report["blockers"]
    assert check.validate_report(report) == []


def test_rc_advances_failure_stage_after_all_provider_stage_runtime_reports_verify() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    push = _stage_worker_push_probe()
    push["mode"] = "import"
    push["live_run_performed"] = True
    push["stage_runtime_reports_collected"] = len(pack.REQUIRED_PROVIDERS)
    push["stage_runtime_reports_verified"] = len(pack.REQUIRED_PROVIDERS)
    push["blockers"] = []
    for item in push["pushes"]:
        item["output_collected"] = True
        item["stage_report_present"] = True
        item["stage_runtime_verified"] = True
        item["stage_report_check"] = {
            "ok": True,
            "stage_runtime_verified": True,
            "provider": item["provider"],
            "stage_id": item["stage_id"],
        }
        item["cleanup_performed"] = True
        item["terminal_status"] = "IMPORTED"
    push_probe = _write(base / "push.json", push)

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--stage-worker-push-probe-report",
            str(push_probe),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["stage_worker_push_probe"]["required_stage_runtime_reports_verified"] is True
    assert report["stage_worker_push_probe"]["required_provider_stage_runtime_reports_verified"] is True
    assert report["stage_worker_push_probe"]["all_planned_stage_runtime_reports_verified"] is True
    assert report["stage_worker_push_probe"]["planned_stage_count"] == len(pack.REQUIRED_PROVIDERS)
    assert set(report["stage_worker_push_probe"]["verified_provider_coverage"]) == set(pack.REQUIRED_PROVIDERS)
    assert report["stage_worker_push_probe"]["verified_stage_count"] == len(pack.REQUIRED_PROVIDERS)
    assert report["failure_stage"] == "glm52_same_request_decode_not_verified"
    assert check.validate_report(report) == []


def test_stage_worker_push_summary_records_multiple_cpu_stage_coverage_without_provider_success() -> None:
    push = _stage_worker_push_probe()
    push["mode"] = "import"
    push["live_run_performed"] = True
    push["stage_runtime_reports_collected"] = 3
    push["stage_runtime_reports_verified"] = 3
    push["pushes"] = [
        {
            "provider": "kaggle_cpu",
            "stage_id": stage_id,
            "pushed": False,
            "terminal_status": "IMPORTED",
            "output_collected": True,
            "stage_report_present": True,
            "stage_runtime_verified": True,
            "stage_report_path": f"stage-{stage_id}.json",
            "stage_report_check": {
                "ok": True,
                "stage_runtime_verified": True,
                "provider": "kaggle_cpu",
                "stage_id": stage_id,
            },
            "cleanup_performed": True,
            "public_artifact_safe": True,
        }
        for stage_id in [2, 3, 4]
    ]

    summary = pack.summarize_stage_worker_push_probe(push)

    assert summary["verified_provider_coverage"] == ["kaggle_cpu"]
    assert summary["verified_stage_count"] == 3
    assert [item["stage_id"] for item in summary["verified_stage_coverage"]] == [2, 3, 4]
    assert summary["required_stage_runtime_reports_verified"] is False
    assert summary["required_provider_stage_runtime_reports_verified"] is False
    assert summary["all_planned_stage_runtime_reports_verified"] is True


def test_stage_worker_push_summary_keeps_missing_planned_stages_out_of_required_success() -> None:
    push = _stage_worker_push_probe()
    push["mode"] = "import"
    push["live_run_performed"] = True
    push["stage_runtime_reports_collected"] = 3
    push["stage_runtime_reports_verified"] = 3
    push["pushes"] = [
        {
            "provider": "kaggle_cuda",
            "stage_id": 0,
            "terminal_status": "IMPORTED",
            "stage_runtime_verified": True,
            "stage_report_path": "cuda.json",
            "cleanup_performed": True,
        },
        {
            "provider": "kaggle_jax_tpu",
            "stage_id": 13,
            "terminal_status": "IMPORTED",
            "stage_runtime_verified": True,
            "stage_report_path": "tpu.json",
            "cleanup_performed": True,
        },
        {
            "provider": "kaggle_cpu",
            "stage_id": 14,
            "terminal_status": "IMPORTED",
            "stage_runtime_verified": True,
            "stage_report_path": "cpu.json",
            "cleanup_performed": True,
        },
        {
            "provider": "kaggle_cpu",
            "stage_id": 15,
            "terminal_status": "MISSING",
            "stage_runtime_verified": False,
            "stage_report_path": "",
            "cleanup_performed": False,
        },
    ]

    summary = pack.summarize_stage_worker_push_probe(push)

    assert set(summary["verified_provider_coverage"]) == set(pack.REQUIRED_PROVIDERS)
    assert summary["required_provider_stage_runtime_reports_verified"] is True
    assert summary["all_planned_stage_runtime_reports_verified"] is False
    assert summary["required_stage_runtime_reports_verified"] is False
    assert summary["planned_stage_count"] == 4


def test_stage_worker_push_summary_uses_package_plan_when_import_is_stage_filtered() -> None:
    push = _stage_worker_push_probe()
    push["mode"] = "import"
    push["live_run_performed"] = True
    push["stage_runtime_reports_collected"] = 3
    push["stage_runtime_reports_verified"] = 3
    push["pushes"] = [
        {
            "provider": "kaggle_cuda",
            "stage_id": 0,
            "terminal_status": "IMPORTED",
            "stage_runtime_verified": True,
            "stage_report_path": "cuda.json",
            "cleanup_performed": True,
        },
        {
            "provider": "kaggle_jax_tpu",
            "stage_id": 13,
            "terminal_status": "IMPORTED",
            "stage_runtime_verified": True,
            "stage_report_path": "tpu.json",
            "cleanup_performed": True,
        },
        {
            "provider": "kaggle_cpu",
            "stage_id": 14,
            "terminal_status": "IMPORTED",
            "stage_runtime_verified": True,
            "stage_report_path": "cpu.json",
            "cleanup_performed": True,
        },
    ]
    package = {
        "provider_packages": [
            {"provider": "kaggle_cuda", "stage_id": 0},
            {"provider": "kaggle_cpu", "stage_id": 1},
            {"provider": "kaggle_jax_tpu", "stage_id": 13},
            {"provider": "kaggle_cpu", "stage_id": 14},
        ]
    }

    summary = pack.summarize_stage_worker_push_probe(push, stage_worker_package=package)

    assert set(summary["verified_provider_coverage"]) == set(pack.REQUIRED_PROVIDERS)
    assert summary["required_provider_stage_runtime_reports_verified"] is True
    assert summary["all_planned_stage_runtime_reports_verified"] is False
    assert summary["required_stage_runtime_reports_verified"] is False
    assert summary["planned_stage_count"] == 4
    assert summary["missing_planned_stage_count"] == 1
    assert summary["missing_planned_stage_coverage"] == [{"provider": "kaggle_cpu", "stage_id": 1}]


def test_rc_imports_mcp_tpu_stage_runtime_watch() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["tpu_request"]["source_schema"] == "glm52_mcp_tpu_stage_runtime_watch_v1"
    assert report["tpu_request"]["tpu_runtime_ready"] is True
    assert report["tpu_request"]["tpu_stage_runtime_ready"] is True
    assert report["tpu_request"]["stage_runtime_report_verified"] is True
    assert "kaggle_tpu_runtime_not_ready" not in report["blockers"]
    assert "glm52_same_request_decode_not_verified" in report["blockers"]
    assert check.validate_report(report) == []


def test_rc_imports_decode_adapter_gap_as_current_precise_blocker() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    gap = _write(base / "gap.json", _decode_adapter_gap())
    push = _stage_worker_push_probe()
    push["mode"] = "import"
    push["live_run_performed"] = True
    push["stage_runtime_reports_collected"] = len(pack.REQUIRED_PROVIDERS)
    push["stage_runtime_reports_verified"] = len(pack.REQUIRED_PROVIDERS)
    push["blockers"] = []
    for item in push["pushes"]:
        item["output_collected"] = True
        item["stage_report_present"] = True
        item["stage_runtime_verified"] = True
        item["stage_report_check"] = {
            "ok": True,
            "stage_runtime_verified": True,
            "provider": item["provider"],
            "stage_id": item["stage_id"],
        }
        item["cleanup_performed"] = True
        item["terminal_status"] = "IMPORTED"
    push_probe = _write(base / "push.json", push)

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--stage-worker-push-probe-report",
            str(push_probe),
            "--decode-adapter-gap-report",
            str(gap),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["decode_adapter_gap"]["present"] is True
    assert report["decode_adapter_gap"]["decode_adapter_ready"] is False
    assert "awq_int4_dequant_linear_runtime" in report["decode_adapter_gap"]["missing_capabilities"]
    assert "glm52_full_decode_adapter_not_ready" in report["blockers"]
    assert report["failure_stage"] == "glm52_full_decode_adapter_not_ready"
    assert check.validate_report(report) == []


def test_rc_imports_transformers_preflight_as_foundation_not_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    preflight_report = _write(base / "preflight.json", _transformers_decode_preflight())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--transformers-decode-preflight-report",
            str(preflight_report),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["transformers_decode_preflight"]["adapter_foundation_ready"] is True
    assert report["transformers_decode_preflight"]["decode_adapter_ready"] is False
    assert report["transformers_decode_preflight"]["missing_required_key_count"] == 0
    assert "glm52_full_decode_adapter_not_ready" in report["blockers"]
    assert check.validate_report(report) == []


def test_rc_imports_attention_projection_as_projection_not_decode_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    attention = _write(base / "attention.json", _attention_projection())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--attention-projection-report",
            str(attention),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["attention_projection"]["attention_projection_verified"] is True
    assert report["attention_projection"]["query_shape"] == [64, 256]
    assert report["attention_projection"]["value_shape"] == [64, 256]
    assert "glm52_attention_projection_is_not_stage_decode" in report["blockers"]
    assert "glm52_attention_projection_missing_attention_scores" in report["blockers"]
    assert check.validate_report(report) == []


def test_checker_rejects_attention_projection_stage_decode_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    attention = _write(base / "attention.json", _attention_projection(stage_decode_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--attention-projection-report",
            str(attention),
        ])
    )

    assert "attention_projection_stage_decode_overclaim" in check.validate_report(report)


def test_rc_imports_attention_single_token_as_attention_not_decode_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    attention = _write(base / "attention-single.json", _attention_single_token())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--attention-single-token-report",
            str(attention),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["attention_single_token"]["single_token_attention_verified"] is True
    assert report["attention_single_token"]["o_proj_verified"] is True
    assert report["attention_single_token"]["o_proj_output_shape"] == [6144]
    assert "glm52_attention_single_token_is_not_stage_decode" in report["blockers"]
    assert "glm52_attention_single_token_is_not_kv_cache_decode" in report["blockers"]
    assert check.validate_report(report) == []


def test_checker_rejects_attention_single_token_stage_decode_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    attention = _write(base / "attention-single.json", _attention_single_token(stage_decode_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--attention-single-token-report",
            str(attention),
        ])
    )

    assert "attention_single_token_stage_decode_overclaim" in check.validate_report(report)


def test_rc_imports_kv_cache_decode_as_cache_decode_not_stage_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    kv_decode = _write(base / "kv-decode.json", _kv_cache_decode())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--kv-cache-decode-report",
            str(kv_decode),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["kv_cache_decode"]["kv_cache_prefill_verified"] is True
    assert report["kv_cache_decode"]["kv_cache_update_verified"] is True
    assert report["kv_cache_decode"]["kv_cache_decode_attention_verified"] is True
    assert report["kv_cache_decode"]["o_proj_verified"] is True
    assert report["kv_cache_decode"]["updated_key_cache_shape"] == [5, 64, 256]
    assert "glm52_kv_cache_decode_is_not_stage_decode" in report["blockers"]
    assert "glm52_kv_cache_decode_missing_lm_head" in report["blockers"]
    assert check.validate_report(report) == []


def test_checker_rejects_kv_cache_decode_stage_decode_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    kv_decode = _write(base / "kv-decode.json", _kv_cache_decode(stage_decode_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--kv-cache-decode-report",
            str(kv_decode),
        ])
    )

    assert "kv_cache_decode_stage_decode_overclaim" in check.validate_report(report)


def test_rc_imports_layer_decode_as_single_layer_not_stage_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    layer_decode = _write(base / "layer-decode.json", _layer_decode())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--layer-decode-report",
            str(layer_decode),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["layer_decode"]["layer_decode_verified"] is True
    assert report["layer_decode"]["attention_residual_verified"] is True
    assert report["layer_decode"]["post_attention_norm_verified"] is True
    assert report["layer_decode"]["full_moe_mlp_verified"] is True
    assert report["layer_decode"]["layer_output_shape"] == [6144]
    assert "glm52_layer_decode_is_not_stage_decode" in report["blockers"]
    assert "glm52_layer_decode_missing_lm_head" in report["blockers"]
    assert check.validate_report(report) == []


def test_checker_rejects_layer_decode_stage_decode_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    layer_decode = _write(base / "layer-decode.json", _layer_decode(stage_decode_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--layer-decode-report",
            str(layer_decode),
        ])
    )

    assert "layer_decode_stage_decode_verified_overclaim" in check.validate_report(report)


def test_rc_imports_lm_head_token_selection_without_claiming_generated_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    lm_head = _write(base / "lm-head.json", _lm_head_token())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--lm-head-token-report",
            str(lm_head),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["lm_head_token"]["lm_head_logits_token_selection_verified"] is True
    assert report["lm_head_token"]["lm_head_streamed_full_vocab"] is True
    assert report["lm_head_token"]["lm_head_rows_scanned"] == 154880
    assert report["lm_head_token"]["selected_token_id_hash_present"] is True
    assert "glm52_lm_head_token_selection_is_not_stage_decode" in report["blockers"]
    assert "glm52_lm_head_token_selection_is_not_same_request" in report["blockers"]
    assert check.validate_report(report) == []


def test_checker_rejects_lm_head_token_stage_decode_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    lm_head = _write(base / "lm-head.json", _lm_head_token(stage_decode_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--lm-head-token-report",
            str(lm_head),
        ])
    )

    assert "lm_head_token_stage_decode_verified_overclaim" in check.validate_report(report)


def test_checker_rejects_lm_head_token_generated_token_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    lm_head = _write(base / "lm-head.json", _lm_head_token(generated_token_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--lm-head-token-report",
            str(lm_head),
        ])
    )

    assert "lm_head_token_generated_token_verified_overclaim" in check.validate_report(report)


def test_rc_imports_dsa_masked_layer_decode_without_claiming_stage_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    dsa_layer = _write(base / "dsa-layer.json", _dsa_masked_layer_decode())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--dsa-masked-layer-decode-report",
            str(dsa_layer),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["dsa_masked_layer_decode"]["dsa_masked_attention_integrated"] is True
    assert report["dsa_masked_layer_decode"]["dsa_mask_topk_count"] == 4
    assert report["dsa_masked_layer_decode"]["dsa_mask_pruned_position_count"] == 5
    assert report["dsa_masked_layer_decode"]["layer_decode_verified"] is True
    assert "glm52_dsa_masked_layer_decode_is_not_stage_decode" in report["blockers"]
    assert "glm52_dsa_masked_layer_decode_is_not_same_request" in report["blockers"]
    assert check.validate_report(report) == []


def test_checker_rejects_dsa_masked_layer_decode_stage_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    dsa_layer = _write(base / "dsa-layer.json", _dsa_masked_layer_decode(stage_decode_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--dsa-masked-layer-decode-report",
            str(dsa_layer),
        ])
    )

    assert "dsa_masked_layer_decode_stage_decode_verified_overclaim" in check.validate_report(report)


def test_checker_rejects_dsa_masked_layer_decode_same_request_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    dsa_layer = _write(base / "dsa-layer.json", _dsa_masked_layer_decode(same_request_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--dsa-masked-layer-decode-report",
            str(dsa_layer),
        ])
    )

    assert "dsa_masked_layer_decode_same_request_decode_verified_overclaim" in check.validate_report(report)


def test_rc_imports_stage_hidden_lm_head_without_claiming_generated_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    stage_lm = _write(base / "stage-lm.json", _stage_hidden_lm_head())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--stage-hidden-lm-head-report",
            str(stage_lm),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["stage_hidden_lm_head"]["stage_hidden_lm_head_token_selection_verified"] is True
    assert report["stage_hidden_lm_head"]["stage_hidden_to_lm_head_verified"] is True
    assert report["stage_hidden_lm_head"]["stage_dsa_masked_attention_integrated"] is True
    assert report["stage_hidden_lm_head"]["lm_head_rows_scanned"] == 154880
    assert "glm52_stage_hidden_lm_head_is_not_stage_decode" in report["blockers"]
    assert "glm52_stage_hidden_lm_head_is_not_same_request" in report["blockers"]
    assert check.validate_report(report) == []


def test_checker_rejects_stage_hidden_lm_head_generated_token_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    stage_lm = _write(base / "stage-lm.json", _stage_hidden_lm_head(generated_token_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--stage-hidden-lm-head-report",
            str(stage_lm),
        ])
    )

    assert "stage_hidden_lm_head_generated_token_verified_overclaim" in check.validate_report(report)


def test_checker_rejects_stage_hidden_lm_head_stage_decode_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    stage_lm = _write(base / "stage-lm.json", _stage_hidden_lm_head(stage_decode_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--stage-hidden-lm-head-report",
            str(stage_lm),
        ])
    )

    assert "stage_hidden_lm_head_stage_decode_verified_overclaim" in check.validate_report(report)


def test_rc_imports_multi_layer_stage_decode_without_claiming_runtime_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    multi_layer = _write(base / "multi-layer.json", _multi_layer_stage_decode())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--multi-layer-stage-decode-report",
            str(multi_layer),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["multi_layer_stage_decode"]["multi_layer_stage_hidden_verified"] is True
    assert report["multi_layer_stage_decode"]["stage_hidden_lm_head_token_selection_verified"] is True
    assert report["multi_layer_stage_decode"]["stage_layer_range"] == [6, 8]
    assert report["multi_layer_stage_decode"]["layer_summary_count"] == 2
    assert "glm52_multi_layer_stage_decode_is_not_kaggle_runtime" in report["blockers"]
    assert "glm52_multi_layer_stage_decode_is_not_same_request" in report["blockers"]
    assert check.validate_report(report) == []


def test_checker_rejects_multi_layer_stage_decode_generated_token_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    multi_layer = _write(base / "multi-layer.json", _multi_layer_stage_decode(generated_token_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--multi-layer-stage-decode-report",
            str(multi_layer),
        ])
    )

    assert "multi_layer_stage_decode_generated_token_verified_overclaim" in check.validate_report(report)


def test_checker_rejects_multi_layer_stage_decode_stage_decode_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    multi_layer = _write(base / "multi-layer.json", _multi_layer_stage_decode(stage_decode_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--multi-layer-stage-decode-report",
            str(multi_layer),
        ])
    )

    assert "multi_layer_stage_decode_stage_decode_verified_overclaim" in check.validate_report(report)


def test_checker_rejects_multi_layer_stage_decode_live_kaggle_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    multi_layer = _write(base / "multi-layer.json", _multi_layer_stage_decode(live_kaggle_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--multi-layer-stage-decode-report",
            str(multi_layer),
        ])
    )

    assert "multi_layer_stage_decode_live_kaggle_runtime_verified_overclaim" in check.validate_report(report)


def test_rc_imports_full_prefix_stage_decode_without_claiming_runtime_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    full_prefix = _write(base / "full-prefix.json", _full_prefix_stage_decode())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--full-prefix-stage-decode-report",
            str(full_prefix),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["full_prefix_stage_decode"]["full_prefix_stage_hidden_verified"] is True
    assert report["full_prefix_stage_decode"]["stage_hidden_lm_head_token_selection_verified"] is True
    assert report["full_prefix_stage_decode"]["stage_layer_range"] == [6, 8]
    assert report["full_prefix_stage_decode"]["stage_hidden_sequence_shape"] == [3, 6144]
    assert "glm52_full_prefix_stage_decode_is_not_kaggle_runtime" in report["blockers"]
    assert "glm52_full_prefix_stage_decode_is_not_same_request" in report["blockers"]
    assert check.validate_report(report) == []


def test_checker_rejects_full_prefix_stage_decode_generated_token_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    full_prefix = _write(base / "full-prefix.json", _full_prefix_stage_decode(generated_token_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--full-prefix-stage-decode-report",
            str(full_prefix),
        ])
    )

    assert "full_prefix_stage_decode_generated_token_verified_overclaim" in check.validate_report(report)


def test_checker_rejects_full_prefix_stage_decode_stage_decode_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    full_prefix = _write(base / "full-prefix.json", _full_prefix_stage_decode(stage_decode_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--full-prefix-stage-decode-report",
            str(full_prefix),
        ])
    )

    assert "full_prefix_stage_decode_stage_decode_verified_overclaim" in check.validate_report(report)


def test_checker_rejects_full_prefix_stage_decode_live_kaggle_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    full_prefix = _write(base / "full-prefix.json", _full_prefix_stage_decode(live_kaggle_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--full-prefix-stage-decode-report",
            str(full_prefix),
        ])
    )

    assert "full_prefix_stage_decode_live_kaggle_runtime_verified_overclaim" in check.validate_report(report)


def test_rc_imports_dsa_indexer_as_indexer_not_decode_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    dsa = _write(base / "dsa.json", _dsa_indexer())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--dsa-indexer-report",
            str(dsa),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["dsa_indexer"]["dsa_indexer_verified"] is True
    assert report["dsa_indexer"]["topk_indices_shape"] == [8, 8]
    assert report["dsa_indexer"]["indexer_cache_updated"] is False
    assert "glm52_dsa_indexer_is_not_stage_decode" in report["blockers"]
    assert "glm52_dsa_indexer_is_not_kv_cache_decode" in report["blockers"]
    assert check.validate_report(report) == []


def test_checker_rejects_dsa_indexer_stage_decode_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    dsa = _write(base / "dsa.json", _dsa_indexer(stage_decode_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--dsa-indexer-report",
            str(dsa),
        ])
    )

    assert "dsa_indexer_stage_decode_overclaim" in check.validate_report(report)


def test_rc_imports_pack_quantized_dequant_as_slice_not_decode_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    dequant = _write(base / "dequant.json", _pack_quantized_dequant())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--pack-quantized-dequant-report",
            str(dequant),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["pack_quantized_dequant"]["pack_quantized_dequant_verified"] is True
    assert report["pack_quantized_dequant"]["pack_quantized_linear_slice_verified"] is True
    assert report["pack_quantized_dequant"]["stage_decode_verified"] is False
    assert "glm52_pack_quantized_dequant_slice_is_not_full_layer" in report["blockers"]
    assert "glm52_pack_quantized_linear_slice_is_not_stage_decode" in report["blockers"]
    assert check.validate_report(report) == []


def test_checker_rejects_pack_quantized_dequant_stage_decode_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    dequant = _write(base / "dequant.json", _pack_quantized_dequant(stage_decode_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--pack-quantized-dequant-report",
            str(dequant),
        ])
    )

    assert "pack_quantized_dequant_stage_decode_overclaim" in check.validate_report(report)


def test_rc_imports_pack_quantized_expert_mlp_as_single_expert_not_decode_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    expert_mlp = _write(base / "expert-mlp.json", _pack_quantized_expert_mlp())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--pack-quantized-expert-mlp-report",
            str(expert_mlp),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["pack_quantized_expert_mlp"]["pack_quantized_expert_mlp_verified"] is True
    assert report["pack_quantized_expert_mlp"]["final_output_shape"] == [6144]
    assert report["pack_quantized_expert_mlp"]["stage_decode_verified"] is False
    assert "glm52_pack_quantized_expert_mlp_is_single_expert_only" in report["blockers"]
    assert "glm52_pack_quantized_expert_mlp_is_not_attention" in report["blockers"]
    assert check.validate_report(report) == []


def test_checker_rejects_pack_quantized_expert_mlp_stage_decode_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    expert_mlp = _write(base / "expert-mlp.json", _pack_quantized_expert_mlp(stage_decode_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--pack-quantized-expert-mlp-report",
            str(expert_mlp),
        ])
    )

    assert "pack_quantized_expert_mlp_stage_decode_overclaim" in check.validate_report(report)


def test_rc_imports_pack_quantized_router_gather_as_subset_not_decode_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    router = _write(base / "router-gather.json", _pack_quantized_router_gather())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--pack-quantized-router-gather-report",
            str(router),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["pack_quantized_router_gather"]["router_topk_verified"] is True
    assert report["pack_quantized_router_gather"]["routed_expert_subset_verified"] is True
    assert report["pack_quantized_router_gather"]["executed_expert_count"] == 2
    assert "glm52_pack_quantized_router_gather_is_subset_only" in report["blockers"]
    assert "glm52_pack_quantized_router_gather_missing_shared_experts" in report["blockers"]
    assert check.validate_report(report) == []


def test_checker_rejects_pack_quantized_router_gather_stage_decode_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    router = _write(base / "router-gather.json", _pack_quantized_router_gather(stage_decode_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--pack-quantized-router-gather-report",
            str(router),
        ])
    )

    assert "pack_quantized_router_gather_stage_decode_overclaim" in check.validate_report(report)


def test_rc_imports_pack_quantized_moe_mlp_as_mlp_not_decode_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    moe_mlp = _write(base / "moe-mlp.json", _pack_quantized_moe_mlp())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--pack-quantized-moe-mlp-report",
            str(moe_mlp),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["pack_quantized_moe_mlp"]["full_moe_mlp_verified"] is True
    assert report["pack_quantized_moe_mlp"]["executed_expert_count"] == 8
    assert report["pack_quantized_moe_mlp"]["shared_experts_mlp_verified"] is True
    assert report["pack_quantized_moe_mlp"]["full_moe_output_shape"] == [6144]
    assert "glm52_pack_quantized_moe_mlp_is_not_attention" in report["blockers"]
    assert "glm52_pack_quantized_moe_mlp_is_not_stage_decode" in report["blockers"]
    assert check.validate_report(report) == []


def test_checker_rejects_pack_quantized_moe_mlp_stage_decode_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _mcp_tpu_stage_runtime_watch(ready=True))
    moe_mlp = _write(base / "moe-mlp.json", _pack_quantized_moe_mlp(stage_decode_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--pack-quantized-moe-mlp-report",
            str(moe_mlp),
        ])
    )

    assert "pack_quantized_moe_mlp_stage_decode_overclaim" in check.validate_report(report)


def test_rc_rejects_live_push_reports_collected_but_not_verified() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _tpu_report(ready=True, queued=False))
    push = _stage_worker_push_probe()
    push["mode"] = "live"
    push["live_run_performed"] = True
    push["stage_runtime_reports_collected"] = len(pack.REQUIRED_PROVIDERS)
    push["stage_runtime_reports_verified"] = 0
    for item in push["pushes"]:
        item["pushed"] = True
        item["terminal_status"] = "COMPLETE"
        item["output_collected"] = True
        item["stage_report_present"] = True
        item["stage_runtime_verified"] = False
        item["stage_report_check"] = {"ok": False, "stage_runtime_verified": False}
        item["cleanup_performed"] = True
    push_probe = _write(base / "push.json", push)

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--stage-worker-push-probe-report",
            str(push_probe),
        ])
    )

    errors = check.validate_report(report)

    assert "stage_worker_push_reports_collected_but_not_verified" in errors


def test_rc_imports_tpu_stage_smoke_without_claiming_same_request_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _tpu_report(ready=True, queued=False))
    awq = _write(base / "awq.json", _awq_stage_header())
    smoke = _write(base / "smoke.json", _tpu_stage_smoke())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--awq-stage-header-report",
            str(awq),
            "--tpu-stage-smoke-report",
            str(smoke),
        ])
    )

    assert report["goal_achieved"] is False
    assert report["tpu_stage_smoke"]["stage_runtime_adapter_smoke_ready"] is True
    assert report["tpu_stage_smoke"]["jax_tpu_device_count"] == 8
    assert "glm52_awq_tpu_stage_runtime_adapter_not_verified" in report["blockers"]
    assert "glm52_same_request_decode_not_verified" in report["blockers"]
    assert check.validate_report(report) == []


def test_checker_rejects_tpu_stage_smoke_same_request_overclaim() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report())
    tpu = _write(base / "tpu.json", _tpu_report(ready=True, queued=False))
    smoke = _write(base / "smoke.json", _tpu_stage_smoke(same_request_overclaim=True))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--tpu-stage-smoke-report",
            str(smoke),
        ])
    )

    assert "tpu_stage_smoke_same_request_overclaim" in check.validate_report(report)


def test_rc_accepts_full_live_same_request_success_only_with_strict_evidence() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report(stage_runtime=True, route=True))
    tpu = _write(base / "tpu.json", _tpu_report(ready=True, queued=False))
    awq_payload = _awq_stage_header()
    awq_payload["stage_runtime_adapter_verified"] = True
    awq_payload["same_request_route_verified"] = True
    awq = _write(base / "awq.json", awq_payload)
    same = _write(base / "same.json", _same_request())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--awq-stage-header-report",
            str(awq),
            "--same-request-report",
            str(same),
        ])
    )

    assert report["goal_achieved"] is True
    assert report["same_request"]["live_run_performed"] is True
    assert report["same_request"]["stage_provider_coverage_verified"] is True
    assert report["same_request"]["cleanup_verified"] is True
    assert report["blockers"] == []
    assert check.validate_report(report) == []


def test_rc_accepts_live_same_request_as_final_runtime_proof_over_metadata_gaps() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report(stage_runtime=False, route=False))
    tpu = _write(base / "tpu.json", _tpu_report(ready=True, queued=False))
    same = _write(base / "same.json", _same_request())

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--same-request-report",
            str(same),
        ])
    )

    assert report["goal_achieved"] is True
    assert report["source"]["compatible_with_glm52"] is True
    assert report["same_request"]["same_request_decode_verified"] is True
    assert report["same_request"]["generated_token_hash_present"] is True
    assert report["success"]["accepted_providers"] == pack.REQUIRED_PROVIDERS
    assert report["blockers"] == []
    assert check.validate_report(report) == []


def test_checker_rejects_queued_tpu_claimed_as_success() -> None:
    report = {
        "schema": pack.SCHEMA,
        "ok": True,
        "glm52_kaggle_accelerator_deployment_rc_ready": True,
        "public_artifact_safe": True,
        "goal_achieved": True,
        "model": {"model_id": pack.MODEL_ID, "fallback_model_allowed_for_success": False},
        "source": {
            "present": True,
            "resolver_ready": True,
            "compatible_with_glm52": True,
            "model_id": pack.MODEL_ID,
            "official_weight_key_count": 1,
            "recommended_repo": pack.MODEL_ID,
            "stage_runtime_adapter_verified": True,
            "same_request_route_verified": True,
            "public_artifact_safe": True,
        },
        "tpu_request": {"present": True, "queued": True, "tpu_runtime_ready": False, "public_artifact_safe": True},
        "same_request": {
            "present": True,
            "same_request_decode_verified": True,
            "generated_token_count": 1,
            "accepted_providers": pack.REQUIRED_PROVIDERS,
            "model_id": pack.MODEL_ID,
        },
        "success": {"same_request_decode_verified": True, "generated_token_count": 1, "accepted_providers": pack.REQUIRED_PROVIDERS},
        "blockers": [],
        "completion_boundary": {
            "queue_evidence_is_not_success": True,
            "metadata_only_source_is_not_success": True,
            "kaggle_source_search_is_not_success": True,
            "stage_header_evidence_is_not_success": True,
            "stage_smoke_evidence_is_not_success": True,
            "single_backend_inference_is_not_success": True,
            "fallback_model_is_not_success": True,
            "requires_real_kaggle_cpu_gpu_tpu_same_request": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
        },
    }

    errors = check.validate_report(report)

    assert "success_without_tpu_runtime_ready" in errors
    assert "tpu_queued_and_ready_conflict" not in errors


def test_checker_rejects_non_glm_same_request_success() -> None:
    base = _tmp_dir()
    source = _write(base / "source.json", _source_report(stage_runtime=True, route=True))
    tpu = _write(base / "tpu.json", _tpu_report(ready=True, queued=False))
    same = _write(base / "same.json", _same_request(model_id="Qwen/Qwen2.5-32B-Instruct"))

    report = pack.build_report(
        pack.parse_args([
            "--output-dir",
            str(base / "rc"),
            "--source-report",
            str(source),
            "--tpu-watch-report",
            str(tpu),
            "--same-request-report",
            str(same),
        ])
    )

    assert report["goal_achieved"] is False
    assert "same_request_model_id_not_glm52" in report["blockers"]
    assert check.validate_report(report) == []
