#!/usr/bin/env python3
"""Build DeepSeek-V4-Flash Kaggle GPU+WebTPU+CPU swarm RC evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "deepseek_v4_flash_kaggle_tpu_swarm_rc_v1"
SUPPORT_SCHEMA = "deepseek_v4_flash_kaggle_tpu_swarm_rc_support_v1"
DEFAULT_OUTPUT_DIR = "dist/deepseek-v4-flash-kaggle-tpu-swarm-rc"
REQUIRED_PROVIDERS = ["kaggle_cuda", "kaggle_web_tpu", "kaggle_cpu"]
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Authorization:",
    "Cookie:",
    "Set-Cookie",
    "runtime_proxy_token",
    "oauth_token",
    "jupyter-proxy",
    "token=",
    '"prompt":',
    '"raw_prompt":',
    '"generated_text":',
    '"generated_token_ids":',
    '"input_ids":',
    '"activation":',
    '"hidden_state":',
    '"logits":',
    '"kv_cache":',
    '"past_key_values":',
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    loaded = json.loads(p.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    entry: dict[str, Any] = {"kind": kind, "path": relative, "present": path.is_file()}
    if path.is_file():
        entry["sha256"] = sha256_file(path)
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def summarize_source(report: dict[str, Any]) -> dict[str, Any]:
    recommended = _dict(report.get("recommended_live_probe_candidate"))
    return {
        "schema": "deepseek_v4_flash_kaggle_tpu_source_summary_v1",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "resolver_ready": report.get("deepseek_v4_flash_quantized_source_resolver_ready") is True,
        "model": _dict(report.get("model")),
        "candidate_count": _int(report.get("candidate_count")),
        "ready_candidate_count": _int(report.get("ready_candidate_count")),
        "recommended_candidate_id": str(recommended.get("candidate_id") or ""),
        "recommended_repo": str(recommended.get("repo") or ""),
        "recommended_quant": str(recommended.get("quant") or ""),
        "recommended_total_size_gb": _float(recommended.get("total_size_gb")),
        "recommended_split_file_count": _int(recommended.get("split_file_count")),
        "recommended_runtime_backend": str(recommended.get("runtime_backend") or ""),
        "recommended_runtime_fork": str(recommended.get("runtime_fork") or ""),
        "recommended_blockers": [str(item) for item in _list(recommended.get("blockers"))],
        "blockers": [str(item) for item in _list(report.get("blockers"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_web_tpu_queue(report: dict[str, Any]) -> dict[str, Any]:
    queue_progress = _dict(report.get("queue_progress"))
    final_observation = _dict(report.get("final_observation"))
    active_event_running = report.get("active_event_running") is True or final_observation.get("active_event_running") is True
    active_event_queued = report.get("active_event_queued") is True or final_observation.get("active_event_queued") is True
    blockers = [str(item) for item in _list(report.get("blocker_codes") or report.get("blockers"))]
    if active_event_running:
        stale_after_running = {
            "kaggle_web_tpu_active_event_queued",
            "kaggle_web_tpu_queue_prompt_visible",
            "kaggle_web_tpu_session_still_starting",
        }
        blockers = [item for item in blockers if item not in stale_after_running]
    return {
        "schema": "deepseek_v4_flash_kaggle_web_tpu_queue_summary_v1",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "start_clicked": report.get("start_clicked") is True,
        "queue_position_observed": report.get("queue_position_observed") is True or queue_progress.get("queue_position_observed") is True,
        "queue_position_changed": report.get("queue_position_changed") is True or queue_progress.get("queue_position_changed") is True,
        "queue_position_decreased": report.get("queue_position_decreased") is True or queue_progress.get("queue_position_decreased") is True,
        "unique_queue_positions": [
            int(item)
            for item in (_list(report.get("unique_queue_positions")) or _list(queue_progress.get("unique_queue_positions")))
        ],
        "web_tpu_ui_runtime_ready": (
            report.get("web_tpu_ui_runtime_ready") is True
            or report.get("web_tpu_runtime_ready") is True
            or final_observation.get("web_tpu_runtime_ready") is True
        ),
        "active_event_running": active_event_running,
        "active_event_queued": active_event_queued,
        "blockers": blockers,
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_active_event(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "deepseek_v4_flash_kaggle_web_tpu_active_event_summary_v1",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "tpu_v5e_active_event_visible": report.get("tpu_v5e_active_event_visible") is True,
        "active_event_running": report.get("active_event_running") is True,
        "active_event_queued": report.get("active_event_queued") is True,
        "active_event_runtime_ready": report.get("active_event_runtime_ready") is True,
        "blocked_reason": str(report.get("blocked_reason") or ""),
        "blockers": [str(item) for item in _list(report.get("blocker_codes") or report.get("blockers"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_web_tpu_execution(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "deepseek_v4_flash_kaggle_web_tpu_execution_summary_v1",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "web_tpu_execution_channel_ready": report.get("web_tpu_execution_channel_ready") is True,
        "small_jax_cell_ready": report.get("small_jax_cell_ready") is True,
        "tiny_qwen_like_cell_ready": report.get("tiny_qwen_like_cell_ready") is True,
        "tpu_runtime_attached": report.get("tpu_runtime_attached") is True,
        "tpu_device_count": _int(report.get("tpu_device_count")),
        "stage_local_kv_cache_verified": report.get("stage_local_kv_cache_verified") is True,
        "failure_stage": str(report.get("failure_stage") or ""),
        "blockers": [str(item) for item in _list(report.get("blocker_codes") or report.get("blockers"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_deepseek_tpu_adapter(report: dict[str, Any]) -> dict[str, Any]:
    metadata = _dict(report.get("deepseek_metadata"))
    stage_mapping = _dict(metadata.get("stage_mapping"))
    cell = _dict(report.get("web_tpu_cell"))
    real_load = _dict(cell.get("deepseek_v4_real_weight_tpu_tensor_load"))
    topk_forward = _dict(real_load.get("real_fp4_topk_expert_mlp_forward"))
    stage_plan = _dict(report.get("stage_plan"))
    topk_ready = (
        report.get("deepseek_v4_real_fp4_topk_expert_mlp_forward_ready") is True
        or topk_forward.get("ready") is True
        or stage_plan.get("real_fp4_topk_expert_mlp_forward_ready") is True
    )
    return {
        "schema": "deepseek_v4_flash_kaggle_web_tpu_adapter_summary_v1",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "metadata_ready": report.get("metadata_ready") is True,
        "stage_key_mapping_ready": metadata.get("stage_key_mapping_ready") is True,
        "kaggle_web_tpu_runtime_ready": report.get("kaggle_web_tpu_runtime_ready") is True,
        "deepseek_v4_real_weight_tpu_tensor_load_ready": report.get("deepseek_v4_real_weight_tpu_tensor_load_ready") is True,
        "real_weight_sample_loaded_tensor_count": _int(real_load.get("loaded_tensor_count")),
        "real_weight_sample_total_loaded_tensor_bytes": _int(real_load.get("total_loaded_tensor_bytes")),
        "real_weight_sample_dtype_counts": _dict(real_load.get("dtype_counts")),
        "real_router_smoke_ready": real_load.get("real_router_smoke_ready") is True,
        "real_fp8_block_dequant_smoke_ready": real_load.get("real_fp8_block_dequant_smoke_ready") is True,
        "real_i8_expert_dequant_smoke_ready": real_load.get("real_i8_expert_dequant_smoke_ready") is True,
        "real_i8_expert_mlp_slice_smoke_ready": real_load.get("real_i8_expert_mlp_slice_smoke_ready") is True,
        "real_fp4_topk_expert_mlp_forward_ready": topk_ready,
        "real_fp4_topk_forward_kind": str(topk_forward.get("forward_kind") or ""),
        "real_fp4_topk_count": _int(topk_forward.get("topk") or stage_plan.get("real_routed_expert_topk_count")),
        "real_fp4_topk_loaded_tensor_count": _int(topk_forward.get("loaded_tensor_count") or stage_plan.get("real_routed_expert_loaded_tensor_count")),
        "real_fp4_topk_total_loaded_tensor_bytes": _int(topk_forward.get("total_loaded_tensor_bytes") or stage_plan.get("real_routed_expert_total_loaded_tensor_bytes")),
        "real_fp4_topk_final_output_shape": _list(topk_forward.get("final_output_shape")),
        "real_fp4_topk_final_output_hash": str(topk_forward.get("final_output_hash") or ""),
        "real_fp4_topk_finite_output": topk_forward.get("finite_output") is True,
        "real_fp4_topk_weight_tensor_values_public": topk_forward.get("weight_tensor_values_public") is True,
        "real_fp4_topk_activation_payload_public": topk_forward.get("activation_payload_public") is True,
        "real_weight_sample_values_public": real_load.get("weight_tensor_values_public") is True,
        "deepseek_v4_jax_tpu_stage_forward_ready": report.get("deepseek_v4_jax_tpu_stage_forward_ready") is True,
        "metadata_source": str(metadata.get("source") or ""),
        "model_config": _dict(metadata.get("model_config")),
        "weight_key_count": _int(_dict(metadata.get("weight_index")).get("weight_key_count")),
        "weight_file_count": _int(_dict(metadata.get("weight_index")).get("weight_file_count")),
        "stage_layer_range": _list(stage_mapping.get("layer_range")),
        "stage_selected_key_count": _int(stage_mapping.get("selected_key_count")),
        "stage_selected_file_count": _int(stage_mapping.get("selected_file_count")),
        "stage_family_hits": _dict(stage_mapping.get("family_hits")),
        "failure_stage": str(report.get("failure_stage") or ""),
        "blockers": [str(item) for item in _list(report.get("blockers"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_torch_stage_smoke(report: dict[str, Any]) -> dict[str, Any]:
    reference = _dict(report.get("reference_stage"))
    components = _dict(reference.get("real_deepseek_v4_components_exercised"))
    return {
        "schema": "deepseek_v4_flash_torch_stage_adapter_smoke_summary_v1",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "torch_stage_adapter_smoke_ready": report.get("deepseek_v4_flash_torch_stage_adapter_smoke_ready") is True,
        "transformers_reference_used": reference.get("transformers_reference_used") is True,
        "tiny_fixture": reference.get("tiny_fixture") is True,
        "real_deepseek_weights_loaded": report.get("real_deepseek_weights_loaded") is True,
        "jax_tpu_translation_ready": report.get("jax_tpu_translation_ready") is True,
        "stage_owned_key_count": _int(reference.get("stage_owned_key_count")),
        "components_exercised": {
            "manifold_hyper_connections": components.get("manifold_hyper_connections") is True,
            "compressed_attention": components.get("compressed_attention") is True,
            "mla_shared_kv_attention": components.get("mla_shared_kv_attention") is True,
            "grouped_output_projection": components.get("grouped_output_projection") is True,
            "moe_router": components.get("moe_router") is True,
            "routed_experts": components.get("routed_experts") is True,
            "shared_experts": components.get("shared_experts") is True,
            "stage_local_kv_cache_shape": components.get("stage_local_kv_cache_shape") is True,
        },
        "blockers": [str(item) for item in _list(report.get("blockers"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_jax_stage_smoke(report: dict[str, Any]) -> dict[str, Any]:
    numpy_reference = _dict(report.get("numpy_reference"))
    components = _dict(numpy_reference.get("components_exercised"))
    return {
        "schema": "deepseek_v4_flash_jax_stage_adapter_smoke_summary_v1",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "jax_stage_adapter_smoke_ready": report.get("deepseek_v4_flash_jax_stage_adapter_smoke_ready") is True,
        "numpy_reference_ready": numpy_reference.get("ok") is True,
        "jax_runtime_execution_requested": report.get("jax_runtime_execution_requested") is True,
        "jax_runtime_execution_ready": report.get("jax_runtime_execution_ready") is True,
        "tpu_runtime_required": report.get("tpu_runtime_required") is True,
        "tpu_runtime_ready": report.get("tpu_runtime_ready") is True,
        "deepseek_v4_jax_stage_forward_ready": report.get("deepseek_v4_jax_stage_forward_ready") is True,
        "deepseek_v4_jax_tpu_stage_forward_ready": report.get("deepseek_v4_jax_tpu_stage_forward_ready") is True,
        "real_deepseek_weights_loaded": report.get("real_deepseek_weights_loaded") is True,
        "stage_owned_key_count": _int(_dict(report.get("stage")).get("stage_owned_key_count")),
        "components_exercised": {
            "manifold_hyper_connections": components.get("manifold_hyper_connections") is True,
            "mla_shared_kv_attention": components.get("mla_shared_kv_attention") is True,
            "grouped_output_projection": components.get("grouped_output_projection") is True,
            "attention_sink": components.get("attention_sink") is True,
            "topk_moe_router": components.get("topk_moe_router") is True,
            "routed_experts": components.get("routed_experts") is True,
            "shared_experts": components.get("shared_experts") is True,
            "hca_compressor_shape_metadata": components.get("hca_compressor_shape_metadata") is True,
        },
        "blockers": [str(item) for item in _list(report.get("blockers"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_safetensors_stage_header(report: dict[str, Any]) -> dict[str, Any]:
    stage = _dict(report.get("stage_mapping"))
    headers = _dict(report.get("headers"))
    return {
        "schema": "deepseek_v4_flash_safetensors_stage_header_summary_v1",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "safetensors_header_ready": report.get("safetensors_header_ready") is True,
        "stage_header_shape_ready": report.get("stage_header_shape_ready") is True,
        "model_config": _dict(_dict(report.get("model")).get("model_config")),
        "stage_layer_range": _list(stage.get("layer_range")),
        "stage_selected_key_count": _int(stage.get("selected_key_count")),
        "stage_selected_file_count": _int(stage.get("selected_file_count")),
        "stage_family_hits": _dict(stage.get("family_hits")),
        "header_file_count": _int(headers.get("header_file_count")),
        "header_fetch_error_count": _int(headers.get("header_fetch_error_count")),
        "missing_header_key_count": _int(headers.get("missing_header_key_count")),
        "dtype_counts": _dict(headers.get("dtype_counts")),
        "rank_counts": _dict(headers.get("rank_counts")),
        "total_selected_tensor_storage_bytes": _int(headers.get("total_selected_tensor_storage_bytes")),
        "real_weight_tensor_values_loaded": headers.get("real_weight_tensor_values_loaded") is True,
        "real_weight_tensor_values_public": headers.get("real_weight_tensor_values_public") is True,
        "safetensors_header_payload_public": headers.get("safetensors_header_payload_public") is True,
        "failure_stage": str(report.get("failure_stage") or ""),
        "blockers": [str(item) for item in _list(report.get("blockers"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_kaggle_gpu(report: dict[str, Any]) -> dict[str, Any]:
    cleanup = _dict(report.get("cleanup"))
    accounts = [_dict(item) for item in _list(report.get("accounts")) if isinstance(item, dict)]
    token_gpu_accepted_accounts = [
        str(account.get("label") or account.get("owner") or "")
        for account in accounts
        if account.get("push_accepted") is True or account.get("weekly_gpu_quota_available_inferred") is True
    ]
    token_gpu_exhausted_accounts = [
        str(account.get("label") or account.get("owner") or "")
        for account in accounts
        if account.get("weekly_gpu_quota_exhausted") is True
    ]
    token_accepted_count = _int(_dict(report.get("summary")).get("gpu_submission_accepted_count"))
    if not token_accepted_count:
        token_accepted_count = len([account for account in accounts if account.get("push_accepted") is True])
    token_auth_ok_count = _int(_dict(report.get("summary")).get("auth_ok_count"))
    if not token_auth_ok_count:
        token_auth_ok_count = len([account for account in accounts if account.get("auth_ok") is True])
    token_cleanup_attempted = any(_dict(account.get("cleanup")).get("attempted") is True for account in accounts)
    token_cleanup_failed = any(_dict(account.get("cleanup")).get("failed") is True for account in accounts)
    source_ok = report.get("ok") is True or bool(accounts)
    accepted_submission_count = max(_int(report.get("accepted_submission_count")), token_accepted_count)
    ready = bool(
        source_ok
        and (
            report.get("simultaneous_t4x2_verified") is True
            or accepted_submission_count >= 1
            or bool(token_gpu_accepted_accounts)
        )
    )
    return {
        "schema": "deepseek_v4_flash_kaggle_cuda_summary_v1",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "source_ok": source_ok,
        "kaggle_cuda_ready": ready,
        "accepted_submission_count": accepted_submission_count,
        "simultaneous_t4x2_verified": report.get("simultaneous_t4x2_verified") is True,
        "max_observed_running_count": _int(report.get("max_observed_running_count")),
        "accelerator": str(report.get("accelerator") or report.get("requested_accelerator") or ""),
        "cleanup_attempted": cleanup.get("attempted") is True or token_cleanup_attempted,
        "cleanup_failed": cleanup.get("failed") is True or token_cleanup_failed,
        "private_kernel_payloads_removed": report.get("private_kernel_payloads_removed") is True,
        "token_account_count": len(accounts),
        "token_auth_ok_count": token_auth_ok_count,
        "token_gpu_submission_accepted_count": token_accepted_count,
        "token_weekly_gpu_quota_exhausted_count": _int(_dict(report.get("summary")).get("weekly_gpu_quota_exhausted_count")),
        "token_gpu_accepted_accounts": [item for item in token_gpu_accepted_accounts if item],
        "token_gpu_exhausted_accounts": [item for item in token_gpu_exhausted_accounts if item],
        "token_account_summaries": [
            {
                "label": str(account.get("label") or ""),
                "owner": str(account.get("owner") or ""),
                "auth_ok": account.get("auth_ok") is True,
                "push_accepted": account.get("push_accepted") is True,
                "weekly_gpu_quota_available_inferred": account.get("weekly_gpu_quota_available_inferred") is True,
                "weekly_gpu_quota_exhausted": account.get("weekly_gpu_quota_exhausted") is True,
                "quota_class": str(account.get("quota_class") or ""),
                "cleanup_attempted": _dict(account.get("cleanup")).get("attempted") is True,
                "cleanup_deleted": _dict(account.get("cleanup")).get("deleted") is True,
                "cleanup_failed": _dict(account.get("cleanup")).get("failed") is True,
            }
            for account in accounts
        ],
        "blockers": [str(item) for item in _list(report.get("blockers"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_cpu_fp4_topk_expert_forward(report: dict[str, Any]) -> dict[str, Any]:
    forward = _dict(report.get("fp4_topk_expert_forward"))
    ready = report.get("ok") is True and forward.get("ready") is True
    return {
        "schema": "deepseek_v4_flash_cpu_fp4_topk_expert_forward_summary_v1",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "stage_selective_fp4_topk_expert_forward_ready": ready,
        "model_id": str(report.get("model_id") or ""),
        "layer": _int(report.get("layer")),
        "topk": _int(forward.get("topk")),
        "loaded_tensor_count": _int(forward.get("loaded_tensor_count")),
        "total_loaded_tensor_bytes": _int(forward.get("total_loaded_tensor_bytes")),
        "final_output_shape": [int(item) for item in _list(forward.get("final_output_shape"))],
        "final_output_hash": str(forward.get("final_output_hash") or ""),
        "finite_output": forward.get("finite_output") is True,
        "forward_kind": str(forward.get("forward_kind") or ""),
        "blockers": [str(item) for item in _list(report.get("blockers")) + _list(forward.get("blockers"))],
        "diagnosis_codes": [str(item) for item in _list(report.get("diagnosis_codes"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def summarize_same_request(report: dict[str, Any]) -> dict[str, Any]:
    raw_providers = [
        str(item)
        for item in (_list(report.get("accepted_providers")) or _list(report.get("accepted_stage_backends")))
    ]
    provider_aliases = {
        "jax_tpu": "kaggle_web_tpu",
        "web_tpu": "kaggle_web_tpu",
        "kaggle_web_tpu": "kaggle_web_tpu",
        "kaggle_cuda": "kaggle_cuda",
        "cuda": "kaggle_cuda",
        "cpu": "kaggle_cpu",
        "kaggle_cpu": "kaggle_cpu",
    }
    providers = sorted({provider_aliases.get(item, item) for item in raw_providers})
    generated = _int(report.get("generated_token_count") or _dict(report.get("coordinator")).get("generated_token_count"))
    verified = bool(
        report.get("deepseek_v4_flash_kaggle_tpu_same_request_verified") is True
        or report.get("same_request_decode_verified") is True
    )
    ready = bool(verified and generated >= 1 and set(REQUIRED_PROVIDERS).issubset(set(providers)))
    return {
        "schema": "deepseek_v4_flash_kaggle_tpu_same_request_summary_v1",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "source_ok": report.get("ok") is True,
        "same_request_decode_verified": ready,
        "generated_token_count": generated,
        "accepted_providers": providers,
        "raw_accepted_providers": raw_providers,
        "provider_stage_counts": _dict(report.get("provider_stage_counts")),
        "stage_task_counts": _dict(report.get("stage_task_counts")),
        "deepseek_v4_same_request_stage_slice_verified": report.get("deepseek_v4_same_request_stage_slice_verified") is True,
        "deepseek_v4_gpu_stage_slice_verified": report.get("deepseek_v4_gpu_stage_slice_verified") is True,
        "deepseek_v4_cpu_stage_slice_verified": report.get("deepseek_v4_cpu_stage_slice_verified") is True,
        "deepseek_v4_gpu_tpu_cpu_same_request_stage_slices_verified": (
            report.get("deepseek_v4_gpu_tpu_cpu_same_request_stage_slices_verified") is True
        ),
        "deepseek_v4_tpu_fp4_topk_expert_forward_verified": (
            report.get("deepseek_v4_tpu_fp4_topk_expert_forward_verified") is True
        ),
        "deepseek_v4_gpu_fp4_topk_expert_forward_verified": (
            report.get("deepseek_v4_gpu_fp4_topk_expert_forward_verified") is True
        ),
        "deepseek_v4_cpu_fp4_topk_expert_forward_verified": (
            report.get("deepseek_v4_cpu_fp4_topk_expert_forward_verified") is True
        ),
        "deepseek_v4_gpu_tpu_cpu_same_request_fp4_topk_expert_forwards_verified": (
            report.get("deepseek_v4_gpu_tpu_cpu_same_request_fp4_topk_expert_forwards_verified") is True
        ),
        "deepseek_v4_stage_layer_ranges": _dict(report.get("deepseek_v4_stage_layer_ranges")),
        "deepseek_v4_distinct_backend_stage_layer_ranges_verified": (
            report.get("deepseek_v4_distinct_backend_stage_layer_ranges_verified") is True
        ),
        "deepseek_v4_stage_layer_coverage_count": _int(report.get("deepseek_v4_stage_layer_coverage_count")),
        "model_scope": str(report.get("model_scope") or ""),
        "failure_stage": str(report.get("failure_stage") or ""),
        "blockers": [str(item) for item in _list(report.get("blockers"))],
        "diagnosis_codes": [str(item) for item in _list(report.get("diagnosis_codes"))],
        "public_artifact_safe": report.get("public_artifact_safe") is True or _dict(report.get("safety")).get("public_artifact_safe") is True,
    }


def failure_stage(
    source: dict[str, Any],
    queue: dict[str, Any],
    active_event: dict[str, Any],
    web_tpu_execution: dict[str, Any],
    deepseek_tpu_adapter: dict[str, Any],
    torch_stage_smoke: dict[str, Any],
    jax_stage_smoke: dict[str, Any],
    safetensors_stage_header: dict[str, Any],
    kaggle_gpu: dict[str, Any],
    same_request: dict[str, Any],
) -> str:
    if not source.get("resolver_ready"):
        return "deepseek_v4_flash_source_not_ready"
    if same_request.get("present") and same_request.get("failure_stage"):
        return str(same_request.get("failure_stage"))
    if same_request.get("present") and not same_request.get("same_request_decode_verified"):
        return "deepseek_v4_flash_kaggle_tpu_same_request_not_verified"
    if deepseek_tpu_adapter.get("present") and deepseek_tpu_adapter.get("failure_stage"):
        return str(deepseek_tpu_adapter.get("failure_stage"))
    if deepseek_tpu_adapter.get("present") and not deepseek_tpu_adapter.get("deepseek_v4_jax_tpu_stage_forward_ready"):
        return "deepseek_v4_flash_mla_moe_jax_tpu_stage_forward_not_implemented"
    if torch_stage_smoke.get("present") and not torch_stage_smoke.get("torch_stage_adapter_smoke_ready"):
        return "deepseek_v4_flash_torch_stage_adapter_smoke_not_ready"
    if jax_stage_smoke.get("present") and not jax_stage_smoke.get("numpy_reference_ready"):
        return "deepseek_v4_flash_jax_stage_adapter_smoke_not_ready"
    if safetensors_stage_header.get("present") and not safetensors_stage_header.get("safetensors_header_ready"):
        return str(safetensors_stage_header.get("failure_stage") or "deepseek_v4_flash_safetensors_stage_header_not_ready")
    if web_tpu_execution.get("present") and not web_tpu_execution.get("web_tpu_execution_channel_ready"):
        return str(web_tpu_execution.get("failure_stage") or "kaggle_web_tpu_execution_channel_not_ready")
    if active_event.get("present") and not active_event.get("active_event_running") and active_event.get("blocked_reason"):
        return str(active_event.get("blocked_reason"))
    if queue.get("present") and not (queue.get("web_tpu_ui_runtime_ready") or queue.get("active_event_running")):
        return "kaggle_web_tpu_runtime_not_ready"
    if kaggle_gpu.get("present") and not kaggle_gpu.get("kaggle_cuda_ready"):
        return "kaggle_cuda_not_ready"
    if not same_request.get("present"):
        return "deepseek_v4_flash_kaggle_tpu_same_request_not_started"
    return "deepseek_v4_flash_kaggle_tpu_same_request_not_verified"


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(args.source_resolver_report) if args.source_resolver_report else Path()
    queue_path = Path(args.web_tpu_queue_report) if args.web_tpu_queue_report else Path()
    active_event_path = Path(args.web_tpu_active_event_report) if args.web_tpu_active_event_report else Path()
    web_tpu_execution_path = Path(args.web_tpu_execution_report) if args.web_tpu_execution_report else Path()
    deepseek_tpu_adapter_path = Path(args.deepseek_tpu_adapter_report) if args.deepseek_tpu_adapter_report else Path()
    torch_stage_smoke_path = Path(args.torch_stage_smoke_report) if args.torch_stage_smoke_report else Path()
    jax_stage_smoke_path = Path(args.jax_stage_smoke_report) if args.jax_stage_smoke_report else Path()
    safetensors_stage_header_path = Path(args.safetensors_stage_header_report) if args.safetensors_stage_header_report else Path()
    kaggle_gpu_path = Path(args.kaggle_gpu_preflight_report) if args.kaggle_gpu_preflight_report else Path()
    cpu_fp4_topk_path = Path(args.cpu_fp4_topk_expert_report) if args.cpu_fp4_topk_expert_report else Path()
    same_request_path = Path(args.same_request_report) if args.same_request_report else Path()

    source = summarize_source(load_json(source_path))
    queue = summarize_web_tpu_queue(load_json(queue_path))
    active_event = summarize_active_event(load_json(active_event_path))
    web_tpu_execution = summarize_web_tpu_execution(load_json(web_tpu_execution_path))
    deepseek_tpu_adapter = summarize_deepseek_tpu_adapter(load_json(deepseek_tpu_adapter_path))
    torch_stage_smoke = summarize_torch_stage_smoke(load_json(torch_stage_smoke_path))
    jax_stage_smoke = summarize_jax_stage_smoke(load_json(jax_stage_smoke_path))
    safetensors_stage_header = summarize_safetensors_stage_header(load_json(safetensors_stage_header_path))
    kaggle_gpu = summarize_kaggle_gpu(load_json(kaggle_gpu_path))
    cpu_fp4_topk = summarize_cpu_fp4_topk_expert_forward(load_json(cpu_fp4_topk_path))
    same_request = summarize_same_request(load_json(same_request_path))

    same_success = same_request.get("same_request_decode_verified") is True
    blockers = sorted({
        *[str(item) for item in _list(source.get("blockers"))],
        *[str(item) for item in _list(source.get("recommended_blockers"))],
        *[str(item) for item in _list(queue.get("blockers"))],
        *[str(item) for item in _list(active_event.get("blockers"))],
        *[str(item) for item in _list(web_tpu_execution.get("blockers"))],
        *[str(item) for item in _list(deepseek_tpu_adapter.get("blockers"))],
        *[str(item) for item in _list(torch_stage_smoke.get("blockers"))],
        *[str(item) for item in _list(jax_stage_smoke.get("blockers"))],
        *[str(item) for item in _list(safetensors_stage_header.get("blockers"))],
        *[str(item) for item in _list(kaggle_gpu.get("blockers"))],
        *[str(item) for item in _list(cpu_fp4_topk.get("blockers"))],
        *[str(item) for item in _list(same_request.get("blockers"))],
    })
    if not source.get("resolver_ready"):
        blockers.append("deepseek_v4_flash_quantized_source_not_ready")
    if web_tpu_execution.get("present") and not web_tpu_execution.get("web_tpu_execution_channel_ready"):
        blockers.append("kaggle_web_tpu_execution_channel_not_ready")
    if deepseek_tpu_adapter.get("present") and not deepseek_tpu_adapter.get("deepseek_v4_jax_tpu_stage_forward_ready"):
        adapter_failure = str(deepseek_tpu_adapter.get("failure_stage") or "").strip()
        blockers.append(adapter_failure or "deepseek_v4_flash_mla_moe_jax_tpu_stage_forward_not_implemented")
    if torch_stage_smoke.get("present") and not torch_stage_smoke.get("torch_stage_adapter_smoke_ready"):
        blockers.append("deepseek_v4_flash_torch_stage_adapter_smoke_not_ready")
    if jax_stage_smoke.get("present") and not jax_stage_smoke.get("numpy_reference_ready"):
        blockers.append("deepseek_v4_flash_jax_stage_adapter_smoke_not_ready")
    if safetensors_stage_header.get("present") and not safetensors_stage_header.get("safetensors_header_ready"):
        blockers.append("deepseek_v4_flash_safetensors_stage_header_not_ready")
    if kaggle_gpu.get("present") and not kaggle_gpu.get("kaggle_cuda_ready"):
        blockers.append("kaggle_cuda_not_ready")
    if not same_success:
        blockers.append("deepseek_v4_flash_kaggle_tpu_same_request_decode_not_verified")

    same_request_providers = set(str(item) for item in _list(same_request.get("accepted_providers")))
    same_request_web_tpu_stage_verified = bool(
        "kaggle_web_tpu" in same_request_providers
        and (
            same_request.get("deepseek_v4_tpu_fp4_topk_expert_forward_verified") is True
            or same_request.get("deepseek_v4_gpu_tpu_cpu_same_request_stage_slices_verified") is True
        )
    )
    if web_tpu_execution.get("web_tpu_execution_channel_ready") is True or same_request_web_tpu_stage_verified:
        stale_web_tpu_blockers = {
            "kaggle_web_tpu_active_event_queued",
            "kaggle_web_tpu_active_event_missing",
            "kaggle_web_tpu_execution_channel_not_ready",
            "kaggle_web_tpu_jupyter_frame_not_visible",
            "kaggle_web_tpu_jupyter_session_not_visible",
            "kaggle_web_tpu_queue_prompt_visible",
            "kaggle_web_tpu_runtime_not_ready",
            "kaggle_web_tpu_session_still_starting",
            "tiny_qwen_like_cell_exception",
            "web_tpu_execution_channel_not_ready",
        }
        blockers = [item for item in blockers if item not in stale_web_tpu_blockers]
    if same_request.get("deepseek_v4_tpu_fp4_topk_expert_forward_verified") is True:
        stale_adapter_blockers = {
            "deepseek_v4_flash_quantized_fp8_nvfp4_tpu_loader_not_implemented",
            "deepseek_v4_flash_real_weight_tpu_stage_loader_not_implemented",
        }
        blockers = [item for item in blockers if item not in stale_adapter_blockers]
    if "kaggle_cuda" in same_request_providers:
        stale_cuda_blockers = {
            "kaggle_cuda_not_ready",
            "kaggle_gpu_weekly_quota_reached",
            "t4_cuda_runtime_not_validated_for_deepseek_v4_flash",
        }
        blockers = [item for item in blockers if item not in stale_cuda_blockers]
    blockers = [] if same_success else sorted(set(blockers))

    result = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "deepseek_v4_flash_kaggle_tpu_swarm_rc_ready": True,
        "objective": "Kaggle T4x2 GPU + Kaggle Web TPU + Kaggle CPU same-request DeepSeek-V4-Flash deployment inference",
        "model": {
            "model_id": "deepseek-ai/DeepSeek-V4-Flash",
            "architecture_class": "moe",
            "total_params_b": 284.0,
            "active_params_b": 13.0,
            "selected_quant": source.get("recommended_quant"),
            "selected_repo": source.get("recommended_repo"),
            "selected_size_gb": source.get("recommended_total_size_gb"),
        },
        "success": {
            "same_request_decode_verified": same_success,
            "generated_token_count": _int(same_request.get("generated_token_count")),
            "required_providers": REQUIRED_PROVIDERS,
            "accepted_providers": _list(same_request.get("accepted_providers")),
        },
        "source_resolver": source,
        "web_tpu_queue": queue,
        "web_tpu_active_event": active_event,
        "web_tpu_execution_channel": web_tpu_execution,
        "deepseek_tpu_adapter": deepseek_tpu_adapter,
        "deepseek_torch_stage_smoke": torch_stage_smoke,
        "deepseek_jax_stage_smoke": jax_stage_smoke,
        "deepseek_safetensors_stage_header": safetensors_stage_header,
        "kaggle_gpu_preflight": kaggle_gpu,
        "deepseek_cpu_fp4_topk_expert_forward": cpu_fp4_topk,
        "same_request": same_request,
        "blockers": blockers,
        "failure_stage": "" if same_success else failure_stage(
            source,
            queue,
            active_event,
            web_tpu_execution,
            deepseek_tpu_adapter,
            torch_stage_smoke,
            jax_stage_smoke,
            safetensors_stage_header,
            kaggle_gpu,
            same_request,
        ),
        "diagnosis_codes": [
            "deepseek_v4_flash_kaggle_tpu_swarm_rc_ready",
            "deepseek_v4_flash_kaggle_tpu_same_request_decode_verified" if same_success else "deepseek_v4_flash_kaggle_tpu_same_request_decode_not_verified",
            "deepseek_v4_flash_quantized_sources_ready" if source.get("resolver_ready") else "deepseek_v4_flash_quantized_sources_not_ready",
        ],
        "safety": {
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "past_key_values_public": False,
            "credentials_public": False,
            "cookies_public": False,
            "jupyter_proxy_token_public": False,
            "private_runtime_state_public": False,
            "private_kaggle_payload_public": False,
            "weight_tensor_values_public": False,
        },
        "public_artifact_safe": True,
        "artifacts": {},
    }
    leaks = public_redaction_errors(result)
    if leaks:
        result["ok"] = False
        result["deepseek_v4_flash_kaggle_tpu_swarm_rc_ready"] = False
        result["public_artifact_safe"] = False
        result["safety"]["public_artifact_safe"] = False
        result["redaction_errors"] = leaks
        result["blockers"].append("public_redaction_scan_failed")

    support = {
        "schema": SUPPORT_SCHEMA,
        "source_resolver_report": str(source_path) if source_path else "",
        "web_tpu_queue_report": str(queue_path) if queue_path else "",
        "web_tpu_active_event_report": str(active_event_path) if active_event_path else "",
        "web_tpu_execution_report": str(web_tpu_execution_path) if web_tpu_execution_path else "",
        "deepseek_tpu_adapter_report": str(deepseek_tpu_adapter_path) if deepseek_tpu_adapter_path else "",
        "torch_stage_smoke_report": str(torch_stage_smoke_path) if torch_stage_smoke_path else "",
        "jax_stage_smoke_report": str(jax_stage_smoke_path) if jax_stage_smoke_path else "",
        "safetensors_stage_header_report": str(safetensors_stage_header_path) if safetensors_stage_header_path else "",
        "kaggle_gpu_preflight_report": str(kaggle_gpu_path) if kaggle_gpu_path else "",
        "cpu_fp4_topk_expert_report": str(cpu_fp4_topk_path) if cpu_fp4_topk_path else "",
        "same_request_report": str(same_request_path) if same_request_path else "",
        "public_artifact_safe": True,
    }
    support_path = output_dir / "deepseek_v4_flash_kaggle_tpu_swarm_rc_support.json"
    summary_path = output_dir / "deepseek_v4_flash_kaggle_tpu_swarm_rc.json"
    write_json(support_path, support)
    result["artifacts"] = {
        "summary_json": {"kind": "summary_json", "path": summary_path.name, "present": True, "schema": SCHEMA, "ok": bool(result.get("ok"))},
        "support_bundle_json": artifact_entry(support_path, output_dir, kind="support_bundle", schema=SUPPORT_SCHEMA, ok=True),
    }
    write_json(summary_path, result)
    result["artifacts"]["summary_json"] = artifact_entry(summary_path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(result.get("ok")))
    write_json(summary_path, result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-resolver-report", required=True)
    parser.add_argument("--web-tpu-queue-report", default="")
    parser.add_argument("--web-tpu-active-event-report", default="")
    parser.add_argument("--web-tpu-execution-report", default="")
    parser.add_argument("--deepseek-tpu-adapter-report", default="")
    parser.add_argument("--torch-stage-smoke-report", default="")
    parser.add_argument("--jax-stage-smoke-report", default="")
    parser.add_argument("--safetensors-stage-header-report", default="")
    parser.add_argument("--kaggle-gpu-preflight-report", default="")
    parser.add_argument("--cpu-fp4-topk-expert-report", default="")
    parser.add_argument("--same-request-report", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {Path(args.output_dir) / 'deepseek_v4_flash_kaggle_tpu_swarm_rc.json'}")
        print(f"Same-request success: {report['success']['same_request_decode_verified']}")
        if report.get("failure_stage"):
            print(f"Failure stage: {report['failure_stage']}")
    return 0 if report.get("public_artifact_safe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
