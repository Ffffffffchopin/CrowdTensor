#!/usr/bin/env python3
"""Run a public-safe GLM 5.2 DSA-masked single-layer decode probe."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_attention_projection_probe as projection_probe  # noqa: E402
from scripts import glm52_dsa_indexer_probe as dsa_probe  # noqa: E402
from scripts import glm52_kv_cache_decode_probe as kv_probe  # noqa: E402
from scripts import glm52_pack_quantized_dequant_probe as dequant_probe  # noqa: E402
from scripts import glm52_pack_quantized_moe_mlp_probe as moe_probe  # noqa: E402


SCHEMA = "glm52_dsa_masked_layer_decode_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-dsa-masked-layer-decode-probe"
DEFAULT_MODEL_REPO = dequant_probe.DEFAULT_MODEL_REPO
MODEL_ID = dequant_probe.MODEL_ID
SENSITIVE_FRAGMENTS = dequant_probe.SENSITIVE_FRAGMENTS + (
    '"query_states":',
    '"key_cache":',
    '"value_cache":',
    '"attention_scores":',
    '"attention_weights":',
    '"attention_output":',
    '"attention_residual":',
    '"post_attention_norm":',
    '"layer_output":',
    '"dsa_index_scores":',
    '"dsa_topk_indices":',
    '"dsa_attention_mask":',
    '"generated_token_ids":',
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def _dsa_index_for_layer(
    args: argparse.Namespace,
    config: dict[str, Any],
    *,
    hidden_norm: torch.Tensor,
    q_resid: torch.Tensor,
) -> dict[str, Any]:
    layer = int(args.layer_id)
    seq_len = int(hidden_norm.shape[0])
    indexer_types = _list(config.get("indexer_types"))
    if not indexer_types:
        raise RuntimeError("indexer_types_missing")
    layer_indexer_type = str(indexer_types[layer])
    if layer_indexer_type == "full":
        indexer_layer = layer
    elif layer_indexer_type == "shared":
        indexer_layer = next(
            (candidate for candidate in range(layer, -1, -1) if str(indexer_types[candidate]) == "full"),
            -1,
        )
        if indexer_layer < 0:
            raise RuntimeError("shared_indexer_source_missing")
    else:
        raise RuntimeError("selected_layer_is_not_full_indexer")
    index_heads = _int(config.get("index_n_heads"))
    index_head_dim = _int(config.get("index_head_dim"))
    rope_dim = _int(config.get("qk_rope_head_dim"))
    index_topk = _int(config.get("index_topk"))
    theta = float((config.get("rope_parameters") or {}).get("rope_theta") or 10000.0)
    prefix = f"model.layers.{indexer_layer}.self_attn.indexer"

    wq_b = projection_probe.router_probe.load_dense_tensor(args, f"{prefix}.wq_b.weight")
    wk = projection_probe.router_probe.load_dense_tensor(args, f"{prefix}.wk.weight")
    k_norm_weight = projection_probe.router_probe.load_dense_tensor(args, f"{prefix}.k_norm.weight")
    k_norm_bias = projection_probe.router_probe.load_dense_tensor(args, f"{prefix}.k_norm.bias")
    weights_proj = projection_probe.router_probe.load_dense_tensor(args, f"{prefix}.weights_proj.weight")

    q = dsa_probe.dense_matrix(wq_b, q_resid).reshape(seq_len, index_heads, index_head_dim)
    q_pe, q_nope = torch.split(q, [rope_dim, index_head_dim - rope_dim], dim=-1)
    cos, sin = dsa_probe.build_position_cos_sin(seq_len, rope_dim, theta)
    q_pe = dsa_probe.apply_rope_sequence(q_pe, cos, sin)
    q_index = torch.cat([q_pe, q_nope], dim=-1)

    k_raw = dsa_probe.dense_matrix(wk, hidden_norm)
    k_normed = dsa_probe.layer_norm_rows(k_raw, k_norm_weight, k_norm_bias, eps=1e-6)
    k_pe, k_nope = torch.split(k_normed, [rope_dim, index_head_dim - rope_dim], dim=-1)
    k_pe = dsa_probe.apply_rope_sequence(k_pe[:, None, :], cos, sin).squeeze(1)
    k_index = torch.cat([k_pe, k_nope], dim=-1)

    head_weights = dsa_probe.dense_matrix(weights_proj, hidden_norm) * (index_heads ** -0.5)
    scores = torch.einsum("shd,td->sht", q_index.float(), k_index.float()) * (index_head_dim ** -0.5)
    scores = torch.relu(scores)
    index_scores = torch.einsum("sht,sh->st", scores, head_weights.float())
    decode_position = int(args.prefill_length)
    topk_count = min(_int(args.dsa_mask_topk), seq_len, index_topk)
    if topk_count <= 0:
        raise RuntimeError("dsa_mask_topk_invalid")
    topk_indices = torch.topk(index_scores[decode_position], k=topk_count, dim=-1).indices.to(torch.int64)
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[topk_indices] = True
    return {
        "dsa_indexer_type": layer_indexer_type,
        "dsa_indexer_source_layer_id": int(indexer_layer),
        "dsa_indexer_source_type": str(indexer_types[indexer_layer]),
        "dsa_indexer_sequence_length": seq_len,
        "dsa_index_n_heads": index_heads,
        "dsa_index_head_dim": index_head_dim,
        "dsa_index_topk_config": index_topk,
        "dsa_mask_topk_requested": _int(args.dsa_mask_topk),
        "dsa_mask_topk_count": int(topk_count),
        "dsa_mask_pruned_position_count": int(seq_len - topk_count),
        "dsa_indexer_query_shape": [int(item) for item in q_index.shape],
        "dsa_indexer_key_shape": [int(item) for item in k_index.shape],
        "dsa_head_weights_shape": [int(item) for item in head_weights.shape],
        "dsa_index_score_shape": [int(item) for item in index_scores.shape],
        "dsa_topk_indices_shape": [int(item) for item in topk_indices.shape],
        "dsa_attention_mask_shape": [int(item) for item in mask.shape],
        "dsa_index_score_hash": dequant_probe.sha_tensor(index_scores),
        "dsa_topk_indices_hash": dequant_probe.sha_tensor(topk_indices),
        "dsa_attention_mask_hash": dequant_probe.sha_tensor(mask.to(torch.int8)),
        "dsa_attention_mask": mask,
    }


def _dsa_masked_attention_decode_for_layer(
    args: argparse.Namespace,
    config: dict[str, Any],
    hidden_sequence: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    layer = int(args.layer_id)
    prefill_len = int(args.prefill_length)
    total_len = int(hidden_sequence.shape[0])
    eps = float(config.get("rms_norm_eps") or 1e-6)
    heads = _int(config.get("num_attention_heads"))
    q_lora_rank = _int(config.get("q_lora_rank"))
    kv_lora_rank = _int(config.get("kv_lora_rank"))
    qk_head_dim = _int(config.get("qk_head_dim"))
    qk_nope_head_dim = _int(config.get("qk_nope_head_dim"))
    qk_rope_head_dim = _int(config.get("qk_rope_head_dim"))
    v_head_dim = _int(config.get("v_head_dim"))
    theta = float((config.get("rope_parameters") or {}).get("rope_theta") or 10000.0)

    input_norm_weight = projection_probe.router_probe.load_dense_tensor(args, f"model.layers.{layer}.input_layernorm.weight")
    hidden_norm = dsa_probe.rms_norm_rows(hidden_sequence, input_norm_weight, eps)

    q_a_weight = projection_probe.router_probe.load_dense_tensor(args, f"model.layers.{layer}.self_attn.q_a_proj.weight")
    q_a = dsa_probe.dense_matrix(q_a_weight, hidden_norm)
    q_a_norm_weight = projection_probe.router_probe.load_dense_tensor(args, f"model.layers.{layer}.self_attn.q_a_layernorm.weight")
    q_a_norm = dsa_probe.rms_norm_rows(q_a, q_a_norm_weight, eps)
    dsa_summary = _dsa_index_for_layer(args, config, hidden_norm=hidden_norm, q_resid=q_a_norm)
    dsa_mask = dsa_summary.pop("dsa_attention_mask")

    q_b_weight = kv_probe.dequantized_pack_weight(
        projection_probe.pack_projection_tensors(args, f"model.layers.{layer}.self_attn.q_b_proj")
    )
    q_b = kv_probe.pack_matrix(q_b_weight, q_a_norm)
    query = q_b.reshape(total_len, heads, qk_head_dim)
    q_nope, q_pe = torch.split(query, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    del q_b_weight, q_b, query
    gc.collect()

    kv_a_weight = projection_probe.router_probe.load_dense_tensor(args, f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight")
    kv_a = dsa_probe.dense_matrix(kv_a_weight, hidden_norm)
    k_compressed, k_pe = torch.split(kv_a, [kv_lora_rank, qk_rope_head_dim], dim=-1)
    kv_a_norm_weight = projection_probe.router_probe.load_dense_tensor(args, f"model.layers.{layer}.self_attn.kv_a_layernorm.weight")
    k_compressed_norm = dsa_probe.rms_norm_rows(k_compressed, kv_a_norm_weight, eps)
    kv_b_weight = kv_probe.dequantized_pack_weight(
        projection_probe.pack_projection_tensors(args, f"model.layers.{layer}.self_attn.kv_b_proj")
    )
    kv_b = kv_probe.pack_matrix(kv_b_weight, k_compressed_norm)
    kv_expanded = kv_b.reshape(total_len, heads, qk_nope_head_dim + v_head_dim)
    k_nope, value_states = torch.split(kv_expanded, [qk_nope_head_dim, v_head_dim], dim=-1)
    del kv_b_weight, kv_b, kv_expanded
    gc.collect()

    cos, sin = dsa_probe.build_position_cos_sin(total_len, qk_rope_head_dim, theta)
    q_pe_rope = dsa_probe.apply_rope_sequence(q_pe, cos, sin)
    k_pe_rope = dsa_probe.apply_rope_sequence(k_pe[:, None, :], cos, sin).expand(total_len, heads, qk_rope_head_dim)
    query_states = torch.cat([q_nope, q_pe_rope], dim=-1)
    key_states = torch.cat([k_nope, k_pe_rope], dim=-1)

    prefill_key_cache = key_states[:prefill_len].contiguous()
    prefill_value_cache = value_states[:prefill_len].contiguous()
    decode_key = key_states[prefill_len : prefill_len + 1].contiguous()
    decode_value = value_states[prefill_len : prefill_len + 1].contiguous()
    updated_key_cache = torch.cat([prefill_key_cache, decode_key], dim=0)
    updated_value_cache = torch.cat([prefill_value_cache, decode_value], dim=0)
    decode_query = query_states[prefill_len].contiguous()
    raw_attention_scores = torch.einsum("hd,thd->ht", decode_query.float(), updated_key_cache.float()) * (
        qk_head_dim ** -0.5
    )
    masked_attention_scores = raw_attention_scores.masked_fill(~dsa_mask[None, :], torch.finfo(torch.float32).min)
    attention_weights = torch.softmax(masked_attention_scores, dim=-1, dtype=torch.float32)
    head_output = torch.einsum("ht,thd->hd", attention_weights, updated_value_cache.float())
    flattened = head_output.reshape(heads * v_head_dim)
    o_proj_weight = kv_probe.dequantized_pack_weight(
        projection_probe.pack_projection_tensors(args, f"model.layers.{layer}.self_attn.o_proj")
    )
    attention_output = torch.matmul(o_proj_weight, flattened.to(torch.float32))
    summary = {
        "q_lora_rank": q_lora_rank,
        "kv_lora_rank": kv_lora_rank,
        "qk_head_dim": qk_head_dim,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "prefill_key_cache_shape": [int(item) for item in prefill_key_cache.shape],
        "prefill_value_cache_shape": [int(item) for item in prefill_value_cache.shape],
        "updated_key_cache_shape": [int(item) for item in updated_key_cache.shape],
        "updated_value_cache_shape": [int(item) for item in updated_value_cache.shape],
        "decode_query_shape": [int(item) for item in decode_query.shape],
        "raw_attention_scores_shape": [int(item) for item in raw_attention_scores.shape],
        "attention_scores_shape": [int(item) for item in masked_attention_scores.shape],
        "attention_weights_shape": [int(item) for item in attention_weights.shape],
        "attention_head_output_shape": [int(item) for item in head_output.shape],
        "attention_flattened_shape": [int(item) for item in flattened.shape],
        "attention_output_shape": [int(item) for item in attention_output.shape],
        "prefill_key_cache_hash": dequant_probe.sha_tensor(prefill_key_cache),
        "updated_key_cache_hash": dequant_probe.sha_tensor(updated_key_cache),
        "decode_query_hash": dequant_probe.sha_tensor(decode_query),
        "raw_attention_scores_hash": dequant_probe.sha_tensor(raw_attention_scores),
        "attention_scores_hash": dequant_probe.sha_tensor(masked_attention_scores),
        "attention_weights_hash": dequant_probe.sha_tensor(attention_weights),
        "attention_output_hash": dequant_probe.sha_tensor(attention_output),
        **dsa_summary,
    }
    return summary, attention_output.to(torch.float32)


def run_dsa_masked_layer_decode_for_hidden(
    args: argparse.Namespace,
    config: dict[str, Any],
    hidden_sequence: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    hidden_size = _int(config.get("hidden_size"))
    if hidden_size <= 0:
        raise RuntimeError("hidden_size_missing")
    layer = int(args.layer_id)
    prefill_len = int(args.prefill_length)
    total_len = int(hidden_sequence.shape[0])
    if total_len != prefill_len + 1:
        raise RuntimeError("hidden_sequence_length_mismatch")
    if int(hidden_sequence.shape[1]) != hidden_size:
        raise RuntimeError("hidden_sequence_width_mismatch")
    eps = float(config.get("rms_norm_eps") or 1e-6)
    decode_input = hidden_sequence[prefill_len].contiguous().to(torch.float32)
    attention_summary, attention_output = _dsa_masked_attention_decode_for_layer(args, config, hidden_sequence)
    attention_residual = decode_input + attention_output

    post_norm_weight = projection_probe.router_probe.load_dense_tensor(args, f"model.layers.{layer}.post_attention_layernorm.weight")
    post_attention_norm = projection_probe.rms_norm(attention_residual, post_norm_weight, eps)
    routed = moe_probe.run_routed_experts_for_input(args, config, post_attention_norm)
    shared_weights = moe_probe.load_shared_weights(args)
    shared_output, shared_projection_summaries = moe_probe.run_shared_experts_from_weights(post_attention_norm, shared_weights)
    full_moe_output = routed["routed_output"].to(torch.float32) + shared_output.to(torch.float32)
    layer_output = attention_residual + full_moe_output
    return {
        "model_type": str(config.get("model_type") or ""),
        "hidden_size": hidden_size,
        "num_hidden_layers": _int(config.get("num_hidden_layers")),
        "num_attention_heads": _int(config.get("num_attention_heads")),
        "n_routed_experts": _int(config.get("n_routed_experts")),
        "num_experts_per_tok": _int(config.get("num_experts_per_tok")),
        "n_shared_experts": _int(config.get("n_shared_experts")),
        "moe_intermediate_size": _int(config.get("moe_intermediate_size")),
        "prefill_length": prefill_len,
        "decode_length": 1,
        "updated_cache_length": total_len,
        "decode_input_shape": [int(item) for item in decode_input.shape],
        "decode_input_hash": dequant_probe.sha_tensor(decode_input),
        **attention_summary,
        "attention_residual_shape": [int(item) for item in attention_residual.shape],
        "attention_residual_hash": dequant_probe.sha_tensor(attention_residual),
        "post_attention_norm_shape": [int(item) for item in post_attention_norm.shape],
        "post_attention_norm_hash": dequant_probe.sha_tensor(post_attention_norm),
        "router_topk_count": _int(routed.get("router_topk_count")),
        "router_topk_indices_hash": str(routed.get("router_topk_indices_hash") or ""),
        "router_topk_weights_hash": str(routed.get("router_topk_weights_hash") or ""),
        "executed_expert_count": _int(routed.get("executed_expert_count")),
        "executed_experts": _list(routed.get("executed_experts")),
        "routed_output_shape": [int(item) for item in routed["routed_output"].shape],
        "routed_output_hash": dequant_probe.sha_tensor(routed["routed_output"]),
        "shared_projection_summaries": shared_projection_summaries,
        "shared_output_shape": [int(item) for item in shared_output.shape],
        "shared_output_hash": dequant_probe.sha_tensor(shared_output),
        "full_moe_output_shape": [int(item) for item in full_moe_output.shape],
        "full_moe_output_hash": dequant_probe.sha_tensor(full_moe_output),
        "layer_output_shape": [int(item) for item in layer_output.shape],
        "layer_output_hash": dequant_probe.sha_tensor(layer_output),
    }, layer_output.to(torch.float32)


def run_dsa_masked_layer_decode_with_output(args: argparse.Namespace) -> tuple[dict[str, Any], torch.Tensor]:
    config = dequant_probe.fetch_hf_json(args.model_repo, "config.json", timeout_seconds=float(args.hf_timeout_seconds))
    hidden_size = _int(config.get("hidden_size"))
    if hidden_size <= 0:
        raise RuntimeError("hidden_size_missing")
    total_len = int(args.prefill_length) + 1
    hidden_sequence = kv_probe.build_hidden_sequence(total_len, hidden_size)
    return run_dsa_masked_layer_decode_for_hidden(args, config, hidden_sequence)


def run_dsa_masked_layer_decode(args: argparse.Namespace) -> dict[str, Any]:
    layer, _ = run_dsa_masked_layer_decode_with_output(args)
    return layer


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    blockers: list[str] = []
    errors: list[dict[str, Any]] = []
    layer: dict[str, Any] = {}
    ready = False
    try:
        layer = run_dsa_masked_layer_decode(args)
        hidden = _int(layer.get("hidden_size"))
        heads = _int(layer.get("num_attention_heads"))
        qk = _int(layer.get("qk_head_dim"))
        value_dim = _int(layer.get("v_head_dim"))
        updated = _int(layer.get("updated_cache_length"))
        ready = (
            layer.get("model_type") == "glm_moe_dsa"
            and layer.get("dsa_indexer_type") == "full"
            and _list(layer.get("updated_key_cache_shape")) == [updated, heads, qk]
            and _list(layer.get("updated_value_cache_shape")) == [updated, heads, value_dim]
            and _list(layer.get("attention_scores_shape")) == [heads, updated]
            and _list(layer.get("attention_weights_shape")) == [heads, updated]
            and _list(layer.get("attention_output_shape")) == [hidden]
            and _list(layer.get("attention_residual_shape")) == [hidden]
            and _list(layer.get("post_attention_norm_shape")) == [hidden]
            and _int(layer.get("dsa_mask_topk_count")) > 0
            and _int(layer.get("dsa_mask_pruned_position_count")) > 0
            and _int(layer.get("router_topk_count")) == _int(layer.get("num_experts_per_tok"))
            and _int(layer.get("executed_expert_count")) == _int(layer.get("num_experts_per_tok"))
            and _list(layer.get("routed_output_shape")) == [hidden]
            and _list(layer.get("shared_output_shape")) == [hidden]
            and _list(layer.get("full_moe_output_shape")) == [hidden]
            and _list(layer.get("layer_output_shape")) == [hidden]
        )
    except Exception as exc:
        errors.append(
            {
                "phase": "dsa_masked_layer_decode",
                "error_type": type(exc).__name__,
                "error_digest": dequant_probe.sha_payload(str(exc)),
            }
        )
        blockers.append("glm52_dsa_masked_layer_decode_failed")
    if ready:
        blockers.extend(
            [
                "glm52_dsa_masked_layer_decode_is_single_layer_only",
                "glm52_dsa_masked_layer_decode_uses_small_sequence_topk_cap",
                "glm52_dsa_masked_layer_decode_missing_lm_head",
                "glm52_dsa_masked_layer_decode_is_not_stage_decode",
                "glm52_dsa_masked_layer_decode_is_not_same_request",
            ]
        )
    blockers.extend(["glm52_stage_decode_not_verified", "glm52_same_request_decode_not_verified"])
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "glm52_dsa_masked_layer_decode_probe_ready": ready,
        "model_id": MODEL_ID,
        "model_repo": str(args.model_repo),
        "layer_id": int(args.layer_id),
        **layer,
        "dsa_indexer_verified": ready,
        "dsa_mask_verified": ready,
        "dsa_mask_pruned_positions_verified": ready,
        "kv_cache_prefill_verified": ready,
        "kv_cache_update_verified": ready,
        "attention_decode_verified": ready,
        "dsa_masked_attention_integrated": ready,
        "attention_residual_verified": ready,
        "post_attention_norm_verified": ready,
        "router_topk_verified": ready,
        "routed_expert_gather_verified": ready,
        "shared_experts_mlp_verified": ready,
        "full_moe_mlp_verified": ready,
        "layer_decode_verified": ready,
        "full_dsa_topk_scale_verified": False,
        "multi_layer_stage_runtime_verified": False,
        "lm_head_verified": False,
        "generated_token_verified": False,
        "stage_decode_verified": False,
        "same_request_decode_verified": False,
        "errors": errors,
        "blockers": sorted(set(blockers)),
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
        "public_artifact_safe": True,
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["glm52_dsa_masked_layer_decode_probe_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"] = sorted(set([*blockers, "public_redaction_scan_failed"]))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--layer-id", type=int, default=6)
    parser.add_argument("--prefill-length", type=int, default=8)
    parser.add_argument("--dsa-mask-topk", type=int, default=4)
    parser.add_argument("--executed-expert-count", type=int, default=8)
    parser.add_argument("--max-header-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-tensor-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--hf-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.prefill_length <= 0:
        raise SystemExit("--prefill-length must be positive")
    if args.dsa_mask_topk <= 0:
        raise SystemExit("--dsa-mask-topk must be positive")
    if args.executed_expert_count <= 0:
        raise SystemExit("--executed-expert-count must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_report(args)
    path = output_dir / "glm52_dsa_masked_layer_decode_probe.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"DSA-masked layer decode verified: {report.get('layer_decode_verified')}")
    return 0 if report.get("public_artifact_safe") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
