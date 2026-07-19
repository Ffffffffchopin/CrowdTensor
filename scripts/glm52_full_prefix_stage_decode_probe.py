#!/usr/bin/env python3
"""Run a public-safe GLM 5.2 full-prefix multi-layer stage decode probe."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from copy import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_attention_projection_probe as projection_probe  # noqa: E402
from scripts import glm52_dsa_masked_layer_decode_probe as layer_probe  # noqa: E402
from scripts import glm52_kv_cache_decode_probe as kv_probe  # noqa: E402
from scripts import glm52_lm_head_token_probe as lm_head_probe  # noqa: E402
from scripts import glm52_pack_quantized_dequant_probe as dequant_probe  # noqa: E402
from scripts import glm52_pack_quantized_router_gather_probe as router_probe  # noqa: E402


SCHEMA = "glm52_full_prefix_stage_decode_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-full-prefix-stage-decode-probe"
DEFAULT_MODEL_REPO = dequant_probe.DEFAULT_MODEL_REPO
MODEL_ID = dequant_probe.MODEL_ID
SENSITIVE_FRAGMENTS = layer_probe.SENSITIVE_FRAGMENTS + lm_head_probe.SENSITIVE_FRAGMENTS + (
    "CT_GLM52_INPUT_HIDDEN_B64",
    '"input_hidden_b64":',
    '"private_input_hidden_b64":',
    '"hidden_b64":',
    '"output_activation":',
    '"stage_hidden_sequence":',
    '"stage_hidden":',
    '"normalized_stage_hidden":',
    '"selected_token":',
    '"token_id":',
    '"token_ids":',
    '"generated_token_ids":',
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _hash_ok(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) >= 71


def _mlp_type_for_layer(config: dict[str, Any], layer_id: int) -> str:
    mlp_types = [str(item) for item in _list(config.get("mlp_layer_types"))]
    if int(layer_id) < len(mlp_types) and mlp_types[int(layer_id)]:
        return mlp_types[int(layer_id)]
    dense_count = _int(config.get("first_k_dense_replace"), 3)
    return "dense" if int(layer_id) < dense_count else "sparse"


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def public_error_text(exc: Exception) -> str:
    text = str(exc)[-500:]
    return "" if public_redaction_errors({"error_public": text}) else text


def _torch_dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name or "").strip().lower()
    if normalized in {"float16", "fp16", "torch.float16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32", "torch.float32", ""}:
        return torch.float32
    raise RuntimeError("input_hidden_dtype_unsupported")


def _private_input_hidden_arg(args: argparse.Namespace, name: str, env_name: str) -> str:
    value = str(getattr(args, name, "") or "").strip()
    return value or str(os.environ.get(env_name, "") or "").strip()


def load_private_input_hidden_sequence(
    args: argparse.Namespace,
    *,
    hidden_size: int,
    expected_sequence_length: int,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    raw_b64 = _private_input_hidden_arg(args, "input_hidden_b64", "CT_GLM52_INPUT_HIDDEN_B64")
    raw_shape = _private_input_hidden_arg(args, "input_hidden_shape_json", "CT_GLM52_INPUT_HIDDEN_SHAPE_JSON")
    if not raw_b64 and not raw_shape:
        return None, {
            "initial_stage_hidden_sequence_source": "deterministic_probe_carrier",
            "input_activation_consumed": False,
            "input_activation_public": False,
        }
    if not raw_b64 or not raw_shape:
        raise RuntimeError("input_hidden_activation_incomplete")
    shape = json.loads(raw_shape)
    if not isinstance(shape, list) or len(shape) != 2:
        raise RuntimeError("input_hidden_shape_invalid")
    clean_shape = [int(shape[0]), int(shape[1])]
    if clean_shape != [int(expected_sequence_length), int(hidden_size)]:
        raise RuntimeError("input_hidden_shape_mismatch")
    dtype_name = _private_input_hidden_arg(args, "input_hidden_dtype", "CT_GLM52_INPUT_HIDDEN_DTYPE") or "float16"
    dtype = _torch_dtype_from_name(dtype_name)
    raw = base64.b64decode(raw_b64.encode("ascii"), validate=True)
    element_size = torch.empty((), dtype=dtype).element_size()
    expected_bytes = int(clean_shape[0]) * int(clean_shape[1]) * element_size
    if len(raw) != expected_bytes:
        raise RuntimeError("input_hidden_byte_count_mismatch")
    tensor = torch.frombuffer(bytearray(raw), dtype=dtype).clone().reshape(clean_shape).to(torch.float32)
    return tensor, {
        "initial_stage_hidden_sequence_source": "private_upstream_activation",
        "input_activation_consumed": True,
        "input_activation_public": False,
        "input_activation_shape": clean_shape,
        "input_activation_dtype": str(dtype).replace("torch.", ""),
        "input_activation_hash": dequant_probe.sha_tensor(tensor),
    }


def write_private_output_activation(args: argparse.Namespace, hidden_sequence: torch.Tensor) -> dict[str, Any]:
    raw_path = str(getattr(args, "output_activation_path", "") or os.environ.get("CT_GLM52_OUTPUT_ACTIVATION_PATH", "") or "").strip()
    transport = hidden_sequence.detach().to(torch.float16).cpu().contiguous()
    activation_hash = dequant_probe.sha_tensor(transport.to(torch.float32))
    summary = {
        "output_activation_private_ready": False,
        "output_activation_public": False,
        "output_activation_shape": [int(item) for item in transport.shape],
        "output_activation_dtype": "float16",
        "output_activation_hash": activation_hash,
    }
    if not raw_path:
        return summary
    raw = transport.numpy().tobytes()
    payload = {
        "schema": "glm52_private_stage_activation_v1",
        "activation_hash": activation_hash,
        "hidden_shape": [int(item) for item in transport.shape],
        "hidden_dtype": "float16",
        "hidden_b64": base64.b64encode(raw).decode("ascii"),
        "activation_public": False,
    }
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return {**summary, "output_activation_private_ready": True}


def _layer_args(args: argparse.Namespace, *, layer_id: int, prefill_length: int) -> argparse.Namespace:
    layer_args = copy(args)
    layer_args.layer_id = int(layer_id)
    layer_args.prefill_length = int(prefill_length)
    return layer_args


def _token_ready(layer: dict[str, Any], config: dict[str, Any], *, position: int) -> bool:
    hidden = _int(layer.get("hidden_size"))
    heads = _int(layer.get("num_attention_heads"))
    qk = _int(layer.get("qk_head_dim"))
    value_dim = _int(layer.get("v_head_dim"))
    updated = int(position) + 1
    common_ready = bool(
        layer.get("model_type") == "glm_moe_dsa"
        and _list(layer.get("updated_key_cache_shape")) == [updated, heads, qk]
        and _list(layer.get("updated_value_cache_shape")) == [updated, heads, value_dim]
        and _list(layer.get("attention_scores_shape")) == [heads, updated]
        and _list(layer.get("attention_weights_shape")) == [heads, updated]
        and _list(layer.get("attention_output_shape")) == [hidden]
        and _list(layer.get("attention_residual_shape")) == [hidden]
        and _list(layer.get("post_attention_norm_shape")) == [hidden]
        and _list(layer.get("layer_output_shape")) == [hidden]
        and _hash_ok(layer.get("layer_output_hash"))
    )
    if str(layer.get("mlp_layer_type") or "") != "sparse":
        return bool(
            common_ready
            and _list(layer.get("dense_mlp_output_shape")) == [hidden]
            and _hash_ok(layer.get("dense_mlp_output_hash"))
        )
    return bool(
        common_ready
        and str(layer.get("dsa_indexer_type") or "") in {"full", "shared"}
        and _int(layer.get("dsa_indexer_source_layer_id"), -1) >= 0
        and _int(layer.get("dsa_mask_topk_count")) > 0
        and _int(layer.get("router_topk_count")) == _int(config.get("num_experts_per_tok"))
        and _int(layer.get("executed_expert_count")) == _int(config.get("num_experts_per_tok"))
        and _list(layer.get("full_moe_output_shape")) == [hidden]
    )


def _layer_summary(layer_id: int, token_summaries: list[dict[str, Any]], token_ready: list[bool]) -> dict[str, Any]:
    indexer_types = sorted({str(item.get("dsa_indexer_type") or "") for item in token_summaries})
    source_layers = sorted({_int(item.get("dsa_indexer_source_layer_id"), -1) for item in token_summaries})
    final = token_summaries[-1] if token_summaries else {}
    return {
        "layer_id": int(layer_id),
        "mlp_layer_types": sorted({str(item.get("mlp_layer_type") or "") for item in token_summaries}),
        "layer_full_prefix_verified": all(token_ready) and bool(token_ready),
        "token_count": len(token_summaries),
        "verified_token_count": sum(1 for item in token_ready if item),
        "dsa_indexer_types": indexer_types,
        "dsa_indexer_source_layer_ids": source_layers,
        "final_token_dsa_mask_topk_count": _int(final.get("dsa_mask_topk_count")),
        "final_token_dsa_mask_pruned_position_count": _int(final.get("dsa_mask_pruned_position_count")),
        "final_token_attention_scores_shape": _list(final.get("attention_scores_shape")),
        "final_token_full_moe_output_shape": _list(final.get("full_moe_output_shape")),
        "final_token_layer_output_shape": _list(final.get("layer_output_shape")),
        "final_token_layer_output_hash": str(final.get("layer_output_hash") or ""),
    }


def _dense_attention_decode_for_layer(
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

    input_norm_weight = router_probe.load_dense_tensor(args, f"model.layers.{layer}.input_layernorm.weight")
    hidden_norm = layer_probe.dsa_probe.rms_norm_rows(hidden_sequence, input_norm_weight, eps)

    q_a_weight = router_probe.load_dense_tensor(args, f"model.layers.{layer}.self_attn.q_a_proj.weight")
    q_a = layer_probe.dsa_probe.dense_matrix(q_a_weight, hidden_norm)
    q_a_norm_weight = router_probe.load_dense_tensor(args, f"model.layers.{layer}.self_attn.q_a_layernorm.weight")
    q_a_norm = layer_probe.dsa_probe.rms_norm_rows(q_a, q_a_norm_weight, eps)
    q_b_weight = router_probe.load_dense_tensor(args, f"model.layers.{layer}.self_attn.q_b_proj.weight")
    q_b = layer_probe.dsa_probe.dense_matrix(q_b_weight, q_a_norm)
    query = q_b.reshape(total_len, heads, qk_head_dim)
    q_nope, q_pe = torch.split(query, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)

    kv_a_weight = router_probe.load_dense_tensor(args, f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight")
    kv_a = layer_probe.dsa_probe.dense_matrix(kv_a_weight, hidden_norm)
    k_compressed, k_pe = torch.split(kv_a, [kv_lora_rank, qk_rope_head_dim], dim=-1)
    kv_a_norm_weight = router_probe.load_dense_tensor(args, f"model.layers.{layer}.self_attn.kv_a_layernorm.weight")
    k_compressed_norm = layer_probe.dsa_probe.rms_norm_rows(k_compressed, kv_a_norm_weight, eps)
    kv_b_weight = router_probe.load_dense_tensor(args, f"model.layers.{layer}.self_attn.kv_b_proj.weight")
    kv_b = layer_probe.dsa_probe.dense_matrix(kv_b_weight, k_compressed_norm)
    kv_expanded = kv_b.reshape(total_len, heads, qk_nope_head_dim + v_head_dim)
    k_nope, value_states = torch.split(kv_expanded, [qk_nope_head_dim, v_head_dim], dim=-1)

    cos, sin = layer_probe.dsa_probe.build_position_cos_sin(total_len, qk_rope_head_dim, theta)
    q_pe_rope = layer_probe.dsa_probe.apply_rope_sequence(q_pe, cos, sin)
    k_pe_rope = layer_probe.dsa_probe.apply_rope_sequence(k_pe[:, None, :], cos, sin).expand(total_len, heads, qk_rope_head_dim)
    query_states = torch.cat([q_nope, q_pe_rope], dim=-1)
    key_states = torch.cat([k_nope, k_pe_rope], dim=-1)

    prefill_key_cache = key_states[:prefill_len].contiguous()
    prefill_value_cache = value_states[:prefill_len].contiguous()
    decode_key = key_states[prefill_len : prefill_len + 1].contiguous()
    decode_value = value_states[prefill_len : prefill_len + 1].contiguous()
    updated_key_cache = torch.cat([prefill_key_cache, decode_key], dim=0)
    updated_value_cache = torch.cat([prefill_value_cache, decode_value], dim=0)
    decode_query = query_states[prefill_len].contiguous()
    attention_scores = torch.einsum("hd,thd->ht", decode_query.float(), updated_key_cache.float()) * (qk_head_dim ** -0.5)
    attention_weights = torch.softmax(attention_scores, dim=-1, dtype=torch.float32)
    head_output = torch.einsum("ht,thd->hd", attention_weights, updated_value_cache.float())
    flattened = head_output.reshape(heads * v_head_dim)
    o_proj_weight = router_probe.load_dense_tensor(args, f"model.layers.{layer}.self_attn.o_proj.weight")
    attention_output = torch.matmul(o_proj_weight.to(torch.float32), flattened.to(torch.float32))
    return {
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
        "attention_scores_shape": [int(item) for item in attention_scores.shape],
        "attention_weights_shape": [int(item) for item in attention_weights.shape],
        "attention_head_output_shape": [int(item) for item in head_output.shape],
        "attention_flattened_shape": [int(item) for item in flattened.shape],
        "attention_output_shape": [int(item) for item in attention_output.shape],
        "prefill_key_cache_hash": dequant_probe.sha_tensor(prefill_key_cache),
        "updated_key_cache_hash": dequant_probe.sha_tensor(updated_key_cache),
        "decode_query_hash": dequant_probe.sha_tensor(decode_query),
        "attention_scores_hash": dequant_probe.sha_tensor(attention_scores),
        "attention_weights_hash": dequant_probe.sha_tensor(attention_weights),
        "attention_output_hash": dequant_probe.sha_tensor(attention_output),
    }, attention_output.to(torch.float32)


def _dense_mlp_for_layer(args: argparse.Namespace, hidden: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    layer = int(args.layer_id)
    outputs: dict[str, torch.Tensor] = {}
    summaries: list[dict[str, Any]] = []
    for projection in ["gate_proj", "up_proj", "down_proj"]:
        weight = router_probe.load_dense_tensor(args, f"model.layers.{layer}.mlp.{projection}.weight")
        projection_input = hidden if projection != "down_proj" else torch.nn.functional.silu(outputs["gate_proj"]) * outputs["up_proj"]
        output = layer_probe.moe_probe.dense_linear(weight, projection_input)
        outputs[projection] = output
        summaries.append(
            {
                "projection": projection,
                "weight_shape": [int(item) for item in weight.shape],
                "output_shape": [int(item) for item in output.shape],
                "output_hash": dequant_probe.sha_tensor(output),
            }
        )
    return outputs["down_proj"].to(torch.float32), summaries


def run_dense_layer_decode_for_hidden(
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
    attention_summary, attention_output = _dense_attention_decode_for_layer(args, config, hidden_sequence)
    attention_residual = decode_input + attention_output
    post_norm_weight = router_probe.load_dense_tensor(args, f"model.layers.{layer}.post_attention_layernorm.weight")
    post_attention_norm = projection_probe.rms_norm(attention_residual, post_norm_weight, eps)
    dense_mlp_output, dense_projection_summaries = _dense_mlp_for_layer(args, post_attention_norm)
    layer_output = attention_residual + dense_mlp_output
    return {
        "model_type": str(config.get("model_type") or ""),
        "mlp_layer_type": "dense",
        "hidden_size": hidden_size,
        "num_hidden_layers": _int(config.get("num_hidden_layers")),
        "num_attention_heads": _int(config.get("num_attention_heads")),
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
        "dense_projection_summaries": dense_projection_summaries,
        "dense_mlp_output_shape": [int(item) for item in dense_mlp_output.shape],
        "dense_mlp_output_hash": dequant_probe.sha_tensor(dense_mlp_output),
        "layer_output_shape": [int(item) for item in layer_output.shape],
        "layer_output_hash": dequant_probe.sha_tensor(layer_output),
    }, layer_output.to(torch.float32)


def run_full_prefix_stage_decode(args: argparse.Namespace) -> dict[str, Any]:
    config = dequant_probe.fetch_hf_json(args.model_repo, "config.json", timeout_seconds=float(args.hf_timeout_seconds))
    index = dequant_probe.fetch_hf_json(args.model_repo, "model.safetensors.index.json", timeout_seconds=float(args.hf_timeout_seconds))
    weight_map = _dict(index.get("weight_map"))
    hidden_size = _int(config.get("hidden_size"))
    vocab_size = _int(config.get("vocab_size"))
    num_layers = _int(config.get("num_hidden_layers"))
    start = int(args.layer_start)
    end = int(args.layer_end)
    total_len = int(args.prefill_length) + 1
    if hidden_size <= 0 or vocab_size <= 0 or num_layers <= 0:
        raise RuntimeError("config_shape_missing")
    if start < 0 or end <= start or end > num_layers:
        raise RuntimeError("stage_layer_range_invalid")
    if total_len < 2:
        raise RuntimeError("full_prefix_sequence_too_short")

    private_hidden_sequence, input_activation_summary = load_private_input_hidden_sequence(
        args,
        hidden_size=hidden_size,
        expected_sequence_length=total_len,
    )
    hidden_sequence = (
        private_hidden_sequence
        if private_hidden_sequence is not None
        else kv_probe.build_hidden_sequence(total_len, hidden_size).to(torch.float32)
    )
    initial_sequence_hash = dequant_probe.sha_tensor(hidden_sequence)
    layer_summaries: list[dict[str, Any]] = []
    for layer_id in range(start, end):
        next_rows: list[torch.Tensor] = []
        token_summaries: list[dict[str, Any]] = []
        token_ready: list[bool] = []
        for position in range(total_len):
            prefix_hidden = hidden_sequence[: position + 1].contiguous()
            layer_args = _layer_args(args, layer_id=layer_id, prefill_length=position)
            if _mlp_type_for_layer(config, layer_id) == "sparse":
                token_summary, token_output = layer_probe.run_dsa_masked_layer_decode_for_hidden(
                    layer_args,
                    config,
                    prefix_hidden,
                )
                token_summary["mlp_layer_type"] = "sparse"
            else:
                token_summary, token_output = run_dense_layer_decode_for_hidden(layer_args, config, prefix_hidden)
            token_summaries.append(token_summary)
            token_ready.append(_token_ready(token_summary, config, position=position))
            next_rows.append(token_output.to(torch.float32))
        hidden_sequence = torch.stack(next_rows, dim=0).contiguous()
        layer_summaries.append(_layer_summary(layer_id, token_summaries, token_ready))

    stage_hidden = hidden_sequence[int(args.prefill_length)].contiguous().to(torch.float32)
    output_activation_summary = write_private_output_activation(args, hidden_sequence)
    norm_weight = router_probe.load_dense_tensor(args, "model.norm.weight")
    normalized_stage_hidden = projection_probe.rms_norm(
        stage_hidden.to(norm_weight.device), norm_weight, float(config.get("rms_norm_eps") or 1e-6)
    )
    common = {
        "model_type": str(config.get("model_type") or ""),
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "num_hidden_layers": num_layers,
        "tie_word_embeddings": bool(config.get("tie_word_embeddings")),
        "stage_hidden_source": "dsa_masked_full_prefix_multi_layer_stage_hidden",
        "stage_layer_range": [start, end],
        "stage_layer_count": end - start,
        "executed_layer_count": len(layer_summaries),
        "stage_prefill_length": int(args.prefill_length),
        "stage_sequence_length": total_len,
        "stage_updated_cache_length": total_len,
        "dsa_mask_topk_requested": int(args.dsa_mask_topk),
        "full_prefix_token_carrier_verified": True,
        "small_sequence_probe": True,
        **input_activation_summary,
        "initial_stage_hidden_sequence_shape": [total_len, hidden_size],
        "initial_stage_hidden_sequence_hash": initial_sequence_hash,
        "layer_summaries": layer_summaries,
        "all_layers_full_prefix_verified": all(item.get("layer_full_prefix_verified") is True for item in layer_summaries),
        "all_layer_outputs_chained": all(item.get("layer_full_prefix_verified") is True for item in layer_summaries),
        "stage_hidden_sequence_shape": [int(item) for item in hidden_sequence.shape],
        "stage_hidden_sequence_hash": dequant_probe.sha_tensor(hidden_sequence),
        **output_activation_summary,
        "stage_hidden_shape": [int(item) for item in stage_hidden.shape],
        "stage_hidden_hash": dequant_probe.sha_tensor(stage_hidden),
        "norm_weight_shape": [int(item) for item in norm_weight.shape],
        "normalized_stage_hidden_shape": [int(item) for item in normalized_stage_hidden.shape],
        "normalized_stage_hidden_hash": dequant_probe.sha_tensor(normalized_stage_hidden),
    }
    if args.skip_lm_head:
        return {
            **common,
            "lm_head_required": False,
            "lm_head_skipped_for_nonfinal_stage": True,
        }
    lm_head_file = str(weight_map.get("lm_head.weight") or "")
    if not lm_head_file:
        raise RuntimeError("lm_head_weight_missing")
    header_len, header = dequant_probe.load_safetensors_header_with_len(
        args.model_repo,
        lm_head_file,
        timeout_seconds=float(args.hf_timeout_seconds),
        max_header_bytes=int(args.max_header_bytes),
    )
    streamed = lm_head_probe.stream_lm_head_topk(
        args,
        model_repo=str(args.model_repo),
        filename=lm_head_file,
        header_len=header_len,
        item=_dict(header.get("lm_head.weight")),
        hidden=normalized_stage_hidden,
        top_k=int(args.top_k),
    )
    return {
        **common,
        "lm_head_required": True,
        "lm_head_skipped_for_nonfinal_stage": False,
        **streamed,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    blockers: list[str] = []
    errors: list[dict[str, Any]] = []
    result: dict[str, Any] = {}
    ready = False
    try:
        result = run_full_prefix_stage_decode(args)
        hidden = _int(result.get("hidden_size"))
        vocab = _int(result.get("vocab_size"))
        seq_len = _int(result.get("stage_sequence_length"))
        hidden_ready = (
            result.get("model_type") == "glm_moe_dsa"
            and result.get("stage_hidden_source") == "dsa_masked_full_prefix_multi_layer_stage_hidden"
            and _int(result.get("stage_layer_count")) >= 2
            and _int(result.get("executed_layer_count")) == _int(result.get("stage_layer_count"))
            and result.get("all_layers_full_prefix_verified") is True
            and result.get("all_layer_outputs_chained") is True
            and _list(result.get("stage_hidden_sequence_shape")) == [seq_len, hidden]
            and _list(result.get("stage_hidden_shape")) == [hidden]
            and _list(result.get("normalized_stage_hidden_shape")) == [hidden]
        )
        lm_head_required = result.get("lm_head_required") is not False
        lm_head_ready = (
            not lm_head_required
            or (
                _list(result.get("lm_head_shape")) == [vocab, hidden]
                and _int(result.get("lm_head_rows_scanned")) == vocab
                and _int(result.get("top_k_count")) == int(args.top_k)
            )
        )
        ready = hidden_ready and lm_head_ready
    except Exception as exc:
        errors.append(
            {
                "phase": "full_prefix_stage_decode",
                "error_type": type(exc).__name__,
                "error_public": public_error_text(exc),
                "error_digest": dequant_probe.sha_payload(str(exc)),
            }
        )
        blockers.append("glm52_full_prefix_stage_decode_failed")
    if ready:
        blockers.extend(
            [
                "glm52_full_prefix_stage_decode_uses_small_sequence_probe",
                "glm52_full_prefix_stage_decode_is_not_kaggle_runtime",
                "glm52_full_prefix_stage_decode_is_not_same_request",
            ]
        )
    blockers.extend(["glm52_stage_decode_not_verified", "glm52_same_request_decode_not_verified"])
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "glm52_full_prefix_stage_decode_probe_ready": ready,
        "model_id": MODEL_ID,
        "model_repo": str(args.model_repo),
        **result,
        "full_prefix_stage_hidden_verified": ready,
        "multi_layer_stage_hidden_verified": ready,
        "stage_handoff_only_verified": ready and result.get("lm_head_required") is False,
        "stage_hidden_to_lm_head_verified": ready and result.get("lm_head_required") is not False,
        "lm_head_streamed_full_vocab": ready and result.get("lm_head_required") is not False,
        "stage_hidden_lm_head_token_selection_verified": ready and result.get("lm_head_required") is not False,
        "partial_full_prefix_token_hash_verified": ready and result.get("lm_head_required") is not False,
        "full_model_hidden_verified": False,
        "generated_token_verified": False,
        "stage_decode_verified": False,
        "same_request_decode_verified": False,
        "live_kaggle_runtime_verified": False,
        "errors": errors,
        "blockers": sorted(set(blockers)),
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
        "public_artifact_safe": True,
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["glm52_full_prefix_stage_decode_probe_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"] = sorted(set([*blockers, "public_redaction_scan_failed"]))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--layer-start", type=int, default=6)
    parser.add_argument("--layer-end", type=int, default=8)
    parser.add_argument("--prefill-length", type=int, default=2)
    parser.add_argument("--dsa-mask-topk", type=int, default=2)
    parser.add_argument("--executed-expert-count", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--row-block-size", type=int, default=2048)
    parser.add_argument("--max-header-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-tensor-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-block-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--hf-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--input-hidden-b64", default="")
    parser.add_argument("--input-hidden-shape-json", default="")
    parser.add_argument("--input-hidden-dtype", default="")
    parser.add_argument("--output-activation-path", default="")
    parser.add_argument("--skip-lm-head", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.layer_end <= args.layer_start:
        raise SystemExit("--layer-end must be greater than --layer-start")
    if args.prefill_length <= 0:
        raise SystemExit("--prefill-length must be positive")
    if args.dsa_mask_topk <= 0:
        raise SystemExit("--dsa-mask-topk must be positive")
    if args.executed_expert_count <= 0:
        raise SystemExit("--executed-expert-count must be positive")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_report(args)
    path = output_dir / "glm52_full_prefix_stage_decode_probe.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"Full-prefix stage hidden verified: {report.get('full_prefix_stage_hidden_verified')}")
    return 0 if report.get("public_artifact_safe") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
