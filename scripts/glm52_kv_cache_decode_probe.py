#!/usr/bin/env python3
"""Run a public-safe GLM 5.2 KV-cache decode attention probe."""

from __future__ import annotations

import argparse
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
from scripts import glm52_attention_single_token_probe as attention_probe  # noqa: E402
from scripts import glm52_dsa_indexer_probe as dsa_probe  # noqa: E402
from scripts import glm52_pack_quantized_dequant_probe as dequant_probe  # noqa: E402


SCHEMA = "glm52_kv_cache_decode_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-kv-cache-decode-probe"
DEFAULT_MODEL_REPO = dequant_probe.DEFAULT_MODEL_REPO
MODEL_ID = dequant_probe.MODEL_ID
SENSITIVE_FRAGMENTS = dequant_probe.SENSITIVE_FRAGMENTS + (
    '"query_states":',
    '"key_cache":',
    '"value_cache":',
    '"attention_scores":',
    '"attention_weights":',
    '"attention_output":',
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


def dequantized_pack_weight(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    _, _, weight = dequant_probe.dequantize_group_slice(
        packed=tensors["weight_packed"],
        scale=tensors["weight_scale"],
        zero_point=tensors["weight_zero_point"],
        weight_shape=tensors["weight_shape"],
        row_count=int(tensors["weight_shape"][0].item()),
        group_count=int(tensors["weight_scale"].shape[1]),
    )
    return weight.to(torch.float32)


def pack_matrix(weight: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or hidden.ndim != 2 or int(weight.shape[1]) != int(hidden.shape[1]):
        raise RuntimeError("pack_matrix_shape_mismatch")
    return torch.matmul(hidden.to(torch.float32), weight.to(torch.float32).T)


def build_hidden_sequence(seq_len: int, hidden_size: int) -> torch.Tensor:
    base = torch.linspace(-0.05, 0.05, steps=int(hidden_size), dtype=torch.float32)
    return torch.stack([base + (position * 0.001) for position in range(int(seq_len))], dim=0)


def run_kv_cache_decode(args: argparse.Namespace) -> dict[str, Any]:
    config = dequant_probe.fetch_hf_json(args.model_repo, "config.json", timeout_seconds=float(args.hf_timeout_seconds))
    layer = int(args.layer_id)
    hidden_size = _int(config.get("hidden_size"))
    if hidden_size <= 0:
        raise RuntimeError("hidden_size_missing")
    prefill_len = int(args.prefill_length)
    total_len = prefill_len + 1
    eps = float(config.get("rms_norm_eps") or 1e-6)
    heads = _int(config.get("num_attention_heads"))
    q_lora_rank = _int(config.get("q_lora_rank"))
    kv_lora_rank = _int(config.get("kv_lora_rank"))
    qk_head_dim = _int(config.get("qk_head_dim"))
    qk_nope_head_dim = _int(config.get("qk_nope_head_dim"))
    qk_rope_head_dim = _int(config.get("qk_rope_head_dim"))
    v_head_dim = _int(config.get("v_head_dim"))
    theta = float((config.get("rope_parameters") or {}).get("rope_theta") or 10000.0)

    hidden = build_hidden_sequence(total_len, hidden_size)
    input_norm_weight = projection_probe.router_probe.load_dense_tensor(args, f"model.layers.{layer}.input_layernorm.weight")
    hidden_norm = dsa_probe.rms_norm_rows(hidden, input_norm_weight, eps)

    q_a_weight = projection_probe.router_probe.load_dense_tensor(args, f"model.layers.{layer}.self_attn.q_a_proj.weight")
    q_a = dsa_probe.dense_matrix(q_a_weight, hidden_norm)
    q_a_norm_weight = projection_probe.router_probe.load_dense_tensor(args, f"model.layers.{layer}.self_attn.q_a_layernorm.weight")
    q_a_norm = dsa_probe.rms_norm_rows(q_a, q_a_norm_weight, eps)
    q_b_weight = dequantized_pack_weight(projection_probe.pack_projection_tensors(args, f"model.layers.{layer}.self_attn.q_b_proj"))
    q_b = pack_matrix(q_b_weight, q_a_norm)
    query = q_b.reshape(total_len, heads, qk_head_dim)
    q_nope, q_pe = torch.split(query, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)

    kv_a_weight = projection_probe.router_probe.load_dense_tensor(args, f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight")
    kv_a = dsa_probe.dense_matrix(kv_a_weight, hidden_norm)
    k_compressed, k_pe = torch.split(kv_a, [kv_lora_rank, qk_rope_head_dim], dim=-1)
    kv_a_norm_weight = projection_probe.router_probe.load_dense_tensor(args, f"model.layers.{layer}.self_attn.kv_a_layernorm.weight")
    k_compressed_norm = dsa_probe.rms_norm_rows(k_compressed, kv_a_norm_weight, eps)
    kv_b_weight = dequantized_pack_weight(projection_probe.pack_projection_tensors(args, f"model.layers.{layer}.self_attn.kv_b_proj"))
    kv_b = pack_matrix(kv_b_weight, k_compressed_norm)
    kv_expanded = kv_b.reshape(total_len, heads, qk_nope_head_dim + v_head_dim)
    k_nope, value_states = torch.split(kv_expanded, [qk_nope_head_dim, v_head_dim], dim=-1)

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
    attention_scores = torch.einsum("hd,thd->ht", decode_query.float(), updated_key_cache.float()) * (qk_head_dim ** -0.5)
    attention_weights = torch.softmax(attention_scores, dim=-1, dtype=torch.float32)
    head_output = torch.einsum("ht,thd->hd", attention_weights, updated_value_cache.float())
    flattened = head_output.reshape(heads * v_head_dim)
    o_proj_weight = dequantized_pack_weight(projection_probe.pack_projection_tensors(args, f"model.layers.{layer}.self_attn.o_proj"))
    o_proj_output = torch.matmul(o_proj_weight, flattened.to(torch.float32))
    return {
        "model_type": str(config.get("model_type") or ""),
        "hidden_size": hidden_size,
        "num_attention_heads": heads,
        "q_lora_rank": q_lora_rank,
        "kv_lora_rank": kv_lora_rank,
        "qk_head_dim": qk_head_dim,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "prefill_length": prefill_len,
        "decode_length": 1,
        "updated_cache_length": total_len,
        "query_states_shape": [int(item) for item in query_states.shape],
        "key_states_shape": [int(item) for item in key_states.shape],
        "value_states_shape": [int(item) for item in value_states.shape],
        "prefill_key_cache_shape": [int(item) for item in prefill_key_cache.shape],
        "prefill_value_cache_shape": [int(item) for item in prefill_value_cache.shape],
        "updated_key_cache_shape": [int(item) for item in updated_key_cache.shape],
        "updated_value_cache_shape": [int(item) for item in updated_value_cache.shape],
        "decode_query_shape": [int(item) for item in decode_query.shape],
        "attention_scores_shape": [int(item) for item in attention_scores.shape],
        "attention_weights_shape": [int(item) for item in attention_weights.shape],
        "head_output_shape": [int(item) for item in head_output.shape],
        "attention_flattened_shape": [int(item) for item in flattened.shape],
        "o_proj_weight_shape": [int(item) for item in o_proj_weight.shape],
        "o_proj_output_shape": [int(item) for item in o_proj_output.shape],
        "prefill_key_cache_hash": dequant_probe.sha_tensor(prefill_key_cache),
        "prefill_value_cache_hash": dequant_probe.sha_tensor(prefill_value_cache),
        "updated_key_cache_hash": dequant_probe.sha_tensor(updated_key_cache),
        "updated_value_cache_hash": dequant_probe.sha_tensor(updated_value_cache),
        "decode_query_hash": dequant_probe.sha_tensor(decode_query),
        "attention_scores_hash": dequant_probe.sha_tensor(attention_scores),
        "attention_weights_hash": dequant_probe.sha_tensor(attention_weights),
        "head_output_hash": dequant_probe.sha_tensor(head_output),
        "o_proj_output_hash": dequant_probe.sha_tensor(o_proj_output),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    blockers: list[str] = []
    errors: list[dict[str, Any]] = []
    decode: dict[str, Any] = {}
    ready = False
    try:
        decode = run_kv_cache_decode(args)
        heads = _int(decode.get("num_attention_heads"))
        qk = _int(decode.get("qk_head_dim"))
        value = _int(decode.get("v_head_dim"))
        prefill_len = _int(decode.get("prefill_length"))
        updated_len = _int(decode.get("updated_cache_length"))
        hidden_size = _int(decode.get("hidden_size"))
        ready = (
            decode.get("model_type") == "glm_moe_dsa"
            and _list(decode.get("prefill_key_cache_shape")) == [prefill_len, heads, qk]
            and _list(decode.get("prefill_value_cache_shape")) == [prefill_len, heads, value]
            and _list(decode.get("updated_key_cache_shape")) == [updated_len, heads, qk]
            and _list(decode.get("updated_value_cache_shape")) == [updated_len, heads, value]
            and _list(decode.get("attention_scores_shape")) == [heads, updated_len]
            and _list(decode.get("attention_weights_shape")) == [heads, updated_len]
            and _list(decode.get("o_proj_output_shape")) == [hidden_size]
        )
    except Exception as exc:
        errors.append({"phase": "kv_cache_decode", "error_type": type(exc).__name__, "error_digest": dequant_probe.sha_payload(str(exc))})
        blockers.append("glm52_kv_cache_decode_failed")
    if ready:
        blockers.extend(
            [
                "glm52_kv_cache_decode_is_not_dsa_masked_attention",
                "glm52_kv_cache_decode_is_not_transformer_block",
                "glm52_kv_cache_decode_is_not_stage_decode",
                "glm52_kv_cache_decode_missing_mlp_residual",
                "glm52_kv_cache_decode_missing_lm_head",
            ]
        )
    blockers.append("glm52_stage_decode_not_verified")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "glm52_kv_cache_decode_probe_ready": ready,
        "model_id": MODEL_ID,
        "model_repo": str(args.model_repo),
        "model_type": str(decode.get("model_type") or ""),
        "layer_id": int(args.layer_id),
        **decode,
        "kv_cache_prefill_verified": ready,
        "kv_cache_update_verified": ready,
        "kv_cache_decode_attention_verified": ready,
        "o_proj_verified": ready,
        "stage_decode_verified": False,
        "generated_token_verified": False,
        "errors": errors,
        "blockers": sorted(set(blockers)),
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
        "public_artifact_safe": True,
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["glm52_kv_cache_decode_probe_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"] = sorted(set([*blockers, "public_redaction_scan_failed"]))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--layer-id", type=int, default=3)
    parser.add_argument("--prefill-length", type=int, default=4)
    parser.add_argument("--max-header-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-tensor-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--hf-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.prefill_length <= 0:
        raise SystemExit("--prefill-length must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_report(args)
    path = output_dir / "glm52_kv_cache_decode_probe.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"KV-cache decode verified: {report.get('kv_cache_decode_attention_verified')}")
    return 0 if report.get("public_artifact_safe") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
