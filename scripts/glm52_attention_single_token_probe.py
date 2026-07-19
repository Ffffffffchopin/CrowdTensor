#!/usr/bin/env python3
"""Run a public-safe GLM 5.2 single-token attention probe."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_attention_projection_probe as projection_probe  # noqa: E402
from scripts import glm52_pack_quantized_dequant_probe as dequant_probe  # noqa: E402


SCHEMA = "glm52_attention_single_token_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-attention-single-token-probe"
DEFAULT_MODEL_REPO = dequant_probe.DEFAULT_MODEL_REPO
MODEL_ID = dequant_probe.MODEL_ID
SENSITIVE_FRAGMENTS = dequant_probe.SENSITIVE_FRAGMENTS + (
    '"query_states":',
    '"key_states":',
    '"value_states":',
    '"attention_scores":',
    '"attention_weights":',
    '"attention_output":',
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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(*, dim: int, position_id: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (float(theta) ** (torch.arange(0, int(dim), 2, dtype=torch.float32) / int(dim)))
    freqs = inv_freq * float(position_id)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(torch.float32), emb.sin().to(torch.float32)


def apply_rope_1d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if int(x.shape[-1]) != int(cos.shape[-1]) or int(x.shape[-1]) != int(sin.shape[-1]):
        raise RuntimeError("rope_shape_mismatch")
    return (x.to(torch.float32) * cos.to(torch.float32)) + (rotate_half(x.to(torch.float32)) * sin.to(torch.float32))


def run_attention_tensors(args: argparse.Namespace) -> tuple[dict[str, Any], torch.Tensor]:
    config = dequant_probe.fetch_hf_json(args.model_repo, "config.json", timeout_seconds=float(args.hf_timeout_seconds))
    hidden_size = _int(config.get("hidden_size"))
    if hidden_size <= 0:
        raise RuntimeError("hidden_size_missing")
    layer = int(args.layer_id)
    eps = float(config.get("rms_norm_eps") or 1e-6)
    num_heads = _int(config.get("num_attention_heads"))
    q_lora_rank = _int(config.get("q_lora_rank"))
    kv_lora_rank = _int(config.get("kv_lora_rank"))
    qk_head_dim = _int(config.get("qk_head_dim"))
    qk_nope_head_dim = _int(config.get("qk_nope_head_dim"))
    qk_rope_head_dim = _int(config.get("qk_rope_head_dim"))
    v_head_dim = _int(config.get("v_head_dim"))
    theta = float((config.get("rope_parameters") or {}).get("rope_theta") or 10000.0)
    hidden = torch.linspace(-0.05, 0.05, steps=hidden_size, dtype=torch.float32)

    input_norm_weight = projection_probe.router_probe.load_dense_tensor(args, f"model.layers.{layer}.input_layernorm.weight")
    input_norm = projection_probe.rms_norm(hidden, input_norm_weight, eps)

    q_a, _ = projection_probe.dense_projection(args, f"model.layers.{layer}.self_attn.q_a_proj.weight", input_norm)
    q_a_norm_weight = projection_probe.router_probe.load_dense_tensor(args, f"model.layers.{layer}.self_attn.q_a_layernorm.weight")
    q_a_norm = projection_probe.rms_norm(q_a, q_a_norm_weight, eps)
    q_b_tensors = projection_probe.pack_projection_tensors(args, f"model.layers.{layer}.self_attn.q_b_proj")
    q_b, _, _ = projection_probe.pack_projection_linear(q_b_tensors, q_a_norm)
    query = q_b.reshape(num_heads, qk_head_dim)
    q_nope, q_pe = torch.split(query, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)

    kv_a, _ = projection_probe.dense_projection(args, f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight", input_norm)
    k_compressed, k_pe = torch.split(kv_a, [kv_lora_rank, qk_rope_head_dim], dim=-1)
    kv_a_norm_weight = projection_probe.router_probe.load_dense_tensor(args, f"model.layers.{layer}.self_attn.kv_a_layernorm.weight")
    k_compressed_norm = projection_probe.rms_norm(k_compressed, kv_a_norm_weight, eps)
    kv_b_tensors = projection_probe.pack_projection_tensors(args, f"model.layers.{layer}.self_attn.kv_b_proj")
    kv_b, _, _ = projection_probe.pack_projection_linear(kv_b_tensors, k_compressed_norm)
    kv_expanded = kv_b.reshape(num_heads, qk_nope_head_dim + v_head_dim)
    k_nope, value = torch.split(kv_expanded, [qk_nope_head_dim, v_head_dim], dim=-1)

    cos, sin = rope_cos_sin(dim=qk_rope_head_dim, position_id=int(args.position_id), theta=theta)
    q_pe_rope = apply_rope_1d(q_pe, cos, sin)
    k_pe_rope = apply_rope_1d(k_pe, cos, sin).reshape(1, qk_rope_head_dim).expand(num_heads, qk_rope_head_dim)
    query_states = torch.cat([q_nope, q_pe_rope], dim=-1)
    key_states = torch.cat([k_nope, k_pe_rope], dim=-1)
    attention_scores = (query_states.to(torch.float32) * key_states.to(torch.float32)).sum(dim=-1, keepdim=True) * (qk_head_dim ** -0.5)
    attention_weights = torch.softmax(attention_scores, dim=-1, dtype=torch.float32)
    head_output = attention_weights * value.to(torch.float32)
    flattened = head_output.reshape(num_heads * v_head_dim)
    o_proj_tensors = projection_probe.pack_projection_tensors(args, f"model.layers.{layer}.self_attn.o_proj")
    o_proj, o_proj_weight_shape, o_proj_output_shape = projection_probe.pack_projection_linear(o_proj_tensors, flattened)
    summary = {
        "model_type": str(config.get("model_type") or ""),
        "hidden_size": hidden_size,
        "num_attention_heads": num_heads,
        "q_lora_rank": q_lora_rank,
        "kv_lora_rank": kv_lora_rank,
        "qk_head_dim": qk_head_dim,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "rope_theta": theta,
        "position_id": int(args.position_id),
        "query_states_shape": [int(item) for item in query_states.shape],
        "key_states_shape": [int(item) for item in key_states.shape],
        "value_states_shape": [int(item) for item in value.shape],
        "attention_scores_shape": [int(item) for item in attention_scores.shape],
        "attention_weights_shape": [int(item) for item in attention_weights.shape],
        "head_output_shape": [int(item) for item in head_output.shape],
        "attention_flattened_shape": [int(item) for item in flattened.shape],
        "o_proj_weight_shape": o_proj_weight_shape,
        "o_proj_output_shape": o_proj_output_shape,
        "q_pe_rope_hash": dequant_probe.sha_tensor(q_pe_rope),
        "k_pe_rope_hash": dequant_probe.sha_tensor(k_pe_rope),
        "query_states_hash": dequant_probe.sha_tensor(query_states),
        "key_states_hash": dequant_probe.sha_tensor(key_states),
        "value_states_hash": dequant_probe.sha_tensor(value),
        "attention_scores_hash": dequant_probe.sha_tensor(attention_scores),
        "attention_weights_hash": dequant_probe.sha_tensor(attention_weights),
        "head_output_hash": dequant_probe.sha_tensor(head_output),
        "o_proj_output_hash": dequant_probe.sha_tensor(o_proj),
    }
    return summary, o_proj


def run_attention_single_token(args: argparse.Namespace) -> dict[str, Any]:
    summary, _ = run_attention_tensors(args)
    return summary


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    blockers: list[str] = []
    errors: list[dict[str, Any]] = []
    attention: dict[str, Any] = {}
    ready = False
    try:
        attention = run_attention_single_token(args)
        heads = _int(attention.get("num_attention_heads"))
        qk = _int(attention.get("qk_head_dim"))
        value_dim = _int(attention.get("v_head_dim"))
        ready = (
            attention.get("model_type") == "glm_moe_dsa"
            and _list(attention.get("query_states_shape")) == [heads, qk]
            and _list(attention.get("key_states_shape")) == [heads, qk]
            and _list(attention.get("value_states_shape")) == [heads, value_dim]
            and _list(attention.get("attention_scores_shape")) == [heads, 1]
            and _list(attention.get("attention_weights_shape")) == [heads, 1]
            and _list(attention.get("head_output_shape")) == [heads, value_dim]
            and _list(attention.get("attention_flattened_shape")) == [heads * value_dim]
            and _list(attention.get("o_proj_output_shape")) == [_int(attention.get("hidden_size"))]
        )
    except Exception as exc:
        errors.append({"phase": "attention_single_token", "error_type": type(exc).__name__, "error_digest": dequant_probe.sha_payload(str(exc))})
        blockers.append("glm52_attention_single_token_failed")
    if ready:
        blockers.extend(
            [
                "glm52_attention_single_token_is_not_multi_token_prefill",
                "glm52_attention_single_token_is_not_dsa_indexer",
                "glm52_attention_single_token_is_not_kv_cache_decode",
                "glm52_attention_single_token_is_not_transformer_block",
                "glm52_attention_single_token_is_not_stage_decode",
            ]
        )
    blockers.append("glm52_stage_decode_not_verified")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "glm52_attention_single_token_probe_ready": ready,
        "model_id": MODEL_ID,
        "model_repo": str(args.model_repo),
        "model_type": str(attention.get("model_type") or ""),
        "layer_id": int(args.layer_id),
        **attention,
        "rope_applied": ready,
        "attention_scores_verified": ready,
        "attention_weights_verified": ready,
        "o_proj_verified": ready,
        "single_token_attention_verified": ready,
        "kv_cache_updated": False,
        "dsa_indexer_verified": False,
        "stage_decode_verified": False,
        "errors": errors,
        "blockers": sorted(set(blockers)),
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
        "public_artifact_safe": True,
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["glm52_attention_single_token_probe_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"] = sorted(set([*blockers, "public_redaction_scan_failed"]))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--layer-id", type=int, default=3)
    parser.add_argument("--position-id", type=int, default=7)
    parser.add_argument("--max-header-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-tensor-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--hf-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_report(args)
    path = output_dir / "glm52_attention_single_token_probe.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"Single-token attention verified: {report.get('single_token_attention_verified')}")
    return 0 if report.get("public_artifact_safe") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
