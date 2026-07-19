#!/usr/bin/env python3
"""Run a public-safe GLM 5.2 DSA indexer probe."""

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
from scripts import glm52_pack_quantized_dequant_probe as dequant_probe  # noqa: E402


SCHEMA = "glm52_dsa_indexer_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-dsa-indexer-probe"
DEFAULT_MODEL_REPO = dequant_probe.DEFAULT_MODEL_REPO
MODEL_ID = dequant_probe.MODEL_ID
SENSITIVE_FRAGMENTS = dequant_probe.SENSITIVE_FRAGMENTS + (
    '"index_scores":',
    '"topk_indices":',
    '"indexer_queries":',
    '"indexer_keys":',
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


def dense_matrix(weight: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or hidden.ndim != 2 or int(weight.shape[1]) != int(hidden.shape[1]):
        raise RuntimeError("dense_matrix_shape_mismatch")
    return torch.matmul(hidden.to(torch.float32), weight.to(torch.float32).T)


def rms_norm_rows(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if hidden.ndim != 2 or weight.ndim != 1 or int(hidden.shape[-1]) != int(weight.shape[0]):
        raise RuntimeError("rms_norm_rows_shape_mismatch")
    value = hidden.to(torch.float32)
    variance = value.pow(2).mean(dim=-1, keepdim=True)
    return weight.to(torch.float32) * value * torch.rsqrt(variance + float(eps))


def layer_norm_rows(hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if hidden.ndim != 2 or weight.ndim != 1 or bias.ndim != 1 or int(hidden.shape[-1]) != int(weight.shape[0]):
        raise RuntimeError("layer_norm_rows_shape_mismatch")
    value = hidden.to(torch.float32)
    mean = value.mean(dim=-1, keepdim=True)
    variance = (value - mean).pow(2).mean(dim=-1, keepdim=True)
    return (value - mean) * torch.rsqrt(variance + float(eps)) * weight.to(torch.float32) + bias.to(torch.float32)


def apply_rope_sequence(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3 or cos.ndim != 2 or sin.ndim != 2:
        raise RuntimeError("rope_sequence_rank_mismatch")
    if int(x.shape[0]) != int(cos.shape[0]) or int(x.shape[-1]) != int(cos.shape[-1]):
        raise RuntimeError("rope_sequence_shape_mismatch")
    return (x.to(torch.float32) * cos[:, None, :].to(torch.float32)) + (
        attention_probe.rotate_half(x.to(torch.float32)) * sin[:, None, :].to(torch.float32)
    )


def build_position_cos_sin(seq_len: int, rope_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    cos_items: list[torch.Tensor] = []
    sin_items: list[torch.Tensor] = []
    for position in range(int(seq_len)):
        cos, sin = attention_probe.rope_cos_sin(dim=int(rope_dim), position_id=position, theta=float(theta))
        cos_items.append(cos)
        sin_items.append(sin)
    return torch.stack(cos_items, dim=0), torch.stack(sin_items, dim=0)


def run_dsa_indexer(args: argparse.Namespace) -> dict[str, Any]:
    config = dequant_probe.fetch_hf_json(args.model_repo, "config.json", timeout_seconds=float(args.hf_timeout_seconds))
    layer = int(args.layer_id)
    indexer_types = _list(config.get("indexer_types"))
    if not indexer_types or str(indexer_types[layer]) != "full":
        raise RuntimeError("selected_layer_is_not_full_indexer")
    hidden_size = _int(config.get("hidden_size"))
    q_lora_rank = _int(config.get("q_lora_rank"))
    index_heads = _int(config.get("index_n_heads"))
    index_head_dim = _int(config.get("index_head_dim"))
    rope_dim = _int(config.get("qk_rope_head_dim"))
    index_topk = _int(config.get("index_topk"))
    theta = float((config.get("rope_parameters") or {}).get("rope_theta") or 10000.0)
    eps = float(config.get("rms_norm_eps") or 1e-6)
    seq_len = int(args.sequence_length)
    base = torch.linspace(-0.05, 0.05, steps=hidden_size, dtype=torch.float32)
    hidden = torch.stack([base + (position * 0.001) for position in range(seq_len)], dim=0)

    input_norm_weight = projection_probe.router_probe.load_dense_tensor(args, f"model.layers.{layer}.input_layernorm.weight")
    hidden_norm = rms_norm_rows(hidden, input_norm_weight, eps)
    q_a_weight = projection_probe.router_probe.load_dense_tensor(args, f"model.layers.{layer}.self_attn.q_a_proj.weight")
    q_a = dense_matrix(q_a_weight, hidden_norm)
    q_a_norm_weight = projection_probe.router_probe.load_dense_tensor(args, f"model.layers.{layer}.self_attn.q_a_layernorm.weight")
    q_resid = rms_norm_rows(q_a, q_a_norm_weight, eps)

    prefix = f"model.layers.{layer}.self_attn.indexer"
    wq_b = projection_probe.router_probe.load_dense_tensor(args, f"{prefix}.wq_b.weight")
    wk = projection_probe.router_probe.load_dense_tensor(args, f"{prefix}.wk.weight")
    k_norm_weight = projection_probe.router_probe.load_dense_tensor(args, f"{prefix}.k_norm.weight")
    k_norm_bias = projection_probe.router_probe.load_dense_tensor(args, f"{prefix}.k_norm.bias")
    weights_proj = projection_probe.router_probe.load_dense_tensor(args, f"{prefix}.weights_proj.weight")

    q = dense_matrix(wq_b, q_resid).reshape(seq_len, index_heads, index_head_dim)
    q_pe, q_nope = torch.split(q, [rope_dim, index_head_dim - rope_dim], dim=-1)
    cos, sin = build_position_cos_sin(seq_len, rope_dim, theta)
    q_pe = apply_rope_sequence(q_pe, cos, sin)
    q_index = torch.cat([q_pe, q_nope], dim=-1)

    k_raw = dense_matrix(wk, hidden_norm)
    k_normed = layer_norm_rows(k_raw, k_norm_weight, k_norm_bias, eps=1e-6)
    k_pe, k_nope = torch.split(k_normed, [rope_dim, index_head_dim - rope_dim], dim=-1)
    k_pe = apply_rope_sequence(k_pe[:, None, :], cos, sin).squeeze(1)
    k_index = torch.cat([k_pe, k_nope], dim=-1)

    head_weights = dense_matrix(weights_proj, hidden_norm) * (index_heads ** -0.5)
    scores = torch.einsum("shd,td->sht", q_index.float(), k_index.float()) * (index_head_dim ** -0.5)
    scores = torch.relu(scores)
    index_scores = torch.einsum("sht,sh->st", scores, head_weights.float())
    topk = min(index_topk, seq_len)
    topk_indices = torch.topk(index_scores, k=topk, dim=-1).indices.to(torch.int64)

    return {
        "model_type": str(config.get("model_type") or ""),
        "hidden_size": hidden_size,
        "layer_indexer_type": str(indexer_types[layer]),
        "sequence_length": seq_len,
        "q_lora_rank": q_lora_rank,
        "index_n_heads": index_heads,
        "index_head_dim": index_head_dim,
        "qk_rope_head_dim": rope_dim,
        "index_topk_config": index_topk,
        "effective_topk": topk,
        "hidden_norm_shape": [int(item) for item in hidden_norm.shape],
        "q_resid_shape": [int(item) for item in q_resid.shape],
        "indexer_query_shape": [int(item) for item in q_index.shape],
        "indexer_key_shape": [int(item) for item in k_index.shape],
        "head_weights_shape": [int(item) for item in head_weights.shape],
        "index_score_shape": [int(item) for item in index_scores.shape],
        "topk_indices_shape": [int(item) for item in topk_indices.shape],
        "hidden_norm_hash": dequant_probe.sha_tensor(hidden_norm),
        "q_resid_hash": dequant_probe.sha_tensor(q_resid),
        "indexer_query_hash": dequant_probe.sha_tensor(q_index),
        "indexer_key_hash": dequant_probe.sha_tensor(k_index),
        "head_weights_hash": dequant_probe.sha_tensor(head_weights),
        "index_score_hash": dequant_probe.sha_tensor(index_scores),
        "topk_indices_hash": dequant_probe.sha_tensor(topk_indices),
        "wq_b_weight_shape": [int(item) for item in wq_b.shape],
        "wk_weight_shape": [int(item) for item in wk.shape],
        "weights_proj_shape": [int(item) for item in weights_proj.shape],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    blockers: list[str] = []
    errors: list[dict[str, Any]] = []
    indexer: dict[str, Any] = {}
    ready = False
    try:
        indexer = run_dsa_indexer(args)
        seq_len = _int(indexer.get("sequence_length"))
        heads = _int(indexer.get("index_n_heads"))
        head_dim = _int(indexer.get("index_head_dim"))
        ready = (
            indexer.get("model_type") == "glm_moe_dsa"
            and indexer.get("layer_indexer_type") == "full"
            and _list(indexer.get("indexer_query_shape")) == [seq_len, heads, head_dim]
            and _list(indexer.get("indexer_key_shape")) == [seq_len, head_dim]
            and _list(indexer.get("index_score_shape")) == [seq_len, seq_len]
            and _list(indexer.get("topk_indices_shape")) == [seq_len, _int(indexer.get("effective_topk"))]
        )
    except Exception as exc:
        errors.append({"phase": "dsa_indexer", "error_type": type(exc).__name__, "error_digest": dequant_probe.sha_payload(str(exc))})
        blockers.append("glm52_dsa_indexer_failed")
    if ready:
        blockers.extend(
            [
                "glm52_dsa_indexer_small_sequence_is_not_full_prefill",
                "glm52_dsa_indexer_is_not_kv_cache_decode",
                "glm52_dsa_indexer_is_not_attention_output",
                "glm52_dsa_indexer_is_not_stage_decode",
            ]
        )
    blockers.append("glm52_stage_decode_not_verified")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "glm52_dsa_indexer_probe_ready": ready,
        "model_id": MODEL_ID,
        "model_repo": str(args.model_repo),
        "model_type": str(indexer.get("model_type") or ""),
        "layer_id": int(args.layer_id),
        **indexer,
        "dsa_indexer_verified": ready,
        "dsa_topk_verified": ready,
        "indexer_cache_updated": False,
        "attention_output_verified": False,
        "stage_decode_verified": False,
        "errors": errors,
        "blockers": sorted(set(blockers)),
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
        "public_artifact_safe": True,
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["glm52_dsa_indexer_probe_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"] = sorted(set([*blockers, "public_redaction_scan_failed"]))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--layer-id", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--max-header-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-tensor-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--hf-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.sequence_length <= 0:
        raise SystemExit("--sequence-length must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_report(args)
    path = output_dir / "glm52_dsa_indexer_probe.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"DSA indexer verified: {report.get('dsa_indexer_verified')}")
    return 0 if report.get("public_artifact_safe") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
