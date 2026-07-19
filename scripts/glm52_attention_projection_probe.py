#!/usr/bin/env python3
"""Run a public-safe GLM 5.2 attention projection-path probe."""

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

from scripts import glm52_pack_quantized_dequant_probe as dequant_probe  # noqa: E402
from scripts import glm52_pack_quantized_moe_mlp_probe as moe_probe  # noqa: E402
from scripts import glm52_pack_quantized_router_gather_probe as router_probe  # noqa: E402


SCHEMA = "glm52_attention_projection_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-attention-projection-probe"
DEFAULT_MODEL_REPO = dequant_probe.DEFAULT_MODEL_REPO
MODEL_ID = dequant_probe.MODEL_ID
PACK_FIELDS = dequant_probe.PACK_FIELDS
SENSITIVE_FRAGMENTS = dequant_probe.SENSITIVE_FRAGMENTS + (
    '"query_states":',
    '"key_states":',
    '"value_states":',
    '"attention_scores":',
    '"attention_output":',
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


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if hidden.ndim != 1 or weight.ndim != 1 or int(hidden.shape[0]) != int(weight.shape[0]):
        raise RuntimeError("rms_norm_shape_mismatch")
    value = hidden.to(torch.float32)
    variance = value.pow(2).mean(dim=-1, keepdim=True)
    return weight.to(torch.float32) * value * torch.rsqrt(variance + float(eps))


def dense_projection(args: argparse.Namespace, key: str, input_vec: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    weight = router_probe.load_dense_tensor(args, key)
    output = moe_probe.dense_linear(weight, input_vec)
    return output, [int(item) for item in weight.shape]


def pack_projection_tensors(args: argparse.Namespace, prefix: str) -> dict[str, torch.Tensor]:
    index = dequant_probe.fetch_hf_json(args.model_repo, "model.safetensors.index.json", timeout_seconds=float(args.hf_timeout_seconds))
    weight_map = _dict(index.get("weight_map"))
    tensors: dict[str, torch.Tensor] = {}
    headers_by_file: dict[str, tuple[int, dict[str, Any]]] = {}
    for field in PACK_FIELDS:
        key = f"{prefix}.{field}"
        filename = str(weight_map.get(key) or "")
        if not filename:
            raise RuntimeError(f"pack_projection_key_missing:{key}")
        if filename not in headers_by_file:
            headers_by_file[filename] = dequant_probe.load_safetensors_header_with_len(
                args.model_repo,
                filename,
                timeout_seconds=float(args.hf_timeout_seconds),
                max_header_bytes=int(args.max_header_bytes),
            )
        header_len, header = headers_by_file[filename]
        tensors[field] = dequant_probe.load_tensor(args.model_repo, filename, header_len, _dict(header.get(key)), args)
    return tensors


def pack_projection_linear(tensors: dict[str, torch.Tensor], input_vec: torch.Tensor) -> tuple[torch.Tensor, list[int], list[int]]:
    _, _, weight = dequant_probe.dequantize_group_slice(
        packed=tensors["weight_packed"],
        scale=tensors["weight_scale"],
        zero_point=tensors["weight_zero_point"],
        weight_shape=tensors["weight_shape"],
        row_count=int(tensors["weight_shape"][0].item()),
        group_count=int(tensors["weight_scale"].shape[1]),
    )
    if int(weight.shape[1]) != int(input_vec.shape[0]):
        raise RuntimeError(f"pack_projection_width_mismatch:{int(weight.shape[1])}:{int(input_vec.shape[0])}")
    output = torch.matmul(weight.to(torch.float32), input_vec.to(torch.float32))
    return output, [int(item) for item in weight.shape], [int(item) for item in output.shape]


def run_attention_projection(args: argparse.Namespace) -> dict[str, Any]:
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
    hidden = torch.linspace(-0.05, 0.05, steps=hidden_size, dtype=torch.float32)

    input_norm_weight = router_probe.load_dense_tensor(args, f"model.layers.{layer}.input_layernorm.weight")
    input_norm = rms_norm(hidden, input_norm_weight, eps)

    q_a, q_a_weight_shape = dense_projection(args, f"model.layers.{layer}.self_attn.q_a_proj.weight", input_norm)
    q_a_norm_weight = router_probe.load_dense_tensor(args, f"model.layers.{layer}.self_attn.q_a_layernorm.weight")
    q_a_norm = rms_norm(q_a, q_a_norm_weight, eps)
    q_b_tensors = pack_projection_tensors(args, f"model.layers.{layer}.self_attn.q_b_proj")
    q_b, q_b_weight_shape, q_b_output_shape = pack_projection_linear(q_b_tensors, q_a_norm)
    query = q_b.reshape(num_heads, qk_head_dim)
    q_nope, q_pe = torch.split(query, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)

    kv_a, kv_a_weight_shape = dense_projection(args, f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight", input_norm)
    k_compressed, k_pe = torch.split(kv_a, [kv_lora_rank, qk_rope_head_dim], dim=-1)
    kv_a_norm_weight = router_probe.load_dense_tensor(args, f"model.layers.{layer}.self_attn.kv_a_layernorm.weight")
    k_compressed_norm = rms_norm(k_compressed, kv_a_norm_weight, eps)
    kv_b_tensors = pack_projection_tensors(args, f"model.layers.{layer}.self_attn.kv_b_proj")
    kv_b, kv_b_weight_shape, kv_b_output_shape = pack_projection_linear(kv_b_tensors, k_compressed_norm)
    kv_expanded = kv_b.reshape(num_heads, qk_nope_head_dim + v_head_dim)
    k_nope, value = torch.split(kv_expanded, [qk_nope_head_dim, v_head_dim], dim=-1)

    return {
        "model_type": str(config.get("model_type") or ""),
        "hidden_size": hidden_size,
        "num_attention_heads": num_heads,
        "q_lora_rank": q_lora_rank,
        "kv_lora_rank": kv_lora_rank,
        "qk_head_dim": qk_head_dim,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "input_norm_shape": [int(item) for item in input_norm.shape],
        "input_norm_hash": dequant_probe.sha_tensor(input_norm),
        "q_a_weight_shape": q_a_weight_shape,
        "q_a_output_shape": [int(item) for item in q_a.shape],
        "q_a_output_hash": dequant_probe.sha_tensor(q_a),
        "q_a_norm_shape": [int(item) for item in q_a_norm.shape],
        "q_a_norm_hash": dequant_probe.sha_tensor(q_a_norm),
        "q_b_weight_shape": q_b_weight_shape,
        "q_b_output_shape": q_b_output_shape,
        "q_b_output_hash": dequant_probe.sha_tensor(q_b),
        "query_shape": [int(item) for item in query.shape],
        "q_nope_shape": [int(item) for item in q_nope.shape],
        "q_pe_shape": [int(item) for item in q_pe.shape],
        "q_nope_hash": dequant_probe.sha_tensor(q_nope),
        "q_pe_hash": dequant_probe.sha_tensor(q_pe),
        "kv_a_weight_shape": kv_a_weight_shape,
        "kv_a_output_shape": [int(item) for item in kv_a.shape],
        "kv_a_output_hash": dequant_probe.sha_tensor(kv_a),
        "k_compressed_shape": [int(item) for item in k_compressed.shape],
        "k_pe_shape": [int(item) for item in k_pe.shape],
        "k_compressed_norm_shape": [int(item) for item in k_compressed_norm.shape],
        "k_compressed_norm_hash": dequant_probe.sha_tensor(k_compressed_norm),
        "kv_b_weight_shape": kv_b_weight_shape,
        "kv_b_output_shape": kv_b_output_shape,
        "kv_b_output_hash": dequant_probe.sha_tensor(kv_b),
        "k_nope_shape": [int(item) for item in k_nope.shape],
        "value_shape": [int(item) for item in value.shape],
        "k_nope_hash": dequant_probe.sha_tensor(k_nope),
        "value_hash": dequant_probe.sha_tensor(value),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    blockers: list[str] = []
    errors: list[dict[str, Any]] = []
    attention: dict[str, Any] = {}
    ready = False
    try:
        attention = run_attention_projection(args)
        heads = _int(attention.get("num_attention_heads"))
        q_rank = _int(attention.get("q_lora_rank"))
        kv_rank = _int(attention.get("kv_lora_rank"))
        qk = _int(attention.get("qk_head_dim"))
        nope = _int(attention.get("qk_nope_head_dim"))
        rope = _int(attention.get("qk_rope_head_dim"))
        value_dim = _int(attention.get("v_head_dim"))
        ready = (
            attention.get("model_type") == "glm_moe_dsa"
            and _list(attention.get("input_norm_shape")) == [_int(attention.get("hidden_size"))]
            and _list(attention.get("q_a_output_shape")) == [q_rank]
            and _list(attention.get("q_a_norm_shape")) == [q_rank]
            and _list(attention.get("q_b_output_shape")) == [heads * qk]
            and _list(attention.get("query_shape")) == [heads, qk]
            and _list(attention.get("q_nope_shape")) == [heads, nope]
            and _list(attention.get("q_pe_shape")) == [heads, rope]
            and _list(attention.get("kv_a_output_shape")) == [kv_rank + rope]
            and _list(attention.get("k_compressed_norm_shape")) == [kv_rank]
            and _list(attention.get("kv_b_output_shape")) == [heads * (nope + value_dim)]
            and _list(attention.get("k_nope_shape")) == [heads, nope]
            and _list(attention.get("value_shape")) == [heads, value_dim]
        )
    except Exception as exc:
        errors.append({"phase": "attention_projection", "error_type": type(exc).__name__, "error_digest": dequant_probe.sha_payload(str(exc))})
        blockers.append("glm52_attention_projection_failed")
    if ready:
        blockers.extend(
            [
                "glm52_attention_projection_is_not_rope_attention",
                "glm52_attention_projection_is_not_o_proj",
                "glm52_attention_projection_is_not_stage_decode",
                "glm52_attention_projection_missing_attention_scores",
                "glm52_attention_projection_missing_kv_cache_update",
            ]
        )
    blockers.append("glm52_stage_decode_not_verified")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "glm52_attention_projection_probe_ready": ready,
        "model_id": MODEL_ID,
        "model_repo": str(args.model_repo),
        "model_type": str(attention.get("model_type") or ""),
        "layer_id": int(args.layer_id),
        **attention,
        "input_layernorm_verified": ready,
        "q_lora_projection_verified": ready,
        "kv_lora_projection_verified": ready,
        "attention_projection_verified": ready,
        "rope_applied": False,
        "attention_scores_verified": False,
        "o_proj_verified": False,
        "stage_decode_verified": False,
        "errors": errors,
        "blockers": sorted(set(blockers)),
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
        "public_artifact_safe": True,
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["glm52_attention_projection_probe_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"] = sorted(set([*blockers, "public_redaction_scan_failed"]))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--layer-id", type=int, default=3)
    parser.add_argument("--max-header-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-tensor-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--hf-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_report(args)
    path = output_dir / "glm52_attention_projection_probe.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"Attention projection verified: {report.get('attention_projection_verified')}")
    return 0 if report.get("public_artifact_safe") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
