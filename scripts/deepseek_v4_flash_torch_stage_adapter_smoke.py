#!/usr/bin/env python3
"""Smoke-test a tiny DeepSeek-V4 decoder stage using Transformers reference code."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "deepseek_v4_flash_torch_stage_adapter_smoke_v1"
DEFAULT_OUTPUT_DIR = "dist/deepseek-v4-flash-torch-stage-adapter-smoke"
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Cookie:",
    "Set-Cookie",
    '"prompt":',
    '"generated_text":',
    '"generated_token_ids":',
    '"activation":',
    '"hidden_state":',
    '"logits":',
    '"kv_cache":',
    '"past_key_values":',
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def build_tiny_config(args: argparse.Namespace) -> Any:
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    hidden_size = int(args.hidden_size)
    num_heads = int(args.num_attention_heads)
    head_dim = int(args.head_dim)
    o_groups = int(args.o_groups)
    o_lora_rank = int(args.o_lora_rank)
    q_lora_rank = int(args.q_lora_rank)
    layer_count = max(1, int(args.layer_count))
    layer_types = [
        "compressed_sparse_attention" if index % 2 else "heavily_compressed_attention"
        for index in range(layer_count)
    ]
    mlp_layer_types = ["moe"] * layer_count
    return DeepseekV4Config(
        vocab_size=512,
        hidden_size=hidden_size,
        moe_intermediate_size=int(args.moe_intermediate_size),
        num_hidden_layers=layer_count,
        num_attention_heads=num_heads,
        num_key_value_heads=1,
        head_dim=head_dim,
        q_lora_rank=q_lora_rank,
        n_routed_experts=int(args.num_experts),
        num_experts_per_tok=int(args.num_experts_per_tok),
        n_shared_experts=1,
        layer_types=layer_types,
        mlp_layer_types=mlp_layer_types,
        compress_rates={
            "compressed_sparse_attention": int(args.csa_compress_rate),
            "heavily_compressed_attention": int(args.hca_compress_rate),
        },
        sliding_window=int(args.sliding_window),
        o_groups=o_groups,
        o_lora_rank=o_lora_rank,
        index_n_heads=int(args.index_n_heads),
        index_head_dim=int(args.index_head_dim),
        index_topk=int(args.index_topk),
        hc_mult=int(args.hc_mult),
        hc_sinkhorn_iters=int(args.hc_sinkhorn_iters),
        partial_rotary_factor=float(args.partial_rotary_factor),
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        torch_dtype="float32",
        _attn_implementation="eager",
    )


def tensor_summary(tensor: Any) -> dict[str, Any]:
    import torch

    value = tensor.detach().float().cpu()
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "mean": round(float(value.mean()), 8),
        "std": round(float(value.std(unbiased=False)), 8) if value.numel() > 1 else 0.0,
        "min": round(float(value.min()), 8),
        "max": round(float(value.max()), 8),
        "payload_public": False,
    }


def run_reference_stage(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4DecoderLayer,
        DeepseekV4Model,
        DeepseekV4RotaryEmbedding,
        create_sliding_window_causal_mask,
    )

    torch.manual_seed(int(args.seed))
    config = build_tiny_config(args)
    model = DeepseekV4Model(config).eval()
    layer: DeepseekV4DecoderLayer = model.layers[0]
    rotary: DeepseekV4RotaryEmbedding = model.rotary_emb
    batch = 1
    seq = int(args.sequence_length)
    hidden = int(config.hidden_size)
    hc = int(config.hc_mult)
    hidden_states = torch.linspace(-0.15, 0.15, steps=batch * seq * hc * hidden, dtype=torch.float32)
    hidden_states = hidden_states.reshape(batch, seq, hc, hidden)
    input_ids = torch.arange(seq, dtype=torch.long).reshape(batch, seq) % int(config.vocab_size)
    position_ids = torch.arange(seq, dtype=torch.long).reshape(batch, seq)
    collapsed_for_rope = hidden_states[:, :, 0, :]
    position_embeddings = {
        name: rotary(collapsed_for_rope, position_ids=position_ids, layer_type=name)
        for name in ["main", "compress"]
    }
    past_key_values = DynamicCache(config=config)
    attention_mask = create_sliding_window_causal_mask(
        config,
        inputs_embeds=collapsed_for_rope,
        attention_mask=None,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    with torch.no_grad():
        output = layer(
            hidden_states,
            input_ids=input_ids,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
    state = layer.state_dict()
    return {
        "schema": "deepseek_v4_flash_torch_stage_reference_v1",
        "ok": True,
        "transformers_reference_used": True,
        "model_type": str(config.model_type),
        "architectures": ["DeepseekV4ForCausalLM"],
        "tiny_fixture": True,
        "real_deepseek_v4_components_exercised": {
            "manifold_hyper_connections": True,
            "compressed_attention": config.layer_types[0] in {"compressed_sparse_attention", "heavily_compressed_attention"},
            "mla_shared_kv_attention": True,
            "grouped_output_projection": True,
            "moe_router": True,
            "routed_experts": True,
            "shared_experts": True,
            "stage_local_kv_cache_shape": True,
        },
        "layer_type": str(config.layer_types[0]),
        "mlp_layer_type": str(config.mlp_layer_types[0]),
        "shape_metadata": {
            "input_shape": [batch, seq, hc, hidden],
            "output_shape": [int(item) for item in output.shape],
            "layout": "batch_seq_hc_hidden",
            "dtype": "float32",
            "activation_payload_public": False,
        },
        "config_summary": {
            "hidden_size": int(config.hidden_size),
            "head_dim": int(config.head_dim),
            "num_attention_heads": int(config.num_attention_heads),
            "q_lora_rank": int(config.q_lora_rank),
            "o_groups": int(config.o_groups),
            "o_lora_rank": int(config.o_lora_rank),
            "hc_mult": int(config.hc_mult),
            "num_local_experts": int(config.num_local_experts),
            "num_experts_per_tok": int(config.num_experts_per_tok),
            "moe_intermediate_size": int(config.moe_intermediate_size),
            "compress_rates": dict(config.compress_rates),
            "config_payload_public": False,
        },
        "stage_owned_key_count": len(state),
        "stage_owned_key_digest": stable_hash(sorted(state)),
        "output_summary": tensor_summary(output),
        "output_summary_hash": stable_hash(tensor_summary(output)),
        "weight_tensor_values_public": False,
        "activation_payload_public": False,
        "blockers": [],
        "diagnosis_codes": ["deepseek_v4_flash_torch_stage_reference_ready"],
        "public_artifact_safe": True,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    diagnosis: list[str] = []
    try:
        reference = run_reference_stage(args)
    except Exception as exc:
        reference = {
            "schema": "deepseek_v4_flash_torch_stage_reference_v1",
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message_public": str(exc)[:500],
            "error_digest": stable_hash(str(exc)),
            "blockers": ["deepseek_v4_flash_torch_stage_reference_failed"],
            "diagnosis_codes": ["deepseek_v4_flash_torch_stage_reference_failed"],
            "public_artifact_safe": True,
        }
    blockers.extend(str(item) for item in reference.get("blockers") or [])
    diagnosis.extend(str(item) for item in reference.get("diagnosis_codes") or [])
    if reference.get("ok") is True:
        diagnosis.append("deepseek_v4_flash_torch_stage_adapter_smoke_ready")
    else:
        blockers.append("deepseek_v4_flash_torch_stage_reference_not_ready")
        diagnosis.append("deepseek_v4_flash_torch_stage_adapter_smoke_not_ready")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": reference.get("ok") is True,
        "deepseek_v4_flash_torch_stage_adapter_smoke_ready": reference.get("ok") is True,
        "model": {
            "model_id": "deepseek-ai/DeepSeek-V4-Flash",
            "architecture_class": "moe",
            "model_type": "deepseek_v4",
            "reference_implementation": "transformers.models.deepseek_v4",
            "full_model_weight_values_loaded": False,
        },
        "reference_stage": reference,
        "jax_tpu_translation_ready": False,
        "real_deepseek_weights_loaded": False,
        "blockers": sorted(set(blockers)),
        "diagnosis_codes": sorted(set(diagnosis)),
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
            "weight_tensor_values_public": False,
        },
        "public_artifact_safe": True,
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["deepseek_v4_flash_torch_stage_adapter_smoke_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"].append("public_redaction_scan_failed")
        report["diagnosis_codes"].append("public_redaction_scan_failed")
        report["redaction_errors"] = leaks
    summary_path = output_dir / "deepseek_v4_flash_torch_stage_adapter_smoke.json"
    write_json(summary_path, report)
    report["artifacts"] = {
        "summary_json": artifact_entry(summary_path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(report.get("ok"))),
    }
    write_json(summary_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test a tiny DeepSeek-V4 decoder stage with the Transformers reference.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--q-lora-rank", type=int, default=16)
    parser.add_argument("--o-groups", type=int, default=2)
    parser.add_argument("--o-lora-rank", type=int, default=8)
    parser.add_argument("--hc-mult", type=int, default=2)
    parser.add_argument("--hc-sinkhorn-iters", type=int, default=3)
    parser.add_argument("--moe-intermediate-size", type=int, default=32)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--num-experts-per-tok", type=int, default=2)
    parser.add_argument("--csa-compress-rate", type=int, default=2)
    parser.add_argument("--hca-compress-rate", type=int, default=4)
    parser.add_argument("--sliding-window", type=int, default=8)
    parser.add_argument("--index-n-heads", type=int, default=4)
    parser.add_argument("--index-head-dim", type=int, default=8)
    parser.add_argument("--index-topk", type=int, default=2)
    parser.add_argument("--partial-rotary-factor", type=float, default=0.25)
    parser.add_argument("--layer-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=270701)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.sequence_length < 1 or args.sequence_length > 256:
        raise SystemExit("--sequence-length must be between 1 and 256")
    if args.hidden_size < 8:
        raise SystemExit("--hidden-size must be >= 8")
    if args.hidden_size != args.num_attention_heads * args.head_dim:
        raise SystemExit("--hidden-size must equal --num-attention-heads * --head-dim")
    if args.o_groups < 1 or args.num_attention_heads % args.o_groups != 0:
        raise SystemExit("--o-groups must divide --num-attention-heads")
    if args.num_experts_per_tok < 1 or args.num_experts_per_tok > args.num_experts:
        raise SystemExit("--num-experts-per-tok must be between 1 and --num-experts")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {Path(args.output_dir) / 'deepseek_v4_flash_torch_stage_adapter_smoke.json'}")
        print(f"Torch stage adapter smoke ready: {report.get('deepseek_v4_flash_torch_stage_adapter_smoke_ready')}")
        if report.get("blockers"):
            print("Blockers: " + ", ".join(str(item) for item in report.get("blockers") or []))
    return 0 if report.get("public_artifact_safe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
