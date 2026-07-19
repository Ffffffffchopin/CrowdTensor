#!/usr/bin/env python3
"""Smoke-test a DeepSeek-V4 stage adapter translation with optional JAX execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "deepseek_v4_flash_jax_stage_adapter_smoke_v1"
DEFAULT_OUTPUT_DIR = "dist/deepseek-v4-flash-jax-stage-adapter-smoke"
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


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def fixture_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_id": "deepseek-ai/DeepSeek-V4-Flash",
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "hidden_size": int(args.hidden_size),
        "head_dim": int(args.head_dim),
        "num_attention_heads": int(args.num_attention_heads),
        "num_key_value_heads": 1,
        "q_lora_rank": int(args.q_lora_rank),
        "o_groups": int(args.o_groups),
        "o_lora_rank": int(args.o_lora_rank),
        "hc_mult": int(args.hc_mult),
        "hc_sinkhorn_iters": int(args.hc_sinkhorn_iters),
        "hc_eps": 1.0e-8,
        "rms_norm_eps": 1.0e-6,
        "num_local_experts": int(args.num_experts),
        "num_experts_per_tok": int(args.num_experts_per_tok),
        "moe_intermediate_size": int(args.moe_intermediate_size),
        "n_shared_experts": 1,
        "routed_scaling_factor": 1.0,
        "swiglu_limit": 7.0,
        "layer_type": "heavily_compressed_attention",
        "mlp_layer_type": "moe",
        "compress_rates": {
            "compressed_sparse_attention": int(args.csa_compress_rate),
            "heavily_compressed_attention": int(args.hca_compress_rate),
        },
        "partial_rotary_factor": float(args.partial_rotary_factor),
        "rope_theta": 10000.0,
        "compress_rope_theta": 160000.0,
    }


def np_summary(value: Any) -> dict[str, Any]:
    import numpy as np

    arr = np.asarray(value, dtype=np.float32)
    return {
        "shape": [int(item) for item in arr.shape],
        "dtype": "float32",
        "mean": round(float(arr.mean()), 8),
        "std": round(float(arr.std()), 8) if arr.size > 1 else 0.0,
        "min": round(float(arr.min()), 8),
        "max": round(float(arr.max()), 8),
        "payload_public": False,
    }


def build_fixture_state(config: dict[str, Any], *, seed: int) -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    head_dim = int(config["head_dim"])
    q_rank = int(config["q_lora_rank"])
    o_groups = int(config["o_groups"])
    o_rank = int(config["o_lora_rank"])
    hc = int(config["hc_mult"])
    experts = int(config["num_local_experts"])
    inter = int(config["moe_intermediate_size"])
    mix = (2 + hc) * hc
    grouped_in = heads * head_dim // o_groups

    def normal(shape: tuple[int, ...], scale: float = 0.025) -> Any:
        return rng.normal(0.0, scale, size=shape).astype(np.float32)

    return {
        "self_attn.sinks": np.zeros((heads,), dtype=np.float32),
        "self_attn.q_a_proj.weight": normal((q_rank, hidden)),
        "self_attn.q_a_norm.weight": np.ones((q_rank,), dtype=np.float32),
        "self_attn.q_b_proj.weight": normal((heads * head_dim, q_rank)),
        "self_attn.kv_proj.weight": normal((head_dim, hidden)),
        "self_attn.kv_norm.weight": np.ones((head_dim,), dtype=np.float32),
        "self_attn.o_a_proj.weight": normal((o_groups * o_rank, grouped_in)),
        "self_attn.o_b_proj.weight": normal((hidden, o_groups * o_rank)),
        "self_attn.compressor.position_bias": np.zeros((int(config["compress_rates"]["heavily_compressed_attention"]), head_dim), dtype=np.float32),
        "self_attn.compressor.kv_proj.weight": normal((head_dim, hidden)),
        "self_attn.compressor.gate_proj.weight": normal((head_dim, hidden)),
        "self_attn.compressor.kv_norm.weight": np.ones((head_dim,), dtype=np.float32),
        "mlp.gate.weight": normal((experts, hidden)),
        "mlp.gate.e_score_correction_bias": np.zeros((experts,), dtype=np.float32),
        "mlp.experts.gate_up_proj": normal((experts, 2 * inter, hidden)),
        "mlp.experts.down_proj": normal((experts, hidden, inter)),
        "mlp.shared_experts.gate_proj.weight": normal((inter, hidden)),
        "mlp.shared_experts.up_proj.weight": normal((inter, hidden)),
        "mlp.shared_experts.down_proj.weight": normal((hidden, inter)),
        "input_layernorm.weight": np.ones((hidden,), dtype=np.float32),
        "post_attention_layernorm.weight": np.ones((hidden,), dtype=np.float32),
        "attn_hc.fn": normal((mix, hc * hidden)),
        "attn_hc.base": np.zeros((mix,), dtype=np.float32),
        "attn_hc.scale": np.ones((3,), dtype=np.float32),
        "ffn_hc.fn": normal((mix, hc * hidden)),
        "ffn_hc.base": np.zeros((mix,), dtype=np.float32),
        "ffn_hc.scale": np.ones((3,), dtype=np.float32),
    }


def run_numpy_reference(config: dict[str, Any], state: dict[str, Any], *, sequence_length: int) -> dict[str, Any]:
    import numpy as np

    hidden = int(config["hidden_size"])
    hc = int(config["hc_mult"])
    heads = int(config["num_attention_heads"])
    head_dim = int(config["head_dim"])
    q_rank = int(config["q_lora_rank"])
    o_groups = int(config["o_groups"])
    o_rank = int(config["o_lora_rank"])
    top_k = int(config["num_experts_per_tok"])
    inter = int(config["moe_intermediate_size"])
    x = np.linspace(-0.15, 0.15, num=sequence_length * hc * hidden, dtype=np.float32).reshape(1, sequence_length, hc, hidden)

    def rms_norm(value: Any, weight: Any | None = None) -> Any:
        out = value / np.sqrt(np.mean(np.square(value.astype(np.float32)), axis=-1, keepdims=True) + float(config["rms_norm_eps"]))
        return out if weight is None else out * weight

    def sigmoid(value: Any) -> Any:
        return 1.0 / (1.0 + np.exp(-value))

    def softmax(value: Any, axis: int = -1) -> Any:
        shifted = value - np.max(value, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=axis, keepdims=True)

    def silu(value: Any) -> Any:
        return value * sigmoid(value)

    def hc_map(prefix: str, hidden_streams: Any) -> tuple[Any, Any, Any]:
        flat = rms_norm(hidden_streams.reshape(1, sequence_length, hc * hidden))
        logits = flat @ state[f"{prefix}.fn"].T
        pre_w, post_w, comb_w = np.split(logits, [hc, 2 * hc], axis=-1)
        pre_b, post_b, comb_b = np.split(state[f"{prefix}.base"], [hc, 2 * hc])
        pre_scale, post_scale, comb_scale = state[f"{prefix}.scale"]
        pre = sigmoid(pre_w * pre_scale + pre_b) + float(config["hc_eps"])
        post = 2.0 * sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.reshape(1, sequence_length, hc, hc) * comb_scale + comb_b.reshape(hc, hc)
        comb = softmax(comb_logits, axis=-1) + float(config["hc_eps"])
        comb = comb / (np.sum(comb, axis=-2, keepdims=True) + float(config["hc_eps"]))
        for _ in range(max(0, int(config["hc_sinkhorn_iters"]) - 1)):
            comb = comb / (np.sum(comb, axis=-1, keepdims=True) + float(config["hc_eps"]))
            comb = comb / (np.sum(comb, axis=-2, keepdims=True) + float(config["hc_eps"]))
        collapsed = np.sum(pre[..., None] * hidden_streams, axis=2)
        return post.astype(np.float32), comb.astype(np.float32), collapsed.astype(np.float32)

    def rotate_half(value: Any) -> Any:
        even = value[..., 0::2]
        odd = value[..., 1::2]
        return np.stack((-odd, even), axis=-1).reshape(value.shape)

    def apply_rope(value: Any, *, layout: str = "bhsd") -> Any:
        rope_dim = max(2, int(head_dim * float(config["partial_rotary_factor"])))
        rope_dim = rope_dim if rope_dim % 2 == 0 else rope_dim - 1
        inv_freq = 1.0 / (float(config["compress_rope_theta"]) ** (np.arange(0, rope_dim, 2, dtype=np.float32) / rope_dim))
        positions = np.arange(sequence_length, dtype=np.float32)
        freqs = np.einsum("i,j->ij", positions, inv_freq)
        if layout == "bhsd":
            cos = np.repeat(np.cos(freqs), 2, axis=-1)[None, None, :, :]
            sin = np.repeat(np.sin(freqs), 2, axis=-1)[None, None, :, :]
        elif layout == "bshd":
            cos = np.repeat(np.cos(freqs), 2, axis=-1)[None, :, None, :]
            sin = np.repeat(np.sin(freqs), 2, axis=-1)[None, :, None, :]
        else:
            raise ValueError(f"unsupported_rope_layout:{layout}")
        nope = value[..., :-rope_dim]
        rope = value[..., -rope_dim:]
        rotated = rope * cos + rotate_half(rope) * sin
        return np.concatenate([nope, rotated], axis=-1)

    def attention(collapsed: Any) -> tuple[Any, dict[str, Any]]:
        normed = rms_norm(collapsed, state["input_layernorm.weight"])
        q_residual = rms_norm(normed @ state["self_attn.q_a_proj.weight"].T, state["self_attn.q_a_norm.weight"])
        q = q_residual @ state["self_attn.q_b_proj.weight"].T
        q = q.reshape(1, sequence_length, heads, head_dim).transpose(0, 2, 1, 3)
        q = rms_norm(q)
        kv = rms_norm(normed @ state["self_attn.kv_proj.weight"].T, state["self_attn.kv_norm.weight"])
        kv = kv.reshape(1, sequence_length, 1, head_dim).transpose(0, 2, 1, 3)
        q = apply_rope(q)
        kv = apply_rope(kv)
        kv_full = np.repeat(kv, heads, axis=1)
        scores = np.matmul(q, np.swapaxes(kv_full, -1, -2)) / math.sqrt(head_dim)
        causal = np.tril(np.ones((sequence_length, sequence_length), dtype=bool))[None, None, :, :]
        scores = np.where(causal, scores, -1.0e4)
        sinks = state["self_attn.sinks"].reshape(1, heads, 1, 1)
        combined = np.concatenate([scores, np.broadcast_to(sinks, (1, heads, sequence_length, 1))], axis=-1)
        probs = softmax(combined, axis=-1)[..., :-1]
        context = np.matmul(probs, kv_full)
        context = apply_rope(context.transpose(0, 2, 1, 3), layout="bshd").transpose(0, 2, 1, 3)
        grouped = context.transpose(0, 2, 1, 3).reshape(1, sequence_length, o_groups, -1)
        o_a_w = state["self_attn.o_a_proj.weight"].reshape(o_groups, o_rank, -1)
        grouped_out = np.einsum("bsgh,goh->bsgo", grouped, o_a_w).reshape(1, sequence_length, o_groups * o_rank)
        out = grouped_out @ state["self_attn.o_b_proj.weight"].T
        kv_cache = {
            "stage_local_only": True,
            "k_shape": [int(item) for item in kv.shape],
            "v_shape": [int(item) for item in kv.shape],
            "compressed_kv_shape_metadata": [1, 1, max(1, sequence_length // int(config["compress_rates"]["heavily_compressed_attention"])), head_dim],
            "kv_payload_public": False,
            "past_key_values_public": False,
        }
        return out.astype(np.float32), kv_cache

    def moe(collapsed: Any) -> Any:
        normed = rms_norm(collapsed, state["post_attention_layernorm.weight"])
        flat = normed.reshape(sequence_length, hidden)
        logits = flat @ state["mlp.gate.weight"].T + state["mlp.gate.e_score_correction_bias"]
        scores = sigmoid(logits)
        indices = np.argsort(scores, axis=-1)[:, -top_k:]
        gathered = np.take_along_axis(scores, indices, axis=-1)
        weights = gathered / (np.sum(gathered, axis=-1, keepdims=True) + 1.0e-20)
        routed = np.zeros_like(flat)
        for token in range(sequence_length):
            for pos in range(top_k):
                expert = int(indices[token, pos])
                gate_up = flat[token] @ state["mlp.experts.gate_up_proj"][expert].T
                gate, up = np.split(gate_up, 2, axis=-1)
                current = silu(np.clip(gate, None, float(config["swiglu_limit"]))) * np.clip(up, -float(config["swiglu_limit"]), float(config["swiglu_limit"]))
                routed[token] += (current @ state["mlp.experts.down_proj"][expert].T) * weights[token, pos]
        shared = silu(flat @ state["mlp.shared_experts.gate_proj.weight"].T) * (flat @ state["mlp.shared_experts.up_proj.weight"].T)
        shared = shared @ state["mlp.shared_experts.down_proj.weight"].T
        return (routed + shared).reshape(1, sequence_length, hidden).astype(np.float32)

    post, comb, collapsed = hc_map("attn_hc", x)
    attn_output, kv_cache = attention(collapsed)
    x = post[..., None] * attn_output[:, :, None, :] + np.einsum("bsji,bsjd->bsid", comb, x)
    post, comb, collapsed = hc_map("ffn_hc", x)
    mlp_output = moe(collapsed)
    x = post[..., None] * mlp_output[:, :, None, :] + np.einsum("bsji,bsjd->bsid", comb, x)
    return {
        "ok": True,
        "output_summary": np_summary(x),
        "output_hash": stable_hash(np_summary(x)),
        "stage_local_kv_cache_metadata": kv_cache,
        "components_exercised": {
            "manifold_hyper_connections": True,
            "mla_shared_kv_attention": True,
            "grouped_output_projection": True,
            "attention_sink": True,
            "topk_moe_router": True,
            "routed_experts": True,
            "shared_experts": True,
            "hca_compressor_shape_metadata": True,
        },
    }


def run_jax_stage(config: dict[str, Any], state: dict[str, Any], *, sequence_length: int, require_tpu: bool) -> dict[str, Any]:
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
    except Exception as exc:
        return {
            "ok": False,
            "jax_imported": False,
            "jax_runtime_execution_ready": False,
            "tpu_runtime_ready": False,
            "error_type": type(exc).__name__,
            "error_digest": stable_hash(str(exc)),
            "blockers": ["jax_missing"],
            "diagnosis_codes": ["deepseek_v4_flash_jax_missing"],
        }

    devices = list(jax.devices())
    tpu_devices = [device for device in devices if str(getattr(device, "platform", "")).lower() == "tpu"]
    if require_tpu and not tpu_devices:
        return {
            "ok": False,
            "jax_imported": True,
            "jax_runtime_execution_ready": False,
            "tpu_runtime_ready": False,
            "jax_device_count": len(devices),
            "jax_tpu_device_count": 0,
            "jax_devices_public": [
                {"platform": str(getattr(device, "platform", "")), "device_kind": str(getattr(device, "device_kind", ""))}
                for device in devices
            ],
            "blockers": ["jax_tpu_device_missing"],
            "diagnosis_codes": ["deepseek_v4_flash_jax_tpu_device_missing"],
        }

    hidden = int(config["hidden_size"])
    hc = int(config["hc_mult"])
    heads = int(config["num_attention_heads"])
    head_dim = int(config["head_dim"])
    q_rank = int(config["q_lora_rank"])
    o_groups = int(config["o_groups"])
    o_rank = int(config["o_lora_rank"])
    top_k = int(config["num_experts_per_tok"])
    hca_rate = int(config["compress_rates"]["heavily_compressed_attention"])

    np_state = {key: np.asarray(value, dtype=np.float32) for key, value in state.items()}
    x_np = np.linspace(-0.15, 0.15, num=sequence_length * hc * hidden, dtype=np.float32).reshape(1, sequence_length, hc, hidden)

    def rms_norm(value: Any, weight: Any | None = None) -> Any:
        out = value * jax.lax.rsqrt(jnp.mean(jnp.square(value.astype(jnp.float32)), axis=-1, keepdims=True) + float(config["rms_norm_eps"]))
        return out if weight is None else out * weight

    def rotate_half(value: Any) -> Any:
        even = value[..., 0::2]
        odd = value[..., 1::2]
        return jnp.reshape(jnp.stack((-odd, even), axis=-1), value.shape)

    def apply_rope(value: Any, *, layout: str = "bhsd") -> Any:
        rope_dim = max(2, int(head_dim * float(config["partial_rotary_factor"])))
        rope_dim = rope_dim if rope_dim % 2 == 0 else rope_dim - 1
        inv_freq = 1.0 / (float(config["compress_rope_theta"]) ** (jnp.arange(0, rope_dim, 2, dtype=jnp.float32) / rope_dim))
        positions = jnp.arange(sequence_length, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", positions, inv_freq)
        if layout == "bhsd":
            cos = jnp.repeat(jnp.cos(freqs), 2, axis=-1)[None, None, :, :]
            sin = jnp.repeat(jnp.sin(freqs), 2, axis=-1)[None, None, :, :]
        elif layout == "bshd":
            cos = jnp.repeat(jnp.cos(freqs), 2, axis=-1)[None, :, None, :]
            sin = jnp.repeat(jnp.sin(freqs), 2, axis=-1)[None, :, None, :]
        else:
            raise ValueError(f"unsupported_rope_layout:{layout}")
        nope = value[..., :-rope_dim]
        rope = value[..., -rope_dim:]
        return jnp.concatenate([nope, rope * cos + rotate_half(rope) * sin], axis=-1)

    def hc_map(p: dict[str, Any], prefix: str, hidden_streams: Any) -> tuple[Any, Any, Any]:
        flat = rms_norm(jnp.reshape(hidden_streams, (1, sequence_length, hc * hidden)))
        logits = flat @ jnp.swapaxes(p[f"{prefix}.fn"], -1, -2)
        pre_w, post_w, comb_w = jnp.split(logits, [hc, 2 * hc], axis=-1)
        pre_b, post_b, comb_b = jnp.split(p[f"{prefix}.base"], [hc, 2 * hc])
        pre_scale, post_scale, comb_scale = p[f"{prefix}.scale"]
        pre = jax.nn.sigmoid(pre_w * pre_scale + pre_b) + float(config["hc_eps"])
        post = 2.0 * jax.nn.sigmoid(post_w * post_scale + post_b)
        comb_logits = jnp.reshape(comb_w, (1, sequence_length, hc, hc)) * comb_scale + jnp.reshape(comb_b, (hc, hc))
        comb = jax.nn.softmax(comb_logits, axis=-1) + float(config["hc_eps"])
        comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + float(config["hc_eps"]))
        for _ in range(max(0, int(config["hc_sinkhorn_iters"]) - 1)):
            comb = comb / (jnp.sum(comb, axis=-1, keepdims=True) + float(config["hc_eps"]))
            comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + float(config["hc_eps"]))
        collapsed = jnp.sum(pre[..., None] * hidden_streams, axis=2)
        return post, comb, collapsed

    def attention(p: dict[str, Any], collapsed: Any) -> tuple[Any, Any, Any]:
        normed = rms_norm(collapsed, p["input_layernorm.weight"])
        q_residual = rms_norm(normed @ jnp.swapaxes(p["self_attn.q_a_proj.weight"], -1, -2), p["self_attn.q_a_norm.weight"])
        q = q_residual @ jnp.swapaxes(p["self_attn.q_b_proj.weight"], -1, -2)
        q = jnp.transpose(jnp.reshape(q, (1, sequence_length, heads, head_dim)), (0, 2, 1, 3))
        q = rms_norm(q)
        kv = rms_norm(normed @ jnp.swapaxes(p["self_attn.kv_proj.weight"], -1, -2), p["self_attn.kv_norm.weight"])
        kv = jnp.transpose(jnp.reshape(kv, (1, sequence_length, 1, head_dim)), (0, 2, 1, 3))
        q = apply_rope(q)
        kv = apply_rope(kv)
        kv_full = jnp.repeat(kv, heads, axis=1)
        scores = jnp.matmul(q, jnp.swapaxes(kv_full, -1, -2)) / jnp.sqrt(jnp.asarray(head_dim, dtype=jnp.float32))
        causal = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))[None, None, :, :]
        scores = jnp.where(causal, scores, jnp.asarray(-1.0e4, dtype=jnp.float32))
        sinks = jnp.reshape(p["self_attn.sinks"], (1, heads, 1, 1))
        combined = jnp.concatenate([scores, jnp.broadcast_to(sinks, (1, heads, sequence_length, 1))], axis=-1)
        probs = jax.nn.softmax(combined, axis=-1)[..., :-1]
        context = jnp.matmul(probs, kv_full)
        context = jnp.transpose(apply_rope(jnp.transpose(context, (0, 2, 1, 3)), layout="bshd"), (0, 2, 1, 3))
        grouped = jnp.reshape(jnp.transpose(context, (0, 2, 1, 3)), (1, sequence_length, o_groups, -1))
        o_a_w = jnp.reshape(p["self_attn.o_a_proj.weight"], (o_groups, o_rank, -1))
        grouped_out = jnp.reshape(jnp.einsum("bsgh,goh->bsgo", grouped, o_a_w), (1, sequence_length, o_groups * o_rank))
        return grouped_out @ jnp.swapaxes(p["self_attn.o_b_proj.weight"], -1, -2), kv, kv

    def moe(p: dict[str, Any], collapsed: Any) -> Any:
        normed = rms_norm(collapsed, p["post_attention_layernorm.weight"])
        flat = jnp.reshape(normed, (sequence_length, hidden))
        logits = flat @ jnp.swapaxes(p["mlp.gate.weight"], -1, -2) + p["mlp.gate.e_score_correction_bias"]
        scores = jax.nn.sigmoid(logits)
        weights, indices = jax.lax.top_k(scores, top_k)
        weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1.0e-20)
        routed = jnp.zeros_like(flat)
        for expert_id in range(int(config["num_local_experts"])):
            gate_up = flat @ jnp.swapaxes(p["mlp.experts.gate_up_proj"][expert_id], -1, -2)
            gate, up = jnp.split(gate_up, 2, axis=-1)
            current = jax.nn.silu(jnp.minimum(gate, float(config["swiglu_limit"]))) * jnp.clip(
                up,
                -float(config["swiglu_limit"]),
                float(config["swiglu_limit"]),
            )
            current = current @ jnp.swapaxes(p["mlp.experts.down_proj"][expert_id], -1, -2)
            mask = (indices == expert_id).astype(jnp.float32)
            routed = routed + current * jnp.sum(mask * weights, axis=-1, keepdims=True)
        shared = jax.nn.silu(flat @ jnp.swapaxes(p["mlp.shared_experts.gate_proj.weight"], -1, -2)) * (
            flat @ jnp.swapaxes(p["mlp.shared_experts.up_proj.weight"], -1, -2)
        )
        shared = shared @ jnp.swapaxes(p["mlp.shared_experts.down_proj.weight"], -1, -2)
        return jnp.reshape(routed + shared, (1, sequence_length, hidden))

    @jax.jit
    def stage_forward(p: dict[str, Any], x: Any) -> tuple[Any, Any, Any]:
        post, comb, collapsed = hc_map(p, "attn_hc", x)
        attn_output, k, v = attention(p, collapsed)
        x = post[..., None] * attn_output[:, :, None, :] + jnp.einsum("bsji,bsjd->bsid", comb, x)
        post, comb, collapsed = hc_map(p, "ffn_hc", x)
        mlp_output = moe(p, collapsed)
        x = post[..., None] * mlp_output[:, :, None, :] + jnp.einsum("bsji,bsjd->bsid", comb, x)
        return x, k, v

    device = tpu_devices[0] if tpu_devices else devices[0]
    output, k, v = stage_forward(jax.device_put(np_state, device), jax.device_put(x_np, device))
    out_np = jax.device_get(output)
    return {
        "ok": True,
        "jax_imported": True,
        "jax_version": str(getattr(jax, "__version__", "")),
        "jax_runtime_execution_ready": True,
        "tpu_runtime_ready": bool(tpu_devices),
        "jax_device_count": len(devices),
        "jax_tpu_device_count": len(tpu_devices),
        "jax_devices_public": [
            {"platform": str(getattr(device, "platform", "")), "device_kind": str(getattr(device, "device_kind", ""))}
            for device in devices
        ],
        "output_summary": np_summary(out_np),
        "output_hash": stable_hash(np_summary(out_np)),
        "stage_local_kv_cache_metadata": {
            "stage_local_only": True,
            "k_shape": [int(item) for item in k.shape],
            "v_shape": [int(item) for item in v.shape],
            "compressed_kv_shape_metadata": [1, 1, max(1, sequence_length // hca_rate), head_dim],
            "kv_payload_public": False,
            "past_key_values_public": False,
        },
        "blockers": [],
        "diagnosis_codes": ["deepseek_v4_flash_jax_stage_forward_ready"],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = fixture_config(args)
    state = build_fixture_state(config, seed=int(args.seed))
    numpy_reference = run_numpy_reference(config, state, sequence_length=int(args.sequence_length))
    jax_result = run_jax_stage(config, state, sequence_length=int(args.sequence_length), require_tpu=bool(args.require_tpu)) if args.run_jax else {
        "ok": False,
        "jax_runtime_execution_ready": False,
        "tpu_runtime_ready": False,
        "blockers": ["jax_execution_not_requested"],
        "diagnosis_codes": ["deepseek_v4_flash_jax_execution_not_requested"],
    }

    blockers = sorted(set(str(item) for item in jax_result.get("blockers") or []))
    diagnosis = sorted(set(["deepseek_v4_flash_numpy_stage_reference_ready", *[str(item) for item in jax_result.get("diagnosis_codes") or []]]))
    jax_ready = jax_result.get("ok") is True and jax_result.get("jax_runtime_execution_ready") is True
    tpu_ready = jax_result.get("tpu_runtime_ready") is True
    ready = bool(numpy_reference.get("ok") is True and jax_ready and (not args.require_tpu or tpu_ready))
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "deepseek_v4_flash_jax_stage_adapter_smoke_ready": ready,
        "model": {
            "model_id": "deepseek-ai/DeepSeek-V4-Flash",
            "architecture_class": "moe",
            "model_type": "deepseek_v4",
            "fixture_config": True,
            "real_deepseek_weights_loaded": False,
            "full_model_weight_values_loaded": False,
        },
        "stage": {
            "stage_type": "decoder_layer_fixture_translation",
            "stage_owned_key_count": len(state),
            "stage_owned_key_digest": stable_hash(sorted(state)),
            "sequence_length": int(args.sequence_length),
            "layer_type": config["layer_type"],
            "mlp_layer_type": config["mlp_layer_type"],
            "shape_metadata": {
                "input_shape": [1, int(args.sequence_length), int(config["hc_mult"]), int(config["hidden_size"])],
                "layout": "batch_seq_hc_hidden",
                "activation_payload_public": False,
            },
        },
        "numpy_reference": numpy_reference,
        "jax_runtime_execution_requested": bool(args.run_jax),
        "jax_runtime_execution_ready": jax_ready,
        "tpu_runtime_required": bool(args.require_tpu),
        "tpu_runtime_ready": tpu_ready,
        "deepseek_v4_jax_stage_forward_ready": ready,
        "deepseek_v4_jax_tpu_stage_forward_ready": bool(ready and tpu_ready),
        "jax_result": jax_result,
        "real_deepseek_weights_loaded": False,
        "blockers": [] if ready else blockers,
        "diagnosis_codes": diagnosis if not ready else sorted(set([*diagnosis, "deepseek_v4_flash_jax_stage_adapter_smoke_ready"])),
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
        report["deepseek_v4_flash_jax_stage_adapter_smoke_ready"] = False
        report["deepseek_v4_jax_stage_forward_ready"] = False
        report["deepseek_v4_jax_tpu_stage_forward_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"].append("public_redaction_scan_failed")
        report["diagnosis_codes"].append("public_redaction_scan_failed")
        report["redaction_errors"] = leaks
    summary_path = output_dir / "deepseek_v4_flash_jax_stage_adapter_smoke.json"
    write_json(summary_path, report)
    report["artifacts"] = {
        "summary_json": artifact_entry(summary_path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(report.get("ok"))),
    }
    write_json(summary_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--partial-rotary-factor", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=270702)
    parser.add_argument("--run-jax", action="store_true")
    parser.add_argument("--require-tpu", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.sequence_length < 1 or args.sequence_length > 256:
        raise SystemExit("--sequence-length must be between 1 and 256")
    if args.hidden_size != args.num_attention_heads * args.head_dim:
        raise SystemExit("--hidden-size must equal --num-attention-heads * --head-dim")
    if args.o_groups < 1 or args.num_attention_heads % args.o_groups != 0:
        raise SystemExit("--o-groups must divide --num-attention-heads")
    if args.num_experts_per_tok < 1 or args.num_experts_per_tok > args.num_experts:
        raise SystemExit("--num-experts-per-tok must be between 1 and --num-experts")
    if args.require_tpu and not args.run_jax:
        raise SystemExit("--require-tpu requires --run-jax")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {Path(args.output_dir) / 'deepseek_v4_flash_jax_stage_adapter_smoke.json'}")
        print(f"JAX stage adapter ready: {report.get('deepseek_v4_flash_jax_stage_adapter_smoke_ready')}")
        if report.get("blockers"):
            print("Blockers: " + ", ".join(str(item) for item in report.get("blockers") or []))
    return 0 if report.get("public_artifact_safe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
