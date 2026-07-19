#!/usr/bin/env python3
"""Smoke-test a dense Qwen stage adapter with PyTorch reference and optional JAX/TPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "qwen_dense_jax_tpu_stage_adapter_smoke_v1"
DEFAULT_OUTPUT_DIR = "dist/qwen-dense-jax-tpu-stage-adapter-smoke"
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Cookie:",
    "Set-Cookie",
    "kaggle-cookies",
    "kaggle-web-storage-state",
    "operator.private.env",
    "miner.private.env",
    '"prompt":',
    '"generated_text":',
    '"generated_token_ids":',
    '"activation":',
    '"activations":',
    '"hidden_state":',
    '"hidden_states":',
    '"logits":',
    '"kv_cache":',
    '"past_key_values":',
    '"lease_token":',
    '"idempotency_key":',
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


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


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def fixture_config() -> dict[str, Any]:
    return {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 64,
        "intermediate_size": 176,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "num_hidden_layers": 1,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "torch_dtype": "bfloat16",
        "vocab_size": 32000,
    }


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.config_json:
        return load_json(Path(args.config_json))
    return fixture_config()


def layer_key(prefix: str, layer_id: int, suffix: str) -> str:
    return f"{prefix}.{layer_id}.{suffix}"


def torch_tensor_summary(tensor: Any) -> dict[str, Any]:
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


def build_fixture_torch_state(config: dict[str, Any], *, seed: int, layer_id: int) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    hidden = _int(config.get("hidden_size"))
    intermediate = _int(config.get("intermediate_size"))
    heads = _int(config.get("num_attention_heads"))
    kv_heads = _int(config.get("num_key_value_heads"), heads)
    head_dim = hidden // heads
    kv_width = kv_heads * head_dim
    prefix = "model.layers"

    def rand(shape: tuple[int, ...], scale: float = 0.03) -> Any:
        return torch.randn(shape, dtype=torch.float32) * scale

    state = {
        layer_key(prefix, layer_id, "input_layernorm.weight"): torch.ones((hidden,), dtype=torch.float32),
        layer_key(prefix, layer_id, "post_attention_layernorm.weight"): torch.ones((hidden,), dtype=torch.float32),
        layer_key(prefix, layer_id, "self_attn.q_proj.weight"): rand((heads * head_dim, hidden)),
        layer_key(prefix, layer_id, "self_attn.q_proj.bias"): rand((heads * head_dim,), 0.005),
        layer_key(prefix, layer_id, "self_attn.k_proj.weight"): rand((kv_width, hidden)),
        layer_key(prefix, layer_id, "self_attn.k_proj.bias"): rand((kv_width,), 0.005),
        layer_key(prefix, layer_id, "self_attn.v_proj.weight"): rand((kv_width, hidden)),
        layer_key(prefix, layer_id, "self_attn.v_proj.bias"): rand((kv_width,), 0.005),
        layer_key(prefix, layer_id, "self_attn.o_proj.weight"): rand((hidden, heads * head_dim)),
        layer_key(prefix, layer_id, "mlp.gate_proj.weight"): rand((intermediate, hidden)),
        layer_key(prefix, layer_id, "mlp.up_proj.weight"): rand((intermediate, hidden)),
        layer_key(prefix, layer_id, "mlp.down_proj.weight"): rand((hidden, intermediate)),
    }
    return state


def load_stage_state_from_safetensors(model_path: Path, config: dict[str, Any], *, layer_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors.torch import load_file

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError("model_safetensors_index_missing")
    index = load_json(index_path)
    weight_map = _dict(index.get("weight_map"))
    wanted_suffixes = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    wanted_keys = [layer_key("model.layers", layer_id, suffix) for suffix in wanted_suffixes]
    files = sorted({Path(str(weight_map.get(key))).name for key in wanted_keys if weight_map.get(key)})
    if not files:
        raise RuntimeError("stage_owned_safetensors_files_missing")
    tensors: dict[str, Any] = {}
    for filename in files:
        file_path = model_path / filename
        if not file_path.is_file():
            raise RuntimeError("stage_owned_safetensors_file_not_attached")
        loaded = load_file(str(file_path), device="cpu")
        for key in wanted_keys:
            if key in loaded:
                tensors[key] = loaded[key].float()
    missing = [key for key in wanted_keys if key not in tensors]
    if missing:
        raise RuntimeError("stage_owned_safetensors_keys_missing")
    return tensors, {
        "source": "attached_safetensors",
        "stage_owned_file_count": len(files),
        "stage_owned_file_digest": stable_hash(files),
        "stage_owned_key_count": len(tensors),
        "weight_tensor_values_public": False,
    }


def repeat_kv_torch(value: Any, repeat_factor: int) -> Any:
    if repeat_factor <= 1:
        return value
    return value.repeat_interleave(repeat_factor, dim=2)


def rotate_half_torch(value: Any) -> Any:
    import torch

    x1 = value[..., : value.shape[-1] // 2]
    x2 = value[..., value.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_torch(q: Any, k: Any, *, theta: float) -> tuple[Any, Any]:
    import torch

    seq_len = q.shape[1]
    head_dim = q.shape[-1]
    inv_freq = 1.0 / (float(theta) ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)[None, :, None, :]
    cos = emb.cos().to(q.dtype)
    sin = emb.sin().to(q.dtype)
    return (q * cos) + (rotate_half_torch(q) * sin), (k * cos) + (rotate_half_torch(k) * sin)


def torch_stage_forward(config: dict[str, Any], state: dict[str, Any], hidden: Any, *, layer_id: int) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    hidden_size = _int(config.get("hidden_size"))
    heads = _int(config.get("num_attention_heads"))
    kv_heads = _int(config.get("num_key_value_heads"), heads)
    head_dim = hidden_size // heads
    repeat_factor = max(1, heads // max(1, kv_heads))
    eps = float(config.get("rms_norm_eps") or 1e-6)
    theta = float(config.get("rope_theta") or 1000000.0)
    seq_len = hidden.shape[1]
    prefix = "model.layers"

    def rms_norm(value: Any, weight: Any) -> Any:
        variance = value.float().pow(2).mean(dim=-1, keepdim=True)
        return (value * torch.rsqrt(variance + eps)) * weight

    residual = hidden
    normed = rms_norm(hidden, state[layer_key(prefix, layer_id, "input_layernorm.weight")])
    q = F.linear(normed, state[layer_key(prefix, layer_id, "self_attn.q_proj.weight")], state[layer_key(prefix, layer_id, "self_attn.q_proj.bias")])
    k = F.linear(normed, state[layer_key(prefix, layer_id, "self_attn.k_proj.weight")], state[layer_key(prefix, layer_id, "self_attn.k_proj.bias")])
    v = F.linear(normed, state[layer_key(prefix, layer_id, "self_attn.v_proj.weight")], state[layer_key(prefix, layer_id, "self_attn.v_proj.bias")])
    q = q.reshape(1, seq_len, heads, head_dim)
    k = k.reshape(1, seq_len, kv_heads, head_dim)
    v = v.reshape(1, seq_len, kv_heads, head_dim)
    q, k = apply_rope_torch(q, k, theta=theta)
    k_full = repeat_kv_torch(k, repeat_factor)
    v_full = repeat_kv_torch(v, repeat_factor)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k_full) / math.sqrt(head_dim)
    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    scores = scores.masked_fill(~causal[None, None, :, :], -1e4)
    attn = torch.softmax(scores.float(), dim=-1)
    context = torch.einsum("bhqk,bkhd->bqhd", attn, v_full).reshape(1, seq_len, hidden_size)
    hidden = residual + F.linear(context, state[layer_key(prefix, layer_id, "self_attn.o_proj.weight")])
    residual = hidden
    normed = rms_norm(hidden, state[layer_key(prefix, layer_id, "post_attention_layernorm.weight")])
    gate = F.linear(normed, state[layer_key(prefix, layer_id, "mlp.gate_proj.weight")])
    up = F.linear(normed, state[layer_key(prefix, layer_id, "mlp.up_proj.weight")])
    hidden = residual + F.linear(F.silu(gate) * up, state[layer_key(prefix, layer_id, "mlp.down_proj.weight")])
    kv_summary = {
        "stage_local_only": True,
        "kv_payload_public": False,
        "past_key_values_public": False,
        "k_shape": [int(item) for item in k.shape],
        "v_shape": [int(item) for item in v.shape],
        "layer_count": 1,
    }
    return hidden, kv_summary


def tensor_to_jax_params(config: dict[str, Any], state: dict[str, Any], *, layer_id: int) -> dict[str, Any]:
    import numpy as np

    prefix = "model.layers"

    def np_value(suffix: str, *, transpose: bool = False) -> Any:
        value = state[layer_key(prefix, layer_id, suffix)].detach().cpu().numpy().astype(np.float32)
        return value.T if transpose else value

    return {
        "input_layernorm": np_value("input_layernorm.weight"),
        "post_attention_layernorm": np_value("post_attention_layernorm.weight"),
        "q_w": np_value("self_attn.q_proj.weight", transpose=True),
        "q_b": np_value("self_attn.q_proj.bias"),
        "k_w": np_value("self_attn.k_proj.weight", transpose=True),
        "k_b": np_value("self_attn.k_proj.bias"),
        "v_w": np_value("self_attn.v_proj.weight", transpose=True),
        "v_b": np_value("self_attn.v_proj.bias"),
        "o_w": np_value("self_attn.o_proj.weight", transpose=True),
        "gate_w": np_value("mlp.gate_proj.weight", transpose=True),
        "up_w": np_value("mlp.up_proj.weight", transpose=True),
        "down_w": np_value("mlp.down_proj.weight", transpose=True),
    }


def run_jax_forward(config: dict[str, Any], state: dict[str, Any], hidden: Any, *, layer_id: int, require_tpu: bool) -> dict[str, Any]:
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
            "diagnosis_codes": ["qwen_dense_jax_runtime_missing"],
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
            "jax_devices_public": [
                {"platform": str(getattr(device, "platform", "")), "device_kind": str(getattr(device, "device_kind", ""))}
                for device in devices
            ],
            "blockers": ["jax_tpu_device_missing"],
            "diagnosis_codes": ["qwen_dense_jax_tpu_device_missing"],
        }

    hidden_size = _int(config.get("hidden_size"))
    heads = _int(config.get("num_attention_heads"))
    kv_heads = _int(config.get("num_key_value_heads"), heads)
    head_dim = hidden_size // heads
    repeat_factor = max(1, heads // max(1, kv_heads))
    eps = float(config.get("rms_norm_eps") or 1e-6)
    theta = float(config.get("rope_theta") or 1000000.0)
    seq_len = int(hidden.shape[1])
    params = tensor_to_jax_params(config, state, layer_id=layer_id)
    x_np = hidden.detach().cpu().numpy().astype(np.float32)

    def rotate_half(value: Any) -> Any:
        x1 = value[..., : value.shape[-1] // 2]
        x2 = value[..., value.shape[-1] // 2 :]
        return jnp.concatenate([-x2, x1], axis=-1)

    def apply_rope(q: Any, k: Any) -> tuple[Any, Any]:
        inv_freq = 1.0 / (float(theta) ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
        positions = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", positions, inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)[None, :, None, :]
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

    def rms_norm(value: Any, weight: Any) -> Any:
        variance = jnp.mean(jnp.square(value.astype(jnp.float32)), axis=-1, keepdims=True)
        return (value * jax.lax.rsqrt(variance + jnp.asarray(eps, dtype=jnp.float32))) * weight

    @jax.jit
    def stage_forward(p: dict[str, Any], x: Any) -> tuple[Any, Any, Any]:
        residual = x
        normed = rms_norm(x, p["input_layernorm"])
        q = normed @ p["q_w"] + p["q_b"]
        k = normed @ p["k_w"] + p["k_b"]
        v = normed @ p["v_w"] + p["v_b"]
        q = jnp.reshape(q, (1, seq_len, heads, head_dim))
        k = jnp.reshape(k, (1, seq_len, kv_heads, head_dim))
        v = jnp.reshape(v, (1, seq_len, kv_heads, head_dim))
        q, k = apply_rope(q, k)
        k_full = jnp.repeat(k, repeat_factor, axis=2)
        v_full = jnp.repeat(v, repeat_factor, axis=2)
        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k_full) / jnp.sqrt(jnp.asarray(head_dim, dtype=jnp.float32))
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        scores = jnp.where(causal[None, None, :, :], scores, jnp.asarray(-1e4, dtype=jnp.float32))
        attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
        context = jnp.einsum("bhqk,bkhd->bqhd", attn, v_full)
        x = residual + jnp.reshape(context, (1, seq_len, hidden_size)) @ p["o_w"]
        residual = x
        normed = rms_norm(x, p["post_attention_layernorm"])
        x = residual + (jax.nn.silu(normed @ p["gate_w"]) * (normed @ p["up_w"])) @ p["down_w"]
        return x, k, v

    device = tpu_devices[0] if tpu_devices else devices[0]
    output, k, v = stage_forward(jax.device_put(params, device), jax.device_put(x_np, device))
    output_np = jax.device_get(output)
    k_shape = [int(item) for item in k.shape]
    v_shape = [int(item) for item in v.shape]
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
        "output_shape": [int(item) for item in output_np.shape],
        "output_summary": {
            "shape": [int(item) for item in output_np.shape],
            "mean": round(float(output_np.mean()), 8),
            "std": round(float(output_np.std()), 8),
            "payload_public": False,
        },
        "stage_local_kv_cache": {
            "stage_local_only": True,
            "kv_payload_public": False,
            "past_key_values_public": False,
            "k_shape": k_shape,
            "v_shape": v_shape,
            "layer_count": 1,
        },
        "blockers": [],
        "diagnosis_codes": ["qwen_dense_jax_stage_forward_ready"],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    diagnosis: list[str] = []
    config = load_config(args)
    layer_id = int(args.layer_id)
    source_summary: dict[str, Any]
    try:
        if args.model_path:
            state, source_summary = load_stage_state_from_safetensors(Path(args.model_path), config, layer_id=layer_id)
        else:
            state = build_fixture_torch_state(config, seed=args.seed, layer_id=layer_id)
            source_summary = {
                "source": "fixture_dense_qwen_stage_owned_weights",
                "stage_owned_key_count": len(state),
                "stage_owned_file_count": 0,
                "weight_tensor_values_public": False,
            }
    except Exception as exc:
        state = {}
        source_summary = {
            "source": "attached_safetensors",
            "error_type": type(exc).__name__,
            "error_digest": stable_hash(str(exc)),
            "weight_tensor_values_public": False,
        }
        blockers.append("stage_owned_dense_weight_loading_failed")
        diagnosis.append("qwen_dense_stage_owned_weight_loading_failed")

    torch_ready = False
    torch_summary: dict[str, Any] = {}
    kv_summary: dict[str, Any] = {}
    jax_result: dict[str, Any] = {}
    compare_summary: dict[str, Any] = {}
    if state:
        try:
            import torch

            hidden = torch.linspace(-0.25, 0.25, steps=args.sequence_length * _int(config.get("hidden_size")), dtype=torch.float32)
            hidden = hidden.reshape(1, args.sequence_length, _int(config.get("hidden_size")))
            output, kv_summary = torch_stage_forward(config, state, hidden, layer_id=layer_id)
            torch_summary = torch_tensor_summary(output)
            torch_ready = True
            diagnosis.append("qwen_dense_torch_reference_forward_ready")
            if args.run_jax:
                jax_result = run_jax_forward(config, state, hidden, layer_id=layer_id, require_tpu=args.require_tpu)
                diagnosis.extend(str(item) for item in jax_result.get("diagnosis_codes") or [])
                blockers.extend(str(item) for item in jax_result.get("blockers") or [])
                if jax_result.get("ok") is True:
                    jax_output_summary = _dict(jax_result.get("output_summary"))
                    mean_delta = abs(float(torch_summary["mean"]) - float(jax_output_summary.get("mean", 0.0)))
                    std_delta = abs(float(torch_summary["std"]) - float(jax_output_summary.get("std", 0.0)))
                    shape_match = torch_summary["shape"] == jax_result.get("output_shape")
                    compare_summary = {
                        "shape_match": shape_match,
                        "mean_delta": round(mean_delta, 8),
                        "std_delta": round(std_delta, 8),
                        "tolerance": float(args.compare_tolerance),
                        "reference_payload_public": False,
                        "jax_payload_public": False,
                    }
                    if not (shape_match and mean_delta <= args.compare_tolerance and std_delta <= args.compare_tolerance):
                        blockers.append("jax_torch_reference_mismatch")
                        diagnosis.append("qwen_dense_jax_torch_reference_mismatch")
            else:
                blockers.append("jax_execution_not_requested")
                diagnosis.append("qwen_dense_jax_execution_not_requested")
        except Exception as exc:
            blockers.append("torch_reference_forward_failed")
            diagnosis.append("qwen_dense_torch_reference_forward_failed")
            torch_summary = {"error_type": type(exc).__name__, "error_digest": stable_hash(str(exc))}

    jax_runtime_ready = bool(jax_result.get("jax_runtime_execution_ready") is True)
    tpu_ready = bool(jax_result.get("tpu_runtime_ready") is True)
    adapter_ready = bool(torch_ready and jax_runtime_ready and (not args.require_tpu or tpu_ready) and not any(item == "jax_torch_reference_mismatch" for item in blockers))
    if args.require_tpu and not tpu_ready:
        blockers.append("jax_tpu_runtime_not_available")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": adapter_ready,
        "qwen_dense_stage_adapter_smoke_ready": adapter_ready,
        "model_source": source_summary,
        "model_path_used": bool(args.model_path),
        "model_path_public": bool(args.model_path and args.public_model_path),
        "model_type": str(config.get("model_type") or ""),
        "architectures": list(config.get("architectures") or []),
        "dense_full_precision_only": True,
        "quantized_weight_adapter_used": False,
        "layer_id": layer_id,
        "sequence_length": int(args.sequence_length),
        "shape_metadata": {
            "input_shape": [1, int(args.sequence_length), _int(config.get("hidden_size"))],
            "output_shape": torch_summary.get("shape") or [],
            "layout": "batch_seq_hidden",
            "dtype": "float32_reference_bfloat16_compatible",
            "shape_public": True,
            "activation_payload_public": False,
        },
        "qwen_components_exercised": {
            "rms_norm": torch_ready,
            "rope": torch_ready,
            "grouped_query_attention": torch_ready and _int(config.get("num_key_value_heads"), _int(config.get("num_attention_heads"))) != _int(config.get("num_attention_heads")),
            "causal_attention": torch_ready,
            "swiglu_mlp": torch_ready,
            "stage_local_kv_cache": bool(kv_summary),
        },
        "torch_reference_forward_ready": torch_ready,
        "torch_reference_summary_hash": stable_hash(torch_summary) if torch_summary else "",
        "torch_reference_summary": torch_summary,
        "jax_runtime_execution_requested": bool(args.run_jax),
        "jax_runtime_execution_ready": jax_runtime_ready,
        "tpu_runtime_required": bool(args.require_tpu),
        "tpu_runtime_ready": tpu_ready,
        "jax_summary_hash": stable_hash(jax_result) if jax_result else "",
        "jax_result": jax_result,
        "torch_jax_comparison": compare_summary,
        "tpu_jax_qwen_stage_runtime_ready": bool(adapter_ready and tpu_ready),
        "stage_local_kv_cache_verified": bool(torch_ready and (not args.run_jax or jax_runtime_ready)),
        "stage_local_kv_cache_metadata": kv_summary,
        "blockers": sorted(set(item for item in blockers if item)),
        "diagnosis_codes": sorted(set(diagnosis or ["qwen_dense_stage_adapter_smoke_not_started"])),
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
            "private_runtime_state_public": False,
            "weight_tensor_values_public": False,
        },
        "public_artifact_safe": True,
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["qwen_dense_stage_adapter_smoke_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"].append("public_redaction_scan_failed")
        report["diagnosis_codes"].append("public_redaction_scan_failed")
        report["redaction_errors"] = leaks
    summary_path = output_dir / "qwen_dense_jax_tpu_stage_adapter_smoke.json"
    write_json(summary_path, report)
    report["artifacts"] = {
        "summary_json": artifact_entry(summary_path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(report.get("ok"))),
    }
    write_json(summary_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense Qwen stage adapter smoke with optional JAX/TPU.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-json", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--public-model-path", action="store_true")
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--seed", type=int, default=230626)
    parser.add_argument("--run-jax", action="store_true")
    parser.add_argument("--require-tpu", action="store_true")
    parser.add_argument("--compare-tolerance", type=float, default=2e-4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.sequence_length < 1 or args.sequence_length > 2048:
        raise SystemExit("--sequence-length must be between 1 and 2048")
    if args.layer_id < 0:
        raise SystemExit("--layer-id must be non-negative")
    if args.config_json and not Path(args.config_json).is_file():
        raise SystemExit("--config-json must point to an existing JSON file")
    if args.model_path and not Path(args.model_path).is_dir():
        raise SystemExit("--model-path must point to an existing directory")
    if args.require_tpu and not args.run_jax:
        raise SystemExit("--require-tpu requires --run-jax")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {Path(args.output_dir) / 'qwen_dense_jax_tpu_stage_adapter_smoke.json'}")
        print(f"Adapter smoke ready: {report.get('qwen_dense_stage_adapter_smoke_ready')}")
        print(f"JAX runtime ready: {report.get('jax_runtime_execution_ready')}")
        print(f"TPU runtime ready: {report.get('tpu_runtime_ready')}")
        if report.get("blockers"):
            print("Blockers: " + ", ".join(str(item) for item in report.get("blockers") or []))
    return 0 if report.get("public_artifact_safe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
