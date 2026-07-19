#!/usr/bin/env python3
"""Probe DeepSeek-V4-Flash metadata and Kaggle Web TPU adapter readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import gpu_tpu_cpu_same_request_runtime_bridge_probe as web_tpu_bridge  # noqa: E402


SCHEMA = "deepseek_v4_flash_kaggle_web_tpu_stage_adapter_probe_v1"
CELL_SCHEMA = "deepseek_v4_flash_kaggle_web_tpu_stage_adapter_cell_v1"
DEFAULT_OUTPUT_DIR = "dist/deepseek-v4-flash-kaggle-web-tpu-stage-adapter-probe"
DEFAULT_NOTEBOOK_URL = "https://www.kaggle.com/code/tpuowner/notebook8d4184babd/edit"
DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_LAYER_START = 16
DEFAULT_LAYER_END = 18
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Authorization:",
    "Cookie:",
    "Set-Cookie",
    "jupyter-proxy",
    "token=",
    "XSRF-TOKEN",
    "_xsrf",
    "kaggle_session",
    "jupyterServerHttpUrl",
    '"prompt":',
    '"generated_text":',
    '"generated_token_ids":',
    '"activation":',
    '"hidden_state":',
    '"logits":',
    '"kv_cache":',
    '"past_key_values":',
)


DEEPSEEK_V4_TPU_FIXTURE_FORWARD_SOURCE = r'''
def run_deepseek_v4_tpu_fixture_stage_forward(report):
    """Run a tiny DeepSeek-V4-shaped MLA/MoE stage on the current TPU."""
    try:
        import jax
        import jax.numpy as jnp

        devices = list(jax.devices())
        tpu_devices = [device for device in devices if str(getattr(device, "platform", "")).lower() == "tpu"]
        if not tpu_devices:
            return {
                "ok": False,
                "blockers": ["jax_tpu_device_missing"],
                "diagnosis_codes": ["deepseek_v4_flash_jax_tpu_device_missing"],
            }

        seq = 2
        hidden = 16
        hc = 2
        heads = 2
        head_dim = 8
        q_rank = 4
        o_groups = 1
        o_rank = 2
        experts = 2
        top_k = 1
        inter = 8
        eps = 1.0e-6

        def arr(shape, scale=0.025, offset=0):
            size = 1
            for dim in shape:
                size *= int(dim)
            values = jnp.arange(offset, offset + size, dtype=jnp.float32)
            return (jnp.reshape(jnp.sin(values * 0.017), shape) * scale).astype(jnp.float32)

        p = {
            "self_attn.sinks": jnp.zeros((heads,), dtype=jnp.float32),
            "self_attn.q_a_proj.weight": arr((q_rank, hidden), offset=1),
            "self_attn.q_a_norm.weight": jnp.ones((q_rank,), dtype=jnp.float32),
            "self_attn.q_b_proj.weight": arr((heads * head_dim, q_rank), offset=1000),
            "self_attn.kv_proj.weight": arr((head_dim, hidden), offset=2000),
            "self_attn.kv_norm.weight": jnp.ones((head_dim,), dtype=jnp.float32),
            "self_attn.o_a_proj.weight": arr((o_groups * o_rank, heads * head_dim // o_groups), offset=3000),
            "self_attn.o_b_proj.weight": arr((hidden, o_groups * o_rank), offset=4000),
            "mlp.gate.weight": arr((experts, hidden), offset=5000),
            "mlp.gate.e_score_correction_bias": jnp.zeros((experts,), dtype=jnp.float32),
            "mlp.experts.gate_up_proj": arr((experts, 2 * inter, hidden), offset=6000),
            "mlp.experts.down_proj": arr((experts, hidden, inter), offset=7000),
            "mlp.shared_experts.gate_proj.weight": arr((inter, hidden), offset=8000),
            "mlp.shared_experts.up_proj.weight": arr((inter, hidden), offset=9000),
            "mlp.shared_experts.down_proj.weight": arr((hidden, inter), offset=10000),
            "input_layernorm.weight": jnp.ones((hidden,), dtype=jnp.float32),
            "post_attention_layernorm.weight": jnp.ones((hidden,), dtype=jnp.float32),
            "attn_hc.fn": arr(((2 + hc) * hc, hc * hidden), offset=11000),
            "attn_hc.base": jnp.zeros(((2 + hc) * hc,), dtype=jnp.float32),
            "attn_hc.scale": jnp.ones((3,), dtype=jnp.float32),
            "ffn_hc.fn": arr(((2 + hc) * hc, hc * hidden), offset=12000),
            "ffn_hc.base": jnp.zeros(((2 + hc) * hc,), dtype=jnp.float32),
            "ffn_hc.scale": jnp.ones((3,), dtype=jnp.float32),
        }
        x = jnp.reshape(jnp.linspace(-0.15, 0.15, seq * hc * hidden, dtype=jnp.float32), (1, seq, hc, hidden))

        def rms_norm(value, weight=None):
            out = value * jax.lax.rsqrt(jnp.mean(jnp.square(value.astype(jnp.float32)), axis=-1, keepdims=True) + eps)
            return out if weight is None else out * weight

        def rotate_half(value):
            even = value[..., 0::2]
            odd = value[..., 1::2]
            return jnp.reshape(jnp.stack((-odd, even), axis=-1), value.shape)

        def apply_rope(value, layout="bhsd"):
            rope_dim = 4
            inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, rope_dim, 2, dtype=jnp.float32) / rope_dim))
            freqs = jnp.einsum("i,j->ij", jnp.arange(seq, dtype=jnp.float32), inv_freq)
            if layout == "bhsd":
                cos = jnp.repeat(jnp.cos(freqs), 2, axis=-1)[None, None, :, :]
                sin = jnp.repeat(jnp.sin(freqs), 2, axis=-1)[None, None, :, :]
            else:
                cos = jnp.repeat(jnp.cos(freqs), 2, axis=-1)[None, :, None, :]
                sin = jnp.repeat(jnp.sin(freqs), 2, axis=-1)[None, :, None, :]
            nope = value[..., :-rope_dim]
            rope = value[..., -rope_dim:]
            return jnp.concatenate([nope, rope * cos + rotate_half(rope) * sin], axis=-1)

        def hc_map(prefix, hidden_streams):
            flat = rms_norm(jnp.reshape(hidden_streams, (1, seq, hc * hidden)))
            logits = flat @ jnp.swapaxes(p[prefix + ".fn"], -1, -2)
            pre_w, post_w, comb_w = jnp.split(logits, [hc, 2 * hc], axis=-1)
            pre_b, post_b, comb_b = jnp.split(p[prefix + ".base"], [hc, 2 * hc])
            pre_scale, post_scale, comb_scale = p[prefix + ".scale"]
            pre = jax.nn.sigmoid(pre_w * pre_scale + pre_b) + 1.0e-8
            post = 2.0 * jax.nn.sigmoid(post_w * post_scale + post_b)
            comb = jnp.reshape(comb_w, (1, seq, hc, hc)) * comb_scale + jnp.reshape(comb_b, (hc, hc))
            comb = jax.nn.softmax(comb, axis=-1) + 1.0e-8
            comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + 1.0e-8)
            comb = comb / (jnp.sum(comb, axis=-1, keepdims=True) + 1.0e-8)
            comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + 1.0e-8)
            collapsed = jnp.sum(pre[..., None] * hidden_streams, axis=2)
            return post, comb, collapsed

        def attention(collapsed):
            normed = rms_norm(collapsed, p["input_layernorm.weight"])
            q_residual = rms_norm(normed @ jnp.swapaxes(p["self_attn.q_a_proj.weight"], -1, -2), p["self_attn.q_a_norm.weight"])
            q = q_residual @ jnp.swapaxes(p["self_attn.q_b_proj.weight"], -1, -2)
            q = jnp.transpose(jnp.reshape(q, (1, seq, heads, head_dim)), (0, 2, 1, 3))
            q = rms_norm(q)
            kv = rms_norm(normed @ jnp.swapaxes(p["self_attn.kv_proj.weight"], -1, -2), p["self_attn.kv_norm.weight"])
            kv = jnp.transpose(jnp.reshape(kv, (1, seq, 1, head_dim)), (0, 2, 1, 3))
            q = apply_rope(q)
            kv = apply_rope(kv)
            kv_full = jnp.repeat(kv, heads, axis=1)
            scores = jnp.matmul(q, jnp.swapaxes(kv_full, -1, -2)) / jnp.sqrt(jnp.asarray(head_dim, dtype=jnp.float32))
            causal = jnp.tril(jnp.ones((seq, seq), dtype=bool))[None, None, :, :]
            scores = jnp.where(causal, scores, jnp.asarray(-1.0e4, dtype=jnp.float32))
            sinks = jnp.reshape(p["self_attn.sinks"], (1, heads, 1, 1))
            probs = jax.nn.softmax(jnp.concatenate([scores, jnp.broadcast_to(sinks, (1, heads, seq, 1))], axis=-1), axis=-1)[..., :-1]
            context = jnp.matmul(probs, kv_full)
            context = jnp.transpose(apply_rope(jnp.transpose(context, (0, 2, 1, 3)), layout="bshd"), (0, 2, 1, 3))
            grouped = jnp.reshape(jnp.transpose(context, (0, 2, 1, 3)), (1, seq, o_groups, -1))
            o_a_w = jnp.reshape(p["self_attn.o_a_proj.weight"], (o_groups, o_rank, -1))
            grouped_out = jnp.reshape(jnp.einsum("bsgh,goh->bsgo", grouped, o_a_w), (1, seq, o_groups * o_rank))
            return grouped_out @ jnp.swapaxes(p["self_attn.o_b_proj.weight"], -1, -2), kv, kv

        def moe(collapsed):
            normed = rms_norm(collapsed, p["post_attention_layernorm.weight"])
            flat = jnp.reshape(normed, (seq, hidden))
            scores = jax.nn.sigmoid(flat @ jnp.swapaxes(p["mlp.gate.weight"], -1, -2) + p["mlp.gate.e_score_correction_bias"])
            weights, indices = jax.lax.top_k(scores, top_k)
            weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1.0e-20)
            routed = jnp.zeros_like(flat)
            for expert_id in range(experts):
                gate_up = flat @ jnp.swapaxes(p["mlp.experts.gate_up_proj"][expert_id], -1, -2)
                gate, up = jnp.split(gate_up, 2, axis=-1)
                current = jax.nn.silu(jnp.minimum(gate, 7.0)) * jnp.clip(up, -7.0, 7.0)
                current = current @ jnp.swapaxes(p["mlp.experts.down_proj"][expert_id], -1, -2)
                mask = (indices == expert_id).astype(jnp.float32)
                routed = routed + current * jnp.sum(mask * weights, axis=-1, keepdims=True)
            shared = jax.nn.silu(flat @ jnp.swapaxes(p["mlp.shared_experts.gate_proj.weight"], -1, -2)) * (
                flat @ jnp.swapaxes(p["mlp.shared_experts.up_proj.weight"], -1, -2)
            )
            shared = shared @ jnp.swapaxes(p["mlp.shared_experts.down_proj.weight"], -1, -2)
            return jnp.reshape(routed + shared, (1, seq, hidden))

        @jax.jit
        def stage_forward(x):
            post, comb, collapsed = hc_map("attn_hc", x)
            attn_output, k, v = attention(collapsed)
            x = post[..., None] * attn_output[:, :, None, :] + jnp.einsum("bsji,bsjd->bsid", comb, x)
            post, comb, collapsed = hc_map("ffn_hc", x)
            mlp_output = moe(collapsed)
            x = post[..., None] * mlp_output[:, :, None, :] + jnp.einsum("bsji,bsjd->bsid", comb, x)
            return x, k, v

        output, k, v = stage_forward(jax.device_put(x, tpu_devices[0]))
        output = output.block_until_ready()
        summary = {
            "input_shape": [1, seq, hc, hidden],
            "output_shape": [int(item) for item in output.shape],
            "dtype": str(output.dtype),
            "output_mean": round(float(jnp.mean(output)), 8),
            "output_std": round(float(jnp.std(output)), 8),
            "output_finite": bool(jnp.all(jnp.isfinite(output))),
            "stage_output_hash": sha_payload({
                "shape": [int(item) for item in output.shape],
                "mean": round(float(jnp.mean(output)), 8),
                "std": round(float(jnp.std(output)), 8),
            }),
            "stage_local_kv_cache_metadata": {
                "stage_local_only": True,
                "k_shape": [int(item) for item in k.shape],
                "v_shape": [int(item) for item in v.shape],
                "kv_payload_public": False,
                "past_key_values_public": False,
            },
            "components_exercised": {
                "manifold_hyper_connections": True,
                "mla_shared_kv_attention": True,
                "grouped_output_projection": True,
                "attention_sink": True,
                "topk_moe_router": True,
                "routed_experts": True,
                "shared_experts": True,
                "stage_local_kv_cache_shape": True,
            },
            "fixture_weights": True,
            "real_weight_tensor_values_loaded": False,
            "activation_payload_public": False,
            "kv_cache_public": False,
        }
        return {
            "ok": True,
            "summary": summary,
            "blockers": [],
            "diagnosis_codes": ["deepseek_v4_flash_jax_tpu_fixture_stage_forward_ready"],
        }
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_digest": sha_payload(str(exc)),
            "blockers": ["deepseek_v4_flash_jax_tpu_fixture_stage_forward_failed"],
            "diagnosis_codes": ["deepseek_v4_flash_jax_tpu_fixture_stage_forward_failed"],
        }
'''


DEEPSEEK_V4_REAL_TENSOR_LOAD_SOURCE = r'''
def run_deepseek_v4_real_weight_tpu_tensor_load(report):
    """Range-load a small real DeepSeek-V4 safetensors slice and place it on TPU."""
    try:
        import struct
        import urllib.error
        import urllib.request

        import jax
        import jax.numpy as jnp
        import numpy as np

        devices = list(jax.devices())
        tpu_devices = [device for device in devices if str(getattr(device, "platform", "")).lower() == "tpu"]
        if not tpu_devices:
            return {
                "ok": False,
                "blockers": ["jax_tpu_device_missing"],
                "diagnosis_codes": ["deepseek_v4_flash_jax_tpu_device_missing"],
            }

        index = load_hf_json("model.safetensors.index.json")
        weight_map = index.get("weight_map") if isinstance(index.get("weight_map"), dict) else {}
        candidate_keys = [
            "layers." + str(LAYER_START) + ".attn.attn_sink",
            "layers." + str(LAYER_START) + ".attn_norm.weight",
            "layers." + str(LAYER_START) + ".ffn_norm.weight",
            "layers." + str(LAYER_START) + ".ffn.gate.bias",
            "layers." + str(LAYER_START) + ".ffn.gate.weight",
            "layers." + str(LAYER_START) + ".attn.wq_a.weight",
            "layers." + str(LAYER_START) + ".attn.wq_a.scale",
            "layers." + str(LAYER_START) + ".ffn.experts.0.w1.weight",
            "layers." + str(LAYER_START) + ".ffn.experts.0.w1.scale",
            "layers." + str(LAYER_START) + ".ffn.experts.0.w2.weight",
            "layers." + str(LAYER_START) + ".ffn.experts.0.w2.scale",
            "layers." + str(LAYER_START) + ".ffn.experts.0.w3.weight",
            "layers." + str(LAYER_START) + ".ffn.experts.0.w3.scale",
        ]
        selected_keys = [key for key in candidate_keys if key in weight_map]
        if not selected_keys:
            return {
                "ok": False,
                "blockers": ["deepseek_v4_flash_real_weight_sample_keys_missing"],
                "diagnosis_codes": ["deepseek_v4_flash_real_weight_sample_keys_missing"],
            }

        def read_range(filename, start, end, max_bytes):
            request = urllib.request.Request(
                "https://huggingface.co/" + MODEL_ID + "/resolve/main/" + filename,
                headers={
                    "Range": "bytes=" + str(int(start)) + "-" + str(int(end)),
                    "User-Agent": "crowdtensor-deepseek-v4-web-tpu-real-weight-smoke/1",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    status = int(getattr(response, "status", response.getcode()))
                    content_range = response.headers.get("Content-Range", "")
                    payload = response.read(int(max_bytes) + 1)
            except urllib.error.HTTPError as exc:
                raise RuntimeError("hf_range_http_" + str(exc.code)) from exc
            if status != 206 and not str(content_range).lower().startswith("bytes "):
                raise RuntimeError("hf_range_not_honored")
            if len(payload) > int(max_bytes):
                raise RuntimeError("hf_range_response_exceeded_budget")
            return payload

        header_cache = {}

        def load_header(filename):
            if filename in header_cache:
                return header_cache[filename]
            prefix = read_range(filename, 0, 7, 8)
            if len(prefix) != 8:
                raise RuntimeError("safetensors_header_prefix_missing")
            header_len = int(struct.unpack("<Q", prefix)[0])
            if header_len <= 0 or header_len > 128 * 1024 * 1024:
                raise RuntimeError("safetensors_header_length_out_of_budget")
            header_blob = read_range(filename, 8, 8 + header_len - 1, header_len)
            if len(header_blob) != header_len:
                raise RuntimeError("safetensors_header_truncated")
            header = json.loads(header_blob.decode("utf-8"))
            header_cache[filename] = (header_len, header if isinstance(header, dict) else {})
            return header_cache[filename]

        def decode_tensor(meta, payload):
            dtype = str(meta.get("dtype") or "")
            shape = [int(item) for item in (meta.get("shape") or [])]
            if dtype == "F32":
                array = np.frombuffer(payload, dtype="<f4").reshape(shape).astype(np.float32, copy=False)
            elif dtype == "BF16":
                raw = np.frombuffer(payload, dtype="<u2")
                array = (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)
            elif dtype == "I8":
                array = np.frombuffer(payload, dtype=np.int8).reshape(shape).astype(np.float32)
            elif dtype == "F8_E8M0":
                raw = np.frombuffer(payload, dtype=np.uint8).astype(np.int16)
                # UE8M0 is an unsigned exponent-only scale. Keep this smoke finite
                # while preserving deterministic scale ordering for public hashes.
                array = np.exp2(np.clip(raw - 127, -32, 32).astype(np.float32)).reshape(shape)
            elif dtype == "F8_E4M3":
                raw = np.frombuffer(payload, dtype=np.uint8)
                sign = np.where((raw & 0x80) == 0, 1.0, -1.0).astype(np.float32)
                exp = ((raw >> 3) & 0x0F).astype(np.int16)
                mant = (raw & 0x07).astype(np.float32)
                normal = (1.0 + mant / 8.0) * np.exp2((exp - 7).astype(np.float32))
                subnormal = (mant / 8.0) * np.exp2(np.asarray(-6.0, dtype=np.float32))
                value = np.where(exp == 0, subnormal, normal)
                value = np.where((raw & 0x7F) == 0, 0.0, value)
                array = (sign * value).reshape(shape).astype(np.float32)
            else:
                raise RuntimeError("unsupported_real_weight_sample_dtype_" + dtype)
            return array

        loaded = []
        loaded_arrays = {}
        dtype_counts = {}
        total_bytes = 0
        device_put_count = 0
        finite_count = 0
        for key in selected_keys:
            filename = str(weight_map[key])
            header_len, header = load_header(filename)
            meta = header.get(key)
            if not isinstance(meta, dict):
                raise RuntimeError("safetensors_key_missing")
            offsets = meta.get("data_offsets") or []
            if len(offsets) != 2:
                raise RuntimeError("safetensors_offsets_missing")
            start, end = int(offsets[0]), int(offsets[1])
            byte_length = end - start
            if byte_length <= 0 or byte_length > 8 * 1024 * 1024:
                raise RuntimeError("real_weight_sample_tensor_out_of_budget")
            payload = read_range(filename, 8 + header_len + start, 8 + header_len + end - 1, byte_length)
            if len(payload) != byte_length:
                raise RuntimeError("real_weight_sample_tensor_truncated")
            array = decode_tensor(meta, payload)
            device_array = jax.device_put(jnp.asarray(array, dtype=jnp.float32), tpu_devices[0])
            finite = bool(jnp.all(jnp.isfinite(device_array)).block_until_ready())
            loaded_arrays[key] = device_array
            device_put_count += 1
            finite_count += int(finite)
            dtype = str(meta.get("dtype") or "")
            dtype_counts[dtype] = int(dtype_counts.get(dtype, 0)) + 1
            total_bytes += int(byte_length)
            loaded.append({
                "key_digest": sha_payload(key),
                "file_digest": sha_payload(filename),
                "dtype": dtype,
                "shape": [int(item) for item in array.shape],
                "byte_length": int(byte_length),
                "raw_payload_sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "device_put_ready": True,
                "finite_on_device": finite,
                "weight_tensor_values_public": False,
            })

        router_smoke = {
            "ready": False,
            "weight_tensor_values_public": False,
            "activation_payload_public": False,
        }
        fp8_block_smoke = {
            "ready": False,
            "weight_tensor_values_public": False,
            "activation_payload_public": False,
        }
        i8_expert_smoke = {
            "ready": False,
            "weight_tensor_values_public": False,
            "activation_payload_public": False,
        }
        i8_expert_mlp_slice_smoke = {
            "ready": False,
            "weight_tensor_values_public": False,
            "activation_payload_public": False,
        }
        router_norm = None
        routed_topk_indices = []
        routed_topk_weights = []
        gate_key = "layers." + str(LAYER_START) + ".ffn.gate.weight"
        gate_bias_key = "layers." + str(LAYER_START) + ".ffn.gate.bias"
        norm_key = "layers." + str(LAYER_START) + ".ffn_norm.weight"
        if gate_key in loaded_arrays and norm_key in loaded_arrays:
            hidden = int(loaded_arrays[norm_key].shape[0])
            dummy = jnp.linspace(-0.01, 0.01, hidden, dtype=jnp.float32)
            norm = dummy * jax.lax.rsqrt(jnp.mean(jnp.square(dummy)) + 1.0e-6) * loaded_arrays[norm_key]
            router_norm = norm
            bias = loaded_arrays.get(gate_bias_key)
            if bias is None:
                bias = jnp.zeros((int(loaded_arrays[gate_key].shape[0]),), dtype=jnp.float32)
            logits = jnp.matmul(loaded_arrays[gate_key], router_norm) + bias
            scores = jnp.sqrt(jax.nn.softplus(logits))
            values, indices = jax.lax.top_k(scores, min(6, int(scores.shape[0])))
            weights = values / (jnp.sum(values) + 1.0e-20) * 1.5
            routed_topk_indices = [int(item) for item in np.asarray(indices)]
            routed_topk_weights = [float(item) for item in np.asarray(weights, dtype=np.float32)]
            finite_router = bool(jnp.all(jnp.isfinite(values)).block_until_ready())
            router_smoke = {
                "ready": bool(finite_router),
                "router_kind": "deepseek_v4_moe_gate_topk",
                "input_shape": [hidden],
                "gate_shape": [int(item) for item in loaded_arrays[gate_key].shape],
                "gate_bias_shape": [int(item) for item in bias.shape],
                "top_k": int(values.shape[0]),
                "topk_index_digest": sha_payload([int(item) for item in np.asarray(indices)]),
                "topk_weight_hash": "sha256:" + hashlib.sha256(np.asarray(weights, dtype=np.float32).tobytes()).hexdigest(),
                "topk_value_hash": "sha256:" + hashlib.sha256(np.asarray(values, dtype=np.float32).tobytes()).hexdigest(),
                "finite_on_device": finite_router,
                "weight_tensor_values_public": False,
                "activation_payload_public": False,
            }
        fp8_key = "layers." + str(LAYER_START) + ".attn.wq_a.weight"
        fp8_scale_key = "layers." + str(LAYER_START) + ".attn.wq_a.scale"
        if fp8_key in loaded_arrays and fp8_scale_key in loaded_arrays:
            block_size = 128
            weight_block = loaded_arrays[fp8_key][:block_size, :block_size]
            scale = loaded_arrays[fp8_scale_key][0, 0]
            dummy = jnp.linspace(-0.02, 0.02, block_size, dtype=jnp.float32)
            out = jnp.matmul(weight_block * scale, dummy)
            out = out.block_until_ready()
            finite_fp8 = bool(jnp.all(jnp.isfinite(out)).block_until_ready())
            fp8_block_smoke = {
                "ready": bool(finite_fp8),
                "smoke_kind": "deepseek_v4_fp8_e4m3_ue8m0_block_dequant_matmul",
                "weight_block_shape": [block_size, block_size],
                "scale_shape": [int(item) for item in loaded_arrays[fp8_scale_key].shape],
                "block_size": [128, 128],
                "output_shape": [int(item) for item in out.shape],
                "output_hash": "sha256:" + hashlib.sha256(np.asarray(out, dtype=np.float32).tobytes()).hexdigest(),
                "finite_on_device": finite_fp8,
                "weight_tensor_values_public": False,
                "activation_payload_public": False,
            }
        i8_key = "layers." + str(LAYER_START) + ".ffn.experts.0.w1.weight"
        i8_scale_key = "layers." + str(LAYER_START) + ".ffn.experts.0.w1.scale"
        if i8_key in loaded_arrays and i8_scale_key in loaded_arrays:
            block_size = 128
            weight_block = loaded_arrays[i8_key][:block_size, :block_size]
            scale = loaded_arrays[i8_scale_key][:block_size, :]
            full_input_dim = int(loaded_arrays[i8_key].shape[1])
            group_size = max(1, full_input_dim // int(scale.shape[1]))
            scale_group_count = (block_size + group_size - 1) // group_size
            expanded_scale = jnp.repeat(scale[:, :scale_group_count], group_size, axis=1)[:, :block_size]
            dummy = jnp.linspace(-0.015, 0.015, block_size, dtype=jnp.float32)
            out = jnp.matmul(weight_block * expanded_scale, dummy)
            out = out.block_until_ready()
            finite_i8 = bool(jnp.all(jnp.isfinite(out)).block_until_ready())
            i8_expert_smoke = {
                "ready": bool(finite_i8),
                "smoke_kind": "deepseek_v4_i8_ue8m0_expert_w1_block_dequant_matmul",
                "expert_id": 0,
                "weight_block_shape": [block_size, block_size],
                "scale_shape": [int(item) for item in loaded_arrays[i8_scale_key].shape],
                "scale_group_size": int(group_size),
                "scale_group_count_used": int(scale_group_count),
                "output_shape": [int(item) for item in out.shape],
                "output_hash": "sha256:" + hashlib.sha256(np.asarray(out, dtype=np.float32).tobytes()).hexdigest(),
                "finite_on_device": finite_i8,
                "weight_tensor_values_public": False,
                "activation_payload_public": False,
            }
        i8_w2_key = "layers." + str(LAYER_START) + ".ffn.experts.0.w2.weight"
        i8_w2_scale_key = "layers." + str(LAYER_START) + ".ffn.experts.0.w2.scale"
        i8_w3_key = "layers." + str(LAYER_START) + ".ffn.experts.0.w3.weight"
        i8_w3_scale_key = "layers." + str(LAYER_START) + ".ffn.experts.0.w3.scale"

        def dequant_i8_block(weight_key, scale_key, rows, cols):
            weight_block = loaded_arrays[weight_key][:rows, :cols]
            scale = loaded_arrays[scale_key][:rows, :]
            full_input_dim = int(loaded_arrays[weight_key].shape[1])
            group_size = max(1, full_input_dim // int(scale.shape[1]))
            scale_group_count = (cols + group_size - 1) // group_size
            expanded_scale = jnp.repeat(scale[:, :scale_group_count], group_size, axis=1)[:, :cols]
            return weight_block * expanded_scale, group_size, scale_group_count

        if all(key in loaded_arrays for key in [i8_key, i8_scale_key, i8_w2_key, i8_w2_scale_key, i8_w3_key, i8_w3_scale_key]):
            block_size = 128
            w1, w1_group_size, w1_group_count = dequant_i8_block(i8_key, i8_scale_key, block_size, block_size)
            w3, w3_group_size, w3_group_count = dequant_i8_block(i8_w3_key, i8_w3_scale_key, block_size, block_size)
            w2, w2_group_size, w2_group_count = dequant_i8_block(i8_w2_key, i8_w2_scale_key, block_size, block_size)
            dummy = jnp.linspace(-0.015, 0.015, block_size, dtype=jnp.float32)
            gate = jnp.matmul(w1, dummy)
            up = jnp.matmul(w3, dummy)
            intermediate = jax.nn.silu(jnp.minimum(gate, 7.0)) * jnp.clip(up, -7.0, 7.0)
            out = jnp.matmul(w2, intermediate)
            out = out.block_until_ready()
            finite_mlp = bool(jnp.all(jnp.isfinite(out)).block_until_ready())
            i8_expert_mlp_slice_smoke = {
                "ready": bool(finite_mlp),
                "smoke_kind": "deepseek_v4_i8_ue8m0_expert_mlp_slice_forward",
                "expert_id": 0,
                "input_shape": [block_size],
                "intermediate_shape": [block_size],
                "output_shape": [int(item) for item in out.shape],
                "w1_block_shape": [block_size, block_size],
                "w2_block_shape": [block_size, block_size],
                "w3_block_shape": [block_size, block_size],
                "w1_scale_group_size": int(w1_group_size),
                "w2_scale_group_size": int(w2_group_size),
                "w3_scale_group_size": int(w3_group_size),
                "w1_scale_group_count_used": int(w1_group_count),
                "w2_scale_group_count_used": int(w2_group_count),
                "w3_scale_group_count_used": int(w3_group_count),
                "output_hash": "sha256:" + hashlib.sha256(np.asarray(out, dtype=np.float32).tobytes()).hexdigest(),
                "finite_on_device": finite_mlp,
                "weight_tensor_values_public": False,
                "activation_payload_public": False,
            }

        def unpack_fp4_e2m1_numpy(packed):
            lut = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
            u8 = np.asarray(packed).astype(np.int8, copy=False).view(np.uint8)
            low = u8 & np.asarray(0x0F, dtype=np.uint8)
            high = (u8 >> np.asarray(4, dtype=np.uint8)) & np.asarray(0x0F, dtype=np.uint8)
            unpacked = np.stack([lut[low], lut[high]], axis=-1)
            return unpacked.reshape(*np.asarray(packed).shape[:-1], 2 * int(np.asarray(packed).shape[-1])).astype(np.float32, copy=False)

        def dequant_block_scaled_numpy(quantized, scales):
            q = np.asarray(quantized, dtype=np.float32)
            s = np.asarray(scales, dtype=np.float32)
            rows, cols = [int(item) for item in q.shape[-2:]]
            scale_rows, scale_cols = [int(item) for item in s.shape[-2:]]
            if rows % scale_rows or cols % scale_cols:
                raise RuntimeError("dequant_block_shape_mismatch")
            block_m = rows // scale_rows
            block_n = cols // scale_cols
            q_blocks = q.reshape(*q.shape[:-2], scale_rows, block_m, scale_cols, block_n)
            s_blocks = s.reshape(*q.shape[:-2], scale_rows, 1, scale_cols, 1)
            return (q_blocks * s_blocks).reshape(q.shape).astype(np.float32, copy=False), int(block_m), int(block_n)

        def load_array_for_forward(key, max_bytes=32 * 1024 * 1024):
            if key not in weight_map:
                raise RuntimeError("deepseek_v4_forward_key_missing")
            filename = str(weight_map[key])
            header_len, header = load_header(filename)
            meta = header.get(key)
            if not isinstance(meta, dict):
                raise RuntimeError("safetensors_key_missing")
            offsets = meta.get("data_offsets") or []
            if len(offsets) != 2:
                raise RuntimeError("safetensors_offsets_missing")
            start, end = int(offsets[0]), int(offsets[1])
            byte_length = end - start
            if byte_length <= 0 or byte_length > int(max_bytes):
                raise RuntimeError("real_weight_forward_tensor_out_of_budget")
            payload = read_range(filename, 8 + header_len + start, 8 + header_len + end - 1, byte_length)
            if len(payload) != byte_length:
                raise RuntimeError("real_weight_forward_tensor_truncated")
            dtype = str(meta.get("dtype") or "")
            shape = [int(item) for item in (meta.get("shape") or [])]
            if dtype == "I8":
                array = np.frombuffer(payload, dtype=np.int8).reshape(shape)
            else:
                array = decode_tensor(meta, payload)
            return array, {
                "key_digest": sha_payload(key),
                "file_digest": sha_payload(filename),
                "dtype": dtype,
                "shape": [int(item) for item in shape],
                "byte_length": int(byte_length),
                "raw_payload_sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "weight_tensor_values_public": False,
            }

        def dequant_fp4_expert(weight_key, scale_key):
            packed, packed_summary = load_array_for_forward(weight_key, max_bytes=8 * 1024 * 1024)
            scale, scale_summary = load_array_for_forward(scale_key, max_bytes=2 * 1024 * 1024)
            unpacked = unpack_fp4_e2m1_numpy(packed)
            dequantized, block_m, block_n = dequant_block_scaled_numpy(unpacked, scale)
            return jax.device_put(jnp.asarray(dequantized, dtype=jnp.float32), tpu_devices[0]), {
                "weight": packed_summary,
                "scale": scale_summary,
                "unpacked_shape": [int(item) for item in unpacked.shape],
                "dequantized_shape": [int(item) for item in dequantized.shape],
                "scale_block_shape": [int(block_m), int(block_n)],
                "packed_fp4_e2m1_x2": True,
            }

        def dequant_fp8_weight(weight_key, scale_key):
            weight, weight_summary = load_array_for_forward(weight_key, max_bytes=16 * 1024 * 1024)
            scale, scale_summary = load_array_for_forward(scale_key, max_bytes=2 * 1024 * 1024)
            dequantized, block_m, block_n = dequant_block_scaled_numpy(weight, scale)
            return jax.device_put(jnp.asarray(dequantized, dtype=jnp.float32), tpu_devices[0]), {
                "weight": weight_summary,
                "scale": scale_summary,
                "dequantized_shape": [int(item) for item in dequantized.shape],
                "scale_block_shape": [int(block_m), int(block_n)],
                "fp8_e4m3_ue8m0": True,
            }

        topk_expert_forward = {
            "ready": False,
            "weight_tensor_values_public": False,
            "activation_payload_public": False,
        }
        if router_norm is not None and routed_topk_indices and routed_topk_weights:
            try:
                routed_output = jnp.zeros((int(router_norm.shape[0]),), dtype=jnp.float32)
                expert_summaries = []
                loaded_forward_tensors = 0
                loaded_forward_bytes = 0
                for expert_id, expert_weight in zip(routed_topk_indices, routed_topk_weights):
                    prefix = "layers." + str(LAYER_START) + ".ffn.experts." + str(int(expert_id))
                    w1, w1_summary = dequant_fp4_expert(prefix + ".w1.weight", prefix + ".w1.scale")
                    w2, w2_summary = dequant_fp4_expert(prefix + ".w2.weight", prefix + ".w2.scale")
                    w3, w3_summary = dequant_fp4_expert(prefix + ".w3.weight", prefix + ".w3.scale")
                    gate = jnp.matmul(w1, router_norm)
                    up = jnp.matmul(w3, router_norm)
                    intermediate = jax.nn.silu(jnp.minimum(gate, 10.0)) * jnp.clip(up, -10.0, 10.0)
                    out = jnp.matmul(w2, intermediate).block_until_ready()
                    routed_output = routed_output + out * float(expert_weight)
                    tensor_summaries = [w1_summary, w2_summary, w3_summary]
                    loaded_forward_tensors += 6
                    loaded_forward_bytes += sum(int(item[name]["byte_length"]) for item in tensor_summaries for name in ("weight", "scale"))
                    expert_summaries.append({
                        "expert_id_digest": sha_payload(int(expert_id)),
                        "router_weight_hash": "sha256:" + hashlib.sha256(np.asarray([expert_weight], dtype=np.float32).tobytes()).hexdigest(),
                        "w1": w1_summary,
                        "w2": w2_summary,
                        "w3": w3_summary,
                        "intermediate_shape": [int(item) for item in intermediate.shape],
                        "output_shape": [int(item) for item in out.shape],
                        "output_hash": "sha256:" + hashlib.sha256(np.asarray(out, dtype=np.float32).tobytes()).hexdigest(),
                        "finite_output": bool(jnp.all(jnp.isfinite(out)).block_until_ready()),
                        "weight_tensor_values_public": False,
                    })
                    del w1, w2, w3, gate, up, intermediate, out

                shared_prefix = "layers." + str(LAYER_START) + ".ffn.shared_experts"
                sw1, sw1_summary = dequant_fp8_weight(shared_prefix + ".w1.weight", shared_prefix + ".w1.scale")
                sw2, sw2_summary = dequant_fp8_weight(shared_prefix + ".w2.weight", shared_prefix + ".w2.scale")
                sw3, sw3_summary = dequant_fp8_weight(shared_prefix + ".w3.weight", shared_prefix + ".w3.scale")
                shared_gate = jnp.matmul(sw1, router_norm)
                shared_up = jnp.matmul(sw3, router_norm)
                shared_intermediate = jax.nn.silu(shared_gate) * shared_up
                shared_output = jnp.matmul(sw2, shared_intermediate).block_until_ready()
                loaded_forward_tensors += 6
                loaded_forward_bytes += sum(int(item[name]["byte_length"]) for item in [sw1_summary, sw2_summary, sw3_summary] for name in ("weight", "scale"))
                final_output = (routed_output + shared_output).block_until_ready()
                finite_forward = bool(jnp.all(jnp.isfinite(final_output)).block_until_ready())
                topk_expert_forward = {
                    "ready": finite_forward,
                    "forward_kind": "deepseek_v4_jax_tpu_stage_selective_fp4_topk_routed_experts_plus_fp8_shared_expert",
                    "layer": int(LAYER_START),
                    "router_kind": "sqrtsoftplus_topk_normed_scaled",
                    "topk": int(len(routed_topk_indices)),
                    "topk_index_digest": sha_payload(routed_topk_indices),
                    "topk_weight_hash": "sha256:" + hashlib.sha256(np.asarray(routed_topk_weights, dtype=np.float32).tobytes()).hexdigest(),
                    "routed_expert_summaries": expert_summaries,
                    "shared_expert_summary": {
                        "w1": sw1_summary,
                        "w2": sw2_summary,
                        "w3": sw3_summary,
                        "intermediate_shape": [int(item) for item in shared_intermediate.shape],
                        "output_shape": [int(item) for item in shared_output.shape],
                        "output_hash": "sha256:" + hashlib.sha256(np.asarray(shared_output, dtype=np.float32).tobytes()).hexdigest(),
                        "finite_output": bool(jnp.all(jnp.isfinite(shared_output)).block_until_ready()),
                    },
                    "loaded_tensor_count": int(loaded_forward_tensors),
                    "total_loaded_tensor_bytes": int(loaded_forward_bytes),
                    "input_shape": [int(item) for item in router_norm.shape],
                    "routed_output_shape": [int(item) for item in routed_output.shape],
                    "shared_output_shape": [int(item) for item in shared_output.shape],
                    "final_output_shape": [int(item) for item in final_output.shape],
                    "final_output_hash": "sha256:" + hashlib.sha256(np.asarray(final_output, dtype=np.float32).tobytes()).hexdigest(),
                    "finite_output": finite_forward,
                    "weight_tensor_values_public": False,
                    "activation_payload_public": False,
                }
                del sw1, sw2, sw3, shared_gate, shared_up, shared_intermediate, shared_output, final_output
            except Exception as exc:
                topk_expert_forward = {
                    "ready": False,
                    "error_type": type(exc).__name__,
                    "error_digest": sha_payload(str(exc)),
                    "blockers": ["deepseek_v4_flash_real_fp4_topk_expert_forward_failed"],
                    "weight_tensor_values_public": False,
                    "activation_payload_public": False,
                }

        summary = {
            "loaded_tensor_count": len(loaded),
            "requested_tensor_count": len(candidate_keys),
            "loaded_key_digest": sha_payload(selected_keys),
            "dtype_counts": dtype_counts,
            "total_loaded_tensor_bytes": int(total_bytes),
            "device_put_count": int(device_put_count),
            "finite_tensor_count": int(finite_count),
            "tpu_device_kind": str(getattr(tpu_devices[0], "device_kind", "")),
            "tensor_summaries": loaded,
            "real_router_smoke": router_smoke,
            "real_router_smoke_ready": router_smoke.get("ready") is True,
            "real_fp8_block_dequant_smoke": fp8_block_smoke,
            "real_fp8_block_dequant_smoke_ready": fp8_block_smoke.get("ready") is True,
            "real_i8_expert_dequant_smoke": i8_expert_smoke,
            "real_i8_expert_dequant_smoke_ready": i8_expert_smoke.get("ready") is True,
            "real_i8_expert_mlp_slice_smoke": i8_expert_mlp_slice_smoke,
            "real_i8_expert_mlp_slice_smoke_ready": i8_expert_mlp_slice_smoke.get("ready") is True,
            "real_fp4_topk_expert_mlp_forward": topk_expert_forward,
            "real_fp4_topk_expert_mlp_forward_ready": topk_expert_forward.get("ready") is True,
            "real_routed_expert_topk_count": int(topk_expert_forward.get("topk") or 0),
            "real_routed_expert_loaded_tensor_count": int(topk_expert_forward.get("loaded_tensor_count") or 0),
            "real_routed_expert_total_loaded_tensor_bytes": int(topk_expert_forward.get("total_loaded_tensor_bytes") or 0),
            "tensor_payload_hashes_public": True,
            "weight_tensor_values_public": False,
            "real_weight_tensor_values_loaded": True,
            "stage_weight_values_loaded": False,
        }
        diagnosis_codes = ["deepseek_v4_flash_real_weight_tpu_tensor_load_ready"]
        if router_smoke.get("ready") is True:
            diagnosis_codes.append("deepseek_v4_flash_real_router_smoke_ready")
        if fp8_block_smoke.get("ready") is True:
            diagnosis_codes.append("deepseek_v4_flash_real_fp8_block_dequant_smoke_ready")
        if i8_expert_smoke.get("ready") is True:
            diagnosis_codes.append("deepseek_v4_flash_real_i8_expert_dequant_smoke_ready")
        if i8_expert_mlp_slice_smoke.get("ready") is True:
            diagnosis_codes.append("deepseek_v4_flash_real_i8_expert_mlp_slice_smoke_ready")
        if topk_expert_forward.get("ready") is True:
            diagnosis_codes.append("deepseek_v4_flash_real_fp4_topk_expert_mlp_forward_ready")
        return {
            "ok": bool(loaded and device_put_count == len(loaded) and finite_count == len(loaded)),
            "summary": summary,
            "blockers": [],
            "diagnosis_codes": diagnosis_codes,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_digest": sha_payload(str(exc)),
            "blockers": ["deepseek_v4_flash_real_weight_tpu_tensor_load_failed"],
            "diagnosis_codes": ["deepseek_v4_flash_real_weight_tpu_tensor_load_failed"],
        }
'''


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha_payload(value: Any) -> str:
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


def fetch_hf_json(model_id: str, filename: str, *, timeout_seconds: float = 120.0) -> dict[str, Any]:
    with urllib.request.urlopen(
        f"https://huggingface.co/{model_id}/resolve/main/{filename}",
        timeout=timeout_seconds,
    ) as response:
        loaded = json.load(response)
    return loaded if isinstance(loaded, dict) else {}


def build_stage_metadata(model_id: str, *, layer_start: int, layer_end: int, timeout_seconds: float = 120.0) -> dict[str, Any]:
    config = fetch_hf_json(model_id, "config.json", timeout_seconds=timeout_seconds)
    index = fetch_hf_json(model_id, "model.safetensors.index.json", timeout_seconds=timeout_seconds)
    weight_map = _dict(index.get("weight_map"))
    layer_prefixes = [f"layers.{layer}." for layer in range(layer_start, layer_end)]
    selected_keys = sorted(key for key in weight_map if any(str(key).startswith(prefix) for prefix in layer_prefixes))
    selected_files = sorted({str(weight_map[key]) for key in selected_keys if weight_map.get(key)})
    expected_families = {
        "mla_attention": ["attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b"],
        "moe_router": ["ffn.gate"],
        "shared_experts": ["ffn.shared_experts"],
        "routed_experts": ["ffn.experts."],
        "hybrid_compression": ["hc_attn", "hc_ffn"],
        "norms": ["attn_norm", "ffn_norm"],
    }
    family_hits = {
        name: any(fragment in key for key in selected_keys for fragment in fragments)
        for name, fragments in expected_families.items()
    }
    return {
        "schema": "deepseek_v4_flash_stage_metadata_v1",
        "model_id": model_id,
        "metadata_ready": bool(config and weight_map),
        "stage_key_mapping_ready": bool(config and weight_map and selected_keys and all(family_hits.values())),
        "model_config": {
            "architectures": list(config.get("architectures") or [])[:8],
            "model_type": str(config.get("model_type") or ""),
            "num_hidden_layers": _int(config.get("num_hidden_layers")),
            "hidden_size": _int(config.get("hidden_size")),
            "num_attention_heads": _int(config.get("num_attention_heads")),
            "n_routed_experts": _int(config.get("n_routed_experts") or config.get("num_experts")),
            "num_experts_per_tok": _int(config.get("num_experts_per_tok")),
            "n_shared_experts": _int(config.get("n_shared_experts")),
            "q_lora_rank": _int(config.get("q_lora_rank")),
            "qk_rope_head_dim": _int(config.get("qk_rope_head_dim")),
            "moe_intermediate_size": _int(config.get("moe_intermediate_size")),
            "torch_dtype": str(config.get("torch_dtype") or ""),
            "quantization_config_present": isinstance(config.get("quantization_config"), dict),
            "config_payload_public": False,
        },
        "weight_index": {
            "weight_key_count": len(weight_map),
            "weight_file_count": len(set(weight_map.values())),
            "metadata_total_size_bytes": _int(_dict(index.get("metadata")).get("total_size")),
            "weight_map_payload_public": False,
        },
        "stage_mapping": {
            "layer_range": [int(layer_start), int(layer_end)],
            "selected_key_count": len(selected_keys),
            "selected_file_count": len(selected_files),
            "selected_key_digest": sha_payload(selected_keys),
            "selected_file_digest": sha_payload(selected_files),
            "family_hits": family_hits,
            "stage_weight_values_loaded": False,
            "stage_weight_values_public": False,
        },
        "blockers": [] if bool(config and weight_map and selected_keys and all(family_hits.values())) else ["deepseek_v4_flash_stage_key_mapping_incomplete"],
        "public_artifact_safe": True,
    }


def metadata_from_cell_or_local(summary: dict[str, Any], local: dict[str, Any]) -> dict[str, Any]:
    if summary.get("metadata_ready") and summary.get("stage_key_mapping_ready"):
        return {
            "source": "kaggle_web_tpu_cell",
            "metadata_ready": True,
            "stage_key_mapping_ready": True,
            "model_config": _dict(summary.get("model_config")),
            "weight_index": _dict(summary.get("weight_index")),
            "stage_mapping": _dict(summary.get("stage_mapping")),
            "blockers": [],
            "public_artifact_safe": True,
        }
    return {
        "source": "local_hf_api",
        "metadata_ready": local.get("metadata_ready") is True,
        "stage_key_mapping_ready": local.get("stage_key_mapping_ready") is True,
        "model_config": _dict(local.get("model_config")),
        "weight_index": _dict(local.get("weight_index")),
        "stage_mapping": _dict(local.get("stage_mapping")),
        "blockers": [str(item) for item in _list(local.get("blockers"))],
        "public_artifact_safe": local.get("public_artifact_safe") is True,
    }


def render_deepseek_stage_metadata_cell(args: argparse.Namespace) -> str:
    return f'''
import hashlib
import json
import time
import urllib.request

SCHEMA = {CELL_SCHEMA!r}
MODEL_ID = {args.model_id!r}
LAYER_START = {int(args.layer_start)!r}
LAYER_END = {int(args.layer_end)!r}


def sha_payload(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def load_hf_json(name):
    with urllib.request.urlopen("https://huggingface.co/" + MODEL_ID + "/resolve/main/" + name, timeout=120) as response:
        loaded = json.load(response)
    return loaded if isinstance(loaded, dict) else {{}}


started = time.monotonic()
report = {{
    "schema": SCHEMA,
    "cell_kind": "deepseek_stage_metadata",
    "ok": False,
    "model_id": MODEL_ID,
    "layer_range": [LAYER_START, LAYER_END],
    "metadata_ready": False,
    "stage_key_mapping_ready": False,
    "jax_imported": False,
    "tpu_runtime_ready": False,
    "tpu_device_count": 0,
    "deepseek_v4_jax_tpu_stage_forward_ready": False,
    "deepseek_v4_real_weight_tpu_tensor_load_ready": False,
    "real_weight_tensor_values_loaded": False,
    "real_weight_tensor_values_public": False,
    "activation_payload_public": False,
    "kv_cache_public": False,
    "raw_prompt_public": False,
    "generated_token_ids_public": False,
    "blockers": [],
    "diagnosis_codes": [],
    "jupyter_proxy_token_public": False,
    "public_artifact_safe": True,
}}
try:
    import jax
    import jax.numpy as jnp

    report["jax_imported"] = True
    devices = list(jax.devices())
    tpu_devices = [device for device in devices if str(getattr(device, "platform", "")).lower() == "tpu"]
    report["tpu_device_count"] = len(tpu_devices)
    report["tpu_device_kind"] = str(getattr(tpu_devices[0], "device_kind", "")) if tpu_devices else ""
    report["tpu_runtime_ready"] = bool(tpu_devices)
    if not tpu_devices:
        report["blockers"].append("jax_tpu_device_missing")
        report["diagnosis_codes"].append("deepseek_v4_flash_web_tpu_device_missing")
    else:
        x = jax.device_put(jnp.arange(8, dtype=jnp.float32), tpu_devices[0])
        y = (x * x).block_until_ready()
        report["tpu_smoke_hash"] = sha_payload({{"shape": [8], "sum": round(float(jnp.sum(y)), 7)}})
except Exception as exc:
    report["error_type"] = type(exc).__name__
    report["error_digest"] = sha_payload(str(exc))
    report["blockers"].append("jax_tpu_smoke_failed")
    report["diagnosis_codes"].append("deepseek_v4_flash_jax_tpu_smoke_failed")

try:
    config = load_hf_json("config.json")
    index = load_hf_json("model.safetensors.index.json")
    weight_map = index.get("weight_map") if isinstance(index.get("weight_map"), dict) else {{}}
    layer_prefixes = ["layers." + str(i) + "." for i in range(LAYER_START, LAYER_END)]
    selected_keys = sorted(key for key in weight_map if any(str(key).startswith(prefix) for prefix in layer_prefixes))
    selected_files = sorted({{str(weight_map[key]) for key in selected_keys if weight_map.get(key)}})
    expected_families = {{
        "mla_attention": ["attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b"],
        "moe_router": ["ffn.gate"],
        "shared_experts": ["ffn.shared_experts"],
        "routed_experts": ["ffn.experts."],
        "hybrid_compression": ["hc_attn", "hc_ffn"],
        "norms": ["attn_norm", "ffn_norm"],
    }}
    family_hits = {{
        name: any(fragment in key for key in selected_keys for fragment in fragments)
        for name, fragments in expected_families.items()
    }}
    metadata_ready = bool(config and weight_map)
    stage_key_mapping_ready = bool(metadata_ready and selected_keys and all(family_hits.values()))
    report.update({{
        "metadata_ready": metadata_ready,
        "stage_key_mapping_ready": stage_key_mapping_ready,
        "model_config": {{
            "architectures": list(config.get("architectures") or [])[:8],
            "model_type": str(config.get("model_type") or ""),
            "num_hidden_layers": int(config.get("num_hidden_layers") or 0),
            "hidden_size": int(config.get("hidden_size") or 0),
            "num_attention_heads": int(config.get("num_attention_heads") or 0),
            "n_routed_experts": int(config.get("n_routed_experts") or config.get("num_experts") or 0),
            "num_experts_per_tok": int(config.get("num_experts_per_tok") or 0),
            "n_shared_experts": int(config.get("n_shared_experts") or 0),
            "q_lora_rank": int(config.get("q_lora_rank") or 0),
            "qk_rope_head_dim": int(config.get("qk_rope_head_dim") or 0),
            "moe_intermediate_size": int(config.get("moe_intermediate_size") or 0),
            "torch_dtype": str(config.get("torch_dtype") or ""),
            "quantization_config_present": isinstance(config.get("quantization_config"), dict),
            "config_payload_public": False,
        }},
        "weight_index": {{
            "weight_key_count": len(weight_map),
            "weight_file_count": len(set(weight_map.values())),
            "metadata_total_size_bytes": int((index.get("metadata") or {{}}).get("total_size") or 0),
            "weight_map_payload_public": False,
        }},
        "stage_mapping": {{
            "layer_range": [LAYER_START, LAYER_END],
            "selected_key_count": len(selected_keys),
            "selected_file_count": len(selected_files),
            "selected_key_digest": sha_payload(selected_keys),
            "selected_file_digest": sha_payload(selected_files),
            "family_hits": family_hits,
            "stage_weight_values_loaded": False,
            "stage_weight_values_public": False,
        }},
    }})
    if not metadata_ready:
        report["blockers"].append("deepseek_v4_flash_metadata_unavailable")
        report["diagnosis_codes"].append("deepseek_v4_flash_metadata_unavailable")
    if metadata_ready and not stage_key_mapping_ready:
        report["blockers"].append("deepseek_v4_flash_stage_key_mapping_incomplete")
        report["diagnosis_codes"].append("deepseek_v4_flash_stage_key_mapping_incomplete")
    if stage_key_mapping_ready:
        report["diagnosis_codes"].append("deepseek_v4_flash_stage_key_mapping_ready")
except Exception as exc:
    report["error_type"] = type(exc).__name__
    report["error_digest"] = sha_payload(str(exc))
    report["blockers"].append("deepseek_v4_flash_metadata_fetch_failed")
    report["diagnosis_codes"].append("deepseek_v4_flash_metadata_fetch_failed")

{DEEPSEEK_V4_TPU_FIXTURE_FORWARD_SOURCE}

{DEEPSEEK_V4_REAL_TENSOR_LOAD_SOURCE}

if report.get("tpu_runtime_ready") and report.get("stage_key_mapping_ready"):
    real_load_result = run_deepseek_v4_real_weight_tpu_tensor_load(report)
    report["deepseek_v4_real_weight_tpu_tensor_load_ready"] = real_load_result.get("ok") is True
    report["deepseek_v4_real_weight_tpu_tensor_load"] = real_load_result.get("summary") or {{}}
    report["real_weight_tensor_values_loaded"] = real_load_result.get("ok") is True
    report["blockers"].extend(str(item) for item in real_load_result.get("blockers") or [] if item)
    report["diagnosis_codes"].extend(str(item) for item in real_load_result.get("diagnosis_codes") or [] if item)
    fixture_result = run_deepseek_v4_tpu_fixture_stage_forward(report)
    report["deepseek_v4_jax_tpu_fixture_stage_forward_ready"] = fixture_result.get("ok") is True
    report["deepseek_v4_jax_tpu_fixture_stage_forward"] = fixture_result.get("summary") or {{}}
    report["blockers"].extend(str(item) for item in fixture_result.get("blockers") or [] if item)
    report["diagnosis_codes"].extend(str(item) for item in fixture_result.get("diagnosis_codes") or [] if item)
    if report.get("deepseek_v4_jax_tpu_fixture_stage_forward_ready"):
        report["blockers"].append("deepseek_v4_flash_real_weight_tpu_stage_loader_not_implemented")
        report["diagnosis_codes"].append("deepseek_v4_flash_web_tpu_fixture_stage_forward_ready_real_weight_gap")
    else:
        report["blockers"].append("deepseek_v4_flash_mla_moe_jax_tpu_stage_forward_not_implemented")
    report["blockers"].append("deepseek_v4_flash_quantized_fp8_nvfp4_tpu_loader_not_implemented")
    report["diagnosis_codes"].append("deepseek_v4_flash_web_tpu_metadata_ready_adapter_gap")

report["ok"] = bool(report.get("metadata_ready") and report.get("stage_key_mapping_ready") and report.get("tpu_runtime_ready"))
report["elapsed_seconds"] = round(time.monotonic() - started, 3)
print(json.dumps({{"schema": SCHEMA, "report": report}}, sort_keys=True))
'''


def run_web_cell(args: argparse.Namespace) -> dict[str, Any]:
    try:
        report = web_tpu_bridge.execute_web_tpu_code_via_iframe(
            args,
            render_deepseek_stage_metadata_cell(args),
        )
    except Exception as exc:
        blocker, diagnosis = web_tpu_bridge.classify_web_tpu_exception(exc)
        report = {
            "schema": CELL_SCHEMA,
            "cell_kind": "deepseek_stage_metadata",
            "ok": False,
            "blockers": [blocker],
            "diagnosis_codes": [diagnosis],
            "error_type": type(exc).__name__,
            "error_digest": sha_payload(str(exc)),
            "jupyter_proxy_token_public": False,
            "public_artifact_safe": True,
        }
    report.setdefault("cell_kind", "deepseek_stage_metadata")
    report["jupyter_proxy_token_public"] = False
    return report


def summarize_cell(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "deepseek_v4_flash_kaggle_web_tpu_stage_adapter_cell_summary_v1",
        "cell_kind": str(report.get("cell_kind") or "deepseek_stage_metadata"),
        "ok": report.get("ok") is True,
        "model_id": str(report.get("model_id") or ""),
        "metadata_ready": report.get("metadata_ready") is True,
        "stage_key_mapping_ready": report.get("stage_key_mapping_ready") is True,
        "jax_imported": report.get("jax_imported") is True,
        "tpu_runtime_ready": report.get("tpu_runtime_ready") is True,
        "tpu_device_count": _int(report.get("tpu_device_count")),
        "tpu_device_kind": str(report.get("tpu_device_kind") or ""),
        "deepseek_v4_real_weight_tpu_tensor_load_ready": report.get("deepseek_v4_real_weight_tpu_tensor_load_ready") is True,
        "deepseek_v4_real_weight_tpu_tensor_load": _dict(report.get("deepseek_v4_real_weight_tpu_tensor_load")),
        "deepseek_v4_jax_tpu_fixture_stage_forward_ready": report.get("deepseek_v4_jax_tpu_fixture_stage_forward_ready") is True,
        "deepseek_v4_jax_tpu_fixture_stage_forward": _dict(report.get("deepseek_v4_jax_tpu_fixture_stage_forward")),
        "deepseek_v4_jax_tpu_stage_forward_ready": report.get("deepseek_v4_jax_tpu_stage_forward_ready") is True,
        "model_config": _dict(report.get("model_config")),
        "weight_index": _dict(report.get("weight_index")),
        "stage_mapping": _dict(report.get("stage_mapping")),
        "blockers": [str(item) for item in _list(report.get("blockers")) if item],
        "diagnosis_codes": [str(item) for item in _list(report.get("diagnosis_codes")) if item],
        "web_tpu_jupyter_access_mode": str(report.get("web_tpu_jupyter_access_mode") or ""),
        "web_tpu_jupyter_steps": web_tpu_bridge.public_jupyter_steps(report.get("web_tpu_jupyter_steps")),
        "web_tpu_executor_attempts": web_tpu_bridge.public_executor_attempts(report.get("web_tpu_executor_attempts")),
        "jupyter_proxy_token_public": False,
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def failure_stage_from_summary(summary: dict[str, Any]) -> str:
    blockers = set(summary.get("blockers") or [])
    if not summary.get("tpu_runtime_ready"):
        if "jax_tpu_device_missing" in blockers:
            return "kaggle_web_tpu_device_missing"
        return "kaggle_web_tpu_runtime_not_ready"
    if not summary.get("metadata_ready"):
        return "deepseek_v4_flash_metadata_unavailable"
    if not summary.get("stage_key_mapping_ready"):
        return "deepseek_v4_flash_stage_key_mapping_incomplete"
    if not summary.get("deepseek_v4_jax_tpu_stage_forward_ready"):
        if summary.get("deepseek_v4_jax_tpu_fixture_stage_forward_ready"):
            return "deepseek_v4_flash_real_weight_tpu_stage_loader_not_implemented"
        return "deepseek_v4_flash_mla_moe_jax_tpu_stage_forward_not_implemented"
    return ""


def build_report(
    args: argparse.Namespace,
    *,
    cell_report: dict[str, Any],
    output_dir: Path,
    local_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = summarize_cell(cell_report)
    if local_metadata is None:
        try:
            local_metadata = build_stage_metadata(
                args.model_id,
                layer_start=int(args.layer_start),
                layer_end=int(args.layer_end),
                timeout_seconds=float(args.hf_timeout_seconds),
            )
        except Exception as exc:
            local_metadata = {
                "schema": "deepseek_v4_flash_stage_metadata_v1",
                "metadata_ready": False,
                "stage_key_mapping_ready": False,
                "error_type": type(exc).__name__,
                "error_digest": sha_payload(str(exc)),
                "blockers": ["deepseek_v4_flash_local_metadata_fetch_failed"],
                "public_artifact_safe": True,
            }
    metadata = metadata_from_cell_or_local(summary, local_metadata)
    metadata_ready = metadata.get("metadata_ready") is True and metadata.get("stage_key_mapping_ready") is True
    tpu_ready = summary.get("tpu_runtime_ready") is True and _int(summary.get("tpu_device_count")) >= 1
    adapter_ready = summary.get("deepseek_v4_jax_tpu_stage_forward_ready") is True
    fixture_forward_ready = summary.get("deepseek_v4_jax_tpu_fixture_stage_forward_ready") is True
    real_tensor_load_ready = summary.get("deepseek_v4_real_weight_tpu_tensor_load_ready") is True
    real_load = _dict(summary.get("deepseek_v4_real_weight_tpu_tensor_load"))
    real_fp4_topk_forward_ready = real_load.get("real_fp4_topk_expert_mlp_forward_ready") is True
    blockers = set(str(item) for item in _list(summary.get("blockers")) if item)
    blockers.update(str(item) for item in _list(metadata.get("blockers")) if item)
    if not adapter_ready and fixture_forward_ready:
        blockers.add("deepseek_v4_flash_real_weight_tpu_stage_loader_not_implemented")
    elif not adapter_ready:
        blockers.add("deepseek_v4_flash_mla_moe_jax_tpu_stage_forward_not_implemented")
    if not tpu_ready:
        blockers.add("kaggle_web_tpu_runtime_not_ready")
    if not metadata_ready:
        blockers.add("deepseek_v4_flash_metadata_or_stage_mapping_not_ready")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(metadata_ready and tpu_ready and adapter_ready),
        "deepseek_v4_flash_kaggle_web_tpu_stage_adapter_ready": bool(metadata_ready and tpu_ready and adapter_ready),
        "metadata_ready": metadata_ready,
        "kaggle_web_tpu_runtime_ready": tpu_ready,
        "deepseek_v4_jax_tpu_fixture_stage_forward_ready": fixture_forward_ready,
        "deepseek_v4_real_weight_tpu_tensor_load_ready": real_tensor_load_ready,
        "deepseek_v4_real_fp4_topk_expert_mlp_forward_ready": real_fp4_topk_forward_ready,
        "deepseek_v4_jax_tpu_stage_forward_ready": adapter_ready,
        "model": {
            "model_id": args.model_id,
            "architecture_class": "moe",
            "expected_model_type": "deepseek_v4",
            "total_params_b": 284.0,
            "active_params_b": 13.0,
            "source": "huggingface",
            "quantized_or_mixed_precision": True,
        },
        "stage_plan": {
            "layer_range": [int(args.layer_start), int(args.layer_end)],
            "backend": "kaggle_web_tpu_jax",
            "stage_selective": True,
            "real_deepseek_metadata_used": metadata_ready,
            "fixture_stage_forward_executed": fixture_forward_ready,
            "real_weight_sample_tensors_loaded": real_tensor_load_ready,
            "real_fp4_topk_expert_mlp_forward_ready": real_fp4_topk_forward_ready,
            "real_routed_expert_topk_count": _int(real_load.get("real_routed_expert_topk_count")),
            "real_routed_expert_loaded_tensor_count": _int(real_load.get("real_routed_expert_loaded_tensor_count")),
            "real_routed_expert_total_loaded_tensor_bytes": _int(real_load.get("real_routed_expert_total_loaded_tensor_bytes")),
            "real_weight_sample_tensors_public": False,
            "real_weight_values_loaded": False,
            "real_weight_values_public": False,
        },
        "deepseek_metadata": metadata,
        "web_tpu_cell": summary,
        "failure_stage": "" if metadata_ready and tpu_ready and adapter_ready else failure_stage_from_summary(summary),
        "blockers": sorted(blockers),
        "diagnosis_codes": sorted(set(_list(summary.get("diagnosis_codes")) + [
            "deepseek_v4_flash_kaggle_web_tpu_metadata_ready" if metadata_ready else "deepseek_v4_flash_kaggle_web_tpu_metadata_not_ready",
            "deepseek_v4_flash_kaggle_web_tpu_stage_adapter_ready" if adapter_ready else "deepseek_v4_flash_kaggle_web_tpu_stage_adapter_not_ready",
        ])),
        "cleanup_status": {
            "temporary_kaggle_kernels_created": False,
            "temporary_kaggle_kernels_deleted": True,
            "temporary_private_packages_removed": True,
            "live_resources_left_running": False,
            "web_runtime_execution_count": int(bool(summary)),
            "cookie_file_public": False,
            "storage_state_file_public": False,
        },
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
            "weight_tensor_values_public": False,
            "credentials_public": False,
            "cookies_public": False,
            "jupyter_proxy_token_public": False,
            "private_runtime_state_public": False,
        },
        "public_artifact_safe": True,
        "limitations": [
            "This preflight can prove Kaggle Web TPU execution plus real DeepSeek-V4-Flash metadata/stage-key mapping.",
            "It is not a successful DeepSeek-V4-Flash same-request decode until deepseek_v4_jax_tpu_stage_forward_ready is true and GPU/TPU/CPU same-request evidence exists.",
            "Public artifacts contain only metadata, counts, hashes, and blockers; no prompts, generated text, token ids, tensor values, hidden states, logits, or KV-cache tensors are public.",
        ],
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["deepseek_v4_flash_kaggle_web_tpu_stage_adapter_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"].append("public_redaction_scan_failed")
        report["diagnosis_codes"].append("public_redaction_scan_failed")
        report["redaction_errors"] = leaks
    summary_path = output_dir / "deepseek_v4_flash_kaggle_web_tpu_stage_adapter_probe.json"
    write_json(summary_path, report)
    report["artifacts"] = {
        "summary_json": artifact_entry(summary_path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(report.get("ok"))),
    }
    write_json(summary_path, report)
    return report


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    local_metadata = build_stage_metadata(
        args.model_id,
        layer_start=int(args.layer_start),
        layer_end=int(args.layer_end),
        timeout_seconds=float(args.hf_timeout_seconds),
    )
    cell_report = (
        {
            "schema": CELL_SCHEMA,
            "cell_kind": "deepseek_stage_metadata",
            "ok": False,
            "blockers": ["web_tpu_execution_skipped"],
            "diagnosis_codes": ["deepseek_v4_flash_web_tpu_execution_skipped"],
            "jupyter_proxy_token_public": False,
            "public_artifact_safe": True,
        }
        if args.skip_web_tpu_execute
        else run_web_cell(args)
    )
    return build_report(args, cell_report=cell_report, output_dir=output_dir, local_metadata=local_metadata)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe DeepSeek-V4-Flash Kaggle Web TPU adapter readiness.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--layer-start", type=int, default=DEFAULT_LAYER_START)
    parser.add_argument("--layer-end", type=int, default=DEFAULT_LAYER_END)
    parser.add_argument("--hf-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--kaggle-notebook-url", default=DEFAULT_NOTEBOOK_URL)
    parser.add_argument("--kaggle-web-storage-state", default="/root/kaggle-web-storage-state.json")
    parser.add_argument("--chrome-executable", default="/usr/bin/google-chrome")
    parser.add_argument("--web-tpu-execute-timeout-seconds", type=float, default=360.0)
    parser.add_argument("--web-tpu-force-new-session", action="store_true")
    parser.add_argument("--skip-web-tpu-execute", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.layer_start < 0:
        raise SystemExit("--layer-start must be non-negative")
    if args.layer_end <= args.layer_start:
        raise SystemExit("--layer-end must be greater than --layer-start")
    if args.layer_end - args.layer_start > 8:
        raise SystemExit("--layer range must cover at most 8 layers")
    if args.web_tpu_execute_timeout_seconds < 30 or args.web_tpu_execute_timeout_seconds > 900:
        raise SystemExit("--web-tpu-execute-timeout-seconds must be between 30 and 900")
    if args.hf_timeout_seconds < 1 or args.hf_timeout_seconds > 600:
        raise SystemExit("--hf-timeout-seconds must be between 1 and 600")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_probe(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            f"{SCHEMA}: ok={bool(report.get('ok'))} "
            f"metadata={bool(report.get('metadata_ready'))} "
            f"web_tpu={bool(report.get('kaggle_web_tpu_runtime_ready'))} "
            f"adapter={bool(report.get('deepseek_v4_jax_tpu_stage_forward_ready'))} "
            f"failure={report.get('failure_stage') or 'none'}"
        )
    return 0 if report.get("public_artifact_safe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
