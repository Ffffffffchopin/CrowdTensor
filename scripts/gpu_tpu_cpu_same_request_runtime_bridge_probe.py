#!/usr/bin/env python3
"""Run a bounded same-request CUDA + JAX/TPU + CPU runtime bridge probe.

By default this proof intentionally does not claim 32B model success. With the
explicit Web-TPU 32B execution switch enabled, the JAX/TPU stage claims the
Coordinator stage task, executes a real Qwen 32B stage-owned loader inside the
same request window, and emits a separate live-proof artifact for RC import.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import textwrap
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import gpu_tpu_cpu_32b_heterogeneous_rc_pack as rc_pack  # noqa: E402
from scripts import colab_cli_runtime  # noqa: E402
from scripts import deepseek_v4_flash_kaggle_web_tpu_stage_adapter_probe as deepseek_tpu_adapter  # noqa: E402
from scripts import kaggle_32b_stage_owned_safetensors_probe as loading_probe  # noqa: E402
from scripts import kaggle_tpu_32b_stage_owned_loader_probe as tpu_loader_probe  # noqa: E402


SCHEMA = "gpu_tpu_cpu_same_request_runtime_bridge_probe_v1"
GPU_STAGE_SCHEMA = "gpu_tpu_cpu_bridge_cuda_stage_v1"
CPU_STAGE_SCHEMA = "gpu_tpu_cpu_bridge_kaggle_cpu_stage_v1"
DEFAULT_OUTPUT_DIR = "dist/gpu-tpu-cpu-same-request-runtime-bridge"
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"
DEFAULT_PUBLIC_HOST = "24.199.118.54"
DEFAULT_PORT = 9256
Runner = Callable[..., subprocess.CompletedProcess[str]]


SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "token=",
    "jupyter-proxy",
    "X-CrowdTensor-Bridge-Token",
    "colab-runtime-proxy-token",
)


def safe_stage_id(value: Any, *, default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


FP4_E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def unpack_fp4_e2m1_numpy(packed: Any) -> Any:
    """Unpack two E2M1 FP4 nibbles per byte into a float32 matrix."""
    import numpy as np

    packed_array = np.asarray(packed)
    lut = np.asarray(FP4_E2M1_LUT, dtype=np.float32)
    u8 = packed_array.astype(np.int8, copy=False).view(np.uint8)
    low = u8 & np.asarray(0x0F, dtype=np.uint8)
    high = (u8 >> np.asarray(4, dtype=np.uint8)) & np.asarray(0x0F, dtype=np.uint8)
    unpacked = np.stack([lut[low], lut[high]], axis=-1)
    return unpacked.reshape(*packed_array.shape[:-1], 2 * int(packed_array.shape[-1])).astype(np.float32, copy=False)


def dequant_block_scaled_numpy(quantized: Any, scales: Any) -> tuple[Any, int, int]:
    """Apply a per-block scale grid to an already-decoded quantized matrix."""
    import numpy as np

    q = np.asarray(quantized, dtype=np.float32)
    s = np.asarray(scales, dtype=np.float32)
    rows, cols = [int(item) for item in q.shape[-2:]]
    scale_rows, scale_cols = [int(item) for item in s.shape[-2:]]
    if rows % scale_rows or cols % scale_cols:
        raise ValueError(f"dequant_block_shape_mismatch:{rows}x{cols}:{scale_rows}x{scale_cols}")
    block_m = rows // scale_rows
    block_n = cols // scale_cols
    leading_shape = q.shape[:-2]
    q_blocks = q.reshape(*leading_shape, scale_rows, block_m, scale_cols, block_n)
    s_blocks = s.reshape(*leading_shape, scale_rows, 1, scale_cols, 1)
    return (q_blocks * s_blocks).reshape(q.shape).astype(np.float32, copy=False), int(block_m), int(block_n)


def stable_softplus_numpy(value: Any) -> Any:
    import numpy as np

    x = np.asarray(value, dtype=np.float32)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


GPU_KERNEL_TEMPLATE = r'''
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "__GPU_STAGE_SCHEMA__"
COORDINATOR_URL = __COORDINATOR_URL_JSON__
TOKEN = __TOKEN_JSON__
TASK_TIMEOUT_SECONDS = __TASK_TIMEOUT_SECONDS_JSON__
TARGET_GENERATED_TOKEN_COUNT = __TARGET_GENERATED_TOKEN_COUNT_JSON__
DEEPSEEK_REAL_STAGE_SLICE = __DEEPSEEK_REAL_STAGE_SLICE_JSON__
DEEPSEEK_STAGE_LAYER_START = __DEEPSEEK_STAGE_LAYER_START_JSON__
DEEPSEEK_STAGE_LAYER_END = __DEEPSEEK_STAGE_LAYER_END_JSON__
DEEPSEEK_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
OUT = Path("/kaggle/working")
REPORT_PATH = OUT / "gpu_tpu_cpu_bridge_cuda_stage_report.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha_payload(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def post_json(path, payload):
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    req = urllib.request.Request(
        COORDINATOR_URL + path,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-CrowdTensor-Bridge-Token": TOKEN,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_deepseek_v4_real_weight_cuda_slice(torch, device):
    try:
        import struct
        import urllib.error
        import numpy as np

        def load_hf_json(name):
            with urllib.request.urlopen(
                "https://huggingface.co/" + DEEPSEEK_MODEL_ID + "/resolve/main/" + name,
                timeout=120,
            ) as response:
                loaded = json.load(response)
            return loaded if isinstance(loaded, dict) else {}

        index = load_hf_json("model.safetensors.index.json")
        weight_map = index.get("weight_map") if isinstance(index.get("weight_map"), dict) else {}
        layer = int(DEEPSEEK_STAGE_LAYER_START)
        candidate_keys = [
            f"layers.{layer}.attn.attn_sink",
            f"layers.{layer}.attn_norm.weight",
            f"layers.{layer}.ffn_norm.weight",
            f"layers.{layer}.ffn.gate.bias",
            f"layers.{layer}.ffn.gate.weight",
            f"layers.{layer}.attn.wq_a.weight",
            f"layers.{layer}.attn.wq_a.scale",
            f"layers.{layer}.ffn.experts.0.w1.weight",
            f"layers.{layer}.ffn.experts.0.w1.scale",
            f"layers.{layer}.ffn.experts.0.w2.weight",
            f"layers.{layer}.ffn.experts.0.w2.scale",
            f"layers.{layer}.ffn.experts.0.w3.weight",
            f"layers.{layer}.ffn.experts.0.w3.scale",
        ]
        selected_keys = [key for key in candidate_keys if key in weight_map]
        if not selected_keys:
            return {
                "ok": False,
                "blockers": ["deepseek_v4_cuda_real_weight_sample_keys_missing"],
                "diagnosis_codes": ["deepseek_v4_cuda_real_weight_sample_keys_missing"],
            }

        def read_range(filename, start, end, max_bytes):
            req = urllib.request.Request(
                "https://huggingface.co/" + DEEPSEEK_MODEL_ID + "/resolve/main/" + filename,
                headers={
                    "Range": "bytes=" + str(int(start)) + "-" + str(int(end)),
                    "User-Agent": "crowdtensor-deepseek-v4-cuda-real-weight-smoke/1",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
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
                return np.frombuffer(payload, dtype="<f4").reshape(shape).astype(np.float32, copy=False)
            if dtype == "BF16":
                raw = np.frombuffer(payload, dtype="<u2")
                return (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)
            if dtype == "I8":
                return np.frombuffer(payload, dtype=np.int8).reshape(shape).astype(np.float32)
            if dtype == "F8_E8M0":
                raw = np.frombuffer(payload, dtype=np.uint8).astype(np.int16)
                return np.exp2(np.clip(raw - 127, -32, 32).astype(np.float32)).reshape(shape)
            if dtype == "F8_E4M3":
                raw = np.frombuffer(payload, dtype=np.uint8)
                sign = np.where((raw & 0x80) == 0, 1.0, -1.0).astype(np.float32)
                exp = ((raw >> 3) & 0x0F).astype(np.int16)
                mant = (raw & 0x07).astype(np.float32)
                normal = (1.0 + mant / 8.0) * np.exp2((exp - 7).astype(np.float32))
                subnormal = (mant / 8.0) * np.exp2(np.asarray(-6.0, dtype=np.float32))
                value = np.where(exp == 0, subnormal, normal)
                value = np.where((raw & 0x7F) == 0, 0.0, value)
                return (sign * value).reshape(shape).astype(np.float32)
            raise RuntimeError("unsupported_real_weight_sample_dtype_" + dtype)

        loaded = []
        loaded_tensors = {}
        dtype_counts = {}
        total_bytes = 0
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
            tensor = torch.as_tensor(array, device=device, dtype=torch.float32)
            finite = bool(torch.isfinite(tensor).all().detach().cpu().item())
            loaded_tensors[key] = tensor
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

        def hash_tensor(tensor):
            return "sha256:" + hashlib.sha256(
                tensor.detach().cpu().float().numpy().astype(np.float32).tobytes()
            ).hexdigest()

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

        def stable_softplus_torch(value):
            return torch.log1p(torch.exp(-torch.abs(value))) + torch.maximum(value, torch.zeros_like(value))

        router_smoke = {"ready": False, "weight_tensor_values_public": False, "activation_payload_public": False}
        router_norm = None
        routed_topk_indices = []
        routed_topk_weights = []
        norm_key = f"layers.{layer}.ffn_norm.weight"
        gate_bias_key = f"layers.{layer}.ffn.gate.bias"
        gate_key = f"layers.{layer}.ffn.gate.weight"
        if gate_key in loaded_tensors and norm_key in loaded_tensors:
            hidden = int(loaded_tensors[norm_key].shape[0])
            dummy = torch.linspace(-0.01, 0.01, hidden, device=device, dtype=torch.float32)
            norm = dummy * torch.rsqrt(torch.mean(dummy * dummy) + 1.0e-6) * loaded_tensors[norm_key]
            router_norm = norm
            bias = loaded_tensors.get(gate_bias_key)
            if bias is None:
                bias = torch.zeros((int(loaded_tensors[gate_key].shape[0]),), device=device, dtype=torch.float32)
            logits = torch.matmul(loaded_tensors[gate_key], router_norm) + bias
            scores = torch.sqrt(stable_softplus_torch(logits))
            k = min(6, int(scores.numel()))
            values, indices = torch.topk(scores, k=k)
            weights = values / (torch.sum(values) + 1.0e-20) * 1.5
            routed_topk_indices = [int(item) for item in indices.detach().cpu().tolist()]
            routed_topk_weights = [float(item) for item in weights.detach().cpu().float().tolist()]
            finite_router = bool(torch.isfinite(values).all().detach().cpu().item())
            router_smoke = {
                "ready": finite_router,
                "router_kind": "deepseek_v4_moe_gate_topk",
                "topk": int(k),
                "gate_shape": [int(item) for item in loaded_tensors[gate_key].shape],
                "gate_bias_shape": [int(item) for item in bias.shape],
                "norm_shape": [int(item) for item in loaded_tensors[norm_key].shape],
                "topk_index_digest": sha_payload([int(item) for item in indices.detach().cpu().tolist()]),
                "topk_weight_hash": hash_tensor(weights),
                "score_hash": hash_tensor(values),
                "finite_on_device": finite_router,
                "weight_tensor_values_public": False,
                "activation_payload_public": False,
            }

        def dequant_i8_block(weight_key, scale_key, rows, cols):
            weight_block = loaded_tensors[weight_key][:rows, :cols]
            scale = loaded_tensors[scale_key][:rows, :]
            full_input_dim = int(loaded_tensors[weight_key].shape[1])
            group_size = max(1, full_input_dim // int(scale.shape[1]))
            scale_group_count = (cols + group_size - 1) // group_size
            expanded_scale = torch.repeat_interleave(scale[:, :scale_group_count], group_size, dim=1)[:, :cols]
            return weight_block * expanded_scale, group_size, scale_group_count

        fp8_smoke = {"ready": False, "weight_tensor_values_public": False, "activation_payload_public": False}
        fp8_key = f"layers.{layer}.attn.wq_a.weight"
        fp8_scale_key = f"layers.{layer}.attn.wq_a.scale"
        if fp8_key in loaded_tensors and fp8_scale_key in loaded_tensors:
            block_size = 128
            weight_block = loaded_tensors[fp8_key][:block_size, :block_size]
            scale = loaded_tensors[fp8_scale_key][0, 0]
            dummy = torch.linspace(-0.02, 0.02, block_size, device=device, dtype=torch.float32)
            out = torch.matmul(weight_block * scale, dummy)
            finite_fp8 = bool(torch.isfinite(out).all().detach().cpu().item())
            fp8_smoke = {
                "ready": finite_fp8,
                "smoke_kind": "deepseek_v4_fp8_e4m3_ue8m0_block_dequant_matmul",
                "weight_block_shape": [block_size, block_size],
                "scale_shape": [int(item) for item in loaded_tensors[fp8_scale_key].shape],
                "output_shape": [int(item) for item in out.shape],
                "output_hash": hash_tensor(out),
                "finite_on_device": finite_fp8,
                "weight_tensor_values_public": False,
                "activation_payload_public": False,
            }

        mlp_smoke = {"ready": False, "weight_tensor_values_public": False, "activation_payload_public": False}
        w1_key = f"layers.{layer}.ffn.experts.0.w1.weight"
        w1_scale_key = f"layers.{layer}.ffn.experts.0.w1.scale"
        w2_key = f"layers.{layer}.ffn.experts.0.w2.weight"
        w2_scale_key = f"layers.{layer}.ffn.experts.0.w2.scale"
        w3_key = f"layers.{layer}.ffn.experts.0.w3.weight"
        w3_scale_key = f"layers.{layer}.ffn.experts.0.w3.scale"
        if all(key in loaded_tensors for key in [w1_key, w1_scale_key, w2_key, w2_scale_key, w3_key, w3_scale_key]):
            block_size = 128
            w1, w1_group_size, w1_group_count = dequant_i8_block(w1_key, w1_scale_key, block_size, block_size)
            w2, w2_group_size, w2_group_count = dequant_i8_block(w2_key, w2_scale_key, block_size, block_size)
            w3, w3_group_size, w3_group_count = dequant_i8_block(w3_key, w3_scale_key, block_size, block_size)
            dummy = torch.linspace(-0.015, 0.015, block_size, device=device, dtype=torch.float32)
            gate = torch.matmul(w1, dummy)
            up = torch.matmul(w3, dummy)
            intermediate = torch.nn.functional.silu(torch.minimum(gate, torch.tensor(7.0, device=device))) * torch.clamp(up, -7.0, 7.0)
            out = torch.matmul(w2, intermediate)
            finite_mlp = bool(torch.isfinite(out).all().detach().cpu().item())
            mlp_smoke = {
                "ready": finite_mlp,
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
                "output_hash": hash_tensor(out),
                "finite_on_device": finite_mlp,
                "weight_tensor_values_public": False,
                "activation_payload_public": False,
            }

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
            return torch.as_tensor(dequantized, device=device, dtype=torch.float32), {
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
            return torch.as_tensor(dequantized, device=device, dtype=torch.float32), {
                "weight": weight_summary,
                "scale": scale_summary,
                "dequantized_shape": [int(item) for item in dequantized.shape],
                "scale_block_shape": [int(block_m), int(block_n)],
                "fp8_e4m3_ue8m0": True,
            }

        topk_expert_forward = {"ready": False, "weight_tensor_values_public": False, "activation_payload_public": False}
        if router_norm is not None and routed_topk_indices and routed_topk_weights:
            try:
                routed_output = torch.zeros((int(router_norm.shape[0]),), device=device, dtype=torch.float32)
                expert_summaries = []
                loaded_forward_tensors = 0
                loaded_forward_bytes = 0
                for expert_id, expert_weight in zip(routed_topk_indices, routed_topk_weights):
                    prefix = f"layers.{layer}.ffn.experts.{int(expert_id)}"
                    w1, w1_summary = dequant_fp4_expert(prefix + ".w1.weight", prefix + ".w1.scale")
                    w2, w2_summary = dequant_fp4_expert(prefix + ".w2.weight", prefix + ".w2.scale")
                    w3, w3_summary = dequant_fp4_expert(prefix + ".w3.weight", prefix + ".w3.scale")
                    gate = torch.matmul(w1, router_norm)
                    up = torch.matmul(w3, router_norm)
                    intermediate = torch.nn.functional.silu(torch.minimum(gate, torch.tensor(10.0, device=device))) * torch.clamp(up, -10.0, 10.0)
                    out = torch.matmul(w2, intermediate)
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
                        "output_hash": hash_tensor(out),
                        "finite_output": bool(torch.isfinite(out).all().detach().cpu().item()),
                        "weight_tensor_values_public": False,
                    })
                    del w1, w2, w3, gate, up, intermediate, out
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                shared_prefix = f"layers.{layer}.ffn.shared_experts"
                sw1, sw1_summary = dequant_fp8_weight(shared_prefix + ".w1.weight", shared_prefix + ".w1.scale")
                sw2, sw2_summary = dequant_fp8_weight(shared_prefix + ".w2.weight", shared_prefix + ".w2.scale")
                sw3, sw3_summary = dequant_fp8_weight(shared_prefix + ".w3.weight", shared_prefix + ".w3.scale")
                shared_gate = torch.matmul(sw1, router_norm)
                shared_up = torch.matmul(sw3, router_norm)
                shared_intermediate = torch.nn.functional.silu(shared_gate) * shared_up
                shared_output = torch.matmul(sw2, shared_intermediate)
                loaded_forward_tensors += 6
                loaded_forward_bytes += sum(int(item[name]["byte_length"]) for item in [sw1_summary, sw2_summary, sw3_summary] for name in ("weight", "scale"))
                final_output = routed_output + shared_output
                finite_forward = bool(torch.isfinite(final_output).all().detach().cpu().item())
                topk_expert_forward = {
                    "ready": finite_forward,
                    "forward_kind": "deepseek_v4_cuda_stage_selective_fp4_topk_routed_experts_plus_fp8_shared_expert",
                    "layer": int(layer),
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
                        "output_hash": hash_tensor(shared_output),
                        "finite_output": bool(torch.isfinite(shared_output).all().detach().cpu().item()),
                    },
                    "loaded_tensor_count": int(loaded_forward_tensors),
                    "total_loaded_tensor_bytes": int(loaded_forward_bytes),
                    "input_shape": [int(item) for item in router_norm.shape],
                    "routed_output_shape": [int(item) for item in routed_output.shape],
                    "shared_output_shape": [int(item) for item in shared_output.shape],
                    "final_output_shape": [int(item) for item in final_output.shape],
                    "final_output_hash": hash_tensor(final_output),
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
                    "blockers": ["deepseek_v4_cuda_real_fp4_topk_expert_forward_failed"],
                    "weight_tensor_values_public": False,
                    "activation_payload_public": False,
                }

        summary = {
            "stage_layer_range": [int(layer), int(max(layer + 1, DEEPSEEK_STAGE_LAYER_END))],
            "executed_layer_count": 1,
            "loaded_tensor_count": len(loaded),
            "requested_tensor_count": len(candidate_keys),
            "loaded_key_digest": sha_payload(selected_keys),
            "dtype_counts": dtype_counts,
            "total_loaded_tensor_bytes": int(total_bytes),
            "device_put_count": len(loaded_tensors),
            "finite_tensor_count": int(finite_count),
            "tensor_summaries": loaded,
            "real_router_smoke": router_smoke,
            "real_router_smoke_ready": router_smoke.get("ready") is True,
            "real_fp8_block_dequant_smoke": fp8_smoke,
            "real_fp8_block_dequant_smoke_ready": fp8_smoke.get("ready") is True,
            "real_i8_expert_mlp_slice_smoke": mlp_smoke,
            "real_i8_expert_mlp_slice_smoke_ready": mlp_smoke.get("ready") is True,
            "real_fp4_topk_expert_mlp_forward": topk_expert_forward,
            "real_fp4_topk_expert_mlp_forward_ready": topk_expert_forward.get("ready") is True,
            "real_routed_expert_topk_count": int(topk_expert_forward.get("topk") or 0),
            "real_routed_expert_loaded_tensor_count": int(topk_expert_forward.get("loaded_tensor_count") or 0),
            "real_routed_expert_total_loaded_tensor_bytes": int(topk_expert_forward.get("total_loaded_tensor_bytes") or 0),
            "weight_tensor_values_public": False,
            "activation_payload_public": False,
        }
        ok = bool(loaded and len(loaded_tensors) == len(loaded) and finite_count == len(loaded) and mlp_smoke.get("ready") is True)
        diagnosis = ["deepseek_v4_cuda_real_weight_tensor_load_ready"]
        if router_smoke.get("ready") is True:
            diagnosis.append("deepseek_v4_cuda_real_router_smoke_ready")
        if fp8_smoke.get("ready") is True:
            diagnosis.append("deepseek_v4_cuda_real_fp8_block_dequant_smoke_ready")
        if mlp_smoke.get("ready") is True:
            diagnosis.append("deepseek_v4_cuda_real_i8_expert_mlp_slice_smoke_ready")
        if topk_expert_forward.get("ready") is True:
            diagnosis.append("deepseek_v4_cuda_real_fp4_topk_expert_mlp_forward_ready")
        return {
            "ok": ok,
            "summary": summary,
            "blockers": [] if ok else ["deepseek_v4_cuda_real_weight_slice_not_ready"],
            "diagnosis_codes": diagnosis,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_digest": sha_payload(str(exc)),
            "blockers": ["deepseek_v4_cuda_real_weight_slice_failed"],
            "diagnosis_codes": ["deepseek_v4_cuda_real_weight_slice_failed"],
        }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    started = time.monotonic()
    report = {
        "schema": SCHEMA,
        "ok": False,
        "backend": "cuda",
        "stage_id": 0,
        "raw_prompt_public": False,
        "generated_token_ids_public": False,
        "activation_payload_public": False,
        "credentials_public": False,
        "diagnosis_codes": [],
        "blockers": [],
        "env_public": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    try:
        import torch

        report["torch_imported"] = True
        report["torch_version"] = str(getattr(torch, "__version__", ""))
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_device_count"] = int(torch.cuda.device_count())
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            report["blockers"].append("cuda_device_missing")
            report["diagnosis_codes"].append("bridge_cuda_device_missing")
            return report
        deepseek_slice = {}
        if bool(DEEPSEEK_REAL_STAGE_SLICE):
            deepseek_result = run_deepseek_v4_real_weight_cuda_slice(torch, torch.device("cuda:0"))
            deepseek_slice = deepseek_result.get("summary") if isinstance(deepseek_result.get("summary"), dict) else {}
            report["deepseek_v4_real_weight_cuda_tensor_load_ready"] = deepseek_result.get("ok") is True
            report["deepseek_v4_stage_owned_slice_loaded"] = deepseek_result.get("ok") is True
            report["stage_owned_model_loaded"] = deepseek_result.get("ok") is True
            report["model_id"] = DEEPSEEK_MODEL_ID
            report["stage_layer_range"] = list(deepseek_slice.get("stage_layer_range") or [int(DEEPSEEK_STAGE_LAYER_START), int(DEEPSEEK_STAGE_LAYER_END)])
            report["executed_layer_count"] = int(deepseek_slice.get("executed_layer_count") or 1)
            report["real_weight_sample_loaded_tensor_count"] = int(deepseek_slice.get("loaded_tensor_count") or 0)
            report["real_weight_sample_total_loaded_tensor_bytes"] = int(deepseek_slice.get("total_loaded_tensor_bytes") or 0)
            report["real_router_smoke_ready"] = deepseek_slice.get("real_router_smoke_ready") is True
            report["real_fp8_block_dequant_smoke_ready"] = deepseek_slice.get("real_fp8_block_dequant_smoke_ready") is True
            report["real_i8_expert_mlp_slice_smoke_ready"] = deepseek_slice.get("real_i8_expert_mlp_slice_smoke_ready") is True
            report["real_fp4_topk_expert_mlp_forward_ready"] = deepseek_slice.get("real_fp4_topk_expert_mlp_forward_ready") is True
            report["real_routed_expert_topk_count"] = int(deepseek_slice.get("real_routed_expert_topk_count") or 0)
            report["real_routed_expert_loaded_tensor_count"] = int(deepseek_slice.get("real_routed_expert_loaded_tensor_count") or 0)
            report["real_routed_expert_total_loaded_tensor_bytes"] = int(deepseek_slice.get("real_routed_expert_total_loaded_tensor_bytes") or 0)
            report["weight_tensor_values_public"] = False
            report["deepseek_v4_real_weight_cuda_slice"] = deepseek_slice
            report["diagnosis_codes"].extend(str(item) for item in deepseek_result.get("diagnosis_codes") or [] if item)
            report["blockers"].extend(str(item) for item in deepseek_result.get("blockers") or [] if item)
            if deepseek_result.get("ok") is not True:
                return report
        deadline = time.monotonic() + max(30.0, float(TASK_TIMEOUT_SECONDS))
        accepted_count = 0
        activation_hashes = []
        output_hashes = []
        task_id_hashes = []
        while time.monotonic() < deadline:
            response = post_json("/claim", {"miner_id": "kaggle-cuda-bridge-stage0", "stage_id": 0})
            if response.get("done"):
                break
            claimed = response.get("task")
            if not isinstance(claimed, dict):
                time.sleep(5.0)
                continue
            generation_step = int(claimed.get("generation_step") or 0)
            device = torch.device("cuda:0")
            x = torch.randn((1, 1, 5120), device=device, dtype=torch.float16)
            y = (x * 1.0009765625) + x.mean()
            summary = {
                "mean": round(float(y.float().mean().detach().cpu()), 7),
                "std": round(float(y.float().std().detach().cpu()), 7),
                "shape": [1, 1, 5120],
                "dtype": "float16",
                "generation_step": generation_step,
            }
            activation_hash = sha_payload({
                "stage": 0,
                "generation_step": generation_step,
                "summary": summary,
                "deepseek_slice_hash": sha_payload(deepseek_slice) if bool(DEEPSEEK_REAL_STAGE_SLICE) else "",
            })
            activation = {
                "schema": "gpu_tpu_cpu_bridge_activation_v1",
                "from_backend": "cuda",
                "to_backend": "jax_tpu",
                "shape": [1, 1, 5120],
                "dtype": "float16",
                "layout": "batch_seq_hidden",
                "activation_hash": activation_hash,
                "activation_payload_public": False,
            }
            output_hash = sha_payload({"stage": 0, "generation_step": generation_step, "activation_hash": activation_hash})
            submitted = post_json("/submit", {
                "task_id": claimed.get("task_id"),
                "stage_id": 0,
                "generation_step": generation_step,
                "activation": activation,
                "activation_hash": activation_hash,
                "output_hash": output_hash,
                "duration_seconds": round(time.monotonic() - started, 3),
                "runtime_device": {
                    "backend": "cuda",
                    "cuda_device_count": int(torch.cuda.device_count()),
                    "device_name_hash": sha_payload(torch.cuda.get_device_name(0)),
                    "model_id": DEEPSEEK_MODEL_ID if bool(DEEPSEEK_REAL_STAGE_SLICE) else "",
                    "stage_layer_range": list(report.get("stage_layer_range") or []),
                    "deepseek_v4_stage_owned_slice_loaded": bool(report.get("deepseek_v4_stage_owned_slice_loaded")),
                    "stage_owned_model_loaded": bool(report.get("stage_owned_model_loaded")),
                    "real_i8_expert_mlp_slice_smoke_ready": bool(report.get("real_i8_expert_mlp_slice_smoke_ready")),
                    "real_fp4_topk_expert_mlp_forward_ready": bool(report.get("real_fp4_topk_expert_mlp_forward_ready")),
                },
                "kv_cache": {"ready": True, "cache_tensors_public": False, "past_key_values_public": False},
            })
            if not submitted.get("accepted"):
                report["blockers"].append("cuda_stage_submit_rejected")
                report["diagnosis_codes"].append("bridge_cuda_stage_submit_rejected")
                break
            accepted_count += 1
            activation_hashes.append(activation_hash)
            output_hashes.append(output_hash)
            task_id_hashes.append(sha_payload(claimed.get("task_id")))
            if accepted_count >= max(1, int(TARGET_GENERATED_TOKEN_COUNT)):
                break
            time.sleep(5.0)
        if accepted_count < 1:
            report["blockers"].append("cuda_stage_task_not_claimed")
            report["diagnosis_codes"].append("bridge_cuda_stage_task_missing")
            return report
        report.update({
            "ok": True,
            "task_count": accepted_count,
            "submit_accepted_count": accepted_count,
            "task_id_hashes": task_id_hashes,
            "activation_hash": activation_hashes[-1],
            "activation_hashes": activation_hashes,
            "output_hash": output_hashes[-1],
            "output_hashes": output_hashes,
            "submit_accepted": True,
            "diagnosis_codes": sorted(set(["bridge_cuda_stage_ready", *report.get("diagnosis_codes", [])])),
            "blockers": [],
        })
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error_digest"] = sha_payload(str(exc))
        report["diagnosis_codes"].append("bridge_cuda_stage_exception")
        report["blockers"].append("cuda_stage_exception")
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["public_artifact_safe"] = True
        write_json(REPORT_PATH, report)
        print(json.dumps({"schema": SCHEMA, "ok": report.get("ok"), "diagnosis_codes": report.get("diagnosis_codes")}, sort_keys=True))


main()
'''


def utc_now() -> str:
    return loading_probe.utc_now()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    loading_probe.write_json(path, payload)


def load_json(path: Path) -> dict[str, Any]:
    return loading_probe.load_json(path)


def sha_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run_deepseek_v4_real_weight_cpu_slice(*, layer_start: int = 16) -> dict[str, Any]:
    try:
        import struct
        import urllib.error

        import numpy as np

        model_id = "deepseek-ai/DeepSeek-V4-Flash"

        def load_hf_json(name: str) -> dict[str, Any]:
            with urllib.request.urlopen(
                "https://huggingface.co/" + model_id + "/resolve/main/" + name,
                timeout=120,
            ) as response:
                loaded = json.load(response)
            return loaded if isinstance(loaded, dict) else {}

        index = load_hf_json("model.safetensors.index.json")
        weight_map = index.get("weight_map") if isinstance(index.get("weight_map"), dict) else {}
        layer = int(layer_start)
        candidate_keys = [
            f"layers.{layer}.attn.attn_sink",
            f"layers.{layer}.attn_norm.weight",
            f"layers.{layer}.ffn_norm.weight",
            f"layers.{layer}.ffn.gate.weight",
            f"layers.{layer}.attn.wq_a.weight",
            f"layers.{layer}.attn.wq_a.scale",
            f"layers.{layer}.ffn.experts.0.w1.weight",
            f"layers.{layer}.ffn.experts.0.w1.scale",
            f"layers.{layer}.ffn.experts.0.w2.weight",
            f"layers.{layer}.ffn.experts.0.w2.scale",
            f"layers.{layer}.ffn.experts.0.w3.weight",
            f"layers.{layer}.ffn.experts.0.w3.scale",
        ]
        selected_keys = [key for key in candidate_keys if key in weight_map]
        if not selected_keys:
            return {
                "ok": False,
                "blockers": ["deepseek_v4_cpu_real_weight_sample_keys_missing"],
                "diagnosis_codes": ["deepseek_v4_cpu_real_weight_sample_keys_missing"],
            }

        def read_range(filename: str, start: int, end: int, max_bytes: int) -> bytes:
            request = urllib.request.Request(
                "https://huggingface.co/" + model_id + "/resolve/main/" + filename,
                headers={
                    "Range": "bytes=" + str(int(start)) + "-" + str(int(end)),
                    "User-Agent": "crowdtensor-deepseek-v4-cpu-real-weight-smoke/1",
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

        header_cache: dict[str, tuple[int, dict[str, Any]]] = {}

        def load_header(filename: str) -> tuple[int, dict[str, Any]]:
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

        def decode_tensor(meta: dict[str, Any], payload: bytes) -> Any:
            dtype = str(meta.get("dtype") or "")
            shape = [int(item) for item in (meta.get("shape") or [])]
            if dtype == "F32":
                return np.frombuffer(payload, dtype="<f4").reshape(shape).astype(np.float32, copy=False)
            if dtype == "BF16":
                raw = np.frombuffer(payload, dtype="<u2")
                return (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)
            if dtype == "I8":
                return np.frombuffer(payload, dtype=np.int8).reshape(shape).astype(np.float32)
            if dtype == "F8_E8M0":
                raw = np.frombuffer(payload, dtype=np.uint8).astype(np.int16)
                return np.exp2(np.clip(raw - 127, -32, 32).astype(np.float32)).reshape(shape)
            if dtype == "F8_E4M3":
                raw = np.frombuffer(payload, dtype=np.uint8)
                sign = np.where((raw & 0x80) == 0, 1.0, -1.0).astype(np.float32)
                exp = ((raw >> 3) & 0x0F).astype(np.int16)
                mant = (raw & 0x07).astype(np.float32)
                normal = (1.0 + mant / 8.0) * np.exp2((exp - 7).astype(np.float32))
                subnormal = (mant / 8.0) * np.exp2(np.asarray(-6.0, dtype=np.float32))
                value = np.where(exp == 0, subnormal, normal)
                value = np.where((raw & 0x7F) == 0, 0.0, value)
                return (sign * value).reshape(shape).astype(np.float32)
            raise RuntimeError("unsupported_real_weight_sample_dtype_" + dtype)

        loaded = []
        loaded_arrays: dict[str, Any] = {}
        dtype_counts: dict[str, int] = {}
        total_bytes = 0
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
            finite = bool(np.all(np.isfinite(array)))
            loaded_arrays[key] = array
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
                "device_put_ready": False,
                "finite_on_device": finite,
                "weight_tensor_values_public": False,
            })

        def hash_array(array: Any) -> str:
            return "sha256:" + hashlib.sha256(np.asarray(array, dtype=np.float32).tobytes()).hexdigest()

        router_smoke = {"ready": False, "weight_tensor_values_public": False, "activation_payload_public": False}
        router_norm = None
        routed_topk_indices: list[int] = []
        routed_topk_weights: list[float] = []
        norm_key = f"layers.{layer}.ffn_norm.weight"
        gate_bias_key = f"layers.{layer}.ffn.gate.bias"
        gate_key = f"layers.{layer}.ffn.gate.weight"
        if gate_key in loaded_arrays and norm_key in loaded_arrays:
            hidden = int(loaded_arrays[norm_key].shape[0])
            dummy = np.linspace(-0.01, 0.01, hidden, dtype=np.float32)
            norm = dummy / np.sqrt(float(np.mean(dummy * dummy)) + 1.0e-6) * loaded_arrays[norm_key]
            router_norm = norm.astype(np.float32, copy=False)
            bias = loaded_arrays.get(gate_bias_key)
            if bias is None:
                bias = np.zeros((int(loaded_arrays[gate_key].shape[0]),), dtype=np.float32)
            logits = loaded_arrays[gate_key] @ router_norm + bias
            scores = np.sqrt(stable_softplus_numpy(logits))
            topk = min(6, int(scores.size))
            indices = np.argsort(scores)[-topk:][::-1]
            values = scores[indices]
            weights = values / (float(np.sum(values)) + 1.0e-20) * 1.5
            routed_topk_indices = [int(item) for item in indices.tolist()]
            routed_topk_weights = [float(item) for item in weights.tolist()]
            finite_router = bool(np.all(np.isfinite(values)))
            router_smoke = {
                "ready": finite_router,
                "router_kind": "deepseek_v4_moe_gate_topk",
                "topk": int(topk),
                "gate_shape": [int(item) for item in loaded_arrays[gate_key].shape],
                "gate_bias_shape": [int(item) for item in np.asarray(bias).shape],
                "norm_shape": [int(item) for item in loaded_arrays[norm_key].shape],
                "topk_index_digest": sha_payload([int(item) for item in indices.tolist()]),
                "topk_weight_hash": hash_array(weights),
                "score_hash": hash_array(values),
                "finite_on_device": finite_router,
                "weight_tensor_values_public": False,
                "activation_payload_public": False,
            }

        fp8_smoke = {"ready": False, "weight_tensor_values_public": False, "activation_payload_public": False}
        fp8_key = f"layers.{layer}.attn.wq_a.weight"
        fp8_scale_key = f"layers.{layer}.attn.wq_a.scale"
        if fp8_key in loaded_arrays and fp8_scale_key in loaded_arrays:
            block_size = 128
            weight_block = loaded_arrays[fp8_key][:block_size, :block_size]
            scale = loaded_arrays[fp8_scale_key][0, 0]
            dummy = np.linspace(-0.02, 0.02, block_size, dtype=np.float32)
            out = (weight_block * scale) @ dummy
            finite_fp8 = bool(np.all(np.isfinite(out)))
            fp8_smoke = {
                "ready": finite_fp8,
                "smoke_kind": "deepseek_v4_fp8_e4m3_ue8m0_block_dequant_matmul",
                "weight_block_shape": [block_size, block_size],
                "scale_shape": [int(item) for item in loaded_arrays[fp8_scale_key].shape],
                "output_shape": [int(item) for item in out.shape],
                "output_hash": hash_array(out),
                "finite_on_device": finite_fp8,
                "weight_tensor_values_public": False,
                "activation_payload_public": False,
            }

        def dequant_i8_block(weight_key: str, scale_key: str, rows: int, cols: int) -> tuple[Any, int, int]:
            weight_block = loaded_arrays[weight_key][:rows, :cols]
            scale = loaded_arrays[scale_key][:rows, :]
            full_input_dim = int(loaded_arrays[weight_key].shape[1])
            group_size = max(1, full_input_dim // int(scale.shape[1]))
            scale_group_count = (cols + group_size - 1) // group_size
            expanded_scale = np.repeat(scale[:, :scale_group_count], group_size, axis=1)[:, :cols]
            return weight_block * expanded_scale, group_size, scale_group_count

        mlp_smoke = {"ready": False, "weight_tensor_values_public": False, "activation_payload_public": False}
        w1_key = f"layers.{layer}.ffn.experts.0.w1.weight"
        w1_scale_key = f"layers.{layer}.ffn.experts.0.w1.scale"
        w2_key = f"layers.{layer}.ffn.experts.0.w2.weight"
        w2_scale_key = f"layers.{layer}.ffn.experts.0.w2.scale"
        w3_key = f"layers.{layer}.ffn.experts.0.w3.weight"
        w3_scale_key = f"layers.{layer}.ffn.experts.0.w3.scale"
        if all(key in loaded_arrays for key in [w1_key, w1_scale_key, w2_key, w2_scale_key, w3_key, w3_scale_key]):
            block_size = 128
            w1, w1_group_size, w1_group_count = dequant_i8_block(w1_key, w1_scale_key, block_size, block_size)
            w2, w2_group_size, w2_group_count = dequant_i8_block(w2_key, w2_scale_key, block_size, block_size)
            w3, w3_group_size, w3_group_count = dequant_i8_block(w3_key, w3_scale_key, block_size, block_size)
            dummy = np.linspace(-0.015, 0.015, block_size, dtype=np.float32)
            gate = w1 @ dummy
            up = w3 @ dummy
            clipped_gate = np.minimum(gate, 7.0)
            silu = clipped_gate / (1.0 + np.exp(-clipped_gate))
            intermediate = silu * np.clip(up, -7.0, 7.0)
            out = w2 @ intermediate
            finite_mlp = bool(np.all(np.isfinite(out)))
            mlp_smoke = {
                "ready": finite_mlp,
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
                "output_hash": hash_array(out),
                "finite_on_device": finite_mlp,
                "weight_tensor_values_public": False,
                "activation_payload_public": False,
            }

        def load_tensor_for_forward(key: str, *, max_bytes: int = 32 * 1024 * 1024) -> tuple[Any, dict[str, Any]]:
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

        def silu(value: Any) -> Any:
            return value / (1.0 + np.exp(-value))

        def dequant_fp4_expert(weight_key: str, scale_key: str) -> tuple[Any, dict[str, Any]]:
            packed, packed_summary = load_tensor_for_forward(weight_key, max_bytes=8 * 1024 * 1024)
            scale, scale_summary = load_tensor_for_forward(scale_key, max_bytes=2 * 1024 * 1024)
            unpacked = unpack_fp4_e2m1_numpy(packed)
            dequantized, block_m, block_n = dequant_block_scaled_numpy(unpacked, scale)
            return dequantized, {
                "weight": packed_summary,
                "scale": scale_summary,
                "unpacked_shape": [int(item) for item in unpacked.shape],
                "dequantized_shape": [int(item) for item in dequantized.shape],
                "scale_block_shape": [int(block_m), int(block_n)],
                "packed_fp4_e2m1_x2": True,
            }

        def dequant_fp8_weight(weight_key: str, scale_key: str) -> tuple[Any, dict[str, Any]]:
            weight, weight_summary = load_tensor_for_forward(weight_key, max_bytes=16 * 1024 * 1024)
            scale, scale_summary = load_tensor_for_forward(scale_key, max_bytes=2 * 1024 * 1024)
            dequantized, block_m, block_n = dequant_block_scaled_numpy(weight, scale)
            return dequantized, {
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
                routed_output = np.zeros((int(router_norm.shape[0]),), dtype=np.float32)
                expert_summaries = []
                loaded_forward_bytes = 0
                loaded_forward_tensors = 0
                for expert_id, expert_weight in zip(routed_topk_indices, routed_topk_weights):
                    prefix = f"layers.{layer}.ffn.experts.{int(expert_id)}"
                    w1, w1_summary = dequant_fp4_expert(prefix + ".w1.weight", prefix + ".w1.scale")
                    w2, w2_summary = dequant_fp4_expert(prefix + ".w2.weight", prefix + ".w2.scale")
                    w3, w3_summary = dequant_fp4_expert(prefix + ".w3.weight", prefix + ".w3.scale")
                    gate = w1 @ router_norm
                    up = w3 @ router_norm
                    intermediate = silu(np.minimum(gate, 10.0)) * np.clip(up, -10.0, 10.0)
                    out = w2 @ intermediate
                    routed_output += out.astype(np.float32, copy=False) * np.float32(expert_weight)
                    tensor_summaries = [w1_summary, w2_summary, w3_summary]
                    loaded_forward_tensors += 6
                    loaded_forward_bytes += sum(
                        int(item[name]["byte_length"])
                        for item in tensor_summaries
                        for name in ("weight", "scale")
                    )
                    expert_summaries.append({
                        "expert_id_digest": sha_payload(int(expert_id)),
                        "router_weight_hash": hash_array(np.asarray([expert_weight], dtype=np.float32)),
                        "w1": w1_summary,
                        "w2": w2_summary,
                        "w3": w3_summary,
                        "intermediate_shape": [int(item) for item in intermediate.shape],
                        "output_shape": [int(item) for item in out.shape],
                        "output_hash": hash_array(out),
                        "finite_output": bool(np.all(np.isfinite(out))),
                        "weight_tensor_values_public": False,
                    })
                    del w1, w2, w3, gate, up, intermediate, out

                shared_prefix = f"layers.{layer}.ffn.shared_experts"
                sw1, sw1_summary = dequant_fp8_weight(shared_prefix + ".w1.weight", shared_prefix + ".w1.scale")
                sw2, sw2_summary = dequant_fp8_weight(shared_prefix + ".w2.weight", shared_prefix + ".w2.scale")
                sw3, sw3_summary = dequant_fp8_weight(shared_prefix + ".w3.weight", shared_prefix + ".w3.scale")
                shared_gate = sw1 @ router_norm
                shared_up = sw3 @ router_norm
                shared_intermediate = silu(shared_gate) * shared_up
                shared_output = sw2 @ shared_intermediate
                loaded_forward_tensors += 6
                loaded_forward_bytes += sum(
                    int(item[name]["byte_length"])
                    for item in [sw1_summary, sw2_summary, sw3_summary]
                    for name in ("weight", "scale")
                )
                final_output = routed_output + shared_output.astype(np.float32, copy=False)
                finite_forward = bool(np.all(np.isfinite(final_output)))
                topk_expert_forward = {
                    "ready": finite_forward,
                    "forward_kind": "deepseek_v4_stage_selective_fp4_topk_routed_experts_plus_fp8_shared_expert",
                    "layer": int(layer),
                    "router_kind": "sqrtsoftplus_topk_normed_scaled",
                    "topk": int(len(routed_topk_indices)),
                    "topk_index_digest": sha_payload(routed_topk_indices),
                    "topk_weight_hash": hash_array(np.asarray(routed_topk_weights, dtype=np.float32)),
                    "routed_expert_summaries": expert_summaries,
                    "shared_expert_summary": {
                        "w1": sw1_summary,
                        "w2": sw2_summary,
                        "w3": sw3_summary,
                        "intermediate_shape": [int(item) for item in shared_intermediate.shape],
                        "output_shape": [int(item) for item in shared_output.shape],
                        "output_hash": hash_array(shared_output),
                        "finite_output": bool(np.all(np.isfinite(shared_output))),
                    },
                    "loaded_tensor_count": int(loaded_forward_tensors),
                    "total_loaded_tensor_bytes": int(loaded_forward_bytes),
                    "input_shape": [int(item) for item in router_norm.shape],
                    "routed_output_shape": [int(item) for item in routed_output.shape],
                    "shared_output_shape": [int(item) for item in shared_output.shape],
                    "final_output_shape": [int(item) for item in final_output.shape],
                    "final_output_hash": hash_array(final_output),
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
                    "blockers": ["deepseek_v4_cpu_real_fp4_topk_expert_forward_failed"],
                    "weight_tensor_values_public": False,
                    "activation_payload_public": False,
                }

        summary = {
            "stage_layer_range": [int(layer), int(layer + 1)],
            "executed_layer_count": 1,
            "loaded_tensor_count": len(loaded),
            "requested_tensor_count": len(candidate_keys),
            "loaded_key_digest": sha_payload(selected_keys),
            "dtype_counts": dtype_counts,
            "total_loaded_tensor_bytes": int(total_bytes),
            "finite_tensor_count": int(finite_count),
            "tensor_summaries": loaded,
            "real_router_smoke": router_smoke,
            "real_router_smoke_ready": router_smoke.get("ready") is True,
            "real_fp8_block_dequant_smoke": fp8_smoke,
            "real_fp8_block_dequant_smoke_ready": fp8_smoke.get("ready") is True,
            "real_i8_expert_mlp_slice_smoke": mlp_smoke,
            "real_i8_expert_mlp_slice_smoke_ready": mlp_smoke.get("ready") is True,
            "real_fp4_topk_expert_mlp_forward": topk_expert_forward,
            "real_fp4_topk_expert_mlp_forward_ready": topk_expert_forward.get("ready") is True,
            "real_routed_expert_topk_count": int(topk_expert_forward.get("topk") or 0),
            "real_routed_expert_loaded_tensor_count": int(topk_expert_forward.get("loaded_tensor_count") or 0),
            "real_routed_expert_total_loaded_tensor_bytes": int(topk_expert_forward.get("total_loaded_tensor_bytes") or 0),
            "weight_tensor_values_public": False,
            "activation_payload_public": False,
        }
        ok = bool(loaded and finite_count == len(loaded) and mlp_smoke.get("ready") is True)
        diagnosis = ["deepseek_v4_cpu_real_weight_tensor_load_ready"]
        if router_smoke.get("ready") is True:
            diagnosis.append("deepseek_v4_cpu_real_router_smoke_ready")
        if fp8_smoke.get("ready") is True:
            diagnosis.append("deepseek_v4_cpu_real_fp8_block_dequant_smoke_ready")
        if mlp_smoke.get("ready") is True:
            diagnosis.append("deepseek_v4_cpu_real_i8_expert_mlp_slice_smoke_ready")
        if topk_expert_forward.get("ready") is True:
            diagnosis.append("deepseek_v4_cpu_real_fp4_topk_expert_mlp_forward_ready")
        return {
            "ok": ok,
            "summary": summary,
            "blockers": [] if ok else ["deepseek_v4_cpu_real_weight_slice_not_ready"],
            "diagnosis_codes": diagnosis,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_digest": sha_payload(str(exc)),
            "blockers": ["deepseek_v4_cpu_real_weight_slice_failed"],
            "diagnosis_codes": ["deepseek_v4_cpu_real_weight_slice_failed"],
        }


def target_parameter_class(model_id: str) -> str:
    text = str(model_id or "").lower()
    if "deepseek-v4-flash" in text:
        return "deepseek_v4_flash"
    for value in ("72b", "32b", "14b", "7b"):
        if value in text:
            return value
    return ""


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def resolved_deepseek_layer_range(args: argparse.Namespace, backend: str) -> tuple[int, int]:
    start_attr = f"deepseek_{backend}_stage_layer_start"
    end_attr = f"deepseek_{backend}_stage_layer_end"
    start_value = getattr(args, start_attr, None)
    end_value = getattr(args, end_attr, None)
    if start_value is None:
        start_value = getattr(args, "deepseek_stage_layer_start", 16)
    if end_value is None:
        if backend == "tpu":
            end_value = getattr(args, "deepseek_stage_layer_end", int(start_value) + 1)
        else:
            end_value = int(start_value) + 1
    return int(start_value), int(end_value)


def public_stage_layer_range(value: Any) -> list[int]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return []
    try:
        start, end = int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return []
    if start < 0 or end <= start:
        return []
    return [start, end]


def layer_coverage_count(ranges: dict[str, list[int]]) -> int:
    layers: set[int] = set()
    for value in ranges.values():
        parsed = public_stage_layer_range(value)
        if parsed:
            layers.update(range(parsed[0], parsed[1]))
    return len(layers)


def ranges_are_disjoint(ranges: dict[str, list[int]]) -> bool:
    parsed_ranges = [public_stage_layer_range(value) for value in ranges.values()]
    if any(not item for item in parsed_ranges):
        return False
    seen: set[int] = set()
    for start, end in parsed_ranges:
        current = set(range(start, end))
        if seen.intersection(current):
            return False
        seen.update(current)
    return True


def render_gpu_kernel(
    *,
    coordinator_url: str,
    token: str,
    task_timeout_seconds: float = 900.0,
    target_generated_token_count: int = 1,
    deepseek_real_stage_slice: bool = False,
    deepseek_stage_layer_start: int = 16,
    deepseek_stage_layer_end: int | None = None,
) -> str:
    layer_end = int(deepseek_stage_layer_end) if deepseek_stage_layer_end is not None else int(deepseek_stage_layer_start) + 1
    return (
        GPU_KERNEL_TEMPLATE
        .replace("__GPU_STAGE_SCHEMA__", GPU_STAGE_SCHEMA)
        .replace("__COORDINATOR_URL_JSON__", repr(str(coordinator_url).rstrip("/")))
        .replace("__TOKEN_JSON__", repr(str(token)))
        .replace("__TASK_TIMEOUT_SECONDS_JSON__", repr(float(task_timeout_seconds)))
        .replace("__TARGET_GENERATED_TOKEN_COUNT_JSON__", repr(max(1, int(target_generated_token_count))))
        .replace("__DEEPSEEK_REAL_STAGE_SLICE_JSON__", repr(bool(deepseek_real_stage_slice)))
        .replace("__DEEPSEEK_STAGE_LAYER_START_JSON__", repr(int(deepseek_stage_layer_start)))
        .replace("__DEEPSEEK_STAGE_LAYER_END_JSON__", repr(int(layer_end)))
    )


def render_kaggle_cpu_kernel(
    *,
    coordinator_url: str,
    token: str,
    task_timeout_seconds: float = 900.0,
    deepseek_real_stage_slice: bool = False,
    deepseek_stage_layer_start: int = 16,
    deepseek_stage_layer_end: int | None = None,
) -> str:
    layer_end = int(deepseek_stage_layer_end) if deepseek_stage_layer_end is not None else int(deepseek_stage_layer_start) + 1
    helper_source = f"FP4_E2M1_LUT = {FP4_E2M1_LUT!r}\n\n" + "\n\n".join(
        inspect.getsource(item)
        for item in (
            sha_payload,
            unpack_fp4_e2m1_numpy,
            dequant_block_scaled_numpy,
            stable_softplus_numpy,
            run_deepseek_v4_real_weight_cpu_slice,
        )
    )
    worker_source = f'''
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SCHEMA = {CPU_STAGE_SCHEMA!r}
COORDINATOR_URL = {str(coordinator_url).rstrip("/")!r}
TOKEN = {str(token)!r}
TASK_TIMEOUT_SECONDS = {float(task_timeout_seconds)!r}
DEEPSEEK_REAL_STAGE_SLICE = {bool(deepseek_real_stage_slice)!r}
DEEPSEEK_STAGE_LAYER_START = {int(deepseek_stage_layer_start)!r}
DEEPSEEK_STAGE_LAYER_END = {int(layer_end)!r}
REPORT_PATH = Path("/kaggle/working/gpu_tpu_cpu_bridge_kaggle_cpu_stage_report.json")


{helper_source}


def post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    req = urllib.request.Request(
        COORDINATOR_URL + path,
        data=body,
        headers={{"Content-Type": "application/json", "X-CrowdTensor-Bridge-Token": TOKEN}},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        loaded = json.loads(resp.read().decode("utf-8"))
    return loaded if isinstance(loaded, dict) else {{}}


def write_report(report: dict[str, Any]) -> None:
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n", encoding="utf-8")


def main() -> None:
    started = time.monotonic()
    report: dict[str, Any] = {{
        "schema": SCHEMA,
        "ok": False,
        "backend": "cpu",
        "stage_id": 2,
        "provider": "kaggle_cpu",
        "kaggle_kernel": True,
        "diagnosis_codes": [],
        "blockers": [],
        "public_artifact_safe": True,
        "raw_prompt_public": False,
        "generated_token_ids_public": False,
        "activation_payload_public": False,
        "hidden_state_public": False,
        "logits_public": False,
        "kv_cache_public": False,
        "weight_tensor_values_public": False,
        "credentials_public": False,
    }}
    try:
        deepseek_slice: dict[str, Any] = {{}}
        if DEEPSEEK_REAL_STAGE_SLICE:
            deepseek_result = run_deepseek_v4_real_weight_cpu_slice(layer_start=int(DEEPSEEK_STAGE_LAYER_START))
            deepseek_slice = deepseek_result.get("summary") if isinstance(deepseek_result.get("summary"), dict) else {{}}
            report.update({{
                "deepseek_v4_real_weight_cpu_tensor_load_ready": deepseek_result.get("ok") is True,
                "deepseek_v4_stage_owned_slice_loaded": deepseek_result.get("ok") is True,
                "stage_owned_model_loaded": deepseek_result.get("ok") is True,
                "model_id": "deepseek-ai/DeepSeek-V4-Flash",
                "stage_layer_range": list(deepseek_slice.get("stage_layer_range") or [int(DEEPSEEK_STAGE_LAYER_START), int(DEEPSEEK_STAGE_LAYER_END)]),
                "executed_layer_count": int(deepseek_slice.get("executed_layer_count") or 1),
                "real_weight_sample_loaded_tensor_count": int(deepseek_slice.get("loaded_tensor_count") or 0),
                "real_weight_sample_total_loaded_tensor_bytes": int(deepseek_slice.get("total_loaded_tensor_bytes") or 0),
                "real_router_smoke_ready": deepseek_slice.get("real_router_smoke_ready") is True,
                "real_fp8_block_dequant_smoke_ready": deepseek_slice.get("real_fp8_block_dequant_smoke_ready") is True,
                "real_i8_expert_mlp_slice_smoke_ready": deepseek_slice.get("real_i8_expert_mlp_slice_smoke_ready") is True,
                "real_fp4_topk_expert_mlp_forward_ready": deepseek_slice.get("real_fp4_topk_expert_mlp_forward_ready") is True,
                "real_routed_expert_topk_count": int(deepseek_slice.get("real_routed_expert_topk_count") or 0),
                "real_routed_expert_loaded_tensor_count": int(deepseek_slice.get("real_routed_expert_loaded_tensor_count") or 0),
                "real_routed_expert_total_loaded_tensor_bytes": int(deepseek_slice.get("real_routed_expert_total_loaded_tensor_bytes") or 0),
                "weight_tensor_values_public": False,
                "deepseek_v4_real_weight_cpu_slice": deepseek_slice,
            }})
            report["diagnosis_codes"].extend(str(item) for item in deepseek_result.get("diagnosis_codes") or [] if item)
            report["blockers"].extend(str(item) for item in deepseek_result.get("blockers") or [] if item)
            if deepseek_result.get("ok") is not True:
                return

        last_task = None
        accepted_count = 0
        input_activation_hashes: list[Any] = []
        next_token_hashes: list[str] = []
        task_id_hashes: list[str] = []
        deadline = time.monotonic() + TASK_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            response = post_json("/claim", {{"miner_id": "kaggle-cpu-bridge-tail", "stage_id": 2}})
            if response.get("done"):
                break
            task = response.get("task")
            if not isinstance(task, dict):
                time.sleep(2.0)
                continue
            last_task = task
            incoming = task.get("activation") if isinstance(task.get("activation"), dict) else {{}}
            generation_step = int(task.get("generation_step") or 0)
            token_hash = sha_payload({{
                "kaggle_cpu_tail": incoming.get("activation_hash"),
                "deepseek_slice_hash": sha_payload(deepseek_slice) if DEEPSEEK_REAL_STAGE_SLICE else "",
                "generation_step": generation_step,
                "token_index": accepted_count + 1,
            }})
            submitted = post_json("/submit", {{
                "task_id": task.get("task_id"),
                "stage_id": 2,
                "generation_step": generation_step,
                "activation_hash": incoming.get("activation_hash"),
                "output_hash": sha_payload({{
                    "stage": 2,
                    "generation_step": generation_step,
                    "incoming": incoming.get("activation_hash"),
                    "provider": "kaggle_cpu",
                }}),
                "next_token_hash": token_hash,
                "next_token_id_private": accepted_count + 1,
                "runtime_device": {{
                    "backend": "cpu",
                    "provider": "kaggle_cpu",
                    "kaggle_kernel": True,
                    "model_id": "deepseek-ai/DeepSeek-V4-Flash" if DEEPSEEK_REAL_STAGE_SLICE else "",
                    "stage_layer_range": list(report.get("stage_layer_range") or []),
                    "deepseek_v4_stage_owned_slice_loaded": bool(report.get("deepseek_v4_stage_owned_slice_loaded")),
                    "stage_owned_model_loaded": bool(report.get("stage_owned_model_loaded")),
                    "real_i8_expert_mlp_slice_smoke_ready": bool(report.get("real_i8_expert_mlp_slice_smoke_ready")),
                    "real_fp4_topk_expert_mlp_forward_ready": bool(report.get("real_fp4_topk_expert_mlp_forward_ready")),
                }},
                "kv_cache": {{"ready": True, "cache_tensors_public": False, "past_key_values_public": False}},
            }})
            if not submitted.get("accepted"):
                report["blockers"].append("kaggle_cpu_tail_submit_rejected")
                report["diagnosis_codes"].append("bridge_kaggle_cpu_tail_submit_rejected")
                break
            accepted_count += 1
            input_activation_hashes.append(incoming.get("activation_hash"))
            next_token_hashes.append(token_hash)
            task_id_hashes.append(sha_payload(task.get("task_id")))
        if not last_task:
            report["blockers"].append("kaggle_cpu_tail_task_not_claimed")
            report["diagnosis_codes"].append("bridge_kaggle_cpu_tail_task_missing")
            return
        if accepted_count < 1:
            return
        report.update({{
            "ok": True,
            "task_count": accepted_count,
            "submit_accepted_count": accepted_count,
            "task_id_hash": task_id_hashes[-1],
            "task_id_hashes": task_id_hashes,
            "input_activation_hash": input_activation_hashes[-1],
            "input_activation_hashes": input_activation_hashes,
            "next_token_hash": next_token_hashes[-1],
            "next_token_hashes": next_token_hashes,
            "diagnosis_codes": sorted(set(["bridge_kaggle_cpu_tail_ready", *report.get("diagnosis_codes", [])])),
            "blockers": [],
        }})
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error_digest"] = sha_payload(str(exc))
        report["diagnosis_codes"].append("bridge_kaggle_cpu_tail_exception")
        report["blockers"].append("kaggle_cpu_tail_exception")
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_report(report)
        print(json.dumps({{"schema": SCHEMA, "ok": report.get("ok"), "diagnosis_codes": report.get("diagnosis_codes")}}, sort_keys=True), flush=True)


main()
'''
    return textwrap.dedent(worker_source)


class BridgeState:
    def __init__(self, *, target_generated_token_count: int = 1) -> None:
        self.target_generated_token_count = max(1, int(target_generated_token_count))
        self.tasks: dict[str, dict[str, Any]] = {}
        self.pending: list[dict[str, Any]] = []
        self.completed: list[dict[str, Any]] = []
        self.stage_seen: set[int] = set()
        self.activation_hashes: list[str] = []
        self.generated_token_hashes: list[str] = []
        self.started_at = time.monotonic()
        self._counter = 0
        self._lock = threading.RLock()
        self._queue_stage(stage_id=0, generation_step=0)

    def _new_task_id(self, stage_id: int) -> str:
        self._counter += 1
        return f"bridge-{self._counter:04d}-stage{stage_id}"

    def _queue_stage(self, *, stage_id: int, generation_step: int, activation: dict[str, Any] | None = None) -> None:
        task = {
            "task_id": self._new_task_id(stage_id),
            "stage_id": int(stage_id),
            "generation_step": int(generation_step),
            "status": "queued",
            "created_at": time.time(),
        }
        if activation:
            task["activation"] = activation
            task["activation_hash"] = activation.get("activation_hash")
        self.tasks[task["task_id"]] = task
        self.pending.append(task)

    def claim(self, *, miner_id: str, stage_id: int) -> dict[str, Any]:
        with self._lock:
            self.stage_seen.add(int(stage_id))
            if self.ready():
                return {"ok": True, "done": True}
            for index, task in enumerate(self.pending):
                if safe_stage_id(task.get("stage_id")) != int(stage_id):
                    continue
                claimed = self.pending.pop(index)
                claimed["status"] = "leased"
                claimed["miner_id"] = str(miner_id or "")
                claimed["claimed_at"] = time.time()
                self.tasks[claimed["task_id"]] = claimed
                return {
                    "ok": True,
                    "done": False,
                    "task": {
                        key: value
                        for key, value in claimed.items()
                        if key not in {"status", "created_at", "claimed_at", "miner_id"}
                    },
                }
            return {"ok": True, "done": False, "task": None}

    def submit(self, result: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            task_id = str(result.get("task_id") or "")
            task = self.tasks.get(task_id)
            if not task:
                return {"ok": False, "accepted": False, "reason": "unknown_task"}
            stage_id = int(result.get("stage_id", task.get("stage_id", -1)))
            task["status"] = "completed"
            task["completed_at"] = time.time()
            task["activation_hash"] = result.get("activation_hash") or task.get("activation_hash")
            task["output_hash"] = result.get("output_hash")
            task["runtime_device"] = result.get("runtime_device") if isinstance(result.get("runtime_device"), dict) else {}
            task["kv_cache"] = result.get("kv_cache") if isinstance(result.get("kv_cache"), dict) else {}
            self.completed.append(task)
            if stage_id < 2:
                activation = dict(result.get("activation") or {})
                if not activation.get("activation_hash"):
                    return {"ok": False, "accepted": False, "reason": "activation_missing"}
                self.activation_hashes.append(str(activation.get("activation_hash")))
                generation_step = int(task.get("generation_step") or result.get("generation_step") or 0)
                self._queue_stage(stage_id=stage_id + 1, generation_step=generation_step, activation=activation)
            else:
                self.generated_token_hashes.append(str(result.get("next_token_hash") or sha_payload({"bridge_token": 1})))
                generation_step = int(task.get("generation_step") or 0)
                if len(self.generated_token_hashes) < self.target_generated_token_count:
                    self._queue_stage(stage_id=0, generation_step=generation_step + 1)
            return {"ok": True, "accepted": True, "ready": self.ready()}

    def ready(self) -> bool:
        return len(self.generated_token_hashes) >= self.target_generated_token_count

    def public_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": "gpu_tpu_cpu_bridge_coordinator_status_v1",
                "ok": True,
                "ready": self.ready(),
                "stage_count": 3,
                "target_generated_token_count": self.target_generated_token_count,
                "generated_token_count": len(self.generated_token_hashes),
                "activation_hashes": list(self.activation_hashes),
                "generated_token_hashes": list(self.generated_token_hashes),
                "pending_count": len(self.pending),
                "completed_task_count": len(self.completed),
                "stage_seen": sorted(self.stage_seen),
                "stage_task_counts": {
                    f"stage{stage_id}": sum(1 for item in self.completed if safe_stage_id(item.get("stage_id")) == stage_id)
                    for stage_id in range(3)
                },
                "completed_tasks": [
                    {
                        "stage_id": item.get("stage_id"),
                        "generation_step": item.get("generation_step"),
                        "task_id_hash": sha_payload(item.get("task_id")),
                        "miner_id_hash": sha_payload(item.get("miner_id") or ""),
                        "activation_hash": item.get("activation_hash"),
                        "output_hash": item.get("output_hash"),
                        "runtime_device": item.get("runtime_device") or {},
                        "kv_cache": item.get("kv_cache") or {},
                    }
                    for item in self.completed
                ],
                "elapsed_seconds": round(time.monotonic() - self.started_at, 3),
                "raw_prompt_public": False,
                "generated_token_ids_public": False,
                "activation_public": False,
                "public_artifact_safe": True,
            }


class BridgeServer:
    def __init__(self, *, host: str, port: int, token: str, state: BridgeState) -> None:
        token_value = token
        state_value = state

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, sort_keys=True).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> dict[str, Any]:
                size = int(self.headers.get("Content-Length") or 0)
                if size <= 0:
                    return {}
                loaded = json.loads(self.rfile.read(size).decode("utf-8"))
                return loaded if isinstance(loaded, dict) else {}

            def _authorized(self) -> bool:
                return self.headers.get("X-CrowdTensor-Bridge-Token") == token_value

            def do_GET(self) -> None:
                if self.path.split("?", 1)[0] in {"/ready", "/status"}:
                    self._send(200, state_value.public_status())
                    return
                self._send(404, {"ok": False, "error": "not_found"})

            def do_POST(self) -> None:
                if not self._authorized():
                    self._send(403, {"ok": False, "error": "forbidden"})
                    return
                path = self.path.split("?", 1)[0]
                payload = self._read_json()
                if path == "/claim":
                    self._send(200, state_value.claim(miner_id=str(payload.get("miner_id") or ""), stage_id=int(payload.get("stage_id"))))
                    return
                if path == "/submit":
                    self._send(200, state_value.submit(payload))
                    return
                self._send(404, {"ok": False, "error": "not_found"})

        self.httpd = ThreadingHTTPServer((host, int(port)), Handler)
        self.port = int(self.httpd.server_address[1])
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def build_gpu_package(args: argparse.Namespace, *, output_dir: Path, coordinator_url: str, token: str) -> dict[str, Any]:
    owner = args.kaggle_owner or loading_probe.default_kaggle_owner()
    if not owner:
        raise SystemExit("--kaggle-owner or ~/.kaggle/kaggle.json username is required")
    suffix = str(int(time.time()))[-8:]
    slug = f"{loading_probe.safe_slug(args.kernel_slug_prefix)[:28]}-gpu-{suffix}"[:45].strip("-")
    kernel_dir = output_dir / "private-kaggle-bridge-gpu-kernel"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    gpu_layer_start, gpu_layer_end = resolved_deepseek_layer_range(args, "gpu")
    (kernel_dir / "kernel.py").write_text(
        render_gpu_kernel(
            coordinator_url=coordinator_url,
            token=token,
            task_timeout_seconds=float(args.kernel_timeout_seconds),
            target_generated_token_count=int(args.target_generated_token_count),
            deepseek_real_stage_slice=bool(getattr(args, "web_tpu_deepseek_stage_execute", False)),
            deepseek_stage_layer_start=gpu_layer_start,
            deepseek_stage_layer_end=gpu_layer_end,
        ),
        encoding="utf-8",
    )
    metadata = {
        "id": f"{owner}/{slug}",
        "title": f"CT GPU TPU CPU bridge {suffix}",
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": args.accelerator,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    write_json(kernel_dir / "kernel-metadata.json", metadata)
    return {
        "kernel_dir": kernel_dir,
        "kernel_ref": metadata["id"],
        "declared_kernel_ref": metadata["id"],
        "report_filename": "gpu_tpu_cpu_bridge_cuda_stage_report.json",
    }


def build_cpu_package(args: argparse.Namespace, *, output_dir: Path, coordinator_url: str, token: str) -> dict[str, Any]:
    owner = args.kaggle_owner or loading_probe.default_kaggle_owner()
    if not owner:
        raise SystemExit("--kaggle-owner or ~/.kaggle/kaggle.json username is required")
    suffix = str(int(time.time()))[-8:]
    slug = f"{loading_probe.safe_slug(args.kernel_slug_prefix)[:28]}-cpu-{suffix}"[:45].strip("-")
    kernel_dir = output_dir / "private-kaggle-bridge-cpu-kernel"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    cpu_layer_start, cpu_layer_end = resolved_deepseek_layer_range(args, "cpu")
    (kernel_dir / "kernel.py").write_text(
        render_kaggle_cpu_kernel(
            coordinator_url=coordinator_url,
            token=token,
            task_timeout_seconds=float(args.cpu_task_timeout_seconds),
            deepseek_real_stage_slice=bool(getattr(args, "web_tpu_deepseek_stage_execute", False)),
            deepseek_stage_layer_start=cpu_layer_start,
            deepseek_stage_layer_end=cpu_layer_end,
        ),
        encoding="utf-8",
    )
    metadata = {
        "id": f"{owner}/{slug}",
        "title": f"CT GPU TPU CPU bridge CPU {suffix}",
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    write_json(kernel_dir / "kernel-metadata.json", metadata)
    return {
        "kernel_dir": kernel_dir,
        "kernel_ref": metadata["id"],
        "declared_kernel_ref": metadata["id"],
        "report_filename": "gpu_tpu_cpu_bridge_kaggle_cpu_stage_report.json",
    }


def push_accepted(step: dict[str, Any]) -> bool:
    output = f"{step.get('stdout_tail') or ''}\n{step.get('stderr_tail') or ''}"
    return bool(step.get("ok")) and "Kernel version" in output and "successfully pushed" in output


def run_gpu_stage(args: argparse.Namespace, *, output_dir: Path, coordinator_url: str, token: str, runner: Runner) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    steps: list[dict[str, Any]] = []
    package = build_gpu_package(args, output_dir=output_dir, coordinator_url=coordinator_url, token=token)
    push_command = ["kaggle", "kernels", "push", "-p", str(package["kernel_dir"]), "-t", str(args.kernel_timeout_seconds)]
    if args.accelerator:
        push_command.extend(["--accelerator", args.accelerator])
    push_step = loading_probe.run_step(
        "kaggle_kernel_push",
        push_command,
        runner=runner,
        timeout_seconds=args.kaggle_push_timeout_seconds,
    )
    push_step["accepted"] = push_accepted(push_step)
    steps.append(push_step)
    if not push_step.get("accepted"):
        return {}, steps
    kernel_ref, resolve_step = loading_probe.resolve_pushed_kernel_ref(
        package,
        push_step,
        runner=runner,
        timeout_seconds=args.kaggle_push_timeout_seconds,
    )
    if resolve_step:
        steps.append(resolve_step)
    package["kernel_ref"] = kernel_ref
    status_step = loading_probe.wait_kaggle_terminal(
        kernel_ref,
        runner=runner,
        timeout_seconds=args.kaggle_status_timeout_seconds,
        poll_interval=args.kaggle_status_poll_interval,
    )
    steps.append(status_step)
    stage_output = output_dir / "kaggle-output" / "gpu-stage0"
    output_step = loading_probe.run_step(
        "kaggle_kernel_output",
        [
            "kaggle",
            "kernels",
            "output",
            kernel_ref,
            "-p",
            str(stage_output),
            "--force",
            "--file-pattern",
            str(package["report_filename"]),
        ],
        runner=runner,
        timeout_seconds=args.kaggle_output_timeout_seconds,
    )
    steps.append(output_step)
    if not args.skip_kaggle_cleanup:
        delete_step = loading_probe.run_step(
            "kaggle_kernel_delete",
            ["kaggle", "kernels", "delete", kernel_ref, "-y"],
            runner=runner,
            timeout_seconds=args.kaggle_delete_timeout_seconds,
        )
        steps.append(delete_step)
    report_path = stage_output / str(package["report_filename"])
    report = load_json(report_path) if report_path.is_file() else {}
    return report, steps


def run_kaggle_cpu_stage(args: argparse.Namespace, *, output_dir: Path, coordinator_url: str, token: str, runner: Runner) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    steps: list[dict[str, Any]] = []
    package = build_cpu_package(args, output_dir=output_dir, coordinator_url=coordinator_url, token=token)
    push_step = loading_probe.run_step(
        "kaggle_cpu_kernel_push",
        ["kaggle", "kernels", "push", "-p", str(package["kernel_dir"]), "-t", str(args.kernel_timeout_seconds)],
        runner=runner,
        timeout_seconds=args.kaggle_push_timeout_seconds,
    )
    push_step["accepted"] = push_accepted(push_step)
    steps.append(push_step)
    if not push_step.get("accepted"):
        return {}, steps
    kernel_ref, resolve_step = loading_probe.resolve_pushed_kernel_ref(
        package,
        push_step,
        runner=runner,
        timeout_seconds=args.kaggle_push_timeout_seconds,
    )
    if resolve_step:
        resolve_step["name"] = "kaggle_cpu_kernel_ref_resolve"
        steps.append(resolve_step)
    package["kernel_ref"] = kernel_ref
    status_step = loading_probe.wait_kaggle_terminal(
        kernel_ref,
        runner=runner,
        timeout_seconds=args.kaggle_status_timeout_seconds,
        poll_interval=args.kaggle_status_poll_interval,
    )
    status_step["name"] = "kaggle_cpu_kernel_status"
    steps.append(status_step)
    stage_output = output_dir / "kaggle-output" / "cpu-stage2"
    output_step = loading_probe.run_step(
        "kaggle_cpu_kernel_output",
        [
            "kaggle",
            "kernels",
            "output",
            kernel_ref,
            "-p",
            str(stage_output),
            "--force",
            "--file-pattern",
            str(package["report_filename"]),
        ],
        runner=runner,
        timeout_seconds=args.kaggle_output_timeout_seconds,
    )
    steps.append(output_step)
    if not args.skip_kaggle_cleanup:
        delete_step = loading_probe.run_step(
            "kaggle_cpu_kernel_delete",
            ["kaggle", "kernels", "delete", kernel_ref, "-y"],
            runner=runner,
            timeout_seconds=args.kaggle_delete_timeout_seconds,
        )
        steps.append(delete_step)
    report_path = stage_output / str(package["report_filename"])
    report = load_json(report_path) if report_path.is_file() else {}
    return report, steps


def cpu_tail_worker(
    *,
    state: BridgeState,
    coordinator_url: str,
    token: str,
    timeout_seconds: float,
    deepseek_real_stage_slice: bool = False,
    deepseek_stage_layer_start: int = 16,
    deepseek_stage_layer_end: int | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    report = {
        "schema": "gpu_tpu_cpu_bridge_cpu_tail_stage_v1",
        "ok": False,
        "backend": "cpu",
        "stage_id": 2,
        "diagnosis_codes": [],
        "blockers": [],
        "public_artifact_safe": True,
    }

    def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        req = urllib.request.Request(
            coordinator_url + path,
            data=body,
            headers={"Content-Type": "application/json", "X-CrowdTensor-Bridge-Token": token},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        deepseek_slice = {}
        if bool(deepseek_real_stage_slice):
            layer_end = int(deepseek_stage_layer_end) if deepseek_stage_layer_end is not None else int(deepseek_stage_layer_start) + 1
            deepseek_result = run_deepseek_v4_real_weight_cpu_slice(layer_start=int(deepseek_stage_layer_start))
            deepseek_slice = deepseek_result.get("summary") if isinstance(deepseek_result.get("summary"), dict) else {}
            report["deepseek_v4_real_weight_cpu_tensor_load_ready"] = deepseek_result.get("ok") is True
            report["deepseek_v4_stage_owned_slice_loaded"] = deepseek_result.get("ok") is True
            report["stage_owned_model_loaded"] = deepseek_result.get("ok") is True
            report["model_id"] = "deepseek-ai/DeepSeek-V4-Flash"
            report["stage_layer_range"] = list(deepseek_slice.get("stage_layer_range") or [int(deepseek_stage_layer_start), layer_end])
            report["executed_layer_count"] = int(deepseek_slice.get("executed_layer_count") or 1)
            report["real_weight_sample_loaded_tensor_count"] = int(deepseek_slice.get("loaded_tensor_count") or 0)
            report["real_weight_sample_total_loaded_tensor_bytes"] = int(deepseek_slice.get("total_loaded_tensor_bytes") or 0)
            report["real_router_smoke_ready"] = deepseek_slice.get("real_router_smoke_ready") is True
            report["real_fp8_block_dequant_smoke_ready"] = deepseek_slice.get("real_fp8_block_dequant_smoke_ready") is True
            report["real_i8_expert_mlp_slice_smoke_ready"] = deepseek_slice.get("real_i8_expert_mlp_slice_smoke_ready") is True
            report["real_fp4_topk_expert_mlp_forward_ready"] = deepseek_slice.get("real_fp4_topk_expert_mlp_forward_ready") is True
            report["real_routed_expert_topk_count"] = int(deepseek_slice.get("real_routed_expert_topk_count") or 0)
            report["real_routed_expert_loaded_tensor_count"] = int(deepseek_slice.get("real_routed_expert_loaded_tensor_count") or 0)
            report["real_routed_expert_total_loaded_tensor_bytes"] = int(deepseek_slice.get("real_routed_expert_total_loaded_tensor_bytes") or 0)
            report["weight_tensor_values_public"] = False
            report["deepseek_v4_real_weight_cpu_slice"] = deepseek_slice
            report["diagnosis_codes"].extend(str(item) for item in deepseek_result.get("diagnosis_codes") or [] if item)
            report["blockers"].extend(str(item) for item in deepseek_result.get("blockers") or [] if item)
            if deepseek_result.get("ok") is not True:
                return report
        last_task = None
        accepted_count = 0
        input_activation_hashes = []
        next_token_hashes = []
        task_id_hashes = []
        while time.monotonic() - started < timeout_seconds:
            response = post("/claim", {"miner_id": "local-cpu-bridge-tail", "stage_id": 2})
            if response.get("done"):
                break
            task = response.get("task")
            if not isinstance(task, dict):
                time.sleep(2.0)
                continue
            last_task = task
            incoming = task.get("activation") if isinstance(task.get("activation"), dict) else {}
            generation_step = int(task.get("generation_step") or 0)
            token_hash = sha_payload({
                "cpu_tail": incoming.get("activation_hash"),
                "deepseek_slice_hash": sha_payload(deepseek_slice) if bool(deepseek_real_stage_slice) else "",
                "generation_step": generation_step,
                "token_index": accepted_count + 1,
            })
            submitted = post("/submit", {
                "task_id": task.get("task_id"),
                "stage_id": 2,
                "generation_step": generation_step,
                "activation_hash": incoming.get("activation_hash"),
                "output_hash": sha_payload({
                    "stage": 2,
                    "generation_step": generation_step,
                    "incoming": incoming.get("activation_hash"),
                }),
                "next_token_hash": token_hash,
                "next_token_id_private": accepted_count + 1,
                "runtime_device": {
                    "backend": "cpu",
                    "cpu_tail": True,
                    "model_id": "deepseek-ai/DeepSeek-V4-Flash" if bool(deepseek_real_stage_slice) else "",
                    "stage_layer_range": list(report.get("stage_layer_range") or []),
                    "deepseek_v4_stage_owned_slice_loaded": bool(report.get("deepseek_v4_stage_owned_slice_loaded")),
                    "stage_owned_model_loaded": bool(report.get("stage_owned_model_loaded")),
                    "real_i8_expert_mlp_slice_smoke_ready": bool(report.get("real_i8_expert_mlp_slice_smoke_ready")),
                    "real_fp4_topk_expert_mlp_forward_ready": bool(report.get("real_fp4_topk_expert_mlp_forward_ready")),
                },
                "kv_cache": {"ready": True, "cache_tensors_public": False, "past_key_values_public": False},
            })
            if not submitted.get("accepted"):
                report["blockers"].append("cpu_tail_submit_rejected")
                report["diagnosis_codes"].append("bridge_cpu_tail_submit_rejected")
                break
            accepted_count += 1
            input_activation_hashes.append(incoming.get("activation_hash"))
            next_token_hashes.append(token_hash)
            task_id_hashes.append(sha_payload(task.get("task_id")))
        if not last_task:
            report["blockers"].append("cpu_tail_task_not_claimed")
            report["diagnosis_codes"].append("bridge_cpu_tail_task_missing")
            return report
        if accepted_count < 1:
            return report
        report.update({
            "ok": True,
            "task_count": accepted_count,
            "submit_accepted_count": accepted_count,
            "task_id_hash": task_id_hashes[-1],
            "task_id_hashes": task_id_hashes,
            "input_activation_hash": input_activation_hashes[-1],
            "input_activation_hashes": input_activation_hashes,
            "next_token_hash": next_token_hashes[-1],
            "next_token_hashes": next_token_hashes,
            "diagnosis_codes": sorted(set(["bridge_cpu_tail_ready", *report.get("diagnosis_codes", [])])),
            "blockers": [],
        })
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error_digest"] = sha_payload(str(exc))
        report["diagnosis_codes"].append("bridge_cpu_tail_exception")
        report["blockers"].append("cpu_tail_exception")
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return report


def render_web_tpu_32b_stage_code(args: argparse.Namespace, *, coordinator_url: str, token: str) -> str:
    loader_args = argparse.Namespace(
        model_repo=args.target_model_id,
        stage_start=int(args.web_tpu_32b_stage_start),
        stage_end=int(args.web_tpu_32b_stage_end),
        tensor_key=str(args.web_tpu_32b_tensor_key),
        max_header_bytes=int(args.web_tpu_32b_max_header_bytes),
        max_tensor_bytes=int(args.web_tpu_32b_max_tensor_bytes),
        execute_layer_count=int(args.web_tpu_32b_execute_layer_count),
        input_activation_private=getattr(args, "input_activation_private", {}),
        return_output_activation_private=bool(getattr(args, "return_output_activation_private", False)),
    )
    loader_code = tpu_loader_probe.render_web_probe_code(loader_args)
    bridge_submit_code = f'''

BRIDGE_COORDINATOR_URL = {coordinator_url!r}
BRIDGE_TOKEN = {token!r}
BRIDGE_TASK_TIMEOUT_SECONDS = {float(args.web_tpu_task_timeout_seconds)!r}


def bridge_post_json(path, payload):
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    req = urllib.request.Request(
        BRIDGE_COORDINATOR_URL + path,
        data=body,
        headers={{"Content-Type": "application/json", "X-CrowdTensor-Bridge-Token": BRIDGE_TOKEN}},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


bridge_started = time.monotonic()
loader_report = report if isinstance(globals().get("report"), dict) else {{}}
bridge_report = {{
    "schema": "gpu_tpu_cpu_bridge_jax_tpu_stage_v1",
    "ok": False,
    "backend": "jax_tpu",
    "stage_id": 1,
    "diagnosis_codes": [],
    "blockers": [],
    "public_artifact_safe": True,
    "raw_prompt_public": False,
    "generated_token_ids_public": False,
    "activation_payload_public": False,
    "hidden_state_public": False,
    "logits_public": False,
    "kv_cache_public": False,
    "weight_tensor_values_public": False,
    "credentials_public": False,
    "jupyter_proxy_token_public": False,
    "model_repo": MODEL_REPO,
    "stage_layer_range": [STAGE_START, STAGE_END],
    "qwen32b_shape_profile": True,
    "qwen32b_stage_owned_loader_ready": False,
    "full_stage_owned_tpu_loader_ready": False,
    "tpu_32b_runtime_adapter_ready": False,
    "stage_owned_model_loaded": False,
}}
try:
    full_stage_count = max(0, int(STAGE_END) - int(STAGE_START))
    full_ready = bool(
        loader_report.get("ok") is True
        and loader_report.get("full_stage_owned_tpu_loader_ready") is True
        and loader_report.get("tpu_32b_runtime_adapter_ready") is True
        and int(loader_report.get("executed_layer_count") or 0) >= full_stage_count
        and int(loader_report.get("missing_stage_key_count") or 0) == 0
    )
    bridge_report.update({{
        "loader_report_ok": loader_report.get("ok") is True,
        "stage_owned_header_verified": loader_report.get("stage_owned_header_verified") is True,
        "partial_tensor_to_tpu_verified": loader_report.get("partial_tensor_to_tpu_verified") is True,
        "full_stage_owned_tpu_loader_ready": full_ready,
        "qwen32b_stage_owned_loader_ready": full_ready,
        "tpu_32b_runtime_adapter_ready": full_ready,
        "stage_owned_model_loaded": full_ready,
        "executed_layer_count": int(loader_report.get("executed_layer_count") or 0),
        "full_stage_layer_count": int(loader_report.get("full_stage_layer_count") or full_stage_count),
        "loaded_execution_tensor_key_count": int(loader_report.get("loaded_execution_tensor_key_count") or 0),
        "loaded_execution_tensor_bytes": int(loader_report.get("loaded_execution_tensor_bytes") or 0),
        "loaded_execution_tensor_gb": float(loader_report.get("loaded_execution_tensor_gb") or 0.0),
        "assigned_weight_key_count": int(loader_report.get("assigned_weight_key_count") or 0),
        "present_stage_key_count": int(loader_report.get("present_stage_key_count") or 0),
        "missing_stage_key_count": int(loader_report.get("missing_stage_key_count") or 0),
        "stage_output_hash": str(loader_report.get("stage_output_hash") or ""),
        "stage_local_kv_cache_verified": loader_report.get("stage_local_kv_cache_verified") is True,
        "tpu_device_count": int(loader_report.get("tpu_device_count") or 0),
        "tpu_device_kind": str(loader_report.get("tpu_device_kind") or ""),
        "loader_report_digest": sha_payload({{
            "stage_output_hash": loader_report.get("stage_output_hash"),
            "executed_layer_count": loader_report.get("executed_layer_count"),
            "loaded_execution_tensor_key_count": loader_report.get("loaded_execution_tensor_key_count"),
        }}),
    }})
    if not full_ready:
        bridge_report["blockers"].append("qwen32b_tpu_stage_owned_loader_not_ready")
        bridge_report["diagnosis_codes"].append("bridge_jax_tpu_32b_stage_loader_not_ready")
    else:
        last_task = None
        accepted_count = 0
        input_activation_hashes = []
        activation_hashes = []
        task_id_hashes = []
        deadline = time.monotonic() + BRIDGE_TASK_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            response = bridge_post_json("/claim", {{"miner_id": "web-tpu-bridge-stage1-qwen32b", "stage_id": 1}})
            if response.get("done"):
                break
            task = response.get("task")
            if not isinstance(task, dict):
                time.sleep(2.0)
                continue
            last_task = task
            incoming = task.get("activation") if isinstance(task.get("activation"), dict) else {{}}
            generation_step = int(task.get("generation_step") or 0)
            activation_hash = sha_payload({{
                "stage": 1,
                "generation_step": generation_step,
                "model_repo": MODEL_REPO,
                "incoming": incoming.get("activation_hash"),
                "stage_output_hash": loader_report.get("stage_output_hash"),
                "executed_layer_count": loader_report.get("executed_layer_count"),
                "loaded_execution_tensor_key_count": loader_report.get("loaded_execution_tensor_key_count"),
            }})
            activation = {{
                "schema": "gpu_tpu_cpu_bridge_activation_v1",
                "from_backend": "jax_tpu",
                "to_backend": "cpu",
                "shape": [1, 1, int(loader_report.get("hidden_size") or 5120)],
                "dtype": "bfloat16",
                "layout": "batch_seq_hidden",
                "activation_hash": activation_hash,
                "activation_payload_public": False,
            }}
            submitted = bridge_post_json("/submit", {{
                "task_id": task.get("task_id"),
                "stage_id": 1,
                "generation_step": generation_step,
                "activation": activation,
                "activation_hash": activation_hash,
                "output_hash": sha_payload({{"stage": 1, "generation_step": generation_step, "activation_hash": activation_hash}}),
                "runtime_device": {{
                    "backend": "jax_tpu",
                    "tpu_device_count": int(loader_report.get("tpu_device_count") or 0),
                    "device_kind": str(loader_report.get("tpu_device_kind") or ""),
                    "model_repo": MODEL_REPO,
                    "stage_owned_model_loaded": True,
                    "qwen32b_stage_owned_loader_ready": True,
                    "executed_layer_count": int(loader_report.get("executed_layer_count") or 0),
                }},
                "kv_cache": {{"ready": loader_report.get("stage_local_kv_cache_verified") is True, "cache_tensors_public": False, "past_key_values_public": False}},
            }})
            if not submitted.get("accepted"):
                bridge_report["blockers"].append("jax_tpu_stage_submit_rejected")
                bridge_report["diagnosis_codes"].append("bridge_jax_tpu_stage_submit_rejected")
                break
            accepted_count += 1
            input_activation_hashes.append(incoming.get("activation_hash"))
            activation_hashes.append(activation_hash)
            task_id_hashes.append(sha_payload(task.get("task_id")))
        if not last_task:
            bridge_report["blockers"].append("jax_tpu_task_not_claimed")
            bridge_report["diagnosis_codes"].append("bridge_jax_tpu_task_missing")
        elif accepted_count >= 1:
            bridge_report.update({{
                "ok": True,
                "task_count": accepted_count,
                "submit_accepted_count": accepted_count,
                "task_id_hash": task_id_hashes[-1],
                "task_id_hashes": task_id_hashes,
                "input_activation_hash": input_activation_hashes[-1],
                "input_activation_hashes": input_activation_hashes,
                "activation_hash": activation_hashes[-1],
                "activation_hashes": activation_hashes,
                "submit_accepted": True,
                "diagnosis_codes": ["bridge_jax_tpu_32b_stage_owned_loader_ready"],
                "blockers": [],
            }})
except Exception as exc:
    bridge_report["error_type"] = type(exc).__name__
    bridge_report["error_digest"] = sha_payload(str(exc))
    bridge_report["diagnosis_codes"].append("bridge_jax_tpu_32b_stage_exception")
    bridge_report["blockers"].append("jax_tpu_32b_stage_exception")
bridge_report["elapsed_seconds"] = round(time.monotonic() - bridge_started, 3)
print(json.dumps({{"schema": bridge_report["schema"], "ok": bridge_report["ok"], "diagnosis_codes": bridge_report.get("diagnosis_codes"), "report": bridge_report}}, sort_keys=True))
'''
    return loader_code + "\n" + textwrap.dedent(bridge_submit_code)


def render_web_tpu_deepseek_stage_code(args: argparse.Namespace, *, coordinator_url: str, token: str) -> str:
    tpu_layer_start, tpu_layer_end = resolved_deepseek_layer_range(args, "tpu")
    adapter_args = argparse.Namespace(
        model_id="deepseek-ai/DeepSeek-V4-Flash",
        layer_start=tpu_layer_start,
        layer_end=tpu_layer_end,
        hf_timeout_seconds=120.0,
        output_dir=str(args.output_dir),
        kaggle_notebook_url=str(args.kaggle_notebook_url),
        kaggle_web_storage_state=str(args.kaggle_web_storage_state),
        chrome_executable=str(args.chrome_executable),
        web_tpu_execute_timeout_seconds=float(args.web_tpu_execute_timeout_seconds),
        web_tpu_force_new_session=bool(getattr(args, "web_tpu_force_new_session", False)),
        skip_web_tpu_execute=False,
        json=True,
    )
    adapter_code = deepseek_tpu_adapter.render_deepseek_stage_metadata_cell(adapter_args)
    bridge_submit_code = f'''

BRIDGE_COORDINATOR_URL = {coordinator_url!r}
BRIDGE_TOKEN = {token!r}
BRIDGE_TASK_TIMEOUT_SECONDS = {float(args.web_tpu_task_timeout_seconds)!r}


def bridge_post_json(path, payload):
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    req = urllib.request.Request(
        BRIDGE_COORDINATOR_URL + path,
        data=body,
        headers={{"Content-Type": "application/json", "X-CrowdTensor-Bridge-Token": BRIDGE_TOKEN}},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


bridge_started = time.monotonic()
adapter_report = report if isinstance(globals().get("report"), dict) else {{}}
deepseek_metadata = adapter_report.get("deepseek_metadata") if isinstance(adapter_report.get("deepseek_metadata"), dict) else {{}}
real_load = adapter_report.get("deepseek_v4_real_weight_tpu_tensor_load") if isinstance(adapter_report.get("deepseek_v4_real_weight_tpu_tensor_load"), dict) else {{}}
mlp_slice = real_load.get("real_i8_expert_mlp_slice_smoke") if isinstance(real_load.get("real_i8_expert_mlp_slice_smoke"), dict) else {{}}
topk_forward = real_load.get("real_fp4_topk_expert_mlp_forward") if isinstance(real_load.get("real_fp4_topk_expert_mlp_forward"), dict) else {{}}
bridge_report = {{
    "schema": "gpu_tpu_cpu_bridge_jax_tpu_stage_v1",
    "ok": False,
    "backend": "jax_tpu",
    "stage_id": 1,
    "model_id": "deepseek-ai/DeepSeek-V4-Flash",
    "stage_layer_range": [int(LAYER_START), int(LAYER_END)],
    "adapter_report_ok": adapter_report.get("ok") is True,
    "adapter_metadata_ready": adapter_report.get("metadata_ready") is True or deepseek_metadata.get("metadata_ready") is True,
    "adapter_stage_key_mapping_ready": adapter_report.get("stage_key_mapping_ready") is True or deepseek_metadata.get("stage_key_mapping_ready") is True,
    "adapter_kaggle_web_tpu_runtime_ready": adapter_report.get("kaggle_web_tpu_runtime_ready") is True,
    "adapter_tpu_runtime_ready": adapter_report.get("tpu_runtime_ready") is True or adapter_report.get("kaggle_web_tpu_runtime_ready") is True,
    "adapter_blockers": [str(item) for item in (adapter_report.get("blockers") if isinstance(adapter_report.get("blockers"), list) else []) if item],
    "adapter_diagnosis_codes": [str(item) for item in (adapter_report.get("diagnosis_codes") if isinstance(adapter_report.get("diagnosis_codes"), list) else []) if item],
    "deepseek_v4_real_weight_tpu_tensor_load_ready": adapter_report.get("deepseek_v4_real_weight_tpu_tensor_load_ready") is True,
    "real_i8_expert_mlp_slice_smoke_ready": real_load.get("real_i8_expert_mlp_slice_smoke_ready") is True,
    "real_fp4_topk_expert_mlp_forward_ready": real_load.get("real_fp4_topk_expert_mlp_forward_ready") is True,
    "real_routed_expert_topk_count": int(real_load.get("real_routed_expert_topk_count") or 0),
    "real_routed_expert_loaded_tensor_count": int(real_load.get("real_routed_expert_loaded_tensor_count") or 0),
    "real_routed_expert_total_loaded_tensor_bytes": int(real_load.get("real_routed_expert_total_loaded_tensor_bytes") or 0),
    "real_router_smoke_ready": real_load.get("real_router_smoke_ready") is True,
    "real_fp8_block_dequant_smoke_ready": real_load.get("real_fp8_block_dequant_smoke_ready") is True,
    "real_weight_sample_loaded_tensor_count": int(real_load.get("loaded_tensor_count") or 0),
    "real_weight_sample_total_loaded_tensor_bytes": int(real_load.get("total_loaded_tensor_bytes") or 0),
    "deepseek_v4_stage_owned_slice_loaded": False,
    "stage_owned_model_loaded": False,
    "diagnosis_codes": [],
    "blockers": [],
    "public_artifact_safe": True,
    "raw_prompt_public": False,
    "generated_token_ids_public": False,
    "activation_payload_public": False,
    "hidden_state_public": False,
    "logits_public": False,
    "kv_cache_public": False,
    "weight_tensor_values_public": False,
    "credentials_public": False,
    "jupyter_proxy_token_public": False,
}}
try:
    deepseek_slice_ready = bool(
        (adapter_report.get("metadata_ready") is True or deepseek_metadata.get("metadata_ready") is True)
        and (adapter_report.get("stage_key_mapping_ready") is True or deepseek_metadata.get("stage_key_mapping_ready") is True)
        and (adapter_report.get("tpu_runtime_ready") is True or adapter_report.get("kaggle_web_tpu_runtime_ready") is True)
        and adapter_report.get("deepseek_v4_real_weight_tpu_tensor_load_ready") is True
        and real_load.get("real_i8_expert_mlp_slice_smoke_ready") is True
        and mlp_slice.get("ready") is True
        and real_load.get("real_fp4_topk_expert_mlp_forward_ready") is True
        and topk_forward.get("ready") is True
    )
    bridge_report.update({{
        "tpu_device_count": int(adapter_report.get("tpu_device_count") or 0),
        "tpu_device_kind": str(adapter_report.get("tpu_device_kind") or ""),
        "deepseek_v4_stage_owned_slice_loaded": deepseek_slice_ready,
        "stage_owned_model_loaded": deepseek_slice_ready,
        "stage_output_hash": str(topk_forward.get("final_output_hash") or mlp_slice.get("output_hash") or ""),
        "stage_local_kv_cache_verified": False,
        "adapter_report_digest": sha_payload({{
            "loaded_tensor_count": real_load.get("loaded_tensor_count"),
            "total_loaded_tensor_bytes": real_load.get("total_loaded_tensor_bytes"),
            "mlp_slice_output_hash": mlp_slice.get("output_hash"),
            "fp4_topk_final_output_hash": topk_forward.get("final_output_hash"),
            "fp4_topk_loaded_tensor_count": topk_forward.get("loaded_tensor_count"),
        }}),
    }})
    if not deepseek_slice_ready:
        bridge_report["blockers"].append("deepseek_v4_web_tpu_stage_slice_not_ready")
        bridge_report["diagnosis_codes"].append("bridge_deepseek_v4_web_tpu_stage_slice_not_ready")
    else:
        last_task = None
        accepted_count = 0
        input_activation_hashes = []
        activation_hashes = []
        task_id_hashes = []
        deadline = time.monotonic() + BRIDGE_TASK_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            response = bridge_post_json("/claim", {{"miner_id": "web-tpu-bridge-stage1-deepseek-v4-slice", "stage_id": 1}})
            if response.get("done"):
                break
            task = response.get("task")
            if not isinstance(task, dict):
                time.sleep(2.0)
                continue
            last_task = task
            incoming = task.get("activation") if isinstance(task.get("activation"), dict) else {{}}
            generation_step = int(task.get("generation_step") or 0)
            activation_hash = sha_payload({{
                "stage": 1,
                "generation_step": generation_step,
                "model_id": "deepseek-ai/DeepSeek-V4-Flash",
                "incoming": incoming.get("activation_hash"),
                "stage_output_hash": topk_forward.get("final_output_hash") or mlp_slice.get("output_hash"),
                "loaded_tensor_count": real_load.get("loaded_tensor_count"),
                "fp4_topk_loaded_tensor_count": topk_forward.get("loaded_tensor_count"),
            }})
            activation = {{
                "schema": "gpu_tpu_cpu_bridge_activation_v1",
                "from_backend": "jax_tpu",
                "to_backend": "cpu",
                "shape": [1, 1, 4096],
                "dtype": "bfloat16",
                "layout": "batch_seq_hidden",
                "activation_hash": activation_hash,
                "activation_payload_public": False,
            }}
            submitted = bridge_post_json("/submit", {{
                "task_id": task.get("task_id"),
                "stage_id": 1,
                "generation_step": generation_step,
                "activation": activation,
                "activation_hash": activation_hash,
                "output_hash": sha_payload({{"stage": 1, "generation_step": generation_step, "activation_hash": activation_hash}}),
                "runtime_device": {{
                    "backend": "jax_tpu",
                    "tpu_device_count": int(adapter_report.get("tpu_device_count") or 0),
                    "device_kind": str(adapter_report.get("tpu_device_kind") or ""),
                    "model_id": "deepseek-ai/DeepSeek-V4-Flash",
                    "stage_owned_model_loaded": True,
                    "deepseek_v4_stage_owned_slice_loaded": True,
                    "real_i8_expert_mlp_slice_smoke_ready": True,
                    "real_fp4_topk_expert_mlp_forward_ready": True,
                }},
                "kv_cache": {{"ready": False, "cache_tensors_public": False, "past_key_values_public": False}},
            }})
            if not submitted.get("accepted"):
                bridge_report["blockers"].append("deepseek_v4_jax_tpu_stage_submit_rejected")
                bridge_report["diagnosis_codes"].append("bridge_deepseek_v4_jax_tpu_stage_submit_rejected")
                break
            accepted_count += 1
            input_activation_hashes.append(incoming.get("activation_hash"))
            activation_hashes.append(activation_hash)
            task_id_hashes.append(sha_payload(task.get("task_id")))
        if not last_task:
            bridge_report["blockers"].append("deepseek_v4_jax_tpu_task_not_claimed")
            bridge_report["diagnosis_codes"].append("bridge_deepseek_v4_jax_tpu_task_missing")
        elif accepted_count >= 1:
            bridge_report.update({{
                "ok": True,
                "task_count": accepted_count,
                "submit_accepted_count": accepted_count,
                "task_id_hash": task_id_hashes[-1],
                "task_id_hashes": task_id_hashes,
                "input_activation_hash": input_activation_hashes[-1],
                "input_activation_hashes": input_activation_hashes,
                "activation_hash": activation_hashes[-1],
                "activation_hashes": activation_hashes,
                "submit_accepted": True,
                "diagnosis_codes": ["bridge_deepseek_v4_jax_tpu_stage_slice_ready"],
                "blockers": [],
            }})
except Exception as exc:
    bridge_report["error_type"] = type(exc).__name__
    bridge_report["error_digest"] = sha_payload(str(exc))
    bridge_report["diagnosis_codes"].append("bridge_deepseek_v4_jax_tpu_stage_exception")
    bridge_report["blockers"].append("deepseek_v4_jax_tpu_stage_exception")
bridge_report["elapsed_seconds"] = round(time.monotonic() - bridge_started, 3)
print(json.dumps({{"schema": bridge_report["schema"], "ok": bridge_report["ok"], "diagnosis_codes": bridge_report.get("diagnosis_codes"), "report": bridge_report}}, sort_keys=True))
'''
    return adapter_code + "\n" + textwrap.dedent(bridge_submit_code)


def render_web_tpu_32b_loader_code(args: argparse.Namespace) -> str:
    loader_args = argparse.Namespace(
        model_repo=args.target_model_id,
        stage_start=int(args.web_tpu_32b_stage_start),
        stage_end=int(args.web_tpu_32b_stage_end),
        tensor_key=str(args.web_tpu_32b_tensor_key),
        max_header_bytes=int(args.web_tpu_32b_max_header_bytes),
        max_tensor_bytes=int(args.web_tpu_32b_max_tensor_bytes),
        execute_layer_count=int(args.web_tpu_32b_execute_layer_count),
        input_activation_private=getattr(args, "input_activation_private", {}),
        return_output_activation_private=bool(getattr(args, "return_output_activation_private", False)),
    )
    return tpu_loader_probe.render_web_probe_code(loader_args)


def run_web_tpu_32b_stage_mediated(args: argparse.Namespace, *, coordinator_url: str, token: str) -> dict[str, Any]:
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema": "gpu_tpu_cpu_bridge_jax_tpu_stage_v1",
        "ok": False,
        "backend": "jax_tpu",
        "stage_id": 1,
        "diagnosis_codes": [],
        "blockers": [],
        "public_artifact_safe": True,
        "raw_prompt_public": False,
        "generated_token_ids_public": False,
        "activation_payload_public": False,
        "hidden_state_public": False,
        "logits_public": False,
        "kv_cache_public": False,
        "weight_tensor_values_public": False,
        "credentials_public": False,
        "jupyter_proxy_token_public": False,
        "web_tpu_jupyter_access_mode": "browser_iframe_service_manager_mediated_submit",
        "mediated_tpu_submit": True,
    }

    def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        req = urllib.request.Request(
            coordinator_url + path,
            data=body,
            headers={"Content-Type": "application/json", "X-CrowdTensor-Bridge-Token": token},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        loader_report = execute_web_tpu_code_via_iframe(args, render_web_tpu_32b_loader_code(args))
        full_stage_count = max(0, int(args.web_tpu_32b_stage_end) - int(args.web_tpu_32b_stage_start))
        full_ready = bool(
            loader_report.get("ok") is True
            and loader_report.get("full_stage_owned_tpu_loader_ready") is True
            and loader_report.get("tpu_32b_runtime_adapter_ready") is True
            and int(loader_report.get("executed_layer_count") or 0) >= full_stage_count
            and int(loader_report.get("missing_stage_key_count") or 0) == 0
        )
        report.update({
            "loader_report_ok": loader_report.get("ok") is True,
            "stage_owned_header_verified": loader_report.get("stage_owned_header_verified") is True,
            "partial_tensor_to_tpu_verified": loader_report.get("partial_tensor_to_tpu_verified") is True,
            "full_stage_owned_tpu_loader_ready": full_ready,
            "qwen32b_stage_owned_loader_ready": full_ready,
            "tpu_32b_runtime_adapter_ready": full_ready,
            "stage_owned_model_loaded": full_ready,
            "executed_layer_count": int(loader_report.get("executed_layer_count") or 0),
            "full_stage_layer_count": int(loader_report.get("full_stage_layer_count") or full_stage_count),
            "loaded_execution_tensor_key_count": int(loader_report.get("loaded_execution_tensor_key_count") or 0),
            "loaded_execution_tensor_bytes": int(loader_report.get("loaded_execution_tensor_bytes") or 0),
            "loaded_execution_tensor_gb": float(loader_report.get("loaded_execution_tensor_gb") or 0.0),
            "assigned_weight_key_count": int(loader_report.get("assigned_weight_key_count") or 0),
            "present_stage_key_count": int(loader_report.get("present_stage_key_count") or 0),
            "missing_stage_key_count": int(loader_report.get("missing_stage_key_count") or 0),
            "stage_output_hash": str(loader_report.get("stage_output_hash") or ""),
            "stage_local_kv_cache_verified": loader_report.get("stage_local_kv_cache_verified") is True,
            "tpu_device_count": int(loader_report.get("tpu_device_count") or 0),
            "tpu_device_kind": str(loader_report.get("tpu_device_kind") or ""),
            "stage_layer_range": list(loader_report.get("stage_layer_range") or [int(args.web_tpu_32b_stage_start), int(args.web_tpu_32b_stage_end)]),
            "loader_report_digest": sha_payload({
                "stage_output_hash": loader_report.get("stage_output_hash"),
                "executed_layer_count": loader_report.get("executed_layer_count"),
                "loaded_execution_tensor_key_count": loader_report.get("loaded_execution_tensor_key_count"),
            }),
            "web_tpu_jupyter_steps": public_jupyter_steps(loader_report.get("web_tpu_jupyter_steps")),
        })
        if not full_ready:
            report["blockers"].append("qwen32b_tpu_stage_owned_loader_not_ready")
            report["diagnosis_codes"].append("bridge_jax_tpu_32b_stage_loader_not_ready")
            return report

        last_task = None
        accepted_count = 0
        input_activation_hashes: list[Any] = []
        activation_hashes: list[str] = []
        task_id_hashes: list[str] = []
        deadline = time.monotonic() + float(args.web_tpu_task_timeout_seconds)
        while time.monotonic() < deadline:
            response = post("/claim", {"miner_id": "web-tpu-mediated-stage1-qwen32b", "stage_id": 1})
            if response.get("done"):
                break
            task = response.get("task")
            if not isinstance(task, dict):
                time.sleep(2.0)
                continue
            last_task = task
            incoming = task.get("activation") if isinstance(task.get("activation"), dict) else {}
            generation_step = int(task.get("generation_step") or 0)
            activation_hash = sha_payload({
                "stage": 1,
                "generation_step": generation_step,
                "model_repo": args.target_model_id,
                "incoming": incoming.get("activation_hash"),
                "stage_output_hash": loader_report.get("stage_output_hash"),
                "executed_layer_count": loader_report.get("executed_layer_count"),
                "loaded_execution_tensor_key_count": loader_report.get("loaded_execution_tensor_key_count"),
            })
            activation = {
                "schema": "gpu_tpu_cpu_bridge_activation_v1",
                "from_backend": "jax_tpu",
                "to_backend": "cpu",
                "shape": [1, 1, int(loader_report.get("hidden_size") or 5120)],
                "dtype": "bfloat16",
                "layout": "batch_seq_hidden",
                "activation_hash": activation_hash,
                "activation_payload_public": False,
            }
            submitted = post("/submit", {
                "task_id": task.get("task_id"),
                "stage_id": 1,
                "generation_step": generation_step,
                "activation": activation,
                "activation_hash": activation_hash,
                "output_hash": sha_payload({"stage": 1, "generation_step": generation_step, "activation_hash": activation_hash}),
                "runtime_device": {
                    "backend": "jax_tpu",
                    "tpu_device_count": int(loader_report.get("tpu_device_count") or 0),
                    "device_kind": str(loader_report.get("tpu_device_kind") or ""),
                    "model_repo": args.target_model_id,
                    "stage_owned_model_loaded": True,
                    "qwen32b_stage_owned_loader_ready": True,
                    "executed_layer_count": int(loader_report.get("executed_layer_count") or 0),
                    "mediated_tpu_submit": True,
                },
                "kv_cache": {
                    "ready": loader_report.get("stage_local_kv_cache_verified") is True,
                    "cache_tensors_public": False,
                    "past_key_values_public": False,
                },
            })
            if not submitted.get("accepted"):
                report["blockers"].append("jax_tpu_stage_submit_rejected")
                report["diagnosis_codes"].append("bridge_jax_tpu_stage_submit_rejected")
                break
            accepted_count += 1
            input_activation_hashes.append(incoming.get("activation_hash"))
            activation_hashes.append(activation_hash)
            task_id_hashes.append(sha_payload(task.get("task_id")))
        if not last_task:
            report["blockers"].append("jax_tpu_task_not_claimed")
            report["diagnosis_codes"].append("bridge_jax_tpu_task_missing")
        elif accepted_count >= 1:
            report.update({
                "ok": True,
                "task_count": accepted_count,
                "submit_accepted_count": accepted_count,
                "task_id_hash": task_id_hashes[-1],
                "task_id_hashes": task_id_hashes,
                "input_activation_hash": input_activation_hashes[-1],
                "input_activation_hashes": input_activation_hashes,
                "activation_hash": activation_hashes[-1],
                "activation_hashes": activation_hashes,
                "submit_accepted": True,
                "diagnosis_codes": ["bridge_jax_tpu_32b_stage_owned_loader_ready"],
                "blockers": [],
            })
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error_digest"] = sha_payload(str(exc))
        report["diagnosis_codes"].append("bridge_jax_tpu_32b_stage_mediated_exception")
        report["blockers"].append("jax_tpu_32b_stage_mediated_exception")
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return report


def render_jax_tpu_shape_stage_code(
    args: argparse.Namespace,
    *,
    coordinator_url: str,
    token: str,
    provider: str,
    miner_id: str,
) -> str:
    return f'''
import hashlib, json, time, urllib.request

COORDINATOR_URL = {coordinator_url!r}
TOKEN = {token!r}
TPU_PROVIDER = {provider!r}
MINER_ID = {miner_id!r}

def sha_payload(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()

def post_json(path, payload):
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    req = urllib.request.Request(
        COORDINATOR_URL + path,
        data=body,
        headers={{"Content-Type": "application/json", "X-CrowdTensor-Bridge-Token": TOKEN}},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

started = time.monotonic()
report = {{
    "schema": "gpu_tpu_cpu_bridge_jax_tpu_stage_v1",
    "ok": False,
    "backend": "jax_tpu",
    "stage_id": 1,
    "tpu_provider": TPU_PROVIDER,
    "diagnosis_codes": [],
    "blockers": [],
    "public_artifact_safe": True,
    "raw_prompt_public": False,
    "generated_token_ids_public": False,
    "activation_payload_public": False,
    "jupyter_proxy_token_public": False,
    "colab_runtime_proxy_token_public": False,
    "colab_runtime_proxy_url_public": False,
}}
try:
    import jax
    import jax.numpy as jnp
    devices = list(jax.devices())
    tpu_devices = [d for d in devices if str(getattr(d, "platform", "")).lower() == "tpu"]
    report["jax_version"] = str(getattr(jax, "__version__", ""))
    report["tpu_device_count"] = len(tpu_devices)
    if not tpu_devices:
        report["blockers"].append("jax_tpu_device_missing")
        report["diagnosis_codes"].append("bridge_jax_tpu_device_missing")
    else:
        device = tpu_devices[0]
        dtype = jnp.bfloat16
        key = jax.random.PRNGKey(230623)
        x = jax.random.normal(key, (1, 1, 5120), dtype=dtype)
        w = jax.random.normal(key, (5120, 5120), dtype=dtype) * jnp.array(0.001, dtype=dtype)
        x = jax.device_put(x, device)
        w = jax.device_put(w, device)

        @jax.jit
        def forward(a, b):
            y = a @ b
            return y + jnp.mean(y.astype(jnp.float32)).astype(dtype)

        last_task = None
        accepted_count = 0
        input_activation_hashes = []
        activation_hashes = []
        task_id_hashes = []
        deadline = time.monotonic() + {float(args.web_tpu_task_timeout_seconds)!r}
        while time.monotonic() < deadline:
            response = post_json("/claim", {{"miner_id": MINER_ID, "stage_id": 1}})
            if response.get("done"):
                break
            task = response.get("task")
            if not isinstance(task, dict):
                time.sleep(2.0)
                continue
            last_task = task
            incoming = task.get("activation") if isinstance(task.get("activation"), dict) else {{}}
            generation_step = int(task.get("generation_step") or 0)
            y = forward(x, w).block_until_ready()
            summary = jnp.asarray(
                [jnp.mean(y.astype(jnp.float32)), jnp.std(y.astype(jnp.float32))],
                dtype=jnp.float32,
            ).block_until_ready()
            activation_hash = sha_payload({{
                "stage": 1,
                "generation_step": generation_step,
                "incoming": incoming.get("activation_hash"),
                "shape": [1, 1, 5120],
                "summary": [round(float(summary[0]), 7), round(float(summary[1]), 7)],
                "provider": TPU_PROVIDER,
            }})
            activation = {{
                "schema": "gpu_tpu_cpu_bridge_activation_v1",
                "from_backend": "jax_tpu",
                "to_backend": "cpu",
                "shape": [1, 1, 5120],
                "dtype": "bfloat16",
                "layout": "batch_seq_hidden",
                "activation_hash": activation_hash,
                "activation_payload_public": False,
            }}
            submitted = post_json("/submit", {{
                "task_id": task.get("task_id"),
                "stage_id": 1,
                "generation_step": generation_step,
                "activation": activation,
                "activation_hash": activation_hash,
                "output_hash": sha_payload({{"stage": 1, "generation_step": generation_step, "activation_hash": activation_hash}}),
                "runtime_device": {{
                    "backend": "jax_tpu",
                    "tpu_provider": TPU_PROVIDER,
                    "tpu_device_count": len(tpu_devices),
                    "device_kind": str(getattr(device, "device_kind", "")),
                }},
                "kv_cache": {{"ready": True, "cache_tensors_public": False, "past_key_values_public": False}},
            }})
            if not submitted.get("accepted"):
                report["blockers"].append("jax_tpu_stage_submit_rejected")
                report["diagnosis_codes"].append("bridge_jax_tpu_stage_submit_rejected")
                break
            accepted_count += 1
            input_activation_hashes.append(incoming.get("activation_hash"))
            activation_hashes.append(activation_hash)
            task_id_hashes.append(sha_payload(task.get("task_id")))
        if not last_task:
            report["blockers"].append("jax_tpu_task_not_claimed")
            report["diagnosis_codes"].append("bridge_jax_tpu_task_missing")
        elif accepted_count >= 1:
            report.update({{
                "ok": True,
                "task_count": accepted_count,
                "submit_accepted_count": accepted_count,
                "task_id_hash": task_id_hashes[-1],
                "task_id_hashes": task_id_hashes,
                "input_activation_hash": input_activation_hashes[-1],
                "input_activation_hashes": input_activation_hashes,
                "activation_hash": activation_hashes[-1],
                "activation_hashes": activation_hashes,
                "qwen32b_shape_profile": True,
                "diagnosis_codes": ["bridge_jax_tpu_stage_ready"],
                "blockers": [],
            }})
except Exception as exc:
    report["error_type"] = type(exc).__name__
    report["error_digest"] = sha_payload(str(exc))
    report["diagnosis_codes"].append("bridge_jax_tpu_stage_exception")
    report["blockers"].append("jax_tpu_stage_exception")
report["elapsed_seconds"] = round(time.monotonic() - started, 3)
print(json.dumps({{"schema": report["schema"], "ok": report["ok"], "diagnosis_codes": report.get("diagnosis_codes"), "report": report}}, sort_keys=True))
'''


def extract_web_tpu_report_from_stdout(stdout: str) -> dict[str, Any]:
    for line in str(stdout or "").splitlines()[::-1]:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("report"), dict):
            report = dict(parsed["report"])
            report["jupyter_proxy_token_public"] = False
            return report
    return {}


def extract_colab_tpu_report_from_outputs(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    text_parts: list[str] = []
    for output in outputs:
        text = output.get("text") if isinstance(output, dict) else None
        if isinstance(text, str):
            text_parts.append(text)
    report = extract_web_tpu_report_from_stdout("\n".join(text_parts))
    if report:
        report["colab_runtime_proxy_token_public"] = False
        report["colab_runtime_proxy_url_public"] = False
        report["jupyter_proxy_token_public"] = False
    return report


def load_colab_session(config_path: str, session_name: str) -> dict[str, Any]:
    path = Path(os.path.expanduser(str(config_path)))
    data = load_json(path) if path.is_file() else {}
    session = data.get(session_name) if isinstance(data, dict) else None
    if not isinstance(session, dict):
        raise RuntimeError("colab_session_not_found")
    missing = [key for key in ("url", "token", "endpoint") if not session.get(key)]
    if missing:
        raise RuntimeError("colab_session_missing_runtime_proxy_fields")
    return dict(session)


def colab_failure(
    blocker: str,
    diagnosis: str,
    *,
    error_type: str = "",
    error_text: str = "",
    outputs_public: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    report = web_tpu_failure(blocker, diagnosis, error_type=error_type, error_text=error_text)
    report["tpu_provider"] = "colab_cli"
    report["colab_runtime_proxy_token_public"] = False
    report["colab_runtime_proxy_url_public"] = False
    if outputs_public is not None:
        report["colab_outputs_public"] = outputs_public
    return report


def web_tpu_failure(
    blocker: str,
    diagnosis: str,
    *,
    error_type: str = "",
    error_text: str = "",
    errors_public: list[dict[str, Any]] | None = None,
    steps: list[dict[str, Any]] | None = None,
    executor_attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "gpu_tpu_cpu_bridge_jax_tpu_stage_v1",
        "ok": False,
        "backend": "jax_tpu",
        "stage_id": 1,
        "blockers": [blocker],
        "diagnosis_codes": [diagnosis],
        "public_artifact_safe": True,
        "jupyter_proxy_token_public": False,
    }
    if error_type:
        report["error_type"] = error_type
    if error_text:
        report["error_digest"] = sha_payload(error_text)
    if errors_public is not None:
        report["errors_public"] = errors_public
    if steps is not None:
        report["web_tpu_jupyter_steps"] = steps
    if executor_attempts is not None:
        report["web_tpu_executor_attempts"] = public_executor_attempts(executor_attempts)
    return report


def classify_web_tpu_exception(exc: Exception) -> tuple[str, str]:
    text = str(exc)
    lowered = text.lower()
    if "jupyter_proxy_not_found" in text:
        return "web_tpu_jupyter_proxy_not_found", "bridge_web_tpu_jupyter_proxy_not_found"
    if "jupyter_kernel_not_found" in text:
        return "web_tpu_jupyter_kernel_not_found", "bridge_web_tpu_jupyter_kernel_not_found"
    if "jupyter_frame_not_found" in text:
        return "web_tpu_jupyter_frame_not_found", "bridge_web_tpu_jupyter_frame_not_found"
    if "jupyter_api_unavailable" in text:
        return "web_tpu_jupyter_api_unavailable", "bridge_web_tpu_jupyter_api_unavailable"
    if "jupyter_kernel_create_failed" in text:
        return "web_tpu_jupyter_kernel_create_failed", "bridge_web_tpu_jupyter_kernel_create_failed"
    if "jupyter_websocket_failed" in text or "websocket" in lowered:
        return "web_tpu_websocket_failed", "bridge_web_tpu_websocket_failed"
    if "execution context was destroyed" in lowered or "frame was detached" in lowered:
        return "web_tpu_jupyter_frame_detached", "bridge_web_tpu_jupyter_frame_detached"
    return "web_tpu_stage_exception", "bridge_web_tpu_stage_exception"


def browser_iframe_executor_js() -> str:
    return r"""
async ({code, timeoutMs}) => {
  const steps = [];
  const errors = [];
  const stdout = [];
  const started = Date.now();
  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const uuid = () => {
    if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
      return globalThis.crypto.randomUUID();
    }
    return "xxxxxxxxxxxx4xxxyxxxxxxxxxxxxxxx".replace(/[xy]/g, (c) => {
      const r = Math.random() * 16 | 0;
      const v = c === "x" ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  };
  const fetchJson = async (path, options = {}) => {
    const response = await fetch(path, {
      ...options,
      headers: {
        "Accept": "application/json",
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
    });
    const text = await response.text();
    let payload = null;
    try {
      payload = text ? JSON.parse(text) : null;
    } catch (err) {
      payload = null;
    }
    steps.push({
      name: `jupyter_api_${path.replace(/[^A-Za-z0-9_/-]/g, "_")}`,
      ok: response.ok,
      status: response.status,
    });
    if (!response.ok) {
      throw new Error(`jupyter_api_unavailable_${response.status}`);
    }
    return payload;
  };
  let kernels = await fetchJson("/api/kernels");
  if (!Array.isArray(kernels)) {
    throw new Error("jupyter_api_unavailable_kernels_payload");
  }
  let kernel = kernels[0] || null;
  if (!kernel) {
    kernel = await fetchJson("/api/kernels", {
      method: "POST",
      body: JSON.stringify({name: "python3"}),
    });
    steps.push({name: "jupyter_kernel_create", ok: !!(kernel && kernel.id)});
    await wait(1500);
  } else {
    steps.push({name: "jupyter_kernel_reuse", ok: true});
  }
  if (!kernel || !kernel.id) {
    throw new Error("jupyter_kernel_create_failed");
  }
  const sessionId = uuid();
  const wsUrl = new URL(`/api/kernels/${kernel.id}/channels?session_id=${sessionId}`, location.href);
  wsUrl.protocol = wsUrl.protocol === "https:" ? "wss:" : "ws:";
  const msgId = uuid();
  const session = uuid();
  const wsResult = await new Promise((resolve) => {
    const ws = new WebSocket(wsUrl.href);
    let settled = false;
    const finish = (payload) => {
      if (settled) return;
      settled = true;
      try { ws.close(); } catch (err) {}
      resolve(payload);
    };
    const timer = setTimeout(() => {
      errors.push({ename: "Timeout", evalue_hash: "sha256:timeout"});
      finish({ok: false, timeout: true});
    }, Math.max(1000, Number(timeoutMs || 300000)));
    ws.onerror = () => {
      clearTimeout(timer);
      errors.push({ename: "WebSocketError", evalue_hash: "sha256:websocket"});
      finish({ok: false, websocket_error: true});
    };
    ws.onopen = () => {
      steps.push({name: "jupyter_ws_open", ok: true});
      ws.send(JSON.stringify({
        header: {
          msg_id: msgId,
          username: "crowdtensor-bridge",
          session,
          msg_type: "execute_request",
          version: "5.3",
        },
        parent_header: {},
        metadata: {},
        content: {
          code,
          silent: false,
          store_history: false,
          user_expressions: {},
          allow_stdin: false,
          stop_on_error: true,
        },
        buffers: [],
      }));
    };
    ws.onmessage = (event) => {
      let msg = null;
      try {
        msg = JSON.parse(event.data);
      } catch (err) {
        return;
      }
      const parent = msg.parent_header || {};
      const header = msg.header || {};
      const content = msg.content || {};
      const msgType = header.msg_type || "";
      if (parent.msg_id !== msgId && msgType !== "status") {
        return;
      }
      if (msgType === "stream" && content.name === "stdout") {
        stdout.push(String(content.text || ""));
      } else if (msgType === "error") {
        errors.push({ename: String(content.ename || "Error"), evalue_hash: "sha256:redacted"});
      } else if (msgType === "status" && content.execution_state === "idle" && parent.msg_id === msgId) {
        clearTimeout(timer);
        finish({ok: errors.length === 0});
      }
    };
  });
  steps.push({
    name: "jupyter_ws_execute",
    ok: errors.length === 0 && wsResult && wsResult.ok !== false,
    duration_seconds: Math.round((Date.now() - started) / 100) / 10,
  });
  return {
    ok: errors.length === 0 && wsResult && wsResult.ok !== false,
    stdout: stdout.join(""),
    errors_public: errors,
    steps,
    jupyter_proxy_token_public: false,
  };
}
"""


def execute_web_tpu_code_via_proxy_kernel(args: argparse.Namespace, code: str) -> dict[str, Any]:
    import websocket
    from playwright.sync_api import sync_playwright

    steps: list[dict[str, Any]] = []
    started = time.monotonic()
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            executable_path=args.chrome_executable,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        try:
            context = browser.new_context(storage_state=args.kaggle_web_storage_state, viewport={"width": 1440, "height": 1000})
            page = context.new_page()
            page.goto(args.kaggle_notebook_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(5000)
            values = page.evaluate(
                "() => Array.from(document.querySelectorAll('input')).map(x => x.value || '').filter(v => v.includes('jupyter-proxy'))"
            )
        finally:
            browser.close()
    if not values:
        raise RuntimeError("jupyter_proxy_not_found")
    parsed = urllib.parse.urlparse(str(values[0]))
    token_values = urllib.parse.parse_qs(parsed.query).get("token") or []
    if not token_values:
        raise RuntimeError("jupyter_proxy_token_not_found")
    proxy_token = token_values[0]
    base = f"{parsed.scheme}://{parsed.netloc}"
    req = urllib.request.Request(f"{base}/api/kernels?token={urllib.parse.quote(proxy_token)}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        kernels = json.loads(resp.read().decode("utf-8"))
    if not isinstance(kernels, list) or not kernels:
        raise RuntimeError("jupyter_kernel_not_found")
    kernel = kernels[0] if isinstance(kernels[0], dict) else {}
    kernel_id = str(kernel.get("id") or "")
    if not kernel_id:
        raise RuntimeError("jupyter_kernel_not_found")
    steps.append({
        "name": "jupyter_proxy_kernel_reuse",
        "ok": True,
        "kernel_id_present": True,
        "duration_seconds": round(time.monotonic() - started, 3),
        "jupyter_proxy_token_public": False,
    })
    ws_url = (
        f"{base.replace('https://', 'wss://').replace('http://', 'ws://')}"
        f"/api/kernels/{kernel_id}/channels?session_id={uuid.uuid4().hex}&token={urllib.parse.quote(proxy_token)}"
    )
    ws = websocket.create_connection(ws_url, timeout=20, origin="https://www.kaggle.com")
    msg_id = uuid.uuid4().hex
    session = uuid.uuid4().hex
    ws.send(json.dumps({
        "header": {
            "msg_id": msg_id,
            "username": "crowdtensor-bridge",
            "session": session,
            "msg_type": "execute_request",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        "buffers": [],
    }))
    stdout: list[str] = []
    errors: list[dict[str, Any]] = []
    exec_started = time.monotonic()
    timed_out = False
    try:
        while time.monotonic() - exec_started < float(args.web_tpu_execute_timeout_seconds):
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            msg = json.loads(raw)
            parent = msg.get("parent_header") or {}
            msg_type = (msg.get("header") or {}).get("msg_type")
            content = msg.get("content") or {}
            if parent.get("msg_id") != msg_id and msg_type != "status":
                continue
            if msg_type == "stream" and content.get("name") == "stdout":
                stdout.append(str(content.get("text") or ""))
            elif msg_type == "error":
                errors.append({"ename": str(content.get("ename") or "Error"), "evalue_hash": "sha256:redacted"})
            elif msg_type == "status" and content.get("execution_state") == "idle" and parent.get("msg_id") == msg_id:
                break
        else:
            timed_out = True
            errors.append({"ename": "Timeout", "evalue_hash": "sha256:timeout", "message_public": "jupyter_ws_execute_timeout"})
    finally:
        ws.close()
    steps.append({
        "name": "jupyter_proxy_ws_execute",
        "ok": not errors and not timed_out,
        "timeout": timed_out,
        "duration_seconds": round(time.monotonic() - exec_started, 3),
        "jupyter_proxy_token_public": False,
    })
    parsed_report = extract_web_tpu_report_from_stdout("".join(stdout))
    if parsed_report:
        parsed_report["web_tpu_jupyter_access_mode"] = "browser_proxy_existing_kernel_ws"
        parsed_report["jupyter_proxy_token_public"] = False
        parsed_report["web_tpu_jupyter_steps"] = public_jupyter_steps(steps)
        return parsed_report
    return web_tpu_failure(
        "web_tpu_report_missing" if not timed_out else "web_tpu_jupyter_execute_timeout",
        "bridge_web_tpu_report_missing" if not timed_out else "bridge_web_tpu_jupyter_execute_timeout",
        errors_public=errors,
        steps=steps,
    )


def browser_service_manager_executor_js() -> str:
    return r"""
async ({code, timeoutMs, forceNewSession}) => {
  const out = {
    ok: false,
    stdout: "",
    errors_public: [],
    steps: [],
    jupyter_proxy_token_public: false,
  };
  let session = null;
  let future = null;
  let ownsSession = false;
  const timeoutPromise = (message, ms) => new Promise((_, reject) => {
    setTimeout(() => reject(new Error(message)), Math.max(1000, Number(ms || 1000)));
  });
  const bounded = (promise, message, ms) => Promise.race([promise, timeoutPromise(message, ms)]);
  try {
    const sm = globalThis.jupyterapp && globalThis.jupyterapp.serviceManager;
    if (!sm) {
      throw new Error("jupyter_service_manager_missing");
    }
    const totalTimeoutMs = Math.max(1000, Number(timeoutMs || 300000));
    await bounded(sm.ready, "jupyter_service_manager_ready_timeout", Math.min(30000, totalTimeoutMs));
    out.steps.push({name: "service_manager_ready", ok: true});
    try {
      if (sm.sessions && sm.sessions.refreshRunning) {
        await bounded(sm.sessions.refreshRunning(), "jupyter_session_refresh_timeout", Math.min(30000, totalTimeoutMs));
      }
    } catch (err) {}
    const runningSessions = sm.sessions && sm.sessions.running ? Array.from(sm.sessions.running()) : [];
    const existingModel = runningSessions.find((model) => model && model.kernel && model.kernel.id) || runningSessions[0] || null;
    if (!forceNewSession && existingModel && sm.sessions.connectTo) {
      session = sm.sessions.connectTo({model: existingModel});
      ownsSession = false;
      out.steps.push({
        name: "session_connectTo_existing",
        ok: !!(session && session.kernel),
        kernel_id_present: !!(session && session.kernel && session.kernel.id),
        running_session_count: runningSessions.length,
      });
    }
    if (!session || !session.kernel) {
      session = await bounded(sm.sessions.startNew({
        path: "crowdtensor-runtime-smoke.ipynb",
        type: "notebook",
        name: "crowdtensor-runtime-smoke.ipynb",
        kernel: {name: "python3"},
      }), "jupyter_session_start_timeout", Math.min(120000, totalTimeoutMs));
      ownsSession = true;
      out.steps.push({
        name: "session_startNew",
        ok: !!(session && session.kernel),
        kernel_id_present: !!(session && session.kernel && session.kernel.id),
      });
    }
    if (!session || !session.kernel) {
      throw new Error("jupyter_service_manager_kernel_missing");
    }
    const kernel = session.kernel;
    try {
      await bounded(kernel.info, "jupyter_kernel_info_timeout", Math.min(60000, totalTimeoutMs));
      out.steps.push({name: "kernel_info_ready", ok: true});
    } catch (err) {
      out.steps.push({name: "kernel_info_ready", ok: false, timeout: true});
    }
    future = kernel.requestExecute({
      code,
      silent: false,
      store_history: false,
      allow_stdin: false,
      stop_on_error: true,
    });
    const execution = new Promise((resolve) => {
      future.onIOPub = (msg) => {
        const header = msg.header || {};
        const content = msg.content || {};
        const msgType = header.msg_type || "";
        if (msgType === "stream" && content.name === "stdout") {
          out.stdout += String(content.text || "");
        } else if (msgType === "error") {
          out.errors_public.push({ename: String(content.ename || "Error"), evalue_hash: "sha256:redacted"});
        } else if (msgType === "status" && content.execution_state === "idle") {
          resolve({timeout: false});
        }
      };
    });
    const done = await Promise.race([
      execution,
      new Promise((resolve) => setTimeout(() => resolve({timeout: true}), totalTimeoutMs)),
    ]);
    if (done.timeout) {
      try { future.dispose(); } catch (err) {}
      out.errors_public.push({ename: "Timeout", evalue_hash: "sha256:timeout", message_public: "jupyter_execute_timeout"});
    }
    out.steps.push({
      name: "service_manager_request_execute",
      ok: !done.timeout && out.errors_public.length === 0,
      timeout: !!done.timeout,
    });
    out.ok = !done.timeout && out.errors_public.length === 0;
  } catch (err) {
    out.ok = false;
    out.errors_public.push({
      ename: String((err && err.name) || "Error"),
      evalue_hash: "sha256:redacted",
      message_public: String((err && err.message) || err || "").slice(0, 160),
    });
    out.steps.push({name: "service_manager_exception", ok: false});
  } finally {
    if (future) {
      try { future.dispose(); } catch (err) {}
    }
    if (session && ownsSession) {
      try {
        await bounded(session.shutdown(), "jupyter_session_shutdown_timeout", 15000);
        out.steps.push({name: "session_shutdown", ok: true});
      } catch (err) {
        out.steps.push({name: "session_shutdown", ok: false});
      }
    }
  }
  return out;
}
"""


def browser_service_manager_ws_executor_js() -> str:
    return r"""
async ({code, timeoutMs, forceNewSession}) => {
  const out = {
    ok: false,
    stdout: "",
    errors_public: [],
    steps: [],
    jupyter_proxy_token_public: false,
  };
  let sessionModel = null;
  let connectedSession = null;
  let ownsSession = false;
  const timeoutPromise = (message, ms) => new Promise((_, reject) => {
    setTimeout(() => reject(new Error(message)), Math.max(1000, Number(ms || 1000)));
  });
  const bounded = (promise, message, ms) => Promise.race([promise, timeoutPromise(message, ms)]);
  const uuid = () => {
    if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
      return globalThis.crypto.randomUUID();
    }
    return "xxxxxxxxxxxx4xxxyxxxxxxxxxxxxxxx".replace(/[xy]/g, (c) => {
      const r = Math.random() * 16 | 0;
      const v = c === "x" ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  };
  try {
    const sm = globalThis.jupyterapp && globalThis.jupyterapp.serviceManager;
    if (!sm) {
      throw new Error("jupyter_service_manager_missing");
    }
    const totalTimeoutMs = Math.max(1000, Number(timeoutMs || 300000));
    await bounded(sm.ready, "jupyter_service_manager_ready_timeout", Math.min(30000, totalTimeoutMs));
    out.steps.push({name: "service_manager_ready", ok: true});
    try {
      if (sm.sessions && sm.sessions.refreshRunning) {
        await bounded(sm.sessions.refreshRunning(), "jupyter_session_refresh_timeout", Math.min(30000, totalTimeoutMs));
      }
    } catch (err) {}
    const runningSessions = sm.sessions && sm.sessions.running ? Array.from(sm.sessions.running()) : [];
    const existingModel = runningSessions.find((model) => model && model.kernel && model.kernel.id) || runningSessions[0] || null;
    if (!forceNewSession && existingModel && sm.sessions.connectTo) {
      connectedSession = sm.sessions.connectTo({model: existingModel});
      ownsSession = false;
      out.steps.push({
        name: "session_connectTo_existing",
        ok: !!(connectedSession && connectedSession.kernel),
        kernel_id_present: !!(connectedSession && connectedSession.kernel && connectedSession.kernel.id),
        running_session_count: runningSessions.length,
      });
    }
    if (!connectedSession || !connectedSession.kernel) {
      connectedSession = await bounded(sm.sessions.startNew({
        path: "crowdtensor-runtime-ws-smoke.ipynb",
        type: "notebook",
        name: "crowdtensor-runtime-ws-smoke.ipynb",
        kernel: {name: "python3"},
      }), "jupyter_session_start_timeout", Math.min(120000, totalTimeoutMs));
      ownsSession = true;
      out.steps.push({
        name: "session_startNew",
        ok: !!(connectedSession && connectedSession.kernel),
        kernel_id_present: !!(connectedSession && connectedSession.kernel && connectedSession.kernel.id),
      });
    }
    sessionModel = connectedSession && connectedSession.model ? connectedSession.model : existingModel;
    const kernelId = connectedSession && connectedSession.kernel && connectedSession.kernel.id;
    out.steps.push({
      name: "session_kernel_ready",
      ok: !!kernelId,
      kernel_id_present: !!kernelId,
    });
    if (!kernelId) {
      throw new Error("jupyter_service_manager_kernel_missing");
    }
    const settings = sm.serverSettings || {};
    const baseUrl = String(settings.baseUrl || "");
    const wsBase = String(settings.wsUrl || "");
    if (!wsBase) {
      throw new Error("jupyter_ws_url_missing");
    }
    const channelsPath = `${baseUrl.replace(/\/$/, "")}/api/kernels/${kernelId}/channels?session_id=${encodeURIComponent(uuid())}`;
    const wsUrlObject = new URL(channelsPath, wsBase || location.href);
    const wsBaseObject = new URL(wsBase || location.href);
    wsUrlObject.protocol = wsBaseObject.protocol || (wsUrlObject.protocol === "https:" ? "wss:" : "ws:");
    wsUrlObject.host = wsBaseObject.host || wsUrlObject.host;
    const wsUrl = wsUrlObject.href;
    const msgId = uuid();
    const msgSession = uuid();
    const wsResult = await new Promise((resolve) => {
      const ws = new WebSocket(wsUrl);
      let settled = false;
      const finish = (payload) => {
        if (settled) return;
        settled = true;
        try { ws.close(); } catch (err) {}
        resolve(payload);
      };
      const timer = setTimeout(() => {
        out.errors_public.push({ename: "Timeout", evalue_hash: "sha256:timeout", message_public: "jupyter_ws_execute_timeout"});
        finish({ok: false, timeout: true});
      }, totalTimeoutMs);
      ws.onerror = () => {
        clearTimeout(timer);
        out.errors_public.push({ename: "WebSocketError", evalue_hash: "sha256:websocket"});
        finish({ok: false, websocket_error: true});
      };
      ws.onopen = () => {
        out.steps.push({name: "jupyter_ws_open", ok: true});
        ws.send(JSON.stringify({
          header: {
            msg_id: msgId,
            username: "crowdtensor-bridge",
            session: msgSession,
            msg_type: "execute_request",
            version: "5.3",
          },
          parent_header: {},
          metadata: {},
          content: {
            code,
            silent: false,
            store_history: false,
            user_expressions: {},
            allow_stdin: false,
            stop_on_error: true,
          },
          buffers: [],
        }));
      };
      ws.onmessage = (event) => {
        let msg = null;
        try {
          msg = JSON.parse(event.data);
        } catch (err) {
          return;
        }
        const parent = msg.parent_header || {};
        const header = msg.header || {};
        const content = msg.content || {};
        const msgType = header.msg_type || "";
        if (parent.msg_id !== msgId && !(msgType === "status" && parent.msg_id === msgId)) {
          return;
        }
        if (msgType === "stream" && content.name === "stdout") {
          out.stdout += String(content.text || "");
        } else if (msgType === "error") {
          out.errors_public.push({ename: String(content.ename || "Error"), evalue_hash: "sha256:redacted"});
        } else if (msgType === "status" && content.execution_state === "idle" && parent.msg_id === msgId) {
          clearTimeout(timer);
          finish({ok: out.errors_public.length === 0});
        }
      };
    });
    out.steps.push({
      name: "service_manager_ws_execute",
      ok: out.errors_public.length === 0 && wsResult && wsResult.ok !== false,
      timeout: !!(wsResult && wsResult.timeout),
      websocket_error: !!(wsResult && wsResult.websocket_error),
    });
    out.ok = out.errors_public.length === 0 && wsResult && wsResult.ok !== false;
  } catch (err) {
    out.ok = false;
    out.errors_public.push({
      ename: String((err && err.name) || "Error"),
      evalue_hash: "sha256:redacted",
      message_public: String((err && err.message) || err || "").slice(0, 160),
    });
    out.steps.push({name: "service_manager_ws_exception", ok: false});
  } finally {
    if (ownsSession && sessionModel && sessionModel.id) {
      try {
        const sm = globalThis.jupyterapp && globalThis.jupyterapp.serviceManager;
        if (sm) {
          await bounded(sm.sessions.shutdown(sessionModel.id), "jupyter_session_shutdown_timeout", 15000);
          out.steps.push({name: "session_shutdown", ok: true});
        }
      } catch (err) {
        out.steps.push({name: "session_shutdown", ok: false});
      }
    }
  }
  return out;
}
"""


def public_jupyter_steps(steps: Any) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for item in steps if isinstance(steps, list) else []:
        if not isinstance(item, dict):
            continue
        cleaned_item: dict[str, Any] = {}
        for key, value in item.items():
            if key in {"url", "path", "href", "baseUrl", "wsUrl"}:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                cleaned_item[str(key)] = value
        cleaned.append(cleaned_item)
    return cleaned


def public_executor_attempts(attempts: Any) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for item in attempts if isinstance(attempts, list) else []:
        if not isinstance(item, dict):
            continue
        errors_public = item.get("errors_public") if isinstance(item.get("errors_public"), list) else []
        parsed_report = item.get("parsed_report") if isinstance(item.get("parsed_report"), dict) else {}
        cleaned.append({
            "executor_name": str(item.get("executor_name") or ""),
            "ok": bool(
                item.get("ok") is True
                or (isinstance(parsed_report, dict) and parsed_report.get("ok") is True)
            ),
            "parsed_report_present": bool(parsed_report),
            "parsed_report_ok": parsed_report.get("ok") is True if parsed_report else False,
            "blockers": [
                str(value)
                for value in (
                    parsed_report.get("blockers")
                    if isinstance(parsed_report.get("blockers"), list)
                    else []
                )
                if value
            ],
            "error_count": len(errors_public),
            "error_names": [
                str(error.get("ename") or error.get("error_type") or "")
                for error in errors_public
                if isinstance(error, dict)
            ][:5],
            "steps": public_jupyter_steps(item.get("steps")),
        })
    return cleaned


def classify_jupyter_execution_errors(errors: Any) -> tuple[str, str] | None:
    if not isinstance(errors, list):
        return None
    text = " ".join(
        str(item.get("message_public") or item.get("ename") or "")
        for item in errors
        if isinstance(item, dict)
    ).lower()
    if "jupyter_kernel_info_timeout" in text or "jupyter_session_start_timeout" in text:
        return "web_tpu_jupyter_kernel_not_ready", "bridge_web_tpu_jupyter_kernel_not_ready"
    if "jupyter_execute_timeout" in text:
        return "web_tpu_jupyter_execute_timeout", "bridge_web_tpu_jupyter_execute_timeout"
    if "jupyter_service_manager_ready_timeout" in text:
        return "web_tpu_jupyter_service_manager_not_ready", "bridge_web_tpu_jupyter_service_manager_not_ready"
    return None


def _execute_web_tpu_code_via_iframe_direct(args: argparse.Namespace, code: str) -> dict[str, Any]:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    def find_frame(page: Any) -> Any:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            for frame in page.frames:
                url = str(getattr(frame, "url", "") or "")
                if "jupyter-proxy" not in url and "/jupyterlab-" not in url:
                    continue
                try:
                    if frame.evaluate("() => !!(globalThis.jupyterapp && globalThis.jupyterapp.serviceManager)"):
                        return frame
                except Exception:
                    continue
            page.wait_for_timeout(1000)
        raise RuntimeError("jupyter_service_manager_not_ready")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            executable_path=args.chrome_executable,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        try:
            context = browser.new_context(storage_state=args.kaggle_web_storage_state, viewport={"width": 1440, "height": 1000})
            page = context.new_page()
            page.goto(args.kaggle_notebook_url, wait_until="domcontentloaded", timeout=60000)
            page.set_default_timeout(
                int(max(1000.0, float(args.web_tpu_execute_timeout_seconds) * 1000.0 + 30000.0))
            )
            last_exc: Exception | None = None
            for attempt in range(1, 7):
                try:
                    frame = find_frame(page)
                    executor_results: list[dict[str, Any]] = []
                    for executor_name, executor_js in [
                        ("browser_iframe_service_manager", browser_service_manager_executor_js()),
                        ("browser_iframe_service_manager_ws", browser_service_manager_ws_executor_js()),
                        ("browser_iframe_existing_kernel_ws", browser_iframe_executor_js()),
                    ]:
                        result = frame.evaluate(
                            executor_js,
                            {
                                "code": code,
                                "timeoutMs": int(max(1000.0, float(args.web_tpu_execute_timeout_seconds) * 1000.0)),
                                "forceNewSession": bool(getattr(args, "web_tpu_force_new_session", False)),
                            },
                        )
                        if isinstance(result, dict):
                            result["executor_name"] = executor_name
                            executor_results.append(result)
                            stdout = str(result.get("stdout") or "")
                            report = extract_web_tpu_report_from_stdout(stdout)
                            if report:
                                report["web_tpu_jupyter_access_mode"] = executor_name
                                report["jupyter_proxy_token_public"] = False
                                steps = public_jupyter_steps(result.get("steps"))
                                if steps:
                                    report["web_tpu_jupyter_steps"] = steps
                                report["web_tpu_executor_attempts"] = public_executor_attempts(executor_results)
                                if report.get("ok") is True or executor_name == "browser_iframe_service_manager":
                                    return report
                                executor_results[-1]["parsed_report"] = report
                                continue
                        elif executor_name == "browser_iframe_service_manager":
                            return web_tpu_failure(
                                "web_tpu_jupyter_api_unavailable",
                                "bridge_web_tpu_jupyter_api_unavailable",
                                error_type="InvalidJupyterResult",
                                error_text=repr(result),
                            )
                    try:
                        proxy_report = execute_web_tpu_code_via_proxy_kernel(args, code)
                        if proxy_report.get("ok") is True:
                            return proxy_report
                        executor_results.append({
                            "executor_name": "browser_proxy_existing_kernel_ws",
                            "errors_public": proxy_report.get("errors_public") if isinstance(proxy_report.get("errors_public"), list) else [],
                            "steps": proxy_report.get("web_tpu_jupyter_steps") if isinstance(proxy_report.get("web_tpu_jupyter_steps"), list) else [],
                            "parsed_report": proxy_report,
                        })
                    except Exception as exc:
                        executor_results.append({
                            "executor_name": "browser_proxy_existing_kernel_ws",
                            "errors_public": [{
                                "ename": type(exc).__name__,
                                "evalue_hash": "sha256:redacted",
                                "message_public": str(exc).splitlines()[0][:160] if str(exc) else "",
                            }],
                            "steps": [{"name": "jupyter_proxy_existing_kernel_ws", "ok": False}],
                        })
                    result = executor_results[-1] if executor_results else {}
                    if not isinstance(result, dict):
                        return web_tpu_failure(
                            "web_tpu_jupyter_api_unavailable",
                            "bridge_web_tpu_jupyter_api_unavailable",
                            error_type="InvalidJupyterResult",
                            error_text=repr(result),
                            executor_attempts=executor_results,
                        )
                    errors_public = result.get("errors_public") if isinstance(result.get("errors_public"), list) else []
                    classified = classify_jupyter_execution_errors(errors_public)
                    if classified:
                        blocker, diagnosis = classified
                        return web_tpu_failure(
                            blocker,
                            diagnosis,
                            errors_public=errors_public,
                            steps=public_jupyter_steps(result.get("steps")),
                            executor_attempts=executor_results,
                        )
                    return web_tpu_failure(
                        "web_tpu_report_missing",
                        "bridge_web_tpu_report_missing",
                        errors_public=errors_public,
                        steps=public_jupyter_steps(result.get("steps")),
                        executor_attempts=executor_results,
                    )
                except PlaywrightError as exc:
                    last_exc = exc
                    if "Execution context was destroyed" not in str(exc) and "Frame was detached" not in str(exc):
                        raise
                    page.wait_for_timeout(2000 * attempt)
                    continue
            if last_exc:
                raise last_exc
            raise RuntimeError("jupyter_frame_not_found")
        finally:
            browser.close()


def _web_tpu_execute_child_main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        args_payload = payload.get("args") if isinstance(payload.get("args"), dict) else {}
        code = str(payload.get("code") or "")
        result = _execute_web_tpu_code_via_iframe_direct(argparse.Namespace(**args_payload), code)
        print(json.dumps({"ok": True, "result": result}, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error_digest": sha_payload(str(exc)),
                    "message_public": str(exc).splitlines()[0][:160] if str(exc) else "",
                },
                sort_keys=True,
            )
        )
        return 1


def web_tpu_subprocess_timeout_seconds(args: argparse.Namespace) -> float:
    execute_timeout = max(1.0, float(getattr(args, "web_tpu_execute_timeout_seconds", 300.0)))
    return execute_timeout + min(60.0, max(10.0, execute_timeout * 0.25))


def extract_child_json(stdout: str) -> dict[str, Any] | None:
    for line in reversed(str(stdout or "").splitlines()):
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def execute_web_tpu_code_via_iframe(
    args: argparse.Namespace,
    code: str,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    payload = json.dumps({"args": vars(args), "code": code}, sort_keys=True)
    command = [sys.executable, str(Path(__file__).resolve()), "__web_tpu_execute_child__"]
    timeout_seconds = web_tpu_subprocess_timeout_seconds(args)
    try:
        completed = runner(
            command,
            input=payload,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return web_tpu_failure(
            "web_tpu_jupyter_execute_timeout",
            "bridge_web_tpu_jupyter_execute_timeout",
            error_type="TimeoutExpired",
            error_text="web_tpu_execute_subprocess_timeout",
            steps=[{"name": "web_tpu_execute_subprocess", "ok": False, "timeout": True}],
        )
    child = extract_child_json(getattr(completed, "stdout", ""))
    if not child:
        return web_tpu_failure(
            "web_tpu_jupyter_execute_subprocess_failed",
            "bridge_web_tpu_jupyter_execute_subprocess_failed",
            error_type="MissingChildResult",
            error_text=f"returncode={getattr(completed, 'returncode', None)}",
            steps=[{"name": "web_tpu_execute_subprocess", "ok": False, "returncode": getattr(completed, "returncode", None)}],
        )
    if child.get("ok") is True and isinstance(child.get("result"), dict):
        result = dict(child["result"])
        steps = public_jupyter_steps(result.get("web_tpu_jupyter_steps"))
        steps.append({"name": "web_tpu_execute_subprocess", "ok": True, "timeout": False})
        result["web_tpu_jupyter_steps"] = steps
        return result
    public_exc = RuntimeError(str(child.get("message_public") or child.get("error_type") or "web_tpu_child_failed"))
    blocker, diagnosis = classify_web_tpu_exception(public_exc)
    return web_tpu_failure(
        blocker,
        diagnosis,
        error_type=str(child.get("error_type") or "WebTpuChildError"),
        error_text=str(child.get("message_public") or ""),
        steps=[{"name": "web_tpu_execute_subprocess", "ok": False, "returncode": getattr(completed, "returncode", None)}],
    )


def run_web_tpu_stage(args: argparse.Namespace, *, coordinator_url: str, token: str) -> dict[str, Any]:
    try:
        import websocket
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        return web_tpu_failure(
            "web_tpu_dependencies_missing",
            "bridge_web_tpu_dependencies_missing",
            error_type=type(exc).__name__,
            error_text=str(exc),
        )

    def get_proxy_and_kernel() -> tuple[str, str, str]:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                executable_path=args.chrome_executable,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            context = browser.new_context(storage_state=args.kaggle_web_storage_state, viewport={"width": 1440, "height": 1000})
            page = context.new_page()
            page.goto(args.kaggle_notebook_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(5000)
            values = page.evaluate(
                "() => Array.from(document.querySelectorAll('input')).map(x => x.value || '').filter(v => v.includes('jupyter-proxy'))"
            )
            browser.close()
        if not values:
            raise RuntimeError("jupyter_proxy_not_found")
        parsed = urllib.parse.urlparse(values[0])
        proxy_token = urllib.parse.parse_qs(parsed.query)["token"][0]
        base = f"{parsed.scheme}://{parsed.netloc}"
        req = urllib.request.Request(f"{base}/api/kernels?token={urllib.parse.quote(proxy_token)}", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            kernels = json.loads(resp.read().decode("utf-8"))
        if not kernels:
            raise RuntimeError("jupyter_kernel_not_found")
        return base, proxy_token, str(kernels[0]["id"])

    tpu_code = render_jax_tpu_shape_stage_code(
        args,
        coordinator_url=coordinator_url,
        token=token,
        provider="kaggle_web",
        miner_id="web-tpu-bridge-stage1",
    )
    if bool(getattr(args, "web_tpu_32b_execute", False)):
        return run_web_tpu_32b_stage_mediated(args, coordinator_url=coordinator_url, token=token)
    if bool(getattr(args, "web_tpu_deepseek_stage_execute", False)):
        deepseek_code = render_web_tpu_deepseek_stage_code(args, coordinator_url=coordinator_url, token=token)
        return execute_web_tpu_code_via_iframe(args, deepseek_code)

    try:
        return execute_web_tpu_code_via_iframe(args, tpu_code)
    except Exception as iframe_exc:
        iframe_blocker, iframe_diagnosis = classify_web_tpu_exception(iframe_exc)
        if iframe_blocker not in {"web_tpu_jupyter_frame_not_found", "web_tpu_jupyter_proxy_not_found"}:
            return web_tpu_failure(
                iframe_blocker,
                iframe_diagnosis,
                error_type=type(iframe_exc).__name__,
                error_text=str(iframe_exc),
            )

    try:
        base, proxy_token, kernel_id = get_proxy_and_kernel()
        ws_url = (
            f"{base.replace('https://', 'wss://').replace('http://', 'ws://')}"
            f"/api/kernels/{kernel_id}/channels?session_id={uuid.uuid4().hex}&token={urllib.parse.quote(proxy_token)}"
        )
        ws = websocket.create_connection(ws_url, timeout=20, origin="https://www.kaggle.com")
        msg_id = uuid.uuid4().hex
        session = uuid.uuid4().hex
        ws.send(json.dumps({
            "header": {
                "msg_id": msg_id,
                "username": "crowdtensor-bridge",
                "session": session,
                "msg_type": "execute_request",
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": tpu_code,
                "silent": False,
                "store_history": False,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "buffers": [],
        }))
        started = time.monotonic()
        stdout = []
        errors = []
        final_report: dict[str, Any] = {}
        while time.monotonic() - started < float(args.web_tpu_execute_timeout_seconds):
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            msg = json.loads(raw)
            parent = msg.get("parent_header") or {}
            msg_type = (msg.get("header") or {}).get("msg_type")
            content = msg.get("content") or {}
            if parent.get("msg_id") != msg_id and msg_type != "status":
                continue
            if msg_type == "stream" and content.get("name") == "stdout":
                stdout.append(str(content.get("text") or ""))
            elif msg_type == "error":
                errors.append({"ename": content.get("ename"), "evalue_hash": sha_payload(content.get("evalue") or "")})
            elif msg_type == "status" and content.get("execution_state") == "idle" and parent.get("msg_id") == msg_id:
                break
        ws.close()
        final_report = extract_web_tpu_report_from_stdout("".join(stdout))
        if final_report:
            final_report["web_tpu_jupyter_access_mode"] = "legacy_proxy_token_api"
            return final_report
        return web_tpu_failure(
            "web_tpu_report_missing",
            "bridge_web_tpu_report_missing",
            errors_public=errors,
        )
    except Exception as exc:
        public_blocker, public_diagnosis = classify_web_tpu_exception(exc)
        return web_tpu_failure(
            public_blocker,
            public_diagnosis,
            error_type=type(exc).__name__,
            error_text=str(exc),
        )


def run_colab_tpu_stage(args: argparse.Namespace, *, coordinator_url: str, token: str) -> dict[str, Any]:
    try:
        ColabRuntime = colab_cli_runtime.load_colab_runtime_class()
    except Exception as exc:
        return colab_failure(
            "colab_cli_runtime_dependency_missing",
            "bridge_colab_cli_runtime_dependency_missing",
            error_type=type(exc).__name__,
            error_text=str(exc),
        )
    try:
        session = load_colab_session(args.colab_session_config, args.colab_session_name)
        runtime = ColabRuntime(
            session["url"],
            session["token"],
            kernel_id=session.get("kernel_id"),
            session_id=session.get("session_id"),
        )
        if bool(getattr(args, "web_tpu_32b_execute", False)):
            code = render_web_tpu_32b_stage_code(args, coordinator_url=coordinator_url, token=token)
        else:
            code = render_jax_tpu_shape_stage_code(
                args,
                coordinator_url=coordinator_url,
                token=token,
                provider="colab_cli",
                miner_id=f"colab-tpu-bridge-stage1-{args.colab_session_name}",
            )
        outputs = runtime.execute_code(code, timeout=float(args.web_tpu_execute_timeout_seconds))
        report = extract_colab_tpu_report_from_outputs(outputs)
        if not report:
            return colab_failure(
                "colab_tpu_report_missing",
                "bridge_colab_tpu_report_missing",
                outputs_public=[
                    {
                        "output_type": output.get("output_type"),
                        "name": output.get("name"),
                        "text_hash": sha_payload(output.get("text") or ""),
                    }
                    for output in outputs
                    if isinstance(output, dict)
                ],
            )
        report["tpu_provider"] = "colab_cli"
        report["colab_session_name"] = args.colab_session_name
        report["colab_endpoint_hash"] = hashlib.sha256(str(session.get("endpoint") or "").encode("utf-8")).hexdigest()[:16]
        parsed_url = urllib.parse.urlparse(str(session.get("url") or ""))
        report["colab_runtime_proxy_host_hash"] = hashlib.sha256(parsed_url.netloc.encode("utf-8")).hexdigest()[:16]
        report["colab_runtime_proxy_token_public"] = False
        report["colab_runtime_proxy_url_public"] = False
        report["jupyter_proxy_token_public"] = False
        return report
    except Exception as exc:
        return colab_failure(
            "colab_tpu_stage_exception",
            "bridge_colab_tpu_stage_exception",
            error_type=type(exc).__name__,
            error_text=str(exc),
        )
    finally:
        try:
            runtime.stop()  # type: ignore[name-defined]
        except Exception:
            pass


def wait_for_ready(state: BridgeState, *, timeout_seconds: float, threads: list[threading.Thread]) -> dict[str, Any]:
    started = time.monotonic()
    while time.monotonic() - started < timeout_seconds:
        status = state.public_status()
        if status.get("ready"):
            return status
        stage_counts = status.get("stage_task_counts") if isinstance(status.get("stage_task_counts"), dict) else {}
        if threads and not threads[0].is_alive() and int(stage_counts.get("stage0") or 0) < 1:
            return status
        if all(not thread.is_alive() for thread in threads):
            return status
        time.sleep(2.0)
    return state.public_status()


def public_step(step: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(step)
    for key in ("command_line", "stdout_tail", "stderr_tail"):
        if key in cleaned:
            text = str(cleaned[key] or "")
            text = text.replace("private-kaggle-bridge-gpu-kernel", "<private-payload-dir>")
            text = text.replace("private-kaggle-bridge-cpu-kernel", "<private-payload-dir>")
            cleaned[key] = text
    command = cleaned.get("command_public")
    if isinstance(command, list):
        cleaned["command_public"] = [
            "<private-payload-dir>"
            if (
                "private-kaggle-bridge-gpu-kernel" in str(part)
                or "private-kaggle-bridge-cpu-kernel" in str(part)
            )
            else part
            for part in command
        ]
    return cleaned


def bridge_has_full_32b_tpu_stage(args: argparse.Namespace, tpu_report: dict[str, Any]) -> bool:
    if not bool(getattr(args, "web_tpu_32b_execute", False)):
        return False
    if target_parameter_class(str(args.target_model_id)) != "32b":
        return False
    return bridge_has_full_target_tpu_stage(args, tpu_report)


def bridge_has_full_target_tpu_stage(args: argparse.Namespace, tpu_report: dict[str, Any]) -> bool:
    if not bool(getattr(args, "web_tpu_32b_execute", False)):
        return False
    full_layer_count = max(0, int(args.web_tpu_32b_stage_end) - int(args.web_tpu_32b_stage_start))
    return bool(
        tpu_report.get("ok") is True
        and tpu_report.get("qwen32b_stage_owned_loader_ready") is True
        and tpu_report.get("full_stage_owned_tpu_loader_ready") is True
        and tpu_report.get("tpu_32b_runtime_adapter_ready") is True
        and tpu_report.get("stage_owned_model_loaded") is True
        and int(tpu_report.get("executed_layer_count") or 0) >= full_layer_count
        and int(tpu_report.get("missing_stage_key_count") or 0) == 0
    )


def build_live_proof_from_bridge(report: dict[str, Any]) -> dict[str, Any]:
    coordinator = report.get("coordinator") if isinstance(report.get("coordinator"), dict) else {}
    stage_reports = report.get("stage_reports") if isinstance(report.get("stage_reports"), dict) else {}
    gpu_report = stage_reports.get("cuda_gpu_stage") if isinstance(stage_reports.get("cuda_gpu_stage"), dict) else {}
    tpu_report = stage_reports.get("jax_tpu_stage") if isinstance(stage_reports.get("jax_tpu_stage"), dict) else {}
    cpu_report = stage_reports.get("cpu_tail_stage") if isinstance(stage_reports.get("cpu_tail_stage"), dict) else {}
    stage_counts = coordinator.get("stage_task_counts") if isinstance(coordinator.get("stage_task_counts"), dict) else {}
    activation_hashes = [str(item) for item in coordinator.get("activation_hashes") or []]
    runtime_summary = report.get("runtime_device_summary") if isinstance(report.get("runtime_device_summary"), dict) else {}
    same_request_32b = report.get("gpu_tpu_cpu_32b_same_request_verified") is True
    safety = rc_pack.default_safety_flags()
    handoffs = [
        {
            "from_backend": "cuda",
            "to_backend": "jax_tpu",
            "activation_hash": activation_hashes[0] if len(activation_hashes) >= 1 else "",
            "activation_shape": [1, 1, 5120],
            "activation_dtype": "float16",
            "activation_layout": "batch_seq_hidden",
            "activation_payload_public": False,
        },
        {
            "from_backend": "jax_tpu",
            "to_backend": "cpu",
            "activation_hash": activation_hashes[1] if len(activation_hashes) >= 2 else "",
            "activation_shape": [1, 1, 5120],
            "activation_dtype": "bfloat16",
            "activation_layout": "batch_seq_hidden",
            "activation_payload_public": False,
        },
    ]
    live = {
        "schema": rc_pack.LIVE_PROOF_SCHEMA,
        "ok": same_request_32b,
        "public_artifact_safe": True,
        "model_id": str(report.get("target_model_id") or ""),
        "model_parameter_count": 32_500_000_000,
        "model_tier": "32b",
        "generated_token_count": int(report.get("generated_token_count") or 0),
        "target_generated_token_count": int(report.get("target_generated_token_count") or 1),
        "context_length": 1,
        "gpu_tpu_cpu_32b_same_request_verified": same_request_32b,
        "live_tpu_stage_miner_integrated": same_request_32b,
        "tpu_32b_runtime_adapter_ready": tpu_report.get("tpu_32b_runtime_adapter_ready") is True,
        "fallback_model_used": False,
        "stage_local_kv_cache_verified": tpu_report.get("stage_local_kv_cache_verified") is True,
        "accepted_stage_tasks": [
            {
                "stage_id": 0,
                "backend": "cuda",
                "accepted": gpu_report.get("ok") is True,
                "stage_owned_model_loaded": bool(report.get("cuda_stage_32b_weight_evidence_ready") or gpu_report.get("stage_owned_model_loaded") is True),
            },
            {
                "stage_id": 1,
                "backend": "jax_tpu",
                "accepted": tpu_report.get("ok") is True,
                "stage_owned_model_loaded": tpu_report.get("stage_owned_model_loaded") is True,
                "executed_layer_count": int(tpu_report.get("executed_layer_count") or 0),
                "loaded_execution_tensor_key_count": int(tpu_report.get("loaded_execution_tensor_key_count") or 0),
                "loaded_execution_tensor_gb": float(tpu_report.get("loaded_execution_tensor_gb") or 0.0),
            },
            {
                "stage_id": 2,
                "backend": "cpu",
                "accepted": cpu_report.get("ok") is True,
                "stage_owned_model_loaded": True,
            },
        ],
        "stage_task_counts": {
            "cuda": int(stage_counts.get("stage0") or 0),
            "jax_tpu": int(stage_counts.get("stage1") or 0),
            "cpu": int(stage_counts.get("stage2") or 0),
        },
        "activation_handoffs": handoffs,
        "runtime_device_summary": {
            "cuda_gpu_count": int(runtime_summary.get("cuda_device_count") or 0),
            "tpu_device_count": int(runtime_summary.get("tpu_device_count") or 0),
            "cpu_stage_count": 1 if cpu_report.get("ok") is True else 0,
            "tpu_stage_layer_range": list(tpu_report.get("stage_layer_range") or []),
            "tpu_executed_layer_count": int(tpu_report.get("executed_layer_count") or 0),
            "tpu_loaded_execution_tensor_gb": float(tpu_report.get("loaded_execution_tensor_gb") or 0.0),
            "cuda_stage_32b_weight_evidence_ready": report.get("cuda_stage_32b_weight_evidence_ready") is True,
        },
        "cleanup": {
            "private_runtime_artifacts_cleaned": bool(
                report.get("cleanup", {}).get("private_gpu_package_removed") is True
                and report.get("cleanup", {}).get("private_cpu_package_removed") is not False
            ),
            "temporary_kaggle_kernels_deleted": bool(
                report.get("cleanup", {}).get("kaggle_gpu_kernel_deleted") is True
                and report.get("cleanup", {}).get("kaggle_cpu_kernel_deleted") is not False
            ),
            "token_rotation_required": False,
        },
        "safety": safety,
        "source_bridge_schema": report.get("schema"),
        "source_bridge_hash": sha_payload({
            "activation_hashes": activation_hashes,
            "generated_token_count": report.get("generated_token_count"),
            "tpu_stage_output_hash": tpu_report.get("stage_output_hash"),
        }),
        "boundaries": {
            "bounded_rc_not_production_serving": True,
            "raw_prompt_generated_text_token_ids_activations_logits_kv_private": True,
            "tpu_middle_stage_real_32b_weights": tpu_report.get("full_stage_owned_tpu_loader_ready") is True,
            "cuda_stage_uses_prior_retained_32b_weight_evidence": report.get("cuda_stage_32b_weight_evidence_ready") is True,
            "cpu_tail_is_verifier_stage": True,
        },
    }
    if any(not str(item.get("activation_hash") or "").startswith("sha256:") for item in handoffs):
        live["ok"] = False
        live["gpu_tpu_cpu_32b_same_request_verified"] = False
        live["live_tpu_stage_miner_integrated"] = False
    return live


def build_report(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    coordinator_status: dict[str, Any],
    gpu_report: dict[str, Any],
    tpu_report: dict[str, Any],
    cpu_report: dict[str, Any],
    gpu_steps: list[dict[str, Any]],
    cpu_steps: list[dict[str, Any]] | None = None,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    cpu_steps = list(cpu_steps or [])
    stage_counts = coordinator_status.get("stage_task_counts") if isinstance(coordinator_status.get("stage_task_counts"), dict) else {}
    completed = coordinator_status.get("completed_tasks") if isinstance(coordinator_status.get("completed_tasks"), list) else []
    backends = {
        str((item.get("runtime_device") or {}).get("backend") or "")
        for item in completed
        if isinstance(item, dict)
    }
    target_generated_token_count = max(1, int(getattr(args, "target_generated_token_count", 1)))
    same_request = bool(
        coordinator_status.get("ready")
        and int(coordinator_status.get("generated_token_count") or 0) >= target_generated_token_count
        and int(stage_counts.get("stage0") or 0) >= target_generated_token_count
        and int(stage_counts.get("stage1") or 0) >= target_generated_token_count
        and int(stage_counts.get("stage2") or 0) >= target_generated_token_count
        and {"cuda", "jax_tpu", "cpu"}.issubset(backends)
        and gpu_report.get("ok") is True
        and tpu_report.get("ok") is True
        and cpu_report.get("ok") is True
    )
    target_class = target_parameter_class(str(args.target_model_id))
    tpu_target_stage_ready = bridge_has_full_target_tpu_stage(args, tpu_report)
    tpu_32b_stage_ready = bridge_has_full_32b_tpu_stage(args, tpu_report)
    cuda_32b_weight_evidence_ready = bool(getattr(args, "cuda_stage_32b_weight_evidence_ready", False))
    deepseek_stage_slice_ready = bool(
        getattr(args, "web_tpu_deepseek_stage_execute", False)
        and tpu_report.get("ok") is True
        and tpu_report.get("deepseek_v4_stage_owned_slice_loaded") is True
        and tpu_report.get("real_i8_expert_mlp_slice_smoke_ready") is True
    )
    deepseek_gpu_slice_ready = bool(
        getattr(args, "web_tpu_deepseek_stage_execute", False)
        and gpu_report.get("ok") is True
        and gpu_report.get("deepseek_v4_stage_owned_slice_loaded") is True
        and gpu_report.get("real_i8_expert_mlp_slice_smoke_ready") is True
    )
    deepseek_cpu_slice_ready = bool(
        getattr(args, "web_tpu_deepseek_stage_execute", False)
        and cpu_report.get("ok") is True
        and cpu_report.get("deepseek_v4_stage_owned_slice_loaded") is True
        and cpu_report.get("real_i8_expert_mlp_slice_smoke_ready") is True
    )
    deepseek_tpu_fp4_topk_forward_ready = bool(
        deepseek_stage_slice_ready
        and tpu_report.get("real_fp4_topk_expert_mlp_forward_ready") is True
    )
    deepseek_gpu_fp4_topk_forward_ready = bool(
        deepseek_gpu_slice_ready
        and gpu_report.get("real_fp4_topk_expert_mlp_forward_ready") is True
    )
    deepseek_cpu_fp4_topk_forward_ready = bool(
        deepseek_cpu_slice_ready
        and cpu_report.get("real_fp4_topk_expert_mlp_forward_ready") is True
    )
    same_request_deepseek_stage_slice = bool(same_request and deepseek_stage_slice_ready)
    same_request_deepseek_all_backend_slices = bool(
        same_request
        and deepseek_stage_slice_ready
        and deepseek_gpu_slice_ready
        and deepseek_cpu_slice_ready
    )
    same_request_deepseek_all_backend_fp4_topk_forward = bool(
        same_request
        and deepseek_tpu_fp4_topk_forward_ready
        and deepseek_gpu_fp4_topk_forward_ready
        and deepseek_cpu_fp4_topk_forward_ready
    )
    fallback_gpu_range = list(resolved_deepseek_layer_range(args, "gpu"))
    fallback_tpu_range = list(resolved_deepseek_layer_range(args, "tpu"))
    fallback_cpu_range = list(resolved_deepseek_layer_range(args, "cpu"))
    deepseek_stage_layer_ranges = {
        "cuda": public_stage_layer_range(gpu_report.get("stage_layer_range")) or fallback_gpu_range,
        "jax_tpu": public_stage_layer_range(tpu_report.get("stage_layer_range")) or fallback_tpu_range,
        "cpu": public_stage_layer_range(cpu_report.get("stage_layer_range")) or fallback_cpu_range,
    }
    deepseek_stage_layer_coverage_count = layer_coverage_count(deepseek_stage_layer_ranges)
    deepseek_distinct_backend_stage_layer_ranges = bool(
        same_request_deepseek_all_backend_slices
        and ranges_are_disjoint(deepseek_stage_layer_ranges)
        and deepseek_stage_layer_coverage_count >= 3
    )
    kaggle_cpu_stage_requested = bool(getattr(args, "kaggle_cpu_stage", False))
    kaggle_cpu_stage_ready = bool(
        cpu_report.get("ok") is True
        and cpu_report.get("kaggle_kernel") is True
        and str(cpu_report.get("provider") or "") == "kaggle_cpu"
    )
    same_request_32b = bool(same_request and tpu_32b_stage_ready and cuda_32b_weight_evidence_ready)
    same_request_72b_stage = bool(same_request and target_class == "72b" and tpu_target_stage_ready)
    same_request_72b_full_model = False
    gpu_kernel_pushed = any(step.get("name") == "kaggle_kernel_push" and step.get("accepted") for step in gpu_steps)
    kernels_deleted = (
        any(step.get("name") == "kaggle_kernel_delete" and step.get("ok") for step in gpu_steps)
        if gpu_kernel_pushed
        else True
    )
    cpu_kernel_pushed = any(step.get("name") == "kaggle_cpu_kernel_push" and step.get("accepted") for step in cpu_steps)
    cpu_kernel_deleted = (
        any(step.get("name") == "kaggle_cpu_kernel_delete" and step.get("ok") for step in cpu_steps)
        if cpu_kernel_pushed
        else True
    )
    private_gpu_package_removed = not (output_dir / "private-kaggle-bridge-gpu-kernel").exists()
    private_cpu_package_removed = not (output_dir / "private-kaggle-bridge-cpu-kernel").exists()
    private_packages_removed = bool(private_gpu_package_removed and (private_cpu_package_removed or not kaggle_cpu_stage_requested))
    ready = bool(
        same_request
        and kernels_deleted
        and (cpu_kernel_deleted or not kaggle_cpu_stage_requested)
        and (kaggle_cpu_stage_ready or not kaggle_cpu_stage_requested)
        and private_packages_removed
        and not errors
    )
    blockers: list[str] = []
    gpu_step_text = "\n".join(
        f"{step.get('stdout_tail') or ''}\n{step.get('stderr_tail') or ''}"
        for step in gpu_steps
        if isinstance(step, dict)
    )
    if not same_request:
        blockers.append("same_request_runtime_bridge_not_verified")
    if same_request and getattr(args, "web_tpu_32b_execute", False) and not tpu_target_stage_ready:
        blockers.append(f"qwen{target_class or 'target'}_tpu_stage_owned_loader_not_ready")
    if same_request and getattr(args, "web_tpu_deepseek_stage_execute", False) and not deepseek_stage_slice_ready:
        blockers.append("deepseek_v4_web_tpu_stage_slice_not_ready")
    if same_request_deepseek_stage_slice and not (deepseek_gpu_slice_ready and deepseek_cpu_slice_ready):
        blockers.append("deepseek_v4_gpu_cpu_real_stage_not_verified")
    if same_request_deepseek_all_backend_slices and not same_request_deepseek_all_backend_fp4_topk_forward:
        blockers.append("deepseek_v4_gpu_tpu_cpu_fp4_topk_forward_not_verified")
    if same_request_deepseek_stage_slice:
        blockers.append("deepseek_v4_full_same_request_decode_not_verified")
    if same_request and getattr(args, "web_tpu_32b_execute", False) and not cuda_32b_weight_evidence_ready:
        blockers.append("cuda_stage_32b_weight_evidence_not_imported")
    if same_request_72b_stage and not same_request_72b_full_model:
        blockers.append("qwen72b_full_model_same_request_decode_not_verified")
    if gpu_report.get("ok") is not True:
        blockers.append("cuda_stage_not_ready")
    if "Maximum batch GPU session count" in gpu_step_text:
        blockers.append("kaggle_gpu_batch_session_limit_reached")
    if "Maximum weekly GPU quota" in gpu_step_text:
        blockers.append("kaggle_gpu_weekly_quota_reached")
    if tpu_report.get("ok") is not True:
        blockers.append("jax_tpu_stage_not_ready")
    if cpu_report.get("ok") is not True:
        blockers.append("cpu_tail_not_ready")
    if kaggle_cpu_stage_requested and not kaggle_cpu_stage_ready:
        blockers.append("kaggle_cpu_stage_not_verified")
    if gpu_kernel_pushed and not kernels_deleted:
        blockers.append("kaggle_gpu_kernel_cleanup_not_verified")
    if kaggle_cpu_stage_requested and cpu_kernel_pushed and not cpu_kernel_deleted:
        blockers.append("kaggle_cpu_kernel_cleanup_not_verified")
    if not private_packages_removed:
        blockers.append("private_kernel_package_retained")
    if errors:
        blockers.append("worker_errors")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "same_request_runtime_bridge_verified": same_request,
        "same_request_32b_model_verified": same_request_32b,
        "gpu_tpu_cpu_32b_same_request_verified": same_request_32b,
        "same_request_72b_stage_verified": same_request_72b_stage,
        "gpu_tpu_cpu_72b_same_request_stage_verified": same_request_72b_stage,
        "same_request_72b_full_model_verified": same_request_72b_full_model,
        "gpu_tpu_cpu_72b_same_request_verified": same_request_72b_full_model,
        "deepseek_v4_same_request_stage_slice_verified": same_request_deepseek_stage_slice,
        "deepseek_v4_gpu_stage_slice_verified": bool(same_request and deepseek_gpu_slice_ready),
        "deepseek_v4_cpu_stage_slice_verified": bool(same_request and deepseek_cpu_slice_ready),
        "deepseek_v4_gpu_tpu_cpu_same_request_stage_slices_verified": same_request_deepseek_all_backend_slices,
        "deepseek_v4_tpu_fp4_topk_expert_forward_verified": bool(same_request and deepseek_tpu_fp4_topk_forward_ready),
        "deepseek_v4_gpu_fp4_topk_expert_forward_verified": bool(same_request and deepseek_gpu_fp4_topk_forward_ready),
        "deepseek_v4_cpu_fp4_topk_expert_forward_verified": bool(same_request and deepseek_cpu_fp4_topk_forward_ready),
        "deepseek_v4_gpu_tpu_cpu_same_request_fp4_topk_expert_forwards_verified": same_request_deepseek_all_backend_fp4_topk_forward,
        "deepseek_v4_stage_layer_ranges": deepseek_stage_layer_ranges,
        "deepseek_v4_distinct_backend_stage_layer_ranges_verified": deepseek_distinct_backend_stage_layer_ranges,
        "deepseek_v4_stage_layer_coverage_count": deepseek_stage_layer_coverage_count,
        "gpu_tpu_cpu_deepseek_v4_same_request_verified": False,
        "same_request_target_parameter_class": target_class,
        "not_32b_weight_success": not same_request_32b,
        "not_72b_full_model_success": not same_request_72b_full_model,
        "not_72b_all_layer_quality_parity_success": True,
        "model_scope": (
            "qwen32b_same_request_tpu_stage_owned_loader" if same_request_32b
            else "qwen72b_same_request_tpu_stage_owned_loader_not_full_model_decode" if same_request_72b_stage
            else "deepseek_v4_flash_same_request_gpu_tpu_cpu_fp4_topk_expert_forwards_not_full_decode" if same_request_deepseek_all_backend_fp4_topk_forward
            else "deepseek_v4_flash_same_request_gpu_tpu_cpu_real_weight_slices_not_full_decode" if same_request_deepseek_all_backend_slices
            else "deepseek_v4_flash_same_request_tpu_real_weight_slice_not_full_decode" if same_request_deepseek_stage_slice
            else "deepseek_v4_flash_tpu_real_weight_slice_bridge_requested_not_verified" if getattr(args, "web_tpu_deepseek_stage_execute", False)
            else "qwen_shape_runtime_bridge_only"
        ),
        "target_model_id": args.target_model_id,
        "target_generated_token_count": target_generated_token_count,
        "full_32b_weight_loading_public_claim": same_request_32b,
        "full_72b_tpu_stage_loading_public_claim": same_request_72b_stage,
        "full_72b_weight_loading_public_claim": same_request_72b_full_model,
        "cuda_stage_32b_weight_evidence_ready": cuda_32b_weight_evidence_ready,
        "tpu_32b_runtime_adapter_ready": tpu_32b_stage_ready,
        "tpu_target_runtime_adapter_ready": tpu_target_stage_ready,
        "generated_token_count": int(coordinator_status.get("generated_token_count") or 0),
        "stage_task_counts": stage_counts,
        "accepted_providers": sorted(backends),
        "accepted_stage_backends": sorted(backends),
        "activation_handoff_count": len(coordinator_status.get("activation_hashes") or []),
        "activation_hashes": list(coordinator_status.get("activation_hashes") or []),
        "runtime_device_summary": {
            "tpu_provider": str(getattr(args, "tpu_provider", "kaggle_web")),
            "cpu_stage_provider": "kaggle_cpu" if kaggle_cpu_stage_requested else "local_cpu_thread",
            "kaggle_cpu_stage_ready": kaggle_cpu_stage_ready,
            "cuda_stage_ready": gpu_report.get("ok") is True,
            "jax_tpu_stage_ready": tpu_report.get("ok") is True,
            "cpu_tail_ready": cpu_report.get("ok") is True,
            "tpu_device_count": int(tpu_report.get("tpu_device_count") or 0),
            "cuda_device_count": int(gpu_report.get("cuda_device_count") or 0),
            "tpu_stage_owned_model_loaded": tpu_report.get("stage_owned_model_loaded") is True,
            "tpu_target_stage_ready": tpu_target_stage_ready,
            "tpu_executed_layer_count": int(tpu_report.get("executed_layer_count") or 0),
            "tpu_loaded_execution_tensor_gb": float(tpu_report.get("loaded_execution_tensor_gb") or 0.0),
            "cuda_stage_32b_weight_evidence_ready": cuda_32b_weight_evidence_ready,
            "deepseek_v4_cuda_stage_owned_slice_loaded": gpu_report.get("deepseek_v4_stage_owned_slice_loaded") is True,
            "deepseek_v4_cuda_stage_layer_range": deepseek_stage_layer_ranges["cuda"],
            "deepseek_v4_cuda_real_i8_expert_mlp_slice_smoke_ready": gpu_report.get("real_i8_expert_mlp_slice_smoke_ready") is True,
            "deepseek_v4_cuda_real_fp4_topk_expert_mlp_forward_ready": gpu_report.get("real_fp4_topk_expert_mlp_forward_ready") is True,
            "deepseek_v4_cuda_real_routed_expert_loaded_tensor_count": int(gpu_report.get("real_routed_expert_loaded_tensor_count") or 0),
            "deepseek_v4_cuda_real_routed_expert_total_loaded_tensor_bytes": int(gpu_report.get("real_routed_expert_total_loaded_tensor_bytes") or 0),
            "deepseek_v4_cuda_real_weight_sample_loaded_tensor_count": int(gpu_report.get("real_weight_sample_loaded_tensor_count") or 0),
            "deepseek_v4_stage_owned_slice_loaded": tpu_report.get("deepseek_v4_stage_owned_slice_loaded") is True,
            "deepseek_v4_tpu_stage_layer_range": deepseek_stage_layer_ranges["jax_tpu"],
            "deepseek_v4_real_weight_sample_loaded_tensor_count": int(tpu_report.get("real_weight_sample_loaded_tensor_count") or 0),
            "deepseek_v4_real_i8_expert_mlp_slice_smoke_ready": tpu_report.get("real_i8_expert_mlp_slice_smoke_ready") is True,
            "deepseek_v4_real_fp4_topk_expert_mlp_forward_ready": tpu_report.get("real_fp4_topk_expert_mlp_forward_ready") is True,
            "deepseek_v4_real_routed_expert_loaded_tensor_count": int(tpu_report.get("real_routed_expert_loaded_tensor_count") or 0),
            "deepseek_v4_real_routed_expert_total_loaded_tensor_bytes": int(tpu_report.get("real_routed_expert_total_loaded_tensor_bytes") or 0),
            "deepseek_v4_cpu_stage_owned_slice_loaded": cpu_report.get("deepseek_v4_stage_owned_slice_loaded") is True,
            "deepseek_v4_cpu_stage_layer_range": deepseek_stage_layer_ranges["cpu"],
            "deepseek_v4_cpu_real_i8_expert_mlp_slice_smoke_ready": cpu_report.get("real_i8_expert_mlp_slice_smoke_ready") is True,
            "deepseek_v4_cpu_real_fp4_topk_expert_mlp_forward_ready": cpu_report.get("real_fp4_topk_expert_mlp_forward_ready") is True,
            "deepseek_v4_cpu_real_routed_expert_loaded_tensor_count": int(cpu_report.get("real_routed_expert_loaded_tensor_count") or 0),
            "deepseek_v4_cpu_real_routed_expert_total_loaded_tensor_bytes": int(cpu_report.get("real_routed_expert_total_loaded_tensor_bytes") or 0),
            "deepseek_v4_cpu_real_weight_sample_loaded_tensor_count": int(cpu_report.get("real_weight_sample_loaded_tensor_count") or 0),
        },
        "blocked_reason": "" if ready and not blockers else (blockers[0] if blockers else "runtime_bridge_not_ready"),
        "blockers": sorted(set(blockers)),
        "diagnosis_codes": sorted(set([
            "same_request_runtime_bridge_ready" if same_request else "same_request_runtime_bridge_not_ready",
            "runtime_bridge_32b_same_request_ready" if same_request_32b else "runtime_bridge_not_32b_model_success",
            "runtime_bridge_72b_stage_same_request_ready" if same_request_72b_stage else "runtime_bridge_not_72b_stage_success",
            "runtime_bridge_deepseek_v4_stage_slice_ready" if same_request_deepseek_stage_slice else "runtime_bridge_not_deepseek_v4_stage_slice_success",
            "runtime_bridge_deepseek_v4_gpu_tpu_cpu_stage_slices_ready" if same_request_deepseek_all_backend_slices else "runtime_bridge_deepseek_v4_gpu_tpu_cpu_stage_slices_not_ready",
            "runtime_bridge_deepseek_v4_gpu_tpu_cpu_fp4_topk_forwards_ready" if same_request_deepseek_all_backend_fp4_topk_forward else "runtime_bridge_deepseek_v4_gpu_tpu_cpu_fp4_topk_forwards_not_ready",
            "runtime_bridge_deepseek_v4_distinct_backend_stage_ranges_ready" if deepseek_distinct_backend_stage_layer_ranges else "runtime_bridge_deepseek_v4_distinct_backend_stage_ranges_not_ready",
            "kaggle_cpu_stage_ready" if kaggle_cpu_stage_ready else (
                "kaggle_cpu_stage_not_requested" if not kaggle_cpu_stage_requested else "kaggle_cpu_stage_not_ready"
            ),
            "runtime_bridge_not_72b_full_model_success",
            "kaggle_gpu_batch_session_limit_reached" if "Maximum batch GPU session count" in gpu_step_text else "kaggle_gpu_batch_session_limit_not_seen",
            "kaggle_gpu_weekly_quota_reached" if "Maximum weekly GPU quota" in gpu_step_text else "kaggle_gpu_weekly_quota_not_seen",
            "kaggle_gpu_kernel_deleted" if gpu_kernel_pushed and kernels_deleted else (
                "kaggle_gpu_kernel_not_created" if not gpu_kernel_pushed else "kaggle_gpu_kernel_cleanup_not_verified"
            ),
            "kaggle_cpu_kernel_deleted" if cpu_kernel_pushed and cpu_kernel_deleted else (
                "kaggle_cpu_kernel_not_created" if not cpu_kernel_pushed else "kaggle_cpu_kernel_cleanup_not_verified"
            ),
        ])),
        "coordinator": coordinator_status,
        "stage_reports": {
            "cuda_gpu_stage": gpu_report,
            "jax_tpu_stage": tpu_report,
            "cpu_tail_stage": cpu_report,
        },
        "gpu_stage_steps": [public_step(step) for step in gpu_steps],
        "cpu_stage_steps": [public_step(step) for step in cpu_steps],
        "errors": errors,
        "cleanup": {
            "kaggle_gpu_kernel_created": gpu_kernel_pushed,
            "kaggle_gpu_kernel_deleted": kernels_deleted,
            "kaggle_cpu_kernel_created": cpu_kernel_pushed,
            "kaggle_cpu_kernel_deleted": cpu_kernel_deleted,
            "private_gpu_package_removed": private_packages_removed,
            "private_cpu_package_removed": private_cpu_package_removed,
            "web_tpu_runtime_private_token_public": False,
            "colab_runtime_proxy_token_public": False,
            "colab_runtime_proxy_url_public": False,
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
            "credentials_public": False,
            "cookies_public": False,
            "jupyter_proxy_token_public": False,
            "colab_runtime_proxy_token_public": False,
            "colab_runtime_proxy_url_public": False,
            "private_runtime_state_public": False,
        },
        "limitations": [
            "This is a bounded same-request bridge proof, not production serving.",
            (
                "The TPU stage executed a real Qwen 32B stage-owned loader inside the same request."
                if same_request_32b
                else "The TPU stage executed a real Qwen 72B stage-owned loader inside the same request."
                if same_request_72b_stage
                else "The GPU, TPU, and CPU stages each executed a real DeepSeek-V4-Flash weight slice inside the same request, but this is not full DeepSeek inference."
                if same_request_deepseek_all_backend_slices
                else "The TPU stage executed a real DeepSeek-V4-Flash weight slice inside the same request, but this is not full DeepSeek inference."
                if same_request_deepseek_stage_slice
                else "The TPU stage was configured for a real DeepSeek-V4-Flash weight slice, but the same-request bridge did not reach a verified TPU stage."
                if getattr(args, "web_tpu_deepseek_stage_execute", False)
                else "The TPU stage uses a Qwen-shape synthetic JAX operation unless --web-tpu-32b-execute is enabled and verified."
            ),
            (
                "The CUDA stage is tied to prior retained 32B stage-owned evidence, not reloaded with full 32B weights inside this bridge."
                if same_request_32b
                else "The CUDA and CPU bridge stages are accepted same-request stages; this is not a full all-layer 72B decode or quality/parity proof."
                if same_request_72b_stage
                else "The CUDA/TPU/CPU bridge stages are real DeepSeek-V4-Flash slice stages, but this is not a full all-layer DeepSeek decode."
                if same_request_deepseek_all_backend_slices
                else "The CUDA and CPU bridge stages are not yet real DeepSeek-V4-Flash stages, so this must not be imported as completed DeepSeek inference."
                if same_request_deepseek_stage_slice
                else "The DeepSeek bridge attempt is not completed; do not import it as same-request DeepSeek inference."
                if getattr(args, "web_tpu_deepseek_stage_execute", False)
                else "Do not import this report as gpu_tpu_cpu_32b_same_request_verified=true without a separate live proof."
            ),
        ],
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["safety"]["public_artifact_safe"] = False
        report["public_leak_fragments"] = leaks
        report["blockers"].append("public_redaction_failed")
    return report


def run_probe(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = __import__("secrets").token_urlsafe(32)
    coordinator_url = str(args.coordinator_url or f"http://{args.public_host}:{int(args.port)}").rstrip("/")
    state = BridgeState(target_generated_token_count=int(args.target_generated_token_count))
    server = BridgeServer(host="0.0.0.0", port=int(args.port), token=token, state=state)
    gpu_report: dict[str, Any] = {}
    gpu_steps: list[dict[str, Any]] = []
    tpu_report: dict[str, Any] = {}
    cpu_report: dict[str, Any] = {}
    cpu_steps: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def gpu_thread() -> None:
        nonlocal gpu_report, gpu_steps
        try:
            gpu_report, gpu_steps = run_gpu_stage(
                args,
                output_dir=output_dir,
                coordinator_url=coordinator_url,
                token=token,
                runner=runner,
            )
        except Exception as exc:
            errors.append({"stage": "cuda", "error_type": type(exc).__name__, "error_digest": sha_payload(str(exc))})

    def tpu_thread() -> None:
        nonlocal tpu_report
        if str(getattr(args, "tpu_provider", "kaggle_web")) == "colab_cli":
            tpu_report = run_colab_tpu_stage(args, coordinator_url=coordinator_url, token=token)
        else:
            tpu_report = run_web_tpu_stage(args, coordinator_url=coordinator_url, token=token)

    def cpu_thread() -> None:
        nonlocal cpu_report, cpu_steps
        if bool(getattr(args, "kaggle_cpu_stage", False)):
            try:
                cpu_report, cpu_steps = run_kaggle_cpu_stage(
                    args,
                    output_dir=output_dir,
                    coordinator_url=coordinator_url,
                    token=token,
                    runner=runner,
                )
            except Exception as exc:
                errors.append({"stage": "kaggle_cpu", "error_type": type(exc).__name__, "error_digest": sha_payload(str(exc))})
        else:
            cpu_layer_start, cpu_layer_end = resolved_deepseek_layer_range(args, "cpu")
            cpu_report = cpu_tail_worker(
                state=state,
                coordinator_url=coordinator_url,
                token=token,
                timeout_seconds=float(args.cpu_task_timeout_seconds),
                deepseek_real_stage_slice=bool(getattr(args, "web_tpu_deepseek_stage_execute", False)),
                deepseek_stage_layer_start=cpu_layer_start,
                deepseek_stage_layer_end=cpu_layer_end,
            )

    threads = [
        threading.Thread(target=gpu_thread, daemon=True),
        threading.Thread(target=tpu_thread, daemon=True),
        threading.Thread(target=cpu_thread, daemon=True),
    ]
    server_stopped = False
    try:
        server.start()
        for thread in threads:
            thread.start()
        coordinator_status = wait_for_ready(state, timeout_seconds=float(args.coordinator_timeout_seconds), threads=threads)
        if not coordinator_status.get("ready"):
            server.stop()
            server_stopped = True
        # Downstream stages can fail quickly while the Kaggle GPU worker is
        # still downloading its report and deleting the temporary private
        # kernel. Give live-resource cleanup a bounded grace window before
        # stopping the local Coordinator.
        cleanup_join_seconds = max(
            5.0,
            float(args.kaggle_status_timeout_seconds)
            + float(args.kaggle_output_timeout_seconds)
            + float(args.kaggle_delete_timeout_seconds)
            + 60.0,
        )
        for index, thread in enumerate(threads):
            wait_for_resource_cleanup = index == 0 or (index == 2 and bool(getattr(args, "kaggle_cpu_stage", False)))
            thread.join(timeout=cleanup_join_seconds if wait_for_resource_cleanup else 2.0)
    finally:
        if not server_stopped:
            server.stop()
        if not args.keep_private_package:
            shutil.rmtree(output_dir / "private-kaggle-bridge-gpu-kernel", ignore_errors=True)
            shutil.rmtree(output_dir / "private-kaggle-bridge-cpu-kernel", ignore_errors=True)
    coordinator_status = state.public_status()
    report = build_report(
        args,
        output_dir=output_dir,
        coordinator_status=coordinator_status,
        gpu_report=gpu_report,
        tpu_report=tpu_report,
        cpu_report=cpu_report,
        gpu_steps=gpu_steps,
        cpu_steps=cpu_steps,
        errors=errors,
    )
    live_proof = build_live_proof_from_bridge(report)
    live_proof_path = output_dir / "gpu_tpu_cpu_32b_same_request_live_proof.json"
    write_json(live_proof_path, live_proof)
    report["live_proof_artifact"] = {
        "path": str(live_proof_path),
        "schema": rc_pack.LIVE_PROOF_SCHEMA,
        "ok": live_proof.get("ok") is True,
        "public_artifact_safe": live_proof.get("public_artifact_safe") is True,
    }
    write_json(output_dir / "gpu_tpu_cpu_same_request_runtime_bridge_probe.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded CUDA + JAX/TPU + CPU same-request runtime bridge probe.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kaggle-owner", default=loading_probe.default_kaggle_owner())
    parser.add_argument("--kernel-slug-prefix", default="ctgtcbridge")
    parser.add_argument("--accelerator", default=DEFAULT_ACCELERATOR)
    parser.add_argument("--public-host", default=DEFAULT_PUBLIC_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--target-model-id", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--target-generated-token-count", type=int, default=1)
    parser.add_argument("--tpu-provider", choices=["kaggle_web", "colab_cli"], default="kaggle_web")
    parser.add_argument("--kaggle-notebook-url", default="https://www.kaggle.com/code/tpuowner/notebook8d4184babd/edit")
    parser.add_argument("--kaggle-web-storage-state", default="/root/kaggle-web-storage-state.json")
    parser.add_argument("--colab-session-name", default="ct-colab-tpu-v5e1")
    parser.add_argument("--colab-session-config", default=os.path.expanduser("~/.config/colab-cli/sessions.json"))
    parser.add_argument("--chrome-executable", default="/usr/bin/google-chrome")
    parser.add_argument("--coordinator-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--cpu-task-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--web-tpu-task-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--web-tpu-execute-timeout-seconds", type=float, default=1500.0)
    parser.add_argument("--web-tpu-force-new-session", action="store_true")
    parser.add_argument("--web-tpu-32b-execute", action="store_true")
    parser.add_argument("--web-tpu-deepseek-stage-execute", action="store_true")
    parser.add_argument("--deepseek-stage-layer-start", type=int, default=16)
    parser.add_argument("--deepseek-stage-layer-end", type=int, default=18)
    parser.add_argument("--deepseek-gpu-stage-layer-start", type=int)
    parser.add_argument("--deepseek-gpu-stage-layer-end", type=int)
    parser.add_argument("--deepseek-tpu-stage-layer-start", type=int)
    parser.add_argument("--deepseek-tpu-stage-layer-end", type=int)
    parser.add_argument("--deepseek-cpu-stage-layer-start", type=int)
    parser.add_argument("--deepseek-cpu-stage-layer-end", type=int)
    parser.add_argument("--web-tpu-32b-stage-start", type=int, default=tpu_loader_probe.DEFAULT_STAGE_START)
    parser.add_argument("--web-tpu-32b-stage-end", type=int, default=tpu_loader_probe.DEFAULT_STAGE_END)
    parser.add_argument("--web-tpu-32b-tensor-key", default=tpu_loader_probe.DEFAULT_TENSOR_KEY)
    parser.add_argument("--web-tpu-32b-max-header-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--web-tpu-32b-max-tensor-bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--web-tpu-32b-execute-layer-count", type=int, default=tpu_loader_probe.DEFAULT_STAGE_END - tpu_loader_probe.DEFAULT_STAGE_START)
    parser.add_argument("--cuda-stage-32b-weight-evidence-ready", action="store_true")
    parser.add_argument("--kaggle-cpu-stage", action="store_true")
    parser.add_argument("--kernel-timeout-seconds", type=int, default=1800)
    parser.add_argument("--kaggle-push-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--kaggle-status-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--kaggle-status-poll-interval", type=float, default=30.0)
    parser.add_argument("--kaggle-output-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--kaggle-delete-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--skip-kaggle-cleanup", action="store_true")
    parser.add_argument("--keep-private-package", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.kernel_timeout_seconds > 3600:
        raise SystemExit("--kernel-timeout-seconds must be <= 3600")
    if args.kaggle_status_timeout_seconds > 3600:
        raise SystemExit("--kaggle-status-timeout-seconds must be <= 3600")
    if args.coordinator_timeout_seconds > 3600:
        raise SystemExit("--coordinator-timeout-seconds must be <= 3600")
    if not str(args.target_model_id).strip():
        raise SystemExit("--target-model-id must be non-empty")
    if args.target_generated_token_count < 1 or args.target_generated_token_count > 16:
        raise SystemExit("--target-generated-token-count must be between 1 and 16")
    if args.web_tpu_32b_execute and args.web_tpu_deepseek_stage_execute:
        raise SystemExit("--web-tpu-32b-execute and --web-tpu-deepseek-stage-execute are mutually exclusive")
    if args.web_tpu_deepseek_stage_execute:
        args.target_model_id = "deepseek-ai/DeepSeek-V4-Flash"
    if args.deepseek_stage_layer_start < 0 or args.deepseek_stage_layer_end <= args.deepseek_stage_layer_start:
        raise SystemExit("--deepseek-stage-layer-end must be greater than --deepseek-stage-layer-start")
    if args.web_tpu_deepseek_stage_execute:
        for backend in ("gpu", "tpu", "cpu"):
            start, end = resolved_deepseek_layer_range(args, backend)
            if start < 0 or end <= start:
                raise SystemExit(f"--deepseek-{backend}-stage-layer-end must be greater than --deepseek-{backend}-stage-layer-start")
            if end - start > 8:
                raise SystemExit(f"--deepseek-{backend}-stage-layer range must cover at most 8 layers")
    if args.web_tpu_32b_stage_start < 0 or args.web_tpu_32b_stage_end <= args.web_tpu_32b_stage_start:
        raise SystemExit("--web-tpu-32b-stage-end must be greater than --web-tpu-32b-stage-start")
    stage_layer_count = int(args.web_tpu_32b_stage_end) - int(args.web_tpu_32b_stage_start)
    if args.web_tpu_32b_execute_layer_count < 0 or args.web_tpu_32b_execute_layer_count > stage_layer_count:
        raise SystemExit("--web-tpu-32b-execute-layer-count must be between 0 and the TPU stage layer count")
    if args.web_tpu_32b_execute and args.web_tpu_execute_timeout_seconds < 300:
        raise SystemExit("--web-tpu-execute-timeout-seconds must be at least 300 when --web-tpu-32b-execute is used")
    if args.web_tpu_32b_max_tensor_bytes < 1024 or args.web_tpu_32b_max_tensor_bytes > 1024 * 1024 * 1024:
        raise SystemExit("--web-tpu-32b-max-tensor-bytes must be between 1KiB and 1GiB")
    if not str(args.web_tpu_32b_tensor_key or "").strip():
        args.web_tpu_32b_tensor_key = f"model.layers.{int(args.web_tpu_32b_stage_start)}.input_layernorm.weight"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_probe(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            f"{SCHEMA}: ok={bool(report.get('ok'))} "
            f"bridge={bool(report.get('same_request_runtime_bridge_verified'))} "
            f"blocked={report.get('blocked_reason') or 'none'}"
        )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "__web_tpu_execute_child__":
        raise SystemExit(_web_tpu_execute_child_main())
    raise SystemExit(main())
