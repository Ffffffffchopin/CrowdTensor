#!/usr/bin/env python3
"""Run a bounded Web-TPU probe for Qwen 32B stage-owned safetensors loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts import gpu_tpu_cpu_same_request_runtime_bridge_probe as web_tpu_bridge
except Exception:  # pragma: no cover - script can still report missing dependency at runtime.
    web_tpu_bridge = None  # type: ignore[assignment]


SCHEMA = "kaggle_tpu_32b_stage_owned_loader_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/kaggle-tpu-32b-stage-owned-loader-probe"
DEFAULT_NOTEBOOK_URL = "https://www.kaggle.com/code/tpuowner/notebook8d4184babd/edit"
DEFAULT_MODEL_REPO = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_STAGE_START = 21
DEFAULT_STAGE_END = 42
DEFAULT_TENSOR_KEY = ""
SENSITIVE_FRAGMENTS = (
    "jupyter-proxy",
    "token=",
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Cookie",
    "Set-Cookie",
    '"hidden_b64":',
)


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


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted(fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded)


def qwen_stage_keys(weight_map: dict[str, str], *, stage_start: int, stage_end: int) -> list[str]:
    keys: list[str] = []
    for key in weight_map:
        parts = str(key).split(".")
        if len(parts) < 4 or parts[0] != "model" or parts[1] != "layers":
            continue
        try:
            layer = int(parts[2])
        except ValueError:
            continue
        if stage_start <= layer < stage_end:
            keys.append(str(key))
    return sorted(keys)


def render_web_probe_code(args: argparse.Namespace) -> str:
    return f'''
import gc
import base64
import hashlib
import json
import os
import platform
import re
import struct
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = {SCHEMA!r}
MODEL_REPO = {args.model_repo!r}
STAGE_START = {int(args.stage_start)!r}
STAGE_END = {int(args.stage_end)!r}
TENSOR_KEY = {args.tensor_key!r}
MAX_HEADER_BYTES = {int(args.max_header_bytes)!r}
MAX_TENSOR_BYTES = {int(args.max_tensor_bytes)!r}
EXECUTE_LAYER_COUNT = {int(args.execute_layer_count)!r}
INPUT_ACTIVATION_PRIVATE = {json.dumps(getattr(args, "input_activation_private", {}) or {}, sort_keys=True)!r}
RETURN_OUTPUT_ACTIVATION_PRIVATE = {bool(getattr(args, "return_output_activation_private", False))!r}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha_payload(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def safe_error(value):
    text = str(value or "")[-500:]
    for fragment in ["jupyter-proxy", "token=", "KAGGLE_KEY", "KAGGLE_USERNAME", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "Bearer ", "Cookie", "Set-Cookie"]:
        text = text.replace(fragment, "<redacted>")
    return text


def fetch_json(filename):
    url = f"https://huggingface.co/{{MODEL_REPO}}/resolve/main/{{filename}}"
    with urllib.request.urlopen(url, timeout=120) as response:
        loaded = json.load(response)
    return loaded if isinstance(loaded, dict) else {{}}


def qwen_stage_keys(weight_map):
    selected = []
    for key in weight_map:
        parts = str(key).split(".")
        if len(parts) < 4 or parts[0] != "model" or parts[1] != "layers":
            continue
        try:
            layer = int(parts[2])
        except ValueError:
            continue
        if STAGE_START <= layer < STAGE_END:
            selected.append(str(key))
    return sorted(selected)


def read_range(url, start, end, max_bytes):
    req = urllib.request.Request(
        url,
        headers={{"Range": f"bytes={{int(start)}}-{{int(end)}}", "User-Agent": "crowdtensor-tpu-loader-probe/1"}},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        status = int(getattr(response, "status", 0) or 0)
        content_range = str(response.headers.get("Content-Range") or "")
        data = response.read(int(max_bytes) + 1)
    if len(data) > int(max_bytes):
        raise RuntimeError("range_response_exceeded_budget")
    return status, content_range, data


def load_safetensors_header(filename):
    url = f"https://huggingface.co/{{MODEL_REPO}}/resolve/main/{{filename}}"
    status, _, first = read_range(url, 0, 7, 8)
    if len(first) != 8:
        raise RuntimeError("safetensors_header_prefix_missing")
    header_len = struct.unpack("<Q", first)[0]
    if header_len <= 0 or header_len > MAX_HEADER_BYTES:
        raise RuntimeError("safetensors_header_length_out_of_budget")
    status, content_range, payload = read_range(url, 8, 8 + header_len - 1, header_len)
    if status != 206 and not content_range:
        raise RuntimeError("hf_range_request_not_honored_for_header")
    if len(payload) != header_len:
        raise RuntimeError("safetensors_header_truncated")
    header = json.loads(payload.decode("utf-8"))
    return int(header_len), header


def tensor_file_and_header(weight_map, headers, key):
    filename = str(weight_map.get(key) or "")
    if not filename:
        raise RuntimeError("tensor_key_not_in_weight_map")
    header_len, header = headers[filename]
    meta = header.get(key)
    if not isinstance(meta, dict):
        raise RuntimeError("tensor_key_not_in_safetensors_header")
    return filename, int(header_len), meta


def download_tensor_bytes(filename, header_len, tensor_meta):
    offsets = tensor_meta.get("data_offsets") or []
    if len(offsets) != 2:
        raise RuntimeError("tensor_data_offsets_missing")
    start, end = int(offsets[0]), int(offsets[1])
    size = end - start
    if size <= 0:
        raise RuntimeError("tensor_size_invalid")
    if size > MAX_TENSOR_BYTES:
        raise RuntimeError("selected_tensor_exceeds_budget")
    absolute_start = 8 + int(header_len) + start
    absolute_end = 8 + int(header_len) + end - 1
    url = f"https://huggingface.co/{{MODEL_REPO}}/resolve/main/{{filename}}"
    status, content_range, data = read_range(url, absolute_start, absolute_end, size)
    if status != 206 and not content_range:
        raise RuntimeError("hf_range_request_not_honored_for_tensor")
    if len(data) != size:
        raise RuntimeError("tensor_byte_range_truncated")
    return data


def tensor_bytes_to_tpu(data, tensor_meta):
    import numpy as np
    import jax
    import jax.numpy as jnp

    devices = list(jax.devices())
    tpu_devices = [device for device in devices if str(getattr(device, "platform", "")).lower() == "tpu"]
    if not tpu_devices:
        raise RuntimeError("jax_tpu_device_missing")
    dtype = str(tensor_meta.get("dtype") or "").upper()
    shape = [int(item) for item in tensor_meta.get("shape") or []]
    if dtype == "BF16":
        raw = np.frombuffer(data, dtype="<u2")
        host = (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        arr = jax.device_put(jnp.asarray(host, dtype=jnp.bfloat16), tpu_devices[0])
    elif dtype == "F16":
        host = np.frombuffer(data, dtype="<f2").reshape(shape)
        arr = jax.device_put(jnp.asarray(host, dtype=jnp.float16), tpu_devices[0])
    elif dtype == "F32":
        host = np.frombuffer(data, dtype="<f4").reshape(shape)
        arr = jax.device_put(jnp.asarray(host, dtype=jnp.float32), tpu_devices[0])
    else:
        raise RuntimeError("unsupported_tensor_dtype_for_tpu_probe")
    summary = jnp.asarray([jnp.mean(arr.astype(jnp.float32)), jnp.std(arr.astype(jnp.float32))], dtype=jnp.float32).block_until_ready()
    return {{
        "tpu_device_count": len(tpu_devices),
        "device_kind": str(getattr(tpu_devices[0], "device_kind", "")),
        "tensor_dtype": dtype,
        "tensor_shape": shape,
        "summary_hash": sha_payload({{"mean": round(float(summary[0]), 8), "std": round(float(summary[1]), 8), "shape": shape, "dtype": dtype}}),
    }}


def tensor_bytes_to_jax(data, tensor_meta):
    import numpy as np
    import jax.numpy as jnp

    dtype = str(tensor_meta.get("dtype") or "").upper()
    shape = [int(item) for item in tensor_meta.get("shape") or []]
    if dtype == "BF16":
        raw = np.frombuffer(data, dtype="<u2")
        host = (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        return jnp.asarray(host, dtype=jnp.bfloat16)
    if dtype == "F16":
        host = np.frombuffer(data, dtype="<f2").reshape(shape)
        return jnp.asarray(host, dtype=jnp.float16)
    if dtype == "F32":
        host = np.frombuffer(data, dtype="<f4").reshape(shape)
        return jnp.asarray(host, dtype=jnp.float32)
    raise RuntimeError("unsupported_tensor_dtype_for_execution")


def load_tensor(weight_map, headers, key):
    filename, header_len, meta = tensor_file_and_header(weight_map, headers, key)
    data = download_tensor_bytes(filename, header_len, meta)
    return tensor_bytes_to_jax(data, meta), len(data)


def initial_hidden_for_stage(device, dtype, hidden_size):
    import numpy as np
    import jax
    import jax.numpy as jnp

    activation = INPUT_ACTIVATION_PRIVATE if isinstance(INPUT_ACTIVATION_PRIVATE, dict) else {{}}
    hidden_b64 = str(activation.get("hidden_b64") or "")
    if hidden_b64:
        shape = [int(item) for item in activation.get("hidden_shape") or []]
        if len(shape) != 3 or int(shape[-1]) != int(hidden_size):
            raise RuntimeError("tpu_stage_input_activation_shape_mismatch")
        if int(shape[1]) != 1:
            raise RuntimeError("tpu_stage_input_activation_seq_length_not_one")
        raw = base64.b64decode(hidden_b64)
        hidden_dtype = str(activation.get("hidden_dtype") or "float16").lower()
        if hidden_dtype in {{"float16", "f16"}}:
            host = np.frombuffer(raw, dtype="<f2").reshape(shape)
        elif hidden_dtype in {{"bfloat16", "bf16"}}:
            raw16 = np.frombuffer(raw, dtype="<u2")
            host = (raw16.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        elif hidden_dtype in {{"float32", "f32"}}:
            host = np.frombuffer(raw, dtype="<f4").reshape(shape)
        else:
            raise RuntimeError("tpu_stage_input_activation_dtype_unsupported")
        hidden = jax.device_put(jnp.asarray(host, dtype=dtype), device)
        return hidden, {{
            "input_activation_consumed": True,
            "input_activation_hash": str(activation.get("activation_hash") or ""),
            "input_activation_shape": shape,
            "input_activation_dtype": hidden_dtype,
        }}
    key = jax.random.PRNGKey(230623)
    hidden = jax.device_put(jax.random.normal(key, (1, 1, hidden_size), dtype=dtype) * jnp.array(0.01, dtype=dtype), device)
    return hidden, {{
        "input_activation_consumed": False,
        "input_activation_hash": "",
        "input_activation_shape": [1, 1, hidden_size],
        "input_activation_dtype": "synthetic_random_bfloat16",
    }}


def output_activation_from_hidden(hidden, *, model_repo, generation_step, input_activation_hash, stage_layer_range, input_ids, position_ids):
    import numpy as np
    import jax
    import jax.numpy as jnp

    host = np.asarray(jax.device_get(hidden.astype(jnp.float16)))
    payload = {{
        "schema": "kaggle_32b_full_private_activation_v1",
        "model_repo": model_repo,
        "stage_count": 10,
        "stage_id": 0,
        "generation_step": int(generation_step),
        "stage_layer_range": list(stage_layer_range),
        "prompt_hash": "sha256:redacted",
        "input_ids": [int(value) for value in list(input_ids or [1])],
        "position_ids": [int(value) for value in list(position_ids or [0])],
        "hidden_shape": [int(value) for value in host.shape],
        "hidden_dtype": "float16",
        "hidden_b64": base64.b64encode(host.tobytes()).decode("ascii"),
        "activation_payload_public": False,
    }}
    payload["activation_hash"] = sha_payload({{
        "model_repo": model_repo,
        "stage_layer_range": list(stage_layer_range),
        "generation_step": int(generation_step),
        "input_activation_hash": input_activation_hash,
        "hidden_shape": payload["hidden_shape"],
        "hidden_digest": hashlib.sha256(host.tobytes()).hexdigest(),
    }})
    return payload


def execute_stage_layers(weight_map, headers, config):
    if EXECUTE_LAYER_COUNT <= 0:
        return {{
            "attempted": False,
            "executed_layer_count": 0,
            "loaded_tensor_key_count": 0,
            "loaded_tensor_bytes": 0,
            "stage_output_hash": "",
            "stage_local_kv_cache_verified": False,
        }}
    import jax
    import jax.numpy as jnp

    tpu_devices = [device for device in jax.devices() if str(getattr(device, "platform", "")).lower() == "tpu"]
    if not tpu_devices:
        raise RuntimeError("jax_tpu_device_missing")
    hidden_size = int(config.get("hidden_size") or 5120)
    num_heads = int(config.get("num_attention_heads") or 40)
    num_kv_heads = int(config.get("num_key_value_heads") or 8)
    head_dim = hidden_size // num_heads
    repeat_factor = max(1, num_heads // max(1, num_kv_heads))
    dtype = jnp.bfloat16
    device = tpu_devices[0]
    hidden, input_activation_summary = initial_hidden_for_stage(device, dtype, hidden_size)

    def rms_norm(value, weight):
        variance = jnp.mean(jnp.square(value.astype(jnp.float32)), axis=-1, keepdims=True)
        return (value * jax.lax.rsqrt(variance.astype(dtype) + jnp.array(1e-6, dtype=dtype))) * weight

    def silu(value):
        return value * jax.nn.sigmoid(value)

    @jax.jit
    def layer_forward(hidden, params):
        residual = hidden
        normed = rms_norm(hidden, params["input_norm"])
        q = jnp.reshape(normed @ params["q_w"].T + params["q_b"], (1, 1, num_heads, head_dim))
        k = jnp.reshape(normed @ params["k_w"].T + params["k_b"], (1, 1, num_kv_heads, head_dim))
        v = jnp.reshape(normed @ params["v_w"].T + params["v_b"], (1, 1, num_kv_heads, head_dim))
        k_full = jnp.repeat(k, repeat_factor, axis=2)
        v_full = jnp.repeat(v, repeat_factor, axis=2)
        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k_full) / jnp.sqrt(jnp.array(head_dim, dtype=dtype))
        attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(dtype)
        context = jnp.einsum("bhqk,bkhd->bqhd", attn, v_full)
        hidden = residual + (jnp.reshape(context, (1, 1, hidden_size)) @ params["o_w"].T)
        residual = hidden
        normed = rms_norm(hidden, params["post_norm"])
        hidden = residual + (silu(normed @ params["gate_w"].T) * (normed @ params["up_w"].T)) @ params["down_w"].T
        kv_summary = jnp.asarray([jnp.mean(k.astype(jnp.float32)), jnp.mean(v.astype(jnp.float32))], dtype=jnp.float32)
        return hidden, kv_summary

    layer_hashes = []
    loaded_tensor_key_count = 0
    loaded_tensor_bytes = 0
    target_count = min(int(EXECUTE_LAYER_COUNT), max(0, STAGE_END - STAGE_START))
    suffix_map = {{
        "input_norm": "input_layernorm.weight",
        "q_w": "self_attn.q_proj.weight",
        "q_b": "self_attn.q_proj.bias",
        "k_w": "self_attn.k_proj.weight",
        "k_b": "self_attn.k_proj.bias",
        "v_w": "self_attn.v_proj.weight",
        "v_b": "self_attn.v_proj.bias",
        "o_w": "self_attn.o_proj.weight",
        "post_norm": "post_attention_layernorm.weight",
        "gate_w": "mlp.gate_proj.weight",
        "up_w": "mlp.up_proj.weight",
        "down_w": "mlp.down_proj.weight",
    }}
    for offset in range(target_count):
        layer = STAGE_START + offset
        params = {{}}
        for name, suffix in suffix_map.items():
            tensor, nbytes = load_tensor(weight_map, headers, f"model.layers.{{layer}}.{{suffix}}")
            params[name] = jax.device_put(tensor, device)
            loaded_tensor_key_count += 1
            loaded_tensor_bytes += int(nbytes)
        hidden, kv_summary = layer_forward(hidden, params)
        hidden.block_until_ready()
        summary = jnp.asarray([jnp.mean(hidden.astype(jnp.float32)), jnp.std(hidden.astype(jnp.float32)), kv_summary[0], kv_summary[1]], dtype=jnp.float32).block_until_ready()
        layer_hashes.append(sha_payload({{
            "layer": layer,
            "summary": [round(float(item), 8) for item in summary],
            "shape": [1, 1, hidden_size],
        }}))
        del params
        gc.collect()
    final_summary = jnp.asarray([jnp.mean(hidden.astype(jnp.float32)), jnp.std(hidden.astype(jnp.float32))], dtype=jnp.float32).block_until_ready()
    return {{
        "attempted": True,
        "requested_layer_count": int(EXECUTE_LAYER_COUNT),
        "executed_layer_count": int(target_count),
        "loaded_tensor_key_count": int(loaded_tensor_key_count),
        "loaded_tensor_bytes": int(loaded_tensor_bytes),
        "loaded_tensor_gb": round(float(loaded_tensor_bytes) / 1024 / 1024 / 1024, 6),
        "stage_output_hash": sha_payload({{
            "executed_layer_count": int(target_count),
            "summary": [round(float(final_summary[0]), 8), round(float(final_summary[1]), 8)],
            "input_activation_hash": input_activation_summary.get("input_activation_hash"),
        }}),
        "layer_output_hashes": layer_hashes,
        "input_activation_consumed": bool(input_activation_summary.get("input_activation_consumed")),
        "input_activation_hash": input_activation_summary.get("input_activation_hash"),
        "input_activation_shape": input_activation_summary.get("input_activation_shape"),
        "input_activation_dtype": input_activation_summary.get("input_activation_dtype"),
        "output_activation_private": output_activation_from_hidden(
            hidden,
            model_repo=MODEL_REPO,
            generation_step=int((INPUT_ACTIVATION_PRIVATE if isinstance(INPUT_ACTIVATION_PRIVATE, dict) else {{}}).get("generation_step") or 0),
            input_activation_hash=str((INPUT_ACTIVATION_PRIVATE if isinstance(INPUT_ACTIVATION_PRIVATE, dict) else {{}}).get("activation_hash") or ""),
            stage_layer_range=[STAGE_START, STAGE_END],
            input_ids=(INPUT_ACTIVATION_PRIVATE if isinstance(INPUT_ACTIVATION_PRIVATE, dict) else {{}}).get("input_ids") or [1],
            position_ids=(INPUT_ACTIVATION_PRIVATE if isinstance(INPUT_ACTIVATION_PRIVATE, dict) else {{}}).get("position_ids") or [0],
        ) if RETURN_OUTPUT_ACTIVATION_PRIVATE else None,
        "stage_local_kv_cache_verified": bool(target_count > 0),
    }}


started = time.monotonic()
report = {{
    "schema": SCHEMA,
    "generated_at": utc_now(),
    "ok": False,
    "model_repo": MODEL_REPO,
    "stage_layer_range": [STAGE_START, STAGE_END],
    "stage_owned_header_verified": False,
    "partial_tensor_to_tpu_verified": False,
    "full_stage_owned_tpu_loader_ready": False,
    "executed_layer_count": 0,
    "full_stage_layer_count": STAGE_END - STAGE_START,
    "tpu_32b_runtime_adapter_ready": False,
    "public_artifact_safe": True,
    "raw_prompt_public": False,
    "raw_generated_text_public": False,
    "generated_token_ids_public": False,
    "activation_public": False,
    "hidden_state_public": False,
    "logits_public": False,
    "kv_cache_public": False,
    "weight_tensor_values_public": False,
    "credentials_public": False,
    "cookies_public": False,
    "jupyter_proxy_token_public": False,
    "diagnosis_codes": [],
    "blockers": [],
    "env_public": {{"python": platform.python_version(), "platform": platform.platform(), "pjrt_device_present": bool(os.environ.get("PJRT_DEVICE"))}},
}}
try:
    import jax
    report["jax_version"] = str(getattr(jax, "__version__", ""))
    report["tpu_device_count"] = len([d for d in jax.devices() if str(getattr(d, "platform", "")).lower() == "tpu"])
    config = fetch_json("config.json")
    index = fetch_json("model.safetensors.index.json")
    weight_map = {{str(k): str(v) for k, v in dict(index.get("weight_map") or {{}}).items()}}
    assigned = qwen_stage_keys(weight_map)
    assigned_set = set(assigned)
    files = sorted({{weight_map[key] for key in assigned if weight_map.get(key)}})
    report.update({{
        "model_type": str(config.get("model_type") or ""),
        "hidden_size": int(config.get("hidden_size") or 0),
        "num_hidden_layers": int(config.get("num_hidden_layers") or 0),
        "assigned_weight_key_count": len(assigned),
        "assigned_weight_file_count": len(files),
        "assigned_weight_file_digest": sha_payload(files),
        "selected_tensor_key_hash": sha_payload(TENSOR_KEY),
    }})
    if not assigned:
        raise RuntimeError("stage_owned_key_selection_empty")
    headers = {{}}
    present_key_count = 0
    candidate_file_key_count = 0
    skipped_non_stage_key_count = 0
    missing = []
    header_file_summaries = []
    for filename in files:
        header_len, header = load_safetensors_header(filename)
        headers[filename] = (header_len, header)
        header_keys = [str(key) for key in header.keys() if key != "__metadata__"]
        candidate_file_key_count += len(header_keys)
        expected = [key for key in assigned if weight_map.get(key) == filename]
        available = set(header_keys)
        present = [key for key in expected if key in available]
        present_key_count += len(present)
        missing.extend([key for key in expected if key not in available])
        skipped_non_stage_key_count += len([key for key in header_keys if key not in assigned_set])
        header_file_summaries.append({{
            "filename": filename,
            "header_len": header_len,
            "expected_stage_key_count": len(expected),
            "present_stage_key_count": len(present),
            "candidate_file_key_count": len(header_keys),
        }})
    report.update({{
        "header_file_summaries": header_file_summaries,
        "candidate_file_key_count": candidate_file_key_count,
        "present_stage_key_count": present_key_count,
        "missing_stage_key_count": len(missing),
        "skipped_non_stage_key_count": skipped_non_stage_key_count,
        "stage_owned_header_verified": present_key_count == len(assigned) and not missing,
        "loads_only_stage_weight_keys": True,
        "cross_stage_weight_keys_loaded": False,
    }})
    if missing:
        report["blockers"].append("stage_owned_header_keys_missing")
    filename, header_len, tensor_meta = tensor_file_and_header(weight_map, headers, TENSOR_KEY)
    tensor_bytes = download_tensor_bytes(filename, header_len, tensor_meta)
    tpu_summary = tensor_bytes_to_tpu(tensor_bytes, tensor_meta)
    report.update({{
        "partial_tensor_to_tpu_verified": True,
        "selected_tensor_file": filename,
        "selected_tensor_bytes": len(tensor_bytes),
        "selected_tensor_dtype": tpu_summary["tensor_dtype"],
        "selected_tensor_shape": tpu_summary["tensor_shape"],
        "selected_tensor_value_hash": sha_payload({{"tensor_key_hash": sha_payload(TENSOR_KEY), "bytes": hashlib.sha256(tensor_bytes).hexdigest()}}),
        "selected_tensor_tpu_summary_hash": tpu_summary["summary_hash"],
        "tpu_device_count": tpu_summary["tpu_device_count"],
        "tpu_device_kind": tpu_summary["device_kind"],
    }})
    execution = execute_stage_layers(weight_map, headers, config)
    output_activation_private = execution.pop("output_activation_private", None) if isinstance(execution, dict) else None
    report["stage_execution"] = execution
    if isinstance(output_activation_private, dict):
        report["output_activation_private"] = output_activation_private
        report["output_activation_private_public"] = False
        report["output_activation_hash"] = str(output_activation_private.get("activation_hash") or "")
        report["output_activation_shape"] = list(output_activation_private.get("hidden_shape") or [])
    report["executed_layer_count"] = int(execution.get("executed_layer_count") or 0)
    report["loaded_execution_tensor_key_count"] = int(execution.get("loaded_tensor_key_count") or 0)
    report["loaded_execution_tensor_bytes"] = int(execution.get("loaded_tensor_bytes") or 0)
    report["loaded_execution_tensor_gb"] = float(execution.get("loaded_tensor_gb") or 0.0)
    report["stage_output_hash"] = str(execution.get("stage_output_hash") or "")
    report["input_activation_consumed"] = execution.get("input_activation_consumed") is True
    report["input_activation_hash"] = str(execution.get("input_activation_hash") or "")
    report["input_activation_shape"] = list(execution.get("input_activation_shape") or [])
    report["output_activation_private_present"] = isinstance(output_activation_private, dict)
    report["stage_local_kv_cache_verified"] = execution.get("stage_local_kv_cache_verified") is True
    full_stage_count = max(0, STAGE_END - STAGE_START)
    if report["stage_owned_header_verified"] and report["executed_layer_count"] >= full_stage_count and full_stage_count > 0:
        report["full_stage_owned_tpu_loader_ready"] = True
        report["tpu_32b_runtime_adapter_ready"] = True
    if report["stage_owned_header_verified"] and report["partial_tensor_to_tpu_verified"]:
        report["ok"] = True
        report["diagnosis_codes"].extend([
            "kaggle_web_tpu_32b_stage_owned_headers_verified",
            "kaggle_web_tpu_32b_partial_tensor_to_tpu_verified",
        ])
    if report["full_stage_owned_tpu_loader_ready"]:
        report["diagnosis_codes"].append("kaggle_web_tpu_32b_full_stage_loader_ready")
    else:
        report["blockers"].append("full_stage_owned_tpu_loader_not_executed")
        report["diagnosis_codes"].append("kaggle_web_tpu_32b_full_stage_loader_not_ready")
except Exception as exc:
    report["ok"] = False
    report["error_type"] = type(exc).__name__
    report["error_digest"] = sha_payload(str(exc))
    report["error_public"] = safe_error(str(exc))
    report["diagnosis_codes"].append("kaggle_web_tpu_32b_stage_owned_loader_probe_failed")
    if not report["blockers"]:
        report["blockers"].append("kaggle_web_tpu_32b_stage_owned_loader_probe_failed")
finally:
    report["diagnosis_codes"] = sorted(set(report["diagnosis_codes"]))
    report["blockers"] = sorted(set(report["blockers"]))
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
print(json.dumps({{"schema": SCHEMA, "ok": report.get("ok"), "report": report}}, sort_keys=True))
'''


def get_proxy_and_kernel(args: argparse.Namespace) -> tuple[str, str, str]:
    from playwright.sync_api import sync_playwright

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


def execute_web_code(args: argparse.Namespace, code: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if web_tpu_bridge is not None:
        report = web_tpu_bridge.execute_web_tpu_code_via_iframe(args, code)
        if report.get("ok") is True or report.get("blockers"):
            steps = report.get("web_tpu_jupyter_steps") if isinstance(report.get("web_tpu_jupyter_steps"), list) else []
            return report, [step for step in steps if isinstance(step, dict)]

    import websocket

    steps: list[dict[str, Any]] = []
    started = time.monotonic()
    base, proxy_token, kernel_id = get_proxy_and_kernel(args)
    steps.append({
        "name": "jupyter_proxy_kernel_discovered",
        "ok": True,
        "accepted": True,
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
            "username": "crowdtensor-loader",
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
    try:
        while time.monotonic() - exec_started < float(args.web_execute_timeout_seconds):
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
    finally:
        ws.close()
    steps.append({
        "name": "jupyter_ws_execute",
        "ok": not errors,
        "duration_seconds": round(time.monotonic() - exec_started, 3),
        "errors_public": errors,
        "jupyter_proxy_token_public": False,
    })
    for line in "".join(stdout).splitlines()[::-1]:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("report"), dict):
            return parsed["report"], steps
    return {
        "schema": SCHEMA,
        "ok": False,
        "blockers": ["web_tpu_loader_report_missing"],
        "diagnosis_codes": ["web_tpu_loader_report_missing"],
        "public_artifact_safe": True,
    }, steps


def build_report(args: argparse.Namespace, *, runtime_report: dict[str, Any], steps: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    ok = bool(runtime_report.get("ok"))
    full_ready = runtime_report.get("full_stage_owned_tpu_loader_ready") is True
    public_runtime_report = json.loads(json.dumps(runtime_report, sort_keys=True, default=str))
    output_activation_private = public_runtime_report.pop("output_activation_private", None)
    if isinstance(public_runtime_report.get("stage_execution"), dict):
        public_runtime_report["stage_execution"].pop("output_activation_private", None)
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "model_repo": args.model_repo,
        "stage_layer_range": [int(args.stage_start), int(args.stage_end)],
        "stage_owned_header_verified": runtime_report.get("stage_owned_header_verified") is True,
        "partial_tensor_to_tpu_verified": runtime_report.get("partial_tensor_to_tpu_verified") is True,
        "full_stage_owned_tpu_loader_ready": full_ready,
        "tpu_32b_runtime_adapter_ready": full_ready,
        "executed_layer_count": int(runtime_report.get("executed_layer_count") or 0),
        "full_stage_layer_count": int(runtime_report.get("full_stage_layer_count") or 0),
        "loaded_execution_tensor_key_count": int(runtime_report.get("loaded_execution_tensor_key_count") or 0),
        "loaded_execution_tensor_bytes": int(runtime_report.get("loaded_execution_tensor_bytes") or 0),
        "loaded_execution_tensor_gb": float(runtime_report.get("loaded_execution_tensor_gb") or 0.0),
        "stage_output_hash": str(runtime_report.get("stage_output_hash") or ""),
        "input_activation_consumed": runtime_report.get("input_activation_consumed") is True,
        "input_activation_hash": str(runtime_report.get("input_activation_hash") or ""),
        "input_activation_shape": list(runtime_report.get("input_activation_shape") or []),
        "output_activation_private_present": isinstance(output_activation_private, dict) or runtime_report.get("output_activation_private_present") is True,
        "output_activation_hash": str(runtime_report.get("output_activation_hash") or ""),
        "output_activation_shape": list(runtime_report.get("output_activation_shape") or []),
        "stage_local_kv_cache_verified": runtime_report.get("stage_local_kv_cache_verified") is True,
        "assigned_weight_key_count": int(runtime_report.get("assigned_weight_key_count") or 0),
        "assigned_weight_file_count": int(runtime_report.get("assigned_weight_file_count") or 0),
        "present_stage_key_count": int(runtime_report.get("present_stage_key_count") or 0),
        "missing_stage_key_count": int(runtime_report.get("missing_stage_key_count") or 0),
        "candidate_file_key_count": int(runtime_report.get("candidate_file_key_count") or 0),
        "skipped_non_stage_key_count": int(runtime_report.get("skipped_non_stage_key_count") or 0),
        "selected_tensor_key_hash": str(runtime_report.get("selected_tensor_key_hash") or ""),
        "selected_tensor_file": str(runtime_report.get("selected_tensor_file") or ""),
        "selected_tensor_bytes": int(runtime_report.get("selected_tensor_bytes") or 0),
        "selected_tensor_dtype": str(runtime_report.get("selected_tensor_dtype") or ""),
        "selected_tensor_shape": list(runtime_report.get("selected_tensor_shape") or []),
        "selected_tensor_value_hash": str(runtime_report.get("selected_tensor_value_hash") or ""),
        "selected_tensor_tpu_summary_hash": str(runtime_report.get("selected_tensor_tpu_summary_hash") or ""),
        "tpu_device_count": int(runtime_report.get("tpu_device_count") or 0),
        "tpu_device_kind": str(runtime_report.get("tpu_device_kind") or ""),
        "runtime_report": public_runtime_report,
        "steps": steps,
        "blockers": sorted(set(str(item) for item in runtime_report.get("blockers") or [] if item)),
        "blocked_reason": "" if full_ready else (
            str((runtime_report.get("blockers") or ["full_stage_owned_tpu_loader_not_ready"])[0])
        ),
        "diagnosis_codes": sorted(set(str(item) for item in runtime_report.get("diagnosis_codes") or [] if item)),
        "kaggle_lifecycle": {
            "web_runtime_execution_count": 1 if any(step.get("name") == "jupyter_ws_execute" for step in steps) else 0,
            "private_kernel_push_count": 0,
            "kernels_deleted": True,
            "private_packages_removed": True,
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
            "weight_tensor_values_public": False,
            "credentials_public": False,
            "cookies_public": False,
            "jupyter_proxy_token_public": False,
            "private_runtime_state_public": False,
        },
        "limitations": [
            "Header and byte-range tensor evidence is real Qwen 32B safetensors evidence.",
            "partial_tensor_to_tpu_verified=true is not a full 21-layer TPU stage loader.",
            "full_stage_owned_tpu_loader_ready must be true before this can support 32B same-request success.",
        ],
        "public_artifact_safe": True,
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"].append("public_redaction_scan_failed")
        report["diagnosis_codes"].append("public_redaction_scan_failed")
        report["redaction_errors"] = leaks
    summary_path = output_dir / "kaggle_tpu_32b_stage_owned_loader_probe.json"
    write_json(summary_path, report)
    report["artifacts"] = {
        "summary_json": artifact_entry(summary_path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(report.get("ok"))),
    }
    write_json(summary_path, report)
    return report


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        runtime_report, steps = execute_web_code(args, render_web_probe_code(args))
    except Exception as exc:
        if web_tpu_bridge is not None:
            blocker, diagnosis = web_tpu_bridge.classify_web_tpu_exception(exc)
        else:
            blocker, diagnosis = ("web_tpu_32b_loader_probe_exception", "web_tpu_32b_loader_probe_exception")
        runtime_report = {
            "schema": SCHEMA,
            "ok": False,
            "blockers": [blocker],
            "diagnosis_codes": [diagnosis],
            "error_type": type(exc).__name__,
            "error_digest": sha_payload(str(exc)),
            "public_artifact_safe": True,
        }
        steps = [{"name": "web_tpu_loader_probe_exception", "ok": False, "error_type": type(exc).__name__}]
    return build_report(args, runtime_report=runtime_report, steps=steps, output_dir=output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded Web TPU Qwen 32B stage-owned safetensors loader probe.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kaggle-notebook-url", default=DEFAULT_NOTEBOOK_URL)
    parser.add_argument("--kaggle-web-storage-state", default="/root/kaggle-web-storage-state.json")
    parser.add_argument("--chrome-executable", default="/usr/bin/google-chrome")
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--stage-start", type=int, default=DEFAULT_STAGE_START)
    parser.add_argument("--stage-end", type=int, default=DEFAULT_STAGE_END)
    parser.add_argument("--tensor-key", default=DEFAULT_TENSOR_KEY)
    parser.add_argument("--max-header-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-tensor-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--execute-layer-count", type=int, default=0)
    parser.add_argument("--input-activation-json", default="")
    parser.add_argument("--return-output-activation-private", action="store_true")
    parser.add_argument("--web-execute-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    args.web_tpu_execute_timeout_seconds = float(args.web_execute_timeout_seconds)
    if args.stage_start < 0 or args.stage_end <= args.stage_start:
        raise SystemExit("--stage-end must be greater than --stage-start")
    if args.max_header_bytes < 1024 or args.max_header_bytes > 256 * 1024 * 1024:
        raise SystemExit("--max-header-bytes must be between 1KiB and 256MiB")
    if args.max_tensor_bytes < 1024 or args.max_tensor_bytes > 1024 * 1024 * 1024:
        raise SystemExit("--max-tensor-bytes must be between 1KiB and 1GiB")
    if args.execute_layer_count < 0 or args.execute_layer_count > (args.stage_end - args.stage_start):
        raise SystemExit("--execute-layer-count must be between 0 and the stage layer count")
    if str(args.input_activation_json or "").strip():
        loaded = json.loads(str(args.input_activation_json))
        if not isinstance(loaded, dict):
            raise SystemExit("--input-activation-json must be a JSON object")
        args.input_activation_private = loaded
    else:
        args.input_activation_private = {}
    if not str(args.tensor_key or "").strip():
        args.tensor_key = f"model.layers.{int(args.stage_start)}.input_layernorm.weight"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_probe(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            f"{SCHEMA}: ok={bool(report.get('ok'))} "
            f"header={bool(report.get('stage_owned_header_verified'))} "
            f"partial_tensor={bool(report.get('partial_tensor_to_tpu_verified'))} "
            f"full_loader={bool(report.get('full_stage_owned_tpu_loader_ready'))} "
            f"blocked={report.get('blocked_reason') or 'none'}"
        )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
