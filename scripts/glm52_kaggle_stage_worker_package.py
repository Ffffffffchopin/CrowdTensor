#!/usr/bin/env python3
"""Render private Kaggle worker package skeletons for GLM 5.2 stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "glm52_kaggle_stage_worker_package_v1"
MODEL_ID = "zai-org/GLM-5.2"
DEFAULT_OUTPUT_DIR = "dist/glm52-kaggle-stage-worker-package"
REQUIRED_PROVIDERS = ["kaggle_cuda", "kaggle_jax_tpu", "kaggle_cpu"]
RUNTIME_KIND_VALUE_OP = "value_op"
RUNTIME_KIND_FULL_PREFIX_STAGE_DECODE = "full_prefix_stage_decode"
RUNTIME_KINDS = [RUNTIME_KIND_VALUE_OP, RUNTIME_KIND_FULL_PREFIX_STAGE_DECODE]
FULL_PREFIX_PROBE_MODE_DEFAULT = "default"
FULL_PREFIX_PROBE_MODE_FULL_STAGE = "full-stage"
FULL_PREFIX_PROBE_MODES = [FULL_PREFIX_PROBE_MODE_DEFAULT, FULL_PREFIX_PROBE_MODE_FULL_STAGE]
FULL_PREFIX_RUNTIME_BUNDLE = [
    "glm52_full_prefix_stage_decode_probe.py",
    "glm52_attention_projection_probe.py",
    "glm52_attention_single_token_probe.py",
    "glm52_dsa_indexer_probe.py",
    "glm52_dsa_masked_layer_decode_probe.py",
    "glm52_kv_cache_decode_probe.py",
    "glm52_lm_head_token_probe.py",
    "glm52_pack_quantized_dequant_probe.py",
    "glm52_pack_quantized_expert_mlp_probe.py",
    "glm52_pack_quantized_moe_mlp_probe.py",
    "glm52_pack_quantized_router_gather_probe.py",
]
GLM52_KNOWN_FULL_INDEXER_LAYERS = [0, 1, 2, *range(6, 78, 4)]
GLM52_FIRST_DSA_STAGE_LAYER = 3
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Authorization:",
    "Cookie:",
    "Set-Cookie",
    "kaggle-cookies",
    "kaggle-web-storage-state",
    "token=",
    "runtime_proxy",
    "jupyter-proxy",
    '"prompt":',
    '"raw_prompt":',
    '"generated_text":',
    '"raw_generated_text":',
    '"generated_token_ids":',
    '"activation":',
    '"hidden_state":',
    '"logits":',
    '"kv_cache":',
    '"weight_tensor_values":',
    '"safetensors_header_payload":',
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    loaded = json.loads(p.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def sha_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _hash_ok(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) >= 71


def safe_slug(value: str) -> str:
    cleaned = []
    last_dash = False
    for char in str(value or "").lower():
        if char.isalnum():
            cleaned.append(char)
            last_dash = False
        elif not last_dash:
            cleaned.append("-")
            last_dash = True
    return "".join(cleaned).strip("-") or "stage"


def parse_provider_owner_map(value: str) -> dict[str, str]:
    owners: dict[str, str] = {}
    for raw_item in str(value or "").split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            continue
        provider, owner = item.split("=", 1)
        provider = provider.strip()
        owner = owner.strip()
        if provider and owner:
            owners[provider] = owner
    return owners


def owner_for_provider(provider: str, *, default_owner: str, owner_map: dict[str, str]) -> str:
    return str(owner_map.get(provider) or default_owner)


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def full_prefix_runtime_bundle_payload() -> dict[str, str]:
    source_dir = Path(__file__).resolve().parent
    payload: dict[str, str] = {}
    for filename in FULL_PREFIX_RUNTIME_BUNDLE:
        source = source_dir / filename
        if source.is_file():
            payload[f"scripts/{filename}"] = source.read_text(encoding="utf-8")
    payload["scripts/__init__.py"] = ""
    return payload


def default_full_prefix_probe_layer_range(stage: dict[str, Any]) -> list[int]:
    layer_range = _list(stage.get("stage_layer_range"))
    if len(layer_range) != 2:
        return []
    start = _int(layer_range[0])
    end = _int(layer_range[1])
    min_start = max(start, GLM52_FIRST_DSA_STAGE_LAYER)
    for layer_id in GLM52_KNOWN_FULL_INDEXER_LAYERS:
        if layer_id >= min_start and layer_id + 2 <= end:
            return [layer_id, layer_id + 2]
    if end - min_start >= 2:
        return [min_start, min_start + 2]
    if end - start >= 2:
        return [start, start + 2]
    return []


def select_full_prefix_probe_layer_range(stage: dict[str, Any], *, mode: str = FULL_PREFIX_PROBE_MODE_DEFAULT) -> list[int]:
    layer_range = _list(stage.get("stage_layer_range"))
    if len(layer_range) != 2:
        return []
    if mode == FULL_PREFIX_PROBE_MODE_FULL_STAGE:
        start = _int(layer_range[0])
        end = _int(layer_range[1])
        return [start, end] if end > start else []
    return default_full_prefix_probe_layer_range(stage)


def safety_flags() -> dict[str, bool]:
    return {
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
    }


def render_kernel(
    stage: dict[str, Any],
    *,
    coordinator_request_id_hash: str = "",
    runtime_kind: str = RUNTIME_KIND_VALUE_OP,
    full_prefix_timeout_seconds: int = 3600,
) -> str:
    provider = str(stage.get("provider") or "")
    stage_id = _int(stage.get("stage_id"))
    layer_range = _list(stage.get("stage_layer_range"))
    compatible_weight_repo = str(stage.get("compatible_weight_repo") or "")
    runtime_adapter = str(stage.get("runtime_adapter") or "")
    full_prefix_probe_layer_range = _list(stage.get("full_prefix_probe_layer_range"))
    if len(full_prefix_probe_layer_range) != 2:
        full_prefix_probe_layer_range = default_full_prefix_probe_layer_range(stage)
    embedded_bundle = (
        full_prefix_runtime_bundle_payload()
        if runtime_kind == RUNTIME_KIND_FULL_PREFIX_STAGE_DECODE
        else {}
    )
    payload = {
        "provider": provider,
        "stage_id": stage_id,
        "stage_count": _int(stage.get("stage_count"), 3),
        "stage_layer_range": layer_range,
        "model_id": MODEL_ID,
        "compatible_weight_repo": compatible_weight_repo,
        "runtime_adapter": runtime_adapter,
        "stage_runtime_package_kind": runtime_kind,
        "full_prefix_probe_layer_range": full_prefix_probe_layer_range,
        "full_prefix_timeout_seconds": max(1, int(full_prefix_timeout_seconds or 3600)),
    }
    default_request_hash = coordinator_request_id_hash if _hash_ok(coordinator_request_id_hash) else ""
    return f'''#!/usr/bin/env python3
"""Private Kaggle GLM 5.2 stage runtime value-op worker.

This worker verifies stage-owned GLM 5.2 AWQ weight-byte loading and executes a
small provider-local op. It is stage runtime evidence, not full decode success.
The Coordinator same-request proof must be assembled separately.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import struct
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


STAGE = {json.dumps(payload, sort_keys=True)}
MODEL_REPO = STAGE["compatible_weight_repo"] or "cyankiwi/GLM-5.2-AWQ-INT4"
DEFAULT_COORDINATOR_REQUEST_HASH = {json.dumps(default_request_hash)}
STAGE_RUNTIME_PACKAGE_KIND = STAGE.get("stage_runtime_package_kind") or "value_op"
EMBEDDED_FULL_PREFIX_RUNTIME_BUNDLE = {json.dumps(embedded_bundle, sort_keys=True)}
CT_GLM52_PRIVATE_RUNTIME_ENV_INLINE = {{}}


def load_private_runtime_env_file() -> None:
    loaded = {{}}
    if isinstance(CT_GLM52_PRIVATE_RUNTIME_ENV_INLINE, dict):
        loaded.update(CT_GLM52_PRIVATE_RUNTIME_ENV_INLINE)
    raw_path = os.environ.get("CT_GLM52_PRIVATE_RUNTIME_ENV_PATH", "ct_glm52_private_runtime_env.json")
    path = Path(str(raw_path))
    if path.is_file():
        try:
            loaded_from_file = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded_from_file, dict):
                loaded.update(loaded_from_file)
        except Exception:
            pass
    if not loaded:
        return
    for key, value in loaded.items():
        key_text = str(key)
        value_text = str(value) if value is not None else ""
        if (key_text.startswith("CT_GLM52_") or key_text in {{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}}) and value_text and not os.environ.get(key_text):
            os.environ[key_text] = value_text


load_private_runtime_env_file()
if STAGE_RUNTIME_PACKAGE_KIND == "full_prefix_stage_decode":
    os.environ.setdefault("CT_GLM52_FULL_PREFIX_TIMEOUT_SECONDS", str(STAGE.get("full_prefix_timeout_seconds") or 3600))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()


def sha_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha_json(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return sha_text(encoded)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")


def fetch_url_bytes(url: str, *, headers: dict | None = None, timeout: float = 90.0, read_limit: int | None = None) -> bytes:
    last_error = None
    retries = max(1, int(os.environ.get("CT_GLM52_HF_FETCH_RETRIES", "3")))
    for attempt in range(retries):
        request_headers = {{"User-Agent": "crowdtensor-glm52-kaggle-stage-worker/1", **(headers or {{}})}}
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token and "Authorization" not in request_headers:
            request_headers["Authorization"] = "Bearer " + str(hf_token)
        request = urllib.request.Request(
            url,
            headers=request_headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if read_limit is None:
                    return response.read()
                return response.read(read_limit)
        except OSError as exc:
            last_error = exc
            if attempt + 1 >= retries:
                break
            time.sleep(min(8.0, 1.0 + float(attempt)))
    raise last_error if last_error is not None else RuntimeError("fetch_url_bytes_failed")


def fetch_hf_json(repo: str, filename: str) -> dict:
    quoted = urllib.parse.quote(filename)
    raw = fetch_url_bytes(f"https://huggingface.co/{{repo}}/resolve/main/{{quoted}}")
    loaded = json.loads(raw.decode("utf-8"))
    return loaded if isinstance(loaded, dict) else {{}}


def read_hf_range(repo: str, filename: str, start: int, end: int, *, max_bytes: int) -> bytes:
    quoted = urllib.parse.quote(filename)
    raw = fetch_url_bytes(
        f"https://huggingface.co/{{repo}}/resolve/main/{{quoted}}",
        headers={{"Range": f"bytes={{int(start)}}-{{int(end)}}"}},
        read_limit=int(max_bytes) + 1,
    )
    if len(raw) > int(max_bytes):
        raise RuntimeError("hf_range_response_exceeded_budget")
    return raw


def _int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _list(value) -> list:
    return value if isinstance(value, list) else []


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {{}}


def tensor_nbytes(item: dict) -> int:
    offsets = _list(item.get("data_offsets"))
    if len(offsets) != 2:
        return 0
    return max(0, _int(offsets[1]) - _int(offsets[0]))


def load_safetensors_header(repo: str, filename: str, *, max_header_bytes: int) -> tuple[int, dict]:
    prefix = read_hf_range(repo, filename, 0, 7, max_bytes=8)
    if len(prefix) != 8:
        raise RuntimeError("safetensors_header_prefix_missing")
    header_len = struct.unpack("<Q", prefix)[0]
    if header_len <= 0 or header_len > int(max_header_bytes):
        raise RuntimeError("safetensors_header_length_out_of_budget")
    raw_header = read_hf_range(repo, filename, 8, 8 + int(header_len) - 1, max_bytes=int(header_len))
    if len(raw_header) != int(header_len):
        raise RuntimeError("safetensors_header_truncated")
    loaded = json.loads(raw_header.decode("utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError("safetensors_header_not_object")
    return int(header_len), loaded


def stage_prefixes(stage_id: int, stage_count: int, layer_range: list[int]) -> list[str]:
    start, end = int(layer_range[0]), int(layer_range[1])
    prefixes = [f"model.layers.{{layer_id}}." for layer_id in range(start, end)]
    if stage_id == 0:
        prefixes = ["model.embed_tokens.", *prefixes]
    if stage_id == stage_count - 1:
        prefixes = [*prefixes, "model.norm.", "lm_head."]
    return prefixes


def preferred_stage_tensor_keys(stage_id: int, stage_count: int, layer_range: list[int]) -> list[str]:
    start = int(layer_range[0])
    end = int(layer_range[1])
    keys = []
    if end > start:
        keys.extend([
            f"model.layers.{{start}}.input_layernorm.weight",
            f"model.layers.{{start}}.post_attention_layernorm.weight",
        ])
    if stage_id == stage_count - 1:
        keys.append("model.norm.weight")
    return keys


def load_stage_tensor_value(repo: str, key: str, filename: str, *, max_header_bytes: int, max_tensor_bytes: int) -> dict:
    header_len, header = load_safetensors_header(repo, filename, max_header_bytes=max_header_bytes)
    item = _dict(header.get(key))
    nbytes = tensor_nbytes(item)
    if nbytes <= 0 or nbytes > int(max_tensor_bytes):
        raise RuntimeError("preferred_stage_value_tensor_not_within_budget")
    offsets = _list(item.get("data_offsets"))
    absolute_start = 8 + int(header_len) + _int(offsets[0])
    absolute_end = 8 + int(header_len) + _int(offsets[1]) - 1
    raw_value = read_hf_range(repo, filename, absolute_start, absolute_end, max_bytes=max_tensor_bytes)
    if len(raw_value) != int(nbytes):
        raise RuntimeError("stage_value_tensor_read_size_mismatch")
    return {{
        "filename": filename,
        "header_len": header_len,
        "selected_tensor": {{
            "key_digest": sha_json(key),
            "filename": filename,
            "dtype": str(item.get("dtype") or ""),
            "rank": len(_list(item.get("shape"))),
            "shape_digest": sha_json(_list(item.get("shape"))),
            "tensor_nbytes": nbytes,
            "data_offsets_digest": sha_json([_int(offsets[0]), _int(offsets[1])]),
        }},
        "raw_value": raw_value,
    }}


def select_stage_tensor(repo: str, *, max_header_files: int, max_header_bytes: int, max_tensor_bytes: int) -> dict:
    config = fetch_hf_json(repo, "config.json")
    index = fetch_hf_json(repo, "model.safetensors.index.json")
    weight_map = {{
        str(key): str(value).split("/")[-1]
        for key, value in _dict(index.get("weight_map")).items()
        if str(key or "").strip() and str(value or "").strip()
    }}
    layer_range = [int(STAGE["stage_layer_range"][0]), int(STAGE["stage_layer_range"][1])]
    stage_count = int(os.environ.get("CT_GLM52_STAGE_COUNT", str(STAGE.get("stage_count") or 3)))
    prefixes = stage_prefixes(int(STAGE["stage_id"]), stage_count, layer_range)
    assigned_keys = sorted(key for key in weight_map if any(key.startswith(prefix) for prefix in prefixes))
    assigned_files = sorted({{weight_map[key] for key in assigned_keys if weight_map.get(key)}})
    for key in preferred_stage_tensor_keys(int(STAGE["stage_id"]), stage_count, layer_range):
        filename = weight_map.get(key, "")
        if not filename or key not in assigned_keys:
            continue
        try:
            value = load_stage_tensor_value(
                repo,
                key,
                filename,
                max_header_bytes=max_header_bytes,
                max_tensor_bytes=max_tensor_bytes,
            )
            return {{
                "assigned_weight_key_count": len(assigned_keys),
                "assigned_weight_file_count": len(assigned_files),
                "header_file_count": 1,
                "model_type": str(config.get("model_type") or ""),
                "selected_tensor": value["selected_tensor"],
                "weight_value_byte_count": len(value["raw_value"]),
                "weight_value_sha256": sha_bytes(value["raw_value"]),
                "raw_value": value["raw_value"],
            }}
        except Exception:
            pass
    headers_by_file = {{}}
    for filename in assigned_files[:max_header_files]:
        headers_by_file[filename] = load_safetensors_header(repo, filename, max_header_bytes=max_header_bytes)
    candidates = []
    for key in assigned_keys:
        filename = weight_map.get(key, "")
        header_tuple = headers_by_file.get(filename)
        if not header_tuple:
            continue
        header_len, header = header_tuple
        item = _dict(header.get(key))
        nbytes = tensor_nbytes(item)
        if nbytes <= 0 or nbytes > int(max_tensor_bytes):
            continue
        dtype = str(item.get("dtype") or "")
        priority = 0
        if any(fragment in key for fragment in ["qzeros", "scales", "g_idx"]):
            priority -= 100
        if dtype in {{"I32", "I64"}}:
            priority -= 10
        offsets = _list(item.get("data_offsets"))
        candidates.append({{
            "key": key,
            "filename": filename,
            "header_len": header_len,
            "dtype": dtype,
            "shape": _list(item.get("shape")),
            "offset_start": _int(offsets[0]) if len(offsets) == 2 else 0,
            "offset_end": _int(offsets[1]) if len(offsets) == 2 else 0,
            "tensor_nbytes": nbytes,
            "priority": priority,
        }})
    if not candidates:
        raise RuntimeError("stage_value_tensor_not_found_within_budget")
    candidates.sort(key=lambda item: (int(item["priority"]), int(item["tensor_nbytes"]), str(item["key"])))
    selected = candidates[0]
    absolute_start = 8 + int(selected["header_len"]) + int(selected["offset_start"])
    absolute_end = 8 + int(selected["header_len"]) + int(selected["offset_end"]) - 1
    raw_value = read_hf_range(repo, str(selected["filename"]), absolute_start, absolute_end, max_bytes=max_tensor_bytes)
    if len(raw_value) != int(selected["tensor_nbytes"]):
        raise RuntimeError("stage_value_tensor_read_size_mismatch")
    return {{
        "assigned_weight_key_count": len(assigned_keys),
        "assigned_weight_file_count": len(assigned_files),
        "header_file_count": len(headers_by_file),
        "model_type": str(config.get("model_type") or ""),
        "selected_tensor": {{
            "key_digest": sha_json(selected["key"]),
            "filename": selected["filename"],
            "dtype": selected["dtype"],
            "rank": len(_list(selected["shape"])),
            "shape_digest": sha_json(selected["shape"]),
            "tensor_nbytes": selected["tensor_nbytes"],
            "data_offsets_digest": sha_json([selected["offset_start"], selected["offset_end"]]),
        }},
        "weight_value_byte_count": len(raw_value),
        "weight_value_sha256": sha_bytes(raw_value),
        "raw_value": raw_value,
    }}


def provider_runtime_op(raw_value: bytes) -> dict:
    provider = STAGE["provider"]
    sample = list(raw_value[: min(len(raw_value), 4096)])
    if provider == "kaggle_cpu":
        total = sum(sample)
        return {{"provider_runtime_verified": True, "provider_device_count": 1, "provider_op_hash": sha_json(["cpu", total, len(raw_value)])}}
    if provider == "kaggle_cuda":
        errors = []
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append("torch_cuda_device_not_available")
            else:
                tensor = torch.tensor(sample, dtype=torch.uint8, device="cuda")
                total = int(tensor.to(dtype=torch.int64).sum().item())
                return {{
                    "provider_runtime_verified": True,
                    "provider_device_count": int(torch.cuda.device_count()),
                    "provider_op_hash": sha_json(["cuda_torch", total, len(raw_value), torch.cuda.get_device_name(0)]),
                }}
        except Exception as exc:
            errors.append("torch_cuda_provider_op_failed:" + type(exc).__name__)
        try:
            import cupy as cp
            device_count = int(cp.cuda.runtime.getDeviceCount())
            if device_count <= 0:
                errors.append("cupy_cuda_device_not_available")
            else:
                array = cp.asarray(sample, dtype=cp.uint8)
                total = int(cp.sum(array.astype(cp.uint64)).get())
                device_name = cp.cuda.runtime.getDeviceProperties(0).get("name", b"")
                if isinstance(device_name, bytes):
                    device_name = device_name.decode("utf-8", errors="replace")
                return {{
                    "provider_runtime_verified": True,
                    "provider_device_count": device_count,
                    "provider_op_hash": sha_json(["cuda_cupy", total, len(raw_value), str(device_name)]),
                }}
        except Exception as exc:
            errors.append("cupy_cuda_provider_op_failed:" + type(exc).__name__)
        try:
            import numpy as np
            from numba import cuda
            if not cuda.is_available():
                errors.append("numba_cuda_device_not_available")
            else:
                host = np.asarray(sample, dtype=np.uint8)
                device = cuda.to_device(host)
                copied = device.copy_to_host()
                total = int(copied.astype(np.uint64).sum())
                return {{
                    "provider_runtime_verified": True,
                    "provider_device_count": len(list(cuda.gpus)),
                    "provider_op_hash": sha_json(["cuda_numba", total, len(raw_value), len(list(cuda.gpus))]),
                }}
        except Exception as exc:
            errors.append("numba_cuda_provider_op_failed:" + type(exc).__name__)
        try:
            import jax
            import jax.numpy as jnp
            devices = jax.devices("gpu")
            if not devices:
                errors.append("jax_gpu_device_not_available")
            else:
                array = jax.device_put(jnp.asarray(sample, dtype=jnp.uint8), devices[0])
                total = int(jnp.asarray(array, dtype=jnp.uint32).sum().item())
                return {{
                    "provider_runtime_verified": True,
                    "provider_device_count": len(devices),
                    "provider_op_hash": sha_json(["cuda_jax_gpu", total, len(raw_value), len(devices)]),
                }}
        except Exception as exc:
            errors.append("jax_gpu_provider_op_failed:" + type(exc).__name__)
        return {{"provider_runtime_verified": False, "blocker": "cuda_provider_op_failed", "error_digest": sha_json(errors[-8:])}}
    if provider == "kaggle_jax_tpu":
        try:
            import jax
            import jax.numpy as jnp
            devices = jax.devices("tpu")
            if not devices:
                return {{"provider_runtime_verified": False, "blocker": "jax_tpu_device_not_available"}}
            array = jax.device_put(jnp.asarray(sample, dtype=jnp.uint8), devices[0])
            total = int(jnp.asarray(array, dtype=jnp.uint32).sum().item())
            return {{
                "provider_runtime_verified": True,
                "provider_device_count": len(devices),
                "provider_op_hash": sha_json(["jax_tpu", total, len(raw_value), len(devices)]),
            }}
        except Exception as exc:
            return {{"provider_runtime_verified": False, "blocker": "jax_tpu_provider_op_failed", "error_digest": sha_text(type(exc).__name__ + ":" + str(exc))}}
    return {{"provider_runtime_verified": False, "blocker": "provider_not_supported"}}


def runtime_work_dir() -> Path:
    return Path(os.environ.get("CT_GLM52_RUNTIME_WORK_DIR") or os.getcwd()).resolve()


def ensure_embedded_full_prefix_bundle() -> list[str]:
    written = []
    if not isinstance(EMBEDDED_FULL_PREFIX_RUNTIME_BUNDLE, dict):
        return written
    root = runtime_work_dir()
    for relative, content in EMBEDDED_FULL_PREFIX_RUNTIME_BUNDLE.items():
        path = root / str(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = str(content)
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
        written.append(str(relative))
    return written


def run_full_prefix_stage_adapter(env_overrides: dict | None = None) -> dict:
    runtime_env = dict(os.environ)
    if isinstance(env_overrides, dict):
        runtime_env.update({{str(key): str(value) for key, value in env_overrides.items() if value is not None}})
    embedded_files = ensure_embedded_full_prefix_bundle()
    script = runtime_work_dir() / "scripts" / "glm52_full_prefix_stage_decode_probe.py"
    if not script.is_file():
        return {{
            "full_prefix_stage_runtime_bundle_present": False,
            "full_prefix_stage_runtime_embedded_bundle_file_count": len(embedded_files),
            "full_prefix_stage_runtime_adapter_verified": False,
            "blocker": "glm52_full_prefix_stage_runtime_bundle_missing",
        }}
    layer_range = _list(STAGE.get("stage_layer_range"))
    if len(layer_range) != 2:
        return {{
            "full_prefix_stage_runtime_bundle_present": True,
            "full_prefix_stage_runtime_embedded_bundle_file_count": len(embedded_files),
            "full_prefix_stage_runtime_adapter_verified": False,
            "blocker": "glm52_full_prefix_stage_runtime_layer_range_invalid",
        }}
    probe_layer_range = _list(STAGE.get("full_prefix_probe_layer_range"))
    start = int(probe_layer_range[0]) if len(probe_layer_range) == 2 else int(layer_range[0])
    stage_end = int(layer_range[1])
    if len(probe_layer_range) == 2:
        end = min(stage_end, int(probe_layer_range[1]))
    else:
        layer_limit = max(1, int(os.environ.get("CT_GLM52_FULL_PREFIX_LAYER_LIMIT", "2")))
        end = min(stage_end, start + layer_limit)
    if end <= start:
        return {{
            "full_prefix_stage_runtime_bundle_present": True,
            "full_prefix_stage_runtime_embedded_bundle_file_count": len(embedded_files),
            "full_prefix_stage_runtime_adapter_verified": False,
            "blocker": "glm52_full_prefix_stage_runtime_layer_range_empty",
        }}
    output_dir = Path(runtime_env.get("CT_GLM52_FULL_PREFIX_OUTPUT_DIR", "glm52_full_prefix_stage_decode_runtime"))
    cmd = [
        sys.executable,
        str(script),
        "--output-dir",
        str(output_dir),
        "--model-repo",
        MODEL_REPO,
        "--layer-start",
        str(start),
        "--layer-end",
        str(end),
        "--prefill-length",
        runtime_env.get("CT_GLM52_FULL_PREFIX_PREFILL_LENGTH", "2"),
        "--dsa-mask-topk",
        runtime_env.get("CT_GLM52_FULL_PREFIX_DSA_MASK_TOPK", "2"),
        "--executed-expert-count",
        runtime_env.get("CT_GLM52_FULL_PREFIX_EXECUTED_EXPERT_COUNT", "8"),
        "--top-k",
        runtime_env.get("CT_GLM52_FULL_PREFIX_TOP_K", "5"),
        "--row-block-size",
        runtime_env.get("CT_GLM52_FULL_PREFIX_ROW_BLOCK_SIZE", "2048"),
        "--max-header-bytes",
        runtime_env.get("CT_GLM52_MAX_HEADER_BYTES", str(128 * 1024 * 1024)),
        "--max-tensor-bytes",
        runtime_env.get("CT_GLM52_FULL_PREFIX_MAX_TENSOR_BYTES", str(512 * 1024 * 1024)),
        "--max-block-bytes",
        runtime_env.get("CT_GLM52_FULL_PREFIX_MAX_BLOCK_BYTES", str(64 * 1024 * 1024)),
        "--hf-timeout-seconds",
        runtime_env.get("CT_GLM52_HF_TIMEOUT_SECONDS", "60"),
    ]
    stage_count = int(os.environ.get("CT_GLM52_STAGE_COUNT", str(STAGE.get("stage_count") or 3)))
    if int(STAGE.get("stage_id") or 0) != stage_count - 1 and runtime_env.get("CT_GLM52_FULL_PREFIX_FORCE_LM_HEAD", "") not in {{"1", "true", "TRUE"}}:
        cmd.append("--skip-lm-head")
    if runtime_env.get("CT_GLM52_INPUT_HIDDEN_B64"):
        cmd.extend([
            "--input-hidden-b64",
            runtime_env.get("CT_GLM52_INPUT_HIDDEN_B64", ""),
            "--input-hidden-shape-json",
            runtime_env.get("CT_GLM52_INPUT_HIDDEN_SHAPE_JSON", "[]"),
            "--input-hidden-dtype",
            runtime_env.get("CT_GLM52_INPUT_HIDDEN_DTYPE", "float16"),
        ])
    if runtime_env.get("CT_GLM52_OUTPUT_ACTIVATION_PATH"):
        cmd.extend(["--output-activation-path", runtime_env.get("CT_GLM52_OUTPUT_ACTIVATION_PATH", "")])
    timeout_seconds = float(runtime_env.get("CT_GLM52_FULL_PREFIX_TIMEOUT_SECONDS", "3600"))
    try:
        completed = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=timeout_seconds, env=runtime_env)
    except subprocess.TimeoutExpired as exc:
        return {{
            "full_prefix_stage_runtime_bundle_present": True,
            "full_prefix_stage_runtime_embedded_bundle_file_count": len(embedded_files),
            "full_prefix_stage_runtime_adapter_verified": False,
            "full_prefix_stage_runtime_layer_range": [start, end],
            "full_prefix_stage_runtime_full_stage_range": [int(layer_range[0]), int(layer_range[1])],
            "blocker": "glm52_full_prefix_stage_runtime_timeout",
            "error_digest": sha_text(type(exc).__name__),
        }}
    report_path = output_dir / "glm52_full_prefix_stage_decode_probe.json"
    probe_report = {{}}
    if report_path.is_file():
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
            probe_report = loaded if isinstance(loaded, dict) else {{}}
        except Exception:
            probe_report = {{}}
    ready = bool(
        completed.returncode == 0
        and probe_report.get("public_artifact_safe") is True
        and probe_report.get("full_prefix_stage_hidden_verified") is True
        and probe_report.get("multi_layer_stage_hidden_verified") is True
    )
    return {{
        "full_prefix_stage_runtime_bundle_present": True,
        "full_prefix_stage_runtime_embedded_bundle_file_count": len(embedded_files),
        "full_prefix_stage_runtime_adapter_verified": ready,
        "full_prefix_stage_runtime_probe_exit_code": int(completed.returncode),
        "full_prefix_stage_runtime_layer_range": [start, end],
        "full_prefix_stage_runtime_full_stage_range": [int(layer_range[0]), int(layer_range[1])],
        "full_prefix_stage_runtime_layer_limited": [start, end] != [int(layer_range[0]), int(layer_range[1])],
        "full_prefix_stage_runtime_adapter_backend": "torch_host",
        "full_prefix_stage_runtime_accelerator_decode_verified": False,
        "full_prefix_stage_runtime_report_hash": sha_json(probe_report) if probe_report else "",
        "full_prefix_stage_runtime_stdout_hash": sha_text(completed.stdout[-4096:]) if completed.stdout else "",
        "full_prefix_stage_runtime_stderr_hash": sha_text(completed.stderr[-4096:]) if completed.stderr else "",
        "full_prefix_stage_runtime_probe_ready": probe_report.get("glm52_full_prefix_stage_decode_probe_ready") is True,
        "full_prefix_stage_runtime_input_activation_consumed": probe_report.get("input_activation_consumed") is True,
        "full_prefix_stage_runtime_input_activation_hash": str(probe_report.get("input_activation_hash") or ""),
        "full_prefix_stage_runtime_output_activation_private_ready": probe_report.get("output_activation_private_ready") is True,
        "full_prefix_stage_runtime_output_activation_hash": str(probe_report.get("output_activation_hash") or ""),
        "full_prefix_stage_runtime_stage_hidden_sequence_hash": str(probe_report.get("stage_hidden_sequence_hash") or ""),
        "full_prefix_stage_runtime_selected_token_id_hash": str(probe_report.get("selected_token_id_hash") or ""),
        "full_prefix_stage_runtime_partial_token_hash_verified": probe_report.get("partial_full_prefix_token_hash_verified") is True,
        "full_prefix_stage_runtime_probe_blockers": sorted(str(item) for item in _list(probe_report.get("blockers"))),
        "full_prefix_stage_runtime_probe_errors": [
            {{
                "phase": str(_dict(item).get("phase") or ""),
                "error_type": str(_dict(item).get("error_type") or ""),
                "error_public": str(_dict(item).get("error_public") or "")[:500],
                "error_digest": str(_dict(item).get("error_digest") or ""),
            }}
            for item in _list(probe_report.get("errors"))[:8]
            if isinstance(item, dict)
        ],
        "blocker": "" if ready else "glm52_full_prefix_stage_runtime_probe_not_verified",
    }}


def generated_hash_ok(value) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) >= 71


def coordinator_decode_enabled() -> bool:
    return bool(
        STAGE_RUNTIME_PACKAGE_KIND == "full_prefix_stage_decode"
        and os.environ.get("CT_GLM52_COORDINATOR_URL", "").strip()
        and os.environ.get("CT_GLM52_COORDINATOR_TOKEN", "").strip()
    )


def coordinator_post_json(path: str, payload: dict) -> dict:
    base_url = os.environ.get("CT_GLM52_COORDINATOR_URL", "").strip().rstrip("/")
    token = os.environ.get("CT_GLM52_COORDINATOR_TOKEN", "").strip()
    if not base_url or not token:
        return {{"ok": False, "error": "coordinator_not_configured"}}
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    request = urllib.request.Request(
        base_url + path,
        data=body,
        headers={{
            "Content-Type": "application/json",
            "X-CrowdTensor-GLM52-Token": token,
        }},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=float(os.environ.get("CT_GLM52_COORDINATOR_HTTP_TIMEOUT", "120"))) as response:
        loaded = json.loads(response.read().decode("utf-8"))
    return loaded if isinstance(loaded, dict) else {{}}


def read_private_output_activation(path: Path) -> dict:
    if not path.is_file():
        return {{}}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {{}}
    except Exception:
        return {{}}


def activation_env_overrides(task: dict) -> dict:
    activation = _dict(task.get("activation"))
    if not activation:
        return {{}}
    return {{
        "CT_GLM52_INPUT_HIDDEN_B64": str(activation.get("hidden_b64") or ""),
        "CT_GLM52_INPUT_HIDDEN_SHAPE_JSON": json.dumps(_list(activation.get("hidden_shape"))),
        "CT_GLM52_INPUT_HIDDEN_DTYPE": str(activation.get("hidden_dtype") or "float16"),
    }}


def run_coordinator_decode_worker(value: dict, provider_op: dict) -> dict:
    stage_id = int(STAGE.get("stage_id") or 0)
    stage_count = int(os.environ.get("CT_GLM52_STAGE_COUNT", str(STAGE.get("stage_count") or 3)))
    deadline = time.monotonic() + float(os.environ.get("CT_GLM52_COORDINATOR_TASK_TIMEOUT_SECONDS", "7200"))
    poll_interval = max(1.0, float(os.environ.get("CT_GLM52_COORDINATOR_POLL_INTERVAL_SECONDS", "5")))
    stage_task_limit = max(1, int(os.environ.get("CT_GLM52_COORDINATOR_STAGE_TASK_LIMIT", "1")))
    accepted_count = 0
    activation_hashes = []
    stage_output_hashes = []
    token_hashes = []
    blockers = []
    last_full_prefix = {{}}
    last_submit = {{}}
    processed_tasks = []
    while time.monotonic() < deadline:
        claim = coordinator_post_json(
            "/claim",
            {{"miner_id": "glm52-stage-" + str(stage_id), "stage_id": stage_id}},
        )
        if claim.get("done"):
            break
        task = _dict(claim.get("task"))
        if not task:
            time.sleep(poll_interval)
            continue
        task_id = str(task.get("task_id") or "")
        is_final_stage = task.get("is_final_stage") is True
        env_overrides = activation_env_overrides(task)
        if stage_id > 0 and not env_overrides.get("CT_GLM52_INPUT_HIDDEN_B64"):
            blockers.append("glm52_coordinator_input_activation_missing")
            break
        suffix = sha_text(task_id)[7:23]
        output_activation_path = runtime_work_dir() / ("glm52_private_output_activation_stage" + str(stage_id) + "_" + suffix + ".json")
        stage_output_dir = runtime_work_dir() / ("glm52_full_prefix_stage_decode_runtime_stage" + str(stage_id) + "_" + suffix)
        env_overrides.update({{
            "CT_GLM52_OUTPUT_ACTIVATION_PATH": str(output_activation_path),
            "CT_GLM52_FULL_PREFIX_OUTPUT_DIR": str(stage_output_dir),
        }})
        if is_final_stage:
            env_overrides["CT_GLM52_FULL_PREFIX_FORCE_LM_HEAD"] = "1"
        full_prefix = run_full_prefix_stage_adapter(env_overrides=env_overrides)
        last_full_prefix = full_prefix
        if full_prefix.get("full_prefix_stage_runtime_adapter_verified") is not True:
            blockers.append(str(full_prefix.get("blocker") or "glm52_coordinator_full_prefix_stage_not_verified"))
            break
        if stage_id > 0 and full_prefix.get("full_prefix_stage_runtime_input_activation_consumed") is not True:
            blockers.append("glm52_coordinator_input_activation_not_consumed")
            break
        stage_output_hash = sha_json({{
            "stage_id": stage_id,
            "task_id_hash": sha_text(task_id),
            "weight_value_hash": value.get("weight_value_sha256", ""),
            "provider_op_hash": provider_op.get("provider_op_hash", ""),
            "full_prefix_report_hash": full_prefix.get("full_prefix_stage_runtime_report_hash", ""),
            "input_activation_hash": full_prefix.get("full_prefix_stage_runtime_input_activation_hash", ""),
            "output_activation_hash": full_prefix.get("full_prefix_stage_runtime_output_activation_hash", ""),
            "selected_token_hash": full_prefix.get("full_prefix_stage_runtime_selected_token_id_hash", ""),
        }})
        result = {{
            "task_id": task_id,
            "stage_id": stage_id,
            "generation_step": int(task.get("generation_step") or 0),
            "public_artifact_safe": True,
            "stage_decode_verified": True,
            "stage_output_hash": stage_output_hash,
            "output_hash": stage_output_hash,
            "weight_value_sha256": value.get("weight_value_sha256", ""),
            "weight_value_byte_count": int(value.get("weight_value_byte_count") or 0),
            "provider_runtime_verified": provider_op.get("provider_runtime_verified") is True,
            "provider_device_count": int(provider_op.get("provider_device_count") or 0),
            "stage_decode_report_hash": full_prefix.get("full_prefix_stage_runtime_report_hash", ""),
            "duration_seconds": 0.0,
            "kv_cache": {{
                "schema": "glm52_stage_local_cache_summary_v1",
                "stage_id": stage_id,
                "ready": True,
                "cache_tensors_public": False,
                "past_key_values_public": False,
            }},
        }}
        if not is_final_stage:
            activation = read_private_output_activation(output_activation_path)
            activation_hash = str(activation.get("activation_hash") or "")
            if not generated_hash_ok(activation_hash) or not activation.get("hidden_b64"):
                blockers.append("glm52_coordinator_output_activation_missing")
                break
            result["activation"] = activation
            result["activation_hash"] = activation_hash
            activation_hashes.append(activation_hash)
        else:
            token_hash = str(full_prefix.get("full_prefix_stage_runtime_selected_token_id_hash") or "")
            if not generated_hash_ok(token_hash):
                blockers.append("glm52_coordinator_generated_token_hash_missing")
                break
            result["generated_token_hash"] = token_hash
            result["next_token_hash"] = token_hash
            token_hashes.append(token_hash)
        submitted = coordinator_post_json("/submit", result)
        last_submit = submitted
        processed_tasks.append({{
            "task_id_hash": sha_text(task_id),
            "stage_id": stage_id,
            "accepted": submitted.get("accepted") is True,
            "stage_output_hash": stage_output_hash,
        }})
        if submitted.get("accepted") is not True:
            blockers.append("glm52_coordinator_stage_submit_rejected")
            break
        accepted_count += 1
        stage_output_hashes.append(stage_output_hash)
        if submitted.get("ready") is True:
            break
        if accepted_count >= stage_task_limit:
            break
    if accepted_count < 1 and not blockers:
        blockers.append("glm52_coordinator_stage_task_not_accepted")
    verified = bool(accepted_count >= 1 and not blockers)
    return {{
        "coordinator_decode_worker_enabled": True,
        "coordinator_stage_decode_verified": verified,
        "coordinator_stage_tasks_accepted": accepted_count,
        "coordinator_stage_task_limit": stage_task_limit,
        "coordinator_stage_last_submit_ready": last_submit.get("ready") is True,
        "coordinator_stage_output_hashes": stage_output_hashes,
        "coordinator_stage_activation_hashes": activation_hashes,
        "coordinator_stage_generated_token_hashes": token_hashes,
        "coordinator_stage_processed_tasks": processed_tasks,
        "coordinator_stage_last_full_prefix_report_hash": last_full_prefix.get("full_prefix_stage_runtime_report_hash", ""),
        "coordinator_stage_last_full_prefix_layer_range": last_full_prefix.get("full_prefix_stage_runtime_layer_range", []),
        "coordinator_stage_last_full_prefix_adapter_verified": last_full_prefix.get("full_prefix_stage_runtime_adapter_verified") is True,
        "coordinator_stage_last_full_prefix_blocker": str(last_full_prefix.get("blocker") or ""),
        "coordinator_stage_last_full_prefix_probe_exit_code": int(last_full_prefix.get("full_prefix_stage_runtime_probe_exit_code") or -1),
        "coordinator_stage_last_full_prefix_probe_ready": last_full_prefix.get("full_prefix_stage_runtime_probe_ready") is True,
        "coordinator_stage_last_full_prefix_input_activation_consumed": last_full_prefix.get("full_prefix_stage_runtime_input_activation_consumed") is True,
        "coordinator_stage_last_full_prefix_input_activation_hash": str(last_full_prefix.get("full_prefix_stage_runtime_input_activation_hash") or ""),
        "coordinator_stage_last_full_prefix_output_activation_private_ready": last_full_prefix.get("full_prefix_stage_runtime_output_activation_private_ready") is True,
        "coordinator_stage_last_full_prefix_output_activation_hash": str(last_full_prefix.get("full_prefix_stage_runtime_output_activation_hash") or ""),
        "coordinator_stage_last_full_prefix_stdout_hash": str(last_full_prefix.get("full_prefix_stage_runtime_stdout_hash") or ""),
        "coordinator_stage_last_full_prefix_stderr_hash": str(last_full_prefix.get("full_prefix_stage_runtime_stderr_hash") or ""),
        "coordinator_stage_last_full_prefix_probe_blockers": _list(last_full_prefix.get("full_prefix_stage_runtime_probe_blockers")),
        "coordinator_stage_last_full_prefix_probe_errors": _list(last_full_prefix.get("full_prefix_stage_runtime_probe_errors")),
        "coordinator_stage_last_submit_accepted": last_submit.get("accepted") is True,
        "same_request_route_verified": verified,
        "blocker": "" if verified else (blockers[0] if blockers else "glm52_coordinator_stage_decode_not_verified"),
        "blockers": sorted(set(blockers)),
    }}


def run_coordinator_decode_worker_claim_before_load() -> dict:
    stage_id = int(STAGE.get("stage_id") or 0)
    stage_count = int(os.environ.get("CT_GLM52_STAGE_COUNT", str(STAGE.get("stage_count") or 3)))
    deadline = time.monotonic() + float(os.environ.get("CT_GLM52_COORDINATOR_TASK_TIMEOUT_SECONDS", "7200"))
    poll_interval = max(1.0, float(os.environ.get("CT_GLM52_COORDINATOR_POLL_INTERVAL_SECONDS", "5")))
    stage_task_limit = max(1, int(os.environ.get("CT_GLM52_COORDINATOR_STAGE_TASK_LIMIT", "1")))
    accepted_count = 0
    activation_hashes = []
    stage_output_hashes = []
    token_hashes = []
    blockers = []
    last_full_prefix = {{}}
    last_submit = {{}}
    processed_tasks = []
    last_value_summary = {{}}
    last_provider_summary = {{}}
    while time.monotonic() < deadline:
        claim = coordinator_post_json(
            "/claim",
            {{"miner_id": "glm52-stage-" + str(stage_id), "stage_id": stage_id}},
        )
        if claim.get("done"):
            break
        task = _dict(claim.get("task"))
        if not task:
            time.sleep(poll_interval)
            continue
        task_id = str(task.get("task_id") or "")
        is_final_stage = task.get("is_final_stage") is True
        env_overrides = activation_env_overrides(task)
        if stage_id > 0 and not env_overrides.get("CT_GLM52_INPUT_HIDDEN_B64"):
            blockers.append("glm52_coordinator_input_activation_missing")
            break
        try:
            value = select_stage_tensor(
                MODEL_REPO,
                max_header_files=int(os.environ.get("CT_GLM52_MAX_HEADER_FILES", "32")),
                max_header_bytes=int(os.environ.get("CT_GLM52_MAX_HEADER_BYTES", str(128 * 1024 * 1024))),
                max_tensor_bytes=int(os.environ.get("CT_GLM52_MAX_TENSOR_BYTES", str(4 * 1024 * 1024))),
            )
            provider_op = provider_runtime_op(value["raw_value"])
        except Exception as exc:
            blockers.append("glm52_claim_first_stage_value_load_failed")
            last_value_summary = {{
                "weight_value_sha256": "",
                "weight_value_byte_count": 0,
                "error_type": type(exc).__name__,
                "error_digest": sha_text(type(exc).__name__ + ":" + str(exc)),
            }}
            break
        if provider_op.get("provider_runtime_verified") is not True:
            blockers.append(str(provider_op.get("blocker") or "provider_runtime_op_not_verified"))
            last_value_summary = {{
                "weight_value_sha256": value.get("weight_value_sha256", ""),
                "weight_value_byte_count": int(value.get("weight_value_byte_count") or 0),
            }}
            last_provider_summary = {{
                "provider_runtime_verified": False,
                "provider_device_count": int(provider_op.get("provider_device_count") or 0),
                "provider_op_hash": provider_op.get("provider_op_hash", ""),
            }}
            value.pop("raw_value", None)
            break
        last_value_summary = {{
            "weight_value_sha256": value.get("weight_value_sha256", ""),
            "weight_value_byte_count": int(value.get("weight_value_byte_count") or 0),
            "assigned_weight_key_count": int(value.get("assigned_weight_key_count") or 0),
            "assigned_weight_file_count": int(value.get("assigned_weight_file_count") or 0),
            "header_file_count": int(value.get("header_file_count") or 0),
            "selected_tensor": value.get("selected_tensor", {{}}),
        }}
        last_provider_summary = {{
            "provider_runtime_verified": provider_op.get("provider_runtime_verified") is True,
            "provider_device_count": int(provider_op.get("provider_device_count") or 0),
            "provider_op_hash": provider_op.get("provider_op_hash", ""),
        }}
        suffix = sha_text(task_id)[7:23]
        output_activation_path = runtime_work_dir() / ("glm52_private_output_activation_stage" + str(stage_id) + "_" + suffix + ".json")
        stage_output_dir = runtime_work_dir() / ("glm52_full_prefix_stage_decode_runtime_stage" + str(stage_id) + "_" + suffix)
        env_overrides.update({{
            "CT_GLM52_OUTPUT_ACTIVATION_PATH": str(output_activation_path),
            "CT_GLM52_FULL_PREFIX_OUTPUT_DIR": str(stage_output_dir),
        }})
        if is_final_stage:
            env_overrides["CT_GLM52_FULL_PREFIX_FORCE_LM_HEAD"] = "1"
        full_prefix = run_full_prefix_stage_adapter(env_overrides=env_overrides)
        last_full_prefix = full_prefix
        if full_prefix.get("full_prefix_stage_runtime_adapter_verified") is not True:
            blockers.append(str(full_prefix.get("blocker") or "glm52_coordinator_full_prefix_stage_not_verified"))
            value.pop("raw_value", None)
            break
        if stage_id > 0 and full_prefix.get("full_prefix_stage_runtime_input_activation_consumed") is not True:
            blockers.append("glm52_coordinator_input_activation_not_consumed")
            value.pop("raw_value", None)
            break
        stage_output_hash = sha_json({{
            "stage_id": stage_id,
            "task_id_hash": sha_text(task_id),
            "weight_value_hash": value.get("weight_value_sha256", ""),
            "provider_op_hash": provider_op.get("provider_op_hash", ""),
            "full_prefix_report_hash": full_prefix.get("full_prefix_stage_runtime_report_hash", ""),
            "input_activation_hash": full_prefix.get("full_prefix_stage_runtime_input_activation_hash", ""),
            "output_activation_hash": full_prefix.get("full_prefix_stage_runtime_output_activation_hash", ""),
            "selected_token_hash": full_prefix.get("full_prefix_stage_runtime_selected_token_id_hash", ""),
        }})
        result = {{
            "task_id": task_id,
            "stage_id": stage_id,
            "generation_step": int(task.get("generation_step") or 0),
            "public_artifact_safe": True,
            "stage_decode_verified": True,
            "stage_output_hash": stage_output_hash,
            "output_hash": stage_output_hash,
            "weight_value_sha256": value.get("weight_value_sha256", ""),
            "weight_value_byte_count": int(value.get("weight_value_byte_count") or 0),
            "provider_runtime_verified": provider_op.get("provider_runtime_verified") is True,
            "provider_device_count": int(provider_op.get("provider_device_count") or 0),
            "stage_decode_report_hash": full_prefix.get("full_prefix_stage_runtime_report_hash", ""),
            "duration_seconds": 0.0,
            "kv_cache": {{
                "schema": "glm52_stage_local_cache_summary_v1",
                "stage_id": stage_id,
                "ready": True,
                "cache_tensors_public": False,
                "past_key_values_public": False,
            }},
        }}
        if not is_final_stage:
            activation = read_private_output_activation(output_activation_path)
            activation_hash = str(activation.get("activation_hash") or "")
            if not generated_hash_ok(activation_hash) or not activation.get("hidden_b64"):
                blockers.append("glm52_coordinator_output_activation_missing")
                value.pop("raw_value", None)
                break
            result["activation"] = activation
            result["activation_hash"] = activation_hash
            activation_hashes.append(activation_hash)
        else:
            token_hash = str(full_prefix.get("full_prefix_stage_runtime_selected_token_id_hash") or "")
            if not generated_hash_ok(token_hash):
                blockers.append("glm52_coordinator_generated_token_hash_missing")
                value.pop("raw_value", None)
                break
            result["generated_token_hash"] = token_hash
            result["next_token_hash"] = token_hash
            token_hashes.append(token_hash)
        submitted = coordinator_post_json("/submit", result)
        last_submit = submitted
        processed_tasks.append({{
            "task_id_hash": sha_text(task_id),
            "stage_id": stage_id,
            "accepted": submitted.get("accepted") is True,
            "stage_output_hash": stage_output_hash,
        }})
        value.pop("raw_value", None)
        if submitted.get("accepted") is not True:
            blockers.append("glm52_coordinator_stage_submit_rejected")
            break
        accepted_count += 1
        stage_output_hashes.append(stage_output_hash)
        if submitted.get("ready") is True:
            break
        if accepted_count >= stage_task_limit:
            break
    if accepted_count < 1 and not blockers:
        blockers.append("glm52_coordinator_stage_task_not_accepted")
    verified = bool(accepted_count >= 1 and not blockers)
    return {{
        "coordinator_decode_worker_enabled": True,
        "coordinator_claim_before_stage_load": True,
        "coordinator_stage_decode_verified": verified,
        "coordinator_stage_tasks_accepted": accepted_count,
        "coordinator_stage_task_limit": stage_task_limit,
        "coordinator_stage_last_submit_ready": last_submit.get("ready") is True,
        "coordinator_stage_output_hashes": stage_output_hashes,
        "coordinator_stage_activation_hashes": activation_hashes,
        "coordinator_stage_generated_token_hashes": token_hashes,
        "coordinator_stage_processed_tasks": processed_tasks,
        "coordinator_stage_weight_summary": last_value_summary,
        "coordinator_stage_provider_summary": last_provider_summary,
        "coordinator_stage_last_full_prefix_report_hash": last_full_prefix.get("full_prefix_stage_runtime_report_hash", ""),
        "coordinator_stage_last_full_prefix_layer_range": last_full_prefix.get("full_prefix_stage_runtime_layer_range", []),
        "coordinator_stage_last_full_prefix_adapter_verified": last_full_prefix.get("full_prefix_stage_runtime_adapter_verified") is True,
        "coordinator_stage_last_full_prefix_blocker": str(last_full_prefix.get("blocker") or ""),
        "coordinator_stage_last_full_prefix_probe_exit_code": int(last_full_prefix.get("full_prefix_stage_runtime_probe_exit_code") or -1),
        "coordinator_stage_last_full_prefix_probe_ready": last_full_prefix.get("full_prefix_stage_runtime_probe_ready") is True,
        "coordinator_stage_last_full_prefix_input_activation_consumed": last_full_prefix.get("full_prefix_stage_runtime_input_activation_consumed") is True,
        "coordinator_stage_last_full_prefix_input_activation_hash": str(last_full_prefix.get("full_prefix_stage_runtime_input_activation_hash") or ""),
        "coordinator_stage_last_full_prefix_output_activation_private_ready": last_full_prefix.get("full_prefix_stage_runtime_output_activation_private_ready") is True,
        "coordinator_stage_last_full_prefix_output_activation_hash": str(last_full_prefix.get("full_prefix_stage_runtime_output_activation_hash") or ""),
        "coordinator_stage_last_full_prefix_stdout_hash": str(last_full_prefix.get("full_prefix_stage_runtime_stdout_hash") or ""),
        "coordinator_stage_last_full_prefix_stderr_hash": str(last_full_prefix.get("full_prefix_stage_runtime_stderr_hash") or ""),
        "coordinator_stage_last_full_prefix_probe_blockers": _list(last_full_prefix.get("full_prefix_stage_runtime_probe_blockers")),
        "coordinator_stage_last_full_prefix_probe_errors": _list(last_full_prefix.get("full_prefix_stage_runtime_probe_errors")),
        "coordinator_stage_last_submit_accepted": last_submit.get("accepted") is True,
        "same_request_route_verified": verified,
        "blocker": "" if verified else (blockers[0] if blockers else "glm52_coordinator_stage_decode_not_verified"),
        "blockers": sorted(set(blockers)),
    }}


def base_report(request_hash: str) -> dict:
    return {{
        "schema": "glm52_kaggle_stage_runtime_report_v1",
        "generated_at": utc_now(),
        "model_id": STAGE["model_id"],
        "compatible_weight_repo": STAGE["compatible_weight_repo"],
        "provider": STAGE["provider"],
        "stage_id": STAGE["stage_id"],
        "stage_layer_range": STAGE["stage_layer_range"],
        "coordinator_request_id_hash": request_hash,
        "fallback_model_used": False,
        "queue_only_evidence": False,
        "metadata_only": False,
        "stage_smoke_only": False,
        "activation_public": False,
        "kv_cache_public": False,
        "weight_tensor_values_public": False,
        "safetensors_header_payload_public": False,
        "public_artifact_safe": True,
        "safety": {{
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
        }},
    }}


def main() -> None:
    report_path = Path(os.environ.get("CT_GLM52_STAGE_REPORT_PATH", "glm52_kaggle_stage_runtime_report.json"))
    request_hash = os.environ.get("CT_GLM52_COORDINATOR_REQUEST_HASH", "") or DEFAULT_COORDINATOR_REQUEST_HASH
    report = base_report(request_hash)
    if not request_hash:
        report.update({{
            "ok": False,
            "stage_execution_verified": False,
            "stage_decode_verified": False,
            "stage_output_hash": "",
            "live_run_performed": False,
            "blockers": ["glm52_stage_runtime_coordinator_request_hash_missing"],
        }})
        write_json(report_path, report)
        print(json.dumps({{"stage_worker_package_report": str(report_path), "stage_runtime_verified": False}}, sort_keys=True))
        return
    blockers = []
    try:
        full_prefix = {{}}
        coordinator_decode = {{}}
        coordinator_mode = coordinator_decode_enabled()
        claim_before_load = os.environ.get("CT_GLM52_COORDINATOR_CLAIM_BEFORE_STAGE_LOAD", "") in {{"1", "true", "TRUE"}}
        if STAGE_RUNTIME_PACKAGE_KIND == "full_prefix_stage_decode" and coordinator_mode and claim_before_load:
            coordinator_decode = run_coordinator_decode_worker_claim_before_load()
            if coordinator_decode.get("coordinator_stage_decode_verified") is not True:
                blockers.append(str(coordinator_decode.get("blocker") or "glm52_coordinator_stage_decode_not_verified"))
            weight_summary = _dict(coordinator_decode.get("coordinator_stage_weight_summary"))
            provider_summary = _dict(coordinator_decode.get("coordinator_stage_provider_summary"))
            value = {{
                "weight_value_sha256": str(weight_summary.get("weight_value_sha256") or ""),
                "weight_value_byte_count": int(weight_summary.get("weight_value_byte_count") or 0),
                "assigned_weight_key_count": int(weight_summary.get("assigned_weight_key_count") or 0),
                "assigned_weight_file_count": int(weight_summary.get("assigned_weight_file_count") or 0),
                "header_file_count": int(weight_summary.get("header_file_count") or 0),
                "selected_tensor": _dict(weight_summary.get("selected_tensor")),
            }}
            provider_op = {{
                "provider_runtime_verified": provider_summary.get("provider_runtime_verified") is True,
                "provider_device_count": int(provider_summary.get("provider_device_count") or 0),
                "provider_op_hash": str(provider_summary.get("provider_op_hash") or ""),
            }}
        else:
            value = select_stage_tensor(
                MODEL_REPO,
                max_header_files=int(os.environ.get("CT_GLM52_MAX_HEADER_FILES", "32")),
                max_header_bytes=int(os.environ.get("CT_GLM52_MAX_HEADER_BYTES", str(128 * 1024 * 1024))),
                max_tensor_bytes=int(os.environ.get("CT_GLM52_MAX_TENSOR_BYTES", str(4 * 1024 * 1024))),
            )
            provider_op = provider_runtime_op(value["raw_value"])
            if provider_op.get("provider_runtime_verified") is not True:
                blockers.append(str(provider_op.get("blocker") or "provider_runtime_op_not_verified"))
            if STAGE_RUNTIME_PACKAGE_KIND == "full_prefix_stage_decode" and coordinator_mode and not blockers:
                coordinator_decode = run_coordinator_decode_worker(value, provider_op)
                if coordinator_decode.get("coordinator_stage_decode_verified") is not True:
                    blockers.append(str(coordinator_decode.get("blocker") or "glm52_coordinator_stage_decode_not_verified"))
            elif STAGE_RUNTIME_PACKAGE_KIND == "full_prefix_stage_decode":
                full_prefix = run_full_prefix_stage_adapter()
                if full_prefix.get("full_prefix_stage_runtime_adapter_verified") is not True:
                    blockers.append(str(full_prefix.get("blocker") or "glm52_full_prefix_stage_runtime_not_verified"))
        coordinator_stage_output_hashes = _list(coordinator_decode.get("coordinator_stage_output_hashes"))
        stage_output_hash = (
            str(coordinator_stage_output_hashes[-1])
            if coordinator_stage_output_hashes
            else sha_json({{
            "provider": STAGE["provider"],
            "stage_id": STAGE["stage_id"],
            "request_hash": request_hash,
            "weight_value_hash": value["weight_value_sha256"],
            "provider_op_hash": provider_op.get("provider_op_hash", ""),
            "stage_runtime_package_kind": STAGE_RUNTIME_PACKAGE_KIND,
            "full_prefix_stage_runtime_report_hash": full_prefix.get("full_prefix_stage_runtime_report_hash", ""),
            }})
        ) if not blockers else ""
        verified = not blockers
        if STAGE_RUNTIME_PACKAGE_KIND == "full_prefix_stage_decode" and coordinator_mode:
            runtime_kind = "glm52_full_prefix_stage_decode_coordinator_worker"
            boundary_blockers = [] if verified else ["glm52_coordinator_same_request_stage_not_verified"]
        elif STAGE_RUNTIME_PACKAGE_KIND == "full_prefix_stage_decode":
            runtime_kind = "glm52_full_prefix_stage_decode_host_adapter_with_provider_op"
            boundary_blockers = [
                "glm52_full_prefix_stage_runtime_is_host_adapter",
                "glm52_full_prefix_stage_runtime_is_not_same_request",
                "glm52_stage_decode_not_verified",
                "glm52_same_request_decode_not_verified",
            ]
            if full_prefix.get("full_prefix_stage_runtime_layer_limited") is True:
                boundary_blockers.append("glm52_full_prefix_stage_runtime_layer_limited")
        else:
            runtime_kind = "glm52_awq_stage_value_provider_op"
            boundary_blockers = ["glm52_stage_value_provider_op_is_not_full_decode"]
        report.update({{
            "ok": verified,
            "stage_execution_verified": verified,
            "stage_decode_verified": bool(verified and coordinator_mode),
            "stage_runtime_kind": runtime_kind,
            "stage_runtime_package_kind": STAGE_RUNTIME_PACKAGE_KIND,
            "stage_runtime_adapter_verified": verified,
            "same_request_route_verified": bool(verified and coordinator_mode),
            "stage_output_hash": stage_output_hash,
            "live_run_performed": True,
            "stage_owned_weight_values_loaded": bool(value.get("weight_value_sha256")),
            "weight_tensor_values_loaded": bool(value.get("weight_value_sha256")),
            "weight_value_byte_count": int(value.get("weight_value_byte_count") or 0),
            "weight_value_sha256": value.get("weight_value_sha256", ""),
            "assigned_weight_key_count": int(value.get("assigned_weight_key_count") or 0),
            "assigned_weight_file_count": int(value.get("assigned_weight_file_count") or 0),
            "header_file_count": int(value.get("header_file_count") or 0),
            "selected_tensor": value.get("selected_tensor", {{}}),
            "provider_runtime_verified": provider_op.get("provider_runtime_verified") is True,
            "provider_device_count": int(provider_op.get("provider_device_count") or 0),
            "provider_op_hash": provider_op.get("provider_op_hash", ""),
            "stage_full_decode_verified": bool(verified and coordinator_mode),
            **full_prefix,
            **coordinator_decode,
            "blockers": sorted(set(blockers + boundary_blockers)),
        }})
        value.pop("raw_value", None)
    except Exception as exc:
        report.update({{
            "ok": False,
            "stage_execution_verified": False,
            "stage_decode_verified": False,
            "stage_runtime_kind": "glm52_full_prefix_stage_decode_host_adapter_with_provider_op" if STAGE_RUNTIME_PACKAGE_KIND == "full_prefix_stage_decode" else "glm52_awq_stage_value_provider_op",
            "stage_runtime_package_kind": STAGE_RUNTIME_PACKAGE_KIND,
            "stage_runtime_adapter_verified": False,
            "same_request_route_verified": False,
            "stage_output_hash": "",
            "live_run_performed": True,
            "blockers": ["glm52_stage_runtime_value_op_failed"],
            "error_type": type(exc).__name__,
            "os_error_errno": getattr(exc, "errno", None),
            "os_error_strerror_hash": sha_text(str(getattr(exc, "strerror", "") or "")) if isinstance(exc, OSError) else "",
            "error_digest": sha_text(type(exc).__name__ + ":" + str(exc)),
        }})
    write_json(report_path, report)
    print(json.dumps({{"stage_worker_package_report": str(report_path), "stage_runtime_verified": report.get("stage_execution_verified") is True}}, sort_keys=True))


if __name__ == "__main__":
    main()
'''


def render_generic_stage_kernel(
    stage: dict[str, Any],
    *,
    coordinator_request_id_hash: str = "",
    runtime_kind: str = RUNTIME_KIND_VALUE_OP,
    full_prefix_timeout_seconds: int = 3600,
) -> str:
    source = render_kernel(
        stage,
        coordinator_request_id_hash=coordinator_request_id_hash,
        runtime_kind=runtime_kind,
        full_prefix_timeout_seconds=full_prefix_timeout_seconds,
    )
    marker = "\nSTAGE = "
    start = source.index(marker) + 1
    end = source.index("\nMODEL_REPO = ", start)
    stage_assignment = source[start:end]
    default_payload = stage_assignment.split("=", 1)[1].strip()
    replacement = (
        'STAGE = json.loads(os.environ.get("CT_GLM52_STAGE_PAYLOAD_JSON", '
        + json.dumps(default_payload)
        + "))"
    )
    return source[:start] + replacement + source[end:]


def render_cpu_group_driver(
    stages: list[dict[str, Any]],
    *,
    coordinator_request_id_hash: str = "",
    runtime_kind: str = RUNTIME_KIND_VALUE_OP,
    full_prefix_timeout_seconds: int = 3600,
    stage_worker_sources: dict[str, str] | None = None,
) -> str:
    stage_group = [
        {
            "stage_id": _int(stage.get("stage_id")),
            "stage_count": _int(stage.get("stage_count"), len(stages)),
            "provider": str(stage.get("provider") or ""),
            "stage_layer_range": _list(stage.get("stage_layer_range")),
            "model_id": MODEL_ID,
            "compatible_weight_repo": str(stage.get("compatible_weight_repo") or ""),
            "runtime_adapter": str(stage.get("runtime_adapter") or ""),
            "stage_runtime_package_kind": runtime_kind,
            "full_prefix_probe_layer_range": _list(stage.get("full_prefix_probe_layer_range")),
            "full_prefix_timeout_seconds": max(1, int(full_prefix_timeout_seconds or 3600)),
        }
        for stage in stages
    ]
    default_request_hash = coordinator_request_id_hash if _hash_ok(coordinator_request_id_hash) else ""
    return f'''#!/usr/bin/env python3
"""Private Kaggle GLM 5.2 grouped CPU stage worker.

This driver keeps Kaggle CPU session fanout low while preserving fine-grained
Coordinator stages. It runs the existing per-stage worker scripts one task at a
time in a round-robin loop and writes a public-safe aggregate report.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


STAGE_GROUP = {json.dumps(stage_group, sort_keys=True)}
MODEL_ID = {json.dumps(MODEL_ID)}
MODEL_REPO = STAGE_GROUP[0].get("compatible_weight_repo") or "cyankiwi/GLM-5.2-AWQ-INT4"
DEFAULT_COORDINATOR_REQUEST_HASH = {json.dumps(default_request_hash)}
STAGE_RUNTIME_PACKAGE_KIND = {json.dumps(runtime_kind)}
EMBEDDED_STAGE_WORKER_SOURCES = {json.dumps(stage_worker_sources or {}, sort_keys=True)}
CT_GLM52_PRIVATE_RUNTIME_ENV_INLINE = {{}}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()


def sha_json(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return sha_text(encoded)


def _int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _list(value) -> list:
    return value if isinstance(value, list) else []


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {{}}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {{}}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {{}}
    except Exception:
        return {{}}


def load_private_runtime_env_file() -> None:
    loaded = {{}}
    if isinstance(CT_GLM52_PRIVATE_RUNTIME_ENV_INLINE, dict):
        loaded.update(CT_GLM52_PRIVATE_RUNTIME_ENV_INLINE)
    raw_path = os.environ.get("CT_GLM52_PRIVATE_RUNTIME_ENV_PATH", "ct_glm52_private_runtime_env.json")
    path = Path(str(raw_path))
    if path.is_file():
        try:
            loaded_from_file = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded_from_file, dict):
                loaded.update(loaded_from_file)
        except Exception:
            pass
    for key, value in loaded.items():
        key_text = str(key)
        value_text = str(value) if value is not None else ""
        if (key_text.startswith("CT_GLM52_") or key_text in {{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}}) and value_text and not os.environ.get(key_text):
            os.environ[key_text] = value_text


load_private_runtime_env_file()


def runtime_work_dir() -> Path:
    return Path(os.environ.get("CT_GLM52_RUNTIME_WORK_DIR") or os.getcwd()).resolve()


def ensure_embedded_stage_worker_sources() -> list[str]:
    written = []
    root = runtime_work_dir()
    if not isinstance(EMBEDDED_STAGE_WORKER_SOURCES, dict):
        return written
    for relative, source in EMBEDDED_STAGE_WORKER_SOURCES.items():
        path = root / str(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = str(source)
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
        written.append(str(relative))
    return written


def coordinator_status() -> dict:
    base_url = os.environ.get("CT_GLM52_COORDINATOR_URL", "").strip().rstrip("/")
    if not base_url:
        return {{}}
    try:
        request = urllib.request.Request(base_url + "/status", method="GET")
        with urllib.request.urlopen(request, timeout=float(os.environ.get("CT_GLM52_COORDINATOR_HTTP_TIMEOUT", "120"))) as response:
            loaded = json.loads(response.read().decode("utf-8"))
        return loaded if isinstance(loaded, dict) else {{}}
    except Exception:
        return {{}}


def stage_script_path(stage_id: int) -> Path:
    generic = runtime_work_dir() / "kernel_stage_generic.py"
    if generic.is_file():
        return generic
    return runtime_work_dir() / ("kernel_stage_" + str(stage_id) + ".py")


def run_one_stage_attempt(stage: dict, attempt_index: int) -> dict:
    stage_id = _int(stage.get("stage_id"), -1)
    report_path = runtime_work_dir() / ("glm52_group_stage_" + str(stage_id) + "_attempt_" + str(attempt_index) + ".json")
    env = dict(os.environ)
    env["CT_GLM52_STAGE_REPORT_PATH"] = str(report_path)
    env["CT_GLM52_COORDINATOR_CLAIM_BEFORE_STAGE_LOAD"] = "1"
    env["CT_GLM52_STAGE_PAYLOAD_JSON"] = json.dumps(stage, sort_keys=True)
    env["CT_GLM52_COORDINATOR_STAGE_TASK_LIMIT"] = "1"
    env["CT_GLM52_COORDINATOR_TASK_TIMEOUT_SECONDS"] = str(
        max(1.0, float(os.environ.get("CT_GLM52_CPU_GROUP_STAGE_ATTEMPT_SECONDS", "8")))
    )
    env.setdefault("CT_GLM52_COORDINATOR_POLL_INTERVAL_SECONDS", os.environ.get("CT_GLM52_CPU_GROUP_STAGE_POLL_SECONDS", "1"))
    script = stage_script_path(stage_id)
    if not script.is_file():
        return {{
            "stage_id": stage_id,
            "accepted": False,
            "report_present": False,
            "blocker": "glm52_cpu_group_stage_script_missing",
        }}
    timeout_seconds = max(
        float(os.environ.get("CT_GLM52_CPU_GROUP_SUBPROCESS_TIMEOUT_SECONDS", "7200")),
        float(env["CT_GLM52_COORDINATOR_TASK_TIMEOUT_SECONDS"]) + 60.0,
    )
    try:
        completed = subprocess.run(
            [sys.executable, str(script)],
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return {{
            "stage_id": stage_id,
            "accepted": False,
            "report_present": report_path.is_file(),
            "blocker": "glm52_cpu_group_stage_subprocess_timeout",
            "error_digest": sha_text(type(exc).__name__),
        }}
    report = load_json(report_path)
    accepted = _int(report.get("coordinator_stage_tasks_accepted")) > 0
    verified = report.get("stage_decode_verified") is True
    processed = _list(report.get("coordinator_stage_processed_tasks"))
    return {{
        "stage_id": stage_id,
        "accepted": bool(accepted and verified),
        "report_present": bool(report),
        "report_path": str(report_path) if report else "",
        "report_hash": sha_json(report) if report else "",
        "exit_code": int(completed.returncode),
        "stdout_hash": sha_text(completed.stdout[-4096:]) if completed.stdout else "",
        "stderr_hash": sha_text(completed.stderr[-4096:]) if completed.stderr else "",
        "stage_output_hashes": _list(report.get("coordinator_stage_output_hashes")),
        "activation_hashes": _list(report.get("coordinator_stage_activation_hashes")),
        "generated_token_hashes": _list(report.get("coordinator_stage_generated_token_hashes")),
        "processed_tasks": processed,
        "weight_value_sha256": str(report.get("weight_value_sha256") or ""),
        "weight_value_byte_count": _int(report.get("weight_value_byte_count")),
        "blockers": _list(report.get("blockers")),
    }}


def group_layer_range() -> list:
    starts = []
    ends = []
    for stage in STAGE_GROUP:
        layer_range = _list(stage.get("stage_layer_range"))
        if len(layer_range) == 2:
            starts.append(_int(layer_range[0]))
            ends.append(_int(layer_range[1]))
    return [min(starts), max(ends)] if starts and ends else []


def base_report(request_hash: str) -> dict:
    return {{
        "schema": "glm52_kaggle_stage_runtime_report_v1",
        "generated_at": utc_now(),
        "model_id": MODEL_ID,
        "compatible_weight_repo": MODEL_REPO,
        "provider": "kaggle_cpu",
        "stage_id": _int(STAGE_GROUP[0].get("stage_id"), 0) if STAGE_GROUP else 0,
        "stage_ids": [_int(stage.get("stage_id"), -1) for stage in STAGE_GROUP],
        "stage_layer_range": group_layer_range(),
        "coordinator_request_id_hash": request_hash,
        "fallback_model_used": False,
        "queue_only_evidence": False,
        "metadata_only": False,
        "stage_smoke_only": False,
        "activation_public": False,
        "kv_cache_public": False,
        "weight_tensor_values_public": False,
        "safetensors_header_payload_public": False,
        "public_artifact_safe": True,
        "safety": {{
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
        }},
    }}


def main() -> None:
    report_path = Path(os.environ.get("CT_GLM52_STAGE_REPORT_PATH", "glm52_kaggle_stage_runtime_report.json"))
    request_hash = os.environ.get("CT_GLM52_COORDINATOR_REQUEST_HASH", "") or DEFAULT_COORDINATOR_REQUEST_HASH
    report = base_report(request_hash)
    embedded_stage_worker_files = ensure_embedded_stage_worker_sources()
    if not request_hash:
        report.update({{
            "ok": False,
            "stage_execution_verified": False,
            "stage_decode_verified": False,
            "stage_output_hash": "",
            "live_run_performed": False,
            "blockers": ["glm52_stage_runtime_coordinator_request_hash_missing"],
        }})
        write_json(report_path, report)
        print(json.dumps({{"stage_worker_package_report": str(report_path), "stage_runtime_verified": False}}, sort_keys=True))
        return
    total_limit = max(1, _int(os.environ.get("CT_GLM52_COORDINATOR_STAGE_TASK_LIMIT"), len(STAGE_GROUP)))
    deadline = time.monotonic() + max(1.0, float(os.environ.get("CT_GLM52_COORDINATOR_TASK_TIMEOUT_SECONDS", "7200")))
    accepted = []
    attempts = []
    stage_ids_verified = set()
    seen_task_hashes = set()
    attempt_index = 0
    while len(accepted) < total_limit and time.monotonic() < deadline:
        status = coordinator_status()
        if status.get("ready") is True:
            break
        made_progress = False
        for stage in STAGE_GROUP:
            if len(accepted) >= total_limit or time.monotonic() >= deadline:
                break
            attempt_index += 1
            result = run_one_stage_attempt(stage, attempt_index)
            attempts.append(result)
            if result.get("accepted") is not True:
                continue
            new_tasks = []
            for task in _list(result.get("processed_tasks")):
                task_hash = str(_dict(task).get("task_id_hash") or "")
                if task_hash and task_hash in seen_task_hashes:
                    continue
                if task_hash:
                    seen_task_hashes.add(task_hash)
                new_tasks.append(task)
            if not new_tasks and result.get("processed_tasks"):
                continue
            accepted.append(result)
            stage_ids_verified.add(_int(result.get("stage_id"), -1))
            made_progress = True
            if coordinator_status().get("ready") is True:
                break
        if not made_progress:
            time.sleep(max(0.1, float(os.environ.get("CT_GLM52_CPU_GROUP_ROUND_SLEEP_SECONDS", "1"))))
    all_stage_ids = [_int(stage.get("stage_id"), -1) for stage in STAGE_GROUP]
    accepted_stage_output_hashes = []
    accepted_activation_hashes = []
    accepted_generated_token_hashes = []
    processed_tasks = []
    weight_hashes = []
    weight_bytes = 0
    for item in accepted:
        accepted_stage_output_hashes.extend(str(value) for value in _list(item.get("stage_output_hashes")) if value)
        accepted_activation_hashes.extend(str(value) for value in _list(item.get("activation_hashes")) if value)
        accepted_generated_token_hashes.extend(str(value) for value in _list(item.get("generated_token_hashes")) if value)
        processed_tasks.extend(_list(item.get("processed_tasks")))
        if item.get("weight_value_sha256"):
            weight_hashes.append(str(item.get("weight_value_sha256")))
        weight_bytes += _int(item.get("weight_value_byte_count"))
    blockers = []
    if not accepted:
        blockers.append("glm52_cpu_group_no_stage_task_accepted")
    if sorted(stage_ids_verified) != sorted(all_stage_ids):
        blockers.append("glm52_cpu_group_stage_ids_incomplete")
    stage_output_hash = sha_json({{
        "stage_ids": sorted(stage_ids_verified),
        "stage_output_hashes": accepted_stage_output_hashes,
        "processed_task_count": len(processed_tasks),
    }}) if accepted else ""
    weight_hash = sha_json(weight_hashes) if weight_hashes else ""
    report.update({{
        "ok": bool(accepted),
        "stage_execution_verified": bool(accepted),
        "stage_decode_verified": bool(accepted),
        "stage_runtime_kind": "glm52_grouped_cpu_full_prefix_stage_decode_coordinator_worker",
        "stage_runtime_package_kind": STAGE_RUNTIME_PACKAGE_KIND,
        "stage_runtime_adapter_verified": bool(accepted),
        "same_request_route_verified": bool(accepted),
        "grouped_stage_worker": True,
        "grouped_stage_count": len(STAGE_GROUP),
        "embedded_stage_worker_file_count": len(embedded_stage_worker_files),
        "embedded_stage_worker_files": embedded_stage_worker_files,
        "stage_ids_verified": sorted(stage_ids_verified),
        "stage_output_hash": stage_output_hash,
        "live_run_performed": True,
        "stage_owned_weight_values_loaded": bool(weight_hashes),
        "weight_tensor_values_loaded": bool(weight_hashes),
        "weight_value_byte_count": weight_bytes,
        "weight_value_sha256": weight_hash,
        "assigned_weight_key_count": 1 if weight_hashes else 0,
        "assigned_weight_file_count": 1 if weight_hashes else 0,
        "header_file_count": 1 if weight_hashes else 0,
        "selected_tensor": {{"grouped_stage_worker": True, "stage_ids_digest": sha_json(all_stage_ids)}},
        "provider_runtime_verified": bool(accepted),
        "provider_device_count": 1,
        "provider_op_hash": sha_json(["grouped_cpu", sorted(stage_ids_verified), len(processed_tasks)]) if accepted else "",
        "stage_full_decode_verified": bool(accepted),
        "coordinator_decode_worker_enabled": True,
        "coordinator_stage_decode_verified": bool(accepted),
        "coordinator_stage_tasks_accepted": len(processed_tasks),
        "coordinator_stage_task_limit": total_limit,
        "coordinator_stage_output_hashes": accepted_stage_output_hashes,
        "coordinator_stage_activation_hashes": accepted_activation_hashes,
        "coordinator_stage_generated_token_hashes": accepted_generated_token_hashes,
        "coordinator_stage_processed_tasks": processed_tasks,
        "coordinator_stage_last_submit_ready": coordinator_status().get("ready") is True,
        "same_request_route_verified": bool(accepted),
        "per_stage_attempt_count": len(attempts),
        "per_stage_report_summaries": attempts[-64:],
        "blocker": "" if accepted else "glm52_cpu_group_no_stage_task_accepted",
        "blockers": sorted(set(blockers)),
    }})
    write_json(report_path, report)
    print(json.dumps({{"stage_worker_package_report": str(report_path), "stage_runtime_verified": report.get("stage_execution_verified") is True}}, sort_keys=True))


if __name__ == "__main__":
    main()
'''


def metadata_for_stage(stage: dict[str, Any], *, owner: str, title_prefix: str) -> dict[str, Any]:
    provider = str(stage.get("provider") or "")
    stage_id = _int(stage.get("stage_id"))
    title = f"{title_prefix}-{stage_id}-{safe_slug(provider)}"
    machine_shape = "tpuV5e8" if provider == "kaggle_jax_tpu" else ("NvidiaTeslaT4" if provider == "kaggle_cuda" else "")
    return {
        "id": f"{owner}/{title}",
        "title": title,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true" if provider == "kaggle_cuda" else "false",
        "enable_tpu": "true" if provider == "kaggle_jax_tpu" else "false",
        "enable_internet": "true",
        "machine_shape": machine_shape,
        "dataset_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "competition_sources": [],
    }


def metadata_for_group(stages: list[dict[str, Any]], *, owner: str, title_prefix: str) -> dict[str, Any]:
    first = _int(stages[0].get("stage_id"))
    last = _int(stages[-1].get("stage_id"))
    provider = str(stages[0].get("provider") or "kaggle_cpu")
    title = f"{title_prefix}-{first}-{last}-{safe_slug(provider)}-group"
    return {
        "id": f"{owner}/{title}",
        "title": title,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "",
        "dataset_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "competition_sources": [],
    }


def write_full_prefix_runtime_bundle(package_dir: Path) -> list[dict[str, Any]]:
    scripts_dir = package_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(__file__).resolve().parent
    bundled: list[dict[str, Any]] = []
    for filename in FULL_PREFIX_RUNTIME_BUNDLE:
        source = source_dir / filename
        target = scripts_dir / filename
        if not source.is_file():
            continue
        shutil.copyfile(source, target)
        bundled.append(
            {
                "relative_path": str(target.relative_to(package_dir)),
                "sha256": sha_file(target),
            }
        )
    init_path = scripts_dir / "__init__.py"
    if not init_path.exists():
        write_text(init_path, "")
    bundled.append(
        {
            "relative_path": str(init_path.relative_to(package_dir)),
            "sha256": sha_file(init_path),
        }
    )
    return bundled


def grouped_stage_specs(stage_specs: list[dict[str, Any]], *, cpu_stage_group_size: int) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    buffer: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal buffer
        if buffer:
            groups.append(buffer)
            buffer = []

    size = max(1, int(cpu_stage_group_size or 1))
    for stage in stage_specs:
        provider = str(stage.get("provider") or "")
        if provider == "kaggle_cpu" and size > 1:
            buffer.append(stage)
            if len(buffer) >= size:
                flush()
            continue
        flush()
        groups.append([stage])
    flush()
    return groups


def group_layer_range(stages: list[dict[str, Any]]) -> list[int]:
    starts: list[int] = []
    ends: list[int] = []
    for stage in stages:
        layer_range = _list(stage.get("stage_layer_range"))
        if len(layer_range) == 2:
            starts.append(_int(layer_range[0]))
            ends.append(_int(layer_range[1]))
    return [min(starts), max(ends)] if starts and ends else []


def stage_sort_key(stage: dict[str, Any]) -> tuple[int, int]:
    layer_range = _list(stage.get("stage_layer_range"))
    start = _int(layer_range[0], _int(stage.get("stage_id"))) if len(layer_range) == 2 else _int(stage.get("stage_id"))
    return start, _int(stage.get("stage_id"))


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    plan_report = load_json(args.stage_runtime_plan_report)
    stage_specs = sorted(
        [item for item in _list(plan_report.get("stage_specs")) if isinstance(item, dict)],
        key=stage_sort_key,
    )
    packages: list[dict[str, Any]] = []
    blockers = {
        "glm52_stage_worker_packages_not_pushed",
        "glm52_stage_worker_live_reports_missing",
        "glm52_stage_worker_package_is_not_runtime_success",
    }
    request_hash = str(args.coordinator_request_id_hash or "")
    request_hash_bound = _hash_ok(request_hash)
    if request_hash and not request_hash_bound:
        blockers.add("glm52_stage_worker_coordinator_request_hash_invalid")
    if not request_hash_bound:
        blockers.add("glm52_stage_worker_coordinator_request_hash_not_bound")
    runtime_kind = str(args.runtime_kind or RUNTIME_KIND_VALUE_OP)
    if runtime_kind not in RUNTIME_KINDS:
        blockers.add("glm52_stage_worker_runtime_kind_invalid")
    provider_owner_map = parse_provider_owner_map(str(args.provider_owner_map or ""))
    for stage_group in grouped_stage_specs(stage_specs, cpu_stage_group_size=int(args.cpu_stage_group_size)):
        if (
            len(stage_group) > 1
            and all(str(stage.get("provider") or "") == "kaggle_cpu" for stage in stage_group)
        ):
            provider = "kaggle_cpu"
            stage_ids = [_int(stage.get("stage_id")) for stage in stage_group]
            stage_id = stage_ids[0]
            owner = owner_for_provider(provider, default_owner=str(args.kaggle_owner), owner_map=provider_owner_map)
            for stage in stage_group:
                stage["full_prefix_probe_layer_range"] = select_full_prefix_probe_layer_range(
                    stage,
                    mode=str(args.full_prefix_probe_mode or FULL_PREFIX_PROBE_MODE_DEFAULT),
                )
            package_dir = output_dir / f"private-kaggle-glm52-stage-group-{stage_ids[0]}-{stage_ids[-1]}-{safe_slug(provider)}"
            kernel_path = package_dir / "kernel.py"
            metadata_path = package_dir / "kernel-metadata.json"
            metadata = metadata_for_group(stage_group, owner=owner, title_prefix=args.title_prefix)
            bundled_runtime_files: list[dict[str, Any]] = []
            if runtime_kind == RUNTIME_KIND_FULL_PREFIX_STAGE_DECODE:
                bundled_runtime_files = write_full_prefix_runtime_bundle(package_dir)
                bundled_names = {Path(str(item.get("relative_path") or "")).name for item in bundled_runtime_files}
                missing = sorted(set(FULL_PREFIX_RUNTIME_BUNDLE) - bundled_names)
                if missing:
                    blockers.add("glm52_full_prefix_stage_worker_bundle_incomplete")
            stage_script_files = []
            generic_stage_source = render_generic_stage_kernel(
                stage_group[0],
                coordinator_request_id_hash=request_hash if request_hash_bound else "",
                runtime_kind=runtime_kind,
                full_prefix_timeout_seconds=int(args.full_prefix_timeout_seconds),
            )
            generic_stage_path = package_dir / "kernel_stage_generic.py"
            write_text(generic_stage_path, generic_stage_source)
            stage_worker_sources: dict[str, str] = {
                generic_stage_path.name: generic_stage_source,
            }
            stage_script_files.append({
                "stage_id": -1,
                "relative_path": str(generic_stage_path.relative_to(package_dir)),
                "sha256": sha_file(generic_stage_path),
                "generic_stage_worker": True,
            })
            for stage in stage_group:
                per_stage_source = render_kernel(
                    stage,
                    coordinator_request_id_hash=request_hash if request_hash_bound else "",
                    runtime_kind=runtime_kind,
                    full_prefix_timeout_seconds=int(args.full_prefix_timeout_seconds),
                )
                per_stage_path = package_dir / f"kernel_stage_{_int(stage.get('stage_id'))}.py"
                write_text(per_stage_path, per_stage_source)
                stage_script_files.append({
                    "stage_id": _int(stage.get("stage_id")),
                    "relative_path": str(per_stage_path.relative_to(package_dir)),
                    "sha256": sha_file(per_stage_path),
                })
            kernel_source = render_cpu_group_driver(
                stage_group,
                coordinator_request_id_hash=request_hash if request_hash_bound else "",
                runtime_kind=runtime_kind,
                full_prefix_timeout_seconds=int(args.full_prefix_timeout_seconds),
                stage_worker_sources=stage_worker_sources,
            )
            write_text(kernel_path, kernel_source)
            write_json(metadata_path, metadata)
            packages.append({
                "schema": "glm52_kaggle_stage_worker_package_entry_v1",
                "provider": provider,
                "stage_id": stage_id,
                "stage_ids": stage_ids,
                "stage_specs": stage_group,
                "grouped_stage_worker": True,
                "grouped_stage_count": len(stage_group),
                "kaggle_owner": owner,
                "stage_count": _int(stage_group[0].get("stage_count"), len(stage_specs)),
                "stage_layer_range": group_layer_range(stage_group),
                "full_prefix_probe_mode": str(args.full_prefix_probe_mode or FULL_PREFIX_PROBE_MODE_DEFAULT),
                "full_prefix_probe_layer_range": group_layer_range(stage_group),
                "full_prefix_probe_covers_full_stage": True,
                "full_prefix_timeout_seconds": int(args.full_prefix_timeout_seconds),
                "kernel_ref": str(metadata["id"]),
                "package_dir": str(package_dir),
                "kernel_path": str(kernel_path),
                "kernel_sha256": sha_file(kernel_path),
                "metadata_path": str(metadata_path),
                "metadata_sha256": sha_file(metadata_path),
                "expected_stage_report_schema": "glm52_kaggle_stage_runtime_report_v1",
                "stage_runtime_package_kind": runtime_kind,
                "full_prefix_runtime_bundle_present": runtime_kind != RUNTIME_KIND_FULL_PREFIX_STAGE_DECODE or bool(bundled_runtime_files),
                "embedded_runtime_bundle_present": runtime_kind != RUNTIME_KIND_FULL_PREFIX_STAGE_DECODE or bool(FULL_PREFIX_RUNTIME_BUNDLE),
                "embedded_runtime_bundle_file_count": len(FULL_PREFIX_RUNTIME_BUNDLE) + 1 if runtime_kind == RUNTIME_KIND_FULL_PREFIX_STAGE_DECODE else 0,
                "embedded_stage_worker_source_count": len(stage_worker_sources),
                "bundled_runtime_files": bundled_runtime_files,
                "stage_script_files": stage_script_files,
                "private_kernel": True,
                "pushed_to_kaggle": False,
                "live_run_performed": False,
                "stage_runtime_adapter_verified": False,
                "same_request_route_verified": False,
                "coordinator_request_id_hash_bound": request_hash_bound,
                "public_artifact_safe": True,
            })
            continue

        stage = stage_group[0]
        provider = str(stage.get("provider") or "")
        stage_id = _int(stage.get("stage_id"))
        owner = owner_for_provider(provider, default_owner=str(args.kaggle_owner), owner_map=provider_owner_map)
        full_prefix_probe_layer_range_for_stage = select_full_prefix_probe_layer_range(
            stage,
            mode=str(args.full_prefix_probe_mode or FULL_PREFIX_PROBE_MODE_DEFAULT),
        )
        package_dir = output_dir / f"private-kaggle-glm52-stage-{stage_id}-{safe_slug(provider)}"
        stage_payload = {**stage, "full_prefix_probe_layer_range": full_prefix_probe_layer_range_for_stage}
        kernel_source = render_kernel(
            stage_payload,
            coordinator_request_id_hash=request_hash if request_hash_bound else "",
            runtime_kind=runtime_kind,
            full_prefix_timeout_seconds=int(args.full_prefix_timeout_seconds),
        )
        kernel_path = package_dir / "kernel.py"
        metadata_path = package_dir / "kernel-metadata.json"
        metadata = metadata_for_stage(stage, owner=owner, title_prefix=args.title_prefix)
        write_text(kernel_path, kernel_source)
        bundled_runtime_files: list[dict[str, Any]] = []
        if runtime_kind == RUNTIME_KIND_FULL_PREFIX_STAGE_DECODE:
            bundled_runtime_files = write_full_prefix_runtime_bundle(package_dir)
            bundled_names = {Path(str(item.get("relative_path") or "")).name for item in bundled_runtime_files}
            missing = sorted(set(FULL_PREFIX_RUNTIME_BUNDLE) - bundled_names)
            if missing:
                blockers.add("glm52_full_prefix_stage_worker_bundle_incomplete")
        write_json(metadata_path, metadata)
        packages.append({
            "schema": "glm52_kaggle_stage_worker_package_entry_v1",
            "provider": provider,
            "stage_id": stage_id,
            "stage_ids": [stage_id],
            "kaggle_owner": owner,
            "stage_count": _int(stage.get("stage_count"), len(stage_specs)),
            "stage_layer_range": _list(stage.get("stage_layer_range")),
            "full_prefix_probe_mode": str(args.full_prefix_probe_mode or FULL_PREFIX_PROBE_MODE_DEFAULT),
            "full_prefix_probe_layer_range": full_prefix_probe_layer_range_for_stage,
            "full_prefix_probe_covers_full_stage": full_prefix_probe_layer_range_for_stage == _list(stage.get("stage_layer_range")),
            "full_prefix_timeout_seconds": int(args.full_prefix_timeout_seconds),
            "kernel_ref": str(metadata["id"]),
            "package_dir": str(package_dir),
            "kernel_path": str(kernel_path),
            "kernel_sha256": sha_file(kernel_path),
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha_file(metadata_path),
            "expected_stage_report_schema": "glm52_kaggle_stage_runtime_report_v1",
            "stage_runtime_package_kind": runtime_kind,
            "full_prefix_runtime_bundle_present": runtime_kind != RUNTIME_KIND_FULL_PREFIX_STAGE_DECODE or bool(bundled_runtime_files),
            "embedded_runtime_bundle_present": runtime_kind != RUNTIME_KIND_FULL_PREFIX_STAGE_DECODE or bool(FULL_PREFIX_RUNTIME_BUNDLE),
            "embedded_runtime_bundle_file_count": len(FULL_PREFIX_RUNTIME_BUNDLE) + 1 if runtime_kind == RUNTIME_KIND_FULL_PREFIX_STAGE_DECODE else 0,
            "bundled_runtime_files": bundled_runtime_files,
            "private_kernel": True,
            "pushed_to_kaggle": False,
            "live_run_performed": False,
            "stage_runtime_adapter_verified": False,
            "same_request_route_verified": False,
            "coordinator_request_id_hash_bound": request_hash_bound,
            "public_artifact_safe": True,
        })
    providers = {str(pkg.get("provider")) for pkg in packages}
    for provider in REQUIRED_PROVIDERS:
        if provider not in providers:
            blockers.add(f"glm52_stage_worker_package_provider_missing:{provider}")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "glm52_stage_worker_package_ready": True,
        "stage_runtime_adapter_verified": False,
        "same_request_route_verified": False,
        "live_run_performed": False,
        "model": {
            "model_id": MODEL_ID,
            "fallback_model_allowed_for_success": False,
        },
        "source_plan": {
            "source_schema": str(plan_report.get("schema") or ""),
            "plan_ready": plan_report.get("glm52_stage_runtime_plan_ready") is True,
            "stage_runtime_adapter_verified": plan_report.get("stage_runtime_adapter_verified") is True,
        },
        "stage_runtime_package_kind": runtime_kind,
        "default_kaggle_owner": str(args.kaggle_owner),
        "provider_owner_map": provider_owner_map,
        "full_prefix_probe_mode": str(args.full_prefix_probe_mode or FULL_PREFIX_PROBE_MODE_DEFAULT),
        "full_prefix_probe_full_stage_requested": str(args.full_prefix_probe_mode or "") == FULL_PREFIX_PROBE_MODE_FULL_STAGE,
        "full_prefix_timeout_seconds": int(args.full_prefix_timeout_seconds),
        "full_prefix_runtime_bundle_required": runtime_kind == RUNTIME_KIND_FULL_PREFIX_STAGE_DECODE,
        "cpu_stage_group_size": max(1, int(args.cpu_stage_group_size)),
        "grouped_stage_worker_package_count": sum(1 for package in packages if package.get("grouped_stage_worker") is True),
        "coordinator_request_id_hash_bound": request_hash_bound,
        "coordinator_request_id_hash": request_hash if request_hash_bound else "",
        "packages": packages,
        "completion_boundary": {
            "package_is_not_runtime_success": True,
            "kaggle_push_required": True,
            "live_stage_report_required": True,
            "same_request_probe_required": True,
        },
        "blockers": sorted(blockers),
        "safety": safety_flags(),
        "public_artifact_safe": True,
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"] = sorted(set(_list(report.get("blockers")) + ["public_redaction_scan_failed"]))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage-runtime-plan-report", required=True)
    parser.add_argument("--kaggle-owner", default="tpuowner")
    parser.add_argument("--provider-owner-map", default="")
    parser.add_argument("--title-prefix", default="ct-glm52-stage-worker")
    parser.add_argument("--coordinator-request-id-hash", default="")
    parser.add_argument("--runtime-kind", choices=RUNTIME_KINDS, default=RUNTIME_KIND_VALUE_OP)
    parser.add_argument("--full-prefix-probe-mode", choices=FULL_PREFIX_PROBE_MODES, default=FULL_PREFIX_PROBE_MODE_DEFAULT)
    parser.add_argument("--full-prefix-timeout-seconds", type=int, default=3600)
    parser.add_argument("--cpu-stage-group-size", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.cpu_stage_group_size < 1:
        raise SystemExit("--cpu-stage-group-size must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_report(args)
    path = output_dir / "glm52_kaggle_stage_worker_package.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"Worker package ready: {report.get('glm52_stage_worker_package_ready')}")
        print(f"Live run performed: {report.get('live_run_performed')}")
    return 0 if report.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
