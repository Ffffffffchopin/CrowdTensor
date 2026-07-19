#!/usr/bin/env python3
"""Run a bounded 4*T4 + 5*CPU full-precision 32B heterogeneous Kaggle probe."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import kaggle_32b_stage_owned_safetensors_probe as loading_probe

try:
    from scripts import gpu_tpu_cpu_same_request_runtime_bridge_probe as web_tpu_bridge
    from scripts import colab_cli_runtime
    from scripts import colab_cuda_session_manager
except Exception:  # pragma: no cover - live Web TPU support can still fail closed.
    web_tpu_bridge = None  # type: ignore[assignment]
    colab_cli_runtime = None  # type: ignore[assignment]
    colab_cuda_session_manager = None  # type: ignore[assignment]


SCHEMA = "kaggle_32b_full_heterogeneous_probe_v1"
STAGE_REPORT_SCHEMA = "kaggle_32b_full_heterogeneous_stage_v1"
DEFAULT_OUTPUT_DIR = "dist/kaggle-32b-full-heterogeneous-probe"
DEFAULT_MODEL_REPO = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"
DEFAULT_PUBLIC_HOST = "24.199.118.54"
DEFAULT_PORT = 9242
DEFAULT_STAGE_RANGES = [[0, 10], [10, 22], [22, 34], [34, 46], [46, 50], [50, 54], [54, 58], [58, 62], [62, 64]]
COLAB_CUDA_STAGE_REPORT_MARKER = "CT_COLAB_CUDA_STAGE_REPORT"
Runner = Callable[..., subprocess.CompletedProcess[str]]


KERNEL_TEMPLATE = r'''
from __future__ import annotations

import base64
import gc
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "__STAGE_REPORT_SCHEMA__"
MODE = "__MODE__"
MODEL_REPO = __MODEL_REPO_JSON__
KAGGLE_ATTACHED_MODEL_PATHS = __KAGGLE_ATTACHED_MODEL_PATHS_JSON__
STAGE_IDS = __STAGE_IDS_JSON__
STAGE_RANGES = __STAGE_RANGES_JSON__
RESOURCE_KIND = "__RESOURCE_KIND__"
PROMPT_TEXT = __PROMPT_TEXT_JSON__
COORDINATOR_URL = __COORDINATOR_URL_JSON__
COORDINATOR_TOKEN = __COORDINATOR_TOKEN_JSON__
MAX_NEW_TOKENS = __MAX_NEW_TOKENS__
TASK_POLL_INTERVAL_SECONDS = __TASK_POLL_INTERVAL_SECONDS__
TASK_IDLE_TIMEOUT_SECONDS = __TASK_IDLE_TIMEOUT_SECONDS__
CPU_DTYPE = "__CPU_DTYPE__"
OUT = Path("/kaggle/working")
TEMP = Path("/kaggle/temp/ct_32b_full_heterogeneous") / MODE
MODEL_DIR = TEMP / "model"
REPORT_PATH = OUT / f"ct_32b_full_heterogeneous_{MODE}_report.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha_payload(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def sha_text(value):
    return "sha256:" + hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()


def safe_tail(value, limit=1200):
    text = str(value or "")[-limit:]
    for fragment in [PROMPT_TEXT, "KAGGLE_KEY", "KAGGLE_USERNAME", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "Bearer "]:
        if fragment:
            text = text.replace(fragment, "<redacted>")
    if COORDINATOR_TOKEN:
        text = text.replace(COORDINATOR_TOKEN, "<redacted>")
    return text


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def http_json(method, path, payload=None, timeout=120):
    body = None
    headers = {"Content-Type": "application/json", "X-CrowdTensor-32B-Token": COORDINATOR_TOKEN}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        COORDINATOR_URL.rstrip("/") + path,
        data=body,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        loaded = json.loads(response.read().decode("utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def run_command(command, timeout=300):
    started = time.monotonic()
    try:
        completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "duration_seconds": round(time.monotonic() - started, 3)}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout_tail": safe_tail(completed.stdout),
        "stderr_tail": safe_tail(completed.stderr),
    }


def hardware_summary():
    if RESOURCE_KIND == "cpu":
        return {
            "nvidia_smi_ok": False,
            "gpu_count": 0,
            "gpu_names": [],
            "vram_total_mb": [],
            "vram_free_mb": [],
            "kaggle_gpu_verified": False,
            "cpu_kernel": True,
        }
    smi = run_command(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"], timeout=60)
    names = []
    memory_total_mb = []
    memory_free_mb = []
    if smi.get("ok"):
        for raw in str(smi.get("stdout_tail") or "").splitlines():
            parts = [part.strip() for part in raw.split(",")]
            if len(parts) >= 3:
                names.append(parts[0])
                memory_total_mb.append(int("".join(ch for ch in parts[1] if ch.isdigit()) or 0))
                memory_free_mb.append(int("".join(ch for ch in parts[2] if ch.isdigit()) or 0))
    return {
        "nvidia_smi_ok": bool(smi.get("ok")),
        "gpu_count": len(names),
        "gpu_names": names,
        "vram_total_mb": memory_total_mb,
        "vram_free_mb": memory_free_mb,
        "kaggle_gpu_verified": bool(names),
        "nvidia_smi": smi,
    }


def memory_summary():
    fields = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            digits = "".join(ch for ch in value if ch.isdigit())
            if digits:
                fields[key] = int(digits) // 1024
    except Exception:
        pass
    return {
        "mem_total_mb": fields.get("MemTotal", 0),
        "mem_available_mb": fields.get("MemAvailable", 0),
    }


def dependency_setup():
    required = ["torch", "transformers", "accelerate", "safetensors"]
    missing = []
    for name in required:
        try:
            __import__(name)
        except ModuleNotFoundError:
            missing.append(name)
    step = {"ok": True, "missing_before_install": missing, "install_attempted": False}
    if missing:
        step["install_attempted"] = True
        install = run_command([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.40.0", "accelerate", "safetensors", "sentencepiece"], timeout=600)
        step["install_step"] = install
        step["ok"] = bool(install.get("ok"))
    versions = {}
    try:
        import torch
        versions["torch"] = str(torch.__version__)
        versions["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        versions["torch_error"] = type(exc).__name__
        step["ok"] = False
    try:
        import transformers
        versions["transformers"] = str(transformers.__version__)
    except Exception as exc:
        versions["transformers_error"] = type(exc).__name__
        step["ok"] = False
    step["versions"] = versions
    return step


def attached_file(filename):
    for raw in list(KAGGLE_ATTACHED_MODEL_PATHS or []):
        path = Path(str(raw or "")) / filename
        if path.is_file():
            return path
    return None


def fetch_json(filename):
    attached = attached_file(filename)
    if attached is not None:
        return json.loads(attached.read_text(encoding="utf-8"))
    url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{filename}"
    with urllib.request.urlopen(url, timeout=120) as response:
        loaded = json.load(response)
    return loaded if isinstance(loaded, dict) else {}


def fetch_weight_index():
    try:
        index = fetch_json("model.safetensors.index.json")
        if isinstance(index.get("weight_map"), dict) and index.get("weight_map"):
            index["single_file_safetensors_fallback"] = False
            return index
    except Exception:
        pass
    single = download_file("model.safetensors")
    from safetensors import safe_open
    path = MODEL_DIR / "model.safetensors"
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = sorted(str(key) for key in handle.keys())
    return {
        "metadata": {"total_size": int(path.stat().st_size) if path.is_file() else int(single.get("size_bytes") or 0)},
        "weight_map": {key: "model.safetensors" for key in keys},
        "single_file_safetensors_fallback": True,
        "single_file_safetensors_key_count": len(keys),
    }


def model_family_from_config(config):
    model_type = str(config.get("model_type") or "").lower()
    if model_type == "bloom":
        return "bloom"
    return "llama_like"


def stage_prefixes(stage_id, stage_count, layer_range, config=None):
    start, end = int(layer_range[0]), int(layer_range[1])
    family = model_family_from_config(config or {})
    if family == "bloom":
        prefixes = [f"h.{index}." for index in range(start, end)]
        if int(stage_id) == 0:
            prefixes = ["word_embeddings.", "word_embeddings_layernorm.", *prefixes]
        if int(stage_id) == int(stage_count) - 1:
            prefixes = [*prefixes, "ln_f.", "lm_head."]
        return prefixes

    prefixes = [f"model.layers.{index}." for index in range(start, end)]
    if int(stage_id) == 0:
        prefixes = ["model.embed_tokens.", *prefixes]
    if int(stage_id) == int(stage_count) - 1:
        prefixes = [*prefixes, "model.norm.", "lm_head."]
    return prefixes


def build_selection(config, weight_index, stage_id):
    weight_map = {
        str(key): Path(str(value)).name
        for key, value in dict(weight_index.get("weight_map") or {}).items()
        if str(key or "").strip() and str(value or "").strip()
    }
    stage_count = len(STAGE_RANGES)
    selected_stage_id = int(stage_id)
    layer_count = int(config.get("num_hidden_layers") or config.get("n_layer") or 0)
    layer_range = list(STAGE_RANGES[selected_stage_id])
    prefixes = stage_prefixes(selected_stage_id, stage_count, layer_range, config)
    assigned = sorted(key for key in weight_map if any(key.startswith(prefix) for prefix in prefixes))
    assigned_files = sorted({weight_map[key] for key in assigned if weight_map.get(key)})
    return {
        "model_type": str(config.get("model_type") or ""),
        "architectures": list(config.get("architectures") or []),
        "quantization_config": dict(config.get("quantization_config") or {}),
        "num_hidden_layers": layer_count,
        "hidden_size": int(config.get("hidden_size") or config.get("n_embd") or 0),
        "vocab_size": int(config.get("vocab_size") or 0),
        "stage_id": int(selected_stage_id),
        "stage_count": stage_count,
        "stage_layer_range": [int(layer_range[0]), int(layer_range[1])],
        "expected_key_prefixes": prefixes,
        "assigned_weight_keys": assigned,
        "assigned_weight_key_count": len(assigned),
        "assigned_weight_files": assigned_files,
        "assigned_weight_file_count": len(assigned_files),
        "all_weight_file_count": len(set(weight_map.values())),
        "weight_key_count": len(weight_map),
        "weight_map": weight_map,
        "total_size_bytes": int(dict(weight_index.get("metadata") or {}).get("total_size") or 0),
    }


def download_file(filename):
    attached = attached_file(Path(filename).name)
    if attached is not None:
        return {
            "filename": Path(filename).name,
            "size_bytes": int(attached.stat().st_size),
            "size_mb": round(attached.stat().st_size / 1024 / 1024, 3),
            "duration_seconds": 0.0,
            "attached_model_source": True,
            "path_public": False,
        }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = MODEL_DIR / Path(filename).name
    if target.is_file() and target.stat().st_size > 0:
        return {
            "filename": Path(filename).name,
            "size_bytes": int(target.stat().st_size),
            "size_mb": round(target.stat().st_size / 1024 / 1024, 3),
            "duration_seconds": 0.0,
            "cache_hit": True,
        }
    started = time.monotonic()
    size = 0
    with urllib.request.urlopen(f"https://huggingface.co/{MODEL_REPO}/resolve/main/{filename}", timeout=120) as response:
        with target.open("wb") as handle:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                size += len(chunk)
    return {
        "filename": Path(filename).name,
        "size_bytes": int(size),
        "size_mb": round(size / 1024 / 1024, 3),
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def module_by_name(root, dotted):
    current = root
    for part in str(dotted or "").split("."):
        if not part:
            continue
        if part.isdigit() and isinstance(current, (list, tuple)):
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def runtime_dtype(device):
    import torch
    if device.type == "cuda":
        return torch.float16
    if str(CPU_DTYPE) == "float32":
        return torch.float32
    if str(CPU_DTYPE) == "float16":
        return torch.float16
    return torch.bfloat16


def materialize_runtime_buffers(model, device):
    import torch
    materialized = []
    blockers = []
    config = getattr(model, "config", None)
    hidden_size = int(getattr(config, "hidden_size", 0) or 0)
    heads = int(getattr(config, "num_attention_heads", 1) or 1)
    dim = int(getattr(config, "head_dim", 0) or (hidden_size // max(1, heads)))
    theta = float(getattr(config, "rope_theta", 1000000.0) or 1000000.0)
    for name, buffer in list(model.named_buffers(recurse=True)):
        if not bool(getattr(buffer, "is_meta", False)):
            continue
        replacement = None
        if name.endswith("rotary_emb.inv_freq") or name.endswith("rotary_emb.original_inv_freq"):
            replacement = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / max(1, dim)))
        if replacement is None:
            blockers.append("runtime_buffer_materialization_missing:" + name)
            continue
        module_path, attr = name.rsplit(".", 1)
        module = module_by_name(model, module_path)
        module.register_buffer(attr, replacement, persistent=attr not in getattr(module, "_non_persistent_buffers_set", set()))
        materialized.append(name)
    remaining = [
        name for name, buffer in model.named_buffers(recurse=True)
        if bool(getattr(buffer, "is_meta", False))
    ]
    return {
        "ready": not blockers and not remaining,
        "materialized_runtime_buffer_count": len(materialized),
        "remaining_meta_buffer_count": len(remaining),
        "materialized_runtime_buffer_digest": sha_payload(sorted(materialized)),
        "blockers": sorted(set(blockers)),
    }


def load_stage_weights(model, selection, device):
    import torch
    from safetensors.torch import safe_open

    assigned = list(selection.get("assigned_weight_keys") or [])
    assigned_set = set(assigned)
    weight_map = dict(selection.get("weight_map") or {})
    state = model.state_dict()
    assign_state = {}
    loaded = []
    loaded_bytes = 0
    prepared_bytes = 0
    missing = []
    shape_mismatches = []
    downloads = []
    dtype = runtime_dtype(device)

    def state_key_for_weight_key(key):
        if key in state:
            return key
        model_type = str(selection.get("model_type") or "").lower()
        if model_type == "bloom" and key != "lm_head.weight":
            prefixed = "transformer." + str(key)
            if prefixed in state:
                return prefixed
        return key

    for filename in list(selection.get("assigned_weight_files") or []):
        downloads.append(download_file(filename))
    for filename in list(selection.get("assigned_weight_files") or []):
        attached = attached_file(Path(filename).name)
        path = attached if attached is not None else MODEL_DIR / Path(filename).name
        with safe_open(path, framework="pt", device="cpu") as handle:
            available = set(str(key) for key in handle.keys())
            expected = [key for key in assigned if weight_map.get(key) == Path(filename).name]
            for key in expected:
                if key not in available:
                    missing.append(key)
                    continue
                state_key = state_key_for_weight_key(key)
                if state_key not in state:
                    missing.append(key)
                    continue
                source = handle.get_tensor(key)
                target = state[state_key]
                if tuple(source.shape) != tuple(target.shape):
                    shape_mismatches.append(key)
                    continue
                prepared = source.to(device=device, dtype=dtype)
                assign_state[state_key] = prepared
                loaded.append(key)
                loaded_bytes += int(source.numel()) * int(source.element_size())
                prepared_bytes += int(prepared.numel()) * int(prepared.element_size())
                del source
    if assign_state:
        model.load_state_dict(assign_state, strict=False, assign=True)
    loaded_set = set(loaded)
    ready = bool(loaded and loaded_set == assigned_set and not missing and not shape_mismatches)
    del assign_state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "ready": ready,
        "downloads": downloads,
        "loaded_weight_key_count": len(loaded_set),
        "assigned_weight_key_count": len(assigned_set),
        "loaded_tensor_bytes": int(loaded_bytes),
        "loaded_tensor_gb": round(loaded_bytes / 1024 / 1024 / 1024, 6),
        "prepared_tensor_bytes": int(prepared_bytes),
        "prepared_tensor_gb": round(prepared_bytes / 1024 / 1024 / 1024, 6),
        "loaded_weight_key_digest": sha_payload(sorted(loaded)),
        "missing_weight_key_count": len(missing),
        "shape_mismatch_count": len(shape_mismatches),
        "loads_only_stage_weight_keys": bool(loaded_set.issubset(assigned_set)),
        "runtime_dtype": str(dtype).replace("torch.", ""),
        "blockers": ([] if ready else ["stage_weight_apply_not_ready"]),
    }


def prepare_stage_model(config, selection, device):
    import gc
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM

    config.torch_dtype = runtime_dtype(device)
    with init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(config)
    buffers = materialize_runtime_buffers(model, device)
    stage_load = load_stage_weights(model, selection, device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.eval()
    return model, buffers, stage_load


def causal_attention_mask(token_count, dtype, device):
    import torch
    mask_dtype = dtype if device.type == "cuda" else torch.float32
    mask = torch.full((token_count, token_count), torch.finfo(mask_dtype).min, dtype=mask_dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :].to(dtype=dtype if device.type == "cuda" else torch.float32)


def attention_mask_for(query_count, key_value_count, dtype, device):
    import torch
    query_count = max(1, int(query_count))
    key_value_count = max(query_count, int(key_value_count))
    if query_count == key_value_count:
        return causal_attention_mask(query_count, dtype, device)
    mask_dtype = dtype if device.type == "cuda" else torch.float32
    return torch.zeros((1, 1, query_count, key_value_count), dtype=mask_dtype, device=device).to(
        dtype=dtype if device.type == "cuda" else torch.float32
    )


def cache_position(token_count, device):
    import torch
    return torch.arange(max(1, int(token_count)), dtype=torch.long, device=device)


def cache_position_from(start, end, device):
    import torch
    return torch.arange(max(0, int(start)), max(1, int(end)), dtype=torch.long, device=device)


def new_dynamic_cache(model):
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return None
    try:
        return DynamicCache(config=getattr(model, "config", None))
    except Exception:
        try:
            return DynamicCache()
        except Exception:
            return None


def cache_seq_length(cache, layer_index=0):
    if cache is None:
        return 0
    getter = getattr(cache, "get_seq_length", None)
    if not callable(getter):
        return 0
    try:
        return int(getter(int(layer_index)) or 0)
    except TypeError:
        try:
            return int(getter() or 0)
        except Exception:
            return 0
    except Exception:
        return 0


def llama_position_embeddings(base_model, hidden, position_ids):
    rotary = getattr(base_model, "rotary_emb", None)
    if rotary is None:
        return None
    try:
        return rotary(hidden, position_ids)
    except Exception:
        inv_freq = getattr(rotary, "inv_freq", None)
        if inv_freq is None or bool(getattr(inv_freq, "is_meta", False)):
            return None
        import torch
        inv = inv_freq.to(device=hidden.device, dtype=torch.float32)
        pos = position_ids.to(device=hidden.device, dtype=torch.float32)
        freqs = torch.einsum("i,bj->bji", inv, pos)
        emb = torch.cat((freqs, freqs), dim=-1)
        scaling = float(getattr(rotary, "attention_scaling", 1.0) or 1.0)
        return emb.cos().to(dtype=hidden.dtype) * scaling, emb.sin().to(dtype=hidden.dtype) * scaling


def call_layer(layer, hidden, attention_mask, position_ids, cache_pos, position_embeddings, *, kv_cache=None, use_cache=False):
    try:
        params = inspect.signature(layer.forward).parameters
    except Exception:
        params = {}
    kwargs = {}
    if "attention_mask" in params:
        kwargs["attention_mask"] = attention_mask
    if "position_ids" in params:
        kwargs["position_ids"] = position_ids
    if "past_key_value" in params:
        kwargs["past_key_value"] = kv_cache if use_cache else None
    if "past_key_values" in params:
        kwargs["past_key_values"] = kv_cache if use_cache else None
    if "output_attentions" in params:
        kwargs["output_attentions"] = False
    if "use_cache" in params:
        kwargs["use_cache"] = bool(use_cache)
    if "cache_position" in params:
        kwargs["cache_position"] = cache_pos
    if "position_embeddings" in params and position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings
    return layer(hidden, **kwargs)


def output_hidden(output):
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


def model_family(model):
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "") or "").lower()
    if model_type == "bloom":
        return "bloom"
    return "llama_like"


def base_model_for(model):
    if model_family(model) == "bloom":
        return model.transformer
    return model.model


def transformer_layers_for(model):
    base = base_model_for(model)
    if model_family(model) == "bloom":
        return list(base.h)
    return list(base.layers)


def embed_tokens_for(model, input_ids, dtype):
    base = base_model_for(model)
    if model_family(model) == "bloom":
        hidden = base.word_embeddings(input_ids)
        return base.word_embeddings_layernorm(hidden).to(dtype=dtype)
    return base.embed_tokens(input_ids).to(dtype=dtype)


def final_norm_for(model, hidden):
    base = base_model_for(model)
    if model_family(model) == "bloom":
        return base.ln_f(hidden)
    return base.norm(hidden)


def bloom_attention_inputs(model, hidden, key_value_count):
    import torch
    from transformers.models.bloom.modeling_bloom import create_causal_mask

    base = base_model_for(model)
    attention_mask_2d = torch.ones((int(hidden.shape[0]), int(key_value_count)), device=hidden.device)
    alibi = base.build_alibi_tensor(attention_mask_2d, base.num_heads, dtype=hidden.dtype)
    causal_mask = create_causal_mask(
        config=getattr(model, "config", None),
        inputs_embeds=hidden,
        attention_mask=attention_mask_2d,
        past_key_values=None,
    )
    return alibi, causal_mask


def call_bloom_layer(layer, hidden, alibi, attention_mask, *, kv_cache=None, use_cache=False):
    output = layer(
        hidden,
        layer_past=kv_cache if use_cache else None,
        attention_mask=attention_mask,
        use_cache=bool(use_cache),
        output_attentions=False,
        alibi=alibi,
    )
    return output_hidden(output)


def activation_payload_hash(payload):
    return sha_payload({
        "schema": payload["schema"],
        "model_repo": payload["model_repo"],
        "stage_count": payload.get("stage_count"),
        "stage_id": payload.get("stage_id"),
        "stage_layer_range": payload.get("stage_layer_range"),
        "generation_step": payload.get("generation_step"),
        "input_ids": payload["input_ids"],
        "position_ids": payload["position_ids"],
        "hidden_shape": payload["hidden_shape"],
        "hidden_dtype": payload["hidden_dtype"],
        "hidden_b64": payload["hidden_b64"],
    })


def encode_activation_payload(*, hidden, input_ids, position_ids, selection, task_id="", generation_step=0):
    import torch
    hidden_cpu = hidden.detach().to(device="cpu", dtype=torch.float16).contiguous()
    hidden_b64 = base64.b64encode(hidden_cpu.numpy().tobytes()).decode("ascii")
    payload = {
        "schema": "kaggle_32b_full_private_activation_v1",
        "model_repo": MODEL_REPO,
        "stage_count": int(selection.get("stage_count") or len(STAGE_RANGES)),
        "stage_id": int(selection.get("stage_id") or 0),
        "task_id": str(task_id or ""),
        "generation_step": int(generation_step or 0),
        "stage_layer_range": selection.get("stage_layer_range"),
        "prompt_hash": sha_text(PROMPT_TEXT),
        "input_ids": [int(value) for value in input_ids.detach().cpu().tolist()[0]],
        "position_ids": [int(value) for value in position_ids.detach().cpu().tolist()[0]],
        "hidden_shape": [int(value) for value in hidden_cpu.shape],
        "hidden_dtype": "float16",
        "hidden_b64": hidden_b64,
    }
    payload["activation_hash"] = activation_payload_hash(payload)
    return payload


def public_kv_cache_summary(stage_cache):
    hit_count = int(stage_cache.get("hit_count") or 0)
    expected_hit_count = max(0, int(MAX_NEW_TOKENS) - 1)
    return {
        "schema": "kaggle_32b_full_stage_local_kv_cache_v1",
        "stage_id": int(stage_cache.get("stage_id") or 0),
        "ready": bool(stage_cache.get("ready")),
        "hit_count": hit_count,
        "miss_count": int(stage_cache.get("miss_count") or 0),
        "prefill_count": int(stage_cache.get("prefill_count") or 0),
        "expected_hit_count": expected_hit_count,
        "hit_target_ready": bool(hit_count >= expected_hit_count),
        "tokens_before": int(stage_cache.get("last_tokens_before") or 0),
        "tokens_after": int(stage_cache.get("tokens_after") or 0),
        "last_input_token_count": int(stage_cache.get("last_input_token_count") or 0),
        "last_cache_seq_length": int(stage_cache.get("last_cache_seq_length") or 0),
        "last_cache_hit": bool(stage_cache.get("last_cache_hit")),
        "last_cache_ready": bool(stage_cache.get("last_cache_ready")),
        "cache_tensors_public": False,
        "past_key_values_public": False,
        "kv_cache_transport_public": False,
    }


def run_layers(model, hidden, position_ids, device, *, start, end, input_token_ids, stage_cache=None, generation_step=0):
    import torch
    base = base_model_for(model)
    layers = transformer_layers_for(model)
    stage_cache = stage_cache if isinstance(stage_cache, dict) else {}
    input_token_ids = [int(value) for value in list(input_token_ids or [])]
    full_token_count = max(1, len(input_token_ids))
    query_count = int(hidden.shape[1])
    cached_input = [int(value) for value in list(stage_cache.get("input_token_ids") or [])]
    cache_hit = bool(
        generation_step > 0
        and cached_input
        and len(cached_input) == full_token_count - query_count
        and cached_input == input_token_ids[:len(cached_input)]
        and stage_cache.get("cache") is not None
    )
    if generation_step > 0 and not cache_hit and query_count < full_token_count:
        raise RuntimeError("stage_local_kv_cache_required_for_delta_activation")
    kv_cache = stage_cache.get("cache") if cache_hit else new_dynamic_cache(model)
    cache_tokens_before = len(cached_input) if cache_hit else 0
    key_value_count = cache_tokens_before + query_count
    if key_value_count < full_token_count:
        key_value_count = full_token_count
    with torch.no_grad():
        if model_family(model) == "bloom":
            alibi, attention_mask = bloom_attention_inputs(model, hidden, key_value_count)
            for index, layer in enumerate(layers[int(start):int(end)], start=int(start)):
                hidden = call_bloom_layer(
                    layer,
                    hidden,
                    alibi,
                    attention_mask,
                    kv_cache=kv_cache,
                    use_cache=kv_cache is not None,
                )
                if index % 2 == 0:
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
        else:
            attention_mask = attention_mask_for(query_count, key_value_count, hidden.dtype, device)
            cache_pos = cache_position_from(cache_tokens_before, cache_tokens_before + query_count, device)
            position_embeddings = llama_position_embeddings(base, hidden, position_ids)
            for index, layer in enumerate(layers[int(start):int(end)], start=int(start)):
                hidden = output_hidden(call_layer(
                    layer,
                    hidden,
                    attention_mask,
                    position_ids,
                    cache_pos,
                    position_embeddings,
                    kv_cache=kv_cache,
                    use_cache=kv_cache is not None,
                ))
                if index % 2 == 0:
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    seq_length = cache_seq_length(kv_cache, int(start))
    cache_ready = bool(kv_cache is not None and seq_length >= full_token_count)
    stage_cache["cache"] = kv_cache
    stage_cache["stage_id"] = int(stage_cache.get("stage_id") or 0)
    stage_cache["input_token_ids"] = input_token_ids
    stage_cache["tokens_after"] = full_token_count
    stage_cache["last_input_token_count"] = full_token_count
    stage_cache["last_tokens_before"] = cache_tokens_before
    stage_cache["last_cache_seq_length"] = seq_length
    stage_cache["last_cache_hit"] = cache_hit
    stage_cache["last_cache_ready"] = cache_ready
    stage_cache["ready"] = bool(stage_cache.get("ready") or cache_ready)
    if cache_hit:
        stage_cache["hit_count"] = int(stage_cache.get("hit_count") or 0) + 1
    else:
        stage_cache["miss_count"] = int(stage_cache.get("miss_count") or 0) + 1
        if generation_step == 0:
            stage_cache["prefill_count"] = int(stage_cache.get("prefill_count") or 0) + 1
    return hidden, public_kv_cache_summary(stage_cache)


def run_stage_activation(model, tokenizer, selection, device, *, input_ids_values=None, activation=None, task_id="", generation_step=0, stage_cache=None):
    import torch
    dtype = runtime_dtype(device)
    stage_range = list(selection.get("stage_layer_range") or [0, 0])
    start = int(stage_range[0])
    end = int(stage_range[1])
    if input_ids_values:
        input_ids = torch.tensor([list(int(value) for value in input_ids_values)], dtype=torch.long, device=device)
    elif activation:
        input_ids = torch.tensor([list(int(value) for value in activation.get("input_ids") or [])], dtype=torch.long, device=device)
    else:
        encoded = tokenizer(PROMPT_TEXT, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(device)
    if activation:
        shape = [int(value) for value in list(activation.get("hidden_shape") or [])]
        raw = base64.b64decode(str(activation.get("hidden_b64") or ""))
        hidden = torch.frombuffer(bytearray(raw), dtype=torch.float16).reshape(shape).to(device=device, dtype=dtype)
        position_ids = torch.tensor([list(activation.get("position_ids") or range(hidden.shape[1]))], dtype=torch.long, device=device)
    else:
        input_values = [int(value) for value in input_ids.detach().cpu().tolist()[0]]
        cached_input = [int(value) for value in list((stage_cache or {}).get("input_token_ids") or [])]
        can_use_delta = bool(generation_step > 0 and cached_input == input_values[:-1] and (stage_cache or {}).get("cache") is not None)
        if can_use_delta:
            position_ids = torch.tensor([[len(input_values) - 1]], dtype=torch.long, device=device)
            hidden = embed_tokens_for(model, input_ids[:, -1:], dtype)
        else:
            position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
            hidden = embed_tokens_for(model, input_ids, dtype)
    input_values = [int(value) for value in input_ids.detach().cpu().tolist()[0]]
    hidden, kv_cache = run_layers(
        model,
        hidden,
        position_ids,
        device,
        start=start,
        end=end,
        input_token_ids=input_values,
        stage_cache=stage_cache,
        generation_step=generation_step,
    )
    activation_payload = encode_activation_payload(
        hidden=hidden,
        input_ids=input_ids,
        position_ids=position_ids,
        selection=selection,
        task_id=task_id,
        generation_step=generation_step,
    )
    return activation_payload, kv_cache


def final_stage_decode(model, tokenizer, selection, activation, device, *, stage_cache=None, generation_step=0):
    import torch
    dtype = runtime_dtype(device)
    stage_range = list(selection.get("stage_layer_range") or [0, 0])
    start = int(stage_range[0])
    end = int(stage_range[1])
    shape = [int(value) for value in list(activation.get("hidden_shape") or [])]
    raw = base64.b64decode(str(activation.get("hidden_b64") or ""))
    hidden = torch.frombuffer(bytearray(raw), dtype=torch.float16).reshape(shape).to(device=device, dtype=dtype)
    position_ids = torch.tensor([list(activation.get("position_ids") or range(hidden.shape[1]))], dtype=torch.long, device=device)
    input_values = [int(value) for value in list(activation.get("input_ids") or [])]
    hidden, kv_cache = run_layers(
        model,
        hidden,
        position_ids,
        device,
        start=start,
        end=end,
        input_token_ids=input_values,
        stage_cache=stage_cache,
        generation_step=generation_step,
    )
    with torch.no_grad():
        hidden = final_norm_for(model, hidden)
        logits = model.lm_head(hidden)
        next_token_id = int(torch.argmax(logits[0, -1, :]).item())
    next_text = tokenizer.decode([next_token_id], skip_special_tokens=False)
    return {
        "generated_token_count": 1,
        "next_token_id_private": next_token_id,
        "next_token_id_public": False,
        "next_token_text_public": False,
        "next_token_hash": sha_payload({"next_token_id": next_token_id, "next_token_text": next_text}),
        "output_hash": sha_payload({"activation_hash": activation.get("activation_hash"), "next_token_id": next_token_id}),
        "kv_cache": kv_cache,
    }


def cuda_memory_summary(device=None):
    try:
        import torch
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        target_device = device if device is not None else None
        return {
            "cuda_available": True,
            "allocated_mb": round(torch.cuda.memory_allocated(target_device) / 1024 / 1024, 3),
            "reserved_mb": round(torch.cuda.memory_reserved(target_device) / 1024 / 1024, 3),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated(target_device) / 1024 / 1024, 3),
        }
    except Exception as exc:
        return {"cuda_available": False, "error_type": type(exc).__name__}


def device_for_runtime(index):
    import torch
    if RESOURCE_KIND == "gpu":
        return torch.device(f"cuda:{int(index)}")
    return torch.device("cpu")


def worker_loop(stage_runtime, tokenizer, report):
    processed = []
    accepted_counts = {int(item["stage_id"]): 0 for item in stage_runtime}
    deadline = time.monotonic() + float(TASK_IDLE_TIMEOUT_SECONDS)
    miner_id = "kaggle-32b-full-" + MODE
    while time.monotonic() < deadline:
        if accepted_counts and all(count >= int(MAX_NEW_TOKENS) for count in accepted_counts.values()):
            return processed
        claim = {}
        runtime = None
        claim_error = None
        for candidate in stage_runtime:
            try:
                claim = http_json("POST", "/claim", {"miner_id": miner_id, "stage_id": candidate["stage_id"]}, timeout=120)
            except Exception as exc:
                claim_error = exc
                continue
            task = claim.get("task") if isinstance(claim.get("task"), dict) else {}
            if task:
                runtime = candidate
                break
            if claim.get("done"):
                return processed
        if runtime is None and claim_error is not None:
            processed.append({
                "event": "claim_error",
                "error_type": type(claim_error).__name__,
                "error_digest": sha_text(str(claim_error)),
            })
            time.sleep(max(5.0, float(TASK_POLL_INTERVAL_SECONDS)))
            continue
        if claim.get("done"):
            break
        task = claim.get("task") if isinstance(claim.get("task"), dict) else {}
        if not task:
            time.sleep(max(5.0, float(TASK_POLL_INTERVAL_SECONDS)))
            continue
        task_id = str(task.get("task_id") or "")
        generation_step = int(task.get("generation_step") or 0)
        task_started = time.monotonic()
        stage_id = int(runtime["stage_id"])
        model = runtime["model"]
        selection = runtime["selection"]
        device = runtime["device"]
        stage_cache = runtime["kv_cache"]
        if stage_id == 0:
            activation, kv_cache = run_stage_activation(
                model,
                tokenizer,
                selection,
                device,
                input_ids_values=task.get("input_ids") or [],
                task_id=task_id,
                generation_step=generation_step,
                stage_cache=stage_cache,
            )
            result = {
                "task_id": task_id,
                "stage_id": stage_id,
                "generation_step": generation_step,
                "activation": activation,
                "activation_hash": activation.get("activation_hash"),
                "input_token_count": len(activation.get("input_ids") or []),
                "kv_cache": kv_cache,
                "duration_seconds": round(time.monotonic() - task_started, 3),
                "cuda_memory_after_task": cuda_memory_summary(device),
            }
        elif stage_id < len(STAGE_RANGES) - 1:
            activation = dict(task.get("activation") or {})
            next_activation, kv_cache = run_stage_activation(
                model,
                tokenizer,
                selection,
                device,
                activation=activation,
                task_id=task_id,
                generation_step=generation_step,
                stage_cache=stage_cache,
            )
            result = {
                "task_id": task_id,
                "stage_id": stage_id,
                "generation_step": generation_step,
                "activation": next_activation,
                "activation_hash": next_activation.get("activation_hash"),
                "input_token_count": len(next_activation.get("input_ids") or []),
                "kv_cache": kv_cache,
                "duration_seconds": round(time.monotonic() - task_started, 3),
                "cuda_memory_after_task": cuda_memory_summary(device),
            }
        else:
            activation = dict(task.get("activation") or {})
            decoded = final_stage_decode(
                model,
                tokenizer,
                selection,
                activation,
                device,
                stage_cache=stage_cache,
                generation_step=generation_step,
            )
            result = {
                "task_id": task_id,
                "stage_id": stage_id,
                "generation_step": generation_step,
                "activation_hash": activation.get("activation_hash"),
                "next_token_id_private": decoded.get("next_token_id_private"),
                "next_token_hash": decoded.get("next_token_hash"),
                "output_hash": decoded.get("output_hash"),
                "generated_token_count": decoded.get("generated_token_count"),
                "kv_cache": decoded.get("kv_cache") if isinstance(decoded.get("kv_cache"), dict) else {},
                "duration_seconds": round(time.monotonic() - task_started, 3),
                "cuda_memory_after_task": cuda_memory_summary(device),
            }
        try:
            response = http_json("POST", "/submit", result, timeout=120)
            processed.append({
                "task_id": task_id,
                "stage_id": stage_id,
                "generation_step": generation_step,
                "accepted": response.get("accepted") is True,
                "duration_seconds": result.get("duration_seconds"),
                "activation_hash": result.get("activation_hash"),
                "output_hash": result.get("output_hash"),
                "generated_token_count": result.get("generated_token_count"),
                "kv_cache": result.get("kv_cache") if isinstance(result.get("kv_cache"), dict) else {},
            })
            if response.get("accepted") is True:
                accepted_counts[stage_id] = accepted_counts.get(stage_id, 0) + 1
            if response.get("ready") is True:
                break
            if accepted_counts and all(count >= int(MAX_NEW_TOKENS) for count in accepted_counts.values()):
                break
        except Exception as exc:
            processed.append({
                "task_id": task_id,
                "stage_id": stage_id,
                "generation_step": generation_step,
                "submit_error_type": type(exc).__name__,
                "submit_error_digest": sha_text(str(exc)),
                "submit_error_public": safe_tail(str(exc), limit=240),
            })
        report["processed_task_count"] = len([item for item in processed if item.get("accepted")])
        report["processed_tasks"] = processed
        write_json(REPORT_PATH, report)
    return processed


def main():
    started = time.monotonic()
    report = {
        "schema": SCHEMA,
        "mode": MODE,
        "resource_kind": RESOURCE_KIND,
        "stage_ids": STAGE_IDS,
        "stage_ranges": STAGE_RANGES,
        "max_new_tokens": MAX_NEW_TOKENS,
        "model_repo": MODEL_REPO,
        "ok": False,
        "full_precision": True,
        "quantization": "none",
        "public_safe": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "activation_public": False,
        "hidden_state_public": False,
        "credentials_public": False,
        "diagnosis_codes": [],
        "blockers": [],
        "started_at": utc_now(),
    }
    try:
        if TEMP.exists():
            shutil.rmtree(TEMP)
        TEMP.mkdir(parents=True, exist_ok=True)
        report["hardware"] = hardware_summary()
        report["memory_before"] = memory_summary()
        dependency = dependency_setup()
        report["dependency_setup"] = dependency
        if not dependency.get("ok"):
            report["blockers"].append("runtime_dependencies_missing")
            report["diagnosis_codes"].append("kaggle_32b_full_dependencies_missing")
            write_json(REPORT_PATH, report)
            return
        import torch
        from transformers import AutoConfig, AutoTokenizer
        if RESOURCE_KIND == "gpu" and not torch.cuda.is_available():
            report["blockers"].append("cuda_runtime_missing")
            report["diagnosis_codes"].append("kaggle_32b_full_cuda_missing")
            write_json(REPORT_PATH, report)
            return
        if RESOURCE_KIND == "gpu" and len(STAGE_IDS) > torch.cuda.device_count():
            report["blockers"].append("kernel_stage_count_exceeds_cuda_device_count")
            report["diagnosis_codes"].append("kaggle_32b_full_kernel_stage_count_exceeds_cuda_device_count")
            write_json(REPORT_PATH, report)
            return
        torch.set_grad_enabled(False)
        runtime_stage = "config_load"
        config_json = fetch_json("config.json")
        weight_index = fetch_weight_index()
        report["weight_index"] = {
            "single_file_safetensors_fallback": bool(weight_index.get("single_file_safetensors_fallback")),
            "single_file_safetensors_key_count": int(weight_index.get("single_file_safetensors_key_count") or 0),
            "weight_key_count": len(dict(weight_index.get("weight_map") or {})),
        }
        selections = [
            build_selection(config_json, weight_index, stage_id=stage_id)
            for stage_id in STAGE_IDS
        ]
        report["stage_selections"] = [
            {
                key: value
                for key, value in selection.items()
                if key not in {"assigned_weight_keys", "weight_map"}
            }
            for selection in selections
        ]
        if any(not selection["assigned_weight_keys"] for selection in selections):
            report["blockers"].append("stage_weight_selection_empty")
            report["diagnosis_codes"].append("kaggle_32b_full_stage_selection_empty")
            write_json(REPORT_PATH, report)
            return
        runtime_stage = "model_config_tokenizer_load"
        config = AutoConfig.from_pretrained(MODEL_REPO)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
        stage_runtime = []
        for index, selection in enumerate(selections):
            device = device_for_runtime(index)
            runtime_stage = f"stage{selection.get('stage_id')}_model_prepare"
            model, buffers, stage_load = prepare_stage_model(config, selection, device)
            stage_runtime.append({
                "stage_id": int(selection.get("stage_id")),
                "device": device,
                "selection": selection,
                "model": model,
                "buffers": buffers,
                "stage_load": stage_load,
                "kv_cache": {"stage_id": int(selection.get("stage_id"))},
                "cuda_memory_after_load": cuda_memory_summary(device),
                "memory_after_load": memory_summary(),
            })
        report["stage_runtime_summaries"] = [
            {
                "stage_id": item["stage_id"],
                "resource_kind": RESOURCE_KIND,
                "device": str(item["device"]),
                "selection": {
                    key: value
                    for key, value in item["selection"].items()
                    if key not in {"assigned_weight_keys", "weight_map"}
                },
                "runtime_buffers": item["buffers"],
                "stage_weight_load": {
                    key: value
                    for key, value in item["stage_load"].items()
                    if key != "downloads"
                },
                "kv_cache": public_kv_cache_summary(item["kv_cache"]),
                "cuda_memory_after_load": item["cuda_memory_after_load"],
                "memory_after_load": item["memory_after_load"],
            }
            for item in stage_runtime
        ]
        report["downloads"] = {
            f"stage{item['stage_id']}": item["stage_load"].get("downloads") or []
            for item in stage_runtime
        }
        if not all(item["buffers"].get("ready") and item["stage_load"].get("ready") for item in stage_runtime):
            report["blockers"].append("stage_runtime_not_ready")
            report["diagnosis_codes"].append("kaggle_32b_full_stage_runtime_not_ready")
            write_json(REPORT_PATH, report)
            return
        runtime_stage = "coordinator_worker_loop"
        tasks = worker_loop(stage_runtime, tokenizer, report)
        accepted = [item for item in tasks if item.get("accepted")]
        report.update({
            "ok": bool(accepted or MAX_NEW_TOKENS == 0),
            "worker_loop_ready": True,
            "processed_task_count": len(accepted),
            "kv_cache_summaries": {
                f"stage{item['stage_id']}": public_kv_cache_summary(item["kv_cache"])
                for item in stage_runtime
            },
            "diagnosis_codes": [
                *report["diagnosis_codes"],
                "kaggle_32b_full_coordinator_worker_ready",
                "kaggle_32b_full_stage_owned_runtime_ready",
                "kaggle_32b_full_stage_local_kv_cache_ready",
            ],
        })
        if not accepted:
            report["blockers"].append("coordinator_worker_processed_no_tasks")
        report["cuda_memory_after_execution"] = cuda_memory_summary()
        report["memory_after_execution"] = memory_summary()
    except Exception as exc:
        report["ok"] = False
        report["error_type"] = type(exc).__name__
        report["error_stage"] = locals().get("runtime_stage", "unknown")
        report["error_digest"] = sha_text(str(exc))
        report["error_public"] = safe_tail(str(exc), limit=240)
        report["diagnosis_codes"].append("kaggle_32b_full_heterogeneous_exception")
        report["blockers"].append("kaggle_32b_full_heterogeneous_exception")
    finally:
        try:
            if TEMP.exists():
                shutil.rmtree(TEMP)
            report["temp_cleanup"] = {"ok": True, "path_public": False}
        except Exception as exc:
            report["temp_cleanup"] = {"ok": False, "error_type": type(exc).__name__, "error_digest": sha_text(str(exc))}
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["updated_at"] = utc_now()
        write_json(REPORT_PATH, report)
        print(json.dumps({"schema": SCHEMA, "mode": MODE, "ok": report.get("ok"), "diagnosis_codes": report.get("diagnosis_codes")}, sort_keys=True))


main()
'''


def utc_now() -> str:
    return loading_probe.utc_now()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    loading_probe.write_json(path, payload)


def load_json(path: Path) -> dict[str, Any]:
    return loading_probe.load_json(path)


def sha_payload(value: Any) -> str:
    return "sha256:" + __import__("hashlib").sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def int_field(value: Any, default: int = -1) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def parameter_count_b(model_repo: str) -> float:
    text = str(model_repo or "").lower()
    match = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*b\b", text)
    if match:
        return float(match.group(1))
    if "bloom" in text:
        return 176.0
    for value in (235, 176, 150, 128, 123, 120, 110, 100, 72, 32, 14, 7):
        if f"{value}b" in text:
            return float(value)
    return 0.0


def expected_layer_count_for_model(model_repo: str) -> int:
    target_b = parameter_count_b(model_repo)
    if "bloom" in str(model_repo or "").lower():
        return 70
    if target_b == 72:
        return 80
    if target_b == 32:
        return 64
    return 0


def stage_ranges_cover_expected_layers(stage_ranges: list[list[int]], expected_layer_count: int) -> bool:
    if expected_layer_count <= 0 or not stage_ranges:
        return False
    cursor = 0
    for item in stage_ranges:
        if len(item) != 2:
            return False
        start, end = int(item[0]), int(item[1])
        if start != cursor or end <= start:
            return False
        cursor = end
    return cursor == int(expected_layer_count)


def stage_ranges_from_args(args: argparse.Namespace) -> list[list[int]]:
    if not str(args.stage_ranges_json or "").strip():
        return [list(item) for item in DEFAULT_STAGE_RANGES]
    loaded = json.loads(str(args.stage_ranges_json))
    if not isinstance(loaded, list) or not loaded:
        raise SystemExit("--stage-ranges-json must be a non-empty JSON list")
    ranges: list[list[int]] = []
    for item in loaded:
        if not isinstance(item, list) or len(item) != 2:
            raise SystemExit("--stage-ranges-json entries must be [start,end]")
        start, end = int(item[0]), int(item[1])
        if start < 0 or end < start:
            raise SystemExit("--stage-ranges-json contains an invalid range")
        ranges.append([start, end])
    return ranges


def initial_input_ids_from_args(args: argparse.Namespace) -> list[int]:
    text = str(getattr(args, "initial_input_ids_json", "") or "").strip()
    if not text:
        return []
    loaded = json.loads(text)
    if not isinstance(loaded, list):
        raise SystemExit("--initial-input-ids-json must be a JSON list")
    values = [int(item) for item in loaded]
    if len(values) > 16:
        raise SystemExit("--initial-input-ids-json must contain at most 16 token ids")
    if any(value < 0 for value in values):
        raise SystemExit("--initial-input-ids-json token ids must be non-negative")
    return values


def kaggle_model_sources(args: argparse.Namespace) -> list[str]:
    raw = str(getattr(args, "kaggle_model_sources_json", "") or "").strip()
    if not raw:
        return []
    loaded = json.loads(raw)
    if not isinstance(loaded, list):
        raise SystemExit("--kaggle-model-sources-json must be a JSON list")
    return [str(item) for item in loaded if str(item or "").strip()]


def kaggle_attached_model_paths(args: argparse.Namespace) -> list[str]:
    raw = str(getattr(args, "kaggle_attached_model_paths_json", "") or "").strip()
    if not raw:
        return []
    loaded = json.loads(raw)
    if not isinstance(loaded, list):
        raise SystemExit("--kaggle-attached-model-paths-json must be a JSON list")
    return [str(item) for item in loaded if str(item or "").strip()]


def stage_groups_for(args: argparse.Namespace) -> list[dict[str, Any]]:
    ranges = stage_ranges_from_args(args)
    if str(getattr(args, "stage_groups_json", "") or "").strip():
        loaded = json.loads(str(args.stage_groups_json))
        if not isinstance(loaded, list) or not loaded:
            raise SystemExit("--stage-groups-json must be a non-empty JSON list")
        groups: list[dict[str, Any]] = []
        for item in loaded:
            if not isinstance(item, dict):
                raise SystemExit("--stage-groups-json entries must be objects")
            mode = str(item.get("mode") or "").strip()
            resource_kind = str(item.get("resource_kind") or "").strip()
            stage_ids = item.get("stage_ids")
            if not mode:
                raise SystemExit("--stage-groups-json entries require mode")
            if resource_kind not in {"gpu", "cpu", "web_tpu", "colab_gpu"}:
                raise SystemExit("--stage-groups-json resource_kind must be gpu, cpu, web_tpu, or colab_gpu")
            if not isinstance(stage_ids, list) or not stage_ids:
                raise SystemExit("--stage-groups-json entries require non-empty stage_ids")
            clean_ids = [int(value) for value in stage_ids]
            if any(value < 0 or value >= len(ranges) for value in clean_ids):
                raise SystemExit("--stage-groups-json stage id out of range")
            if resource_kind == "web_tpu" and len(clean_ids) != 1:
                raise SystemExit("web_tpu stage groups must contain exactly one stage id")
            groups.append({"mode": mode, "stage_ids": clean_ids, "resource_kind": resource_kind})
        all_ids = [stage_id for group in groups for stage_id in group["stage_ids"]]
        if sorted(all_ids) != list(range(len(ranges))):
            raise SystemExit("--stage-groups-json must cover every stage exactly once")
        if not any(group["resource_kind"] == "gpu" for group in groups):
            raise SystemExit("--stage-groups-json must include at least one gpu group")
        if not any(group["resource_kind"] == "cpu" for group in groups):
            raise SystemExit("--stage-groups-json must include at least one cpu group")
        return groups
    if len(ranges) != 9:
        raise SystemExit("the full heterogeneous demo currently requires exactly 9 stages")
    return [
        {"mode": "gpu-shard0", "stage_ids": [0, 1], "resource_kind": "gpu"},
        {"mode": "gpu-shard1", "stage_ids": [2, 3], "resource_kind": "gpu"},
        {"mode": "cpu-stage4", "stage_ids": [4], "resource_kind": "cpu"},
        {"mode": "cpu-stage5", "stage_ids": [5], "resource_kind": "cpu"},
        {"mode": "cpu-stage6", "stage_ids": [6], "resource_kind": "cpu"},
        {"mode": "cpu-stage7", "stage_ids": [7], "resource_kind": "cpu"},
        {"mode": "cpu-stage8", "stage_ids": [8], "resource_kind": "cpu"},
    ]


class StageCoordinatorState:
    def __init__(self, *, prompt: str, max_new_tokens: int, stage_count: int, initial_input_ids: list[int] | None = None) -> None:
        self.prompt_hash = "sha256:" + __import__("hashlib").sha256(
            str(prompt or "").encode("utf-8", errors="replace")
        ).hexdigest()
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.stage_count = max(2, int(stage_count))
        self.initial_input_ids = [int(value) for value in (initial_input_ids or [])]
        self.input_ids: list[int] = list(self.initial_input_ids)
        self.generated_token_hashes: list[str] = []
        self.output_hashes: list[str] = []
        self.tasks: dict[str, dict[str, Any]] = {}
        self.pending: list[dict[str, Any]] = []
        self.completed: list[dict[str, Any]] = []
        self.stage_seen: set[int] = set()
        self.activation_hashes: list[str] = []
        self.started_at = time.monotonic()
        self._counter = 0
        self._lock = threading.RLock()
        self._queue_stage(generation_step=0, stage_id=0, input_ids=list(self.initial_input_ids))

    def _new_task_id(self, stage_id: int, generation_step: int) -> str:
        self._counter += 1
        return f"ct32bfull-{self._counter:04d}-stage{stage_id}-step{generation_step}"

    def _queue_stage(
        self,
        *,
        generation_step: int,
        stage_id: int,
        input_ids: list[int] | None = None,
        activation: dict[str, Any] | None = None,
    ) -> None:
        task = {
            "task_id": self._new_task_id(stage_id, generation_step),
            "stage_id": int(stage_id),
            "generation_step": int(generation_step),
            "input_ids": [int(value) for value in (input_ids or [])],
            "created_at": time.time(),
            "status": "queued",
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
                if int_field(task.get("stage_id")) != int(stage_id):
                    continue
                claimed = self.pending.pop(index)
                claimed["status"] = "leased"
                claimed["miner_id"] = str(miner_id or "")
                claimed["claimed_at"] = time.time()
                self.tasks[claimed["task_id"]] = claimed
                public_task = {
                    key: value
                    for key, value in claimed.items()
                    if key not in {"status", "created_at", "claimed_at", "miner_id"}
                }
                return {"ok": True, "done": False, "task": public_task}
            return {"ok": True, "done": False, "task": None}

    def submit(self, result: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            task_id = str(result.get("task_id") or "")
            task = self.tasks.get(task_id)
            if not task:
                return {"ok": False, "accepted": False, "reason": "unknown_task"}
            stage_id = int_field(result.get("stage_id"), int_field(task.get("stage_id")))
            step = int_field(result.get("generation_step"), int_field(task.get("generation_step"), 0))
            task["status"] = "completed"
            task["completed_at"] = time.time()
            task["duration_seconds"] = float(result.get("duration_seconds") or 0.0)
            task["activation_hash"] = result.get("activation_hash") or task.get("activation_hash")
            task["output_hash"] = result.get("output_hash")
            task["generated_token_count"] = result.get("generated_token_count")
            task["kv_cache"] = result.get("kv_cache") if isinstance(result.get("kv_cache"), dict) else {}
            task["cuda_memory_after_task"] = result.get("cuda_memory_after_task") if isinstance(result.get("cuda_memory_after_task"), dict) else {}
            self.completed.append(task)
            if stage_id < self.stage_count - 1:
                activation = dict(result.get("activation") or {})
                if not activation.get("activation_hash"):
                    return {"ok": False, "accepted": False, "reason": "activation_missing"}
                self.activation_hashes.append(str(activation.get("activation_hash")))
                self._queue_stage(generation_step=step, stage_id=stage_id + 1, activation=activation)
            else:
                token = result.get("next_token_id_private")
                try:
                    token_id = int(token)
                except (TypeError, ValueError):
                    return {"ok": False, "accepted": False, "reason": "next_token_missing"}
                if not self.input_ids:
                    activation = (self.tasks.get(task_id) or {}).get("activation")
                    self.input_ids = [int(value) for value in (activation or {}).get("input_ids") or []]
                self.input_ids.append(token_id)
                self.generated_token_hashes.append(str(result.get("next_token_hash") or sha_payload({"token_id": token_id})))
                self.output_hashes.append(str(result.get("output_hash") or ""))
                if len(self.generated_token_hashes) < self.max_new_tokens:
                    self._queue_stage(generation_step=step + 1, stage_id=0, input_ids=self.input_ids)
            return {"ok": True, "accepted": True, "ready": self.ready()}

    def ready(self) -> bool:
        return len(self.generated_token_hashes) >= self.max_new_tokens

    def private_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "input_ids": list(self.input_ids),
                "generated_token_hashes": list(self.generated_token_hashes),
                "pending": list(self.pending),
                "tasks": json.loads(json.dumps(self.tasks, default=str)),
            }

    def public_status(self) -> dict[str, Any]:
        with self._lock:
            completed_public = []
            for task in self.completed:
                completed_public.append({
                    "task_id": task.get("task_id"),
                    "stage_id": task.get("stage_id"),
                    "generation_step": task.get("generation_step"),
                    "miner_id_hash": sha_payload(task.get("miner_id") or ""),
                    "duration_seconds": task.get("duration_seconds"),
                    "activation_hash": task.get("activation_hash"),
                    "output_hash": task.get("output_hash"),
                    "generated_token_count": task.get("generated_token_count"),
                    "kv_cache": task.get("kv_cache") or {},
                    "cuda_memory_after_task": task.get("cuda_memory_after_task") or {},
                })
            stage_kv_cache: dict[str, dict[str, Any]] = {}
            for stage_id in range(self.stage_count):
                rows = [
                    item.get("kv_cache") or {}
                    for item in self.completed
                    if int_field(item.get("stage_id")) == stage_id and isinstance(item.get("kv_cache"), dict)
                ]
                latest = rows[-1] if rows else {}
                expected_hit_count = max(0, self.max_new_tokens - 1)
                stage_kv_cache[f"stage{stage_id}"] = {
                    "schema": "kaggle_32b_full_stage_local_kv_cache_v1",
                    "stage_id": stage_id,
                    "ready": bool(rows and latest.get("ready")),
                    "hit_count": int(latest.get("hit_count") or 0),
                    "miss_count": int(latest.get("miss_count") or 0),
                    "prefill_count": int(latest.get("prefill_count") or 0),
                    "expected_hit_count": expected_hit_count,
                    "hit_target_ready": bool(int(latest.get("hit_count") or 0) >= expected_hit_count),
                    "tokens_before": int(latest.get("tokens_before") or 0),
                    "tokens_after": int(latest.get("tokens_after") or 0),
                    "last_input_token_count": int(latest.get("last_input_token_count") or 0),
                    "last_cache_seq_length": int(latest.get("last_cache_seq_length") or 0),
                    "last_cache_hit": bool(latest.get("last_cache_hit")),
                    "last_cache_ready": bool(latest.get("last_cache_ready")),
                    "cache_tensors_public": False,
                    "past_key_values_public": False,
                    "kv_cache_transport_public": False,
                }
            return {
                "schema": "kaggle_32b_full_heterogeneous_coordinator_status_v1",
                "ok": True,
                "ready": self.ready(),
                "prompt_hash": self.prompt_hash,
                "initial_input_token_count": len(self.initial_input_ids),
                "input_token_ids_public": False,
                "stage_count": self.stage_count,
                "max_new_tokens": self.max_new_tokens,
                "generated_token_count": len(self.generated_token_hashes),
                "generated_token_hashes": list(self.generated_token_hashes),
                "output_hashes": list(self.output_hashes),
                "activation_hashes": list(self.activation_hashes),
                "pending_count": len(self.pending),
                "completed_task_count": len(self.completed),
                "stage_task_counts": {
                    f"stage{stage_id}": sum(
                        1 for item in self.completed if int_field(item.get("stage_id")) == stage_id
                    )
                    for stage_id in range(self.stage_count)
                },
                "stage_kv_cache": stage_kv_cache,
                "kv_cache_ready": all(
                    summary.get("hit_target_ready") is True
                    for summary in stage_kv_cache.values()
                ) if stage_kv_cache else False,
                "kv_cache_expected_hit_count_per_stage": max(0, self.max_new_tokens - 1),
                "stage_seen": sorted(self.stage_seen),
                "completed_tasks": completed_public,
                "elapsed_seconds": round(time.monotonic() - self.started_at, 3),
                "raw_prompt_public": False,
                "generated_token_ids_public": False,
                "activation_public": False,
                "hidden_state_public": False,
                "public_artifact_safe": True,
            }


class ProbeCoordinatorServer:
    def __init__(self, *, host: str, port: int, token: str, state: StageCoordinatorState) -> None:
        self.host = host
        self.port = int(port)
        self.token = token
        self.state = state
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

            def _authorized(self) -> bool:
                return self.headers.get("X-CrowdTensor-32B-Token") == token_value

            def _read_json(self) -> dict[str, Any]:
                size = int(self.headers.get("Content-Length") or 0)
                if size <= 0:
                    return {}
                loaded = json.loads(self.rfile.read(size).decode("utf-8"))
                return loaded if isinstance(loaded, dict) else {}

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
                    self._send(200, state_value.claim(miner_id=str(payload.get("miner_id") or ""), stage_id=int_field(payload.get("stage_id"))))
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


def coordinator_url_for(args: argparse.Namespace) -> str:
    if str(getattr(args, "coordinator_url", "") or "").strip():
        return str(args.coordinator_url).rstrip("/")
    return f"http://{args.public_host}:{int(args.port)}"


def render_kernel(
    args: argparse.Namespace,
    *,
    mode: str,
    stage_ids: list[int],
    resource_kind: str,
    coordinator_token: str,
) -> str:
    prompt = str(args.prompt or "Hi")[:64]
    rendered = KERNEL_TEMPLATE
    replacements = {
        "__STAGE_REPORT_SCHEMA__": STAGE_REPORT_SCHEMA,
        "__MODE__": mode,
        "__MODEL_REPO_JSON__": json.dumps(args.model_repo),
        "__KAGGLE_ATTACHED_MODEL_PATHS_JSON__": json.dumps(kaggle_attached_model_paths(args)),
        "__STAGE_IDS_JSON__": json.dumps([int(value) for value in stage_ids]),
        "__STAGE_RANGES_JSON__": json.dumps(stage_ranges_from_args(args)),
        "__RESOURCE_KIND__": str(resource_kind),
        "__PROMPT_TEXT_JSON__": json.dumps(prompt),
        "__COORDINATOR_URL_JSON__": json.dumps(coordinator_url_for(args)),
        "__COORDINATOR_TOKEN_JSON__": json.dumps(coordinator_token),
        "__MAX_NEW_TOKENS__": str(int(args.max_new_tokens)),
        "__TASK_POLL_INTERVAL_SECONDS__": str(float(args.task_poll_interval_seconds)),
        "__TASK_IDLE_TIMEOUT_SECONDS__": str(float(args.task_idle_timeout_seconds)),
        "__CPU_DTYPE__": str(args.cpu_dtype),
    }
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def render_colab_cuda_kernel(
    args: argparse.Namespace,
    *,
    mode: str,
    stage_ids: list[int],
    coordinator_token: str,
) -> str:
    rendered = render_kernel(
        args,
        mode=mode,
        stage_ids=stage_ids,
        resource_kind="gpu",
        coordinator_token=coordinator_token,
    )
    rendered = rendered.replace('OUT = Path("/kaggle/working")', 'OUT = Path("/content/ct_heterogeneous_colab_cuda")')
    rendered = rendered.replace(
        'TEMP = Path("/kaggle/temp/ct_32b_full_heterogeneous") / MODE',
        'TEMP = Path("/content/ct_32b_full_heterogeneous") / MODE',
    )
    return rendered + f'''

try:
    _ct_report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    print("{COLAB_CUDA_STAGE_REPORT_MARKER} " + json.dumps(_ct_report, sort_keys=True))
except Exception as _ct_exc:
    print("{COLAB_CUDA_STAGE_REPORT_MARKER} " + json.dumps({{"ok": False, "error_type": type(_ct_exc).__name__, "error_digest": sha_text(str(_ct_exc))}}, sort_keys=True))
'''


def build_package(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    mode: str,
    stage_ids: list[int],
    resource_kind: str,
    coordinator_token: str,
) -> dict[str, Any]:
    owner = args.kaggle_owner or loading_probe.default_kaggle_owner()
    if not owner:
        raise SystemExit("--kaggle-owner or ~/.kaggle/kaggle.json username is required")
    suffix = str(int(time.time()))[-8:]
    slug = f"{loading_probe.safe_slug(args.kernel_slug_prefix)[:24]}-{mode}-{suffix}"
    slug = slug[:45].strip("-")
    kernel_dir = output_dir / "private-kaggle-kernels" / mode
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text(
        render_kernel(
            args,
            mode=mode,
            stage_ids=stage_ids,
            resource_kind=resource_kind,
            coordinator_token=coordinator_token,
        ),
        encoding="utf-8",
    )
    title = f"CT 32B Full Hetero {mode} {suffix}"
    metadata = {
        "id": f"{owner}/{slug}",
        "title": title,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true" if resource_kind == "gpu" else "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": kaggle_model_sources(args),
    }
    if resource_kind == "gpu":
        metadata["machine_shape"] = args.accelerator
    write_json(kernel_dir / "kernel-metadata.json", metadata)
    return {
        "mode": mode,
        "stage_ids": [int(value) for value in stage_ids],
        "resource_kind": resource_kind,
        "kernel_dir": kernel_dir,
        "declared_kernel_ref": metadata["id"],
        "kernel_ref": metadata["id"],
        "kernel_slug": slug,
        "metadata": metadata,
        "report_filename": f"ct_32b_full_heterogeneous_{mode}_report.json",
    }


def resolve_pushed_kernel_ref(
    package: dict[str, Any],
    push_step: dict[str, Any],
    *,
    runner: Runner,
    timeout_seconds: float,
) -> tuple[str, dict[str, Any] | None]:
    return loading_probe.resolve_pushed_kernel_ref(
        package,
        push_step,
        runner=runner,
        timeout_seconds=timeout_seconds,
    )


def push_accepted(step: dict[str, Any]) -> bool:
    output = f"{step.get('stdout_tail') or ''}\n{step.get('stderr_tail') or ''}"
    return bool(step.get("ok")) and "Kernel version" in output and "successfully pushed" in output


def run_kaggle_package(
    args: argparse.Namespace,
    *,
    package: dict[str, Any],
    output_dir: Path,
    runner: Runner,
    cleanup: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    steps: list[dict[str, Any]] = []
    push_command = ["kaggle", "kernels", "push", "-p", str(package["kernel_dir"]), "-t", str(args.kernel_timeout_seconds)]
    if package.get("resource_kind") == "gpu" and args.accelerator:
        push_command.extend(["--accelerator", args.accelerator])
    print(f"[{utc_now()}] pushing private Kaggle {package['resource_kind']} kernel {package['declared_kernel_ref']}", flush=True)
    push_step = loading_probe.run_step(
        "kaggle_kernel_push",
        push_command,
        runner=runner,
        timeout_seconds=args.kaggle_push_timeout_seconds,
    )
    push_step["accepted"] = push_accepted(push_step)
    steps.append(push_step)
    if not push_step.get("accepted"):
        if push_step.get("ok") and cleanup and not args.skip_kaggle_cleanup:
            kernel_ref, resolve_step = resolve_pushed_kernel_ref(
                package,
                push_step,
                runner=runner,
                timeout_seconds=args.kaggle_push_timeout_seconds,
            )
            if resolve_step:
                steps.append(resolve_step)
            if kernel_ref:
                print(f"[{utc_now()}] deleting non-accepted Kaggle kernel {kernel_ref}", flush=True)
                delete_step = loading_probe.run_step(
                    "kaggle_kernel_delete",
                    ["kaggle", "kernels", "delete", kernel_ref, "-y"],
                    runner=runner,
                    timeout_seconds=args.kaggle_delete_timeout_seconds,
                )
                steps.append(delete_step)
        return {}, steps
    kernel_ref, resolve_step = resolve_pushed_kernel_ref(
        package,
        push_step,
        runner=runner,
        timeout_seconds=args.kaggle_push_timeout_seconds,
    )
    if resolve_step:
        steps.append(resolve_step)
    package["kernel_ref"] = kernel_ref
    print(f"[{utc_now()}] waiting for {kernel_ref}", flush=True)
    status_step = loading_probe.wait_kaggle_terminal(
        kernel_ref,
        runner=runner,
        timeout_seconds=args.kaggle_status_timeout_seconds,
        poll_interval=args.kaggle_status_poll_interval,
    )
    steps.append(status_step)
    stage_output = output_dir / "kaggle-output" / str(package["mode"])
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
    if cleanup and not args.skip_kaggle_cleanup:
        print(f"[{utc_now()}] deleting private Kaggle kernel {kernel_ref}", flush=True)
        delete_step = loading_probe.run_step(
            "kaggle_kernel_delete",
            ["kaggle", "kernels", "delete", kernel_ref, "-y"],
            runner=runner,
            timeout_seconds=args.kaggle_delete_timeout_seconds,
        )
        steps.append(delete_step)
    if not args.keep_kaggle_logs:
        for log_path in stage_output.glob("*.log"):
            log_path.unlink(missing_ok=True)
    report = load_json(stage_output / str(package["report_filename"]))
    return report, steps


def summarize_stage_runtime(item: dict[str, Any], *, parent_report: dict[str, Any]) -> dict[str, Any]:
    selection = item.get("selection") if isinstance(item.get("selection"), dict) else {}
    load = item.get("stage_weight_load") if isinstance(item.get("stage_weight_load"), dict) else {}
    buffers = item.get("runtime_buffers") if isinstance(item.get("runtime_buffers"), dict) else {}
    kv_cache = item.get("kv_cache") if isinstance(item.get("kv_cache"), dict) else {}
    return {
        "mode": parent_report.get("mode"),
        "stage_id": item.get("stage_id"),
        "resource_kind": item.get("resource_kind") or parent_report.get("resource_kind"),
        "device": item.get("device"),
        "ok": parent_report.get("ok") is True,
        "stage_layer_range": selection.get("stage_layer_range") or [],
        "assigned_weight_key_count": selection.get("assigned_weight_key_count"),
        "assigned_weight_file_count": selection.get("assigned_weight_file_count"),
        "loaded_weight_key_count": load.get("loaded_weight_key_count"),
        "loaded_tensor_gb": load.get("loaded_tensor_gb"),
        "prepared_tensor_gb": load.get("prepared_tensor_gb"),
        "runtime_dtype": load.get("runtime_dtype"),
        "runtime_buffers_ready": buffers.get("ready") is True,
        "stage_weight_load_ready": load.get("ready") is True,
        "loads_only_stage_weight_keys": load.get("loads_only_stage_weight_keys") is True,
        "kv_cache": kv_cache,
        "kv_cache_ready": kv_cache.get("ready") is True,
        "kv_cache_hit_count": int(kv_cache.get("hit_count") or 0),
        "kv_cache_expected_hit_count": int(kv_cache.get("expected_hit_count") or 0),
        "kv_cache_hit_target_ready": kv_cache.get("hit_target_ready") is True,
        "cuda_memory_after_load": item.get("cuda_memory_after_load") or {},
        "memory_after_load": item.get("memory_after_load") or {},
        "diagnosis_codes": parent_report.get("diagnosis_codes") or [],
        "blockers": parent_report.get("blockers") or [],
    }


def summarize_stage_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    runtime_summaries = report.get("stage_runtime_summaries")
    if isinstance(runtime_summaries, list) and runtime_summaries:
        return [
            summarize_stage_runtime(item, parent_report=report)
            for item in runtime_summaries
            if isinstance(item, dict)
        ]
    return []


def public_step(step: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(step)
    for key in ("command_line", "stdout_tail", "stderr_tail"):
        if key not in cleaned:
            continue
        cleaned[key] = str(cleaned.get(key) or "").replace("private-kaggle-kernels", "<private-payload-dir>")
    command = cleaned.get("command_public")
    if isinstance(command, list):
        cleaned["command_public"] = [
            "<private-payload-dir>" if "private-kaggle-kernels" in str(part) else part
            for part in command
        ]
    return cleaned


def public_stage_run(run: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(run)
    cleaned["steps"] = [
        public_step(step)
        for step in list(run.get("steps") or [])
        if isinstance(step, dict)
    ]
    return cleaned


def public_colab_outputs(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "output_type": output.get("output_type"),
            "name": output.get("name"),
            "text_hash": sha_payload(output.get("text") or ""),
        }
        for output in outputs
        if isinstance(output, dict)
    ]


def load_colab_session(config_path: str, session_name: str) -> dict[str, Any]:
    if colab_cuda_session_manager is not None:
        return colab_cuda_session_manager.load_session(config_path, session_name)
    path = Path(str(config_path or "")).expanduser()
    data = load_json(path) if path.is_file() else {}
    session = data.get(session_name) if isinstance(data, dict) else None
    if not isinstance(session, dict):
        raise RuntimeError("colab_session_not_found")
    missing = [key for key in ("url", "token", "endpoint") if not session.get(key)]
    if missing:
        raise RuntimeError("colab_session_missing_runtime_proxy_fields")
    return dict(session)


def public_colab_session_manager_result(result: dict[str, Any]) -> dict[str, Any]:
    attempts = []
    for item in list(result.get("attempts") or []):
        if not isinstance(item, dict):
            continue
        attempts.append({
            key: value
            for key, value in item.items()
            if key not in {"token", "url", "endpoint", "runtime_proxy_token", "runtime_proxy_url"}
        })
    return {
        "ok": result.get("ok") is True,
        "blocker": str(result.get("blocker") or ""),
        "attempt_count": len(attempts),
        "attempts": attempts,
        "public_artifact_safe": True,
    }


def load_colab_runtime_class() -> Any:
    if colab_cli_runtime is None:
        from colab_cli.runtime import ColabRuntime

        return ColabRuntime
    return colab_cli_runtime.load_colab_runtime_class()


def extract_colab_cuda_stage_report_from_outputs(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    lines: list[str] = []
    for output in outputs:
        text = output.get("text") if isinstance(output, dict) else None
        if isinstance(text, str):
            lines.extend(text.splitlines())
    for line in reversed(lines):
        if COLAB_CUDA_STAGE_REPORT_MARKER not in line:
            continue
        payload = line.split(COLAB_CUDA_STAGE_REPORT_MARKER, 1)[1].strip()
        try:
            loaded = json.loads(payload)
        except json.JSONDecodeError:
            continue
        return loaded if isinstance(loaded, dict) else {}
    return {}


def public_colab_cuda_stage_report(
    report: dict[str, Any],
    *,
    session: dict[str, Any],
    session_name: str,
) -> dict[str, Any]:
    cleaned = json.loads(json.dumps(report, default=str))
    cleaned["resource_kind"] = "colab_gpu"
    cleaned["provider_kind"] = "colab_cuda"
    cleaned["colab_session_name"] = session_name
    cleaned["colab_endpoint_hash"] = sha_payload(str(session.get("endpoint") or ""))[:24]
    parsed_url = urllib.parse.urlparse(str(session.get("url") or ""))
    cleaned["colab_runtime_proxy_host_hash"] = sha_payload(parsed_url.netloc)[:24]
    cleaned["colab_runtime_proxy_token_public"] = False
    cleaned["colab_runtime_proxy_url_public"] = False
    cleaned["endpoint_public"] = False
    cleaned["credentials_public"] = False
    cleaned["private_runtime_state_public"] = False
    hardware = cleaned.get("hardware")
    if isinstance(hardware, dict):
        gpu_names = [str(value) for value in list(hardware.get("gpu_names") or [])]
        hardware["gpu_name_hashes"] = [sha_payload(value)[:24] for value in gpu_names]
        hardware["gpu_names_public"] = False
        hardware.pop("gpu_names", None)
        smi = hardware.get("nvidia_smi")
        if isinstance(smi, dict):
            for key in ("stdout_tail", "stderr_tail", "command_line"):
                smi.pop(key, None)
            smi["raw_output_public"] = False
    for item in cleaned.get("stage_runtime_summaries") or []:
        if isinstance(item, dict):
            item["resource_kind"] = "colab_gpu"
            item["provider_kind"] = "colab_cuda"
    return cleaned


def colab_cuda_stage_worker(
    args: argparse.Namespace,
    *,
    mode: str,
    stage_ids: list[int],
    token: str,
) -> dict[str, Any]:
    started = time.monotonic()
    base_report: dict[str, Any] = {
        "schema": STAGE_REPORT_SCHEMA,
        "mode": mode,
        "resource_kind": "colab_gpu",
        "provider_kind": "colab_cuda",
        "stage_ids": [int(value) for value in stage_ids],
        "stage_ranges": stage_ranges_from_args(args),
        "model_repo": args.model_repo,
        "ok": False,
        "full_precision": True,
        "quantization": "none",
        "public_safe": True,
        "public_artifact_safe": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "activation_public": False,
        "hidden_state_public": False,
        "credentials_public": False,
        "colab_runtime_proxy_token_public": False,
        "colab_runtime_proxy_url_public": False,
        "endpoint_public": False,
        "private_runtime_state_public": False,
        "diagnosis_codes": [],
        "blockers": [],
    }
    try:
        ColabRuntime = load_colab_runtime_class()
        session_name = str(getattr(args, "colab_gpu_session_name", "ct-colab-cuda-gpu") or "ct-colab-cuda-gpu")
        code = render_colab_cuda_kernel(
            args,
            mode=mode,
            stage_ids=[int(value) for value in stage_ids],
            coordinator_token=token,
        )
        if colab_cuda_session_manager is not None:
            outputs, session, manager_result = colab_cuda_session_manager.execute_with_retry(
                code,
                session_name=session_name,
                state_path=str(getattr(args, "colab_gpu_session_config", "")),
                timeout=float(getattr(args, "colab_gpu_execute_timeout_seconds", 1800.0)),
                max_attempts=int(getattr(args, "colab_gpu_max_attempts", 3)),
                token_cache=str(getattr(args, "colab_gpu_token_cache", Path.home() / ".config" / "colab-exec" / "token.json")),
                accelerator=str(getattr(args, "colab_gpu_accelerator", "T4") or "T4"),
                authuser=str(getattr(args, "colab_gpu_authuser", "0") or "0"),
                cleanup_before_reacquire=bool(getattr(args, "colab_gpu_cleanup_before_reacquire", True)),
                force_reacquire_before=bool(getattr(args, "colab_gpu_reacquire_before", False)),
                heartbeat_code='print("CT_COLAB_CUDA_STAGE_HEARTBEAT")',
            )
            if not manager_result.get("ok"):
                base_report.update({
                    "blockers": [str(manager_result.get("blocker") or "colab_cuda_stage_execute_failed")],
                    "diagnosis_codes": ["kaggle_full_heterogeneous_colab_cuda_stage_execute_failed"],
                    "session_manager": public_colab_session_manager_result(manager_result),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                })
                return base_report
        else:
            session = load_colab_session(str(getattr(args, "colab_gpu_session_config", "")), session_name)
            runtime = ColabRuntime(
                session["url"],
                session["token"],
                kernel_id=session.get("kernel_id"),
                session_id=session.get("session_id"),
            )
            outputs = runtime.execute_code(code, timeout=float(getattr(args, "colab_gpu_execute_timeout_seconds", 1800.0)))
            try:
                runtime.stop()
            except Exception:
                pass
            manager_result = {"ok": True, "attempts": [{"attempt": 1, "ok": True, "fallback": True}]}
        stage_report = extract_colab_cuda_stage_report_from_outputs(outputs)
        if not stage_report:
            base_report.update({
                "blockers": ["colab_cuda_stage_report_missing"],
                "diagnosis_codes": ["kaggle_full_heterogeneous_colab_cuda_stage_report_missing"],
                "colab_outputs_public": public_colab_outputs(outputs),
                "session_manager": public_colab_session_manager_result(manager_result),
            })
            return base_report
        cleaned = public_colab_cuda_stage_report(stage_report, session=session, session_name=session_name)
        cleaned["session_manager"] = public_colab_session_manager_result(manager_result)
        cleaned["elapsed_seconds"] = cleaned.get("elapsed_seconds", round(time.monotonic() - started, 3))
        cleaned["diagnosis_codes"] = sorted(set([
            *list(cleaned.get("diagnosis_codes") or []),
            "kaggle_full_heterogeneous_colab_cuda_stage_ready" if cleaned.get("ok") is True else "kaggle_full_heterogeneous_colab_cuda_stage_not_ready",
        ]))
        cleaned["public_artifact_safe"] = True
        return cleaned
    except Exception as exc:
        base_report.update({
            "error_type": type(exc).__name__,
            "error_digest": sha_payload(str(exc)),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "blockers": ["colab_cuda_stage_worker_exception"],
            "diagnosis_codes": ["kaggle_full_heterogeneous_colab_cuda_stage_exception"],
        })
        return base_report


def web_tpu_stage_worker(
    args: argparse.Namespace,
    *,
    stage_id: int,
    token: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    if web_tpu_bridge is None:
        return {
            "schema": STAGE_REPORT_SCHEMA,
            "mode": f"web-tpu-stage{stage_id}",
            "resource_kind": "web_tpu",
            "stage_ids": [int(stage_id)],
            "model_repo": args.model_repo,
            "ok": False,
            "blockers": ["web_tpu_bridge_dependency_missing"],
            "diagnosis_codes": ["kaggle_full_heterogeneous_web_tpu_bridge_dependency_missing"],
            "public_safe": True,
            "public_artifact_safe": True,
        }
    tpu_provider = str(getattr(args, "tpu_provider", "kaggle_web") or "kaggle_web")
    ranges = stage_ranges_from_args(args)
    stage_range = list(ranges[int(stage_id)])
    loader_args = argparse.Namespace(
        output_dir=args.output_dir,
        kaggle_notebook_url=args.kaggle_notebook_url,
        kaggle_web_storage_state=args.kaggle_web_storage_state,
        chrome_executable=args.chrome_executable,
        target_model_id=args.model_repo,
        model_repo=args.model_repo,
        web_tpu_32b_stage_start=int(stage_range[0]),
        web_tpu_32b_stage_end=int(stage_range[1]),
        web_tpu_32b_tensor_key=str(args.web_tpu_tensor_key or f"model.layers.{int(stage_range[0])}.input_layernorm.weight"),
        web_tpu_32b_max_header_bytes=int(args.web_tpu_max_header_bytes),
        web_tpu_32b_max_tensor_bytes=int(args.web_tpu_max_tensor_bytes),
        web_tpu_32b_execute_layer_count=int(args.web_tpu_execute_layer_count or (int(stage_range[1]) - int(stage_range[0]))),
        web_tpu_execute_timeout_seconds=float(args.web_tpu_execute_timeout_seconds),
        web_tpu_task_timeout_seconds=float(timeout_seconds),
        chrome_user_data_dir="",
        tpu_provider=tpu_provider,
        colab_session_name=getattr(args, "colab_session_name", "ct-colab-tpu-v5e1"),
        colab_session_config=getattr(args, "colab_session_config", ""),
        input_activation_private={},
        return_output_activation_private=True,
    )
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema": STAGE_REPORT_SCHEMA,
        "mode": f"web-tpu-stage{stage_id}",
        "resource_kind": "web_tpu",
        "tpu_provider": tpu_provider,
        "stage_ids": [int(stage_id)],
        "stage_ranges": ranges,
        "model_repo": args.model_repo,
        "ok": False,
        "full_precision": True,
        "quantization": "none",
        "public_safe": True,
        "public_artifact_safe": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "activation_public": False,
        "hidden_state_public": False,
        "credentials_public": False,
        "jupyter_proxy_token_public": False,
        "diagnosis_codes": [],
        "blockers": [],
    }

    def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        request = __import__("urllib.request").request.Request(
            coordinator_url_for(args) + path,
            data=body,
            headers={"Content-Type": "application/json", "X-CrowdTensor-32B-Token": token},
            method="POST",
        )
        with __import__("urllib.request").request.urlopen(request, timeout=120) as response:
            loaded = json.loads(response.read().decode("utf-8"))
        return loaded if isinstance(loaded, dict) else {}

    try:
        accepted_count = 0
        processed_tasks: list[dict[str, Any]] = []
        last_loader_report: dict[str, Any] = {}
        runtime_summary_written = False
        deadline = time.monotonic() + float(timeout_seconds)
        while time.monotonic() < deadline:
            response = post("/claim", {"miner_id": f"web-tpu-full-heterogeneous-stage{stage_id}", "stage_id": int(stage_id)})
            if response.get("done"):
                break
            task = response.get("task") if isinstance(response.get("task"), dict) else {}
            if not task:
                time.sleep(max(2.0, float(args.task_poll_interval_seconds)))
                continue
            incoming = task.get("activation") if isinstance(task.get("activation"), dict) else {}
            loader_args.input_activation_private = incoming
            loader_code = web_tpu_bridge.render_web_tpu_32b_loader_code(loader_args)
            if tpu_provider == "colab_cli":
                try:
                    ColabRuntime = load_colab_runtime_class()
                except Exception as exc:
                    report["error_type"] = type(exc).__name__
                    report["error_digest"] = sha_payload(str(exc))
                    report["blockers"].append("colab_cli_runtime_dependency_missing")
                    report["diagnosis_codes"].append("kaggle_full_heterogeneous_colab_runtime_dependency_missing")
                    return report
                session = web_tpu_bridge.load_colab_session(
                    str(getattr(args, "colab_session_config", "")),
                    str(getattr(args, "colab_session_name", "ct-colab-tpu-v5e1")),
                )
                runtime = ColabRuntime(
                    session["url"],
                    session["token"],
                    kernel_id=session.get("kernel_id"),
                    session_id=session.get("session_id"),
                )
                outputs = runtime.execute_code(loader_code, timeout=float(args.web_tpu_execute_timeout_seconds))
                loader_report = web_tpu_bridge.extract_colab_tpu_report_from_outputs(outputs)
                loader_report["tpu_provider"] = "colab_cli"
                loader_report["colab_runtime_proxy_token_public"] = False
                loader_report["colab_runtime_proxy_url_public"] = False
            else:
                loader_report = web_tpu_bridge.execute_web_tpu_code_via_iframe(
                    loader_args,
                    loader_code,
                )
            last_loader_report = loader_report
            full_stage_count = int(stage_range[1]) - int(stage_range[0])
            full_ready = bool(
                loader_report.get("ok") is True
                and loader_report.get("full_stage_owned_tpu_loader_ready") is True
                and loader_report.get("tpu_32b_runtime_adapter_ready") is True
                and loader_report.get("input_activation_consumed") is True
                and loader_report.get("output_activation_private_present") is True
                and int(loader_report.get("executed_layer_count") or 0) >= full_stage_count
                and int(loader_report.get("missing_stage_key_count") or 0) == 0
            )
            if not runtime_summary_written:
                report.update({
                    "stage_runtime_summaries": [
                        {
                            "stage_id": int(stage_id),
                            "resource_kind": "web_tpu",
                            "tpu_provider": tpu_provider,
                            "device": "jax_tpu",
                            "selection": {
                                "stage_id": int(stage_id),
                                "stage_count": len(ranges),
                                "stage_layer_range": stage_range,
                                "assigned_weight_key_count": int(loader_report.get("assigned_weight_key_count") or 0),
                                "assigned_weight_file_count": int(loader_report.get("assigned_weight_file_count") or 0),
                            },
                            "runtime_buffers": {"ready": full_ready},
                            "stage_weight_load": {
                                "ready": full_ready,
                                "loaded_weight_key_count": int(loader_report.get("loaded_execution_tensor_key_count") or 0),
                                "assigned_weight_key_count": int(loader_report.get("assigned_weight_key_count") or 0),
                                "loaded_tensor_gb": float(loader_report.get("loaded_execution_tensor_gb") or 0.0),
                                "prepared_tensor_gb": float(loader_report.get("loaded_execution_tensor_gb") or 0.0),
                                "runtime_dtype": "bfloat16",
                                "loads_only_stage_weight_keys": True,
                            },
                            "kv_cache": {
                                "schema": "kaggle_32b_full_stage_local_kv_cache_v1",
                                "stage_id": int(stage_id),
                                "ready": loader_report.get("stage_local_kv_cache_verified") is True,
                                "hit_count": 0,
                                "miss_count": 1,
                                "prefill_count": 1,
                                "expected_hit_count": max(0, int(args.max_new_tokens) - 1),
                                "hit_target_ready": int(args.max_new_tokens) <= 1,
                                "cache_tensors_public": False,
                                "past_key_values_public": False,
                            },
                            "cuda_memory_after_load": {},
                            "memory_after_load": {},
                        }
                    ],
                    "loader_report_ok": loader_report.get("ok") is True,
                    "stage_owned_header_verified": loader_report.get("stage_owned_header_verified") is True,
                    "partial_tensor_to_tpu_verified": loader_report.get("partial_tensor_to_tpu_verified") is True,
                    "full_stage_owned_tpu_loader_ready": full_ready,
                    "tpu_32b_runtime_adapter_ready": full_ready,
                    "input_activation_consumed": loader_report.get("input_activation_consumed") is True,
                    "output_activation_private_present": loader_report.get("output_activation_private_present") is True,
                    "executed_layer_count": int(loader_report.get("executed_layer_count") or 0),
                    "loaded_execution_tensor_key_count": int(loader_report.get("loaded_execution_tensor_key_count") or 0),
                    "loaded_execution_tensor_gb": float(loader_report.get("loaded_execution_tensor_gb") or 0.0),
                    "stage_output_hash": str(loader_report.get("stage_output_hash") or ""),
                    "tpu_device_count": int(loader_report.get("tpu_device_count") or 0),
                    "tpu_device_kind": str(loader_report.get("tpu_device_kind") or ""),
                    "web_tpu_jupyter_steps": web_tpu_bridge.public_jupyter_steps(loader_report.get("web_tpu_jupyter_steps")),
                })
                runtime_summary_written = True
            if not full_ready:
                report["blockers"].append("web_tpu_stage_loader_not_ready")
                report["diagnosis_codes"].append("kaggle_full_heterogeneous_web_tpu_stage_loader_not_ready")
                return report
            generation_step = int(task.get("generation_step") or 0)
            task_id = str(task.get("task_id") or "")
            activation = loader_report.get("output_activation_private") if isinstance(loader_report.get("output_activation_private"), dict) else {}
            if not activation:
                report["blockers"].append("web_tpu_output_activation_missing")
                report["diagnosis_codes"].append("kaggle_full_heterogeneous_web_tpu_output_activation_missing")
                return report
            activation_hash = str(activation.get("activation_hash") or "")
            submitted = post("/submit", {
                "task_id": task_id,
                "stage_id": int(stage_id),
                "generation_step": generation_step,
                "activation": activation,
                "activation_hash": activation_hash,
                "output_hash": sha_payload({"stage": int(stage_id), "activation_hash": activation_hash}),
                "kv_cache": {
                    "schema": "kaggle_32b_full_stage_local_kv_cache_v1",
                    "stage_id": int(stage_id),
                    "ready": loader_report.get("stage_local_kv_cache_verified") is True,
                    "hit_count": 0,
                    "miss_count": 1,
                    "prefill_count": 1,
                    "expected_hit_count": max(0, int(args.max_new_tokens) - 1),
                    "hit_target_ready": int(args.max_new_tokens) <= 1,
                    "cache_tensors_public": False,
                    "past_key_values_public": False,
                },
                "duration_seconds": round(time.monotonic() - started, 3),
            })
            processed_tasks.append({
                "task_id": task_id,
                "stage_id": int(stage_id),
                "generation_step": generation_step,
                "accepted": submitted.get("accepted") is True,
                "activation_hash": activation_hash,
            })
            if submitted.get("accepted") is not True:
                report["blockers"].append("web_tpu_stage_submit_rejected")
                report["diagnosis_codes"].append("kaggle_full_heterogeneous_web_tpu_stage_submit_rejected")
                break
            accepted_count += 1
            if submitted.get("ready") is True:
                break
        if accepted_count < 1:
            report["blockers"].append("web_tpu_stage_task_not_claimed")
            report["diagnosis_codes"].append("kaggle_full_heterogeneous_web_tpu_stage_task_missing")
            return report
        report.update({
            "ok": True,
            "worker_loop_ready": True,
            "processed_task_count": accepted_count,
            "processed_tasks": processed_tasks,
            "diagnosis_codes": sorted(set([*report["diagnosis_codes"], "kaggle_full_heterogeneous_web_tpu_stage_ready"])),
            "blockers": [],
        })
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error_digest"] = sha_payload(str(exc))
        report["blockers"].append("web_tpu_full_heterogeneous_stage_exception")
        report["diagnosis_codes"].append("kaggle_full_heterogeneous_web_tpu_stage_exception")
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return report


def wait_for_coordinator_ready(
    state: StageCoordinatorState,
    *,
    timeout_seconds: float,
    worker_threads: list[threading.Thread] | None = None,
    stage_reports_by_mode: dict[str, dict[str, Any]] | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    def terminal_failed_report(report: dict[str, Any]) -> bool:
        if not isinstance(report, dict) or not report or report.get("ok") is True:
            return False
        blockers = set(str(item) for item in report.get("blockers") or [])
        transient_only = blockers and blockers.issubset({"coordinator_worker_processed_no_tasks"})
        return not transient_only

    started = time.monotonic()
    while time.monotonic() - started <= timeout_seconds:
        status = state.public_status()
        if status.get("ready"):
            return status
        if errors:
            return status
        if stage_reports_by_mode and any(terminal_failed_report(report) for report in stage_reports_by_mode.values()):
            return status
        if worker_threads and all(not thread.is_alive() for thread in worker_threads):
            return status
        time.sleep(5.0)
    return state.public_status()


def build_report(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    coordinator_status: dict[str, Any],
    stage_reports: list[dict[str, Any]],
    stage_runs: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    stage_summaries: list[dict[str, Any]] = []
    for report in stage_reports:
        stage_summaries.extend(summarize_stage_report(report))
    groups = stage_groups_for(args)
    stage_ranges = stage_ranges_from_args(args)
    requested_gpu_kernels = sum(1 for group in groups if group.get("resource_kind") == "gpu")
    requested_colab_gpu_runtimes = sum(1 for group in groups if group.get("resource_kind") == "colab_gpu")
    requested_cpu_kernels = sum(1 for group in groups if group.get("resource_kind") == "cpu")
    requested_web_tpu_stages = sum(1 for group in groups if group.get("resource_kind") == "web_tpu")
    requested_gpu_stages = sum(len(group.get("stage_ids") or []) for group in groups if group.get("resource_kind") == "gpu")
    requested_colab_gpu_stages = sum(
        len(group.get("stage_ids") or []) for group in groups if group.get("resource_kind") == "colab_gpu"
    )
    requested_cpu_stages = sum(len(group.get("stage_ids") or []) for group in groups if group.get("resource_kind") == "cpu")
    requested_web_tpu_stage_count = sum(
        len(group.get("stage_ids") or []) for group in groups if group.get("resource_kind") == "web_tpu"
    )
    if requested_colab_gpu_stages:
        requested_topology = (
            f"{requested_gpu_stages}KaggleGPU_stages_"
            f"{requested_colab_gpu_stages}ColabGPU_stages_"
            f"{requested_web_tpu_stage_count}WebTPU_stages_"
            f"{requested_cpu_stages}CPU_stages"
        )
    else:
        requested_topology = (
            f"{requested_gpu_stages}GPU_stages_"
            f"{requested_web_tpu_stage_count}WebTPU_stages_"
            f"{requested_cpu_stages}CPU_stages"
        )
    lifecycle = {
        "requested_topology": requested_topology,
        "requested_gpu_kernels": requested_gpu_kernels,
        "requested_colab_gpu_runtimes": requested_colab_gpu_runtimes,
        "requested_cpu_kernels": requested_cpu_kernels,
        "requested_web_tpu_stages": requested_web_tpu_stages,
        "requested_accelerator": args.accelerator,
        "requested_colab_gpu_session_name": str(getattr(args, "colab_gpu_session_name", "")),
        "coordinator_url": coordinator_url_for(args),
        "coordinator_direct_management": True,
        "actual_push_count": sum(
            1
            for run in stage_runs
            if any(step.get("name") == "kaggle_kernel_push" and step.get("accepted") for step in run.get("steps", []))
        ),
        "actual_gpu_push_count": sum(
            1
            for run in stage_runs
            if run.get("resource_kind") == "gpu"
            and any(step.get("name") == "kaggle_kernel_push" and step.get("accepted") for step in run.get("steps", []))
        ),
        "actual_colab_gpu_runtime_count": sum(
            1
            for run in stage_runs
            if run.get("resource_kind") == "colab_gpu"
            and any(step.get("name") == "colab_cuda_stage_worker" and step.get("accepted") for step in run.get("steps", []))
        ),
        "actual_cpu_push_count": sum(
            1
            for run in stage_runs
            if run.get("resource_kind") == "cpu"
            and any(step.get("name") == "kaggle_kernel_push" and step.get("accepted") for step in run.get("steps", []))
        ),
        "kernels_deleted": all(
            run.get("resource_kind") in {"web_tpu", "colab_gpu"}
            or any(step.get("name") == "kaggle_kernel_delete" and step.get("ok") for step in run.get("steps", []))
            for run in stage_runs
        ) if stage_runs else False,
        "private_packages_removed": not (output_dir / "private-kaggle-kernels").exists(),
    }
    generated = int(coordinator_status.get("generated_token_count") or 0)
    stage_task_counts = coordinator_status.get("stage_task_counts")
    if not isinstance(stage_task_counts, dict):
        stage_task_counts = {}
    stage_count = len(stage_ranges_from_args(args))
    gpu_stage_count = sum(1 for item in stage_summaries if item.get("resource_kind") == "gpu")
    colab_gpu_stage_count = sum(1 for item in stage_summaries if item.get("resource_kind") == "colab_gpu")
    cpu_stage_count = sum(1 for item in stage_summaries if item.get("resource_kind") == "cpu")
    web_tpu_stage_count = sum(1 for item in stage_summaries if item.get("resource_kind") == "web_tpu")
    provider_stage_counts = {
        "kaggle_cuda": gpu_stage_count,
        "colab_cuda": colab_gpu_stage_count,
        "cpu": cpu_stage_count,
        "web_tpu": web_tpu_stage_count,
    }
    accepted_providers = [
        provider
        for provider, count in provider_stage_counts.items()
        if count > 0
    ]
    target_b = parameter_count_b(args.model_repo)
    expected_layer_count = expected_layer_count_for_model(args.model_repo)
    full_model_layer_coverage_ready = stage_ranges_cover_expected_layers(stage_ranges, expected_layer_count)
    all_stage_tasks_done = all(
        int(stage_task_counts.get(f"stage{stage_id}") or 0) >= int(args.max_new_tokens)
        for stage_id in range(stage_count)
    )
    all_stages_seen = set(coordinator_status.get("stage_seen") or []) == set(range(stage_count))
    all_stage_runtime_ready = (
        len(stage_summaries) == stage_count
        and all(item.get("runtime_buffers_ready") and item.get("stage_weight_load_ready") for item in stage_summaries)
    )
    stage_kv_cache = coordinator_status.get("stage_kv_cache")
    if not isinstance(stage_kv_cache, dict):
        stage_kv_cache = {}
    expected_kv_hits = max(0, int(args.max_new_tokens) - 1)
    kv_cache_ready = bool(
        stage_kv_cache
        and len(stage_kv_cache) == stage_count
        and all(
            isinstance(stage_kv_cache.get(f"stage{stage_id}"), dict)
            and int(stage_kv_cache[f"stage{stage_id}"].get("hit_count") or 0) >= expected_kv_hits
            and stage_kv_cache[f"stage{stage_id}"].get("past_key_values_public") is False
            and stage_kv_cache[f"stage{stage_id}"].get("cache_tensors_public") is False
            for stage_id in range(stage_count)
        )
    )
    topology_ready = bool(
        lifecycle["actual_gpu_push_count"] == requested_gpu_kernels
        and lifecycle["actual_colab_gpu_runtime_count"] == requested_colab_gpu_runtimes
        and lifecycle["actual_cpu_push_count"] == requested_cpu_kernels
        and gpu_stage_count == requested_gpu_stages
        and colab_gpu_stage_count == requested_colab_gpu_stages
        and cpu_stage_count == requested_cpu_stages
        and web_tpu_stage_count == requested_web_tpu_stages
    )
    kaggle_colab_gpu_cpu_topology_ready = bool(
        topology_ready
        and requested_gpu_stages >= 1
        and requested_colab_gpu_stages >= 1
        and requested_cpu_stages >= 1
        and gpu_stage_count >= 1
        and colab_gpu_stage_count >= 1
        and cpu_stage_count >= 1
    )
    exact_72b_4t4_1tpu_5cpu_topology_ready = bool(
        target_b == 72
        and topology_ready
        and full_model_layer_coverage_ready
        and stage_count == 10
        and gpu_stage_count == 4
        and web_tpu_stage_count == 1
        and cpu_stage_count == 5
    )
    full_72b_topology_ready = bool(
        target_b == 72
        and full_model_layer_coverage_ready
        and topology_ready
        and gpu_stage_count >= 1
        and web_tpu_stage_count >= 1
        and cpu_stage_count >= 1
    )
    kaggle_colab_gpu_cpu_72b_topology_ready = bool(
        target_b == 72
        and full_model_layer_coverage_ready
        and kaggle_colab_gpu_cpu_topology_ready
    )
    ready = bool(
        generated >= int(args.max_new_tokens)
        and all_stage_tasks_done
        and all_stages_seen
        and all_stage_runtime_ready
        and kv_cache_ready
        and topology_ready
        and (target_b <= 32 or full_model_layer_coverage_ready)
        and lifecycle["kernels_deleted"]
        and lifecycle["private_packages_removed"]
        and not errors
    )
    full_target_same_request_ready = bool(ready and target_b == 72 and full_72b_topology_ready)
    blockers: list[str] = []
    if generated < int(args.max_new_tokens):
        blockers.append("one_token_generation_incomplete")
    if not all_stage_tasks_done:
        blockers.append("coordinator_stage_task_counts_incomplete")
    if not all_stage_runtime_ready:
        blockers.append("stage_runtime_not_ready")
    if not kv_cache_ready:
        blockers.append("stage_local_kv_cache_not_verified")
    if not topology_ready:
        blockers.append("heterogeneous_topology_not_verified")
    if target_b > 32 and not full_model_layer_coverage_ready:
        blockers.append("dense_full_model_layer_coverage_not_verified")
    if not lifecycle["kernels_deleted"]:
        blockers.append("kaggle_kernels_cleanup_not_verified")
    if errors:
        blockers.append("worker_thread_errors")
    diagnosis = [
        "kaggle_32b_full_heterogeneous_ready" if ready else "kaggle_32b_full_heterogeneous_not_ready",
        "kaggle_32b_full_multi_token_generation_ready" if generated >= int(args.max_new_tokens) else "kaggle_32b_full_multi_token_generation_incomplete",
        "kaggle_32b_full_stage_local_kv_cache_ready" if kv_cache_ready else "kaggle_32b_full_stage_local_kv_cache_not_ready",
        "kaggle_32b_4t4_5cpu_topology_ready" if topology_ready and target_b != 72 else "kaggle_32b_4t4_5cpu_topology_not_ready",
        "kaggle_72b_full_layer_coverage_ready" if full_model_layer_coverage_ready else "kaggle_72b_full_layer_coverage_not_ready",
        "kaggle_72b_gpu_tpu_cpu_full_topology_ready" if full_72b_topology_ready else "kaggle_72b_gpu_tpu_cpu_full_topology_not_ready",
        "kaggle_colab_gpu_cpu_topology_ready" if kaggle_colab_gpu_cpu_topology_ready else "kaggle_colab_gpu_cpu_topology_not_ready",
        "kaggle_colab_gpu_cpu_72b_full_topology_ready" if kaggle_colab_gpu_cpu_72b_topology_ready else "kaggle_colab_gpu_cpu_72b_full_topology_not_ready",
        "kaggle_72b_4t4_1tpu_5cpu_topology_ready" if exact_72b_4t4_1tpu_5cpu_topology_ready else "kaggle_72b_4t4_1tpu_5cpu_topology_not_ready",
        "kaggle_kernels_deleted" if lifecycle["kernels_deleted"] else "kaggle_kernels_cleanup_not_verified",
    ]
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "fresh_kaggle_run_performed": lifecycle["actual_push_count"] > 0,
        "execution_mode": "coordinator",
        "heterogeneous_placement_verified": topology_ready,
        "four_t4_five_cpu_topology_verified": topology_ready,
        "kaggle_colab_gpu_cpu_topology_verified": kaggle_colab_gpu_cpu_topology_ready,
        "kaggle_colab_gpu_cpu_same_request_verified": bool(ready and kaggle_colab_gpu_cpu_topology_ready),
        "colab_cuda_provider_verified": bool(ready and colab_gpu_stage_count >= 1),
        "kaggle_cuda_provider_verified": bool(ready and gpu_stage_count >= 1),
        "cpu_provider_verified": bool(ready and cpu_stage_count >= 1),
        "four_t4_one_tpu_five_cpu_topology_verified": exact_72b_4t4_1tpu_5cpu_topology_ready,
        "full_72b_layer_coverage_verified": bool(target_b == 72 and full_model_layer_coverage_ready),
        "gpu_tpu_cpu_72b_full_topology_verified": full_72b_topology_ready,
        "kaggle_colab_gpu_cpu_72b_full_topology_verified": kaggle_colab_gpu_cpu_72b_topology_ready,
        "full_precision_32b": target_b == 32,
        "full_precision_72b": target_b == 72,
        "quantization": "none",
        "coordinator_direct_management_verified": bool(ready and lifecycle["coordinator_direct_management"]),
        "gpu_tpu_cpu_72b_same_request_verified": full_target_same_request_ready,
        "same_request_72b_full_model_verified": full_target_same_request_ready,
        "same_request_72b_kaggle_colab_gpu_cpu_full_model_verified": bool(ready and kaggle_colab_gpu_cpu_72b_topology_ready),
        "full_72b_weight_loading_public_claim": full_target_same_request_ready,
        "cross_kernel_activation_decode_verified": bool(generated >= 1),
        "one_token_generation_verified": bool(generated >= 1),
        "multi_token_generation_verified": bool(generated >= int(args.max_new_tokens) and int(args.max_new_tokens) > 1),
        "generated_token_count": generated,
        "max_new_tokens": int(args.max_new_tokens),
        "stage_owned_full_precision_runtime_verified": all_stage_runtime_ready,
        "stage_local_kv_cache_verified": kv_cache_ready,
        "kv_cache_expected_hit_count_per_stage": expected_kv_hits,
        "stage_kv_cache": stage_kv_cache,
        "activation_handoff_verified": bool(coordinator_status.get("activation_hashes")),
        "accepted_providers": accepted_providers,
        "provider_stage_counts": provider_stage_counts,
        "blocked_reason": "" if ready else (blockers[0] if blockers else "kaggle_32b_full_heterogeneous_not_ready"),
        "blockers": sorted(set(blockers)),
        "diagnosis_codes": sorted(set(diagnosis)),
        "model": {
            "repo": args.model_repo,
            "parameter_count_b": target_b,
            "quantization": "none",
            "precision": "bf16_or_fp16_stage_runtime",
            "stage_count": stage_count,
            "stage_ranges": stage_ranges,
            "expected_layer_count": expected_layer_count,
            "full_layer_coverage_verified": full_model_layer_coverage_ready,
        },
        "coordinator": coordinator_status,
        "stage_task_counts": stage_task_counts,
        "stage_summaries": stage_summaries,
        "kaggle_lifecycle": lifecycle,
        "stage_runs": [public_stage_run(run) for run in stage_runs],
        "worker_errors": errors,
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
            "private_kernel_payload_public": False,
            "coordinator_private_state_public": False,
        },
        "limitations": [
            "This is a temporary proof Coordinator over private Kaggle GPU/CPU kernels and optional authenticated Web TPU stages, not the production CrowdTensor data plane.",
            "The target is bounded full-precision heterogeneous feasibility with stage-local KV-cache reuse, not throughput or production serving latency.",
            "Batching, requeue, pricing, trust, and P2P/NAT traversal are out of scope for this proof.",
        ],
    }


def run_coordinator_probe(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = __import__("secrets").token_urlsafe(32)
    stage_count = len(stage_ranges_from_args(args))
    state = StageCoordinatorState(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        stage_count=stage_count,
        initial_input_ids=initial_input_ids_from_args(args),
    )
    server = ProbeCoordinatorServer(host="0.0.0.0", port=int(args.port), token=token, state=state)
    stage_runs: list[dict[str, Any]] = []
    stage_reports_by_mode: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []

    def run_group(group: dict[str, Any]) -> None:
        mode = str(group["mode"])
        try:
            stagger_seconds = float(getattr(args, "stage_launch_stagger_seconds", 0.0) or 0.0)
            if stagger_seconds > 0:
                group_index = int(group.get("launch_index") or 0)
                if group_index > 0:
                    time.sleep(stagger_seconds * group_index)
            if str(group["resource_kind"]) == "web_tpu":
                report = web_tpu_stage_worker(
                    args,
                    stage_id=int(list(group["stage_ids"])[0]),
                    token=token,
                    timeout_seconds=float(args.task_idle_timeout_seconds),
                )
                stage_reports_by_mode[mode] = report
                stage_runs.append({
                    "mode": mode,
                    "stage_ids": list(group["stage_ids"]),
                    "resource_kind": "web_tpu",
                    "kernel_ref": "authenticated-web-tpu-session",
                    "steps": [
                        {
                            "name": "web_tpu_stage_worker",
                            "ok": report.get("ok") is True,
                            "accepted": report.get("ok") is True,
                            "resource_kind": "web_tpu",
                        }
                    ],
                })
                return
            if str(group["resource_kind"]) == "colab_gpu":
                report = colab_cuda_stage_worker(
                    args,
                    mode=mode,
                    stage_ids=list(group["stage_ids"]),
                    token=token,
                )
                stage_reports_by_mode[mode] = report
                colab_step: dict[str, Any] = {
                    "name": "colab_cuda_stage_worker",
                    "ok": report.get("ok") is True,
                    "accepted": report.get("ok") is True,
                    "resource_kind": "colab_gpu",
                    "provider_kind": "colab_cuda",
                }
                for key in ("blockers", "diagnosis_codes", "elapsed_seconds", "error_type", "error_digest"):
                    if report.get(key):
                        colab_step[key] = report.get(key)
                if isinstance(report.get("session_manager"), dict):
                    colab_step["session_manager"] = report["session_manager"]
                stage_runs.append({
                    "mode": mode,
                    "stage_ids": list(group["stage_ids"]),
                    "resource_kind": "colab_gpu",
                    "kernel_ref": "authenticated-colab-cuda-session",
                    "steps": [colab_step],
                })
                return
            package = build_package(
                args,
                output_dir=output_dir,
                mode=mode,
                stage_ids=list(group["stage_ids"]),
                resource_kind=str(group["resource_kind"]),
                coordinator_token=token,
            )
            report, steps = run_kaggle_package(
                args,
                package=package,
                output_dir=output_dir,
                runner=runner,
                cleanup=True,
            )
            stage_reports_by_mode[mode] = report
            stage_runs.append({
                "mode": mode,
                "stage_ids": list(group["stage_ids"]),
                "resource_kind": str(group["resource_kind"]),
                "kernel_ref": package.get("kernel_ref"),
                "steps": steps,
            })
        except Exception as exc:
            errors.append({
                "mode": mode,
                "stage_ids": list(group.get("stage_ids") or []),
                "resource_kind": str(group.get("resource_kind") or ""),
                "error_type": type(exc).__name__,
                "error_digest": sha_payload(str(exc)),
            })

    groups = stage_groups_for(args)
    try:
        server.start()
        indexed_groups = [
            {**group, "launch_index": index}
            for index, group in enumerate(groups)
        ]
        threads = [
            threading.Thread(target=run_group, args=(group,), daemon=True)
            for group in indexed_groups
        ]
        for thread in threads:
            thread.start()
        coordinator_status = wait_for_coordinator_ready(
            state,
            timeout_seconds=args.coordinator_timeout_seconds,
            worker_threads=threads,
            stage_reports_by_mode=stage_reports_by_mode,
            errors=errors,
        )
        cleanup_join_deadline = time.monotonic() + max(1.0, float(args.kaggle_status_timeout_seconds) + 180.0)
        for thread in threads:
            remaining = cleanup_join_deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)
    finally:
        server.stop()
        if not args.keep_private_package:
            shutil.rmtree(output_dir / "private-kaggle-kernels", ignore_errors=True)
    coordinator_status = state.public_status()
    report = build_report(
        args,
        output_dir=output_dir,
        coordinator_status=coordinator_status,
        stage_reports=[stage_reports_by_mode.get(str(group["mode"]), {}) for group in groups],
        stage_runs=stage_runs,
        errors=errors,
    )
    write_json(output_dir / "kaggle_32b_full_heterogeneous_probe.json", report)
    private_state_path = output_dir / "coordinator-private-state.json"
    if args.keep_coordinator_private_state:
        write_json(private_state_path, state.private_state())
    else:
        private_state_path.unlink(missing_ok=True)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded 4*T4 + 5*CPU full-precision 32B heterogeneous Kaggle probe.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kaggle-owner", default=loading_probe.default_kaggle_owner())
    parser.add_argument("--kernel-slug-prefix", default="ct32bfullhet")
    parser.add_argument("--accelerator", default=DEFAULT_ACCELERATOR)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--kaggle-model-sources-json", default="")
    parser.add_argument("--kaggle-attached-model-paths-json", default="")
    parser.add_argument("--public-host", default=DEFAULT_PUBLIC_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--stage-ranges-json", default="")
    parser.add_argument("--stage-groups-json", default="")
    parser.add_argument("--initial-input-ids-json", default="")
    parser.add_argument("--prompt", default="Hi")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--cpu-dtype", choices=["bfloat16", "float32", "float16"], default="bfloat16")
    parser.add_argument("--task-poll-interval-seconds", type=float, default=15.0)
    parser.add_argument("--task-idle-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--coordinator-timeout-seconds", type=float, default=4200.0)
    parser.add_argument("--kernel-timeout-seconds", type=int, default=3600)
    parser.add_argument("--kaggle-push-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--kaggle-status-timeout-seconds", type=float, default=4200.0)
    parser.add_argument("--kaggle-status-poll-interval", type=float, default=60.0)
    parser.add_argument("--kaggle-output-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--kaggle-delete-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--stage-launch-stagger-seconds", type=float, default=0.0)
    parser.add_argument("--kaggle-notebook-url", default="https://www.kaggle.com/code/tpuowner/notebook8d4184babd/edit")
    parser.add_argument("--kaggle-web-storage-state", default="/root/kaggle-web-storage-state.json")
    parser.add_argument("--chrome-executable", default="/usr/bin/google-chrome")
    parser.add_argument("--tpu-provider", choices=["kaggle_web", "colab_cli"], default="kaggle_web")
    parser.add_argument("--colab-session-name", default="ct-colab-tpu-v5e1")
    parser.add_argument("--colab-session-config", default=str(Path.home() / ".config" / "colab-cli" / "sessions.json"))
    parser.add_argument("--colab-gpu-session-name", default="ct-colab-cuda-gpu")
    parser.add_argument("--colab-gpu-session-config", default=str(Path.home() / ".config" / "colab-cli" / "sessions.json"))
    parser.add_argument("--colab-gpu-execute-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--colab-gpu-max-attempts", type=int, default=3)
    parser.add_argument("--colab-gpu-reacquire-before", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--colab-gpu-cleanup-before-reacquire", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--colab-gpu-token-cache", default=str(Path.home() / ".config" / "colab-exec" / "token.json"))
    parser.add_argument("--colab-gpu-accelerator", default="T4")
    parser.add_argument("--colab-gpu-authuser", default="0")
    parser.add_argument("--web-tpu-tensor-key", default="")
    parser.add_argument("--web-tpu-max-header-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--web-tpu-max-tensor-bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--web-tpu-execute-layer-count", type=int, default=0)
    parser.add_argument("--web-tpu-execute-timeout-seconds", type=float, default=1500.0)
    parser.add_argument("--skip-kaggle-cleanup", action="store_true")
    parser.add_argument("--keep-private-package", action="store_true")
    parser.add_argument("--keep-coordinator-private-state", action="store_true")
    parser.add_argument("--keep-kaggle-logs", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1 or args.max_new_tokens > 4:
        raise SystemExit("--max-new-tokens must be between 1 and 4 for this bounded 32B heterogeneous probe")
    if args.kernel_timeout_seconds > 3600:
        raise SystemExit("--kernel-timeout-seconds must be <= 3600")
    if args.kaggle_status_timeout_seconds > 4500:
        raise SystemExit("--kaggle-status-timeout-seconds must be <= 4500")
    if args.coordinator_timeout_seconds > 4500:
        raise SystemExit("--coordinator-timeout-seconds must be <= 4500")
    if args.stage_launch_stagger_seconds < 0 or args.stage_launch_stagger_seconds > 300:
        raise SystemExit("--stage-launch-stagger-seconds must be between 0 and 300")
    if args.colab_gpu_execute_timeout_seconds < 60 or args.colab_gpu_execute_timeout_seconds > 4500:
        raise SystemExit("--colab-gpu-execute-timeout-seconds must be between 60 and 4500")
    if args.colab_gpu_max_attempts < 1 or args.colab_gpu_max_attempts > 5:
        raise SystemExit("--colab-gpu-max-attempts must be between 1 and 5")
    stage_ranges_from_args(args)
    stage_groups_for(args)
    initial_input_ids_from_args(args)
    kaggle_model_sources(args)
    kaggle_attached_model_paths(args)
    args.coordinator_url = str(args.coordinator_url or "").rstrip("/")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_coordinator_probe(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            f"{SCHEMA}: ok={bool(report.get('ok'))} "
            f"tokens={int(report.get('generated_token_count') or 0)}/{int(args.max_new_tokens)} "
            f"topology={bool(report.get('four_t4_five_cpu_topology_verified'))} "
            f"blocked={report.get('blocked_reason') or 'none'}"
        )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
