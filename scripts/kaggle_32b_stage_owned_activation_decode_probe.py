#!/usr/bin/env python3
"""Run a private two-kernel Kaggle 32B AWQ activation/decode probe."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import kaggle_32b_stage_owned_safetensors_probe as loading_probe


SCHEMA = "kaggle_32b_stage_owned_activation_decode_probe_v1"
STAGE_REPORT_SCHEMA = "kaggle_32b_stage_owned_activation_decode_stage_v1"
DEFAULT_OUTPUT_DIR = "dist/kaggle-32b-stage-owned-activation-decode-probe"
DEFAULT_MODEL_REPO = loading_probe.DEFAULT_MODEL_REPO
DEFAULT_ACCELERATOR = loading_probe.DEFAULT_ACCELERATOR
DEFAULT_PUBLIC_HOST = "24.199.118.54"
DEFAULT_PORT = 9232
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
STAGE_ID = __STAGE_ID__
STAGE_IDS = __STAGE_IDS_JSON__
STAGE_COUNT = __STAGE_COUNT__
SPLIT_INDEX = __SPLIT_INDEX__
PROMPT_TEXT = __PROMPT_TEXT_JSON__
ACTIVATION_PAYLOAD = __ACTIVATION_PAYLOAD_JSON__
EXECUTION_MODE = "__EXECUTION_MODE__"
COORDINATOR_URL = __COORDINATOR_URL_JSON__
COORDINATOR_TOKEN = __COORDINATOR_TOKEN_JSON__
MAX_NEW_TOKENS = __MAX_NEW_TOKENS__
SINGLE_BASELINE_PLACEMENT = "__SINGLE_BASELINE_PLACEMENT__"
TASK_POLL_INTERVAL_SECONDS = __TASK_POLL_INTERVAL_SECONDS__
TASK_IDLE_TIMEOUT_SECONDS = __TASK_IDLE_TIMEOUT_SECONDS__
OUT = Path("/kaggle/working")
TEMP = Path("/kaggle/temp/ct_32b_activation_decode") / MODE
MODEL_DIR = TEMP / "model"
REPORT_PATH = OUT / f"ct_32b_activation_decode_{MODE}_report.json"
PRIVATE_ACTIVATION_PATH = OUT / "ct_32b_stage0_activation_private.json"


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


def fetch_json(filename):
    url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{filename}"
    with urllib.request.urlopen(url, timeout=120) as response:
        loaded = json.load(response)
    return loaded if isinstance(loaded, dict) else {}


def normalize_stage_count(stage_count, layer_count=0):
    count = max(2, min(int(stage_count), 16))
    if layer_count > 0:
        count = min(count, max(2, int(layer_count)))
    return count


def stage_layer_ranges(layer_count, stage_count):
    layers = max(0, int(layer_count))
    count = normalize_stage_count(stage_count, layer_count=layers)
    base = layers // count
    remainder = layers % count
    ranges = []
    cursor = 0
    for index in range(count):
        width = base + (1 if index < remainder else 0)
        start = cursor
        end = min(layers, start + width)
        ranges.append((start, end))
        cursor = end
    return ranges


def stage_prefixes(stage_id, stage_count, layer_range):
    start, end = int(layer_range[0]), int(layer_range[1])
    prefixes = [f"model.layers.{index}." for index in range(start, end)]
    if int(stage_id) == 0:
        prefixes = ["model.embed_tokens.", *prefixes]
    if int(stage_id) == int(stage_count) - 1:
        prefixes = [*prefixes, "model.norm.", "lm_head."]
    return prefixes


def build_selection(config, weight_index, stage_id=None, stage_count=None):
    selected_stage_id = int(STAGE_ID if stage_id is None else stage_id)
    selected_stage_count = int(STAGE_COUNT if stage_count is None else stage_count)
    weight_map = {
        str(key): Path(str(value)).name
        for key, value in dict(weight_index.get("weight_map") or {}).items()
        if str(key or "").strip() and str(value or "").strip()
    }
    layer_count = int(config.get("num_hidden_layers") or 0)
    count = normalize_stage_count(selected_stage_count, layer_count=layer_count)
    ranges = stage_layer_ranges(layer_count, count)
    selected_stage_id = max(0, min(selected_stage_id, count - 1))
    layer_range = ranges[selected_stage_id]
    prefixes = stage_prefixes(selected_stage_id, count, layer_range)
    assigned = sorted(key for key in weight_map if any(key.startswith(prefix) for prefix in prefixes))
    assigned_files = sorted({weight_map[key] for key in assigned if weight_map.get(key)})
    return {
        "model_type": str(config.get("model_type") or ""),
        "architectures": list(config.get("architectures") or []),
        "quantization_config": dict(config.get("quantization_config") or {}),
        "num_hidden_layers": layer_count,
        "hidden_size": int(config.get("hidden_size") or 0),
        "vocab_size": int(config.get("vocab_size") or 0),
        "stage_id": int(selected_stage_id),
        "stage_count": count,
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


def parent_module_and_attr(root, dotted):
    parts = [part for part in str(dotted or "").split(".") if part]
    if not parts:
        raise ValueError("module path is empty")
    parent = module_by_name(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    return parent, parts[-1]


def awq_dequantize_gemm(qweight, qzeros, scales, bits=4, group_size=128):
    import torch
    shifts = torch.arange(0, 32, int(bits), device=qweight.device, dtype=torch.int32)
    iweight = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(torch.int16)
    iweight = iweight.view(iweight.shape[0], -1)
    izeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(torch.int16)
    izeros = izeros.view(izeros.shape[0], -1)
    reverse = torch.arange(iweight.shape[-1], dtype=torch.long, device=iweight.device)
    reverse = reverse.view(-1, 32 // int(bits))[:, [0, 4, 1, 5, 2, 6, 3, 7]].reshape(-1)
    iweight = torch.bitwise_and(iweight[:, reverse], (2**int(bits)) - 1)
    izeros = torch.bitwise_and(izeros[:, reverse], (2**int(bits)) - 1)
    scales = scales.repeat_interleave(int(group_size), dim=0)
    izeros = izeros.repeat_interleave(int(group_size), dim=0)
    return (iweight.to(scales.dtype) - izeros.to(scales.dtype)) * scales


class AWQGemmLinear:
    pass


def make_awq_linear(linear, bits=4, group_size=128):
    import torch
    import torch.nn as nn

    class AWQGemmLinearModule(nn.Module):
        def __init__(self, source):
            super().__init__()
            self.in_features = int(source.in_features)
            self.out_features = int(source.out_features)
            self.w_bit = int(bits)
            self.group_size = int(group_size)
            self.register_buffer("qweight", torch.empty((self.in_features, self.out_features // (32 // self.w_bit)), dtype=torch.int32, device="meta"))
            self.register_buffer("qzeros", torch.empty((self.in_features // self.group_size, self.out_features // (32 // self.w_bit)), dtype=torch.int32, device="meta"))
            self.register_buffer("scales", torch.empty((self.in_features // self.group_size, self.out_features), dtype=torch.float16, device="meta"))
            if getattr(source, "bias", None) is not None:
                self.register_buffer("bias", torch.empty((self.out_features,), dtype=torch.float16, device="meta"))
            else:
                self.bias = None

        def forward(self, x):
            import torch
            out_shape = tuple(x.shape[:-1]) + (self.out_features,)
            original_dtype = x.dtype
            flat = x.reshape(-1, x.shape[-1]).to(dtype=torch.float16)
            weight = awq_dequantize_gemm(self.qweight, self.qzeros, self.scales, self.w_bit, self.group_size)
            out = torch.matmul(flat, weight)
            del weight
            if self.bias is not None:
                out = out + self.bias
            out = out.reshape(out_shape)
            return out.to(dtype=original_dtype) if out.dtype != original_dtype else out

    return AWQGemmLinearModule(linear)


def replace_awq_linears(model, selection):
    import torch.nn as nn
    quant = dict(selection.get("quantization_config") or {})
    bits = int(quant.get("bits") or 4)
    group_size = int(quant.get("group_size") or 128)
    quantized_modules = {
        key[:-len(".qweight")]
        for key in list(selection.get("assigned_weight_keys") or [])
        if str(key).endswith(".qweight")
    }
    replaced = []
    for name, module in list(model.named_modules()):
        if not name or name not in quantized_modules or not isinstance(module, nn.Linear):
            continue
        parent, attr = parent_module_and_attr(model, name)
        setattr(parent, attr, make_awq_linear(module, bits=bits, group_size=group_size))
        replaced.append(name)
    return {
        "awq_linear_replacement_count": len(replaced),
        "awq_linear_replacement_digest": sha_payload(sorted(replaced)),
        "awq_stage_model_prepared": bool(replaced),
        "awq_bits": bits,
        "awq_group_size": group_size,
    }


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
        if name.endswith((".qweight", ".qzeros", ".scales", ".bias")):
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
        if bool(getattr(buffer, "is_meta", False)) and not name.endswith((".qweight", ".qzeros", ".scales", ".bias"))
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
    named_buffers = dict(model.named_buffers(recurse=True))
    state = model.state_dict()
    assign_state = {}
    loaded = []
    loaded_bytes = 0
    missing = []
    shape_mismatches = []
    downloads = []
    for filename in list(selection.get("assigned_weight_files") or []):
        downloads.append(download_file(filename))
    for filename in list(selection.get("assigned_weight_files") or []):
        path = MODEL_DIR / Path(filename).name
        with safe_open(path, framework="pt", device="cpu") as handle:
            available = set(str(key) for key in handle.keys())
            expected = [key for key in assigned if weight_map.get(key) == Path(filename).name]
            for key in expected:
                if key not in available:
                    missing.append(key)
                    continue
                if key not in state:
                    missing.append(key)
                    continue
                source = handle.get_tensor(key)
                target = state[key]
                if tuple(source.shape) != tuple(target.shape):
                    shape_mismatches.append(key)
                    continue
                prepared = source.to(device=device, dtype=target.dtype)
                if key in named_buffers:
                    module_path, attr = key.rsplit(".", 1)
                    module = module_by_name(model, module_path)
                    module.register_buffer(attr, prepared, persistent=attr not in getattr(module, "_non_persistent_buffers_set", set()))
                elif bool(getattr(target, "is_meta", False)):
                    assign_state[key] = prepared
                else:
                    target.copy_(prepared)
                loaded.append(key)
                loaded_bytes += int(source.numel()) * int(source.element_size())
    if assign_state:
        model.load_state_dict(assign_state, strict=False, assign=True)
    loaded_set = set(loaded)
    ready = bool(loaded and loaded_set == assigned_set and not missing and not shape_mismatches)
    return {
        "ready": ready,
        "downloads": downloads,
        "loaded_weight_key_count": len(loaded_set),
        "assigned_weight_key_count": len(assigned_set),
        "loaded_tensor_bytes": int(loaded_bytes),
        "loaded_tensor_gb": round(loaded_bytes / 1024 / 1024 / 1024, 6),
        "loaded_weight_key_digest": sha_payload(sorted(loaded)),
        "missing_weight_key_count": len(missing),
        "shape_mismatch_count": len(shape_mismatches),
        "loads_only_stage_weight_keys": bool(loaded_set.issubset(assigned_set)),
        "blockers": ([] if ready else ["stage_weight_apply_not_ready"]),
    }


def prepare_stage_model(config, selection, device):
    import gc
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM
    with init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(config)
    awq = replace_awq_linears(model, selection)
    buffers = materialize_runtime_buffers(model, device)
    stage_load = load_stage_weights(model, selection, device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.eval()
    return model, awq, buffers, stage_load


def single_kernel_stage_device(stage_id):
    import torch
    placement = str(SINGLE_BASELINE_PLACEMENT or "two_stage_two_gpu")
    if placement == "strict_stage_count":
        if torch.cuda.device_count() < int(STAGE_COUNT):
            raise RuntimeError("single_kernel_gpu_count_below_stage_count")
        return torch.device(f"cuda:{int(stage_id)}")
    return torch.device(f"cuda:{min(int(stage_id), max(0, torch.cuda.device_count() - 1))}")


def run_single_kernel_split_baseline(config, tokenizer, config_json, weight_index):
    import gc
    import torch
    if torch.cuda.device_count() < 2:
        return {
            "ok": False,
            "blockers": ["single_kernel_t4x2_requires_two_cuda_devices"],
            "diagnosis_codes": ["single_kernel_t4x2_gpu_count_insufficient"],
        }
    if str(SINGLE_BASELINE_PLACEMENT or "two_stage_two_gpu") == "strict_stage_count" and torch.cuda.device_count() < int(STAGE_COUNT):
        return {
            "ok": False,
            "schema": "kaggle_32b_single_t4x2_stage_split_baseline_v1",
            "generated_token_count": 0,
            "max_new_tokens": int(MAX_NEW_TOKENS),
            "stage_count": int(STAGE_COUNT),
            "gpu_count": int(torch.cuda.device_count()),
            "blockers": ["single_kernel_t4x2_gpu_count_below_required_stage_count"],
            "diagnosis_codes": ["single_kernel_t4x2_exceeds_gpu_count"],
            "public_artifact_safe": True,
        }
    started = time.monotonic()
    stages = []
    for stage_id in range(int(STAGE_COUNT)):
        device = single_kernel_stage_device(stage_id)
        selection = build_selection(config_json, weight_index, stage_id=stage_id, stage_count=int(STAGE_COUNT))
        model, awq, buffers, load = prepare_stage_model(config, selection, device)
        stages.append({
            "stage_id": stage_id,
            "device": device,
            "selection": selection,
            "model": model,
            "awq": awq,
            "buffers": buffers,
            "load": load,
            "cuda_memory_after_load": cuda_memory_summary(device),
        })
    input_ids_values = None
    token_hashes = []
    output_hashes = []
    step_rows = []
    for step in range(int(MAX_NEW_TOKENS)):
        step_started = time.monotonic()
        activation = run_stage_activation(
            stages[0]["model"],
            tokenizer,
            stages[0]["selection"],
            stages[0]["device"],
            input_ids_values=input_ids_values,
            task_id=f"single-kernel-step-{step}",
            generation_step=step,
        )
        for stage in stages[1:-1]:
            activation = run_stage_activation(
                stage["model"],
                tokenizer,
                stage["selection"],
                stage["device"],
                activation=activation,
                task_id=f"single-kernel-step-{step}-stage{stage['stage_id']}",
                generation_step=step,
            )
        final_stage = stages[-1]
        decoded = final_stage_decode(
            final_stage["model"],
            tokenizer,
            final_stage["selection"],
            activation,
            final_stage["device"],
        )
        input_ids_values = [int(value) for value in activation.get("input_ids") or []]
        input_ids_values.append(int(decoded.get("next_token_id_private")))
        token_hashes.append(str(decoded.get("next_token_hash") or ""))
        output_hashes.append(str(decoded.get("output_hash") or ""))
        step_rows.append({
            "generation_step": step,
            "activation_hash": activation.get("activation_hash"),
            "next_token_hash": decoded.get("next_token_hash"),
            "output_hash": decoded.get("output_hash"),
            "duration_seconds": round(time.monotonic() - step_started, 3),
        })
        gc.collect()
        torch.cuda.empty_cache()
    report = {
        "ok": len(token_hashes) >= int(MAX_NEW_TOKENS),
        "schema": "kaggle_32b_single_t4x2_stage_split_baseline_v1",
        "generated_token_count": len(token_hashes),
        "max_new_tokens": int(MAX_NEW_TOKENS),
        "stage_count": int(STAGE_COUNT),
        "gpu_count": int(torch.cuda.device_count()),
        "stage_summaries": [
            {
                "stage_id": stage["stage_id"],
                "device": str(stage["device"]),
                "loaded_tensor_gb": stage["load"].get("loaded_tensor_gb"),
                "loaded_weight_key_count": stage["load"].get("loaded_weight_key_count"),
                "awq_stage_model_prepared": stage["awq"].get("awq_stage_model_prepared"),
                "runtime_buffers_ready": stage["buffers"].get("ready"),
                "cuda_memory_after_load": stage["cuda_memory_after_load"],
            }
            for stage in stages
        ],
        "step_rows": step_rows,
        "generated_token_hashes": token_hashes,
        "output_hashes": output_hashes,
        "wall_time_seconds": round(time.monotonic() - started, 3),
        "tokens_per_second": round(len(token_hashes) / max(0.001, time.monotonic() - started), 6),
        "raw_prompt_public": False,
        "generated_token_ids_public": False,
        "activation_public": False,
        "hidden_state_public": False,
        "public_artifact_safe": True,
        "diagnosis_codes": ["single_kernel_t4x2_stage_split_baseline_ready"],
        "blockers": [],
    }
    for stage in report["stage_summaries"]:
        report[f"stage{int(stage['stage_id'])}"] = {
            key: value
            for key, value in stage.items()
            if key != "stage_id"
        }
    return report


def causal_attention_mask(token_count, dtype, device):
    import torch
    mask = torch.full((token_count, token_count), torch.finfo(dtype).min, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :]


def cache_position(token_count, device):
    import torch
    return torch.arange(max(1, int(token_count)), dtype=torch.long, device=device)


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


def call_layer(layer, hidden, attention_mask, position_ids, cache_pos, position_embeddings):
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
        kwargs["past_key_value"] = None
    if "output_attentions" in params:
        kwargs["output_attentions"] = False
    if "use_cache" in params:
        kwargs["use_cache"] = False
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
        "schema": "kaggle_32b_private_activation_v2",
        "model_repo": MODEL_REPO,
        "stage_count": int(selection.get("stage_count") or STAGE_COUNT),
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


def run_layers(model, hidden, position_ids, device, *, start, end):
    import torch
    base = model.model
    layers = list(base.layers)
    with torch.no_grad():
        attention_mask = causal_attention_mask(int(hidden.shape[1]), hidden.dtype, device)
        cache_pos = cache_position(int(hidden.shape[1]), device)
        position_embeddings = llama_position_embeddings(base, hidden, position_ids)
        for index, layer in enumerate(layers[int(start):int(end)], start=int(start)):
            hidden = output_hidden(call_layer(layer, hidden, attention_mask, position_ids, cache_pos, position_embeddings))
            if index % 4 == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    return hidden


def run_stage_activation(model, tokenizer, selection, device, *, input_ids_values=None, activation=None, task_id="", generation_step=0):
    import torch
    base = model.model
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
        hidden = torch.frombuffer(bytearray(raw), dtype=torch.float16).reshape(shape).to(device)
        position_ids = torch.tensor([list(activation.get("position_ids") or range(hidden.shape[1]))], dtype=torch.long, device=device)
    else:
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
        hidden = base.embed_tokens(input_ids)
    hidden = run_layers(model, hidden, position_ids, device, start=start, end=end)
    return encode_activation_payload(
        hidden=hidden,
        input_ids=input_ids,
        position_ids=position_ids,
        selection=selection,
        task_id=task_id,
        generation_step=generation_step,
    )


def stage0_activation(model, tokenizer, selection, device, *, input_ids_values=None, task_id="", generation_step=0):
    return run_stage_activation(
        model,
        tokenizer,
        selection,
        device,
        input_ids_values=input_ids_values,
        task_id=task_id,
        generation_step=generation_step,
    )


def final_stage_decode(model, tokenizer, selection, activation, device):
    import torch
    base = model.model
    stage_range = list(selection.get("stage_layer_range") or [0, 0])
    start = int(stage_range[0])
    end = int(stage_range[1])
    shape = [int(value) for value in list(activation.get("hidden_shape") or [])]
    raw = base64.b64decode(str(activation.get("hidden_b64") or ""))
    hidden = torch.frombuffer(bytearray(raw), dtype=torch.float16).reshape(shape).to(device)
    position_ids = torch.tensor([list(activation.get("position_ids") or range(hidden.shape[1]))], dtype=torch.long, device=device)
    hidden = run_layers(model, hidden, position_ids, device, start=start, end=end)
    with torch.no_grad():
        hidden = base.norm(hidden)
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
    }


def stage1_decode(model, tokenizer, activation, device):
    config = getattr(getattr(model, "config", None), "to_dict", lambda: {})()
    weight_index = fetch_json("model.safetensors.index.json")
    selection = build_selection(config, weight_index, stage_id=1, stage_count=2)
    return final_stage_decode(model, tokenizer, selection, activation, device)


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


def worker_loop(stage_runtime, tokenizer, report):
    processed = []
    deadline = time.monotonic() + float(TASK_IDLE_TIMEOUT_SECONDS)
    miner_id = "kaggle-32b-" + MODE
    while time.monotonic() < deadline:
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
        if stage_id == 0:
            activation = run_stage_activation(
                model,
                tokenizer,
                selection,
                device,
                input_ids_values=task.get("input_ids") or [],
                task_id=task_id,
                generation_step=generation_step,
            )
            result = {
                "task_id": task_id,
                "stage_id": 0,
                "generation_step": generation_step,
                "activation": activation,
                "activation_hash": activation.get("activation_hash"),
                "input_token_count": len(activation.get("input_ids") or []),
                "duration_seconds": round(time.monotonic() - task_started, 3),
                "cuda_memory_after_task": cuda_memory_summary(),
            }
        elif stage_id < int(STAGE_COUNT) - 1:
            activation = dict(task.get("activation") or {})
            next_activation = run_stage_activation(
                model,
                tokenizer,
                selection,
                device,
                activation=activation,
                task_id=task_id,
                generation_step=generation_step,
            )
            result = {
                "task_id": task_id,
                "stage_id": stage_id,
                "generation_step": generation_step,
                "activation": next_activation,
                "activation_hash": next_activation.get("activation_hash"),
                "input_token_count": len(next_activation.get("input_ids") or []),
                "duration_seconds": round(time.monotonic() - task_started, 3),
                "cuda_memory_after_task": cuda_memory_summary(),
            }
        else:
            activation = dict(task.get("activation") or {})
            decoded = final_stage_decode(model, tokenizer, selection, activation, device)
            result = {
                "task_id": task_id,
                "stage_id": stage_id,
                "generation_step": generation_step,
                "activation_hash": activation.get("activation_hash"),
                "next_token_id_private": decoded.get("next_token_id_private"),
                "next_token_hash": decoded.get("next_token_hash"),
                "output_hash": decoded.get("output_hash"),
                "generated_token_count": decoded.get("generated_token_count"),
                "duration_seconds": round(time.monotonic() - task_started, 3),
                "cuda_memory_after_task": cuda_memory_summary(),
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
            })
            if response.get("ready") is True:
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
        "execution_mode": EXECUTION_MODE,
        "stage_id": STAGE_ID,
        "stage_count": STAGE_COUNT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "model_repo": MODEL_REPO,
        "ok": False,
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
            report["diagnosis_codes"].append("kaggle_32b_activation_decode_dependencies_missing")
            write_json(REPORT_PATH, report)
            return
        import torch
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        config_json = fetch_json("config.json")
        weight_index = fetch_json("model.safetensors.index.json")
        stage_ids = [int(value) for value in (STAGE_IDS or [STAGE_ID])]
        selections = [
            build_selection(config_json, weight_index, stage_id=stage_id, stage_count=STAGE_COUNT)
            for stage_id in stage_ids
        ]
        report["stage_ids"] = stage_ids
        report["selection"] = {
            key: value
            for key, value in selections[0].items()
            if key not in {"assigned_weight_keys", "weight_map"}
        }
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
            report["diagnosis_codes"].append("kaggle_32b_activation_decode_stage_selection_empty")
            write_json(REPORT_PATH, report)
            return
        if not torch.cuda.is_available():
            report["blockers"].append("cuda_runtime_missing")
            report["diagnosis_codes"].append("kaggle_32b_activation_decode_cuda_missing")
            write_json(REPORT_PATH, report)
            return
        if MODE != "single_baseline" and len(stage_ids) > torch.cuda.device_count():
            report["blockers"].append("kernel_stage_count_exceeds_cuda_device_count")
            report["diagnosis_codes"].append("kaggle_32b_kernel_stage_count_exceeds_cuda_device_count")
            write_json(REPORT_PATH, report)
            return
        torch.set_grad_enabled(False)
        runtime_stage = "config_load"
        config = AutoConfig.from_pretrained(MODEL_REPO)
        runtime_stage = "tokenizer_load"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
        runtime_stage = "model_instantiate_meta"
        if MODE == "single_baseline":
            runtime_stage = "single_kernel_t4x2_baseline"
            baseline = run_single_kernel_split_baseline(config, tokenizer, config_json, weight_index)
            report.update({
                "ok": bool(baseline.get("ok")),
                "single_kernel_baseline": baseline,
                "generated_token_count": int(baseline.get("generated_token_count") or 0),
                "diagnosis_codes": [
                    *report["diagnosis_codes"],
                    *list(baseline.get("diagnosis_codes") or []),
                ],
                "blockers": [
                    *report["blockers"],
                    *list(baseline.get("blockers") or []),
                ],
            })
            report["cuda_memory_after_execution"] = cuda_memory_summary()
            return
        stage_runtime = []
        for index, selection in enumerate(selections):
            device = torch.device(f"cuda:{index}")
            model, awq, buffers, stage_load = prepare_stage_model(config, selection, device)
            stage_runtime.append({
                "stage_id": int(selection.get("stage_id")),
                "device": device,
                "selection": selection,
                "model": model,
                "awq": awq,
                "buffers": buffers,
                "stage_load": stage_load,
                "cuda_memory_after_load": cuda_memory_summary(device),
            })
        first = stage_runtime[0]
        report["awq_stage_preparation"] = first["awq"]
        report["runtime_buffers"] = first["buffers"]
        report["stage_weight_load"] = {key: value for key, value in first["stage_load"].items() if key != "downloads"}
        report["downloads"] = first["stage_load"].get("downloads") or []
        report["cuda_memory_after_load"] = first["cuda_memory_after_load"]
        report["stage_runtime_summaries"] = [
            {
                "stage_id": item["stage_id"],
                "device": str(item["device"]),
                "selection": {
                    key: value
                    for key, value in item["selection"].items()
                    if key not in {"assigned_weight_keys", "weight_map"}
                },
                "awq_stage_preparation": item["awq"],
                "runtime_buffers": item["buffers"],
                "stage_weight_load": {
                    key: value
                    for key, value in item["stage_load"].items()
                    if key != "downloads"
                },
                "cuda_memory_after_load": item["cuda_memory_after_load"],
            }
            for item in stage_runtime
        ]
        if not all(
            item["awq"].get("awq_stage_model_prepared")
            and item["buffers"].get("ready")
            and item["stage_load"].get("ready")
            for item in stage_runtime
        ):
            report["blockers"].append("stage_runtime_not_ready")
            report["diagnosis_codes"].append("kaggle_32b_activation_decode_stage_runtime_not_ready")
            write_json(REPORT_PATH, report)
            return
        if EXECUTION_MODE == "coordinator":
            runtime_stage = "coordinator_worker_loop"
            tasks = worker_loop(stage_runtime, tokenizer, report)
            accepted = [item for item in tasks if item.get("accepted")]
            report.update({
                "ok": bool(accepted or MAX_NEW_TOKENS == 0),
                "worker_loop_ready": True,
                "processed_task_count": len(accepted),
                "stage0_activation_public": False,
                "diagnosis_codes": [
                    *report["diagnosis_codes"],
                    "kaggle_32b_coordinator_worker_ready",
                    "kaggle_32b_stage_owned_awq_runtime_ready",
                ],
            })
            if not accepted:
                report["blockers"].append("coordinator_worker_processed_no_tasks")
        elif MODE == "stage0":
            runtime_stage = "stage0_activation"
            activation = stage0_activation(first["model"], tokenizer, first["selection"], first["device"])
            write_json(PRIVATE_ACTIVATION_PATH, activation)
            report.update({
                "ok": True,
                "activation_ready": True,
                "activation_hash": activation.get("activation_hash"),
                "activation_private_file": PRIVATE_ACTIVATION_PATH.name,
                "activation_bytes": int(len(json.dumps(activation).encode("utf-8"))),
                "activation_shape": activation.get("hidden_shape"),
                "input_token_count": len(activation.get("input_ids") or []),
                "stage0_activation_public": False,
                "diagnosis_codes": [
                    *report["diagnosis_codes"],
                    "kaggle_32b_stage0_activation_ready",
                    "kaggle_32b_stage_owned_awq_runtime_ready",
                ],
            })
        else:
            runtime_stage = "stage1_activation_decode"
            activation = dict(ACTIVATION_PAYLOAD or {})
            result = final_stage_decode(first["model"], tokenizer, first["selection"], activation, first["device"])
            report.update({
                "ok": bool(result.get("generated_token_count") == 1),
                "activation_hash": activation.get("activation_hash"),
                "activation_ready": bool(activation.get("activation_hash")),
                "stage1_decode_ready": bool(result.get("generated_token_count") == 1),
                "generated_token_count": int(result.get("generated_token_count") or 0),
                "next_token_id_public": False,
                "next_token_text_public": False,
                "next_token_hash": result.get("next_token_hash"),
                "output_hash": result.get("output_hash"),
                "diagnosis_codes": [
                    *report["diagnosis_codes"],
                    "kaggle_32b_stage1_decode_ready" if result.get("generated_token_count") == 1 else "kaggle_32b_stage1_decode_not_ready",
                    "kaggle_32b_stage_owned_awq_runtime_ready",
                ],
            })
            if not report["ok"]:
                report["blockers"].append("stage1_decode_failed")
        report["cuda_memory_after_execution"] = cuda_memory_summary()
    except Exception as exc:
        report["ok"] = False
        report["error_type"] = type(exc).__name__
        report["error_stage"] = locals().get("runtime_stage", "unknown")
        report["error_digest"] = sha_text(str(exc))
        report["error_public"] = safe_tail(str(exc), limit=240)
        report["diagnosis_codes"].append("kaggle_32b_activation_decode_exception")
        report["blockers"].append("kaggle_32b_activation_decode_exception")
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


class StageCoordinatorState:
    def __init__(self, *, prompt: str, max_new_tokens: int, stage_count: int = 2) -> None:
        self.prompt_hash = "sha256:" + __import__("hashlib").sha256(
            str(prompt or "").encode("utf-8", errors="replace")
        ).hexdigest()
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.stage_count = max(2, int(stage_count))
        self.input_ids: list[int] = []
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
        self._queue_stage(generation_step=0, stage_id=0, input_ids=[])

    def _new_task_id(self, stage_id: int, generation_step: int) -> str:
        self._counter += 1
        return f"ct32b-{self._counter:04d}-stage{stage_id}-step{generation_step}"

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
                    # The source input ids are private; use activation payload only for the next loop.
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
                    "cuda_memory_after_task": task.get("cuda_memory_after_task") or {},
                })
            return {
                "schema": "kaggle_32b_probe_coordinator_status_v1",
                "ok": True,
                "ready": self.ready(),
                "prompt_hash": self.prompt_hash,
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
                "stage0_task_count": sum(1 for item in self.completed if int_field(item.get("stage_id")) == 0),
                "stage1_task_count": sum(1 for item in self.completed if int_field(item.get("stage_id")) == 1),
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


def render_kernel(
    args: argparse.Namespace,
    *,
    mode: str,
    stage_id: int,
    stage_ids: list[int] | None = None,
    activation_payload: dict[str, Any] | None = None,
    coordinator_token: str = "",
) -> str:
    prompt = str(args.prompt or "Hi")[:64]
    rendered = KERNEL_TEMPLATE
    replacements = {
        "__STAGE_REPORT_SCHEMA__": STAGE_REPORT_SCHEMA,
        "__MODE__": mode,
        "__MODEL_REPO_JSON__": json.dumps(args.model_repo),
        "__STAGE_ID__": str(int(stage_id)),
        "__STAGE_IDS_JSON__": json.dumps([int(value) for value in (stage_ids or [stage_id])]),
        "__STAGE_COUNT__": str(int(args.stage_count)),
        "__SPLIT_INDEX__": str(int(args.split_index)),
        "__PROMPT_TEXT_JSON__": json.dumps(prompt),
        "__ACTIVATION_PAYLOAD_JSON__": json.dumps(activation_payload or {}),
        "__EXECUTION_MODE__": str(args.execution_mode),
        "__COORDINATOR_URL_JSON__": json.dumps(coordinator_url_for(args)),
        "__COORDINATOR_TOKEN_JSON__": json.dumps(coordinator_token),
        "__MAX_NEW_TOKENS__": str(int(args.max_new_tokens)),
        "__SINGLE_BASELINE_PLACEMENT__": str(args.single_baseline_placement),
        "__TASK_POLL_INTERVAL_SECONDS__": str(float(args.task_poll_interval_seconds)),
        "__TASK_IDLE_TIMEOUT_SECONDS__": str(float(args.task_idle_timeout_seconds)),
    }
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def build_package(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    mode: str,
    stage_id: int,
    stage_ids: list[int] | None = None,
    activation_payload: dict[str, Any] | None = None,
    coordinator_token: str = "",
    slug_prefix: str | None = None,
) -> dict[str, Any]:
    owner = args.kaggle_owner or loading_probe.default_kaggle_owner()
    if not owner:
        raise SystemExit("--kaggle-owner or ~/.kaggle/kaggle.json username is required")
    suffix = str(int(time.time()))[-8:]
    prefix = str(slug_prefix or args.kernel_slug_prefix)
    slug = f"{loading_probe.safe_slug(prefix)[:25]}-{mode}-{suffix}"
    slug = slug[:45].strip("-")
    kernel_dir = output_dir / "private-kaggle-kernels" / mode
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text(
        render_kernel(
            args,
            mode=mode,
            stage_id=stage_id,
            stage_ids=stage_ids,
            activation_payload=activation_payload,
            coordinator_token=coordinator_token,
        ),
        encoding="utf-8",
    )
    title = f"CT 32B Activation Decode {mode} {suffix}"
    metadata = {
        "id": f"{owner}/{slug}",
        "title": title,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": args.accelerator,
    }
    write_json(kernel_dir / "kernel-metadata.json", metadata)
    return {
        "mode": mode,
        "stage_id": int(stage_id),
        "stage_ids": [int(value) for value in (stage_ids or [stage_id])],
        "kernel_dir": kernel_dir,
        "declared_kernel_ref": metadata["id"],
        "kernel_ref": metadata["id"],
        "kernel_slug": slug,
        "metadata": metadata,
        "report_filename": f"ct_32b_activation_decode_{mode}_report.json",
    }


def coordinator_url_for(args: argparse.Namespace) -> str:
    if str(getattr(args, "coordinator_url", "") or "").strip():
        return str(args.coordinator_url).rstrip("/")
    return f"http://{args.public_host}:{int(args.port)}"


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


def run_kaggle_package(
    args: argparse.Namespace,
    *,
    package: dict[str, Any],
    output_dir: Path,
    runner: Runner,
    file_patterns: list[str],
    cleanup: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    steps: list[dict[str, Any]] = []
    push_command = ["kaggle", "kernels", "push", "-p", str(package["kernel_dir"]), "-t", str(args.kernel_timeout_seconds)]
    if args.accelerator:
        push_command.extend(["--accelerator", args.accelerator])
    print(f"[{utc_now()}] pushing private Kaggle kernel {package['declared_kernel_ref']}", flush=True)
    push_step = loading_probe.run_step(
        "kaggle_kernel_push",
        push_command,
        runner=runner,
        timeout_seconds=args.kaggle_push_timeout_seconds,
    )
    steps.append(push_step)
    if not push_step.get("ok"):
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
    for pattern in file_patterns:
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
                pattern,
            ],
            runner=runner,
            timeout_seconds=args.kaggle_output_timeout_seconds,
        )
        output_step["file_pattern"] = pattern
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


def summarize_stage(report: dict[str, Any]) -> dict[str, Any]:
    hardware = report.get("hardware") if isinstance(report.get("hardware"), dict) else {}
    selection = report.get("selection") if isinstance(report.get("selection"), dict) else {}
    load = report.get("stage_weight_load") if isinstance(report.get("stage_weight_load"), dict) else {}
    cuda_after_load = report.get("cuda_memory_after_load") if isinstance(report.get("cuda_memory_after_load"), dict) else {}
    cuda_after_execution = (
        report.get("cuda_memory_after_execution")
        if isinstance(report.get("cuda_memory_after_execution"), dict)
        else {}
    )
    return {
        "mode": report.get("mode"),
        "stage_id": report.get("stage_id"),
        "ok": report.get("ok") is True,
        "gpu_verified": hardware.get("kaggle_gpu_verified") is True,
        "gpu_count": hardware.get("gpu_count"),
        "gpu_names": hardware.get("gpu_names") or [],
        "stage_layer_range": selection.get("stage_layer_range") or [],
        "assigned_weight_key_count": selection.get("assigned_weight_key_count"),
        "assigned_weight_file_count": selection.get("assigned_weight_file_count"),
        "loaded_weight_key_count": load.get("loaded_weight_key_count"),
        "loaded_tensor_gb": load.get("loaded_tensor_gb"),
        "cuda_memory_after_load": cuda_after_load,
        "cuda_memory_after_execution": cuda_after_execution,
        "awq_stage_model_prepared": bool((report.get("awq_stage_preparation") or {}).get("awq_stage_model_prepared")),
        "activation_ready": report.get("activation_ready") is True,
        "stage1_decode_ready": report.get("stage1_decode_ready") is True,
        "generated_token_count": report.get("generated_token_count"),
        "activation_hash": report.get("activation_hash"),
        "output_hash": report.get("output_hash"),
        "diagnosis_codes": report.get("diagnosis_codes") or [],
        "blockers": report.get("blockers") or [],
    }


def summarize_stage_runtime(item: dict[str, Any], *, parent_report: dict[str, Any]) -> dict[str, Any]:
    selection = item.get("selection") if isinstance(item.get("selection"), dict) else {}
    load = item.get("stage_weight_load") if isinstance(item.get("stage_weight_load"), dict) else {}
    awq = item.get("awq_stage_preparation") if isinstance(item.get("awq_stage_preparation"), dict) else {}
    buffers = item.get("runtime_buffers") if isinstance(item.get("runtime_buffers"), dict) else {}
    return {
        "mode": parent_report.get("mode"),
        "stage_id": item.get("stage_id"),
        "device": item.get("device"),
        "ok": parent_report.get("ok") is True,
        "gpu_verified": (parent_report.get("hardware") or {}).get("kaggle_gpu_verified") is True
        if isinstance(parent_report.get("hardware"), dict)
        else False,
        "gpu_count": (parent_report.get("hardware") or {}).get("gpu_count")
        if isinstance(parent_report.get("hardware"), dict)
        else None,
        "gpu_names": (parent_report.get("hardware") or {}).get("gpu_names") or []
        if isinstance(parent_report.get("hardware"), dict)
        else [],
        "stage_layer_range": selection.get("stage_layer_range") or [],
        "assigned_weight_key_count": selection.get("assigned_weight_key_count"),
        "assigned_weight_file_count": selection.get("assigned_weight_file_count"),
        "loaded_weight_key_count": load.get("loaded_weight_key_count"),
        "loaded_tensor_gb": load.get("loaded_tensor_gb"),
        "cuda_memory_after_load": item.get("cuda_memory_after_load") or {},
        "cuda_memory_after_execution": parent_report.get("cuda_memory_after_execution") or {},
        "awq_stage_model_prepared": bool(awq.get("awq_stage_model_prepared")),
        "runtime_buffers_ready": buffers.get("ready") is True,
        "activation_ready": parent_report.get("activation_ready") is True,
        "stage1_decode_ready": parent_report.get("stage1_decode_ready") is True,
        "generated_token_count": parent_report.get("generated_token_count"),
        "activation_hash": parent_report.get("activation_hash"),
        "output_hash": parent_report.get("output_hash"),
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
    return [summarize_stage(report)]


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


def build_report(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    stage0_report: dict[str, Any],
    stage1_report: dict[str, Any],
    stage_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    stage0_summary = summarize_stage(stage0_report)
    stage1_summary = summarize_stage(stage1_report)
    lifecycle = {
        "requested_accelerator": args.accelerator,
        "kernel_refs": {
            str(run.get("mode")): str(run.get("kernel_ref") or "")
            for run in stage_runs
            if run.get("kernel_ref")
        },
        "actual_push_count": sum(
            1
            for run in stage_runs
            if any(step.get("name") == "kaggle_kernel_push" and step.get("ok") for step in run.get("steps", []))
        ),
        "kernels_deleted": all(
            any(step.get("name") == "kaggle_kernel_delete" and step.get("ok") for step in run.get("steps", []))
            for run in stage_runs
        ) if stage_runs else False,
        "private_packages_removed": not (output_dir / "private-kaggle-kernels").exists(),
        "private_activation_removed": not (output_dir / "kaggle-output" / "stage0" / "ct_32b_stage0_activation_private.json").exists(),
    }
    activation_match = bool(
        stage0_summary.get("activation_hash")
        and stage0_summary.get("activation_hash") == stage1_summary.get("activation_hash")
    )
    ready = bool(
        stage0_summary.get("ok")
        and stage1_summary.get("ok")
        and activation_match
        and int(stage1_summary.get("generated_token_count") or 0) == 1
        and lifecycle["kernels_deleted"]
        and lifecycle["private_packages_removed"]
        and lifecycle["private_activation_removed"]
    )
    blockers: list[str] = []
    if not stage0_summary.get("ok"):
        blockers.append("stage0_activation_not_ready")
    if not stage1_summary.get("ok"):
        blockers.append("stage1_decode_not_ready")
    if not activation_match:
        blockers.append("activation_hash_handoff_not_verified")
    if not lifecycle["kernels_deleted"]:
        blockers.append("kaggle_kernels_cleanup_not_verified")
    if not lifecycle["private_activation_removed"]:
        blockers.append("private_activation_artifact_retained")
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "fresh_kaggle_run_performed": lifecycle["actual_push_count"] > 0,
        "cross_kernel_activation_decode_verified": ready,
        "one_token_generation_verified": ready,
        "stage_owned_awq_runtime_verified": bool(
            stage0_summary.get("awq_stage_model_prepared")
            and stage1_summary.get("awq_stage_model_prepared")
        ),
        "activation_handoff_verified": activation_match,
        "blocked_reason": "" if ready else (blockers[0] if blockers else "kaggle_32b_activation_decode_not_ready"),
        "blockers": blockers,
        "model": {
            "repo": args.model_repo,
            "parameter_count_b": 32,
            "quantization": "awq",
            "split_index": int(args.split_index),
            "stage_count": int(args.stage_count),
        },
        "stage_summaries": [stage0_summary, stage1_summary],
        "kaggle_lifecycle": lifecycle,
        "stage_runs": [public_stage_run(run) for run in stage_runs],
        "safety": {
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "credentials_public": False,
            "private_kernel_payload_public": False,
        },
        "limitations": [
            "This is a sequential two-private-kernel Kaggle proof with local orchestrator-mediated activation handoff, not a production Coordinator data plane.",
            "The activation is private runtime state and is removed from local retained artifacts by default.",
            "Only a 1-token decode is attempted.",
        ],
    }


def build_coordinator_report(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    coordinator_status: dict[str, Any],
    stage_reports: list[dict[str, Any]],
    stage_runs: list[dict[str, Any]],
    single_kernel_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stage_summaries: list[dict[str, Any]] = []
    for report in stage_reports:
        stage_summaries.extend(summarize_stage_report(report))
    lifecycle = {
        "requested_accelerator": args.accelerator,
        "coordinator_url": coordinator_url_for(args),
        "coordinator_direct_management": True,
        "actual_push_count": sum(
            1
            for run in stage_runs
            if any(step.get("name") == "kaggle_kernel_push" and step.get("ok") for step in run.get("steps", []))
        ),
        "kernels_deleted": all(
            any(step.get("name") == "kaggle_kernel_delete" and step.get("ok") for step in run.get("steps", []))
            for run in stage_runs
        ) if stage_runs else False,
        "private_packages_removed": not (output_dir / "private-kaggle-kernels").exists(),
        "private_activation_removed": not any(
            path.name == "ct_32b_stage0_activation_private.json"
            for path in (output_dir / "kaggle-output").rglob("*")
            if path.is_file()
        ),
    }
    generated = int(coordinator_status.get("generated_token_count") or 0)
    stage_task_counts = coordinator_status.get("stage_task_counts")
    if not isinstance(stage_task_counts, dict):
        stage_task_counts = {
            "stage0": int(coordinator_status.get("stage0_task_count") or 0),
            "stage1": int(coordinator_status.get("stage1_task_count") or 0),
        }
    latency_rows = list(coordinator_status.get("completed_tasks") or [])
    stage_latency: dict[str, dict[str, Any]] = {}
    for stage_id in range(int(args.stage_count)):
        values = [
            float(row.get("duration_seconds") or 0.0)
            for row in latency_rows
            if int_field(row.get("stage_id")) == stage_id and float(row.get("duration_seconds") or 0.0) > 0
        ]
        stage_latency[f"stage{stage_id}"] = {
            "count": len(values),
            "total_seconds": round(sum(values), 3),
            "avg_seconds": round(sum(values) / len(values), 3) if values else 0.0,
            "max_seconds": round(max(values), 3) if values else 0.0,
        }
    stage_summary_by_id = {
        int_field(summary.get("stage_id")): summary
        for summary in stage_summaries
        if 0 <= int_field(summary.get("stage_id")) < int(args.stage_count)
    }
    two_kernel_stage_memory = {}
    for stage_id in range(int(args.stage_count)):
        summary = stage_summary_by_id.get(stage_id, {})
        two_kernel_stage_memory[f"stage{stage_id}"] = {
            "loaded_tensor_gb": summary.get("loaded_tensor_gb"),
            "loaded_weight_key_count": summary.get("loaded_weight_key_count"),
            "cuda_memory_after_load": summary.get("cuda_memory_after_load") or {},
            "cuda_memory_after_execution": summary.get("cuda_memory_after_execution") or {},
        }
    single = single_kernel_report or {}
    single_metrics = single.get("metrics") if isinstance(single.get("metrics"), dict) else {}
    single_kernel_stage_memory = {}
    for stage_id in range(int(args.stage_count)):
        single_stage = single.get(f"stage{stage_id}") if isinstance(single.get(f"stage{stage_id}"), dict) else {}
        single_kernel_stage_memory[f"stage{stage_id}"] = {
            "loaded_tensor_gb": single_stage.get("loaded_tensor_gb"),
            "loaded_weight_key_count": single_stage.get("loaded_weight_key_count"),
            "cuda_memory_after_load": single_stage.get("cuda_memory_after_load") or {},
        }
    comparison = {
        "schema": "kaggle_32b_two_kernel_vs_single_t4x2_comparison_v1",
        "same_model": str(args.model_repo) == str((single.get("model") or {}).get("repo") or args.model_repo),
        "same_prompt_hash": coordinator_status.get("prompt_hash"),
        "same_context_policy": True,
        "two_kernel_generated_token_count": generated,
        "two_kernel_ready": generated >= int(args.max_new_tokens),
        "two_kernel_stage_latency": stage_latency,
        "two_kernel_stage_memory": two_kernel_stage_memory,
        "two_kernel_completed_task_count": int(coordinator_status.get("completed_task_count") or 0),
        "single_kernel_attempted": bool(single),
        "single_kernel_ok": single.get("ok") is True,
        "single_kernel_generated_token_count": int(single_metrics.get("generated_token_count") or 0),
        "single_kernel_wall_time_seconds": single_metrics.get("wall_time_seconds"),
        "single_kernel_tokens_per_second": single_metrics.get("tokens_per_second"),
        "single_kernel_stage_memory": single_kernel_stage_memory,
        "single_kernel_blockers": single.get("blockers") or [],
        "single_kernel_diagnosis_codes": single.get("diagnosis_codes") or [],
        "single_kernel_stability": "completed" if single.get("ok") is True else ("failed_or_killed" if single else "not_attempted"),
        "two_kernel_stability": "completed" if generated >= int(args.max_new_tokens) else "incomplete",
        "upper_bound_crossing_verified": bool(
            generated >= int(args.max_new_tokens)
            and single
            and single.get("ok") is not True
            and "single_kernel_t4x2_gpu_count_below_required_stage_count" in list(single.get("blockers") or [])
        ),
        "raw_outputs_public": False,
        "generated_token_ids_public": False,
        "activation_public": False,
        "public_artifact_safe": True,
    }
    ready = bool(
        generated >= int(args.max_new_tokens)
        and all(int(stage_task_counts.get(f"stage{stage_id}") or 0) >= int(args.max_new_tokens) for stage_id in range(int(args.stage_count)))
        and set(coordinator_status.get("stage_seen") or []) == set(range(int(args.stage_count)))
        and lifecycle["kernels_deleted"]
        and lifecycle["private_packages_removed"]
        and lifecycle["private_activation_removed"]
    )
    blockers: list[str] = []
    if generated < int(args.max_new_tokens):
        blockers.append("multi_token_generation_incomplete")
    if not all(int(stage_task_counts.get(f"stage{stage_id}") or 0) >= int(args.max_new_tokens) for stage_id in range(int(args.stage_count))):
        blockers.append("coordinator_stage_task_counts_incomplete")
    if not lifecycle["kernels_deleted"]:
        blockers.append("kaggle_kernels_cleanup_not_verified")
    if not lifecycle["private_activation_removed"]:
        blockers.append("private_activation_artifact_retained")
    if args.run_single_kernel_baseline and not single:
        blockers.append("single_kernel_baseline_missing")
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "execution_mode": "coordinator",
        "fresh_kaggle_run_performed": lifecycle["actual_push_count"] > 0,
        "coordinator_direct_management_verified": bool(ready and lifecycle["coordinator_direct_management"]),
        "upper_bound_crossing_verified": comparison.get("upper_bound_crossing_verified") is True,
        "cross_kernel_activation_decode_verified": bool(generated >= 1),
        "one_token_generation_verified": bool(generated >= 1),
        "multi_token_decode_verified": bool(generated >= int(args.max_new_tokens) and int(args.max_new_tokens) > 1),
        "generated_token_count": generated,
        "max_new_tokens": int(args.max_new_tokens),
        "stage_owned_awq_runtime_verified": all(summary.get("awq_stage_model_prepared") for summary in stage_summaries),
        "activation_handoff_verified": bool(coordinator_status.get("activation_hashes")),
        "blocked_reason": "" if ready else (blockers[0] if blockers else "kaggle_32b_coordinator_probe_not_ready"),
        "blockers": blockers,
        "model": {
            "repo": args.model_repo,
            "parameter_count_b": 32,
            "quantization": "awq",
            "split_index": int(args.split_index),
            "stage_count": int(args.stage_count),
        },
        "coordinator": coordinator_status,
        "stage_task_counts": stage_task_counts,
        "stage_summaries": stage_summaries,
        "kaggle_lifecycle": lifecycle,
        "stage_runs": [public_stage_run(run) for run in stage_runs],
        "single_kernel_baseline": (
            {**single, "steps": [public_step(step) for step in list(single.get("steps") or []) if isinstance(step, dict)]}
            if single
            else {}
        ),
        "comparison": comparison,
        "safety": {
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "credentials_public": False,
            "private_kernel_payload_public": False,
            "coordinator_private_state_public": False,
        },
        "limitations": [
            "This is a temporary proof Coordinator over HTTP for two private Kaggle GPU kernels, not the production CrowdTensor Coordinator data plane.",
            "Token ids and activations are private runtime state and excluded from retained public artifacts.",
            "KV cache reuse is not implemented; each token recomputes the full prompt prefix through both stages.",
        ],
    }


def run_live_probe(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_runs: list[dict[str, Any]] = []
    try:
        stage0_package = build_package(args, output_dir=output_dir, mode="stage0", stage_id=0)
        stage0_report, stage0_steps = run_kaggle_package(
            args,
            package=stage0_package,
            output_dir=output_dir,
            runner=runner,
            file_patterns=["ct_32b_activation_decode_stage0_report.json", "ct_32b_stage0_activation_private.json"],
        )
        stage_runs.append({
            "mode": "stage0",
            "stage_id": 0,
            "kernel_ref": stage0_package.get("kernel_ref"),
            "steps": stage0_steps,
        })
        activation_path = output_dir / "kaggle-output" / "stage0" / "ct_32b_stage0_activation_private.json"
        activation_payload = load_json(activation_path)
        if not activation_payload:
            stage1_report: dict[str, Any] = {}
        else:
            stage1_package = build_package(
                args,
                output_dir=output_dir,
                mode="stage1",
                stage_id=1,
                activation_payload=activation_payload,
            )
            stage1_report, stage1_steps = run_kaggle_package(
                args,
                package=stage1_package,
                output_dir=output_dir,
                runner=runner,
                file_patterns=["ct_32b_activation_decode_stage1_report.json"],
            )
            stage_runs.append({
                "mode": "stage1",
                "stage_id": 1,
                "kernel_ref": stage1_package.get("kernel_ref"),
                "steps": stage1_steps,
            })
        if not args.keep_private_activation:
            activation_path.unlink(missing_ok=True)
    finally:
        if not args.keep_private_package:
            shutil.rmtree(output_dir / "private-kaggle-kernels", ignore_errors=True)
    report = build_report(
        args,
        output_dir=output_dir,
        stage0_report=stage0_report if "stage0_report" in locals() else {},
        stage1_report=stage1_report if "stage1_report" in locals() else {},
        stage_runs=stage_runs,
    )
    write_json(output_dir / "kaggle_32b_stage_owned_activation_decode_probe.json", report)
    return report


def wait_for_coordinator_ready(
    state: StageCoordinatorState,
    *,
    timeout_seconds: float,
    worker_threads: list[threading.Thread] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    while time.monotonic() - started <= timeout_seconds:
        status = state.public_status()
        if status.get("ready"):
            return status
        if worker_threads and all(not thread.is_alive() for thread in worker_threads):
            return status
        time.sleep(5.0)
    return state.public_status()


def stage_groups_for(args: argparse.Namespace) -> list[tuple[str, list[int]]]:
    stage_count = int(args.stage_count)
    if stage_count == 2:
        return [("stage0", [0]), ("stage1", [1])]
    if stage_count == 4:
        return [("shard0", [0, 1]), ("shard1", [2, 3])]
    return [(f"stage{stage_id}", [stage_id]) for stage_id in range(stage_count)]


def run_single_kernel_baseline(args: argparse.Namespace, *, output_dir: Path, runner: Runner = subprocess.run) -> dict[str, Any]:
    if not args.run_single_kernel_baseline:
        return {}
    package = build_package(
        args,
        output_dir=output_dir,
        mode="single_baseline",
        stage_id=0,
        stage_ids=list(range(int(args.stage_count))),
        coordinator_token="",
        slug_prefix=args.single_kernel_slug_prefix,
    )
    package["report_filename"] = "ct_32b_activation_decode_single_baseline_report.json"
    try:
        report, steps = run_kaggle_package(
            args,
            package=package,
            output_dir=output_dir,
            runner=runner,
            file_patterns=["ct_32b_activation_decode_single_baseline_report.json"],
            cleanup=True,
        )
    finally:
        if not args.keep_private_package:
            shutil.rmtree(output_dir / "private-kaggle-kernels" / "single_baseline", ignore_errors=True)
            try:
                (output_dir / "private-kaggle-kernels").rmdir()
            except OSError:
                pass
    baseline = report.get("single_kernel_baseline") if isinstance(report.get("single_kernel_baseline"), dict) else {}
    if baseline or report:
        baseline = baseline or {
            "ok": report.get("ok") is True,
            "schema": "kaggle_32b_single_t4x2_stage_split_baseline_v1",
            "generated_token_count": report.get("generated_token_count", 0),
            "max_new_tokens": int(args.max_new_tokens),
            "stage_count": int(args.stage_count),
            "blockers": report.get("blockers") or ["single_kernel_baseline_not_ready"],
            "diagnosis_codes": report.get("diagnosis_codes") or ["single_kernel_baseline_not_ready"],
        }
        baseline = {
            **baseline,
            "kernel_ref": package.get("kernel_ref"),
            "steps": steps,
            "model": {
                "repo": args.model_repo,
                "parameter_count_b": 32,
                "quantization": "awq",
            },
            "metrics": {
                "generated_token_count": baseline.get("generated_token_count"),
                "wall_time_seconds": baseline.get("wall_time_seconds"),
                "tokens_per_second": baseline.get("tokens_per_second"),
            },
        }
    return baseline or {
        "ok": False,
        "schema": "kaggle_32b_single_t4x2_stage_split_baseline_v1",
        "kernel_ref": package.get("kernel_ref"),
        "steps": steps,
        "blockers": ["single_kernel_baseline_report_missing"],
        "diagnosis_codes": ["single_kernel_baseline_report_missing"],
    }


def run_coordinator_probe(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = __import__("secrets").token_urlsafe(32)
    state = StageCoordinatorState(prompt=args.prompt, max_new_tokens=args.max_new_tokens, stage_count=args.stage_count)
    server = ProbeCoordinatorServer(host="0.0.0.0", port=int(args.port), token=token, state=state)
    stage_runs: list[dict[str, Any]] = []
    stage_reports_by_mode: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []

    def run_stage(mode: str, stage_ids: list[int]) -> None:
        try:
            package = build_package(
                args,
                output_dir=output_dir,
                mode=mode,
                stage_id=int(stage_ids[0]),
                stage_ids=stage_ids,
                coordinator_token=token,
            )
            report, steps = run_kaggle_package(
                args,
                package=package,
                output_dir=output_dir,
                runner=runner,
                file_patterns=[f"ct_32b_activation_decode_{mode}_report.json"],
                cleanup=True,
            )
            stage_reports_by_mode[mode] = report
            stage_runs.append({
                "mode": mode,
                "stage_id": int(stage_ids[0]),
                "stage_ids": [int(value) for value in stage_ids],
                "kernel_ref": package.get("kernel_ref"),
                "steps": steps,
            })
        except Exception as exc:
            errors.append({
                "mode": mode,
                "stage_ids": [int(value) for value in stage_ids],
                "error_type": type(exc).__name__,
                "error_digest": sha_payload(str(exc)),
            })

    try:
        server.start()
        threads = [
            threading.Thread(target=run_stage, args=(mode, stage_ids), daemon=True)
            for mode, stage_ids in stage_groups_for(args)
        ]
        for thread in threads:
            thread.start()
        coordinator_status = wait_for_coordinator_ready(
            state,
            timeout_seconds=args.coordinator_timeout_seconds,
            worker_threads=threads,
        )
        for thread in threads:
            thread.join(timeout=max(1.0, float(args.kaggle_status_timeout_seconds) + 120.0))
    finally:
        server.stop()
        if not args.keep_private_package:
            shutil.rmtree(output_dir / "private-kaggle-kernels", ignore_errors=True)
    single_report = run_single_kernel_baseline(args, output_dir=output_dir, runner=runner)
    coordinator_status = state.public_status()
    if errors:
        coordinator_status["worker_errors"] = errors
    report = build_coordinator_report(
        args,
        output_dir=output_dir,
        coordinator_status=coordinator_status,
        stage_reports=[stage_reports_by_mode.get(mode, {}) for mode, _stage_ids in stage_groups_for(args)],
        stage_runs=stage_runs,
        single_kernel_report=single_report,
    )
    write_json(output_dir / "kaggle_32b_stage_owned_activation_decode_probe.json", report)
    write_json(output_dir / "kaggle_32b_multitoken_comparison_report.json", report)
    private_state_path = output_dir / "coordinator-private-state.json"
    if args.keep_coordinator_private_state:
        write_json(private_state_path, state.private_state())
    else:
        private_state_path.unlink(missing_ok=True)
    return report


def merge_single_baseline_into_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    args.run_single_kernel_baseline = True
    existing_path = Path(args.existing_coordinator_report)
    existing = load_json(existing_path)
    if not existing:
        raise SystemExit("--existing-coordinator-report must point to a completed coordinator report")
    single_report = run_single_kernel_baseline(args, output_dir=output_dir, runner=runner)
    if not args.keep_private_package:
        shutil.rmtree(output_dir / "private-kaggle-kernels", ignore_errors=True)
    coordinator_status = existing.get("coordinator") if isinstance(existing.get("coordinator"), dict) else {}
    stage_reports = []
    for mode, _stage_ids in stage_groups_for(args):
        stage_reports.append(load_json(output_dir / "kaggle-output" / mode / f"ct_32b_activation_decode_{mode}_report.json"))
    report = build_coordinator_report(
        args,
        output_dir=output_dir,
        coordinator_status=coordinator_status,
        stage_reports=stage_reports,
        stage_runs=list(existing.get("stage_runs") or []),
        single_kernel_report=single_report,
    )
    report["source_coordinator_report"] = str(existing_path)
    write_json(output_dir / "kaggle_32b_stage_owned_activation_decode_probe.json", report)
    write_json(output_dir / "kaggle_32b_multitoken_comparison_report.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a private Kaggle two-kernel 32B AWQ activation/decode probe.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kaggle-owner", default=loading_probe.default_kaggle_owner())
    parser.add_argument("--kernel-slug-prefix", default="ct32bactdecode")
    parser.add_argument("--accelerator", default=DEFAULT_ACCELERATOR)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--execution-mode", choices=["local-handoff", "coordinator"], default="local-handoff")
    parser.add_argument("--single-baseline-only", action="store_true")
    parser.add_argument("--existing-coordinator-report", default="")
    parser.add_argument("--public-host", default=DEFAULT_PUBLIC_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--stage-count", type=int, default=2)
    parser.add_argument("--split-index", type=int, default=32)
    parser.add_argument("--prompt", default="Hi")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--task-poll-interval-seconds", type=float, default=15.0)
    parser.add_argument("--task-idle-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--coordinator-timeout-seconds", type=float, default=3900.0)
    parser.add_argument("--kernel-timeout-seconds", type=int, default=3600)
    parser.add_argument("--kaggle-push-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--kaggle-status-timeout-seconds", type=float, default=3900.0)
    parser.add_argument("--kaggle-status-poll-interval", type=float, default=60.0)
    parser.add_argument("--kaggle-output-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--kaggle-delete-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--run-single-kernel-baseline", action="store_true")
    parser.add_argument("--single-kernel-slug-prefix", default="ct32bsingle")
    parser.add_argument(
        "--single-baseline-placement",
        choices=["two_stage_two_gpu", "strict_stage_count"],
        default="two_stage_two_gpu",
    )
    parser.add_argument("--skip-kaggle-cleanup", action="store_true")
    parser.add_argument("--keep-private-package", action="store_true")
    parser.add_argument("--keep-private-activation", action="store_true")
    parser.add_argument("--keep-coordinator-private-state", action="store_true")
    parser.add_argument("--keep-kaggle-logs", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.stage_count not in {2, 4}:
        raise SystemExit("--stage-count currently must be 2 or 4 for activation/decode")
    if args.split_index < 1 or args.split_index > 63:
        raise SystemExit("--split-index must be in 1..63")
    if args.max_new_tokens < 1 or args.max_new_tokens > 8:
        raise SystemExit("--max-new-tokens must be in 1..8")
    if args.execution_mode == "local-handoff" and args.max_new_tokens != 1 and not args.single_baseline_only:
        raise SystemExit("--execution-mode local-handoff only supports --max-new-tokens 1")
    if args.kernel_timeout_seconds > 3600:
        raise SystemExit("--kernel-timeout-seconds must be <= 3600")
    if args.kaggle_status_timeout_seconds > 4200:
        raise SystemExit("--kaggle-status-timeout-seconds must be <= 4200")
    if args.coordinator_timeout_seconds > 4200:
        raise SystemExit("--coordinator-timeout-seconds must be <= 4200")
    args.coordinator_url = str(args.coordinator_url or "").rstrip("/")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.single_baseline_only:
        report = merge_single_baseline_into_report(args)
    else:
        report = run_coordinator_probe(args) if args.execution_mode == "coordinator" else run_live_probe(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            f"{SCHEMA}: ok={bool(report.get('ok'))} "
            f"tokens={int(report.get('generated_token_count') or (1 if report.get('one_token_generation_verified') else 0))}/"
            f"{int(report.get('max_new_tokens') or args.max_new_tokens)} "
            f"coordinator={bool(report.get('coordinator_direct_management_verified'))} "
            f"blocked={report.get('blocked_reason') or 'none'}"
        )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
