#!/usr/bin/env python3
"""Run private Kaggle probes for stage-owned 32B quantized safetensors loading."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCHEMA = "kaggle_32b_stage_owned_safetensors_probe_v1"
STAGE_REPORT_SCHEMA = "kaggle_32b_stage_owned_safetensors_stage_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/kaggle-32b-stage-owned-safetensors-probe"
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"
DEFAULT_MODEL_REPO = "Qwen/Qwen2.5-32B-Instruct-AWQ"
DEFAULT_STAGE_COUNT = 2
DEFAULT_STAGE_IDS = "0,1"
KAGGLE_TABLE_SPLIT = re.compile(r"\s{2,}")
KAGGLE_CODE_URL = re.compile(r"https://www\.kaggle\.com/code/([^/\s]+)/([^/\s]+)")
Runner = Callable[..., subprocess.CompletedProcess[str]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def default_kaggle_owner() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return str(os.environ["KAGGLE_USERNAME"])
    config = Path.home() / ".kaggle" / "kaggle.json"
    if config.is_file():
        try:
            loaded = json.loads(config.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return ""
        if isinstance(loaded, dict):
            return str(loaded.get("username") or "")
    return ""


def safe_slug(value: str, *, default: str = "ct32bstage") -> str:
    cleaned: list[str] = []
    last_dash = False
    for char in str(value or "").lower():
        if char.isalnum():
            cleaned.append(char)
            last_dash = False
        elif not last_dash:
            cleaned.append("-")
            last_dash = True
    return "".join(cleaned).strip("-") or default


def shell_command(command: list[Any]) -> str:
    return shlex.join([str(part) for part in command])


def safe_tail(text: str, limit: int = 1600) -> str:
    redacted = str(text or "")[-limit:]
    for fragment in [
        "KAGGLE_KEY",
        "KAGGLE_USERNAME",
        "Bearer ",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "CROWDTENSOR_MINER_TOKEN=",
        "CROWDTENSOR_OBSERVER_TOKEN=",
        "CROWDTENSOR_ADMIN_TOKEN=",
    ]:
        redacted = redacted.replace(fragment, "<redacted>")
    return redacted


def run_step(
    name: str,
    command: list[str],
    *,
    runner: Runner,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = runner(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "ok": False,
            "error": "timeout",
            "duration_seconds": round(time.monotonic() - started, 3),
            "command_line": shell_command(command),
        }
    output = f"{completed.stdout or ''}\n{completed.stderr or ''}"
    return {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout_tail": safe_tail(completed.stdout or ""),
        "stderr_tail": safe_tail(completed.stderr or ""),
        "command_line": shell_command(command),
        "actual_kernel_ref": extract_kernel_ref(output),
    }


def extract_kernel_ref(text: str) -> str:
    match = KAGGLE_CODE_URL.search(text or "")
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return ""


def extract_status(text: str) -> str:
    upper = str(text or "").upper()
    if "CANCEL_ACKNOWLEDGED" in upper:
        return "CANCEL_ACKNOWLEDGED"
    for status in ["COMPLETE", "RUNNING", "QUEUED", "PENDING", "ERROR", "FAILED", "CANCELLED", "CANCELED"]:
        if status in upper:
            return status
    return "UNKNOWN"


def extract_kernel_list_ref(text: str, *, owner: str, title: str) -> str:
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("Warning:") or line.startswith("ref "):
            continue
        if set(line) <= {"-", " "}:
            continue
        parts = KAGGLE_TABLE_SPLIT.split(line)
        if len(parts) < 2:
            continue
        ref, row_title = parts[0], parts[1]
        if "/" not in ref:
            continue
        if owner and not ref.startswith(f"{owner}/"):
            continue
        if row_title == title:
            return ref
    return ""


def wait_kaggle_terminal(
    kernel_ref: str,
    *,
    runner: Runner,
    timeout_seconds: float,
    poll_interval: float,
) -> dict[str, Any]:
    started = time.monotonic()
    attempts = 0
    last_step: dict[str, Any] = {}
    while time.monotonic() - started <= timeout_seconds:
        attempts += 1
        last_step = run_step(
            "kaggle_kernel_status",
            ["kaggle", "kernels", "status", kernel_ref],
            runner=runner,
            timeout_seconds=60,
        )
        status = extract_status(f"{last_step.get('stdout_tail') or ''}\n{last_step.get('stderr_tail') or ''}")
        print(f"[{utc_now()}] status kernel={kernel_ref} attempt={attempts} status={status}", flush=True)
        if status in {"COMPLETE", "ERROR", "FAILED", "CANCELLED", "CANCELED", "CANCEL_ACKNOWLEDGED"}:
            last_step.update({
                "duration_seconds": round(time.monotonic() - started, 3),
                "attempts": attempts,
                "status": status,
                "terminal": True,
                "kernel_ref": kernel_ref,
                "ok": bool(last_step.get("ok") and status == "COMPLETE"),
            })
            return last_step
        time.sleep(max(5.0, float(poll_interval)))
    last_step.update({
        "duration_seconds": round(time.monotonic() - started, 3),
        "attempts": attempts,
        "status": extract_status(f"{last_step.get('stdout_tail') or ''}\n{last_step.get('stderr_tail') or ''}"),
        "terminal": False,
        "kernel_ref": kernel_ref,
        "ok": False,
        "error": "timeout_waiting_for_terminal_kernel_status",
    })
    return last_step


def fetch_hf_json(model_repo: str, filename: str, *, timeout_seconds: float = 60.0) -> dict[str, Any]:
    url = f"https://huggingface.co/{model_repo}/resolve/main/{filename}"
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        loaded = json.load(response)
    return loaded if isinstance(loaded, dict) else {}


def normalize_stage_count(stage_count: int, *, layer_count: int = 0) -> int:
    count = max(2, min(int(stage_count), 16))
    if layer_count > 0:
        count = min(count, max(2, int(layer_count)))
    return count


def stage_layer_ranges(layer_count: int, stage_count: int) -> list[tuple[int, int]]:
    layers = max(0, int(layer_count))
    count = normalize_stage_count(stage_count, layer_count=layers)
    if layers <= 0:
        return [(0, 0) for _ in range(count)]
    base = layers // count
    remainder = layers % count
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for index in range(count):
        width = base + (1 if index < remainder else 0)
        start = cursor
        end = min(layers, start + width)
        if end <= start and layers > 0:
            end = min(layers, start + 1)
        ranges.append((start, end))
        cursor = end
    return ranges


def stage_prefixes(*, stage_id: int, stage_count: int, layer_range: tuple[int, int]) -> list[str]:
    start, end = int(layer_range[0]), int(layer_range[1])
    prefixes = [f"model.layers.{index}." for index in range(start, end)]
    if int(stage_id) == 0:
        prefixes = ["model.embed_tokens.", *prefixes]
    if int(stage_id) == int(stage_count) - 1:
        prefixes = [*prefixes, "model.norm.", "lm_head."]
    return prefixes


def build_stage_selection(
    *,
    config: dict[str, Any],
    weight_index: dict[str, Any],
    stage_id: int,
    stage_count: int,
) -> dict[str, Any]:
    weight_map = {
        str(key): Path(str(value)).name
        for key, value in dict(weight_index.get("weight_map") or {}).items()
        if str(key or "").strip() and str(value or "").strip()
    }
    layer_count = int(config.get("num_hidden_layers") or config.get("n_layer") or 0)
    count = normalize_stage_count(stage_count, layer_count=layer_count)
    stage = int(stage_id)
    if stage < 0 or stage >= count:
        raise ValueError(f"stage_id must be between 0 and {count - 1}")
    ranges = stage_layer_ranges(layer_count, count)
    layer_range = ranges[stage]
    prefixes = stage_prefixes(stage_id=stage, stage_count=count, layer_range=layer_range)
    assigned = sorted(key for key in weight_map if any(key.startswith(prefix) for prefix in prefixes))
    assigned_files = sorted({weight_map[key] for key in assigned if weight_map.get(key)})
    all_files = sorted(set(weight_map.values()))
    shared_boundary_files = sorted(
        filename
        for filename in assigned_files
        if any(
            other_key not in set(assigned) and weight_map.get(other_key) == filename
            for other_key in weight_map
        )
    )
    return {
        "model_type": str(config.get("model_type") or ""),
        "architectures": list(config.get("architectures") or []),
        "num_hidden_layers": layer_count,
        "hidden_size": int(config.get("hidden_size") or config.get("n_embd") or 0),
        "stage_id": stage,
        "stage_count": count,
        "stage_layer_range": [int(layer_range[0]), int(layer_range[1])],
        "expected_key_prefixes": prefixes,
        "assigned_weight_keys": assigned,
        "assigned_weight_key_count": len(assigned),
        "assigned_weight_files": assigned_files,
        "assigned_weight_file_count": len(assigned_files),
        "all_weight_file_count": len(all_files),
        "all_weight_files": all_files,
        "shared_boundary_files": shared_boundary_files,
        "shared_boundary_file_count": len(shared_boundary_files),
        "weight_key_count": len(weight_map),
        "weight_map": weight_map,
        "total_size_bytes": int(dict(weight_index.get("metadata") or {}).get("total_size") or 0),
    }


def render_kernel(args: argparse.Namespace, *, stage_id: int) -> str:
    model_repo_json = json.dumps(args.model_repo)
    return f'''from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = {json.dumps(STAGE_REPORT_SCHEMA)}
MODEL_REPO = {model_repo_json}
STAGE_ID = {int(stage_id)}
STAGE_COUNT = {int(args.stage_count)}
RETAIN_TENSORS = {repr(bool(args.retain_tensors))}
MATERIALIZE_CLONE = {repr(bool(args.materialize_clone))}
RETAIN_LIMIT_BYTES = {int(args.retain_limit_gb) * 1024 * 1024 * 1024}
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
OUT = Path("/kaggle/working")
TEMP = Path("/kaggle/temp/ct_32b_stage_owned_safetensors") / f"stage{{STAGE_ID}}"
MODEL_DIR = TEMP / "model"
REPORT_PATH = OUT / f"ct_32b_stage_owned_safetensors_stage{{STAGE_ID}}_report.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha_text(text):
    return "sha256:" + hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def write_report(payload):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")


def run_command(command, timeout=120):
    started = time.monotonic()
    try:
        completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {{"ok": False, "error": "timeout", "duration_seconds": round(time.monotonic() - started, 3)}}
    return {{
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout_tail": (completed.stdout or "")[-1200:],
        "stderr_tail": (completed.stderr or "")[-1200:],
    }}


def fetch_json(filename):
    url = f"https://huggingface.co/{{MODEL_REPO}}/resolve/main/{{filename}}"
    with urllib.request.urlopen(url, timeout=120) as response:
        loaded = json.load(response)
    return loaded if isinstance(loaded, dict) else {{}}


def normalize_stage_count(stage_count, layer_count=0):
    count = max(2, min(int(stage_count), 16))
    if layer_count > 0:
        count = min(count, max(2, int(layer_count)))
    return count


def stage_layer_ranges(layer_count, stage_count):
    layers = max(0, int(layer_count))
    count = normalize_stage_count(stage_count, layer_count=layers)
    if layers <= 0:
        return [(0, 0) for _ in range(count)]
    base = layers // count
    remainder = layers % count
    ranges = []
    cursor = 0
    for index in range(count):
        width = base + (1 if index < remainder else 0)
        start = cursor
        end = min(layers, start + width)
        if end <= start and layers > 0:
            end = min(layers, start + 1)
        ranges.append((start, end))
        cursor = end
    return ranges


def stage_prefixes(stage_id, stage_count, layer_range):
    start, end = int(layer_range[0]), int(layer_range[1])
    prefixes = [f"model.layers.{{index}}." for index in range(start, end)]
    if int(stage_id) == 0:
        prefixes = ["model.embed_tokens.", *prefixes]
    if int(stage_id) == int(stage_count) - 1:
        prefixes = [*prefixes, "model.norm.", "lm_head."]
    return prefixes


def build_selection(config, weight_index):
    weight_map = {{
        str(key): Path(str(value)).name
        for key, value in dict(weight_index.get("weight_map") or {{}}).items()
        if str(key or "").strip() and str(value or "").strip()
    }}
    layer_count = int(config.get("num_hidden_layers") or config.get("n_layer") or 0)
    count = normalize_stage_count(STAGE_COUNT, layer_count=layer_count)
    ranges = stage_layer_ranges(layer_count, count)
    if STAGE_ID < 0 or STAGE_ID >= count:
        raise ValueError(f"stage_id must be between 0 and {{count - 1}}")
    layer_range = ranges[STAGE_ID]
    prefixes = stage_prefixes(STAGE_ID, count, layer_range)
    assigned = sorted(key for key in weight_map if any(key.startswith(prefix) for prefix in prefixes))
    assigned_files = sorted({{weight_map[key] for key in assigned if weight_map.get(key)}})
    all_files = sorted(set(weight_map.values()))
    assigned_set = set(assigned)
    shared = sorted(
        filename
        for filename in assigned_files
        if any(other_key not in assigned_set and weight_map.get(other_key) == filename for other_key in weight_map)
    )
    return {{
        "model_type": str(config.get("model_type") or ""),
        "architectures": list(config.get("architectures") or []),
        "num_hidden_layers": layer_count,
        "hidden_size": int(config.get("hidden_size") or config.get("n_embd") or 0),
        "stage_count": count,
        "stage_layer_range": [int(layer_range[0]), int(layer_range[1])],
        "expected_key_prefixes": prefixes,
        "assigned_weight_keys": assigned,
        "assigned_weight_key_count": len(assigned),
        "assigned_weight_files": assigned_files,
        "assigned_weight_file_count": len(assigned_files),
        "all_weight_file_count": len(all_files),
        "weight_key_count": len(weight_map),
        "weight_map": weight_map,
        "shared_boundary_files": shared,
        "shared_boundary_file_count": len(shared),
        "total_size_bytes": int(dict(weight_index.get("metadata") or {{}}).get("total_size") or 0),
    }}


def install_safetensors_if_needed():
    try:
        import safetensors.torch  # noqa: F401
        return {{"ok": True, "already_available": True}}
    except ModuleNotFoundError:
        step = run_command([sys.executable, "-m", "pip", "install", "-q", "safetensors"], timeout=300)
        try:
            import safetensors.torch  # noqa: F401
            step["import_after_install"] = True
        except ModuleNotFoundError:
            step["import_after_install"] = False
            step["ok"] = False
        return step


def hardware_summary():
    smi = run_command(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], timeout=60)
    names = []
    memory_mb = []
    if smi.get("ok"):
        for raw in str(smi.get("stdout_tail") or "").splitlines():
            parts = [part.strip() for part in raw.split(",")]
            if not parts or not parts[0]:
                continue
            names.append(parts[0])
            if len(parts) > 1:
                digits = "".join(ch for ch in parts[1] if ch.isdigit())
                memory_mb.append(int(digits or 0))
    return {{
        "nvidia_smi_ok": bool(smi.get("ok")),
        "gpu_count": len(names),
        "gpu_names": names,
        "vram_total_mb": memory_mb,
        "kaggle_gpu_verified": bool(names),
        "nvidia_smi": smi,
    }}


def memory_summary():
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except Exception:
        return {{}}
    fields = {{}}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        digits = "".join(ch for ch in value if ch.isdigit())
        if digits:
            fields[key] = int(digits) // 1024
    return {{
        "mem_total_mb": fields.get("MemTotal", 0),
        "mem_available_mb": fields.get("MemAvailable", 0),
    }}


def download_file(filename):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = MODEL_DIR / Path(filename).name
    url = f"https://huggingface.co/{{MODEL_REPO}}/resolve/main/{{filename}}"
    started = time.monotonic()
    size = 0
    with urllib.request.urlopen(url, timeout=120) as response:
        with target.open("wb") as handle:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                handle.write(chunk)
                size += len(chunk)
    return {{
        "filename": Path(filename).name,
        "size_bytes": int(size),
        "size_mb": round(size / 1024 / 1024, 3),
        "duration_seconds": round(time.monotonic() - started, 3),
    }}


def tensor_nbytes(tensor):
    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except Exception:
        return 0


def main():
    started = time.monotonic()
    report = {{
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": False,
        "stage_owned_quantized_32b_loading_ready": False,
        "model_repo": MODEL_REPO,
        "quantization_format": "awq_safetensors",
        "stage_id": STAGE_ID,
        "stage_count": STAGE_COUNT,
        "read_only": True,
        "materialize_clone_requested": MATERIALIZE_CLONE,
        "public_safe": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "activation_public": False,
        "credentials_public": False,
        "local_weight_root_public": False,
        "diagnosis_codes": [],
        "blockers": [],
    }}
    try:
        if TEMP.exists():
            shutil.rmtree(TEMP)
        TEMP.mkdir(parents=True, exist_ok=True)
        report["hardware"] = hardware_summary()
        report["memory_before"] = memory_summary()
        install_step = install_safetensors_if_needed()
        report["dependency_setup"] = install_step
        if not install_step.get("ok"):
            report["blockers"].append("safetensors_dependency_missing")
            report["diagnosis_codes"].append("kaggle_32b_stage_owned_dependency_missing")
            write_report(report)
            return
        config = fetch_json("config.json")
        weight_index = fetch_json("model.safetensors.index.json")
        selection = build_selection(config, weight_index)
        report.update({{
            "model_type": selection["model_type"],
            "architectures": selection["architectures"],
            "num_hidden_layers": selection["num_hidden_layers"],
            "hidden_size": selection["hidden_size"],
            "stage_count": selection["stage_count"],
            "stage_layer_range": selection["stage_layer_range"],
            "assigned_weight_key_count": selection["assigned_weight_key_count"],
            "assigned_weight_file_count": selection["assigned_weight_file_count"],
            "assigned_weight_files": selection["assigned_weight_files"],
            "all_weight_file_count": selection["all_weight_file_count"],
            "weight_key_count": selection["weight_key_count"],
            "shared_boundary_file_count": selection["shared_boundary_file_count"],
            "shared_boundary_files": selection["shared_boundary_files"],
            "total_model_size_bytes": selection["total_size_bytes"],
            "stage_weight_download_scope": "stage_owned_weight_files",
            "stage_weight_download_stage_id": STAGE_ID,
            "stage_weight_download_file_count": selection["assigned_weight_file_count"],
            "stage_weight_downloads_only_stage_files": selection["assigned_weight_file_count"] < selection["all_weight_file_count"],
            "downloads_all_model_weight_files": selection["assigned_weight_file_count"] == selection["all_weight_file_count"],
        }})
        if not selection["assigned_weight_keys"]:
            report["blockers"].append("stage_weight_selection_empty")
            report["diagnosis_codes"].append("kaggle_32b_stage_owned_selection_empty")
            write_report(report)
            return
        downloads = []
        for filename in selection["assigned_weight_files"]:
            downloads.append(download_file(filename))
            report["downloads"] = downloads
            write_report(report)
        from safetensors.torch import safe_open  # type: ignore

        assigned_keys = list(selection["assigned_weight_keys"])
        assigned_set = set(assigned_keys)
        weight_map = dict(selection["weight_map"])
        loaded_keys = []
        loaded_files = set()
        opened_files = set()
        missing_files = []
        missing_key_count_by_file = {{}}
        candidate_file_key_count = 0
        skipped_non_stage_key_count = 0
        loaded_tensor_bytes = 0
        retained_tensor_bytes = 0
        materialized_tensor_bytes = 0
        materialized_weight_key_count = 0
        retained_tensors = {{}}
        for filename in selection["assigned_weight_files"]:
            safe_filename = Path(filename).name
            path = MODEL_DIR / safe_filename
            if not path.is_file():
                missing_files.append(safe_filename)
                missing_key_count_by_file[safe_filename] = len([key for key in assigned_keys if weight_map.get(key) == safe_filename])
                continue
            opened_files.add(safe_filename)
            with safe_open(path, framework="pt", device="cpu") as handle:
                available = set(str(key) for key in handle.keys())
                candidate_file_key_count += len(available)
                skipped_non_stage_key_count += len(available - assigned_set)
                expected_in_file = [key for key in assigned_keys if weight_map.get(key) == safe_filename]
                missing = [key for key in expected_in_file if key not in available]
                if missing:
                    missing_key_count_by_file[safe_filename] = len(missing)
                for key in expected_in_file:
                    if key not in available:
                        continue
                    source_tensor = handle.get_tensor(key)
                    nbytes = tensor_nbytes(source_tensor)
                    loaded_tensor_bytes += nbytes
                    loaded_keys.append(key)
                    loaded_files.add(safe_filename)
                    if RETAIN_TENSORS and retained_tensor_bytes + nbytes <= RETAIN_LIMIT_BYTES:
                        tensor = source_tensor.clone() if MATERIALIZE_CLONE else source_tensor
                        if MATERIALIZE_CLONE:
                            materialized_tensor_bytes += tensor_nbytes(tensor)
                            materialized_weight_key_count += 1
                        retained_tensors[key] = tensor
                        retained_tensor_bytes += nbytes
                    else:
                        if MATERIALIZE_CLONE:
                            materialized = source_tensor.clone()
                            materialized_tensor_bytes += tensor_nbytes(materialized)
                            materialized_weight_key_count += 1
                            del materialized
                        del source_tensor
            gc.collect()
        loaded_key_set = set(loaded_keys)
        unexpected_loaded = sorted(loaded_key_set - assigned_set)
        missing_key_count = sum(int(value) for value in missing_key_count_by_file.values())
        ready = bool(
            loaded_keys
            and not missing_files
            and missing_key_count == 0
            and not unexpected_loaded
            and loaded_key_set.issubset(assigned_set)
            and len(loaded_key_set) == len(assigned_set)
            and (
                not MATERIALIZE_CLONE
                or (
                    materialized_weight_key_count == len(assigned_set)
                    and materialized_tensor_bytes == loaded_tensor_bytes
                )
            )
        )
        report.update({{
            "opened_weight_file_count": len(opened_files),
            "loaded_weight_file_count": len(loaded_files),
            "loaded_weight_key_count": len(loaded_key_set),
            "loaded_tensor_bytes": int(loaded_tensor_bytes),
            "loaded_tensor_gb": round(loaded_tensor_bytes / 1024 / 1024 / 1024, 6),
            "materialized_tensor_bytes": int(materialized_tensor_bytes),
            "materialized_tensor_gb": round(materialized_tensor_bytes / 1024 / 1024 / 1024, 6),
            "materialized_weight_key_count": int(materialized_weight_key_count),
            "materialize_clone_requested": MATERIALIZE_CLONE,
            "retained_tensor_bytes": int(retained_tensor_bytes),
            "retained_tensor_gb": round(retained_tensor_bytes / 1024 / 1024 / 1024, 6),
            "retained_weight_key_count": len(retained_tensors),
            "retain_tensors_requested": RETAIN_TENSORS,
            "retain_limit_bytes": int(RETAIN_LIMIT_BYTES),
            "candidate_file_key_count": int(candidate_file_key_count),
            "skipped_non_stage_weight_key_count": int(skipped_non_stage_key_count),
            "missing_weight_file_count": len(missing_files),
            "missing_weight_key_count": int(missing_key_count),
            "missing_weight_files": sorted(set(missing_files))[:8],
            "missing_weight_key_count_by_file": {{key: int(value) for key, value in sorted(missing_key_count_by_file.items())}},
            "loaded_weight_key_digest": sha_text(json.dumps(sorted(loaded_keys))),
            "loaded_weight_file_digest": sha_text(json.dumps(sorted(loaded_files))),
            "loads_only_stage_weight_keys": bool(ready and not unexpected_loaded),
            "cross_stage_weight_keys_loaded": bool(unexpected_loaded),
            "memory_after_load": memory_summary(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }})
        if ready:
            report["ok"] = True
            report["stage_owned_quantized_32b_loading_ready"] = True
            report["diagnosis_codes"].extend([
                "kaggle_32b_gpu_hardware_verified" if report.get("hardware", {{}}).get("kaggle_gpu_verified") else "kaggle_32b_gpu_hardware_not_verified",
                "kaggle_32b_stage_owned_weight_download_ready",
                "kaggle_32b_stage_owned_tensor_load_ready",
                "kaggle_32b_loads_only_stage_weight_keys",
                "kaggle_32b_stage_owned_tensor_clone_ready" if MATERIALIZE_CLONE else "kaggle_32b_stage_owned_tensor_mmap_ready",
            ])
        else:
            report["diagnosis_codes"].append("kaggle_32b_stage_owned_tensor_load_not_ready")
            if missing_files:
                report["blockers"].append("stage_owned_weight_files_missing")
            if missing_key_count:
                report["blockers"].append("stage_owned_weight_keys_missing")
            if unexpected_loaded:
                report["blockers"].append("cross_stage_weight_keys_loaded")
            if not loaded_keys:
                report["blockers"].append("stage_owned_weight_keys_not_loaded")
            if MATERIALIZE_CLONE and materialized_weight_key_count != len(assigned_set):
                report["blockers"].append("stage_owned_weight_keys_not_materialized")
    except Exception as exc:
        report["ok"] = False
        report["stage_owned_quantized_32b_loading_ready"] = False
        report["error_type"] = type(exc).__name__
        report["error_digest"] = sha_text(str(exc))
        report["diagnosis_codes"].append("kaggle_32b_stage_owned_probe_exception")
        report["blockers"].append("kaggle_32b_stage_owned_probe_exception")
    finally:
        try:
            if TEMP.exists():
                shutil.rmtree(TEMP)
            report["temp_cleanup"] = {{"ok": True, "path_public": False}}
        except Exception as exc:
            report["temp_cleanup"] = {{"ok": False, "error_type": type(exc).__name__, "error_digest": sha_text(str(exc))}}
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_report(report)
        print(json.dumps({{"schema": SCHEMA, "ok": report.get("ok"), "stage_id": STAGE_ID, "diagnosis_codes": report.get("diagnosis_codes")}}, sort_keys=True))


main()
'''


def build_package(args: argparse.Namespace, *, output_dir: Path, stage_id: int) -> dict[str, Any]:
    owner = args.kaggle_owner or default_kaggle_owner()
    if not owner:
        raise SystemExit("--kaggle-owner or ~/.kaggle/kaggle.json username is required")
    suffix = str(int(time.time()))[-8:]
    slug = f"{safe_slug(args.kernel_slug_prefix)[:28]}-s{int(stage_id)}-{suffix}"
    slug = slug[:45].strip("-")
    kernel_dir = output_dir / "private-kaggle-kernels" / f"stage{int(stage_id)}"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text(render_kernel(args, stage_id=stage_id), encoding="utf-8")
    title = f"CT 32B Stage Owned S{int(stage_id)} {suffix}"
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
        "stage_id": int(stage_id),
        "kernel_dir": kernel_dir,
        "declared_kernel_ref": metadata["id"],
        "kernel_ref": metadata["id"],
        "kernel_slug": slug,
        "metadata": metadata,
        "report_filename": f"ct_32b_stage_owned_safetensors_stage{int(stage_id)}_report.json",
    }


def resolve_pushed_kernel_ref(
    package: dict[str, Any],
    push_step: dict[str, Any],
    *,
    runner: Runner,
    timeout_seconds: float,
) -> tuple[str, dict[str, Any] | None]:
    actual = str(push_step.get("actual_kernel_ref") or "")
    if actual:
        return actual, None
    metadata = package.get("metadata") if isinstance(package.get("metadata"), dict) else {}
    declared = str(package.get("declared_kernel_ref") or "")
    owner = declared.split("/", 1)[0]
    title = str(metadata.get("title") or "")
    step = run_step(
        "kaggle_kernel_ref_resolve",
        ["kaggle", "kernels", "list", "--user", owner, "--sort-by", "dateRun", "--page-size", "20"],
        runner=runner,
        timeout_seconds=timeout_seconds,
    )
    output = f"{step.get('stdout_tail') or ''}\n{step.get('stderr_tail') or ''}"
    resolved = extract_kernel_list_ref(output, owner=owner, title=title)
    step["resolved_kernel_ref"] = resolved
    step["resolved"] = bool(resolved)
    return (resolved or declared), step


def parse_stage_ids(text: str, *, stage_count: int) -> list[int]:
    ids: list[int] = []
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        stage = int(item)
        if stage < 0 or stage >= int(stage_count):
            raise SystemExit(f"stage id {stage} is outside 0..{int(stage_count) - 1}")
        ids.append(stage)
    if not ids:
        raise SystemExit("--stage-ids must include at least one stage")
    return sorted(dict.fromkeys(ids))


def summarize_stage_report(report: dict[str, Any]) -> dict[str, Any]:
    hardware = report.get("hardware") if isinstance(report.get("hardware"), dict) else {}
    return {
        "stage_id": report.get("stage_id"),
        "stage_ok": report.get("ok") is True,
        "stage_owned_quantized_32b_loading_ready": report.get("stage_owned_quantized_32b_loading_ready") is True,
        "gpu_verified": hardware.get("kaggle_gpu_verified") is True,
        "gpu_count": hardware.get("gpu_count"),
        "gpu_names": hardware.get("gpu_names") or [],
        "stage_layer_range": report.get("stage_layer_range") or [],
        "assigned_weight_key_count": report.get("assigned_weight_key_count"),
        "assigned_weight_file_count": report.get("assigned_weight_file_count"),
        "assigned_weight_files": report.get("assigned_weight_files") or [],
        "downloaded_file_count": len(report.get("downloads") or []) if isinstance(report.get("downloads"), list) else 0,
        "loaded_weight_key_count": report.get("loaded_weight_key_count"),
        "loaded_tensor_gb": report.get("loaded_tensor_gb"),
        "materialized_tensor_gb": report.get("materialized_tensor_gb"),
        "materialized_weight_key_count": report.get("materialized_weight_key_count"),
        "materialize_clone_requested": report.get("materialize_clone_requested") is True,
        "retained_tensor_gb": report.get("retained_tensor_gb"),
        "loads_only_stage_weight_keys": report.get("loads_only_stage_weight_keys") is True,
        "cross_stage_weight_keys_loaded": report.get("cross_stage_weight_keys_loaded") is True,
        "downloads_all_model_weight_files": report.get("downloads_all_model_weight_files") is True,
        "stage_weight_downloads_only_stage_files": report.get("stage_weight_downloads_only_stage_files") is True,
        "shared_boundary_file_count": report.get("shared_boundary_file_count"),
        "diagnosis_codes": report.get("diagnosis_codes") or [],
        "blockers": report.get("blockers") or [],
    }


def local_expected_plan(args: argparse.Namespace, *, stage_ids: list[int]) -> dict[str, Any]:
    try:
        config = fetch_hf_json(args.model_repo, "config.json")
        weight_index = fetch_hf_json(args.model_repo, "model.safetensors.index.json")
    except Exception as exc:
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "error_digest": "sha256:" + str(abs(hash(str(exc)))),
        }
    stage_plans = [
        build_stage_selection(
            config=config,
            weight_index=weight_index,
            stage_id=stage_id,
            stage_count=args.stage_count,
        )
        for stage_id in stage_ids
    ]
    all_keys = set(dict(weight_index.get("weight_map") or {}))
    covered_keys: set[str] = set()
    for plan in stage_plans:
        covered_keys.update(str(key) for key in plan.get("assigned_weight_keys") or [])
    ranges = [list(plan.get("stage_layer_range") or []) for plan in stage_plans]
    return {
        "available": True,
        "model_repo": args.model_repo,
        "model_type": str(config.get("model_type") or ""),
        "architectures": list(config.get("architectures") or []),
        "num_hidden_layers": int(config.get("num_hidden_layers") or 0),
        "hidden_size": int(config.get("hidden_size") or 0),
        "stage_count": int(args.stage_count),
        "requested_stage_ids": stage_ids,
        "requested_stage_ranges": ranges,
        "weight_key_count": len(all_keys),
        "covered_weight_key_count": len(covered_keys),
        "requested_stages_cover_all_weight_keys": len(covered_keys) == len(all_keys),
        "all_weight_file_count": len({Path(str(value)).name for value in dict(weight_index.get("weight_map") or {}).values()}),
        "total_size_bytes": int(dict(weight_index.get("metadata") or {}).get("total_size") or 0),
        "stage_plans": [
            {
                key: value
                for key, value in plan.items()
                if key not in {"assigned_weight_keys", "weight_map", "all_weight_files"}
            }
            for plan in stage_plans
        ],
        "public_safe": True,
    }


def build_report(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    stage_ids: list[int],
    packages: list[dict[str, Any]],
    stage_runs: list[dict[str, Any]],
    stage_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    stage_summaries = [summarize_stage_report(report) for report in stage_reports]
    expected_plan = local_expected_plan(args, stage_ids=stage_ids)
    pushed_refs = {
        f"stage{int(run.get('stage_id'))}": str(run.get("kernel_ref") or "")
        for run in stage_runs
        if run.get("kernel_ref")
    }
    lifecycle = {
        "requested_accelerator": args.accelerator,
        "requested_stage_ids": stage_ids,
        "expected_push_count": len(stage_ids),
        "actual_push_count": sum(1 for run in stage_runs if any(step.get("name") == "kaggle_kernel_push" and step.get("ok") for step in run.get("steps", []))),
        "pushed_refs": pushed_refs,
        "cleanup_attempted": all(any(step.get("name") == "kaggle_kernel_delete" for step in run.get("steps", [])) for run in stage_runs) if stage_runs else False,
        "kernels_deleted": all(any(step.get("name") == "kaggle_kernel_delete" and step.get("ok") for step in run.get("steps", [])) for run in stage_runs) if stage_runs else False,
        "private_packages_removed": not (output_dir / "private-kaggle-kernels").exists(),
    }
    all_reports_downloaded = len(stage_reports) == len(stage_ids) and all(bool(report) for report in stage_reports)
    all_stage_ready = bool(stage_reports) and all(summary["stage_owned_quantized_32b_loading_ready"] for summary in stage_summaries)
    all_load_scope_ready = bool(stage_reports) and all(summary["stage_weight_downloads_only_stage_files"] for summary in stage_summaries)
    all_only_stage_keys = bool(stage_reports) and all(summary["loads_only_stage_weight_keys"] for summary in stage_summaries)
    gpu_verified = bool(stage_reports) and all(summary["gpu_verified"] for summary in stage_summaries)
    coverage_ready = bool(expected_plan.get("requested_stages_cover_all_weight_keys"))
    ready = bool(
        all_reports_downloaded
        and all_stage_ready
        and all_load_scope_ready
        and all_only_stage_keys
        and coverage_ready
        and lifecycle["kernels_deleted"]
    )
    blockers: list[str] = []
    if not all_reports_downloaded:
        blockers.append("kaggle_stage_reports_not_downloaded")
    if not all_stage_ready:
        blockers.append("stage_owned_quantized_32b_loading_not_ready")
    if not all_load_scope_ready:
        blockers.append("stage_owned_weight_download_scope_not_ready")
    if not all_only_stage_keys:
        blockers.append("stage_owned_tensor_key_load_not_ready")
    if not coverage_ready:
        blockers.append("requested_stages_do_not_cover_full_weight_index")
    if not lifecycle["kernels_deleted"]:
        blockers.append("kaggle_kernels_cleanup_not_verified")
    diagnosis = [
        "fresh_kaggle_run_performed" if lifecycle["actual_push_count"] else "fresh_kaggle_run_not_performed",
        "kaggle_32b_gpu_hardware_verified" if gpu_verified else "kaggle_32b_gpu_hardware_not_verified",
        "kaggle_32b_stage_owned_weight_download_ready" if all_load_scope_ready else "kaggle_32b_stage_owned_weight_download_not_ready",
        "kaggle_32b_stage_owned_tensor_load_ready" if all_stage_ready else "kaggle_32b_stage_owned_tensor_load_not_ready",
        "kaggle_32b_loads_only_stage_weight_keys" if all_only_stage_keys else "kaggle_32b_cross_stage_weight_load_risk",
        "kaggle_32b_requested_stage_coverage_ready" if coverage_ready else "kaggle_32b_requested_stage_coverage_not_ready",
        "kaggle_kernels_deleted" if lifecycle["kernels_deleted"] else "kaggle_kernels_cleanup_not_verified",
    ]
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "stage_owned_quantized_32b_loading_ready": ready,
        "fresh_kaggle_run_performed": bool(lifecycle["actual_push_count"]),
        "all_stage_reports_downloaded": all_reports_downloaded,
        "all_stage_owned_loading_ready": all_stage_ready,
        "stage_owned_download_scope_ready": all_load_scope_ready,
        "loads_only_stage_weight_keys_ready": all_only_stage_keys,
        "gpu_hardware_verified": gpu_verified,
        "coverage_ready": coverage_ready,
        "output_dir": str(output_dir),
        "model": {
            "repo": args.model_repo,
            "parameter_count_b": 32,
            "quantization_format": "awq_safetensors",
            "safetensors_weight_index": True,
        },
        "runtime": {
            "stage_count": int(args.stage_count),
            "stage_ids": stage_ids,
            "kaggle_gpu_kernel_per_stage": True,
            "cross_kernel_network_inference": False,
            "one_token_generation_verified": False,
            "stage_owned_loading_only": True,
        },
        "expected_plan": expected_plan,
        "stage_summaries": stage_summaries,
        "kaggle_lifecycle": lifecycle,
        "stage_runs": stage_runs,
        "safety": {
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "credentials_public": False,
            "private_kernel_payload_public": False,
            "tensor_values_public": False,
            "local_weight_root_public": False,
        },
        "limitations": [
            "This verifies stage-owned quantized 32B safetensors download/load, not generated-token correctness.",
            "Hugging Face safetensors shard files are size-based; a boundary file may contain keys for adjacent logical stages, so the proof enforces stage-owned tensor-key loading inside selected files.",
            "This does not yet prove production P2P routing, pricing, trust, or arbitrary user serving.",
        ],
        "diagnosis_codes": sorted(set(diagnosis)),
        "blockers": sorted(set(blockers)),
    }


def run_stage(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    stage_id: int,
    runner: Runner,
) -> tuple[dict[str, Any], dict[str, Any]]:
    package = build_package(args, output_dir=output_dir, stage_id=stage_id)
    steps: list[dict[str, Any]] = []
    kernel_ref = str(package["declared_kernel_ref"])
    try:
        push_command = ["kaggle", "kernels", "push", "-p", str(package["kernel_dir"]), "-t", str(args.kernel_timeout_seconds)]
        if args.accelerator:
            push_command.extend(["--accelerator", args.accelerator])
        print(f"[{utc_now()}] pushing private Kaggle stage{stage_id} kernel {package['declared_kernel_ref']}", flush=True)
        push_step = run_step("kaggle_kernel_push", push_command, runner=runner, timeout_seconds=args.kaggle_push_timeout_seconds)
        steps.append(push_step)
        if push_step.get("ok"):
            kernel_ref, resolve_step = resolve_pushed_kernel_ref(
                package,
                push_step,
                runner=runner,
                timeout_seconds=args.kaggle_push_timeout_seconds,
            )
            if resolve_step:
                steps.append(resolve_step)
            package["kernel_ref"] = kernel_ref
            print(f"[{utc_now()}] waiting for stage{stage_id} {kernel_ref}", flush=True)
            status_step = wait_kaggle_terminal(
                kernel_ref,
                runner=runner,
                timeout_seconds=args.kaggle_status_timeout_seconds,
                poll_interval=args.kaggle_status_poll_interval,
            )
            steps.append(status_step)
            output_path = output_dir / "kaggle-output" / f"stage{stage_id}"
            output_step = run_step(
                "kaggle_kernel_output",
                [
                    "kaggle",
                    "kernels",
                    "output",
                    kernel_ref,
                    "-p",
                    str(output_path),
                    "--force",
                    "--file-pattern",
                    str(package["report_filename"]),
                ],
                runner=runner,
                timeout_seconds=args.kaggle_output_timeout_seconds,
            )
            steps.append(output_step)
            if not args.skip_kaggle_cleanup:
                print(f"[{utc_now()}] deleting private Kaggle stage{stage_id} kernel {kernel_ref}", flush=True)
                delete_step = run_step(
                    "kaggle_kernel_delete",
                    ["kaggle", "kernels", "delete", kernel_ref, "-y"],
                    runner=runner,
                    timeout_seconds=args.kaggle_delete_timeout_seconds,
                )
                steps.append(delete_step)
        report = load_json(output_dir / "kaggle-output" / f"stage{stage_id}" / str(package["report_filename"]))
    except Exception as exc:
        steps.append({
            "name": "stage_probe_exception",
            "ok": False,
            "stage_id": stage_id,
            "error_type": type(exc).__name__,
            "error_digest": "sha256:" + str(abs(hash(str(exc)))),
        })
        report = {}
    return {
        "stage_id": int(stage_id),
        "kernel_ref": kernel_ref,
        "kernel_slug": package.get("kernel_slug"),
        "steps": steps,
    }, report


def run_live_probe(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_ids = parse_stage_ids(args.stage_ids, stage_count=args.stage_count)
    packages: list[dict[str, Any]] = []
    stage_runs: list[dict[str, Any]] = []
    stage_reports: list[dict[str, Any]] = []
    try:
        for stage_id in stage_ids:
            stage_run, stage_report = run_stage(
                args,
                output_dir=output_dir,
                stage_id=stage_id,
                runner=runner,
            )
            stage_runs.append(stage_run)
            stage_reports.append(stage_report)
    finally:
        if not args.keep_private_package:
            shutil.rmtree(output_dir / "private-kaggle-kernels", ignore_errors=True)
    report = build_report(
        args,
        output_dir=output_dir,
        stage_ids=stage_ids,
        packages=packages,
        stage_runs=stage_runs,
        stage_reports=stage_reports,
    )
    write_json(output_dir / "kaggle_32b_stage_owned_safetensors_probe.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run private Kaggle stage-owned 32B quantized safetensors loading probes.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kaggle-owner", default=default_kaggle_owner())
    parser.add_argument("--kernel-slug-prefix", default="ct32bstage")
    parser.add_argument("--accelerator", default=DEFAULT_ACCELERATOR)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--stage-count", type=int, default=DEFAULT_STAGE_COUNT)
    parser.add_argument("--stage-ids", default=DEFAULT_STAGE_IDS)
    parser.add_argument("--retain-tensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--materialize-clone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retain-limit-gb", type=int, default=24)
    parser.add_argument("--kernel-timeout-seconds", type=int, default=3600)
    parser.add_argument("--kaggle-push-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--kaggle-status-timeout-seconds", type=float, default=3900.0)
    parser.add_argument("--kaggle-status-poll-interval", type=float, default=60.0)
    parser.add_argument("--kaggle-output-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--kaggle-delete-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--skip-kaggle-cleanup", action="store_true")
    parser.add_argument("--keep-private-package", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.stage_count < 2:
        raise SystemExit("--stage-count must be at least 2")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_live_probe(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {Path(args.output_dir) / 'kaggle_32b_stage_owned_safetensors_probe.json'}")
        print(f"Ready: {report.get('stage_owned_quantized_32b_loading_ready')}")
        if report.get("blockers"):
            print("Blockers: " + ", ".join(str(item) for item in report.get("blockers") or []))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
