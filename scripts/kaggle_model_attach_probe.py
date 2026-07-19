#!/usr/bin/env python3
"""Bounded Kaggle Models attach probe for dense Qwen model sources."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import kaggle_dense_model_source_resolver as resolver  # noqa: E402


SCHEMA = "kaggle_model_attach_probe_v1"
STATUS_RE = re.compile(r'has status "([^"]+)"')
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


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", str(value).lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug[:63].strip("-") or "ct-model-attach"


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


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def redact_output(output: str) -> str:
    output = re.sub(r"(?i)(kaggle[_-]?key|api[_-]?key|token)[=:]\S+", r"\1=<redacted>", output)
    return output


def redact_command(command: list[str]) -> list[str]:
    return ["<redacted>" if "token" in item.lower() or "KAGGLE_KEY" in item else item for item in command]


def run_command(command: list[str], *, timeout: float) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "duration_seconds": round(time.time() - started, 3),
            "command": redact_command(command),
            "output_tail": redact_output(proc.stdout)[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        return {
            "ok": False,
            "returncode": None,
            "duration_seconds": round(time.time() - started, 3),
            "command": redact_command(command),
            "timed_out": True,
            "output_tail": redact_output(output)[-4000:],
        }


def load_kaggle_owner() -> str:
    code = (
        "from kaggle.api.kaggle_api_extended import KaggleApi\n"
        "api=KaggleApi(); api.authenticate()\n"
        "print(api.config_values.get('username') or api.config_values.get('user') or '')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=20,
    )
    return safe_slug(proc.stdout.strip()) if proc.returncode == 0 and proc.stdout.strip() else ""


def push_accepted(step: dict[str, Any]) -> bool:
    output = str(step.get("output_tail") or "")
    return bool(step.get("ok")) and "Kernel version" in output and "successfully pushed" in output


def parse_status(output: str) -> str:
    match = STATUS_RE.search(output)
    if match:
        return match.group(1)
    lines = output.strip().splitlines()
    return lines[-1][:160] if lines else ""


def status_class(status: str) -> str:
    upper = status.upper()
    if "COMPLETE" in upper or "SUCCESS" in upper:
        return "complete"
    if "FAIL" in upper or "ERROR" in upper or "CANCEL" in upper:
        return "failed"
    if "RUNNING" in upper:
        return "running"
    if "QUEUE" in upper or "PENDING" in upper or "PREPAR" in upper or "INITIAL" in upper:
        return "queued"
    return "unknown"


def kernel_code(
    *,
    expected_path: str,
    expected_paths: list[str] | None = None,
    model_source: str,
    stage_plan_enabled: bool = False,
    stage_count: int = 10,
    stage_backends: list[str] | None = None,
    max_header_bytes: int = 16 * 1024 * 1024,
) -> str:
    settings = {
        "expected_path": expected_path,
        "expected_paths": [str(item) for item in (expected_paths or [expected_path]) if str(item).strip()],
        "model_source": model_source,
        "stage_plan_enabled": bool(stage_plan_enabled),
        "stage_count": max(1, int(stage_count)),
        "stage_backends": [str(item).strip() for item in (stage_backends or []) if str(item).strip()],
        "max_header_bytes": max(1024, int(max_header_bytes)),
    }
    return r'''
import json
import struct
from pathlib import Path
from datetime import datetime, timezone

SETTINGS = __SETTINGS__


def _int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def _stage_layer_ranges(layer_count, stage_count):
    layer_count = max(0, int(layer_count))
    stage_count = max(1, int(stage_count))
    base = layer_count // stage_count
    remainder = layer_count % stage_count
    ranges = []
    start = 0
    for stage_id in range(stage_count):
        width = base + (1 if stage_id < remainder else 0)
        end = start + width
        ranges.append([start, end])
        start = end
    return ranges


def _stage_prefixes(stage_id, stage_count, layer_range):
    start, end = layer_range
    prefixes = []
    if stage_id == 0:
        prefixes.append("model.embed_tokens.")
    for layer_id in range(start, end):
        prefixes.append(f"model.layers.{layer_id}.")
    if stage_id == stage_count - 1:
        prefixes.extend(["model.norm.", "lm_head."])
    return prefixes


def _read_safetensors_header(path, max_header_bytes):
    if not path.is_file():
        return {}, "missing_file"
    with path.open("rb") as handle:
        first = handle.read(8)
        if len(first) != 8:
            return {}, "header_length_prefix_missing"
        header_len = struct.unpack("<Q", first)[0]
        if header_len > max_header_bytes:
            return {}, "header_too_large"
        raw = handle.read(header_len)
        if len(raw) != header_len:
            return {}, "header_truncated"
    try:
        loaded = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        return {}, "header_json_" + type(exc).__name__
    if not isinstance(loaded, dict):
        return {}, "header_not_object"
    return loaded, ""


def _tensor_nbytes(item):
    if not isinstance(item, dict):
        return 0
    offsets = item.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        return 0
    try:
        start = int(offsets[0])
        end = int(offsets[1])
    except Exception:
        return 0
    return max(0, end - start)


def _build_stage_plan(expected, config, index):
    weight_map = index.get("weight_map") if isinstance(index, dict) else {}
    if not isinstance(weight_map, dict):
        weight_map = {}
    layer_count = _int(config.get("num_hidden_layers"))
    hidden_size = _int(config.get("hidden_size"))
    stage_count = max(1, _int(SETTINGS.get("stage_count"), 10))
    backends = [str(item) for item in SETTINGS.get("stage_backends") or [] if str(item)]
    if len(backends) < stage_count:
        backends.extend(["cpu"] * (stage_count - len(backends)))
    backends = backends[:stage_count]
    ranges = _stage_layer_ranges(layer_count, stage_count)
    header_cache = {}
    plans = []
    total_logical_bytes = 0
    max_stage_logical_bytes = 0
    total_assigned_keys = 0
    total_present_keys = 0
    all_files = set()
    all_verified = bool(weight_map and layer_count > 0)
    for stage_id, layer_range in enumerate(ranges):
        prefixes = _stage_prefixes(stage_id, stage_count, layer_range)
        assigned_keys = sorted(
            key for key in weight_map
            if any(str(key).startswith(prefix) for prefix in prefixes)
        )
        assigned_files = sorted({Path(str(weight_map[key])).name for key in assigned_keys})
        all_files.update(assigned_files)
        stage_present_keys = 0
        stage_missing_keys = 0
        stage_logical_bytes = 0
        file_summaries = []
        header_errors = []
        for file_name in assigned_files:
            path = expected / file_name
            if file_name not in header_cache:
                header_cache[file_name] = _read_safetensors_header(
                    path,
                    _int(SETTINGS.get("max_header_bytes"), 16 * 1024 * 1024),
                )
            header, error = header_cache[file_name]
            file_assigned_keys = [key for key in assigned_keys if Path(str(weight_map[key])).name == file_name]
            file_present_keys = 0
            file_logical_bytes = 0
            if error:
                header_errors.append(file_name + ":" + error)
                stage_missing_keys += len(file_assigned_keys)
            else:
                for key in file_assigned_keys:
                    item = header.get(key)
                    if isinstance(item, dict):
                        file_present_keys += 1
                        file_logical_bytes += _tensor_nbytes(item)
                    else:
                        stage_missing_keys += 1
                public_header_key_count = len([key for key in header if key != "__metadata__"])
                file_summaries.append(
                    {
                        "file_name": file_name,
                        "assigned_key_count": len(file_assigned_keys),
                        "present_key_count": file_present_keys,
                        "logical_tensor_bytes": file_logical_bytes,
                        "logical_tensor_gb": round(file_logical_bytes / 1_000_000_000, 6),
                        "skipped_non_stage_key_count": max(0, public_header_key_count - file_present_keys),
                    }
                )
            stage_present_keys += file_present_keys
            stage_logical_bytes += file_logical_bytes
        total_assigned_keys += len(assigned_keys)
        total_present_keys += stage_present_keys
        total_logical_bytes += stage_logical_bytes
        max_stage_logical_bytes = max(max_stage_logical_bytes, stage_logical_bytes)
        stage_verified = bool(
            assigned_keys
            and not header_errors
            and stage_missing_keys == 0
            and stage_present_keys == len(assigned_keys)
        )
        all_verified = all_verified and stage_verified
        plans.append(
            {
                "stage_id": stage_id,
                "backend": backends[stage_id],
                "layer_range": layer_range,
                "assigned_key_count": len(assigned_keys),
                "present_key_count": stage_present_keys,
                "missing_key_count": stage_missing_keys,
                "assigned_file_count": len(assigned_files),
                "logical_tensor_bytes": stage_logical_bytes,
                "logical_tensor_gb": round(stage_logical_bytes / 1_000_000_000, 6),
                "header_error_count": len(header_errors),
                "header_errors": header_errors[:8],
                "stage_owned_header_verified": stage_verified,
                "file_summaries": file_summaries,
            }
        )
    return {
        "schema": "kaggle_model_attach_stage_plan_v1",
        "enabled": True,
        "stage_count": stage_count,
        "stage_backends": backends,
        "num_hidden_layers": layer_count,
        "hidden_size": hidden_size,
        "assigned_key_count_total": total_assigned_keys,
        "present_key_count_total": total_present_keys,
        "assigned_file_count_total": len(all_files),
        "total_planned_logical_tensor_bytes": total_logical_bytes,
        "total_planned_logical_tensor_gb": round(total_logical_bytes / 1_000_000_000, 6),
        "max_stage_planned_logical_tensor_bytes": max_stage_logical_bytes,
        "max_stage_planned_logical_tensor_gb": round(max_stage_logical_bytes / 1_000_000_000, 6),
        "stage_owned_preflight_verified": bool(all_verified),
        "stage_plans": plans,
        "weight_tensor_values_public": False,
        "public_artifact_safe": True,
    }


expected_paths = [Path(str(item)) for item in (SETTINGS.get("expected_paths") or []) if str(item)]
if not expected_paths:
    expected_paths = [Path(str(SETTINGS.get("expected_path") or ""))]
attached_path_probes = []
for path in expected_paths:
    attached_path_probes.append(
        {
            "path": str(path),
            "path_present": path.is_dir(),
            "config_json_present": (path / "config.json").is_file(),
            "weight_index_present": (path / "model.safetensors.index.json").is_file(),
            "safetensors_file_count": len(list(path.glob("*.safetensors"))) if path.is_dir() else 0,
        }
    )
selected_path = next(
    (Path(item["path"]) for item in attached_path_probes if item.get("path_present")),
    expected_paths[0],
)
expected = selected_path
input_root = Path("/kaggle/input")
input_root_listing = []
if input_root.is_dir():
    for child in sorted(input_root.iterdir(), key=lambda p: p.name)[:80]:
        try:
            input_root_listing.append({
                "name": child.name,
                "is_dir": child.is_dir(),
                "child_names": sorted([p.name for p in child.iterdir()])[:40] if child.is_dir() else [],
            })
        except Exception as exc:
            input_root_listing.append({"name": child.name, "is_dir": child.is_dir(), "error_type": type(exc).__name__})
result = {
    "schema": "kaggle_model_attach_runtime_report_v1",
    "ok": False,
    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "model_source": str(SETTINGS.get("model_source") or ""),
    "expected_attached_path": str(expected_paths[0]),
    "expected_attached_paths": [str(path) for path in expected_paths],
    "attached_path_probes": attached_path_probes,
    "resolved_attached_path": str(expected),
    "path_present": expected.is_dir(),
    "input_root_present": input_root.is_dir(),
    "input_root_listing": input_root_listing,
    "config_json_present": (expected / "config.json").is_file(),
    "weight_index_present": (expected / "model.safetensors.index.json").is_file(),
    "tokenizer_json_present": (expected / "tokenizer.json").is_file(),
    "tokenizer_config_present": (expected / "tokenizer_config.json").is_file(),
    "safetensors_file_count": len(list(expected.glob("*.safetensors"))) if expected.is_dir() else 0,
    "top_level_file_names": sorted([p.name for p in expected.iterdir() if p.is_file()])[:80] if expected.is_dir() else [],
    "top_level_dir_names": sorted([p.name for p in expected.iterdir() if p.is_dir()])[:40] if expected.is_dir() else [],
    "weight_tensor_values_public": False,
    "raw_prompt_public": False,
    "generated_text_public": False,
    "generated_token_ids_public": False,
    "activation_public": False,
    "kv_cache_public": False,
    "credentials_public": False,
    "public_artifact_safe": True,
    "stage_plan_enabled": bool(SETTINGS.get("stage_plan_enabled")),
    "stage_owned_preflight_verified": False,
}
index = {}
index_path = expected / "model.safetensors.index.json"
if index_path.is_file():
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") if isinstance(index, dict) else {}
        files = sorted(set(str(v) for v in weight_map.values())) if isinstance(weight_map, dict) else []
        result["weight_index_key_count"] = len(weight_map) if isinstance(weight_map, dict) else 0
        result["weight_index_file_count"] = len(files)
        result["weight_index_first_files"] = [Path(name).name for name in files[:10]]
    except Exception as exc:
        result["weight_index_error_type"] = type(exc).__name__
config = {}
config_path = expected / "config.json"
if config_path.is_file():
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result["model_type"] = str(config.get("model_type") or "")
        result["architectures"] = list(config.get("architectures") or [])[:8]
        result["num_hidden_layers"] = config.get("num_hidden_layers")
        result["hidden_size"] = config.get("hidden_size")
        result["torch_dtype"] = str(config.get("torch_dtype") or "")
        result["quantization_config_present"] = bool(config.get("quantization_config"))
    except Exception as exc:
        result["config_error_type"] = type(exc).__name__
attach_ok = bool(result["path_present"] and result["config_json_present"] and result["weight_index_present"] and result["safetensors_file_count"] > 0)
result["attach_ok"] = attach_ok
if attach_ok and bool(SETTINGS.get("stage_plan_enabled")):
    try:
        stage_plan = _build_stage_plan(expected, config, index)
        result["stage_plan"] = stage_plan
        result["stage_owned_preflight_verified"] = stage_plan.get("stage_owned_preflight_verified") is True
    except Exception as exc:
        result["stage_plan_error_type"] = type(exc).__name__
        result["stage_owned_preflight_verified"] = False
result["ok"] = bool(attach_ok and (not bool(SETTINGS.get("stage_plan_enabled")) or result["stage_owned_preflight_verified"] is True))
Path("kaggle_model_attach_runtime_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"ok": result["ok"], "attach_ok": result["attach_ok"], "path_present": result["path_present"], "safetensors_file_count": result["safetensors_file_count"], "stage_owned_preflight_verified": result["stage_owned_preflight_verified"]}, sort_keys=True), flush=True)
'''.replace("__SETTINGS__", repr(settings))


def write_kernel_package(
    kernel_dir: Path,
    *,
    owner: str,
    slug: str,
    model_source: str,
    expected_path: str,
    expected_paths: list[str],
    enable_internet: bool,
    stage_plan_enabled: bool,
    stage_count: int,
    stage_backends: list[str],
    max_header_bytes: int,
) -> str:
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text(
        kernel_code(
            expected_path=expected_path,
            expected_paths=expected_paths,
            model_source=model_source,
            stage_plan_enabled=stage_plan_enabled,
            stage_count=stage_count,
            stage_backends=stage_backends,
            max_header_bytes=max_header_bytes,
        ),
        encoding="utf-8",
    )
    write_json(
        kernel_dir / "kernel-metadata.json",
        {
            "id": f"{owner}/{slug}",
            "title": slug.replace("-", " ").title(),
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "false",
            "enable_internet": "true" if enable_internet else "false",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [model_source],
        },
    )
    return f"{owner}/{slug}"


def candidate_for_parameter(parameter_class: str) -> dict[str, Any]:
    requested = str(parameter_class).lower().strip()
    for raw in resolver.DEFAULT_CANDIDATES:
        candidate = resolver.parse_candidate(raw)
        if str(candidate["parameter_class"]).lower() == requested:
            return candidate
    raise SystemExit(f"unknown --parameter-class {parameter_class!r}")


def candidate_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.hf_repo or "").strip():
        return {
            "parameter_class": str(args.parameter_class),
            "hf_repo": str(args.hf_repo),
            "owner_slug": "",
            "model_slug": "",
            "framework": "Transformers",
            "instance_slug": "",
            "version_number": 1,
        }
    return candidate_for_parameter(args.parameter_class)


def default_expected_paths(candidate: dict[str, Any], *, input_root: str = "/kaggle/input") -> list[str]:
    owner = str(candidate.get("owner_slug") or "").strip()
    model = str(candidate.get("model_slug") or "").strip()
    framework = str(candidate.get("framework") or "Transformers").strip()
    instance = str(candidate.get("instance_slug") or "").strip()
    version = str(candidate.get("version_number") or "1").strip()
    if not (owner and model and framework and instance):
        return []
    root = input_root.rstrip("/")
    framework_lower = framework.lower()
    candidates = [
        f"{root}/models/{owner}/{model}/{framework_lower}/{instance}/{version}",
        f"{root}/models/{owner}/{model}/{framework_lower}/{instance}",
        f"{root}/{owner}/{model}/{framework_lower}/{instance}/{version}",
        f"{root}/{owner}/{model}/{framework_lower}/{instance}",
        f"{root}/{model}/{framework_lower}/{instance}/{version}",
        f"{root}/{model}/{framework_lower}/{instance}",
    ]
    deduped: list[str] = []
    for item in candidates:
        if item not in deduped:
            deduped.append(item)
    return deduped


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    package_dir = output_dir / "private-kaggle-model-attach-kernel"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    owner = safe_slug(args.owner) if args.owner else load_kaggle_owner()
    candidate = candidate_from_args(args)
    model_source = args.model_source or resolver.kernel_model_source_ref(candidate)
    expected_paths = [item.strip() for item in str(args.expected_path or "").split(",") if item.strip()]
    if not expected_paths:
        expected_paths = default_expected_paths(candidate) or [
            resolver.attached_runtime_path(candidate, input_root="/kaggle/input")
        ]
    expected_path = expected_paths[0]
    stage_backends = [item.strip() for item in str(args.stage_backends or "").split(",") if item.strip()]
    slug = safe_slug(f"{args.slug_prefix}-{args.parameter_class}-{str(int(time.time()))[-8:]}")
    kernel_ref = f"{owner}/{slug}" if owner else ""
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": False,
        "generated_at": utc_now(),
        "owner": owner,
        "parameter_class": str(candidate["parameter_class"]),
        "hf_repo": str(candidate["hf_repo"]),
        "model_source": model_source,
        "expected_attached_path": expected_path,
        "expected_attached_paths": expected_paths,
        "kernel_ref": kernel_ref,
        "cpu_only": True,
        "private_kernel": True,
        "kaggle_model_attach_probe_ready": False,
        "kaggle_model_attach_used": False,
        "stage_plan_requested": bool(args.stage_plan),
        "stage_owned_preflight_verified": False,
        "runtime_report": {},
        "steps": [],
        "cleanup_status": {
            "temporary_kaggle_kernel_created": False,
            "temporary_kaggle_kernel_deleted": False,
            "temporary_private_package_removed": False,
            "live_resources_left_running": False,
        },
        "blocker_codes": [],
        "safety": {
            "public_artifact_safe": True,
            "cpu_only": True,
            "gpu_enabled": False,
            "tpu_enabled": False,
            "private_kernel": True,
            "credentials_public": False,
            "raw_prompt_public": False,
            "generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
        },
        "public_artifact_safe": True,
    }
    if not owner:
        report["blocker_codes"].append("kaggle_owner_missing")
        return report
    created = False
    try:
        write_kernel_package(
            package_dir,
            owner=owner,
            slug=slug,
            model_source=model_source,
            expected_path=expected_path,
            expected_paths=expected_paths,
            enable_internet=bool(args.enable_internet),
            stage_plan_enabled=bool(args.stage_plan),
            stage_count=int(args.stage_count),
            stage_backends=stage_backends,
            max_header_bytes=int(args.max_header_bytes),
        )
        push = run_command(
            ["kaggle", "kernels", "push", "-p", str(package_dir), "-t", str(int(args.kernel_timeout_seconds))],
            timeout=float(args.push_timeout_seconds),
        )
        push["accepted"] = push_accepted(push)
        report["steps"].append({"name": "kaggle_kernel_push", **push})
        if not push["accepted"]:
            report["blocker_codes"].append("kaggle_kernel_push_rejected")
            return report
        created = True
        report["cleanup_status"]["temporary_kaggle_kernel_created"] = True
        terminal = False
        last_status = ""
        for poll_index in range(max(1, int(args.status_polls))):
            status_step = run_command(["kaggle", "kernels", "status", kernel_ref], timeout=float(args.status_timeout_seconds))
            last_status = parse_status(str(status_step.get("output_tail") or ""))
            status_step["status"] = last_status
            status_step["status_class"] = status_class(last_status)
            status_step["poll_index"] = poll_index
            report["steps"].append({"name": "kaggle_kernel_status", **status_step})
            if status_step["status_class"] in {"complete", "failed"}:
                terminal = True
                break
            if poll_index + 1 < int(args.status_polls):
                time.sleep(max(0.0, float(args.status_poll_interval)))
        if not terminal:
            report["blocker_codes"].append("kaggle_kernel_status_timeout")
        elif status_class(last_status) == "failed":
            report["blocker_codes"].append("kaggle_kernel_failed")
        output_dir_runtime = output_dir / "kaggle-output"
        output_step = run_command(
            [
                "kaggle",
                "kernels",
                "output",
                kernel_ref,
                "-p",
                str(output_dir_runtime),
                "--force",
                "--file-pattern",
                "kaggle_model_attach_runtime_report.json",
            ],
            timeout=float(args.output_timeout_seconds),
        )
        report["steps"].append({"name": "kaggle_kernel_output", **output_step})
        runtime_report = load_json(output_dir_runtime / "kaggle_model_attach_runtime_report.json")
        report["runtime_report"] = runtime_report
        if runtime_report.get("resolved_attached_path"):
            report["resolved_attached_path"] = runtime_report.get("resolved_attached_path")
        report["stage_owned_preflight_verified"] = runtime_report.get("stage_owned_preflight_verified") is True
        attach_ready = bool(runtime_report.get("attach_ok") is True or (runtime_report.get("ok") is True and not args.stage_plan))
        report["kaggle_model_attach_used"] = attach_ready
        report["kaggle_model_attach_probe_ready"] = bool(
            attach_ready and (not args.stage_plan or runtime_report.get("stage_owned_preflight_verified") is True)
        )
        if runtime_report.get("ok") is not True:
            if not runtime_report:
                report["blocker_codes"].append("kaggle_attach_runtime_report_missing")
            elif runtime_report.get("path_present") is not True:
                report["blocker_codes"].append("kaggle_attach_path_missing_in_runtime")
            elif runtime_report.get("weight_index_present") is not True:
                report["blocker_codes"].append("kaggle_attach_weight_index_missing")
            elif _int(runtime_report.get("safetensors_file_count")) < 1:
                report["blocker_codes"].append("kaggle_attach_safetensors_missing")
            elif args.stage_plan and runtime_report.get("stage_owned_preflight_verified") is not True:
                report["blocker_codes"].append("stage_owned_preflight_not_verified")
    finally:
        if created and not args.keep_kernel:
            delete_step = run_command(["kaggle", "kernels", "delete", "-y", kernel_ref], timeout=float(args.delete_timeout_seconds))
            report["steps"].append({"name": "kaggle_kernel_delete", **delete_step})
            report["cleanup_status"]["temporary_kaggle_kernel_deleted"] = bool(delete_step.get("ok"))
        elif created:
            report["cleanup_status"]["live_resources_left_running"] = True
        shutil.rmtree(package_dir, ignore_errors=True)
        report["cleanup_status"]["temporary_private_package_removed"] = not package_dir.exists()
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["kaggle_model_attach_probe_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blocker_codes"].append("public_redaction_scan_failed")
        report["redaction_errors"] = leaks
    else:
        report["ok"] = bool(
            report["kaggle_model_attach_probe_ready"]
            and report["cleanup_status"]["temporary_private_package_removed"] is True
            and (report["cleanup_status"]["temporary_kaggle_kernel_deleted"] is True or args.keep_kernel)
        )
    return report


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Kaggle Models attach path in a private CPU kernel.")
    parser.add_argument("--output-dir", default="dist/kaggle-model-attach-probe")
    parser.add_argument("--owner", default="")
    parser.add_argument("--parameter-class", default="7b")
    parser.add_argument("--hf-repo", default="")
    parser.add_argument("--model-source", default="")
    parser.add_argument("--expected-path", default="")
    parser.add_argument("--slug-prefix", default="ct-model-attach")
    parser.add_argument("--kernel-timeout-seconds", type=int, default=600)
    parser.add_argument("--push-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--status-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--status-polls", type=int, default=30)
    parser.add_argument("--status-poll-interval", type=float, default=20.0)
    parser.add_argument("--output-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--delete-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--enable-internet", action="store_true")
    parser.add_argument("--keep-kernel", action="store_true")
    parser.add_argument("--stage-plan", action="store_true", help="Verify stage-owned safetensors headers in the attached model.")
    parser.add_argument("--stage-count", type=int, default=10)
    parser.add_argument(
        "--stage-backends",
        default="cuda,cuda,cuda,cuda,jax_tpu,cpu,cpu,cpu,cpu,cpu",
        help="Comma-separated public backend labels for the stage plan.",
    )
    parser.add_argument("--max-header-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.stage_count < 1 or args.stage_count > 128:
        raise SystemExit("--stage-count must be between 1 and 128")
    if args.max_header_bytes < 1024:
        raise SystemExit("--max-header-bytes must be at least 1024")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args)
    report_path = output_dir / "kaggle_model_attach_probe.json"
    write_json(report_path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {report_path}")
        print(f"Ready: {report.get('kaggle_model_attach_probe_ready')}")
        if report.get("blocker_codes"):
            print("Blockers: " + ", ".join(str(item) for item in report.get("blocker_codes") or []))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
