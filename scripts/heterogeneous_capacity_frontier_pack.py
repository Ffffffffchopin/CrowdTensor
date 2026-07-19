#!/usr/bin/env python3
"""Build GPU+TPU+CPU heterogeneous large-model capacity frontier evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "heterogeneous_capacity_frontier_v1"
SUPPORT_BUNDLE_SCHEMA = "heterogeneous_capacity_frontier_support_bundle_v1"
DEFAULT_OUTPUT_DIR = "dist/heterogeneous-capacity-frontier"
DEFAULT_32B_BRIDGE_REPORT = (
    "dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260625-r6-existing-session-4token/"
    "gpu_tpu_cpu_same_request_runtime_bridge_probe.json"
)
DEFAULT_32B_SERVING_REPORT = (
    "dist/heterogeneous-32b-serving-20260625-r6-live-4token-success/"
    "heterogeneous_32b_serving.json"
)
DEFAULT_72B_STAGE_LOAD_REPORT = (
    "dist/kaggle-72b-stage-owned-safetensors-probe-awq-live-r2-full10/"
    "kaggle_32b_stage_owned_safetensors_probe.json"
)
DEFAULT_100B_PARTIAL_STAGE_REPORT = (
    "dist/kaggle-100b-stage-owned-safetensors-probe-compressed-live-r1-stage8/"
    "kaggle-output/stage8/ct_32b_stage_owned_safetensors_stage8_report.json"
)
DEFAULT_CANDIDATES = (
    "72b-awq|Qwen/Qwen2.5-72B-Instruct-AWQ|awq_safetensors|decode",
    "72b-gptq|Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4|gptq_safetensors|decode",
    "72b-full|Qwen/Qwen2.5-72B-Instruct|full_precision_safetensors|decode",
    "100b-compressed|cyankiwi/Solar-Open-100B-AWQ-4bit|compressed_tensors_4bit_safetensors|stage_load",
    "235b-awq|QuixiAI/Qwen3-235B-A22B-AWQ|awq_safetensors|stage_load",
)
DEFAULT_STAGE_BACKENDS = (
    "cuda",
    "cuda",
    "cuda",
    "cuda",
    "jax_tpu",
    "cpu",
    "cpu",
    "cpu",
    "cpu",
    "cpu",
)
EXECUTION_MODES = ("metadata-preflight", "external-existing")
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Authorization:",
    "Set-Cookie",
    "Cookie:",
    "jupyter-proxy",
    "token=",
    "kaggle-cookies",
    "kaggle-web-storage-state",
    "operator.private.env",
    "miner.private.env",
    "kernel.py",
    '"prompt":',
    '"prompt_text":',
    '"raw_prompt":',
    '"generated_text":',
    '"output_text":',
    '"generated_token_ids":',
    '"token_ids":',
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


def load_optional_json(path: Path) -> dict[str, Any]:
    return load_json(path) if path.is_file() else {}


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


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def fetch_hf_json(model_repo: str, filename: str, *, timeout_seconds: float = 60.0) -> dict[str, Any]:
    url = f"https://huggingface.co/{model_repo}/resolve/main/{filename}"
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        loaded = json.load(response)
    return loaded if isinstance(loaded, dict) else {}


def read_range(model_repo: str, filename: str, start: int, end: int, *, timeout_seconds: float, max_bytes: int) -> bytes:
    url = f"https://huggingface.co/{model_repo}/resolve/main/{filename}"
    req = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={int(start)}-{int(end)}",
            "User-Agent": "crowdtensor-capacity-frontier/1",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
        payload = response.read(int(max_bytes) + 1)
    if len(payload) > int(max_bytes):
        raise RuntimeError("hf_range_response_exceeded_budget")
    return payload


def load_safetensors_header(
    model_repo: str,
    filename: str,
    *,
    timeout_seconds: float,
    max_header_bytes: int,
) -> tuple[int, dict[str, Any]]:
    prefix = read_range(model_repo, filename, 0, 7, timeout_seconds=timeout_seconds, max_bytes=8)
    if len(prefix) != 8:
        raise RuntimeError("safetensors_header_prefix_missing")
    header_len = int(struct.unpack("<Q", prefix)[0])
    if header_len <= 0 or header_len > int(max_header_bytes):
        raise RuntimeError("safetensors_header_length_out_of_budget")
    payload = read_range(
        model_repo,
        filename,
        8,
        8 + header_len - 1,
        timeout_seconds=timeout_seconds,
        max_bytes=header_len,
    )
    if len(payload) != header_len:
        raise RuntimeError("safetensors_header_truncated")
    loaded = json.loads(payload.decode("utf-8"))
    return header_len, loaded if isinstance(loaded, dict) else {}


def parse_candidate(raw: str) -> dict[str, str]:
    parts = [part.strip() for part in str(raw or "").split("|")]
    if len(parts) != 4 or not all(parts):
        raise SystemExit("--candidate must be parameter_class|model_id|quantization|target")
    return {
        "parameter_class": parts[0],
        "model_id": parts[1],
        "quantization": parts[2],
        "target": parts[3],
    }


def parameter_class_value(value: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*b", str(value or "").lower())
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)", str(value or ""))
    return float(match.group(1)) if match else 0.0


def normalize_stage_count(stage_count: int, *, layer_count: int) -> int:
    count = max(3, min(int(stage_count), 32))
    if layer_count > 0:
        count = min(count, max(3, int(layer_count)))
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
        prefixes = [*prefixes, "model.norm.", "model.rotary_emb.", "lm_head."]
    return prefixes


def classify_weight_format(config: dict[str, Any], weight_map: dict[str, str], declared: str) -> str:
    quant = _dict(config.get("quantization_config")) or _dict(config.get("compression_config"))
    method = str(quant.get("quant_method") or quant.get("format") or declared or "").lower()
    keys = list(weight_map)[:5000]
    if "gptq" in method or "gptq" in declared.lower():
        return "gptq_safetensors"
    if "awq" in method or "awq" in declared.lower():
        return "awq_safetensors"
    if any(".qweight" in key or ".qzeros" in key for key in keys):
        return "quantized_safetensors"
    if any(".weight_packed" in key or ".weight_scale" in key for key in keys):
        return "compressed_tensors_safetensors"
    return "full_precision_safetensors"


def backend_for_stage(stage_id: int, stage_count: int, configured: list[str]) -> str:
    if stage_id < len(configured):
        return configured[stage_id]
    return "cpu"


def build_stage_plan(
    *,
    model_repo: str,
    config: dict[str, Any],
    weight_index: dict[str, Any],
    stage_id: int,
    stage_count: int,
    backend: str,
    header_cache: dict[str, tuple[int, dict[str, Any]]],
    timeout_seconds: float,
    max_header_bytes: int,
    fetch_headers: bool,
) -> dict[str, Any]:
    weight_map = {
        str(key): Path(str(value)).name
        for key, value in dict(weight_index.get("weight_map") or {}).items()
        if str(key or "").strip() and str(value or "").strip()
    }
    layer_count = _int(config.get("num_hidden_layers") or config.get("n_layer"))
    count = normalize_stage_count(stage_count, layer_count=layer_count)
    ranges = stage_layer_ranges(layer_count, count)
    layer_range = ranges[int(stage_id)]
    prefixes = stage_prefixes(stage_id=stage_id, stage_count=count, layer_range=layer_range)
    assigned = sorted(key for key in weight_map if any(key.startswith(prefix) for prefix in prefixes))
    assigned_set = set(assigned)
    files = sorted({weight_map[key] for key in assigned if weight_map.get(key)})
    present_key_count = 0
    missing_key_count = 0
    candidate_file_key_count = 0
    skipped_non_stage_key_count = 0
    logical_bytes = 0
    header_file_summaries: list[dict[str, Any]] = []
    header_errors: list[dict[str, Any]] = []
    if fetch_headers:
        for filename in files:
            try:
                if filename not in header_cache:
                    header_cache[filename] = load_safetensors_header(
                        model_repo,
                        filename,
                        timeout_seconds=timeout_seconds,
                        max_header_bytes=max_header_bytes,
                    )
                header_len, header = header_cache[filename]
                header_keys = [str(key) for key in header if key != "__metadata__"]
                available = set(header_keys)
                expected = [key for key in assigned if weight_map.get(key) == filename]
                present = [key for key in expected if key in available]
                missing = [key for key in expected if key not in available]
                present_key_count += len(present)
                missing_key_count += len(missing)
                candidate_file_key_count += len(header_keys)
                skipped_non_stage_key_count += len([key for key in header_keys if key not in assigned_set])
                file_bytes = 0
                for key in present:
                    meta = _dict(header.get(key))
                    offsets = _list(meta.get("data_offsets"))
                    if len(offsets) == 2:
                        size = max(0, int(offsets[1]) - int(offsets[0]))
                        logical_bytes += size
                        file_bytes += size
                header_file_summaries.append(
                    {
                        "filename": filename,
                        "header_len": int(header_len),
                        "expected_stage_key_count": len(expected),
                        "present_stage_key_count": len(present),
                        "candidate_file_key_count": len(header_keys),
                        "stage_logical_tensor_bytes": int(file_bytes),
                    }
                )
            except Exception as exc:
                header_errors.append(
                    {
                        "filename": filename,
                        "error_type": type(exc).__name__,
                        "error_digest": sha_payload(str(exc)),
                    }
                )
    shared_boundary_files = sorted(
        filename
        for filename in files
        if any(other_key not in assigned_set and weight_map.get(other_key) == filename for other_key in weight_map)
    )
    header_verified = bool(fetch_headers and assigned and present_key_count == len(assigned) and missing_key_count == 0 and not header_errors)
    return {
        "stage_id": int(stage_id),
        "backend": backend,
        "stage_layer_range": [int(layer_range[0]), int(layer_range[1])],
        "assigned_weight_key_count": len(assigned),
        "assigned_weight_file_count": len(files),
        "assigned_weight_file_digest": sha_payload(files),
        "shared_boundary_file_count": len(shared_boundary_files),
        "stage_owned_header_verified": header_verified,
        "present_stage_key_count": int(present_key_count),
        "missing_stage_key_count": int(missing_key_count),
        "candidate_file_key_count": int(candidate_file_key_count),
        "skipped_non_stage_key_count": int(skipped_non_stage_key_count),
        "planned_logical_tensor_bytes": int(logical_bytes),
        "planned_logical_tensor_gb": round(float(logical_bytes) / 1024 / 1024 / 1024, 6),
        "header_file_summaries": header_file_summaries,
        "header_errors": header_errors,
        "loads_only_stage_weight_keys_preflight": header_verified,
        "cross_stage_weight_keys_loaded": False,
    }


def build_candidate(
    candidate: dict[str, str],
    *,
    stage_count: int,
    stage_backends: list[str],
    timeout_seconds: float,
    max_header_bytes: int,
    fetch_headers: bool,
) -> dict[str, Any]:
    model_id = candidate["model_id"]
    report: dict[str, Any] = {
        "schema": "heterogeneous_capacity_candidate_v1",
        "model_id": model_id,
        "parameter_class": candidate["parameter_class"],
        "parameter_class_value_b": parameter_class_value(candidate["parameter_class"]),
        "quantization": candidate["quantization"],
        "target": candidate["target"],
        "stage_owned_load_preflight_attempted": True,
        "stage_owned_load_preflight_verified": False,
        "stage_owned_load_verified": False,
        "activation_handoff_verified": False,
        "one_token_decode_verified": False,
        "multitoken_decode_verified": False,
        "gpu_tpu_cpu_same_request_verified": False,
        "public_artifact_safe": True,
        "blockers": [],
        "diagnosis_codes": [],
    }
    try:
        config = fetch_hf_json(model_id, "config.json", timeout_seconds=timeout_seconds)
        index = fetch_hf_json(model_id, "model.safetensors.index.json", timeout_seconds=timeout_seconds)
        weight_map = {
            str(key): Path(str(value)).name
            for key, value in dict(index.get("weight_map") or {}).items()
            if str(key or "").strip() and str(value or "").strip()
        }
        layers = _int(config.get("num_hidden_layers") or config.get("n_layer"))
        normalized_count = normalize_stage_count(stage_count, layer_count=layers)
        header_cache: dict[str, tuple[int, dict[str, Any]]] = {}
        stages = [
            build_stage_plan(
                model_repo=model_id,
                config=config,
                weight_index=index,
                stage_id=stage_id,
                stage_count=normalized_count,
                backend=backend_for_stage(stage_id, normalized_count, stage_backends),
                header_cache=header_cache,
                timeout_seconds=timeout_seconds,
                max_header_bytes=max_header_bytes,
                fetch_headers=fetch_headers,
            )
            for stage_id in range(normalized_count)
        ]
        stage_bytes = [int(stage.get("planned_logical_tensor_bytes") or 0) for stage in stages]
        report.update(
            {
                "model_type": str(config.get("model_type") or ""),
                "architectures": list(config.get("architectures") or []),
                "weight_format": classify_weight_format(config, weight_map, candidate["quantization"]),
                "num_hidden_layers": layers,
                "hidden_size": _int(config.get("hidden_size") or config.get("n_embd")),
                "num_attention_heads": _int(config.get("num_attention_heads") or config.get("n_head")),
                "num_key_value_heads": _int(config.get("num_key_value_heads")),
                "num_experts": _int(config.get("num_experts") or config.get("num_local_experts")),
                "vocab_size": _int(config.get("vocab_size")),
                "weight_key_count": len(weight_map),
                "all_weight_file_count": len({value for value in weight_map.values()}),
                "total_size_bytes": _int(_dict(index.get("metadata")).get("total_size")),
                "total_size_gb": round(_int(_dict(index.get("metadata")).get("total_size")) / 1024 / 1024 / 1024, 6),
                "topology": {
                    "schema": "heterogeneous_capacity_topology_v1",
                    "name": "single_kaggle_account_gpu_tpu_cpu_capacity_ladder",
                    "stage_count": normalized_count,
                    "stage_backends": [str(stage.get("backend") or "") for stage in stages],
                    "cuda_stage_count": sum(1 for stage in stages if stage.get("backend") == "cuda"),
                    "jax_tpu_stage_count": sum(1 for stage in stages if stage.get("backend") == "jax_tpu"),
                    "cpu_stage_count": sum(1 for stage in stages if stage.get("backend") == "cpu"),
                },
                "stage_plans": stages,
                "max_stage_planned_logical_tensor_gb": round(max(stage_bytes or [0]) / 1024 / 1024 / 1024, 6),
                "total_planned_logical_tensor_gb": round(sum(stage_bytes) / 1024 / 1024 / 1024, 6),
                "header_file_count_fetched": len(header_cache),
                "stage_owned_load_preflight_verified": bool(stages and all(stage.get("stage_owned_header_verified") for stage in stages)),
            }
        )
        if report["stage_owned_load_preflight_verified"]:
            report["diagnosis_codes"].append("stage_owned_load_preflight_verified")
        else:
            report["blockers"].append("stage_owned_load_preflight_not_verified")
        if candidate["target"] == "decode":
            report["blockers"].append("fresh_larger_than_32b_decode_not_yet_verified")
            if "awq" in report["weight_format"] or "gptq" in report["weight_format"]:
                report["blockers"].append("quantized_jax_tpu_runtime_adapter_missing")
            if "full_precision" in report["weight_format"]:
                report["blockers"].append("full_precision_larger_than_32b_live_resource_budget_unverified")
        else:
            report["blockers"].append("fresh_larger_than_32b_stage_owned_load_not_yet_verified")
        report["blocked_reason"] = str(report["blockers"][0]) if report["blockers"] else ""
    except Exception as exc:
        report.update(
            {
                "blocked_reason": "candidate_metadata_preflight_failed",
                "blockers": ["candidate_metadata_preflight_failed"],
                "error_type": type(exc).__name__,
                "error_digest": sha_payload(str(exc)),
                "diagnosis_codes": ["candidate_metadata_preflight_failed"],
            }
        )
    return report


def summarize_larger_stage_load_report(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    model = _dict(report.get("model"))
    lifecycle = _dict(report.get("kaggle_lifecycle"))
    expected = _dict(report.get("expected_plan"))
    stage_summaries = [item for item in _list(report.get("stage_summaries")) if isinstance(item, dict)]
    return {
        "schema": "heterogeneous_capacity_larger_stage_load_import_v1",
        "source": source_summary(path, report, kind="larger_stage_owned_load_report"),
        "model_id": str(model.get("repo") or report.get("model_repo") or expected.get("model_repo") or ""),
        "stage_owned_load_verified": bool(
            report.get("ok") is True
            and report.get("stage_owned_quantized_32b_loading_ready") is True
            and report.get("coverage_ready") is True
            and report.get("all_stage_reports_downloaded") is True
            and report.get("all_stage_owned_loading_ready") is True
            and report.get("loads_only_stage_weight_keys_ready") is True
            and lifecycle.get("kernels_deleted") is True
            and lifecycle.get("private_packages_removed") is True
        ),
        "stage_count": _int(_dict(report.get("runtime")).get("stage_count") or expected.get("stage_count")),
        "stage_report_count": len(stage_summaries),
        "covered_weight_key_count": _int(expected.get("covered_weight_key_count")),
        "weight_key_count": _int(expected.get("weight_key_count")),
        "actual_push_count": _int(lifecycle.get("actual_push_count")),
        "kernels_deleted": lifecycle.get("kernels_deleted") is True,
        "private_packages_removed": lifecycle.get("private_packages_removed") is True,
        "max_loaded_tensor_gb": max((_float(item.get("loaded_tensor_gb")) for item in stage_summaries), default=0.0),
        "total_loaded_tensor_gb": round(sum(_float(item.get("loaded_tensor_gb")) for item in stage_summaries), 6),
        "stage_loaded_tensor_gb": [_float(item.get("loaded_tensor_gb")) for item in stage_summaries],
        "public_artifact_safe": True,
    }


def summarize_partial_stage_load_report(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "heterogeneous_capacity_partial_stage_load_import_v1",
        "source": source_summary(path, report, kind="partial_stage_owned_load_report"),
        "model_id": str(report.get("model_repo") or ""),
        "partial_stage_load_verified": bool(
            report.get("ok") is True
            and report.get("stage_owned_quantized_32b_loading_ready") is True
            and report.get("loads_only_stage_weight_keys") is True
            and report.get("cross_stage_weight_keys_loaded") is False
            and report.get("stage_weight_downloads_only_stage_files") is True
            and _dict(report.get("temp_cleanup")).get("ok") is True
        ),
        "stage_id": _int(report.get("stage_id")),
        "stage_count": _int(report.get("stage_count")),
        "stage_layer_range": list(report.get("stage_layer_range") or []),
        "assigned_weight_key_count": _int(report.get("assigned_weight_key_count")),
        "loaded_weight_key_count": _int(report.get("loaded_weight_key_count")),
        "loaded_tensor_gb": _float(report.get("loaded_tensor_gb")),
        "materialized_tensor_gb": _float(report.get("materialized_tensor_gb")),
        "temp_cleanup_ok": _dict(report.get("temp_cleanup")).get("ok") is True,
        "public_artifact_safe": True,
    }


def apply_larger_stage_load_import(
    candidates: list[dict[str, Any]],
    *,
    path: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    imported = summarize_larger_stage_load_report(path, report)
    if not imported.get("stage_owned_load_verified"):
        return imported
    model_id = str(imported.get("model_id") or "")
    for candidate in candidates:
        if str(candidate.get("model_id") or "") != model_id:
            continue
        candidate["stage_owned_load_verified"] = True
        candidate["stage_owned_load_live_import"] = imported
        candidate["live_stage_owned_load_proof_path"] = str(path)
        candidate["blockers"] = [
            blocker
            for blocker in _list(candidate.get("blockers"))
            if blocker != "fresh_larger_than_32b_stage_owned_load_not_yet_verified"
        ]
        if candidate.get("target") == "stage_load":
            candidate["blocked_reason"] = str(candidate["blockers"][0]) if candidate.get("blockers") else ""
        candidate["diagnosis_codes"] = sorted(
            set(str(item) for item in _list(candidate.get("diagnosis_codes")))
            | {"larger_stage_owned_load_verified"}
        )
        break
    return imported


def apply_partial_stage_load_import(
    candidates: list[dict[str, Any]],
    *,
    path: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    imported = summarize_partial_stage_load_report(path, report)
    if not imported.get("partial_stage_load_verified"):
        return imported
    model_id = str(imported.get("model_id") or "")
    for candidate in candidates:
        if str(candidate.get("model_id") or "") != model_id:
            continue
        candidate["partial_stage_owned_load_verified"] = True
        candidate["partial_stage_owned_load_live_import"] = imported
        candidate["partial_live_stage_owned_load_proof_path"] = str(path)
        candidate["diagnosis_codes"] = sorted(
            set(str(item) for item in _list(candidate.get("diagnosis_codes")))
            | {"partial_stage_owned_load_verified"}
        )
        break
    return imported


def source_summary(path: Path, report: dict[str, Any], *, kind: str) -> dict[str, Any]:
    safety = _dict(report.get("safety"))
    return {
        "kind": kind,
        "path": str(path),
        "present": path.is_file(),
        "schema": str(report.get("schema") or ""),
        "ok": report.get("ok") is True,
        "sha256": sha256_file(path) if path.is_file() else "",
        "public_artifact_safe": bool(
            report.get("public_artifact_safe") is True
            or safety.get("public_artifact_safe") is True
            or (
                report.get("raw_prompt_public") is False
                and report.get("raw_generated_text_public") is False
                and report.get("generated_token_ids_public") is False
                and report.get("activation_public") is False
                and report.get("credentials_public") is False
            )
        ),
    }


def build_32b_baseline(bridge_path: Path, serving_path: Path) -> dict[str, Any]:
    bridge = load_optional_json(bridge_path)
    serving = load_optional_json(serving_path)
    stage_counts = _dict(bridge.get("stage_task_counts"))
    backends = {str(item) for item in _list(bridge.get("accepted_stage_backends"))}
    generated = _int(bridge.get("generated_token_count") or serving.get("generated_token_count"))
    target = _int(bridge.get("target_generated_token_count") or serving.get("target_generated_token_count"), 1)
    verified = bool(
        bridge.get("schema") == "gpu_tpu_cpu_same_request_runtime_bridge_probe_v1"
        and bridge.get("ok") is True
        and bridge.get("same_request_runtime_bridge_verified") is True
        and bridge.get("gpu_tpu_cpu_32b_same_request_verified") is True
        and bridge.get("same_request_32b_model_verified") is True
        and generated >= max(1, target)
        and {"cuda", "jax_tpu", "cpu"}.issubset(backends)
        and all(_int(stage_counts.get(name)) >= max(1, target) for name in ["stage0", "stage1", "stage2"])
    )
    return {
        "schema": "heterogeneous_capacity_32b_baseline_v1",
        "model_id": str(bridge.get("target_model_id") or serving.get("target_model_id") or "Qwen/Qwen2.5-32B-Instruct"),
        "parameter_class": "32b",
        "quantization": "none",
        "weight_format": "full_precision_safetensors",
        "source_bridge": source_summary(bridge_path, bridge, kind="gpu_tpu_cpu_same_request_runtime_bridge"),
        "source_serving": source_summary(serving_path, serving, kind="heterogeneous_32b_serving"),
        "stage_owned_load_verified": verified,
        "activation_handoff_verified": verified,
        "one_token_decode_verified": bool(verified and generated >= 1),
        "multitoken_decode_verified": bool(verified and generated >= 2),
        "gpu_tpu_cpu_same_request_verified": verified,
        "generated_token_count": generated,
        "target_generated_token_count": target,
        "accepted_stage_backends": sorted(backends),
        "stage_task_counts": stage_counts,
        "best_successful_topology": {
            "stage_count": 3,
            "stage_backends": ["cuda", "jax_tpu", "cpu"],
            "source": "retained_r6_same_request_4token_live_proof",
        },
        "blocked_reason": "" if verified else "retained_32b_baseline_not_verified",
        "blockers": [] if verified else ["retained_32b_baseline_not_verified"],
        "public_artifact_safe": True,
    }


def highest_parameter_class(items: list[dict[str, Any]], predicate: str) -> str:
    matching = [item for item in items if item.get(predicate) is True]
    if not matching:
        return ""
    return str(max(matching, key=lambda item: parameter_class_value(str(item.get("parameter_class") or ""))).get("parameter_class") or "")


def largest_failed_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [item for item in candidates if item.get("blocked_reason")]
    if not failed:
        return {}
    item = max(failed, key=lambda candidate: parameter_class_value(str(candidate.get("parameter_class") or "")))
    return {
        "parameter_class": item.get("parameter_class"),
        "model_id": item.get("model_id"),
        "quantization": item.get("quantization"),
        "blocked_reason": item.get("blocked_reason"),
        "blockers": item.get("blockers") or [],
        "stage_owned_load_preflight_verified": item.get("stage_owned_load_preflight_verified") is True,
    }


def build_support_bundle(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SUPPORT_BUNDLE_SCHEMA,
        "ok": report.get("ok") is True,
        "generated_at": report.get("generated_at"),
        "report_schema": report.get("schema"),
        "conclusions": report.get("conclusions"),
        "candidate_count": len(_list(report.get("candidates"))),
        "diagnosis_codes": report.get("diagnosis_codes") or [],
        "blockers": report.get("blockers") or [],
        "public_artifact_safe": True,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_backends = [item.strip() for item in str(args.stage_backends).split(",") if item.strip()]
    baseline = build_32b_baseline(Path(args.baseline_32b_bridge_report), Path(args.baseline_32b_serving_report))
    candidates = [
        build_candidate(
            parse_candidate(raw),
            stage_count=args.stage_count,
            stage_backends=stage_backends,
            timeout_seconds=args.hf_timeout_seconds,
            max_header_bytes=args.max_header_bytes,
            fetch_headers=not args.skip_safetensors_headers,
        )
        for raw in (args.candidate or list(DEFAULT_CANDIDATES))
    ]
    stage_load_imports: list[dict[str, Any]] = []
    for raw_path in args.larger_stage_owned_load_report or []:
        path = Path(raw_path)
        imported_report = load_optional_json(path) if str(raw_path or "").strip() else {}
        if imported_report:
            stage_load_imports.append(apply_larger_stage_load_import(candidates, path=path, report=imported_report))
    partial_stage_load_imports: list[dict[str, Any]] = []
    for raw_path in args.partial_stage_owned_load_report or []:
        path = Path(raw_path)
        imported_report = load_optional_json(path) if str(raw_path or "").strip() else {}
        if imported_report:
            partial_stage_load_imports.append(apply_partial_stage_load_import(candidates, path=path, report=imported_report))
    all_items = [baseline, *candidates]
    max_stage_load = highest_parameter_class(all_items, "stage_owned_load_verified")
    max_partial_stage_load = highest_parameter_class(candidates, "partial_stage_owned_load_verified")
    max_stage_preflight = highest_parameter_class(candidates, "stage_owned_load_preflight_verified")
    max_one = highest_parameter_class(all_items, "one_token_decode_verified")
    max_multi = highest_parameter_class(all_items, "multitoken_decode_verified")
    max_same = highest_parameter_class(all_items, "gpu_tpu_cpu_same_request_verified")
    largest_failed = largest_failed_candidate(candidates)
    next_bottlenecks = sorted(
        {
            blocker
            for candidate in candidates
            for blocker in _list(candidate.get("blockers"))
            if blocker
        }
    )
    diagnosis = [
        "retained_32b_gpu_tpu_cpu_baseline_verified" if baseline.get("gpu_tpu_cpu_same_request_verified") else "retained_32b_gpu_tpu_cpu_baseline_missing",
        "capacity_ladder_72b_candidate_attempted" if any(str(item.get("parameter_class") or "").startswith("72b") for item in candidates) else "capacity_ladder_72b_candidate_missing",
        "capacity_ladder_100b_plus_candidate_attempted" if any(parameter_class_value(str(item.get("parameter_class") or "")) >= 100 for item in candidates) else "capacity_ladder_100b_plus_candidate_missing",
        "stage_owned_load_preflight_ready" if max_stage_preflight else "stage_owned_load_preflight_missing",
        "larger_than_32b_decode_not_verified" if max_same == "32b" else "larger_than_32b_decode_verified",
    ]
    blockers: list[str] = []
    if not baseline.get("gpu_tpu_cpu_same_request_verified"):
        blockers.append("retained_32b_baseline_not_verified")
    if max_same == "32b":
        blockers.append("larger_than_32b_same_request_decode_not_verified")
    if not max_stage_preflight:
        blockers.append("larger_candidate_stage_owned_preflight_missing")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "heterogeneous_capacity_frontier_ready": True,
        "execution_mode": args.execution_mode,
        "output_dir": str(output_dir),
        "baseline_32b": baseline,
        "candidates": candidates,
        "larger_stage_load_imports": stage_load_imports,
        "partial_stage_load_imports": partial_stage_load_imports,
        "conclusions": {
            "max_stage_owned_load_parameter_class": max_stage_load,
            "max_partial_stage_owned_load_parameter_class": max_partial_stage_load,
            "max_stage_owned_load_preflight_parameter_class": max_stage_preflight,
            "max_1token_decode_parameter_class": max_one,
            "max_multitoken_decode_parameter_class": max_multi,
            "max_gpu_tpu_cpu_same_request_parameter_class": max_same,
            "best_successful_topology": baseline.get("best_successful_topology") if baseline.get("gpu_tpu_cpu_same_request_verified") else {},
            "largest_failed_candidate": largest_failed,
            "next_bottleneck": next_bottlenecks,
            "larger_than_32b_decode_verified": bool(max_same and parameter_class_value(max_same) >= 70),
            "capacity_frontier_validation_complete": True,
            "stop_condition": (
                "larger_than_32b_same_request_decode_verified"
                if max_same and parameter_class_value(max_same) >= 70
                else "bounded_capacity_ladder_completed_with_structured_blockers"
            ),
        },
        "resource_bounds": {
            "single_account_policy": "respect_kaggle_limits_no_multi_account_bypass",
            "planned_stage_backends": stage_backends,
            "planned_stage_count": int(args.stage_count),
            "gpu_quota_observed_public": args.gpu_quota_observed,
            "tpu_quota_observed_public": args.tpu_quota_observed,
            "provider_queue_and_runtime_allocation_risk": True,
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
            "credentials_public": False,
            "cookies_public": False,
            "jupyter_proxy_token_public": False,
            "private_runtime_state_public": False,
            "private_kaggle_payload_public": False,
            "weight_tensor_values_public": False,
        },
        "public_artifact_safe": True,
        "diagnosis_codes": diagnosis,
        "blockers": sorted(set(blockers)),
        "limitations": [
            "This report distinguishes retained 32B live decode success from larger-model metadata/header preflight.",
            "A larger candidate is not counted as stage-owned load verified unless a real live loader report is imported.",
            "A larger candidate is not counted as 1-token decode verified without a same-request GPU+TPU+CPU live proof.",
            "Kaggle kernels and Web TPU runtimes are temporary proof vehicles, not production scheduling infrastructure.",
        ],
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["heterogeneous_capacity_frontier_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"].append("public_redaction_scan_failed")
        report["diagnosis_codes"].append("public_redaction_scan_failed")
        report["redaction_errors"] = leaks
    summary_path = output_dir / "heterogeneous_capacity_frontier.json"
    support_path = output_dir / "support_bundle.json"
    write_json(summary_path, report)
    support = build_support_bundle(report)
    write_json(support_path, support)
    report["artifacts"] = {
        "summary_json": artifact_entry(summary_path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(report.get("ok"))),
        "support_bundle_json": artifact_entry(support_path, output_dir, kind="support_bundle_json", schema=SUPPORT_BUNDLE_SCHEMA, ok=bool(support.get("ok"))),
    }
    write_json(summary_path, report)
    write_json(support_path, build_support_bundle(report))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GPU+TPU+CPU heterogeneous capacity frontier evidence.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execution-mode", choices=EXECUTION_MODES, default="metadata-preflight")
    parser.add_argument("--baseline-32b-bridge-report", default=DEFAULT_32B_BRIDGE_REPORT)
    parser.add_argument("--baseline-32b-serving-report", default=DEFAULT_32B_SERVING_REPORT)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--larger-stage-owned-load-report", action="append", default=[DEFAULT_72B_STAGE_LOAD_REPORT])
    parser.add_argument("--partial-stage-owned-load-report", action="append", default=[DEFAULT_100B_PARTIAL_STAGE_REPORT])
    parser.add_argument("--stage-count", type=int, default=len(DEFAULT_STAGE_BACKENDS))
    parser.add_argument("--stage-backends", default=",".join(DEFAULT_STAGE_BACKENDS))
    parser.add_argument("--hf-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--max-header-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--skip-safetensors-headers", action="store_true")
    parser.add_argument("--gpu-quota-observed", default="")
    parser.add_argument("--tpu-quota-observed", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.stage_count < 3:
        raise SystemExit("--stage-count must be at least 3")
    if args.stage_count > 32:
        raise SystemExit("--stage-count must be <= 32")
    if args.max_header_bytes < 1024 or args.max_header_bytes > 1024 * 1024 * 1024:
        raise SystemExit("--max-header-bytes must be between 1KiB and 1GiB")
    backends = [item.strip() for item in str(args.stage_backends).split(",") if item.strip()]
    if not {"cuda", "jax_tpu", "cpu"}.issubset(set(backends)):
        raise SystemExit("--stage-backends must include cuda,jax_tpu,cpu")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        conclusions = _dict(report.get("conclusions"))
        print(f"Report: {Path(args.output_dir) / 'heterogeneous_capacity_frontier.json'}")
        print(f"Ready: {report.get('heterogeneous_capacity_frontier_ready')}")
        print(f"Max same-request decode: {conclusions.get('max_gpu_tpu_cpu_same_request_parameter_class') or 'none'}")
        print(f"Max stage-owned load preflight: {conclusions.get('max_stage_owned_load_preflight_parameter_class') or 'none'}")
        if report.get("blockers"):
            print("Blockers: " + ", ".join(str(item) for item in report.get("blockers") or []))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
