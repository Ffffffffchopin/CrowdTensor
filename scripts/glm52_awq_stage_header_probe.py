#!/usr/bin/env python3
"""Probe GLM 5.2 AWQ stage-owned safetensors headers without tensor download."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "glm52_awq_stage_header_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-awq-stage-header-probe"
DEFAULT_MODEL_REPO = "cyankiwi/GLM-5.2-AWQ-INT4"
BASE_MODEL_ID = "zai-org/GLM-5.2"
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Authorization:",
    "Cookie:",
    "Set-Cookie",
    "token=",
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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


def fetch_url_bytes(url: str, *, timeout_seconds: float, headers: dict[str, str] | None = None, read_limit: int | None = None) -> bytes:
    request_headers = {
        "User-Agent": "crowdtensor-glm52-awq-stage-header-probe/1",
        **(headers or {}),
    }
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token and "Authorization" not in request_headers:
        request_headers["Authorization"] = "Bearer " + str(hf_token)
    request = urllib.request.Request(
        url,
        headers=request_headers,
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        if read_limit is None:
            return response.read()
        return response.read(read_limit)


def fetch_hf_json(repo: str, filename: str, *, timeout_seconds: float) -> dict[str, Any]:
    quoted = urllib.parse.quote(filename)
    raw = fetch_url_bytes(
        f"https://huggingface.co/{repo}/resolve/main/{quoted}",
        timeout_seconds=timeout_seconds,
    )
    loaded = json.loads(raw.decode("utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def read_hf_range(repo: str, filename: str, start: int, end: int, *, timeout_seconds: float, max_bytes: int) -> bytes:
    quoted = urllib.parse.quote(filename)
    raw = fetch_url_bytes(
        f"https://huggingface.co/{repo}/resolve/main/{quoted}",
        timeout_seconds=timeout_seconds,
        headers={"Range": f"bytes={int(start)}-{int(end)}"},
        read_limit=int(max_bytes) + 1,
    )
    if len(raw) > int(max_bytes):
        raise RuntimeError("hf_range_response_exceeded_budget")
    return raw


def load_safetensors_header(repo: str, filename: str, *, timeout_seconds: float, max_header_bytes: int) -> dict[str, Any]:
    prefix = read_hf_range(repo, filename, 0, 7, timeout_seconds=timeout_seconds, max_bytes=8)
    if len(prefix) != 8:
        raise RuntimeError("safetensors_header_prefix_missing")
    header_len = struct.unpack("<Q", prefix)[0]
    if header_len <= 0 or header_len > int(max_header_bytes):
        raise RuntimeError("safetensors_header_length_out_of_budget")
    raw_header = read_hf_range(
        repo,
        filename,
        8,
        8 + int(header_len) - 1,
        timeout_seconds=timeout_seconds,
        max_bytes=int(header_len),
    )
    if len(raw_header) != int(header_len):
        raise RuntimeError("safetensors_header_truncated")
    loaded = json.loads(raw_header.decode("utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError("safetensors_header_not_object")
    return loaded


def normalize_stage_count(stage_count: int, *, layer_count: int) -> int:
    count = max(1, min(int(stage_count), max(1, int(layer_count) or int(stage_count))))
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
    for stage_id in range(count):
        width = base + (1 if stage_id < remainder else 0)
        start = cursor
        end = min(layers, start + width)
        ranges.append((start, end))
        cursor = end
    return ranges


def stage_prefixes(*, stage_id: int, stage_count: int, layer_range: tuple[int, int]) -> list[str]:
    start, end = layer_range
    prefixes = [f"model.layers.{layer_id}." for layer_id in range(int(start), int(end))]
    if int(stage_id) == 0:
        prefixes = ["model.embed_tokens.", *prefixes]
    if int(stage_id) == int(stage_count) - 1:
        prefixes = [*prefixes, "model.norm.", "lm_head."]
    return prefixes


def build_stage_selection(config: dict[str, Any], index: dict[str, Any], *, stage_id: int, stage_count: int) -> dict[str, Any]:
    weight_map = {
        str(key): Path(str(value)).name
        for key, value in _dict(index.get("weight_map")).items()
        if str(key or "").strip() and str(value or "").strip()
    }
    layer_count = _int(config.get("num_hidden_layers"))
    count = normalize_stage_count(stage_count, layer_count=layer_count)
    if int(stage_id) < 0 or int(stage_id) >= count:
        raise ValueError(f"stage_id must be between 0 and {count - 1}")
    layer_range = stage_layer_ranges(layer_count, count)[int(stage_id)]
    prefixes = stage_prefixes(stage_id=int(stage_id), stage_count=count, layer_range=layer_range)
    assigned = sorted(key for key in weight_map if any(key.startswith(prefix) for prefix in prefixes))
    assigned_files = sorted({weight_map[key] for key in assigned if weight_map.get(key)})
    return {
        "model_type": str(config.get("model_type") or ""),
        "architectures": [str(item) for item in _list(config.get("architectures"))],
        "num_hidden_layers": layer_count,
        "hidden_size": _int(config.get("hidden_size")),
        "stage_id": int(stage_id),
        "stage_count": count,
        "stage_layer_range": [int(layer_range[0]), int(layer_range[1])],
        "expected_key_prefixes": prefixes,
        "assigned_weight_keys": assigned,
        "assigned_weight_key_count": len(assigned),
        "assigned_weight_files": assigned_files,
        "assigned_weight_file_count": len(assigned_files),
        "all_weight_file_count": len(set(weight_map.values())),
        "weight_key_count": len(weight_map),
        "total_size_bytes": _int(_dict(index.get("metadata")).get("total_size")),
        "weight_map": weight_map,
    }


def tensor_nbytes(header_item: dict[str, Any]) -> int:
    offsets = _list(header_item.get("data_offsets"))
    if len(offsets) != 2:
        return 0
    return max(0, _int(offsets[1]) - _int(offsets[0]))


def family_hits(keys: list[str]) -> dict[str, bool]:
    joined = "\n".join(keys)
    return {
        "embeddings": "model.embed_tokens." in joined,
        "attention": bool(re.search(r"self_attn|attention|q_proj|k_proj|v_proj|o_proj", joined)),
        "mlp_or_moe": bool(re.search(r"mlp|moe|experts|gate", joined)),
        "awq_quantized_tensors": bool(
            re.search(r"qweight|qzeros|scales|g_idx|weight_packed|weight_scale|weight_shape|weight_zero_point", joined)
        ),
        "norms": ".norm" in joined or "layernorm" in joined.lower(),
        "lm_head": "lm_head." in joined,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    header_errors: list[dict[str, Any]] = []
    try:
        config = fetch_hf_json(args.model_repo, "config.json", timeout_seconds=float(args.hf_timeout_seconds))
        index = fetch_hf_json(args.model_repo, "model.safetensors.index.json", timeout_seconds=float(args.hf_timeout_seconds))
        selection = build_stage_selection(
            config,
            index,
            stage_id=int(args.stage_id),
            stage_count=int(args.stage_count),
        )
    except Exception as exc:
        config = {}
        index = {}
        selection = {
            "assigned_weight_keys": [],
            "assigned_weight_files": [],
            "assigned_weight_key_count": 0,
            "assigned_weight_file_count": 0,
            "stage_layer_range": [],
        }
        blockers.append("glm52_awq_source_metadata_not_ready")
        header_errors.append({"phase": "source_metadata", "error_type": type(exc).__name__, "error_digest": sha_payload(str(exc))})

    assigned_keys = [str(key) for key in selection.get("assigned_weight_keys") or []]
    weight_map = _dict(selection.get("weight_map"))
    assigned_files = [str(item) for item in selection.get("assigned_weight_files") or []]
    dtype_counts: Counter[str] = Counter()
    shape_rank_counts: Counter[str] = Counter()
    present_keys: set[str] = set()
    header_file_summaries: list[dict[str, Any]] = []
    total_tensor_storage_bytes = 0
    max_files = max(1, int(args.max_header_files))
    for filename in assigned_files[:max_files]:
        try:
            header = load_safetensors_header(
                args.model_repo,
                filename,
                timeout_seconds=float(args.hf_timeout_seconds),
                max_header_bytes=int(args.max_header_bytes),
            )
        except Exception as exc:
            header_errors.append({"filename": filename, "error_type": type(exc).__name__, "error_digest": sha_payload(str(exc))})
            continue
        file_keys = [key for key in assigned_keys if weight_map.get(key) == filename]
        file_present = [key for key in file_keys if key in header]
        present_keys.update(file_present)
        file_storage = 0
        for key in file_present:
            item = _dict(header.get(key))
            dtype_counts[str(item.get("dtype") or "unknown")] += 1
            shape = _list(item.get("shape"))
            shape_rank_counts[str(len(shape))] += 1
            nbytes = tensor_nbytes(item)
            file_storage += nbytes
            total_tensor_storage_bytes += nbytes
        header_file_summaries.append(
            {
                "filename": filename,
                "stage_key_count_in_file": len(file_keys),
                "present_stage_key_count": len(file_present),
                "header_tensor_count": len([key for key in header if key != "__metadata__"]),
                "stage_tensor_storage_bytes": file_storage,
                "stage_key_digest": sha_payload(sorted(file_present)),
            }
        )
    missing_keys = sorted(set(assigned_keys) - present_keys)
    if not assigned_keys:
        blockers.append("glm52_awq_stage_key_selection_empty")
    if header_errors:
        blockers.append("glm52_awq_safetensors_header_fetch_errors")
    if missing_keys:
        blockers.append("glm52_awq_stage_header_missing_keys")
    if len(assigned_files) > max_files:
        blockers.append("glm52_awq_stage_header_file_limit_reached")
    ready = bool(assigned_keys and present_keys and not missing_keys and not header_errors and len(assigned_files) <= max_files)
    if ready:
        diagnosis = ["glm52_awq_stage_header_ready", "glm52_awq_stage_owned_key_shape_ready"]
    else:
        diagnosis = ["glm52_awq_stage_header_not_ready"]

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "glm52_awq_stage_header_ready": ready,
        "model_repo": str(args.model_repo),
        "base_model_id": BASE_MODEL_ID,
        "model_type": str(selection.get("model_type") or ""),
        "architectures": [str(item) for item in _list(selection.get("architectures"))],
        "quantization": "AWQ-INT4",
        "stage_id": int(args.stage_id),
        "stage_count": int(selection.get("stage_count") or args.stage_count),
        "stage_layer_range": selection.get("stage_layer_range") or [],
        "assigned_weight_key_count": int(selection.get("assigned_weight_key_count") or 0),
        "assigned_weight_file_count": int(selection.get("assigned_weight_file_count") or 0),
        "assigned_weight_file_sample": assigned_files[:24],
        "header_file_count": len(header_file_summaries),
        "header_file_limit": max_files,
        "present_stage_key_count": len(present_keys),
        "missing_stage_key_count": len(missing_keys),
        "missing_stage_key_digest": sha_payload(missing_keys[:200]),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "shape_rank_counts": dict(sorted(shape_rank_counts.items())),
        "stage_family_hits": family_hits(assigned_keys),
        "total_selected_tensor_storage_bytes": int(total_tensor_storage_bytes),
        "total_selected_tensor_storage_gb": round(total_tensor_storage_bytes / 1_000_000_000, 6),
        "total_model_size_bytes": int(selection.get("total_size_bytes") or 0),
        "total_model_size_gb": round(int(selection.get("total_size_bytes") or 0) / 1_000_000_000, 6),
        "header_file_summaries": header_file_summaries,
        "header_errors": header_errors,
        "weight_tensor_values_loaded": False,
        "weight_tensor_values_public": False,
        "safetensors_header_payload_public": False,
        "stage_runtime_adapter_verified": False,
        "same_request_route_verified": False,
        "diagnosis_codes": diagnosis,
        "blockers": sorted(set(blockers)),
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "signed_url_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
            "activation_public": False,
            "generated_token_ids_public": False,
        },
        "public_artifact_safe": True,
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["glm52_awq_stage_header_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["redaction_errors"] = leaks
        report["blockers"].append("public_redaction_scan_failed")
    write_json(output_dir / "glm52_awq_stage_header_probe.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--stage-id", type=int, default=1)
    parser.add_argument("--stage-count", type=int, default=12)
    parser.add_argument("--max-header-files", type=int, default=24)
    parser.add_argument("--max-header-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--hf-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.stage_id < 0:
        raise SystemExit("--stage-id must be non-negative")
    if args.stage_count <= 0:
        raise SystemExit("--stage-count must be positive")
    if args.max_header_files <= 0 or args.max_header_files > 128:
        raise SystemExit("--max-header-files must be between 1 and 128")
    if args.max_header_bytes < 1024:
        raise SystemExit("--max-header-bytes must be at least 1024")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "glm52_awq_stage_header_probe: "
            f"ok={bool(report.get('ok'))} stage={report.get('stage_id')} "
            f"keys={report.get('present_stage_key_count')}/{report.get('assigned_weight_key_count')} "
            f"blockers={report.get('blockers')}"
        )
    return 0 if report.get("public_artifact_safe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
