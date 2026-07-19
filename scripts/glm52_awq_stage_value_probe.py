#!/usr/bin/env python3
"""Probe one GLM 5.2 AWQ stage-owned tensor value without publishing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_awq_stage_header_probe as header_probe  # noqa: E402


SCHEMA = "glm52_awq_stage_value_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-awq-stage-value-probe"
DEFAULT_MODEL_REPO = header_probe.DEFAULT_MODEL_REPO
BASE_MODEL_ID = header_probe.BASE_MODEL_ID
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
    '"weight_tensor_values":',
    '"safetensors_header_payload":',
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


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


def load_safetensors_header_with_len(
    repo: str,
    filename: str,
    *,
    timeout_seconds: float,
    max_header_bytes: int,
) -> tuple[int, dict[str, Any]]:
    prefix = header_probe.read_hf_range(repo, filename, 0, 7, timeout_seconds=timeout_seconds, max_bytes=8)
    if len(prefix) != 8:
        raise RuntimeError("safetensors_header_prefix_missing")
    header_len = struct.unpack("<Q", prefix)[0]
    if header_len <= 0 or header_len > int(max_header_bytes):
        raise RuntimeError("safetensors_header_length_out_of_budget")
    raw_header = header_probe.read_hf_range(
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
    return int(header_len), loaded


def tensor_nbytes(item: dict[str, Any]) -> int:
    return header_probe.tensor_nbytes(item)


def tensor_offsets(item: dict[str, Any]) -> tuple[int, int]:
    offsets = _list(item.get("data_offsets"))
    if len(offsets) != 2:
        return (0, 0)
    return (_int(offsets[0]), _int(offsets[1]))


def choose_tensor(
    *,
    assigned_keys: list[str],
    weight_map: dict[str, Any],
    headers_by_file: dict[str, tuple[int, dict[str, Any]]],
    max_tensor_bytes: int,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for key in assigned_keys:
        filename = str(weight_map.get(key) or "")
        header_tuple = headers_by_file.get(filename)
        if not header_tuple:
            continue
        _header_len, header = header_tuple
        item = _dict(header.get(key))
        nbytes = tensor_nbytes(item)
        if nbytes <= 0 or nbytes > int(max_tensor_bytes):
            continue
        dtype = str(item.get("dtype") or "")
        shape = _list(item.get("shape"))
        priority = 0
        if any(fragment in key for fragment in ["qzeros", "scales", "g_idx"]):
            priority -= 100
        if dtype in {"I32", "I64"}:
            priority -= 10
        candidates.append(
            {
                "key": key,
                "filename": filename,
                "dtype": dtype,
                "shape": shape,
                "tensor_nbytes": nbytes,
                "priority": priority,
            }
        )
    if not candidates:
        return {}
    candidates.sort(key=lambda item: (int(item["priority"]), int(item["tensor_nbytes"]), str(item["key"])))
    return candidates[0]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    errors: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    value_hash = ""
    value_byte_count = 0
    try:
        config = header_probe.fetch_hf_json(args.model_repo, "config.json", timeout_seconds=float(args.hf_timeout_seconds))
        index = header_probe.fetch_hf_json(args.model_repo, "model.safetensors.index.json", timeout_seconds=float(args.hf_timeout_seconds))
        selection = header_probe.build_stage_selection(
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
            "weight_map": {},
        }
        blockers.append("glm52_awq_source_metadata_not_ready")
        errors.append({"phase": "source_metadata", "error_type": type(exc).__name__, "error_digest": sha_payload(str(exc))})

    assigned_keys = [str(key) for key in selection.get("assigned_weight_keys") or []]
    assigned_files = [str(item) for item in selection.get("assigned_weight_files") or []]
    weight_map = _dict(selection.get("weight_map"))
    headers_by_file: dict[str, tuple[int, dict[str, Any]]] = {}
    for filename in assigned_files[: max(1, int(args.max_header_files))]:
        try:
            headers_by_file[filename] = load_safetensors_header_with_len(
                args.model_repo,
                filename,
                timeout_seconds=float(args.hf_timeout_seconds),
                max_header_bytes=int(args.max_header_bytes),
            )
        except Exception as exc:
            errors.append({"filename": filename, "error_type": type(exc).__name__, "error_digest": sha_payload(str(exc))})
    if len(assigned_files) > int(args.max_header_files):
        blockers.append("glm52_awq_stage_value_probe_file_limit_reached")
    selected = choose_tensor(
        assigned_keys=assigned_keys,
        weight_map=weight_map,
        headers_by_file=headers_by_file,
        max_tensor_bytes=int(args.max_tensor_bytes),
    )
    if not selected:
        blockers.append("glm52_awq_stage_value_tensor_not_found_within_budget")
    else:
        filename = str(selected["filename"])
        header_len, header = headers_by_file[filename]
        item = _dict(header.get(str(selected["key"])))
        offset_start, offset_end = tensor_offsets(item)
        absolute_start = 8 + int(header_len) + int(offset_start)
        absolute_end = 8 + int(header_len) + int(offset_end) - 1
        try:
            raw_value = header_probe.read_hf_range(
                args.model_repo,
                filename,
                absolute_start,
                absolute_end,
                timeout_seconds=float(args.hf_timeout_seconds),
                max_bytes=int(args.max_tensor_bytes),
            )
            value_hash = sha_bytes(raw_value)
            value_byte_count = len(raw_value)
            selected.update(
                {
                    "absolute_byte_range_start": absolute_start,
                    "absolute_byte_range_end": absolute_end,
                    "data_offsets_digest": sha_payload([offset_start, offset_end]),
                    "shape_digest": sha_payload(selected.get("shape")),
                }
            )
        except Exception as exc:
            blockers.append("glm52_awq_stage_value_range_read_failed")
            errors.append({"phase": "value_range_read", "error_type": type(exc).__name__, "error_digest": sha_payload(str(exc))})

    ready = bool(selected and value_hash and value_byte_count == _int(selected.get("tensor_nbytes")))
    if not assigned_keys:
        blockers.append("glm52_awq_stage_key_selection_empty")
    if errors:
        blockers.append("glm52_awq_stage_value_probe_errors")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "glm52_awq_stage_value_probe_ready": ready,
        "model_repo": str(args.model_repo),
        "base_model_id": BASE_MODEL_ID,
        "model_type": str(selection.get("model_type") or ""),
        "quantization": "AWQ-INT4",
        "stage_id": int(args.stage_id),
        "stage_count": int(selection.get("stage_count") or args.stage_count),
        "stage_layer_range": selection.get("stage_layer_range") or [],
        "assigned_weight_key_count": int(selection.get("assigned_weight_key_count") or 0),
        "assigned_weight_file_count": int(selection.get("assigned_weight_file_count") or 0),
        "header_file_count": len(headers_by_file),
        "selected_tensor": {
            "key_digest": sha_payload(selected.get("key", "")),
            "filename": selected.get("filename", ""),
            "dtype": selected.get("dtype", ""),
            "shape_digest": selected.get("shape_digest", ""),
            "rank": len(_list(selected.get("shape"))),
            "tensor_nbytes": _int(selected.get("tensor_nbytes")),
            "data_offsets_digest": selected.get("data_offsets_digest", ""),
        },
        "weight_value_byte_count": value_byte_count,
        "weight_value_sha256": value_hash,
        "weight_tensor_values_loaded": ready,
        "weight_tensor_values_public": False,
        "safetensors_header_payload_public": False,
        "stage_runtime_adapter_verified": False,
        "same_request_route_verified": False,
        "same_request_decode_verified": False,
        "stage_smoke_only": True,
        "diagnosis_codes": [
            "glm52_awq_stage_owned_weight_value_loaded" if ready else "glm52_awq_stage_owned_weight_value_not_loaded",
            "glm52_stage_value_probe_is_not_same_request_success",
        ],
        "blockers": sorted(set(blockers)),
        "errors": errors,
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
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"] = sorted(set([*blockers, "public_redaction_scan_failed"]))
        report["redaction_errors"] = leaks
    write_json(output_dir / "glm52_awq_stage_value_probe.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--stage-id", type=int, default=4)
    parser.add_argument("--stage-count", type=int, default=12)
    parser.add_argument("--max-header-files", type=int, default=8)
    parser.add_argument("--max-header-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-tensor-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--hf-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_tensor_bytes <= 0:
        raise SystemExit("--max-tensor-bytes must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {Path(args.output_dir) / 'glm52_awq_stage_value_probe.json'}")
        print(f"Stage value probe ready: {report.get('glm52_awq_stage_value_probe_ready')}")
    return 0 if report.get("public_artifact_safe") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
