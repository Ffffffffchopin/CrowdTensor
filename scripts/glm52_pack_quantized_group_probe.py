#!/usr/bin/env python3
"""Load one real GLM 5.2 pack-quantized tensor group public-safely."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "glm52_pack_quantized_group_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-pack-quantized-group-probe"
DEFAULT_MODEL_REPO = "cyankiwi/GLM-5.2-AWQ-INT4"
MODEL_ID = "zai-org/GLM-5.2"
PACK_FIELDS = ["weight_packed", "weight_scale", "weight_zero_point", "weight_shape"]
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
    '"raw_prompt":',
    '"generated_text":',
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


def fetch_url_bytes(url: str, *, timeout_seconds: float, headers: dict[str, str] | None = None, read_limit: int | None = None) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "crowdtensor-glm52-pack-group-probe/1", **(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        if read_limit is None:
            return response.read()
        return response.read(read_limit)


def fetch_hf_json(repo: str, filename: str, *, timeout_seconds: float) -> dict[str, Any]:
    quoted = urllib.parse.quote(filename)
    raw = fetch_url_bytes(f"https://huggingface.co/{repo}/resolve/main/{quoted}", timeout_seconds=timeout_seconds)
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


def load_safetensors_header_with_len(repo: str, filename: str, *, timeout_seconds: float, max_header_bytes: int) -> tuple[int, dict[str, Any]]:
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
    loaded = json.loads(raw_header.decode("utf-8"))
    return int(header_len), loaded if isinstance(loaded, dict) else {}


def tensor_nbytes(item: dict[str, Any]) -> int:
    offsets = _list(item.get("data_offsets"))
    if len(offsets) != 2:
        return 0
    return max(0, _int(offsets[1]) - _int(offsets[0]))


def target_keys(layer_id: int, expert_id: int, projection: str) -> list[str]:
    prefix = f"model.layers.{int(layer_id)}.mlp.experts.{int(expert_id)}.{projection}"
    return [f"{prefix}.{field}" for field in PACK_FIELDS]


def load_tensor_value(repo: str, filename: str, header_len: int, item: dict[str, Any], args: argparse.Namespace) -> bytes:
    offsets = _list(item.get("data_offsets"))
    if len(offsets) != 2:
        raise RuntimeError("tensor_offsets_missing")
    absolute_start = 8 + int(header_len) + _int(offsets[0])
    absolute_end = 8 + int(header_len) + _int(offsets[1]) - 1
    return read_hf_range(
        repo,
        filename,
        absolute_start,
        absolute_end,
        timeout_seconds=float(args.hf_timeout_seconds),
        max_bytes=int(args.max_tensor_bytes),
    )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    blockers: list[str] = []
    errors: list[dict[str, Any]] = []
    tensor_summaries: list[dict[str, Any]] = []
    total_bytes = 0
    keys = target_keys(args.layer_id, args.expert_id, args.projection)
    try:
        config = fetch_hf_json(args.model_repo, "config.json", timeout_seconds=float(args.hf_timeout_seconds))
        index = fetch_hf_json(args.model_repo, "model.safetensors.index.json", timeout_seconds=float(args.hf_timeout_seconds))
    except Exception as exc:
        config = {}
        index = {}
        blockers.append("glm52_pack_group_source_metadata_not_ready")
        errors.append({"phase": "metadata", "error_type": type(exc).__name__, "error_digest": sha_payload(str(exc))})

    weight_map = _dict(index.get("weight_map"))
    headers_by_file: dict[str, tuple[int, dict[str, Any]]] = {}
    missing_keys = [key for key in keys if key not in weight_map]
    if missing_keys:
        blockers.append("glm52_pack_group_required_keys_missing")
    for key in keys:
        filename = str(weight_map.get(key) or "")
        if not filename:
            continue
        try:
            if filename not in headers_by_file:
                headers_by_file[filename] = load_safetensors_header_with_len(
                    args.model_repo,
                    filename,
                    timeout_seconds=float(args.hf_timeout_seconds),
                    max_header_bytes=int(args.max_header_bytes),
                )
            header_len, header = headers_by_file[filename]
            item = _dict(header.get(key))
            nbytes = tensor_nbytes(item)
            if nbytes <= 0:
                raise RuntimeError("tensor_nbytes_missing")
            if total_bytes + nbytes > int(args.max_total_bytes):
                raise RuntimeError("pack_group_total_byte_budget_exceeded")
            raw = load_tensor_value(args.model_repo, filename, header_len, item, args)
            if len(raw) != nbytes:
                raise RuntimeError("tensor_value_read_size_mismatch")
            total_bytes += len(raw)
            tensor_summaries.append(
                {
                    "key_digest": sha_payload(key),
                    "field": key.rsplit(".", 1)[-1],
                    "filename": filename,
                    "dtype": str(item.get("dtype") or ""),
                    "rank": len(_list(item.get("shape"))),
                    "shape_digest": sha_payload(_list(item.get("shape"))),
                    "tensor_nbytes": nbytes,
                    "value_sha256": sha_bytes(raw),
                    "value_loaded": True,
                }
            )
        except Exception as exc:
            errors.append({"key_digest": sha_payload(key), "error_type": type(exc).__name__, "error_digest": sha_payload(str(exc))})
    if errors:
        blockers.append("glm52_pack_group_tensor_load_errors")

    loaded_fields = {item["field"] for item in tensor_summaries if item.get("value_loaded") is True}
    group_loaded = bool(set(PACK_FIELDS).issubset(loaded_fields) and not missing_keys and not errors)
    if not group_loaded:
        blockers.append("glm52_pack_quantized_group_not_loaded")
    blockers.append("glm52_pack_quantized_group_is_not_dequantized")
    blockers.append("glm52_stage_decode_not_verified")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": group_loaded,
        "glm52_pack_quantized_group_probe_ready": group_loaded,
        "model_id": MODEL_ID,
        "model_repo": str(args.model_repo),
        "model_type": str(config.get("model_type") or ""),
        "quantization_format": str(_dict(config.get("quantization_config")).get("format") or ""),
        "layer_id": int(args.layer_id),
        "expert_id": int(args.expert_id),
        "projection": str(args.projection),
        "required_fields": PACK_FIELDS,
        "loaded_fields": sorted(loaded_fields),
        "group_tensor_count": len(tensor_summaries),
        "group_value_total_bytes": total_bytes,
        "group_value_hash": sha_payload([item.get("value_sha256") for item in tensor_summaries]),
        "pack_quantized_group_loaded": group_loaded,
        "pack_quantized_group_dequantized": False,
        "stage_decode_verified": False,
        "tensor_summaries": tensor_summaries,
        "missing_key_count": len(missing_keys),
        "missing_key_digest": sha_payload(missing_keys),
        "errors": errors,
        "blockers": sorted(set(blockers)),
        "completion_boundary": {
            "pack_group_load_is_not_dequant_success": True,
            "weight_value_hash_is_not_raw_value_publication": True,
            "requires_dequant_linear_runtime": True,
            "requires_stage_decode_verified": True,
        },
        "safety": {
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
        },
        "public_artifact_safe": True,
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["glm52_pack_quantized_group_probe_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"] = sorted(set([*blockers, "public_redaction_scan_failed"]))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--layer-id", type=int, default=3)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--projection", choices=["gate_proj", "up_proj", "down_proj"], default="gate_proj")
    parser.add_argument("--max-header-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-tensor-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--max-total-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--hf-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_report(args)
    path = output_dir / "glm52_pack_quantized_group_probe.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"Pack group loaded: {report.get('pack_quantized_group_loaded')}")
    return 0 if report.get("public_artifact_safe") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
