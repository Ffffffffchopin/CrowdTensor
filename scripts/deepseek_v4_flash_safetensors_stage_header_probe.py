#!/usr/bin/env python3
"""Probe DeepSeek-V4-Flash stage-selective safetensors header readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "deepseek_v4_flash_safetensors_stage_header_probe_v1"
SUPPORT_SCHEMA = "deepseek_v4_flash_safetensors_stage_header_support_v1"
DEFAULT_OUTPUT_DIR = "dist/deepseek-v4-flash-safetensors-stage-header-probe"
DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_LAYER_START = 16
DEFAULT_LAYER_END = 18
DEFAULT_MAX_HEADER_BYTES = 128 * 1024 * 1024
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Authorization:",
    "Cookie:",
    "Set-Cookie",
    "jupyter-proxy",
    "token=",
    "XSRF-TOKEN",
    "_xsrf",
    "kaggle_session",
    "runtime_proxy_token",
    "oauth_token",
    "X-Amz-Credential",
    "X-Amz-Signature",
    '"prompt":',
    '"raw_prompt":',
    '"generated_text":',
    '"generated_token_ids":',
    '"activation":',
    '"hidden_state":',
    '"logits":',
    '"kv_cache":',
    '"past_key_values":',
    '"tensor_values":',
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


def fetch_hf_json(model_id: str, filename: str, *, timeout_seconds: float = 120.0) -> dict[str, Any]:
    with urllib.request.urlopen(
        f"https://huggingface.co/{model_id}/resolve/main/{filename}",
        timeout=timeout_seconds,
    ) as response:
        loaded = json.load(response)
    return loaded if isinstance(loaded, dict) else {}


def safe_hf_filename(filename: str) -> bool:
    if not filename or filename.startswith(("/", "\\")):
        return False
    if "://" in filename or "\\" in filename:
        return False
    parts = [part for part in filename.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9._/\-]+", filename))


def read_hf_range(
    model_id: str,
    filename: str,
    start: int,
    end: int,
    *,
    timeout_seconds: float,
    max_bytes: int,
) -> bytes:
    if not safe_hf_filename(filename):
        raise RuntimeError("unsafe_hf_filename")
    url = f"https://huggingface.co/{model_id}/resolve/main/{filename}"
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={int(start)}-{int(end)}",
            "User-Agent": "crowdtensor-deepseek-v4-stage-header/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(getattr(response, "status", response.getcode()))
            content_range = response.headers.get("Content-Range", "")
            payload = response.read(int(max_bytes) + 1)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"hf_range_http_{exc.code}") from exc
    if status != 206 and not content_range.lower().startswith("bytes "):
        raise RuntimeError("hf_range_not_honored")
    if len(payload) > int(max_bytes):
        raise RuntimeError("hf_range_response_exceeded_budget")
    return payload


def parse_safetensors_header_from_bytes(blob: bytes, *, max_header_bytes: int) -> tuple[int, dict[str, Any]]:
    if len(blob) < 8:
        raise RuntimeError("safetensors_header_prefix_missing")
    header_len = int(struct.unpack("<Q", blob[:8])[0])
    if header_len <= 0 or header_len > int(max_header_bytes):
        raise RuntimeError("safetensors_header_length_out_of_budget")
    payload = blob[8 : 8 + header_len]
    if len(payload) != header_len:
        raise RuntimeError("safetensors_header_truncated")
    loaded = json.loads(payload.decode("utf-8"))
    return header_len, loaded if isinstance(loaded, dict) else {}


def load_safetensors_header(
    model_id: str,
    filename: str,
    *,
    timeout_seconds: float,
    max_header_bytes: int,
) -> tuple[int, dict[str, Any]]:
    prefix = read_hf_range(model_id, filename, 0, 7, timeout_seconds=timeout_seconds, max_bytes=8)
    if len(prefix) != 8:
        raise RuntimeError("safetensors_header_prefix_missing")
    header_len = int(struct.unpack("<Q", prefix)[0])
    if header_len <= 0 or header_len > int(max_header_bytes):
        raise RuntimeError("safetensors_header_length_out_of_budget")
    payload = read_hf_range(
        model_id,
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


def build_stage_plan(model_id: str, *, layer_start: int, layer_end: int, timeout_seconds: float) -> dict[str, Any]:
    config = fetch_hf_json(model_id, "config.json", timeout_seconds=timeout_seconds)
    index = fetch_hf_json(model_id, "model.safetensors.index.json", timeout_seconds=timeout_seconds)
    weight_map = _dict(index.get("weight_map"))
    layer_prefixes = [
        prefix
        for layer in range(int(layer_start), int(layer_end))
        for prefix in (f"layers.{layer}.", f"model.layers.{layer}.")
    ]
    selected_keys = sorted(key for key in weight_map if any(str(key).startswith(prefix) for prefix in layer_prefixes))
    selected_files = sorted({str(weight_map[key]) for key in selected_keys if weight_map.get(key)})
    expected_families = {
        "mla_attention": ["attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b"],
        "moe_router": ["ffn.gate"],
        "shared_experts": ["ffn.shared_experts"],
        "routed_experts": ["ffn.experts."],
        "hybrid_compression": ["hc_attn", "hc_ffn"],
        "norms": ["attn_norm", "ffn_norm"],
    }
    family_hits = {
        name: any(fragment in key for key in selected_keys for fragment in fragments)
        for name, fragments in expected_families.items()
    }
    return {
        "schema": "deepseek_v4_flash_safetensors_stage_plan_v1",
        "model_id": model_id,
        "metadata_ready": bool(config and weight_map),
        "stage_key_mapping_ready": bool(config and weight_map and selected_keys and all(family_hits.values())),
        "model_config": {
            "architectures": list(config.get("architectures") or [])[:8],
            "model_type": str(config.get("model_type") or ""),
            "num_hidden_layers": _int(config.get("num_hidden_layers")),
            "hidden_size": _int(config.get("hidden_size")),
            "num_attention_heads": _int(config.get("num_attention_heads")),
            "n_routed_experts": _int(config.get("n_routed_experts") or config.get("num_experts")),
            "num_experts_per_tok": _int(config.get("num_experts_per_tok")),
            "n_shared_experts": _int(config.get("n_shared_experts")),
            "q_lora_rank": _int(config.get("q_lora_rank")),
            "qk_rope_head_dim": _int(config.get("qk_rope_head_dim")),
            "moe_intermediate_size": _int(config.get("moe_intermediate_size")),
            "torch_dtype": str(config.get("torch_dtype") or ""),
            "quantization_config_present": isinstance(config.get("quantization_config"), dict),
            "config_payload_public": False,
        },
        "weight_index": {
            "weight_key_count": len(weight_map),
            "weight_file_count": len(set(weight_map.values())),
            "metadata_total_size_bytes": _int(_dict(index.get("metadata")).get("total_size")),
            "weight_map_payload_public": False,
        },
        "stage_mapping": {
            "layer_range": [int(layer_start), int(layer_end)],
            "selected_key_count": len(selected_keys),
            "selected_file_count": len(selected_files),
            "selected_key_digest": sha_payload(selected_keys),
            "selected_file_digest": sha_payload(selected_files),
            "family_hits": family_hits,
            "stage_weight_values_loaded": False,
            "stage_weight_values_public": False,
        },
        "selected_keys": selected_keys,
        "selected_files": selected_files,
        "weight_map": weight_map,
        "public_artifact_safe": True,
    }


def _tensor_storage_bytes(entry: dict[str, Any]) -> int:
    offsets = _list(entry.get("data_offsets"))
    if len(offsets) != 2:
        return 0
    start, end = _int(offsets[0]), _int(offsets[1])
    return max(0, end - start)


def summarize_selected_headers(
    plan: dict[str, Any],
    *,
    timeout_seconds: float,
    max_header_bytes: int,
) -> dict[str, Any]:
    selected_keys = [str(item) for item in _list(plan.get("selected_keys"))]
    selected_files = [str(item) for item in _list(plan.get("selected_files"))]
    weight_map = _dict(plan.get("weight_map"))
    model_id = str(plan.get("model_id") or DEFAULT_MODEL_ID)
    file_summaries: list[dict[str, Any]] = []
    missing_keys: list[str] = []
    fetch_errors: list[dict[str, Any]] = []
    dtype_counts: dict[str, int] = {}
    rank_counts: dict[str, int] = {}
    shape_signatures: list[dict[str, Any]] = []
    total_storage_bytes = 0

    for filename in selected_files:
        file_keys = [key for key in selected_keys if str(weight_map.get(key)) == filename]
        try:
            header_len, header = load_safetensors_header(
                model_id,
                filename,
                timeout_seconds=timeout_seconds,
                max_header_bytes=max_header_bytes,
            )
        except Exception as exc:  # noqa: BLE001
            fetch_errors.append(
                {
                    "file_name": filename,
                    "file_digest": sha_payload(filename),
                    "error_code": str(exc).splitlines()[0][:160],
                }
            )
            continue

        tensors = {key: value for key, value in header.items() if key != "__metadata__" and isinstance(value, dict)}
        file_dtype_counts: dict[str, int] = {}
        file_rank_counts: dict[str, int] = {}
        file_storage_bytes = 0
        file_missing: list[str] = []
        file_shape_signatures: list[dict[str, Any]] = []
        malformed_count = 0
        for key in file_keys:
            entry = _dict(tensors.get(key))
            if not entry:
                missing_keys.append(key)
                file_missing.append(key)
                continue
            dtype = str(entry.get("dtype") or "")
            shape = [_int(item) for item in _list(entry.get("shape"))]
            offsets = _list(entry.get("data_offsets"))
            if not dtype or not shape or len(offsets) != 2:
                malformed_count += 1
            file_dtype_counts[dtype] = file_dtype_counts.get(dtype, 0) + 1
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
            rank_key = str(len(shape))
            file_rank_counts[rank_key] = file_rank_counts.get(rank_key, 0) + 1
            rank_counts[rank_key] = rank_counts.get(rank_key, 0) + 1
            storage = _tensor_storage_bytes(entry)
            file_storage_bytes += storage
            total_storage_bytes += storage
            signature = {
                "key_digest": sha_payload(key),
                "dtype": dtype,
                "shape": shape,
                "storage_bytes": storage,
            }
            file_shape_signatures.append(signature)
            shape_signatures.append(signature)

        file_summaries.append(
            {
                "file_name": filename,
                "file_digest": sha_payload(filename),
                "header_length_bytes": int(header_len),
                "metadata_entry_present": "__metadata__" in header,
                "header_tensor_key_count": len(tensors),
                "selected_key_count": len(file_keys),
                "selected_key_digest": sha_payload(file_keys),
                "selected_shape_digest": sha_payload(file_shape_signatures),
                "selected_tensor_storage_bytes": file_storage_bytes,
                "dtype_counts": dict(sorted(file_dtype_counts.items())),
                "rank_counts": dict(sorted(file_rank_counts.items())),
                "missing_selected_key_count": len(file_missing),
                "missing_selected_key_digest": sha_payload(file_missing) if file_missing else "",
                "malformed_selected_key_count": malformed_count,
            }
        )

    return {
        "schema": "deepseek_v4_flash_safetensors_stage_header_summary_v1",
        "header_file_count": len(file_summaries),
        "selected_file_count": len(selected_files),
        "selected_key_count": len(selected_keys),
        "header_fetch_error_count": len(fetch_errors),
        "header_fetch_errors": fetch_errors,
        "missing_header_key_count": len(missing_keys),
        "missing_header_key_digest": sha_payload(missing_keys) if missing_keys else "",
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "rank_counts": dict(sorted(rank_counts.items())),
        "selected_shape_digest": sha_payload(shape_signatures),
        "total_selected_tensor_storage_bytes": total_storage_bytes,
        "file_summaries": file_summaries,
        "safetensors_header_payload_public": False,
        "real_weight_tensor_values_loaded": False,
        "real_weight_tensor_values_public": False,
        "stage_activation_payload_public": False,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    plan: dict[str, Any] = {}
    headers: dict[str, Any] = {
        "schema": "deepseek_v4_flash_safetensors_stage_header_summary_v1",
        "header_file_count": 0,
        "selected_file_count": 0,
        "selected_key_count": 0,
        "header_fetch_error_count": 0,
        "missing_header_key_count": 0,
        "dtype_counts": {},
        "rank_counts": {},
        "file_summaries": [],
        "safetensors_header_payload_public": False,
        "real_weight_tensor_values_loaded": False,
        "real_weight_tensor_values_public": False,
        "stage_activation_payload_public": False,
    }
    try:
        plan = build_stage_plan(
            args.model_id,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            timeout_seconds=float(args.timeout_seconds),
        )
        if plan.get("stage_key_mapping_ready") is not True:
            blockers.append("deepseek_v4_flash_stage_key_mapping_incomplete")
        else:
            headers = summarize_selected_headers(
                plan,
                timeout_seconds=float(args.timeout_seconds),
                max_header_bytes=int(args.max_header_bytes),
            )
            if _int(headers.get("header_fetch_error_count")):
                blockers.append("safetensors_range_header_fetch_failed")
            if _int(headers.get("missing_header_key_count")):
                blockers.append("safetensors_header_key_missing")
            if _int(headers.get("header_file_count")) != _int(headers.get("selected_file_count")):
                blockers.append("safetensors_header_file_count_mismatch")
    except Exception as exc:  # noqa: BLE001
        blockers.append(str(exc).splitlines()[0][:160] or "deepseek_v4_flash_safetensors_stage_header_probe_failed")

    stage_mapping = _dict(plan.get("stage_mapping"))
    family_hits = _dict(stage_mapping.get("family_hits"))
    header_ready = bool(
        not blockers
        and plan.get("stage_key_mapping_ready") is True
        and _int(headers.get("selected_key_count")) > 0
        and _int(headers.get("header_file_count")) == _int(headers.get("selected_file_count")) > 0
        and _int(headers.get("missing_header_key_count")) == 0
        and _int(headers.get("header_fetch_error_count")) == 0
        and bool(_dict(headers.get("dtype_counts")))
    )
    result = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "deepseek_v4_flash_safetensors_stage_header_probe_ready": True,
        "safetensors_header_ready": header_ready,
        "stage_header_shape_ready": header_ready,
        "model": {
            "model_id": args.model_id,
            "expected_model_id": DEFAULT_MODEL_ID,
            "architecture_class": "moe",
            "model_config": _dict(plan.get("model_config")),
        },
        "weight_index": _dict(plan.get("weight_index")),
        "stage_mapping": {
            "layer_range": _list(stage_mapping.get("layer_range")) or [int(args.layer_start), int(args.layer_end)],
            "selected_key_count": _int(stage_mapping.get("selected_key_count")),
            "selected_file_count": _int(stage_mapping.get("selected_file_count")),
            "selected_key_digest": str(stage_mapping.get("selected_key_digest") or ""),
            "selected_file_digest": str(stage_mapping.get("selected_file_digest") or ""),
            "family_hits": family_hits,
            "stage_weight_values_loaded": False,
            "stage_weight_values_public": False,
        },
        "headers": headers,
        "blockers": sorted(set(blockers)),
        "failure_stage": "" if header_ready else (sorted(set(blockers))[0] if blockers else "safetensors_header_not_ready"),
        "safety": {
            "public_artifact_safe": True,
            "safetensors_header_payload_public": False,
            "weight_index_payload_public": False,
            "weight_tensor_values_loaded": False,
            "weight_tensor_values_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "credentials_public": False,
            "cookies_public": False,
            "jupyter_proxy_token_public": False,
            "private_runtime_state_public": False,
        },
        "public_artifact_safe": True,
        "artifacts": {},
    }
    leaks = public_redaction_errors(result)
    if leaks:
        result["ok"] = False
        result["deepseek_v4_flash_safetensors_stage_header_probe_ready"] = False
        result["safetensors_header_ready"] = False
        result["stage_header_shape_ready"] = False
        result["public_artifact_safe"] = False
        result["safety"]["public_artifact_safe"] = False
        result["redaction_errors"] = leaks
        result["blockers"] = sorted(set([*result["blockers"], "public_redaction_scan_failed"]))
        result["failure_stage"] = "public_redaction_scan_failed"

    support = {
        "schema": SUPPORT_SCHEMA,
        "model_id": args.model_id,
        "layer_range": [int(args.layer_start), int(args.layer_end)],
        "selected_file_count": _int(stage_mapping.get("selected_file_count")),
        "selected_key_count": _int(stage_mapping.get("selected_key_count")),
        "selected_key_digest": str(stage_mapping.get("selected_key_digest") or ""),
        "selected_file_digest": str(stage_mapping.get("selected_file_digest") or ""),
        "safetensors_header_payload_public": False,
        "weight_tensor_values_public": False,
        "public_artifact_safe": True,
    }
    support_path = output_dir / "deepseek_v4_flash_safetensors_stage_header_support.json"
    summary_path = output_dir / "deepseek_v4_flash_safetensors_stage_header_probe.json"
    write_json(support_path, support)
    result["artifacts"] = {
        "summary_json": {"kind": "summary_json", "path": summary_path.name, "present": True, "schema": SCHEMA, "ok": bool(result.get("ok"))},
        "support_bundle_json": artifact_entry(support_path, output_dir, kind="support_bundle", schema=SUPPORT_SCHEMA, ok=True),
    }
    write_json(summary_path, result)
    result["artifacts"]["summary_json"] = artifact_entry(summary_path, output_dir, kind="summary_json", schema=SCHEMA, ok=bool(result.get("ok")))
    write_json(summary_path, result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--layer-start", type=int, default=DEFAULT_LAYER_START)
    parser.add_argument("--layer-end", type=int, default=DEFAULT_LAYER_END)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-header-bytes", type=int, default=DEFAULT_MAX_HEADER_BYTES)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {Path(args.output_dir) / 'deepseek_v4_flash_safetensors_stage_header_probe.json'}")
        print(f"Safetensors header ready: {report['safetensors_header_ready']}")
        if report.get("failure_stage"):
            print(f"Failure stage: {report['failure_stage']}")
    return 0 if report.get("public_artifact_safe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
