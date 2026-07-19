#!/usr/bin/env python3
"""Run a public-safe GLM 5.2 pack-quantized dequant + linear slice probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


SCHEMA = "glm52_pack_quantized_dequant_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-pack-quantized-dequant-probe"
DEFAULT_MODEL_REPO = "cyankiwi/GLM-5.2-AWQ-INT4"
MODEL_ID = "zai-org/GLM-5.2"
PACK_FIELDS = ["weight_packed", "weight_scale", "weight_zero_point", "weight_shape"]
DEFAULT_HF_FETCH_ATTEMPTS = 8
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


def sha_tensor(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    if tensor.dtype is torch.bfloat16:
        tensor = tensor.to(torch.float32)
    return sha_bytes(tensor.numpy().tobytes())


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


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def hf_fetch_attempts(default: int = DEFAULT_HF_FETCH_ATTEMPTS) -> int:
    return max(1, _env_int("CT_GLM52_HF_FETCH_ATTEMPTS", int(default)))


def _retry_delay(attempt: int) -> float:
    return min(8.0, 0.5 * (2**attempt))


def fetch_url_bytes(
    url: str,
    *,
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
    read_limit: int | None = None,
    attempts: int | None = None,
) -> bytes:
    last_error: Exception | None = None
    attempt_count = hf_fetch_attempts() if attempts is None else max(1, int(attempts))
    for attempt in range(attempt_count):
        request_headers = {"User-Agent": "crowdtensor-glm52-pack-dequant-probe/1", **(headers or {})}
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token and "Authorization" not in request_headers:
            request_headers["Authorization"] = "Bearer " + str(hf_token)
        request = urllib.request.Request(
            url,
            headers=request_headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read() if read_limit is None else response.read(read_limit)
            if raw:
                return raw
            last_error = RuntimeError("empty_http_response")
        except Exception as exc:  # pragma: no cover - concrete urllib subclasses vary across runtimes.
            last_error = exc
        if attempt + 1 < attempt_count:
            time.sleep(_retry_delay(attempt))
    raise last_error if last_error is not None else RuntimeError("fetch_url_bytes_failed")


def fetch_hf_json(repo: str, filename: str, *, timeout_seconds: float) -> dict[str, Any]:
    quoted = urllib.parse.quote(filename)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            raw = fetch_url_bytes(f"https://huggingface.co/{repo}/resolve/main/{quoted}", timeout_seconds=timeout_seconds)
            loaded = json.loads(raw.decode("utf-8"))
            return loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt + 1 < 3:
                time.sleep(_retry_delay(attempt))
        except Exception:
            raise
    raise RuntimeError("hf_json_decode_failed") from last_error


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
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            raw_header = read_hf_range(
                repo,
                filename,
                8,
                8 + int(header_len) - 1,
                timeout_seconds=timeout_seconds,
                max_bytes=int(header_len),
            )
            if len(raw_header) != int(header_len):
                raise RuntimeError("safetensors_header_read_size_mismatch")
            loaded = json.loads(raw_header.decode("utf-8"))
            return int(header_len), loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError as exc:
            last_error = exc
        except RuntimeError as exc:
            last_error = exc
        if attempt + 1 < 3:
            time.sleep(_retry_delay(attempt))
    raise RuntimeError("safetensors_header_decode_failed") from last_error


def tensor_nbytes(item: dict[str, Any]) -> int:
    offsets = _list(item.get("data_offsets"))
    if len(offsets) != 2:
        return 0
    return max(0, _int(offsets[1]) - _int(offsets[0]))


def target_keys(layer_id: int, expert_id: int, projection: str) -> dict[str, str]:
    prefix = f"model.layers.{int(layer_id)}.mlp.experts.{int(expert_id)}.{projection}"
    return {field: f"{prefix}.{field}" for field in PACK_FIELDS}


def torch_dtype(dtype: str) -> torch.dtype:
    mapping = {"I32": torch.int32, "I64": torch.int64, "BF16": torch.bfloat16, "F32": torch.float32}
    if dtype not in mapping:
        raise RuntimeError(f"unsupported_dtype:{dtype}")
    return mapping[dtype]


def load_tensor(repo: str, filename: str, header_len: int, item: dict[str, Any], args: argparse.Namespace) -> torch.Tensor:
    offsets = _list(item.get("data_offsets"))
    if len(offsets) != 2:
        raise RuntimeError("tensor_offsets_missing")
    nbytes = tensor_nbytes(item)
    absolute_start = 8 + int(header_len) + _int(offsets[0])
    absolute_end = 8 + int(header_len) + _int(offsets[1]) - 1
    raw = read_hf_range(
        repo,
        filename,
        absolute_start,
        absolute_end,
        timeout_seconds=float(args.hf_timeout_seconds),
        max_bytes=int(args.max_tensor_bytes),
    )
    if len(raw) != nbytes:
        raise RuntimeError("tensor_value_read_size_mismatch")
    shape = [int(item) for item in _list(item.get("shape"))]
    return torch.frombuffer(bytearray(raw), dtype=torch_dtype(str(item.get("dtype") or ""))).reshape(shape).clone()


def unpack_from_int32(value: torch.Tensor, num_bits: int, shape: tuple[int, int], *, packed_dim: int = 1) -> torch.Tensor:
    if value.dtype is not torch.int32:
        raise RuntimeError("expected_int32_packed_tensor")
    pack_factor = 32 // int(num_bits)
    mask = (1 << int(num_bits)) - 1
    if packed_dim == 1:
        unpacked = torch.zeros((value.shape[0], value.shape[1] * pack_factor), dtype=torch.int32)
        for index in range(pack_factor):
            unpacked[:, index::pack_factor] = (value >> (int(num_bits) * index)) & mask
        unpacked = unpacked[:, : int(shape[1])]
    else:
        unpacked = torch.zeros((value.shape[0] * pack_factor, value.shape[1]), dtype=torch.int32)
        for index in range(pack_factor):
            unpacked[index::pack_factor, :] = (value >> (int(num_bits) * index)) & mask
        unpacked = unpacked[: int(shape[0]), :]
    offset = 1 << (int(num_bits) - 1)
    return (unpacked - offset).to(torch.int8)


def dequantize_group_slice(
    *,
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    weight_shape: torch.Tensor,
    row_count: int,
    group_count: int,
    num_bits: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    original_shape = tuple(int(item) for item in weight_shape.tolist())
    if len(original_shape) != 2:
        raise RuntimeError("weight_shape_not_2d")
    input_cols = int(original_shape[1])
    groups = int(scale.shape[1])
    group_size = input_cols // groups
    if input_cols % groups != 0:
        raise RuntimeError("group_size_not_integral")
    rows = min(int(row_count), int(original_shape[0]))
    selected_groups = min(int(group_count), groups)
    selected_cols = selected_groups * group_size
    pack_factor = 32 // int(num_bits)
    packed_cols = (selected_cols + pack_factor - 1) // pack_factor
    q = unpack_from_int32(packed[:rows, :packed_cols].contiguous(), num_bits, (rows, selected_cols), packed_dim=1).to(torch.float32)
    zp_unpacked = unpack_from_int32(
        zero_point[: ((rows + (32 // int(num_bits)) - 1) // (32 // int(num_bits))), :selected_groups].contiguous(),
        num_bits,
        (rows, selected_groups),
        packed_dim=0,
    ).to(torch.float32)
    scale_slice = scale[:rows, :selected_groups].to(torch.float32)
    dequant = (q.reshape(rows, selected_groups, group_size) - zp_unpacked.unsqueeze(-1)) * scale_slice.unsqueeze(-1)
    return q, zp_unpacked, dequant.reshape(rows, selected_cols)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    blockers: list[str] = []
    errors: list[dict[str, Any]] = []
    tensors: dict[str, torch.Tensor] = {}
    tensor_summaries: list[dict[str, Any]] = []
    keys = target_keys(args.layer_id, args.expert_id, args.projection)
    try:
        config = fetch_hf_json(args.model_repo, "config.json", timeout_seconds=float(args.hf_timeout_seconds))
        index = fetch_hf_json(args.model_repo, "model.safetensors.index.json", timeout_seconds=float(args.hf_timeout_seconds))
        weight_map = _dict(index.get("weight_map"))
        headers_by_file: dict[str, tuple[int, dict[str, Any]]] = {}
        for field, key in keys.items():
            filename = str(weight_map.get(key) or "")
            if not filename:
                raise RuntimeError(f"field_weight_key_missing:{field}")
            if filename not in headers_by_file:
                headers_by_file[filename] = load_safetensors_header_with_len(
                    args.model_repo,
                    filename,
                    timeout_seconds=float(args.hf_timeout_seconds),
                    max_header_bytes=int(args.max_header_bytes),
                )
            header_len, header = headers_by_file[filename]
            item = _dict(header.get(key))
            tensor = load_tensor(args.model_repo, filename, header_len, item, args)
            tensors[field] = tensor
            tensor_summaries.append(
                {
                    "field": field,
                    "filename": filename,
                    "dtype": str(item.get("dtype") or ""),
                    "shape": [int(item) for item in tensor.shape],
                    "tensor_nbytes": tensor_nbytes(item),
                    "value_sha256": sha_tensor(tensor),
                    "value_loaded": True,
                }
            )
    except Exception as exc:
        config = {}
        errors.append({"phase": "load_group", "error_type": type(exc).__name__, "error_digest": sha_payload(str(exc))})
        blockers.append("glm52_pack_quantized_group_load_failed")

    dequant_ready = False
    linear_ready = False
    dequant_hash = ""
    linear_hash = ""
    q_hash = ""
    zp_hash = ""
    dequant_shape: list[int] = []
    linear_shape: list[int] = []
    if not errors:
        try:
            q, zp, dequant = dequantize_group_slice(
                packed=tensors["weight_packed"],
                scale=tensors["weight_scale"],
                zero_point=tensors["weight_zero_point"],
                weight_shape=tensors["weight_shape"],
                row_count=int(args.row_count),
                group_count=int(args.group_count),
            )
            dequant_ready = True
            dequant_shape = [int(item) for item in dequant.shape]
            q_hash = sha_tensor(q)
            zp_hash = sha_tensor(zp)
            dequant_hash = sha_tensor(dequant.to(torch.float32))
            input_vec = torch.linspace(-0.25, 0.25, steps=dequant.shape[1], dtype=torch.float32)
            output = torch.matmul(dequant.to(torch.float32), input_vec)
            linear_ready = True
            linear_shape = [int(item) for item in output.shape]
            linear_hash = sha_tensor(output)
        except Exception as exc:
            errors.append({"phase": "dequant_linear", "error_type": type(exc).__name__, "error_digest": sha_payload(str(exc))})
            blockers.append("glm52_pack_quantized_dequant_failed")

    if dequant_ready:
        blockers.append("glm52_pack_quantized_dequant_slice_is_not_full_layer")
    if linear_ready:
        blockers.append("glm52_pack_quantized_linear_slice_is_not_stage_decode")
    blockers.append("glm52_stage_decode_not_verified")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": dequant_ready and linear_ready,
        "glm52_pack_quantized_dequant_probe_ready": dequant_ready and linear_ready,
        "model_id": MODEL_ID,
        "model_repo": str(args.model_repo),
        "model_type": str(config.get("model_type") or ""),
        "quantization_format": str(_dict(config.get("quantization_config")).get("format") or ""),
        "layer_id": int(args.layer_id),
        "expert_id": int(args.expert_id),
        "projection": str(args.projection),
        "row_count": int(args.row_count),
        "group_count": int(args.group_count),
        "pack_quantized_group_loaded": set(tensors) == set(PACK_FIELDS),
        "pack_quantized_dequant_verified": dequant_ready,
        "pack_quantized_linear_slice_verified": linear_ready,
        "stage_decode_verified": False,
        "q_unpacked_hash": q_hash,
        "zero_point_unpacked_hash": zp_hash,
        "dequant_slice_shape": dequant_shape,
        "dequant_slice_hash": dequant_hash,
        "linear_slice_shape": linear_shape,
        "linear_slice_hash": linear_hash,
        "tensor_summaries": tensor_summaries,
        "errors": errors,
        "blockers": sorted(set(blockers)),
        "completion_boundary": {
            "dequant_slice_is_not_full_layer": True,
            "linear_slice_is_not_stage_decode": True,
            "weight_values_not_public": True,
            "requires_full_projection_runtime": True,
            "requires_transformer_block_runtime": True,
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
        report["glm52_pack_quantized_dequant_probe_ready"] = False
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
    parser.add_argument("--row-count", type=int, default=4)
    parser.add_argument("--group-count", type=int, default=2)
    parser.add_argument("--max-header-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-tensor-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--hf-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.row_count <= 0 or args.group_count <= 0:
        raise SystemExit("--row-count and --group-count must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_report(args)
    path = output_dir / "glm52_pack_quantized_dequant_probe.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"Dequant verified: {report.get('pack_quantized_dequant_verified')}")
    return 0 if report.get("public_artifact_safe") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
