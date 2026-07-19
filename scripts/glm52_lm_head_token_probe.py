#!/usr/bin/env python3
"""Run a public-safe GLM 5.2 full-vocab lm_head token-selection probe."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_attention_projection_probe as projection_probe  # noqa: E402
from scripts import glm52_pack_quantized_dequant_probe as dequant_probe  # noqa: E402
from scripts import glm52_pack_quantized_router_gather_probe as router_probe  # noqa: E402


SCHEMA = "glm52_lm_head_token_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-lm-head-token-probe"
DEFAULT_MODEL_REPO = dequant_probe.DEFAULT_MODEL_REPO
MODEL_ID = dequant_probe.MODEL_ID
SENSITIVE_FRAGMENTS = dequant_probe.SENSITIVE_FRAGMENTS + (
    '"logits":',
    '"token_id":',
    '"token_ids":',
    '"selected_token":',
    '"top_token":',
    '"generated_token_ids":',
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def bf16_row_bytes(width: int) -> int:
    return int(width) * 2


def build_hidden(hidden_size: int) -> torch.Tensor:
    return torch.linspace(-0.04, 0.04, steps=int(hidden_size), dtype=torch.float32)


def _tensor_item_hash(value: int | float, *, dtype: torch.dtype) -> str:
    return dequant_probe.sha_tensor(torch.tensor([value], dtype=dtype))


def stream_lm_head_topk(
    args: argparse.Namespace,
    *,
    model_repo: str,
    filename: str,
    header_len: int,
    item: dict[str, Any],
    hidden: torch.Tensor,
    top_k: int,
) -> dict[str, Any]:
    shape = _list(item.get("shape"))
    if len(shape) != 2:
        raise RuntimeError("lm_head_shape_rank_invalid")
    vocab_size = int(shape[0])
    hidden_size = int(shape[1])
    if int(hidden.shape[0]) != hidden_size:
        raise RuntimeError("lm_head_hidden_width_mismatch")
    if str(item.get("dtype") or "") != "BF16":
        raise RuntimeError("lm_head_dtype_not_bf16")
    offsets = _list(item.get("data_offsets"))
    if len(offsets) != 2:
        raise RuntimeError("lm_head_offsets_missing")
    data_start = int(offsets[0])
    data_end = int(offsets[1])
    expected_nbytes = vocab_size * hidden_size * 2
    if data_end - data_start != expected_nbytes:
        raise RuntimeError("lm_head_nbytes_mismatch")

    top_values = torch.empty(0, dtype=torch.float32)
    top_indices = torch.empty(0, dtype=torch.int64)
    rows_scanned = 0
    block_count = 0
    row_block_size = int(args.row_block_size)
    if row_block_size <= 0:
        raise RuntimeError("row_block_size_invalid")
    for row_start in range(0, vocab_size, row_block_size):
        rows = min(row_block_size, vocab_size - row_start)
        byte_count = rows * bf16_row_bytes(hidden_size)
        if byte_count > int(args.max_block_bytes):
            raise RuntimeError("lm_head_block_exceeds_budget")
        absolute_start = 8 + int(header_len) + data_start + row_start * bf16_row_bytes(hidden_size)
        absolute_end = absolute_start + byte_count - 1
        raw = dequant_probe.read_hf_range(
            model_repo,
            filename,
            absolute_start,
            absolute_end,
            timeout_seconds=float(args.hf_timeout_seconds),
            max_bytes=byte_count,
        )
        if len(raw) != byte_count:
            raise RuntimeError("lm_head_block_range_size_mismatch")
        block = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(rows, hidden_size)
        values = torch.matmul(block.to(torch.float32), hidden.to(torch.float32))
        block_top_count = min(int(top_k), int(values.numel()))
        block_values, block_indices = torch.topk(values, k=block_top_count)
        block_indices = block_indices.to(torch.int64) + int(row_start)
        combined_values = torch.cat([top_values, block_values.to(torch.float32)])
        combined_indices = torch.cat([top_indices, block_indices.to(torch.int64)])
        keep_count = min(int(top_k), int(combined_values.numel()))
        keep_values, keep_positions = torch.topk(combined_values, k=keep_count)
        top_values = keep_values.to(torch.float32)
        top_indices = combined_indices[keep_positions].to(torch.int64)
        rows_scanned += rows
        block_count += 1
    return {
        "lm_head_shape": [vocab_size, hidden_size],
        "lm_head_dtype": "BF16",
        "lm_head_nbytes": expected_nbytes,
        "lm_head_file_count": 1,
        "lm_head_rows_scanned": rows_scanned,
        "lm_head_block_count": block_count,
        "lm_head_row_block_size": row_block_size,
        "top_k": int(top_k),
        "top_k_count": int(top_indices.numel()),
        "selected_token_id_hash": dequant_probe.sha_tensor(top_indices[:1].to(torch.int64)),
        "selected_logit_hash": dequant_probe.sha_tensor(top_values[:1].to(torch.float32)),
        "top_token_ids_hash": dequant_probe.sha_tensor(top_indices.to(torch.int64)),
        "top_logits_hash": dequant_probe.sha_tensor(top_values.to(torch.float32)),
    }


def run_lm_head_token_selection(args: argparse.Namespace) -> dict[str, Any]:
    config = dequant_probe.fetch_hf_json(args.model_repo, "config.json", timeout_seconds=float(args.hf_timeout_seconds))
    index = dequant_probe.fetch_hf_json(args.model_repo, "model.safetensors.index.json", timeout_seconds=float(args.hf_timeout_seconds))
    weight_map = _dict(index.get("weight_map"))
    hidden_size = _int(config.get("hidden_size"))
    vocab_size = _int(config.get("vocab_size"))
    if hidden_size <= 0 or vocab_size <= 0:
        raise RuntimeError("config_shape_missing")
    norm_weight = router_probe.load_dense_tensor(args, "model.norm.weight")
    hidden = build_hidden(hidden_size)
    normalized_hidden = projection_probe.rms_norm(hidden, norm_weight, float(config.get("rms_norm_eps") or 1e-6))
    lm_head_file = str(weight_map.get("lm_head.weight") or "")
    if not lm_head_file:
        raise RuntimeError("lm_head_weight_missing")
    header_len, header = dequant_probe.load_safetensors_header_with_len(
        args.model_repo,
        lm_head_file,
        timeout_seconds=float(args.hf_timeout_seconds),
        max_header_bytes=int(args.max_header_bytes),
    )
    lm_head_item = _dict(header.get("lm_head.weight"))
    streamed = stream_lm_head_topk(
        args,
        model_repo=str(args.model_repo),
        filename=lm_head_file,
        header_len=header_len,
        item=lm_head_item,
        hidden=normalized_hidden,
        top_k=int(args.top_k),
    )
    return {
        "model_type": str(config.get("model_type") or ""),
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "tie_word_embeddings": bool(config.get("tie_word_embeddings")),
        "norm_weight_shape": [int(item) for item in norm_weight.shape],
        "hidden_source": "deterministic_probe_vector",
        "hidden_shape": [int(item) for item in hidden.shape],
        "normalized_hidden_shape": [int(item) for item in normalized_hidden.shape],
        "hidden_hash": dequant_probe.sha_tensor(hidden),
        "normalized_hidden_hash": dequant_probe.sha_tensor(normalized_hidden),
        **streamed,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    blockers: list[str] = []
    errors: list[dict[str, Any]] = []
    lm_head: dict[str, Any] = {}
    ready = False
    try:
        lm_head = run_lm_head_token_selection(args)
        ready = (
            lm_head.get("model_type") == "glm_moe_dsa"
            and _list(lm_head.get("norm_weight_shape")) == [_int(lm_head.get("hidden_size"))]
            and _list(lm_head.get("hidden_shape")) == [_int(lm_head.get("hidden_size"))]
            and _list(lm_head.get("normalized_hidden_shape")) == [_int(lm_head.get("hidden_size"))]
            and _list(lm_head.get("lm_head_shape")) == [_int(lm_head.get("vocab_size")), _int(lm_head.get("hidden_size"))]
            and _int(lm_head.get("lm_head_rows_scanned")) == _int(lm_head.get("vocab_size"))
            and _int(lm_head.get("top_k_count")) == int(args.top_k)
        )
    except Exception as exc:
        errors.append({"phase": "lm_head_token_selection", "error_type": type(exc).__name__, "error_digest": dequant_probe.sha_payload(str(exc))})
        blockers.append("glm52_lm_head_token_selection_failed")
    if ready:
        blockers.extend(
            [
                "glm52_lm_head_token_selection_uses_probe_hidden_not_full_model_hidden",
                "glm52_lm_head_token_selection_is_not_stage_decode",
                "glm52_lm_head_token_selection_is_not_same_request",
            ]
        )
    blockers.extend(["glm52_stage_decode_not_verified", "glm52_same_request_decode_not_verified"])
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "glm52_lm_head_token_probe_ready": ready,
        "model_id": MODEL_ID,
        "model_repo": str(args.model_repo),
        **lm_head,
        "final_norm_verified": ready,
        "lm_head_streamed_full_vocab": ready,
        "lm_head_logits_token_selection_verified": ready,
        "selected_token_hash_verified": ready,
        "full_model_hidden_verified": False,
        "generated_token_verified": False,
        "stage_decode_verified": False,
        "same_request_decode_verified": False,
        "errors": errors,
        "blockers": sorted(set(blockers)),
        "completion_boundary": {
            "lm_head_token_selection_uses_probe_hidden_not_full_model_hidden": True,
            "lm_head_token_selection_is_not_stage_decode": True,
            "lm_head_token_selection_is_not_same_request": True,
            "requires_full_model_or_stage_hidden": True,
            "requires_stage_decode_verified": True,
            "requires_kaggle_cpu_gpu_tpu_same_request": True,
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
        report["glm52_lm_head_token_probe_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"] = sorted(set([*blockers, "public_redaction_scan_failed"]))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--row-block-size", type=int, default=2048)
    parser.add_argument("--max-header-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-tensor-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-block-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--hf-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    if args.row_block_size <= 0:
        raise SystemExit("--row-block-size must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_report(args)
    path = output_dir / "glm52_lm_head_token_probe.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"LM head token selection verified: {report.get('lm_head_logits_token_selection_verified')}")
    return 0 if report.get("public_artifact_safe") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
