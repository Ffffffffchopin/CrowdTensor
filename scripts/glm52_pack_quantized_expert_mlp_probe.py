#!/usr/bin/env python3
"""Run a public-safe GLM 5.2 pack-quantized expert MLP probe."""

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

from scripts import glm52_pack_quantized_dequant_probe as dequant_probe


SCHEMA = "glm52_pack_quantized_expert_mlp_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-pack-quantized-expert-mlp-probe"
DEFAULT_MODEL_REPO = dequant_probe.DEFAULT_MODEL_REPO
MODEL_ID = dequant_probe.MODEL_ID
PROJECTIONS = ["gate_proj", "up_proj", "down_proj"]
SENSITIVE_FRAGMENTS = dequant_probe.SENSITIVE_FRAGMENTS + (
    '"expert_input":',
    '"expert_output":',
    '"mlp_intermediate":',
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


def load_projection_tensors(args: argparse.Namespace, projection: str) -> dict[str, torch.Tensor]:
    config = dequant_probe.fetch_hf_json(args.model_repo, "config.json", timeout_seconds=float(args.hf_timeout_seconds))
    index = dequant_probe.fetch_hf_json(args.model_repo, "model.safetensors.index.json", timeout_seconds=float(args.hf_timeout_seconds))
    weight_map = _dict(index.get("weight_map"))
    keys = dequant_probe.target_keys(args.layer_id, args.expert_id, projection)
    headers_by_file: dict[str, tuple[int, dict[str, Any]]] = {}
    tensors: dict[str, torch.Tensor] = {}
    for field, key in keys.items():
        filename = str(weight_map.get(key) or "")
        if not filename:
            raise RuntimeError(f"field_weight_key_missing:{projection}:{field}")
        if filename not in headers_by_file:
            headers_by_file[filename] = dequant_probe.load_safetensors_header_with_len(
                args.model_repo,
                filename,
                timeout_seconds=float(args.hf_timeout_seconds),
                max_header_bytes=int(args.max_header_bytes),
            )
        header_len, header = headers_by_file[filename]
        item = _dict(header.get(key))
        tensors[field] = dequant_probe.load_tensor(args.model_repo, filename, header_len, item, args)
    tensors["_model_type"] = torch.tensor([1 if config.get("model_type") == "glm_moe_dsa" else 0], dtype=torch.int64)
    return tensors


def projection_linear(tensors: dict[str, torch.Tensor], input_vec: torch.Tensor) -> tuple[torch.Tensor, list[int], list[int]]:
    _, _, weight = dequant_probe.dequantize_group_slice(
        packed=tensors["weight_packed"],
        scale=tensors["weight_scale"],
        zero_point=tensors["weight_zero_point"],
        weight_shape=tensors["weight_shape"],
        row_count=int(tensors["weight_shape"][0].item()),
        group_count=int(tensors["weight_scale"].shape[1]),
    )
    if weight.shape[1] != input_vec.shape[0]:
        raise RuntimeError(f"input_width_mismatch:{weight.shape[1]}:{input_vec.shape[0]}")
    output = torch.matmul(weight.to(torch.float32), input_vec.to(torch.float32))
    return output, [int(item) for item in weight.shape], [int(item) for item in output.shape]


def run_expert_mlp(args: argparse.Namespace) -> dict[str, Any]:
    config = dequant_probe.fetch_hf_json(args.model_repo, "config.json", timeout_seconds=float(args.hf_timeout_seconds))
    hidden_size = _int(config.get("hidden_size"))
    if hidden_size <= 0:
        raise RuntimeError("hidden_size_missing")
    input_vec = torch.linspace(-0.05, 0.05, steps=hidden_size, dtype=torch.float32)
    summaries: list[dict[str, Any]] = []
    outputs: dict[str, torch.Tensor] = {}
    for projection in PROJECTIONS:
        tensors = load_projection_tensors(args, projection)
        output_input = input_vec if projection != "down_proj" else torch.nn.functional.silu(outputs["gate_proj"]) * outputs["up_proj"]
        output, weight_shape, output_shape = projection_linear(tensors, output_input)
        outputs[projection] = output
        summaries.append(
            {
                "projection": projection,
                "weight_shape": weight_shape,
                "output_shape": output_shape,
                "output_hash": dequant_probe.sha_tensor(output),
                "pack_quantized_group_loaded": all(field in tensors for field in dequant_probe.PACK_FIELDS),
            }
        )
    final = outputs["down_proj"].to(torch.float32)
    return {
        "model_type": str(config.get("model_type") or ""),
        "hidden_size": hidden_size,
        "projection_summaries": summaries,
        "final_output_shape": [int(item) for item in final.shape],
        "final_output_hash": dequant_probe.sha_tensor(final),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    blockers: list[str] = []
    errors: list[dict[str, Any]] = []
    mlp: dict[str, Any] = {}
    ready = False
    try:
        mlp = run_expert_mlp(args)
        ready = (
            mlp.get("model_type") == "glm_moe_dsa"
            and len(_list(mlp.get("projection_summaries"))) == len(PROJECTIONS)
            and _list(mlp.get("final_output_shape")) == [_int(mlp.get("hidden_size"))]
        )
    except Exception as exc:
        errors.append({"phase": "expert_mlp", "error_type": type(exc).__name__, "error_digest": dequant_probe.sha_payload(str(exc))})
        blockers.append("glm52_pack_quantized_expert_mlp_failed")
    if ready:
        blockers.extend(
            [
                "glm52_pack_quantized_expert_mlp_is_single_expert_only",
                "glm52_pack_quantized_expert_mlp_is_not_attention",
                "glm52_pack_quantized_expert_mlp_is_not_topk_router",
                "glm52_pack_quantized_expert_mlp_is_not_stage_decode",
            ]
        )
    blockers.append("glm52_stage_decode_not_verified")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "glm52_pack_quantized_expert_mlp_probe_ready": ready,
        "model_id": MODEL_ID,
        "model_repo": str(args.model_repo),
        "model_type": str(mlp.get("model_type") or ""),
        "layer_id": int(args.layer_id),
        "expert_id": int(args.expert_id),
        "hidden_size": _int(mlp.get("hidden_size")),
        "projection_summaries": _list(mlp.get("projection_summaries")),
        "final_output_shape": _list(mlp.get("final_output_shape")),
        "final_output_hash": str(mlp.get("final_output_hash") or ""),
        "pack_quantized_expert_mlp_verified": ready,
        "single_expert_mlp_verified": ready,
        "stage_decode_verified": False,
        "errors": errors,
        "blockers": sorted(set(blockers)),
        "completion_boundary": {
            "single_expert_mlp_is_not_full_moe_layer": True,
            "single_expert_mlp_is_not_attention": True,
            "single_expert_mlp_is_not_topk_router": True,
            "single_expert_mlp_is_not_stage_decode": True,
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
        report["glm52_pack_quantized_expert_mlp_probe_ready"] = False
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
    parser.add_argument("--max-header-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-tensor-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--hf-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_report(args)
    path = output_dir / "glm52_pack_quantized_expert_mlp_probe.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"Expert MLP verified: {report.get('pack_quantized_expert_mlp_verified')}")
    return 0 if report.get("public_artifact_safe") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
