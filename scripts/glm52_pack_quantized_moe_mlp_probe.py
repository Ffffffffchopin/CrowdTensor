#!/usr/bin/env python3
"""Run a public-safe GLM 5.2 routed-plus-shared MoE MLP probe."""

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

from scripts import glm52_pack_quantized_dequant_probe as dequant_probe  # noqa: E402
from scripts import glm52_pack_quantized_expert_mlp_probe as expert_probe  # noqa: E402
from scripts import glm52_pack_quantized_router_gather_probe as router_probe  # noqa: E402


SCHEMA = "glm52_pack_quantized_moe_mlp_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-pack-quantized-moe-mlp-probe"
DEFAULT_MODEL_REPO = dequant_probe.DEFAULT_MODEL_REPO
MODEL_ID = dequant_probe.MODEL_ID
PROJECTIONS = expert_probe.PROJECTIONS
SENSITIVE_FRAGMENTS = dequant_probe.SENSITIVE_FRAGMENTS + (
    '"router_logits":',
    '"router_weights":',
    '"expert_output":',
    '"routed_output":',
    '"shared_output":',
    '"full_moe_output":',
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


def shared_projection_key(layer_id: int, projection: str) -> str:
    return f"model.layers.{int(layer_id)}.mlp.shared_experts.{projection}.weight"


def dense_linear(weight: torch.Tensor, input_vec: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or input_vec.ndim != 1:
        raise RuntimeError("dense_linear_shape_mismatch")
    if int(weight.shape[1]) != int(input_vec.shape[0]):
        raise RuntimeError(f"dense_linear_width_mismatch:{int(weight.shape[1])}:{int(input_vec.shape[0])}")
    return torch.matmul(weight.to(torch.float32), input_vec.to(torch.float32))


def run_shared_experts_from_weights(
    hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    outputs: dict[str, torch.Tensor] = {}
    summaries: list[dict[str, Any]] = []
    for projection in PROJECTIONS:
        weight = weights[projection]
        projection_input = hidden if projection != "down_proj" else torch.nn.functional.silu(outputs["gate_proj"]) * outputs["up_proj"]
        output = dense_linear(weight, projection_input)
        outputs[projection] = output
        summaries.append(
            {
                "projection": projection,
                "weight_dtype": str(weight.dtype).replace("torch.", ""),
                "weight_shape": [int(item) for item in weight.shape],
                "output_shape": [int(item) for item in output.shape],
                "output_hash": dequant_probe.sha_tensor(output),
            }
        )
    return outputs["down_proj"].to(torch.float32), summaries


def load_shared_weights(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    return {
        projection: router_probe.load_dense_tensor(args, shared_projection_key(int(args.layer_id), projection))
        for projection in PROJECTIONS
    }


def run_routed_experts_for_input(
    args: argparse.Namespace,
    config: dict[str, Any],
    hidden: torch.Tensor,
) -> dict[str, Any]:
    hidden_size = _int(config.get("hidden_size"))
    gate_key = f"model.layers.{int(args.layer_id)}.mlp.gate.weight"
    bias_key = f"model.layers.{int(args.layer_id)}.mlp.gate.e_score_correction_bias"
    gate_weight = router_probe.load_dense_tensor(args, gate_key)
    correction_bias = router_probe.load_dense_tensor(args, bias_key)
    topk_indices, topk_weights = router_probe.route_topk(config, gate_weight, correction_bias, hidden)
    execute_count = min(int(args.executed_expert_count), int(topk_indices.numel()))
    routed = torch.zeros(hidden_size, dtype=torch.float32)
    executed: list[dict[str, Any]] = []
    for position in range(execute_count):
        expert_id = int(topk_indices[position].item())
        expert_weight = topk_weights[position].to(torch.float32)
        output = router_probe.run_expert_for_input(args, expert_id, hidden)
        routed = routed + output * expert_weight
        executed.append(
            {
                "topk_position": position,
                "expert_id": expert_id,
                "expert_weight_hash": dequant_probe.sha_tensor(expert_weight.reshape(1)),
                "expert_output_shape": [int(item) for item in output.shape],
                "expert_output_hash": dequant_probe.sha_tensor(output),
            }
        )
    return {
        "router_topk_count": int(topk_indices.numel()),
        "router_topk_indices_hash": dequant_probe.sha_tensor(topk_indices),
        "router_topk_weights_hash": dequant_probe.sha_tensor(topk_weights),
        "executed_expert_count": execute_count,
        "executed_experts": executed,
        "routed_output": routed,
    }


def run_moe_mlp(args: argparse.Namespace) -> dict[str, Any]:
    config = dequant_probe.fetch_hf_json(args.model_repo, "config.json", timeout_seconds=float(args.hf_timeout_seconds))
    hidden_size = _int(config.get("hidden_size"))
    if hidden_size <= 0:
        raise RuntimeError("hidden_size_missing")
    hidden = torch.linspace(-0.05, 0.05, steps=hidden_size, dtype=torch.float32)
    routed = run_routed_experts_for_input(args, config, hidden)
    shared_weights = load_shared_weights(args)
    shared_output, shared_projection_summaries = run_shared_experts_from_weights(hidden, shared_weights)
    full_output = routed["routed_output"].to(torch.float32) + shared_output.to(torch.float32)
    return {
        "model_type": str(config.get("model_type") or ""),
        "hidden_size": hidden_size,
        "n_routed_experts": _int(config.get("n_routed_experts")),
        "num_experts_per_tok": _int(config.get("num_experts_per_tok")),
        "n_shared_experts": _int(config.get("n_shared_experts")),
        "moe_intermediate_size": _int(config.get("moe_intermediate_size")),
        "routed_scaling_factor": float(config.get("routed_scaling_factor") or 1.0),
        "router_topk_count": _int(routed.get("router_topk_count")),
        "router_topk_indices_hash": str(routed.get("router_topk_indices_hash") or ""),
        "router_topk_weights_hash": str(routed.get("router_topk_weights_hash") or ""),
        "executed_expert_count": _int(routed.get("executed_expert_count")),
        "executed_experts": _list(routed.get("executed_experts")),
        "routed_output_shape": [int(item) for item in routed["routed_output"].shape],
        "routed_output_hash": dequant_probe.sha_tensor(routed["routed_output"]),
        "shared_projection_summaries": shared_projection_summaries,
        "shared_output_shape": [int(item) for item in shared_output.shape],
        "shared_output_hash": dequant_probe.sha_tensor(shared_output),
        "full_moe_output_shape": [int(item) for item in full_output.shape],
        "full_moe_output_hash": dequant_probe.sha_tensor(full_output),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    blockers: list[str] = []
    errors: list[dict[str, Any]] = []
    moe: dict[str, Any] = {}
    ready = False
    try:
        moe = run_moe_mlp(args)
        ready = (
            moe.get("model_type") == "glm_moe_dsa"
            and _int(moe.get("router_topk_count")) == _int(moe.get("num_experts_per_tok"))
            and _int(moe.get("executed_expert_count")) == _int(moe.get("num_experts_per_tok"))
            and _list(moe.get("routed_output_shape")) == [_int(moe.get("hidden_size"))]
            and _list(moe.get("shared_output_shape")) == [_int(moe.get("hidden_size"))]
            and _list(moe.get("full_moe_output_shape")) == [_int(moe.get("hidden_size"))]
            and len(_list(moe.get("shared_projection_summaries"))) == len(PROJECTIONS)
        )
    except Exception as exc:
        errors.append({"phase": "moe_mlp", "error_type": type(exc).__name__, "error_digest": dequant_probe.sha_payload(str(exc))})
        blockers.append("glm52_pack_quantized_moe_mlp_failed")
    if ready:
        blockers.extend(
            [
                "glm52_pack_quantized_moe_mlp_is_not_attention",
                "glm52_pack_quantized_moe_mlp_is_not_transformer_block",
                "glm52_pack_quantized_moe_mlp_is_not_stage_decode",
                "glm52_pack_quantized_moe_mlp_missing_kv_cache",
                "glm52_pack_quantized_moe_mlp_missing_lm_head",
            ]
        )
    blockers.append("glm52_stage_decode_not_verified")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "glm52_pack_quantized_moe_mlp_probe_ready": ready,
        "model_id": MODEL_ID,
        "model_repo": str(args.model_repo),
        "model_type": str(moe.get("model_type") or ""),
        "layer_id": int(args.layer_id),
        "hidden_size": _int(moe.get("hidden_size")),
        "n_routed_experts": _int(moe.get("n_routed_experts")),
        "num_experts_per_tok": _int(moe.get("num_experts_per_tok")),
        "n_shared_experts": _int(moe.get("n_shared_experts")),
        "moe_intermediate_size": _int(moe.get("moe_intermediate_size")),
        "routed_scaling_factor": moe.get("routed_scaling_factor", 0),
        "router_topk_count": _int(moe.get("router_topk_count")),
        "router_topk_indices_hash": str(moe.get("router_topk_indices_hash") or ""),
        "router_topk_weights_hash": str(moe.get("router_topk_weights_hash") or ""),
        "executed_expert_count": _int(moe.get("executed_expert_count")),
        "requested_executed_expert_count": int(args.executed_expert_count),
        "executed_experts": _list(moe.get("executed_experts")),
        "routed_output_shape": _list(moe.get("routed_output_shape")),
        "routed_output_hash": str(moe.get("routed_output_hash") or ""),
        "shared_projection_summaries": _list(moe.get("shared_projection_summaries")),
        "shared_output_shape": _list(moe.get("shared_output_shape")),
        "shared_output_hash": str(moe.get("shared_output_hash") or ""),
        "full_moe_output_shape": _list(moe.get("full_moe_output_shape")),
        "full_moe_output_hash": str(moe.get("full_moe_output_hash") or ""),
        "router_topk_verified": ready,
        "routed_expert_gather_verified": ready,
        "shared_experts_mlp_verified": ready,
        "pack_quantized_moe_mlp_verified": ready,
        "full_moe_mlp_verified": ready,
        "stage_decode_verified": False,
        "errors": errors,
        "blockers": sorted(set(blockers)),
        "completion_boundary": {
            "full_moe_mlp_is_not_attention": True,
            "full_moe_mlp_is_not_transformer_block": True,
            "full_moe_mlp_is_not_stage_decode": True,
            "requires_attention_runtime": True,
            "requires_residual_norm_runtime": True,
            "requires_stage_local_kv_cache": True,
            "requires_lm_head_token_selection": True,
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
        report["glm52_pack_quantized_moe_mlp_probe_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"] = sorted(set([*blockers, "public_redaction_scan_failed"]))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--layer-id", type=int, default=3)
    parser.add_argument("--executed-expert-count", type=int, default=8)
    parser.add_argument("--max-header-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-tensor-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--hf-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.executed_expert_count <= 0:
        raise SystemExit("--executed-expert-count must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_report(args)
    path = output_dir / "glm52_pack_quantized_moe_mlp_probe.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"Full MoE MLP verified: {report.get('full_moe_mlp_verified')}")
    return 0 if report.get("public_artifact_safe") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
