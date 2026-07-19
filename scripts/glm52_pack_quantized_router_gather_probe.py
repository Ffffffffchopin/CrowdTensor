#!/usr/bin/env python3
"""Run a public-safe GLM 5.2 router + routed expert gather subset probe."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_pack_quantized_dequant_probe as dequant_probe  # noqa: E402
from scripts import glm52_pack_quantized_expert_mlp_probe as expert_probe  # noqa: E402


SCHEMA = "glm52_pack_quantized_router_gather_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-pack-quantized-router-gather-probe"
DEFAULT_MODEL_REPO = dequant_probe.DEFAULT_MODEL_REPO
MODEL_ID = dequant_probe.MODEL_ID
SENSITIVE_FRAGMENTS = dequant_probe.SENSITIVE_FRAGMENTS + (
    '"router_logits":',
    '"router_weights":',
    '"expert_output":',
    '"routed_output":',
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


def load_dense_tensor(args: argparse.Namespace, key: str) -> torch.Tensor:
    index = dequant_probe.fetch_hf_json(args.model_repo, "model.safetensors.index.json", timeout_seconds=float(args.hf_timeout_seconds))
    filename = str(_dict(index.get("weight_map")).get(key) or "")
    if not filename:
        raise RuntimeError(f"dense_weight_key_missing:{key}")
    header_len, header = dequant_probe.load_safetensors_header_with_len(
        args.model_repo,
        filename,
        timeout_seconds=float(args.hf_timeout_seconds),
        max_header_bytes=int(args.max_header_bytes),
    )
    item = _dict(header.get(key))
    return dequant_probe.load_tensor(args.model_repo, filename, header_len, item, args)


def route_topk(config: dict[str, Any], gate_weight: torch.Tensor, correction_bias: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n_routed = _int(config.get("n_routed_experts"))
    top_k = _int(config.get("num_experts_per_tok"))
    n_group = _int(config.get("n_group"), 1)
    topk_group = _int(config.get("topk_group"), 1)
    scaling = float(config.get("routed_scaling_factor") or 1.0)
    logits = torch.matmul(gate_weight.to(torch.float32), hidden.to(torch.float32))
    sigmoid_scores = torch.sigmoid(logits).reshape(1, n_routed)
    choice_scores = sigmoid_scores + correction_bias.to(torch.float32).reshape(1, n_routed)
    group_scores = choice_scores.view(-1, n_group, n_routed // n_group).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(-1, n_group, n_routed // n_group).reshape(-1, n_routed)
    scores_for_choice = choice_scores.masked_fill(~score_mask.bool(), float("-inf"))
    topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=True)[1].reshape(-1)
    topk_weights = sigmoid_scores.gather(1, topk_indices.reshape(1, -1)).reshape(-1)
    if bool(config.get("norm_topk_prob")):
        topk_weights = topk_weights / (topk_weights.sum() + 1e-20)
    topk_weights = topk_weights * scaling
    return topk_indices.to(torch.int64), topk_weights.to(torch.float32)


def run_expert_for_input(args: argparse.Namespace, expert_id: int, hidden: torch.Tensor) -> torch.Tensor:
    expert_args = SimpleNamespace(**vars(args))
    expert_args.expert_id = int(expert_id)
    outputs: dict[str, torch.Tensor] = {}
    for projection in expert_probe.PROJECTIONS:
        tensors = expert_probe.load_projection_tensors(expert_args, projection)
        projection_input = hidden if projection != "down_proj" else torch.nn.functional.silu(outputs["gate_proj"]) * outputs["up_proj"]
        output, _, _ = expert_probe.projection_linear(tensors, projection_input)
        outputs[projection] = output
    return outputs["down_proj"].to(torch.float32)


def run_router_gather(args: argparse.Namespace) -> dict[str, Any]:
    config = dequant_probe.fetch_hf_json(args.model_repo, "config.json", timeout_seconds=float(args.hf_timeout_seconds))
    hidden_size = _int(config.get("hidden_size"))
    if hidden_size <= 0:
        raise RuntimeError("hidden_size_missing")
    hidden = torch.linspace(-0.05, 0.05, steps=hidden_size, dtype=torch.float32)
    gate_key = f"model.layers.{int(args.layer_id)}.mlp.gate.weight"
    bias_key = f"model.layers.{int(args.layer_id)}.mlp.gate.e_score_correction_bias"
    gate_weight = load_dense_tensor(args, gate_key)
    correction_bias = load_dense_tensor(args, bias_key)
    topk_indices, topk_weights = route_topk(config, gate_weight, correction_bias, hidden)
    execute_count = min(int(args.executed_expert_count), int(topk_indices.numel()))
    routed = torch.zeros(hidden_size, dtype=torch.float32)
    executed: list[dict[str, Any]] = []
    for position in range(execute_count):
        expert_id = int(topk_indices[position].item())
        weight = topk_weights[position].to(torch.float32)
        output = run_expert_for_input(args, expert_id, hidden)
        routed = routed + output * weight
        executed.append(
            {
                "topk_position": position,
                "expert_id": expert_id,
                "expert_weight_hash": dequant_probe.sha_tensor(weight.reshape(1)),
                "expert_output_shape": [int(item) for item in output.shape],
                "expert_output_hash": dequant_probe.sha_tensor(output),
            }
        )
    return {
        "model_type": str(config.get("model_type") or ""),
        "hidden_size": hidden_size,
        "n_routed_experts": _int(config.get("n_routed_experts")),
        "num_experts_per_tok": _int(config.get("num_experts_per_tok")),
        "routed_scaling_factor": float(config.get("routed_scaling_factor") or 1.0),
        "router_topk_count": int(topk_indices.numel()),
        "router_topk_indices_hash": dequant_probe.sha_tensor(topk_indices),
        "router_topk_weights_hash": dequant_probe.sha_tensor(topk_weights),
        "executed_expert_count": execute_count,
        "executed_experts": executed,
        "routed_subset_output_shape": [int(item) for item in routed.shape],
        "routed_subset_output_hash": dequant_probe.sha_tensor(routed),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    blockers: list[str] = []
    errors: list[dict[str, Any]] = []
    routed: dict[str, Any] = {}
    ready = False
    try:
        routed = run_router_gather(args)
        ready = (
            routed.get("model_type") == "glm_moe_dsa"
            and _int(routed.get("router_topk_count")) == _int(routed.get("num_experts_per_tok"))
            and _int(routed.get("executed_expert_count")) > 0
            and _list(routed.get("routed_subset_output_shape")) == [_int(routed.get("hidden_size"))]
        )
    except Exception as exc:
        errors.append({"phase": "router_gather", "error_type": type(exc).__name__, "error_digest": dequant_probe.sha_payload(str(exc))})
        blockers.append("glm52_pack_quantized_router_gather_failed")
    if ready:
        blockers.extend(
            [
                "glm52_pack_quantized_router_gather_is_subset_only",
                "glm52_pack_quantized_router_gather_missing_shared_experts",
                "glm52_pack_quantized_router_gather_is_not_attention",
                "glm52_pack_quantized_router_gather_is_not_stage_decode",
            ]
        )
    blockers.append("glm52_stage_decode_not_verified")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "glm52_pack_quantized_router_gather_probe_ready": ready,
        "model_id": MODEL_ID,
        "model_repo": str(args.model_repo),
        "model_type": str(routed.get("model_type") or ""),
        "layer_id": int(args.layer_id),
        "hidden_size": _int(routed.get("hidden_size")),
        "n_routed_experts": _int(routed.get("n_routed_experts")),
        "num_experts_per_tok": _int(routed.get("num_experts_per_tok")),
        "routed_scaling_factor": routed.get("routed_scaling_factor", 0),
        "router_topk_count": _int(routed.get("router_topk_count")),
        "router_topk_indices_hash": str(routed.get("router_topk_indices_hash") or ""),
        "router_topk_weights_hash": str(routed.get("router_topk_weights_hash") or ""),
        "executed_expert_count": _int(routed.get("executed_expert_count")),
        "requested_executed_expert_count": int(args.executed_expert_count),
        "executed_experts": _list(routed.get("executed_experts")),
        "routed_subset_output_shape": _list(routed.get("routed_subset_output_shape")),
        "routed_subset_output_hash": str(routed.get("routed_subset_output_hash") or ""),
        "router_topk_verified": ready,
        "routed_expert_subset_verified": ready,
        "stage_decode_verified": False,
        "errors": errors,
        "blockers": sorted(set(blockers)),
        "completion_boundary": {
            "routed_subset_is_not_full_moe_layer": True,
            "shared_experts_not_included": True,
            "attention_not_included": True,
            "stage_decode_not_included": True,
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
        report["glm52_pack_quantized_router_gather_probe_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"] = sorted(set([*blockers, "public_redaction_scan_failed"]))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--layer-id", type=int, default=3)
    parser.add_argument("--executed-expert-count", type=int, default=2)
    parser.add_argument("--max-header-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-tensor-bytes", type=int, default=16 * 1024 * 1024)
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
    path = output_dir / "glm52_pack_quantized_router_gather_probe.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"Router gather subset verified: {report.get('routed_expert_subset_verified')}")
    return 0 if report.get("public_artifact_safe") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
