#!/usr/bin/env python3
"""Run a public-safe GLM 5.2 multi-layer stage-hidden decode-token probe."""

from __future__ import annotations

import argparse
import json
import sys
from copy import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_attention_projection_probe as projection_probe  # noqa: E402
from scripts import glm52_dsa_masked_layer_decode_probe as layer_probe  # noqa: E402
from scripts import glm52_kv_cache_decode_probe as kv_probe  # noqa: E402
from scripts import glm52_lm_head_token_probe as lm_head_probe  # noqa: E402
from scripts import glm52_pack_quantized_dequant_probe as dequant_probe  # noqa: E402
from scripts import glm52_pack_quantized_router_gather_probe as router_probe  # noqa: E402


SCHEMA = "glm52_multi_layer_stage_decode_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-multi-layer-stage-decode-probe"
DEFAULT_MODEL_REPO = dequant_probe.DEFAULT_MODEL_REPO
MODEL_ID = dequant_probe.MODEL_ID
SENSITIVE_FRAGMENTS = layer_probe.SENSITIVE_FRAGMENTS + lm_head_probe.SENSITIVE_FRAGMENTS + (
    '"stage_hidden_sequence":',
    '"stage_hidden":',
    '"normalized_stage_hidden":',
    '"selected_token":',
    '"token_id":',
    '"token_ids":',
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


def _hash_ok(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) >= 71


def _layer_ready(layer: dict[str, Any], config: dict[str, Any]) -> bool:
    hidden = _int(layer.get("hidden_size"))
    heads = _int(layer.get("num_attention_heads"))
    qk = _int(layer.get("qk_head_dim"))
    value_dim = _int(layer.get("v_head_dim"))
    updated = _int(layer.get("updated_cache_length"))
    return bool(
        layer.get("model_type") == "glm_moe_dsa"
        and str(layer.get("dsa_indexer_type") or "") in {"full", "shared"}
        and _int(layer.get("dsa_indexer_source_layer_id"), -1) >= 0
        and _list(layer.get("updated_key_cache_shape")) == [updated, heads, qk]
        and _list(layer.get("updated_value_cache_shape")) == [updated, heads, value_dim]
        and _list(layer.get("attention_scores_shape")) == [heads, updated]
        and _list(layer.get("attention_weights_shape")) == [heads, updated]
        and _list(layer.get("attention_output_shape")) == [hidden]
        and _list(layer.get("attention_residual_shape")) == [hidden]
        and _list(layer.get("post_attention_norm_shape")) == [hidden]
        and _int(layer.get("dsa_mask_topk_count")) > 0
        and _int(layer.get("dsa_mask_pruned_position_count")) > 0
        and _int(layer.get("router_topk_count")) == _int(config.get("num_experts_per_tok"))
        and _int(layer.get("executed_expert_count")) == _int(config.get("num_experts_per_tok"))
        and _list(layer.get("full_moe_output_shape")) == [hidden]
        and _list(layer.get("layer_output_shape")) == [hidden]
        and _hash_ok(layer.get("layer_output_hash"))
    )


def _public_layer_summary(layer_id: int, layer: dict[str, Any], *, ready: bool) -> dict[str, Any]:
    return {
        "layer_id": int(layer_id),
        "layer_decode_token_verified": ready,
        "dsa_indexer_type": str(layer.get("dsa_indexer_type") or ""),
        "dsa_indexer_source_layer_id": _int(layer.get("dsa_indexer_source_layer_id"), -1),
        "dsa_indexer_source_type": str(layer.get("dsa_indexer_source_type") or ""),
        "dsa_mask_topk_count": _int(layer.get("dsa_mask_topk_count")),
        "dsa_mask_pruned_position_count": _int(layer.get("dsa_mask_pruned_position_count")),
        "attention_scores_shape": _list(layer.get("attention_scores_shape")),
        "attention_output_shape": _list(layer.get("attention_output_shape")),
        "attention_residual_shape": _list(layer.get("attention_residual_shape")),
        "post_attention_norm_shape": _list(layer.get("post_attention_norm_shape")),
        "full_moe_output_shape": _list(layer.get("full_moe_output_shape")),
        "layer_output_shape": _list(layer.get("layer_output_shape")),
        "router_topk_count": _int(layer.get("router_topk_count")),
        "executed_expert_count": _int(layer.get("executed_expert_count")),
        "dsa_index_score_hash_present": _hash_ok(layer.get("dsa_index_score_hash")),
        "dsa_attention_mask_hash_present": _hash_ok(layer.get("dsa_attention_mask_hash")),
        "attention_output_hash_present": _hash_ok(layer.get("attention_output_hash")),
        "full_moe_output_hash_present": _hash_ok(layer.get("full_moe_output_hash")),
        "layer_output_hash": str(layer.get("layer_output_hash") or ""),
    }


def _layer_args(args: argparse.Namespace, layer_id: int) -> argparse.Namespace:
    layer_args = copy(args)
    layer_args.layer_id = int(layer_id)
    return layer_args


def run_multi_layer_stage_decode(args: argparse.Namespace) -> dict[str, Any]:
    config = dequant_probe.fetch_hf_json(args.model_repo, "config.json", timeout_seconds=float(args.hf_timeout_seconds))
    index = dequant_probe.fetch_hf_json(args.model_repo, "model.safetensors.index.json", timeout_seconds=float(args.hf_timeout_seconds))
    weight_map = _dict(index.get("weight_map"))
    hidden_size = _int(config.get("hidden_size"))
    vocab_size = _int(config.get("vocab_size"))
    num_layers = _int(config.get("num_hidden_layers"))
    start = int(args.layer_start)
    end = int(args.layer_end)
    if hidden_size <= 0 or vocab_size <= 0 or num_layers <= 0:
        raise RuntimeError("config_shape_missing")
    if start < 0 or end <= start or end > num_layers:
        raise RuntimeError("stage_layer_range_invalid")

    total_len = int(args.prefill_length) + 1
    hidden_sequence = kv_probe.build_hidden_sequence(total_len, hidden_size).to(torch.float32)
    initial_decode_hidden = hidden_sequence[int(args.prefill_length)].contiguous()
    layer_summaries: list[dict[str, Any]] = []
    layer_ready_flags: list[bool] = []
    for layer_id in range(start, end):
        layer_summary, layer_output = layer_probe.run_dsa_masked_layer_decode_for_hidden(
            _layer_args(args, layer_id),
            config,
            hidden_sequence,
        )
        ready = _layer_ready(layer_summary, config)
        layer_ready_flags.append(ready)
        layer_summaries.append(_public_layer_summary(layer_id, layer_summary, ready=ready))
        hidden_sequence = hidden_sequence.clone()
        hidden_sequence[int(args.prefill_length)] = layer_output.to(torch.float32)

    stage_hidden = hidden_sequence[int(args.prefill_length)].contiguous().to(torch.float32)
    norm_weight = router_probe.load_dense_tensor(args, "model.norm.weight")
    normalized_stage_hidden = projection_probe.rms_norm(
        stage_hidden.to(norm_weight.device), norm_weight, float(config.get("rms_norm_eps") or 1e-6)
    )
    lm_head_file = str(weight_map.get("lm_head.weight") or "")
    if not lm_head_file:
        raise RuntimeError("lm_head_weight_missing")
    header_len, header = dequant_probe.load_safetensors_header_with_len(
        args.model_repo,
        lm_head_file,
        timeout_seconds=float(args.hf_timeout_seconds),
        max_header_bytes=int(args.max_header_bytes),
    )
    streamed = lm_head_probe.stream_lm_head_topk(
        args,
        model_repo=str(args.model_repo),
        filename=lm_head_file,
        header_len=header_len,
        item=_dict(header.get("lm_head.weight")),
        hidden=normalized_stage_hidden,
        top_k=int(args.top_k),
    )
    return {
        "model_type": str(config.get("model_type") or ""),
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "num_hidden_layers": num_layers,
        "tie_word_embeddings": bool(config.get("tie_word_embeddings")),
        "stage_hidden_source": "dsa_masked_multi_layer_decode_token_chain",
        "stage_layer_range": [start, end],
        "stage_layer_count": end - start,
        "executed_layer_count": len(layer_summaries),
        "stage_prefill_length": int(args.prefill_length),
        "stage_updated_cache_length": total_len,
        "dsa_mask_topk_requested": int(args.dsa_mask_topk),
        "decode_token_chain_only": True,
        "prefill_hidden_carrier_full_layer_outputs_verified": False,
        "initial_decode_hidden_shape": [int(item) for item in initial_decode_hidden.shape],
        "initial_decode_hidden_hash": dequant_probe.sha_tensor(initial_decode_hidden),
        "layer_summaries": layer_summaries,
        "all_layers_dsa_masked_attention_integrated": all(
            item.get("dsa_indexer_type") in {"full", "shared"}
            and _int(item.get("dsa_indexer_source_layer_id"), -1) >= 0
            and _int(item.get("dsa_mask_topk_count")) > 0
            and _int(item.get("dsa_mask_pruned_position_count")) > 0
            for item in layer_summaries
        ),
        "all_layers_moe_mlp_verified": all(
            _list(item.get("full_moe_output_shape")) == [hidden_size] and item.get("full_moe_output_hash_present") is True
            for item in layer_summaries
        ),
        "all_layer_outputs_chained": all(layer_ready_flags) and bool(layer_summaries),
        "stage_hidden_shape": [int(item) for item in stage_hidden.shape],
        "stage_hidden_hash": dequant_probe.sha_tensor(stage_hidden),
        "norm_weight_shape": [int(item) for item in norm_weight.shape],
        "normalized_stage_hidden_shape": [int(item) for item in normalized_stage_hidden.shape],
        "normalized_stage_hidden_hash": dequant_probe.sha_tensor(normalized_stage_hidden),
        **streamed,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    blockers: list[str] = []
    errors: list[dict[str, Any]] = []
    result: dict[str, Any] = {}
    ready = False
    try:
        result = run_multi_layer_stage_decode(args)
        hidden = _int(result.get("hidden_size"))
        vocab = _int(result.get("vocab_size"))
        ready = (
            result.get("model_type") == "glm_moe_dsa"
            and result.get("stage_hidden_source") == "dsa_masked_multi_layer_decode_token_chain"
            and _int(result.get("stage_layer_count")) >= 2
            and _int(result.get("executed_layer_count")) == _int(result.get("stage_layer_count"))
            and result.get("all_layers_dsa_masked_attention_integrated") is True
            and result.get("all_layers_moe_mlp_verified") is True
            and result.get("all_layer_outputs_chained") is True
            and _list(result.get("stage_hidden_shape")) == [hidden]
            and _list(result.get("normalized_stage_hidden_shape")) == [hidden]
            and _list(result.get("lm_head_shape")) == [vocab, hidden]
            and _int(result.get("lm_head_rows_scanned")) == vocab
            and _int(result.get("top_k_count")) == int(args.top_k)
        )
    except Exception as exc:
        errors.append(
            {
                "phase": "multi_layer_stage_decode",
                "error_type": type(exc).__name__,
                "error_digest": dequant_probe.sha_payload(str(exc)),
            }
        )
        blockers.append("glm52_multi_layer_stage_decode_failed")
    if ready:
        blockers.extend(
            [
                "glm52_multi_layer_stage_decode_uses_decode_token_chain_only",
                "glm52_multi_layer_stage_decode_prefill_carrier_not_full_layer_outputs",
                "glm52_multi_layer_stage_decode_is_not_full_model_hidden",
                "glm52_multi_layer_stage_decode_is_not_kaggle_runtime",
                "glm52_multi_layer_stage_decode_is_not_same_request",
            ]
        )
    blockers.extend(["glm52_stage_decode_not_verified", "glm52_same_request_decode_not_verified"])
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "glm52_multi_layer_stage_decode_probe_ready": ready,
        "model_id": MODEL_ID,
        "model_repo": str(args.model_repo),
        **result,
        "multi_layer_stage_hidden_verified": ready,
        "multi_layer_decode_token_chain_verified": ready,
        "stage_hidden_to_lm_head_verified": ready,
        "lm_head_streamed_full_vocab": ready,
        "stage_hidden_lm_head_token_selection_verified": ready,
        "partial_multi_layer_token_hash_verified": ready,
        "full_prefill_stage_hidden_verified": False,
        "full_model_hidden_verified": False,
        "generated_token_verified": False,
        "stage_decode_verified": False,
        "same_request_decode_verified": False,
        "live_kaggle_runtime_verified": False,
        "errors": errors,
        "blockers": sorted(set(blockers)),
        "completion_boundary": {
            "multi_layer_stage_decode_uses_decode_token_chain_only": True,
            "multi_layer_stage_decode_prefill_carrier_not_full_layer_outputs": True,
            "multi_layer_stage_decode_is_not_full_model_hidden": True,
            "multi_layer_stage_decode_is_not_kaggle_runtime": True,
            "multi_layer_stage_decode_is_not_same_request": True,
            "requires_full_prefill_layer_outputs": True,
            "requires_kaggle_stage_runtime": True,
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
        report["glm52_multi_layer_stage_decode_probe_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"] = sorted(set([*blockers, "public_redaction_scan_failed"]))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--layer-start", type=int, default=6)
    parser.add_argument("--layer-end", type=int, default=8)
    parser.add_argument("--prefill-length", type=int, default=8)
    parser.add_argument("--dsa-mask-topk", type=int, default=4)
    parser.add_argument("--executed-expert-count", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--row-block-size", type=int, default=2048)
    parser.add_argument("--max-header-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-tensor-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-block-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--hf-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.layer_end <= args.layer_start:
        raise SystemExit("--layer-end must be greater than --layer-start")
    if args.prefill_length <= 0:
        raise SystemExit("--prefill-length must be positive")
    if args.dsa_mask_topk <= 0:
        raise SystemExit("--dsa-mask-topk must be positive")
    if args.executed_expert_count <= 0:
        raise SystemExit("--executed-expert-count must be positive")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_report(args)
    path = output_dir / "glm52_multi_layer_stage_decode_probe.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"Multi-layer stage hidden verified: {report.get('multi_layer_stage_hidden_verified')}")
    return 0 if report.get("public_artifact_safe") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
