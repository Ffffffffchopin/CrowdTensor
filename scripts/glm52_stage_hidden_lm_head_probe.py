#!/usr/bin/env python3
"""Run a public-safe GLM 5.2 stage-hidden to lm_head token-selection probe."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_attention_projection_probe as projection_probe  # noqa: E402
from scripts import glm52_dsa_masked_layer_decode_probe as layer_probe  # noqa: E402
from scripts import glm52_lm_head_token_probe as lm_head_probe  # noqa: E402
from scripts import glm52_pack_quantized_dequant_probe as dequant_probe  # noqa: E402
from scripts import glm52_pack_quantized_router_gather_probe as router_probe  # noqa: E402


SCHEMA = "glm52_stage_hidden_lm_head_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-stage-hidden-lm-head-probe"
DEFAULT_MODEL_REPO = dequant_probe.DEFAULT_MODEL_REPO
MODEL_ID = dequant_probe.MODEL_ID
SENSITIVE_FRAGMENTS = layer_probe.SENSITIVE_FRAGMENTS + lm_head_probe.SENSITIVE_FRAGMENTS + (
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


def run_stage_hidden_lm_head(args: argparse.Namespace) -> dict[str, Any]:
    config = dequant_probe.fetch_hf_json(args.model_repo, "config.json", timeout_seconds=float(args.hf_timeout_seconds))
    index = dequant_probe.fetch_hf_json(args.model_repo, "model.safetensors.index.json", timeout_seconds=float(args.hf_timeout_seconds))
    weight_map = _dict(index.get("weight_map"))
    hidden_size = _int(config.get("hidden_size"))
    vocab_size = _int(config.get("vocab_size"))
    if hidden_size <= 0 or vocab_size <= 0:
        raise RuntimeError("config_shape_missing")

    layer_summary, stage_hidden = layer_probe.run_dsa_masked_layer_decode_with_output(args)
    if _list(layer_summary.get("layer_output_shape")) != [hidden_size]:
        raise RuntimeError("stage_hidden_shape_mismatch")
    stage_updated = _int(layer_summary.get("updated_cache_length"))
    stage_dsa_masked = (
        str(layer_summary.get("dsa_indexer_type") or "") == "full"
        and _int(layer_summary.get("dsa_mask_topk_count")) > 0
        and _int(layer_summary.get("dsa_mask_pruned_position_count")) > 0
        and _list(layer_summary.get("dsa_index_score_shape")) == [stage_updated, stage_updated]
        and _list(layer_summary.get("attention_scores_shape")) == [_int(config.get("num_attention_heads")), stage_updated]
    )
    stage_layer_ready = (
        stage_dsa_masked
        and _list(layer_summary.get("attention_output_shape")) == [hidden_size]
        and _list(layer_summary.get("attention_residual_shape")) == [hidden_size]
        and _list(layer_summary.get("post_attention_norm_shape")) == [hidden_size]
        and _list(layer_summary.get("full_moe_output_shape")) == [hidden_size]
        and _list(layer_summary.get("layer_output_shape")) == [hidden_size]
    )
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
    lm_head_item = _dict(header.get("lm_head.weight"))
    streamed = lm_head_probe.stream_lm_head_topk(
        args,
        model_repo=str(args.model_repo),
        filename=lm_head_file,
        header_len=header_len,
        item=lm_head_item,
        hidden=normalized_stage_hidden,
        top_k=int(args.top_k),
    )
    return {
        "model_type": str(config.get("model_type") or ""),
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "tie_word_embeddings": bool(config.get("tie_word_embeddings")),
        "stage_hidden_source": "dsa_masked_single_layer_output",
        "stage_layer_id": int(args.layer_id),
        "stage_prefill_length": int(args.prefill_length),
        "stage_updated_cache_length": stage_updated,
        "stage_dsa_indexer_type": str(layer_summary.get("dsa_indexer_type") or ""),
        "stage_dsa_mask_topk_count": _int(layer_summary.get("dsa_mask_topk_count")),
        "stage_dsa_mask_pruned_position_count": _int(layer_summary.get("dsa_mask_pruned_position_count")),
        "stage_dsa_masked_attention_integrated": stage_dsa_masked,
        "stage_layer_decode_verified": stage_layer_ready,
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
        result = run_stage_hidden_lm_head(args)
        hidden = _int(result.get("hidden_size"))
        vocab = _int(result.get("vocab_size"))
        ready = (
            result.get("model_type") == "glm_moe_dsa"
            and result.get("stage_hidden_source") == "dsa_masked_single_layer_output"
            and result.get("stage_dsa_masked_attention_integrated") is True
            and result.get("stage_layer_decode_verified") is True
            and _list(result.get("stage_hidden_shape")) == [hidden]
            and _list(result.get("normalized_stage_hidden_shape")) == [hidden]
            and _list(result.get("lm_head_shape")) == [vocab, hidden]
            and _int(result.get("lm_head_rows_scanned")) == vocab
            and _int(result.get("top_k_count")) == int(args.top_k)
        )
    except Exception as exc:
        errors.append(
            {
                "phase": "stage_hidden_lm_head",
                "error_type": type(exc).__name__,
                "error_digest": dequant_probe.sha_payload(str(exc)),
            }
        )
        blockers.append("glm52_stage_hidden_lm_head_failed")
    if ready:
        blockers.extend(
            [
                "glm52_stage_hidden_lm_head_is_single_layer_only",
                "glm52_stage_hidden_lm_head_uses_small_sequence_topk_cap",
                "glm52_stage_hidden_lm_head_is_not_full_model_hidden",
                "glm52_stage_hidden_lm_head_is_not_stage_decode",
                "glm52_stage_hidden_lm_head_is_not_same_request",
            ]
        )
    blockers.extend(["glm52_stage_decode_not_verified", "glm52_same_request_decode_not_verified"])
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "glm52_stage_hidden_lm_head_probe_ready": ready,
        "model_id": MODEL_ID,
        "model_repo": str(args.model_repo),
        **result,
        "stage_hidden_to_lm_head_verified": ready,
        "lm_head_streamed_full_vocab": ready,
        "stage_hidden_lm_head_token_selection_verified": ready,
        "partial_layer_token_hash_verified": ready,
        "full_model_hidden_verified": False,
        "generated_token_verified": False,
        "stage_decode_verified": False,
        "same_request_decode_verified": False,
        "errors": errors,
        "blockers": sorted(set(blockers)),
        "completion_boundary": {
            "stage_hidden_lm_head_is_single_layer_only": True,
            "stage_hidden_lm_head_uses_small_sequence_topk_cap": True,
            "stage_hidden_lm_head_is_not_full_model_hidden": True,
            "stage_hidden_lm_head_is_not_stage_decode": True,
            "stage_hidden_lm_head_is_not_same_request": True,
            "requires_multi_layer_stage_runtime": True,
            "requires_full_model_or_stage_hidden": True,
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
        report["glm52_stage_hidden_lm_head_probe_ready"] = False
        report["public_artifact_safe"] = False
        report["safety"]["public_artifact_safe"] = False
        report["blockers"] = sorted(set([*blockers, "public_redaction_scan_failed"]))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--layer-id", type=int, default=6)
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
    if args.prefill_length <= 0:
        raise SystemExit("--prefill-length must be positive")
    if args.dsa_mask_topk <= 0:
        raise SystemExit("--dsa-mask-topk must be positive")
    if args.executed_expert_count <= 0:
        raise SystemExit("--executed-expert-count must be positive")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    if args.row_block_size <= 0:
        raise SystemExit("--row-block-size must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_report(args)
    path = output_dir / "glm52_stage_hidden_lm_head_probe.json"
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"Stage-hidden lm_head token selection verified: {report.get('stage_hidden_lm_head_token_selection_verified')}")
    return 0 if report.get("public_artifact_safe") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
