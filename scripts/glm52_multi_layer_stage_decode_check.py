#!/usr/bin/env python3
"""Validate GLM 5.2 multi-layer stage-hidden decode-token reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_multi_layer_stage_decode_probe as probe  # noqa: E402


SCHEMA = "glm52_multi_layer_stage_decode_check_v1"


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _hash_ok(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) >= 71


def validate_report(report: dict[str, Any], *, require_verified: bool = False) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != probe.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = probe.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))
    if report.get("model_id") != probe.MODEL_ID:
        errors.append("model_id_mismatch")
    if report.get("model_type") != "glm_moe_dsa":
        errors.append("model_type_not_glm_moe_dsa")

    for key in [
        "multi_layer_stage_hidden_verified",
        "multi_layer_decode_token_chain_verified",
        "all_layers_dsa_masked_attention_integrated",
        "all_layers_moe_mlp_verified",
        "all_layer_outputs_chained",
        "stage_hidden_to_lm_head_verified",
        "lm_head_streamed_full_vocab",
        "stage_hidden_lm_head_token_selection_verified",
        "partial_multi_layer_token_hash_verified",
    ]:
        if report.get(key) is not True:
            errors.append(f"{key}_missing")
    for key in [
        "full_prefill_stage_hidden_verified",
        "full_model_hidden_verified",
        "generated_token_verified",
        "stage_decode_verified",
        "same_request_decode_verified",
        "live_kaggle_runtime_verified",
    ]:
        if report.get(key) is not False:
            errors.append(f"{key}_overclaim")

    hidden = _int(report.get("hidden_size"))
    vocab = _int(report.get("vocab_size"))
    layer_range = _list(report.get("stage_layer_range"))
    layer_count = _int(report.get("stage_layer_count"))
    if len(layer_range) != 2 or _int(layer_range[1]) <= _int(layer_range[0]):
        errors.append("stage_layer_range_invalid")
    elif _int(layer_range[1]) - _int(layer_range[0]) != layer_count:
        errors.append("stage_layer_count_mismatch")
    if layer_count < 2:
        errors.append("stage_layer_count_too_small")
    if _int(report.get("executed_layer_count")) != layer_count:
        errors.append("executed_layer_count_mismatch")
    if report.get("stage_hidden_source") != "dsa_masked_multi_layer_decode_token_chain":
        errors.append("stage_hidden_source_invalid")
    if report.get("decode_token_chain_only") is not True:
        errors.append("decode_token_chain_only_flag_missing")
    if report.get("prefill_hidden_carrier_full_layer_outputs_verified") is not False:
        errors.append("prefill_hidden_carrier_overclaim")

    for key in ["initial_decode_hidden_shape", "stage_hidden_shape", "norm_weight_shape", "normalized_stage_hidden_shape"]:
        if _list(report.get(key)) != [hidden]:
            errors.append(f"shape_mismatch:{key}")
    if _list(report.get("lm_head_shape")) != [vocab, hidden]:
        errors.append("lm_head_shape_mismatch")
    if report.get("lm_head_dtype") != "BF16":
        errors.append("lm_head_dtype_not_bf16")
    if _int(report.get("lm_head_rows_scanned")) != vocab:
        errors.append("lm_head_rows_scanned_mismatch")
    if _int(report.get("lm_head_nbytes")) != vocab * hidden * 2:
        errors.append("lm_head_nbytes_mismatch")
    if _int(report.get("lm_head_block_count")) <= 0:
        errors.append("lm_head_block_count_missing")
    if _int(report.get("top_k_count")) != _int(report.get("top_k")):
        errors.append("top_k_count_mismatch")

    for key in [
        "initial_decode_hidden_hash",
        "stage_hidden_hash",
        "normalized_stage_hidden_hash",
        "selected_token_id_hash",
        "selected_logit_hash",
        "top_token_ids_hash",
        "top_logits_hash",
    ]:
        if not _hash_ok(report.get(key)):
            errors.append(f"hash_missing:{key}")

    layers = [item for item in _list(report.get("layer_summaries")) if isinstance(item, dict)]
    if len(layers) != layer_count:
        errors.append("layer_summary_count_mismatch")
    expected_layer_ids = list(range(_int(layer_range[0]), _int(layer_range[1]))) if len(layer_range) == 2 else []
    seen_layer_ids = [_int(item.get("layer_id"), -1) for item in layers]
    if expected_layer_ids and seen_layer_ids != expected_layer_ids:
        errors.append("layer_summary_ids_mismatch")
    for item in layers:
        prefix = f"layer{_int(item.get('layer_id'), -1)}"
        if item.get("layer_decode_token_verified") is not True:
            errors.append(f"{prefix}_decode_token_not_verified")
        if item.get("dsa_indexer_type") not in {"full", "shared"}:
            errors.append(f"{prefix}_dsa_indexer_type_invalid")
        if _int(item.get("dsa_indexer_source_layer_id"), -1) < 0:
            errors.append(f"{prefix}_dsa_indexer_source_missing")
        if _int(item.get("dsa_mask_topk_count")) <= 0:
            errors.append(f"{prefix}_dsa_topk_missing")
        if _int(item.get("dsa_mask_pruned_position_count")) <= 0:
            errors.append(f"{prefix}_dsa_pruned_missing")
        if _list(item.get("attention_output_shape")) != [hidden]:
            errors.append(f"{prefix}_attention_output_shape_mismatch")
        if _list(item.get("full_moe_output_shape")) != [hidden]:
            errors.append(f"{prefix}_moe_output_shape_mismatch")
        if _list(item.get("layer_output_shape")) != [hidden]:
            errors.append(f"{prefix}_layer_output_shape_mismatch")
        if item.get("attention_output_hash_present") is not True:
            errors.append(f"{prefix}_attention_output_hash_missing")
        if item.get("full_moe_output_hash_present") is not True:
            errors.append(f"{prefix}_moe_output_hash_missing")
        if not _hash_ok(item.get("layer_output_hash")):
            errors.append(f"{prefix}_layer_output_hash_missing")

    if require_verified and report.get("multi_layer_stage_hidden_verified") is not True:
        errors.append("multi_layer_stage_decode_verification_required")

    blockers = set(str(item) for item in _list(report.get("blockers")))
    for blocker in [
        "glm52_multi_layer_stage_decode_uses_decode_token_chain_only",
        "glm52_multi_layer_stage_decode_prefill_carrier_not_full_layer_outputs",
        "glm52_multi_layer_stage_decode_is_not_full_model_hidden",
        "glm52_multi_layer_stage_decode_is_not_kaggle_runtime",
        "glm52_multi_layer_stage_decode_is_not_same_request",
        "glm52_stage_decode_not_verified",
        "glm52_same_request_decode_not_verified",
    ]:
        if blocker not in blockers:
            errors.append(f"blocker_missing:{blocker}")
    boundary = _dict(report.get("completion_boundary"))
    for key in [
        "multi_layer_stage_decode_uses_decode_token_chain_only",
        "multi_layer_stage_decode_prefill_carrier_not_full_layer_outputs",
        "multi_layer_stage_decode_is_not_full_model_hidden",
        "multi_layer_stage_decode_is_not_kaggle_runtime",
        "multi_layer_stage_decode_is_not_same_request",
        "requires_full_prefill_layer_outputs",
        "requires_kaggle_stage_runtime",
        "requires_kaggle_cpu_gpu_tpu_same_request",
    ]:
        if boundary.get(key) is not True:
            errors.append(f"completion_boundary_missing:{key}")
    safety = _dict(report.get("safety"))
    for key in [
        "credentials_public",
        "cookies_public",
        "signed_url_public",
        "raw_prompt_public",
        "raw_generated_text_public",
        "generated_token_ids_public",
        "activation_public",
        "hidden_state_public",
        "logits_public",
        "kv_cache_public",
        "weight_tensor_values_public",
        "safetensors_header_payload_public",
    ]:
        if safety.get(key) is not False:
            errors.append(f"safety_flag_not_false:{key}")
    return sorted(set(errors))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-verified", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = load_json(Path(args.report))
    errors = validate_report(report, require_verified=bool(args.require_verified))
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors,
        "multi_layer_stage_hidden_verified": report.get("multi_layer_stage_hidden_verified") is True,
        "stage_layer_range": _list(report.get("stage_layer_range")),
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_multi_layer_stage_decode_check: ok={result['ok']} "
            f"errors={len(errors)} multi_layer={result['multi_layer_stage_hidden_verified']} "
            f"range={result['stage_layer_range']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
