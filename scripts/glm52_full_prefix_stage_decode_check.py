#!/usr/bin/env python3
"""Validate GLM 5.2 full-prefix multi-layer stage decode reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_full_prefix_stage_decode_probe as probe  # noqa: E402


SCHEMA = "glm52_full_prefix_stage_decode_check_v1"


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

    lm_head_required = report.get("lm_head_required") is not False
    for key in [
        "full_prefix_stage_hidden_verified",
        "multi_layer_stage_hidden_verified",
        "full_prefix_token_carrier_verified",
        "all_layers_full_prefix_verified",
        "all_layer_outputs_chained",
    ]:
        if report.get(key) is not True:
            errors.append(f"{key}_missing")
    if lm_head_required:
        for key in [
            "stage_hidden_to_lm_head_verified",
            "lm_head_streamed_full_vocab",
            "stage_hidden_lm_head_token_selection_verified",
            "partial_full_prefix_token_hash_verified",
        ]:
            if report.get(key) is not True:
                errors.append(f"{key}_missing")
    else:
        if report.get("stage_handoff_only_verified") is not True:
            errors.append("stage_handoff_only_verified_missing")
        if report.get("lm_head_skipped_for_nonfinal_stage") is not True:
            errors.append("lm_head_skip_flag_missing")
        for key in [
            "stage_hidden_to_lm_head_verified",
            "lm_head_streamed_full_vocab",
            "stage_hidden_lm_head_token_selection_verified",
            "partial_full_prefix_token_hash_verified",
        ]:
            if report.get(key) is not False:
                errors.append(f"{key}_handoff_overclaim")
    for key in [
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
    seq_len = _int(report.get("stage_sequence_length"))
    layer_range = _list(report.get("stage_layer_range"))
    layer_count = _int(report.get("stage_layer_count"))
    if seq_len != _int(report.get("stage_prefill_length")) + 1 or seq_len < 2:
        errors.append("stage_sequence_length_invalid")
    if len(layer_range) != 2 or _int(layer_range[1]) <= _int(layer_range[0]):
        errors.append("stage_layer_range_invalid")
    elif _int(layer_range[1]) - _int(layer_range[0]) != layer_count:
        errors.append("stage_layer_count_mismatch")
    if layer_count < 2:
        errors.append("stage_layer_count_too_small")
    if _int(report.get("executed_layer_count")) != layer_count:
        errors.append("executed_layer_count_mismatch")
    if report.get("stage_hidden_source") != "dsa_masked_full_prefix_multi_layer_stage_hidden":
        errors.append("stage_hidden_source_invalid")
    if report.get("small_sequence_probe") is not True:
        errors.append("small_sequence_probe_flag_missing")

    expected_shapes = {
        "initial_stage_hidden_sequence_shape": [seq_len, hidden],
        "stage_hidden_sequence_shape": [seq_len, hidden],
        "stage_hidden_shape": [hidden],
        "norm_weight_shape": [hidden],
        "normalized_stage_hidden_shape": [hidden],
    }
    if lm_head_required:
        expected_shapes["lm_head_shape"] = [vocab, hidden]
    for key, expected in expected_shapes.items():
        if _list(report.get(key)) != expected:
            errors.append(f"shape_mismatch:{key}")
    if lm_head_required:
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
        "initial_stage_hidden_sequence_hash",
        "stage_hidden_sequence_hash",
        "stage_hidden_hash",
        "normalized_stage_hidden_hash",
    ]:
        if not _hash_ok(report.get(key)):
            errors.append(f"hash_missing:{key}")
    if lm_head_required:
        for key in [
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
        if item.get("layer_full_prefix_verified") is not True:
            errors.append(f"{prefix}_full_prefix_not_verified")
        if _int(item.get("token_count")) != seq_len:
            errors.append(f"{prefix}_token_count_mismatch")
        if _int(item.get("verified_token_count")) != seq_len:
            errors.append(f"{prefix}_verified_token_count_mismatch")
        if not set(str(v) for v in _list(item.get("dsa_indexer_types"))).issubset({"full", "shared"}):
            errors.append(f"{prefix}_dsa_indexer_type_invalid")
        if not _list(item.get("dsa_indexer_source_layer_ids")):
            errors.append(f"{prefix}_dsa_source_missing")
        if _int(item.get("final_token_dsa_mask_topk_count")) <= 0:
            errors.append(f"{prefix}_final_dsa_topk_missing")
        if _int(item.get("final_token_dsa_mask_pruned_position_count")) <= 0:
            errors.append(f"{prefix}_final_dsa_pruned_missing")
        if _list(item.get("final_token_full_moe_output_shape")) != [hidden]:
            errors.append(f"{prefix}_final_moe_shape_mismatch")
        if _list(item.get("final_token_layer_output_shape")) != [hidden]:
            errors.append(f"{prefix}_final_layer_shape_mismatch")
        if not _hash_ok(item.get("final_token_layer_output_hash")):
            errors.append(f"{prefix}_final_layer_hash_missing")

    if require_verified and report.get("full_prefix_stage_hidden_verified") is not True:
        errors.append("full_prefix_stage_decode_verification_required")

    blockers = set(str(item) for item in _list(report.get("blockers")))
    for blocker in [
        "glm52_full_prefix_stage_decode_uses_small_sequence_probe",
        "glm52_full_prefix_stage_decode_is_not_kaggle_runtime",
        "glm52_full_prefix_stage_decode_is_not_same_request",
        "glm52_stage_decode_not_verified",
        "glm52_same_request_decode_not_verified",
    ]:
        if blocker not in blockers:
            errors.append(f"blocker_missing:{blocker}")
    boundary = _dict(report.get("completion_boundary"))
    for key in [
        "full_prefix_stage_decode_uses_small_sequence_probe",
        "full_prefix_stage_decode_is_not_kaggle_runtime",
        "full_prefix_stage_decode_is_not_same_request",
        "requires_kaggle_stage_runtime",
        "requires_full_model_or_stage_partition",
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
        "full_prefix_stage_hidden_verified": report.get("full_prefix_stage_hidden_verified") is True,
        "stage_layer_range": _list(report.get("stage_layer_range")),
        "stage_sequence_length": _int(report.get("stage_sequence_length")),
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_full_prefix_stage_decode_check: ok={result['ok']} "
            f"errors={len(errors)} full_prefix={result['full_prefix_stage_hidden_verified']} "
            f"range={result['stage_layer_range']} seq={result['stage_sequence_length']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
