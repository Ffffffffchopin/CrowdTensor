#!/usr/bin/env python3
"""Validate GLM 5.2 stage-hidden to lm_head token-selection reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_stage_hidden_lm_head_probe as probe  # noqa: E402


SCHEMA = "glm52_stage_hidden_lm_head_check_v1"


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
        "stage_dsa_masked_attention_integrated",
        "stage_layer_decode_verified",
        "stage_hidden_to_lm_head_verified",
        "lm_head_streamed_full_vocab",
        "stage_hidden_lm_head_token_selection_verified",
        "partial_layer_token_hash_verified",
    ]:
        if report.get(key) is not True:
            errors.append(f"{key}_missing")
    for key in [
        "full_model_hidden_verified",
        "generated_token_verified",
        "stage_decode_verified",
        "same_request_decode_verified",
    ]:
        if report.get(key) is not False:
            errors.append(f"{key}_overclaim")

    hidden = _int(report.get("hidden_size"))
    vocab = _int(report.get("vocab_size"))
    if _list(report.get("stage_hidden_shape")) != [hidden]:
        errors.append("stage_hidden_shape_mismatch")
    if _list(report.get("norm_weight_shape")) != [hidden]:
        errors.append("norm_weight_shape_mismatch")
    if _list(report.get("normalized_stage_hidden_shape")) != [hidden]:
        errors.append("normalized_stage_hidden_shape_mismatch")
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
    if _int(report.get("stage_dsa_mask_topk_count")) <= 0:
        errors.append("stage_dsa_mask_topk_missing")
    if _int(report.get("stage_dsa_mask_pruned_position_count")) <= 0:
        errors.append("stage_dsa_mask_pruned_missing")
    for key in [
        "stage_hidden_hash",
        "normalized_stage_hidden_hash",
        "selected_token_id_hash",
        "selected_logit_hash",
        "top_token_ids_hash",
        "top_logits_hash",
    ]:
        if not _hash_ok(report.get(key)):
            errors.append(f"hash_missing:{key}")
    if require_verified and report.get("stage_hidden_lm_head_token_selection_verified") is not True:
        errors.append("stage_hidden_lm_head_verification_required")

    blockers = set(str(item) for item in _list(report.get("blockers")))
    for blocker in [
        "glm52_stage_hidden_lm_head_is_single_layer_only",
        "glm52_stage_hidden_lm_head_uses_small_sequence_topk_cap",
        "glm52_stage_hidden_lm_head_is_not_full_model_hidden",
        "glm52_stage_hidden_lm_head_is_not_stage_decode",
        "glm52_stage_hidden_lm_head_is_not_same_request",
        "glm52_stage_decode_not_verified",
        "glm52_same_request_decode_not_verified",
    ]:
        if blocker not in blockers:
            errors.append(f"blocker_missing:{blocker}")
    boundary = _dict(report.get("completion_boundary"))
    for key in [
        "stage_hidden_lm_head_is_single_layer_only",
        "stage_hidden_lm_head_uses_small_sequence_topk_cap",
        "stage_hidden_lm_head_is_not_full_model_hidden",
        "stage_hidden_lm_head_is_not_stage_decode",
        "stage_hidden_lm_head_is_not_same_request",
        "requires_multi_layer_stage_runtime",
        "requires_full_model_or_stage_hidden",
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
        "stage_hidden_lm_head_token_selection_verified": report.get("stage_hidden_lm_head_token_selection_verified") is True,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_stage_hidden_lm_head_check: ok={result['ok']} "
            f"errors={len(errors)} stage_hidden_lm_head={result['stage_hidden_lm_head_token_selection_verified']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
