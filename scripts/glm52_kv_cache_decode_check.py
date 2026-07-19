#!/usr/bin/env python3
"""Validate GLM 5.2 KV-cache decode probe reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_kv_cache_decode_probe as probe  # noqa: E402


SCHEMA = "glm52_kv_cache_decode_check_v1"


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


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
    for key in ["kv_cache_prefill_verified", "kv_cache_update_verified", "kv_cache_decode_attention_verified", "o_proj_verified"]:
        if report.get(key) is not True:
            errors.append(f"{key}_missing")
    for key in ["stage_decode_verified", "generated_token_verified"]:
        if report.get(key) is not False:
            errors.append(f"{key}_overclaim")
    heads = _int(report.get("num_attention_heads"))
    qk = _int(report.get("qk_head_dim"))
    value = _int(report.get("v_head_dim"))
    prefill = _int(report.get("prefill_length"))
    updated = _int(report.get("updated_cache_length"))
    hidden = _int(report.get("hidden_size"))
    expected_shapes = {
        "prefill_key_cache_shape": [prefill, heads, qk],
        "prefill_value_cache_shape": [prefill, heads, value],
        "updated_key_cache_shape": [updated, heads, qk],
        "updated_value_cache_shape": [updated, heads, value],
        "decode_query_shape": [heads, qk],
        "attention_scores_shape": [heads, updated],
        "attention_weights_shape": [heads, updated],
        "head_output_shape": [heads, value],
        "attention_flattened_shape": [heads * value],
        "o_proj_output_shape": [hidden],
    }
    for key, expected in expected_shapes.items():
        if _list(report.get(key)) != expected:
            errors.append(f"shape_mismatch:{key}")
    for key in [
        "prefill_key_cache_hash",
        "prefill_value_cache_hash",
        "updated_key_cache_hash",
        "updated_value_cache_hash",
        "decode_query_hash",
        "attention_scores_hash",
        "attention_weights_hash",
        "head_output_hash",
        "o_proj_output_hash",
    ]:
        if not _hash_ok(report.get(key)):
            errors.append(f"hash_missing:{key}")
    if len(_list(report.get("o_proj_weight_shape"))) != 2:
        errors.append("o_proj_weight_shape_invalid")
    if require_verified and report.get("kv_cache_decode_attention_verified") is not True:
        errors.append("kv_cache_decode_verification_required")
    boundary = report.get("completion_boundary") if isinstance(report.get("completion_boundary"), dict) else {}
    for key in [
        "kv_cache_decode_is_not_dsa_masked_attention",
        "kv_cache_decode_is_not_transformer_block",
        "kv_cache_decode_is_not_stage_decode",
        "requires_mlp_residual_runtime",
        "requires_lm_head_token_selection",
        "requires_stage_decode_verified",
    ]:
        if boundary.get(key) is not True:
            errors.append(f"completion_boundary_missing:{key}")
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
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
        "kv_cache_decode_attention_verified": report.get("kv_cache_decode_attention_verified") is True,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_kv_cache_decode_check: ok={result['ok']} "
            f"errors={len(errors)} kv_decode={result['kv_cache_decode_attention_verified']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
