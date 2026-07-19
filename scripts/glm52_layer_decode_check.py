#!/usr/bin/env python3
"""Validate GLM 5.2 single-layer decode composition reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_layer_decode_probe as probe  # noqa: E402


SCHEMA = "glm52_layer_decode_check_v1"


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
    for key in [
        "kv_cache_prefill_verified",
        "kv_cache_update_verified",
        "attention_decode_verified",
        "attention_residual_verified",
        "post_attention_norm_verified",
        "router_topk_verified",
        "routed_expert_gather_verified",
        "shared_experts_mlp_verified",
        "full_moe_mlp_verified",
        "layer_decode_verified",
    ]:
        if report.get(key) is not True:
            errors.append(f"{key}_missing")
    for key in [
        "dsa_masked_attention_integrated",
        "multi_layer_stage_runtime_verified",
        "lm_head_verified",
        "generated_token_verified",
        "stage_decode_verified",
        "same_request_decode_verified",
    ]:
        if report.get(key) is not False:
            errors.append(f"{key}_overclaim")

    hidden = _int(report.get("hidden_size"))
    heads = _int(report.get("num_attention_heads"))
    qk = _int(report.get("qk_head_dim"))
    value = _int(report.get("v_head_dim"))
    prefill = _int(report.get("prefill_length"))
    updated = _int(report.get("updated_cache_length"))
    expected_shapes = {
        "decode_input_shape": [hidden],
        "prefill_key_cache_shape": [prefill, heads, qk],
        "prefill_value_cache_shape": [prefill, heads, value],
        "updated_key_cache_shape": [updated, heads, qk],
        "updated_value_cache_shape": [updated, heads, value],
        "decode_query_shape": [heads, qk],
        "attention_scores_shape": [heads, updated],
        "attention_weights_shape": [heads, updated],
        "attention_head_output_shape": [heads, value],
        "attention_flattened_shape": [heads * value],
        "attention_output_shape": [hidden],
        "attention_residual_shape": [hidden],
        "post_attention_norm_shape": [hidden],
        "routed_output_shape": [hidden],
        "shared_output_shape": [hidden],
        "full_moe_output_shape": [hidden],
        "layer_output_shape": [hidden],
    }
    for key, expected in expected_shapes.items():
        if _list(report.get(key)) != expected:
            errors.append(f"shape_mismatch:{key}")
    if _int(report.get("router_topk_count")) != _int(report.get("num_experts_per_tok")):
        errors.append("router_topk_count_mismatch")
    if _int(report.get("executed_expert_count")) != _int(report.get("num_experts_per_tok")):
        errors.append("executed_expert_count_mismatch")
    executed = [item for item in _list(report.get("executed_experts")) if isinstance(item, dict)]
    if len(executed) != _int(report.get("executed_expert_count")):
        errors.append("executed_expert_summary_count_mismatch")
    for item in executed:
        if _int(item.get("expert_id"), -1) < 0:
            errors.append("executed_expert_id_invalid")
        if item.get("expert_weight_hash") and not _hash_ok(item.get("expert_weight_hash")):
            errors.append("executed_expert_weight_hash_invalid")
        if item.get("expert_output_hash") and not _hash_ok(item.get("expert_output_hash")):
            errors.append("executed_expert_output_hash_invalid")
        if _list(item.get("expert_output_shape")) != [hidden]:
            errors.append("executed_expert_output_shape_mismatch")
    shared = [item for item in _list(report.get("shared_projection_summaries")) if isinstance(item, dict)]
    if [str(item.get("projection") or "") for item in shared] != ["gate_proj", "up_proj", "down_proj"]:
        errors.append("shared_projection_order_invalid")
    for item in shared:
        projection = str(item.get("projection") or "missing")
        if len(_list(item.get("weight_shape"))) != 2:
            errors.append(f"shared_weight_shape_invalid:{projection}")
        if len(_list(item.get("output_shape"))) != 1:
            errors.append(f"shared_output_shape_invalid:{projection}")
        if not _hash_ok(item.get("output_hash")):
            errors.append(f"shared_output_hash_missing:{projection}")
    for key in [
        "decode_input_hash",
        "prefill_key_cache_hash",
        "updated_key_cache_hash",
        "decode_query_hash",
        "attention_scores_hash",
        "attention_weights_hash",
        "attention_output_hash",
        "attention_residual_hash",
        "post_attention_norm_hash",
        "router_topk_indices_hash",
        "router_topk_weights_hash",
        "routed_output_hash",
        "shared_output_hash",
        "full_moe_output_hash",
        "layer_output_hash",
    ]:
        if not _hash_ok(report.get(key)):
            errors.append(f"hash_missing:{key}")
    if require_verified and report.get("layer_decode_verified") is not True:
        errors.append("layer_decode_verification_required")
    blockers = set(str(item) for item in _list(report.get("blockers")))
    for blocker in [
        "glm52_layer_decode_is_single_layer_only",
        "glm52_layer_decode_uses_basic_attention_not_dsa_masked_attention",
        "glm52_layer_decode_missing_lm_head",
        "glm52_layer_decode_is_not_stage_decode",
        "glm52_layer_decode_is_not_same_request",
    ]:
        if blocker not in blockers:
            errors.append(f"blocker_missing:{blocker}")
    boundary = report.get("completion_boundary") if isinstance(report.get("completion_boundary"), dict) else {}
    for key in [
        "layer_decode_is_single_layer_only",
        "layer_decode_uses_basic_attention_not_dsa_masked_attention",
        "layer_decode_is_not_stage_decode",
        "layer_decode_is_not_same_request",
        "requires_multi_layer_stage_runtime",
        "requires_dsa_masked_attention_integration",
        "requires_lm_head_token_selection",
        "requires_kaggle_cpu_gpu_tpu_same_request",
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
        "layer_decode_verified": report.get("layer_decode_verified") is True,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_layer_decode_check: ok={result['ok']} "
            f"errors={len(errors)} layer_decode={result['layer_decode_verified']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
