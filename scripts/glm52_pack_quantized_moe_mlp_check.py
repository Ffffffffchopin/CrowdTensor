#!/usr/bin/env python3
"""Validate GLM 5.2 routed-plus-shared MoE MLP probe reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_pack_quantized_moe_mlp_probe as probe  # noqa: E402


SCHEMA = "glm52_pack_quantized_moe_mlp_check_v1"


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
        "router_topk_verified",
        "routed_expert_gather_verified",
        "shared_experts_mlp_verified",
        "pack_quantized_moe_mlp_verified",
        "full_moe_mlp_verified",
    ]:
        if report.get(key) is not True:
            errors.append(f"{key}_missing")
    if report.get("stage_decode_verified") is not False:
        errors.append("stage_decode_overclaim")
    hidden_size = _int(report.get("hidden_size"))
    if _int(report.get("router_topk_count")) != _int(report.get("num_experts_per_tok")):
        errors.append("router_topk_count_mismatch")
    if _int(report.get("executed_expert_count")) != _int(report.get("num_experts_per_tok")):
        errors.append("executed_expert_count_not_full_topk")
    executed = [item for item in _list(report.get("executed_experts")) if isinstance(item, dict)]
    if len(executed) != _int(report.get("executed_expert_count")):
        errors.append("executed_expert_summary_count_mismatch")
    for item in executed:
        if _int(item.get("expert_id"), -1) < 0:
            errors.append("executed_expert_id_invalid")
        if not _hash_ok(item.get("expert_weight_hash")):
            errors.append("executed_expert_weight_hash_missing")
        if not _hash_ok(item.get("expert_output_hash")):
            errors.append("executed_expert_output_hash_missing")
        if _list(item.get("expert_output_shape")) != [hidden_size]:
            errors.append("executed_expert_output_shape_mismatch")
    shared = [item for item in _list(report.get("shared_projection_summaries")) if isinstance(item, dict)]
    if [str(item.get("projection") or "") for item in shared] != probe.PROJECTIONS:
        errors.append("shared_projection_order_invalid")
    for item in shared:
        projection = str(item.get("projection") or "missing")
        if str(item.get("weight_dtype") or "") not in {"bfloat16", "float32", "float16"}:
            errors.append(f"shared_projection_dtype_invalid:{projection}")
        if len(_list(item.get("weight_shape"))) != 2:
            errors.append(f"shared_projection_weight_shape_invalid:{projection}")
        if len(_list(item.get("output_shape"))) != 1:
            errors.append(f"shared_projection_output_shape_invalid:{projection}")
        if not _hash_ok(item.get("output_hash")):
            errors.append(f"shared_projection_output_hash_missing:{projection}")
    for key in ["router_topk_indices_hash", "router_topk_weights_hash", "routed_output_hash", "shared_output_hash", "full_moe_output_hash"]:
        if not _hash_ok(report.get(key)):
            errors.append(f"hash_missing:{key}")
    for key in ["routed_output_shape", "shared_output_shape", "full_moe_output_shape"]:
        if _list(report.get(key)) != [hidden_size]:
            errors.append(f"shape_mismatch:{key}")
    if require_verified and report.get("full_moe_mlp_verified") is not True:
        errors.append("moe_mlp_verification_required")
    boundary = report.get("completion_boundary") if isinstance(report.get("completion_boundary"), dict) else {}
    for key in [
        "full_moe_mlp_is_not_attention",
        "full_moe_mlp_is_not_transformer_block",
        "full_moe_mlp_is_not_stage_decode",
        "requires_attention_runtime",
        "requires_residual_norm_runtime",
        "requires_stage_local_kv_cache",
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
        "full_moe_mlp_verified": report.get("full_moe_mlp_verified") is True,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_pack_quantized_moe_mlp_check: ok={result['ok']} "
            f"errors={len(errors)} full_moe_mlp={result['full_moe_mlp_verified']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
