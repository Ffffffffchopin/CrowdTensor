#!/usr/bin/env python3
"""Validate GLM 5.2 pack-quantized expert MLP probe reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_pack_quantized_expert_mlp_probe as probe  # noqa: E402


SCHEMA = "glm52_pack_quantized_expert_mlp_check_v1"


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
    if report.get("pack_quantized_expert_mlp_verified") is not True:
        errors.append("pack_quantized_expert_mlp_not_verified")
    if report.get("single_expert_mlp_verified") is not True:
        errors.append("single_expert_mlp_not_verified")
    if report.get("stage_decode_verified") is not False:
        errors.append("stage_decode_overclaim")
    projections = [item for item in _list(report.get("projection_summaries")) if isinstance(item, dict)]
    if [item.get("projection") for item in projections] != probe.PROJECTIONS:
        errors.append("projection_summary_order_invalid")
    for item in projections:
        projection = str(item.get("projection") or "missing")
        if item.get("pack_quantized_group_loaded") is not True:
            errors.append(f"projection_group_not_loaded:{projection}")
        if not _hash_ok(item.get("output_hash")):
            errors.append(f"projection_output_hash_missing:{projection}")
        if len(_list(item.get("weight_shape"))) != 2:
            errors.append(f"projection_weight_shape_invalid:{projection}")
        if len(_list(item.get("output_shape"))) != 1:
            errors.append(f"projection_output_shape_invalid:{projection}")
    if _list(report.get("final_output_shape")) != [_int(report.get("hidden_size"))]:
        errors.append("final_output_shape_mismatch")
    if not _hash_ok(report.get("final_output_hash")):
        errors.append("final_output_hash_missing")
    if require_verified and report.get("pack_quantized_expert_mlp_verified") is not True:
        errors.append("expert_mlp_verification_required")
    boundary = report.get("completion_boundary") if isinstance(report.get("completion_boundary"), dict) else {}
    for key in [
        "single_expert_mlp_is_not_full_moe_layer",
        "single_expert_mlp_is_not_attention",
        "single_expert_mlp_is_not_topk_router",
        "single_expert_mlp_is_not_stage_decode",
        "requires_transformer_block_runtime",
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
        "pack_quantized_expert_mlp_verified": report.get("pack_quantized_expert_mlp_verified") is True,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_pack_quantized_expert_mlp_check: ok={result['ok']} "
            f"errors={len(errors)} expert_mlp={result['pack_quantized_expert_mlp_verified']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
