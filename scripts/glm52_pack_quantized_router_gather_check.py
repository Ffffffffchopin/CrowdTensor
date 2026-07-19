#!/usr/bin/env python3
"""Validate GLM 5.2 router + routed expert gather subset probe reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_pack_quantized_router_gather_probe as probe  # noqa: E402


SCHEMA = "glm52_pack_quantized_router_gather_check_v1"


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
    if report.get("router_topk_verified") is not True:
        errors.append("router_topk_not_verified")
    if report.get("routed_expert_subset_verified") is not True:
        errors.append("routed_expert_subset_not_verified")
    if report.get("stage_decode_verified") is not False:
        errors.append("stage_decode_overclaim")
    if _int(report.get("router_topk_count")) != _int(report.get("num_experts_per_tok")):
        errors.append("router_topk_count_mismatch")
    if _int(report.get("executed_expert_count")) <= 0:
        errors.append("executed_expert_count_missing")
    if len(_list(report.get("executed_experts"))) != _int(report.get("executed_expert_count")):
        errors.append("executed_expert_summary_count_mismatch")
    if _list(report.get("routed_subset_output_shape")) != [_int(report.get("hidden_size"))]:
        errors.append("routed_subset_output_shape_mismatch")
    for key in ["router_topk_indices_hash", "router_topk_weights_hash", "routed_subset_output_hash"]:
        if not _hash_ok(report.get(key)):
            errors.append(f"hash_missing:{key}")
    for item in _list(report.get("executed_experts")):
        if not isinstance(item, dict):
            errors.append("executed_expert_entry_invalid")
            continue
        if _int(item.get("expert_id"), -1) < 0:
            errors.append("executed_expert_id_invalid")
        if not _hash_ok(item.get("expert_weight_hash")):
            errors.append("executed_expert_weight_hash_missing")
        if not _hash_ok(item.get("expert_output_hash")):
            errors.append("executed_expert_output_hash_missing")
        if _list(item.get("expert_output_shape")) != [_int(report.get("hidden_size"))]:
            errors.append("executed_expert_output_shape_mismatch")
    if require_verified and report.get("routed_expert_subset_verified") is not True:
        errors.append("router_gather_verification_required")
    boundary = report.get("completion_boundary") if isinstance(report.get("completion_boundary"), dict) else {}
    for key in [
        "routed_subset_is_not_full_moe_layer",
        "shared_experts_not_included",
        "attention_not_included",
        "stage_decode_not_included",
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
        "routed_expert_subset_verified": report.get("routed_expert_subset_verified") is True,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_pack_quantized_router_gather_check: ok={result['ok']} "
            f"errors={len(errors)} routed_subset={result['routed_expert_subset_verified']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
