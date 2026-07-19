#!/usr/bin/env python3
"""Validate GLM 5.2 pack-quantized dequant slice probe reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_pack_quantized_dequant_probe as probe  # noqa: E402


SCHEMA = "glm52_pack_quantized_dequant_check_v1"


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
    if "quant" not in str(report.get("quantization_format") or "").lower():
        errors.append("quantization_format_not_pack_quantized")
    if report.get("pack_quantized_group_loaded") is not True:
        errors.append("pack_quantized_group_not_loaded")
    if report.get("pack_quantized_dequant_verified") is not True:
        errors.append("pack_quantized_dequant_not_verified")
    if report.get("pack_quantized_linear_slice_verified") is not True:
        errors.append("pack_quantized_linear_slice_not_verified")
    if report.get("stage_decode_verified") is not False:
        errors.append("stage_decode_overclaim")
    for key in ["q_unpacked_hash", "zero_point_unpacked_hash", "dequant_slice_hash", "linear_slice_hash"]:
        if not _hash_ok(report.get(key)):
            errors.append(f"hash_missing:{key}")
    if len(_list(report.get("dequant_slice_shape"))) != 2:
        errors.append("dequant_slice_shape_invalid")
    if len(_list(report.get("linear_slice_shape"))) != 1:
        errors.append("linear_slice_shape_invalid")
    if _int(report.get("row_count")) <= 0 or _int(report.get("group_count")) <= 0:
        errors.append("slice_config_invalid")
    if require_verified and report.get("pack_quantized_dequant_verified") is not True:
        errors.append("dequant_verification_required")

    boundary = _dict(report.get("completion_boundary"))
    for key in [
        "dequant_slice_is_not_full_layer",
        "linear_slice_is_not_stage_decode",
        "weight_values_not_public",
        "requires_full_projection_runtime",
        "requires_transformer_block_runtime",
        "requires_stage_decode_verified",
    ]:
        if boundary.get(key) is not True:
            errors.append(f"completion_boundary_missing:{key}")

    safety = _dict(report.get("safety"))
    if safety.get("public_artifact_safe") is not True:
        errors.append("safety_public_artifact_safe_missing")
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
        "pack_quantized_dequant_verified": report.get("pack_quantized_dequant_verified") is True,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_pack_quantized_dequant_check: ok={result['ok']} "
            f"errors={len(errors)} dequant={result['pack_quantized_dequant_verified']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
