#!/usr/bin/env python3
"""Validate GLM 5.2 pack-quantized group probe reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_pack_quantized_group_probe as probe  # noqa: E402


SCHEMA = "glm52_pack_quantized_group_check_v1"


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


def validate_report(report: dict[str, Any], *, require_loaded: bool = False) -> list[str]:
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
    if report.get("pack_quantized_group_dequantized") is not False:
        errors.append("pack_group_dequant_overclaim")
    if report.get("stage_decode_verified") is not False:
        errors.append("stage_decode_overclaim")
    if set(_list(report.get("loaded_fields"))) != set(probe.PACK_FIELDS):
        errors.append("loaded_fields_incomplete")
    if _int(report.get("group_tensor_count")) != len(probe.PACK_FIELDS):
        errors.append("group_tensor_count_mismatch")
    if _int(report.get("group_value_total_bytes")) <= 0:
        errors.append("group_value_total_bytes_missing")
    if not _hash_ok(report.get("group_value_hash")):
        errors.append("group_value_hash_missing")
    for item in _list(report.get("tensor_summaries")):
        if not isinstance(item, dict):
            continue
        label = str(item.get("field") or "missing")
        if item.get("value_loaded") is not True:
            errors.append(f"tensor_value_not_loaded:{label}")
        if not _hash_ok(item.get("value_sha256")):
            errors.append(f"tensor_value_hash_missing:{label}")
        if _int(item.get("tensor_nbytes")) <= 0:
            errors.append(f"tensor_nbytes_missing:{label}")
    if require_loaded and report.get("pack_quantized_group_loaded") is not True:
        errors.append("pack_quantized_group_load_required")

    boundary = _dict(report.get("completion_boundary"))
    for key in [
        "pack_group_load_is_not_dequant_success",
        "weight_value_hash_is_not_raw_value_publication",
        "requires_dequant_linear_runtime",
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
    parser.add_argument("--require-loaded", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = load_json(Path(args.report))
    errors = validate_report(report, require_loaded=bool(args.require_loaded))
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors,
        "pack_quantized_group_loaded": report.get("pack_quantized_group_loaded") is True,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_pack_quantized_group_check: ok={result['ok']} "
            f"errors={len(errors)} loaded={result['pack_quantized_group_loaded']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
