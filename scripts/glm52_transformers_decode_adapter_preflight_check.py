#!/usr/bin/env python3
"""Validate GLM 5.2 transformers decode adapter preflight reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_transformers_decode_adapter_preflight as preflight  # noqa: E402


SCHEMA = "glm52_transformers_decode_adapter_preflight_check_v1"


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


def validate_report(report: dict[str, Any], *, require_foundation: bool = False, require_decode_ready: bool = False) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != preflight.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("ok") is not True:
        errors.append("report_not_ok")
    if report.get("glm52_transformers_decode_adapter_preflight_ready") is not True:
        errors.append("preflight_ready_missing")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = preflight.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))

    model = _dict(report.get("model"))
    if model.get("model_id") != preflight.MODEL_ID:
        errors.append("model_id_mismatch")
    if model.get("config_ready") is not True:
        errors.append("config_not_ready")
    if model.get("index_ready") is not True:
        errors.append("index_not_ready")
    if model.get("model_type") != "glm_moe_dsa":
        errors.append("model_type_not_glm_moe_dsa")
    if _int(model.get("num_hidden_layers")) <= 0:
        errors.append("layer_count_missing")
    if "quant" not in str(model.get("quantization_format") or "").lower():
        errors.append("quantization_format_not_pack_quantized")
    if 4 not in [int(item) for item in _list(model.get("quantization_weight_bits")) if str(item).isdigit()]:
        errors.append("quantization_bits_not_int4")

    transformer = _dict(report.get("transformers_runtime"))
    for key in [
        "transformers_available",
        "glm_moe_dsa_config_class_available",
        "glm_moe_dsa_model_class_available",
        "awq_config_normalized_ready",
        "tiny_forward_ready",
    ]:
        if transformer.get(key) is not True:
            errors.append(f"transformers_runtime_missing:{key}")
    if transformer.get("tiny_forward_logits_shape") and _list(transformer.get("tiny_forward_logits_shape"))[-1] != 32:
        errors.append("tiny_forward_logits_shape_unexpected")

    mapping = _dict(report.get("stage_weight_mapping"))
    if mapping.get("stage_weight_mapping_ready") is not True:
        errors.append("stage_weight_mapping_not_ready")
    if _int(mapping.get("required_key_count")) <= 0:
        errors.append("stage_weight_mapping_required_keys_missing")
    if _int(mapping.get("missing_required_key_count")) != 0:
        errors.append("stage_weight_mapping_missing_keys_present")
    if _int(mapping.get("pack_required_key_count")) <= 0:
        errors.append("stage_weight_mapping_pack_keys_missing")
    if _int(mapping.get("sparse_layer_count")) <= 0:
        errors.append("stage_weight_mapping_sparse_layers_missing")

    pack_runtime = _dict(report.get("pack_quantized_runtime"))
    dependencies = [item for item in _list(pack_runtime.get("dependencies")) if isinstance(item, dict)]
    if not dependencies:
        errors.append("pack_runtime_dependencies_missing")
    if report.get("decode_adapter_ready") is True:
        if report.get("adapter_foundation_ready") is not True:
            errors.append("decode_ready_without_foundation")
        if pack_runtime.get("ready") is not True:
            errors.append("decode_ready_without_pack_runtime")
        if report.get("blockers"):
            errors.append("decode_ready_with_blockers")
    else:
        if not report.get("blockers"):
            errors.append("not_ready_missing_blockers")
    if require_foundation and report.get("adapter_foundation_ready") is not True:
        errors.append("adapter_foundation_not_ready")
    if require_decode_ready and report.get("decode_adapter_ready") is not True:
        errors.append("decode_adapter_not_ready")

    boundary = _dict(report.get("completion_boundary"))
    for key in [
        "preflight_is_not_decode_success",
        "tiny_random_forward_is_not_glm52_inference",
        "weight_mapping_is_not_weight_loading",
        "requires_pack_quantized_dequant_runtime",
        "requires_stage_decode_verified",
        "requires_same_request_generated_token_hash",
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
    parser.add_argument("--require-foundation", action="store_true")
    parser.add_argument("--require-decode-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = load_json(Path(args.report))
    errors = validate_report(
        report,
        require_foundation=bool(args.require_foundation),
        require_decode_ready=bool(args.require_decode_ready),
    )
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors,
        "adapter_foundation_ready": report.get("adapter_foundation_ready") is True,
        "decode_adapter_ready": report.get("decode_adapter_ready") is True,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_transformers_decode_adapter_preflight_check: ok={result['ok']} "
            f"errors={len(errors)} foundation={result['adapter_foundation_ready']} "
            f"decode={result['decode_adapter_ready']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
