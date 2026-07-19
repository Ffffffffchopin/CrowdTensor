#!/usr/bin/env python3
"""Validate the DeepSeek-V4-Flash torch stage adapter smoke artifact."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import deepseek_v4_flash_torch_stage_adapter_smoke as smoke  # noqa: E402


SCHEMA = "deepseek_v4_flash_torch_stage_adapter_smoke_check_v1"


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


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != smoke.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("ok") is not True:
        errors.append("report_not_ok")
    if report.get("deepseek_v4_flash_torch_stage_adapter_smoke_ready") is not True:
        errors.append("smoke_ready_missing")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    if smoke.public_redaction_errors(report):
        errors.append("public_redaction_scan_failed")

    model = _dict(report.get("model"))
    if model.get("model_id") != "deepseek-ai/DeepSeek-V4-Flash":
        errors.append("model_id_mismatch")
    if model.get("model_type") != "deepseek_v4":
        errors.append("model_type_mismatch")
    if model.get("full_model_weight_values_loaded") is not False:
        errors.append("full_model_weight_overclaimed")
    if report.get("real_deepseek_weights_loaded") is not False:
        errors.append("real_weight_load_overclaimed")
    if report.get("jax_tpu_translation_ready") is not False:
        errors.append("jax_tpu_translation_overclaimed")

    reference = _dict(report.get("reference_stage"))
    if reference.get("ok") is not True:
        errors.append("reference_stage_not_ok")
    if reference.get("transformers_reference_used") is not True:
        errors.append("transformers_reference_missing")
    if reference.get("model_type") != "deepseek_v4":
        errors.append("reference_model_type_mismatch")
    if reference.get("tiny_fixture") is not True:
        errors.append("tiny_fixture_flag_missing")
    if _int(reference.get("stage_owned_key_count")) < 1:
        errors.append("stage_owned_keys_missing")

    components = _dict(reference.get("real_deepseek_v4_components_exercised"))
    for component in [
        "manifold_hyper_connections",
        "compressed_attention",
        "mla_shared_kv_attention",
        "grouped_output_projection",
        "moe_router",
        "routed_experts",
        "shared_experts",
        "stage_local_kv_cache_shape",
    ]:
        if components.get(component) is not True:
            errors.append(f"component_missing:{component}")

    output = _dict(reference.get("output_summary"))
    for field in ["mean", "std", "min", "max"]:
        if not _finite_number(output.get(field)):
            errors.append(f"non_finite_output_summary:{field}")
    if not _list(output.get("shape")):
        errors.append("output_shape_missing")
    if output.get("payload_public") is not False:
        errors.append("output_payload_public_flag_mismatch")

    safety = _dict(report.get("safety"))
    for flag in [
        "raw_prompt_public",
        "raw_generated_text_public",
        "generated_token_ids_public",
        "activation_public",
        "hidden_state_public",
        "logits_public",
        "kv_cache_public",
        "past_key_values_public",
        "credentials_public",
        "cookies_public",
        "weight_tensor_values_public",
    ]:
        if safety.get(flag) is not False:
            errors.append(f"safety_flag_mismatch:{flag}")
    if safety.get("public_artifact_safe") is not True:
        errors.append("safety_public_artifact_safe_missing")

    if _dict(_dict(report.get("artifacts")).get("summary_json")).get("present") is not True:
        errors.append("summary_artifact_missing")
    return sorted(set(errors))


def build_check(args: argparse.Namespace) -> dict[str, Any]:
    report = load_json(Path(args.report))
    errors = validate_report(report)
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "report_schema": report.get("schema"),
        "report_path": args.report,
        "torch_stage_adapter_smoke_ready": report.get("deepseek_v4_flash_torch_stage_adapter_smoke_ready") is True,
        "jax_tpu_translation_ready": report.get("jax_tpu_translation_ready") is True,
        "real_deepseek_weights_loaded": report.get("real_deepseek_weights_loaded") is True,
        "errors": errors,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_check(args)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Check ok: {result['ok']}")
        print(f"Report: {result['report_path']}")
        if result["errors"]:
            print("Errors: " + ", ".join(result["errors"]))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
