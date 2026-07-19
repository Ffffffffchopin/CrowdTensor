#!/usr/bin/env python3
"""Validate the DeepSeek-V4 JAX stage adapter smoke artifact."""

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

from scripts import deepseek_v4_flash_jax_stage_adapter_smoke as smoke  # noqa: E402


SCHEMA = "deepseek_v4_flash_jax_stage_adapter_smoke_check_v1"


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


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != smoke.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    if smoke.public_redaction_errors(report):
        errors.append("public_redaction_scan_failed")

    model = _dict(report.get("model"))
    if model.get("model_id") != "deepseek-ai/DeepSeek-V4-Flash":
        errors.append("model_id_mismatch")
    if model.get("model_type") != "deepseek_v4":
        errors.append("model_type_mismatch")
    if model.get("fixture_config") is not True:
        errors.append("fixture_config_flag_missing")
    if model.get("real_deepseek_weights_loaded") is not False or report.get("real_deepseek_weights_loaded") is not False:
        errors.append("real_weight_load_overclaimed")

    stage = _dict(report.get("stage"))
    if stage.get("stage_type") != "decoder_layer_fixture_translation":
        errors.append("stage_type_mismatch")
    if _int(stage.get("stage_owned_key_count")) < 20:
        errors.append("stage_owned_key_count_too_low")
    shape = _dict(stage.get("shape_metadata"))
    if shape.get("layout") != "batch_seq_hc_hidden":
        errors.append("stage_layout_mismatch")

    numpy_ref = _dict(report.get("numpy_reference"))
    if numpy_ref.get("ok") is not True:
        errors.append("numpy_reference_not_ok")
    components = _dict(numpy_ref.get("components_exercised"))
    for component in [
        "manifold_hyper_connections",
        "mla_shared_kv_attention",
        "grouped_output_projection",
        "attention_sink",
        "topk_moe_router",
        "routed_experts",
        "shared_experts",
        "hca_compressor_shape_metadata",
    ]:
        if components.get(component) is not True:
            errors.append(f"component_missing:{component}")
    output = _dict(numpy_ref.get("output_summary"))
    for field in ["mean", "std", "min", "max"]:
        if not _finite(output.get(field)):
            errors.append(f"non_finite_numpy_summary:{field}")
    kv = _dict(numpy_ref.get("stage_local_kv_cache_metadata"))
    if kv.get("stage_local_only") is not True:
        errors.append("kv_cache_not_stage_local")
    if kv.get("kv_payload_public") is not False or kv.get("past_key_values_public") is not False:
        errors.append("kv_cache_public_flag_mismatch")

    ready = report.get("deepseek_v4_flash_jax_stage_adapter_smoke_ready") is True
    jax_ready = report.get("jax_runtime_execution_ready") is True
    tpu_ready = report.get("tpu_runtime_ready") is True
    if ready:
        if report.get("ok") is not True:
            errors.append("ready_report_not_ok")
        if jax_ready is not True:
            errors.append("ready_without_jax_runtime")
        if report.get("deepseek_v4_jax_stage_forward_ready") is not True:
            errors.append("ready_without_jax_stage_forward")
        if report.get("tpu_runtime_required") is True and tpu_ready is not True:
            errors.append("ready_without_required_tpu")
        if _list(report.get("blockers")):
            errors.append("ready_with_blockers")
    else:
        if report.get("ok") is True:
            errors.append("not_ready_report_ok")
        if report.get("tpu_runtime_required") is True and "jax_tpu_device_missing" not in set(report.get("blockers") or []) and tpu_ready is not True:
            errors.append("required_tpu_blocker_missing")
        if report.get("jax_runtime_execution_requested") is True and jax_ready is not True:
            blockers = set(str(item) for item in _list(report.get("blockers")))
            if not blockers.intersection({"jax_missing", "jax_tpu_device_missing"}):
                errors.append("jax_requested_blocker_missing")

    if report.get("deepseek_v4_jax_tpu_stage_forward_ready") is True and tpu_ready is not True:
        errors.append("tpu_stage_forward_overclaimed")
    if report.get("deepseek_v4_jax_stage_forward_ready") is True and jax_ready is not True:
        errors.append("jax_stage_forward_overclaimed")

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
        "jax_stage_adapter_smoke_ready": report.get("deepseek_v4_flash_jax_stage_adapter_smoke_ready") is True,
        "jax_runtime_execution_ready": report.get("jax_runtime_execution_ready") is True,
        "tpu_runtime_ready": report.get("tpu_runtime_ready") is True,
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
