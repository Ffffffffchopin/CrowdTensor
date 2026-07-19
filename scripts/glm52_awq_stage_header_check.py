#!/usr/bin/env python3
"""Validate GLM 5.2 AWQ stage-owned safetensors header evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_awq_stage_header_probe as probe  # noqa: E402


SCHEMA = "glm52_awq_stage_header_check_v1"


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def validate_report(report: dict[str, Any], *, require_ready: bool = False) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != probe.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = probe.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))
    if report.get("model_repo") != probe.DEFAULT_MODEL_REPO:
        errors.append("model_repo_mismatch")
    if report.get("base_model_id") != probe.BASE_MODEL_ID:
        errors.append("base_model_id_mismatch")
    if report.get("model_type") not in {"glm_moe_dsa", ""}:
        errors.append("model_type_unexpected")
    if report.get("weight_tensor_values_loaded") is not False:
        errors.append("weight_tensor_values_loaded_overclaim")
    if report.get("weight_tensor_values_public") is not False:
        errors.append("weight_tensor_values_public_unsafe")
    if report.get("safetensors_header_payload_public") is not False:
        errors.append("safetensors_header_payload_public_unsafe")
    if report.get("stage_runtime_adapter_verified") is not False:
        errors.append("stage_runtime_adapter_overclaim")
    if report.get("same_request_route_verified") is not False:
        errors.append("same_request_route_overclaim")
    safety = _dict(report.get("safety"))
    for key in [
        "credentials_public",
        "signed_url_public",
        "weight_tensor_values_public",
        "safetensors_header_payload_public",
        "activation_public",
        "generated_token_ids_public",
    ]:
        if safety.get(key) is not False:
            errors.append(f"safety_flag_not_false:{key}")
    ready = report.get("glm52_awq_stage_header_ready") is True
    if ready:
        if report.get("ok") is not True:
            errors.append("ready_but_report_not_ok")
        if _int(report.get("assigned_weight_key_count")) <= 0:
            errors.append("assigned_weight_key_count_missing")
        if _int(report.get("present_stage_key_count")) != _int(report.get("assigned_weight_key_count")):
            errors.append("present_stage_key_count_mismatch")
        if _int(report.get("missing_stage_key_count")) != 0:
            errors.append("missing_stage_keys_present")
        if _int(report.get("header_file_count")) <= 0:
            errors.append("header_file_count_missing")
        if not _dict(report.get("dtype_counts")):
            errors.append("dtype_counts_missing")
        if _int(report.get("total_selected_tensor_storage_bytes")) <= 0:
            errors.append("selected_tensor_storage_bytes_missing")
        family_hits = _dict(report.get("stage_family_hits"))
        if family_hits.get("awq_quantized_tensors") is not True:
            errors.append("awq_quantized_tensor_family_missing")
    elif require_ready:
        errors.append("stage_header_not_ready")
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = load_json(Path(args.report))
    errors = validate_report(report, require_ready=bool(args.require_ready))
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors,
        "stage_header_ready": report.get("glm52_awq_stage_header_ready") is True,
        "stage_id": report.get("stage_id"),
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_awq_stage_header_check: ok={result['ok']} "
            f"errors={len(errors)} ready={result['stage_header_ready']} stage={result['stage_id']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
