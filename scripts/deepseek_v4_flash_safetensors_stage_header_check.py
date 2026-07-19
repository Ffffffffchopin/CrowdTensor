#!/usr/bin/env python3
"""Validate DeepSeek-V4-Flash safetensors stage header evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import deepseek_v4_flash_safetensors_stage_header_probe as probe  # noqa: E402


SCHEMA = "deepseek_v4_flash_safetensors_stage_header_check_v1"


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


def validate_report(report: dict[str, Any], *, require_ready: bool = False) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != probe.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("ok") is not True:
        errors.append("report_not_ok")
    if report.get("deepseek_v4_flash_safetensors_stage_header_probe_ready") is not True:
        errors.append("stage_header_probe_ready_missing")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = probe.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))

    model = _dict(report.get("model"))
    if model.get("model_id") != probe.DEFAULT_MODEL_ID:
        errors.append("model_id_mismatch")
    config = _dict(model.get("model_config"))
    if config.get("model_type") and config.get("model_type") != "deepseek_v4":
        errors.append("model_type_mismatch")

    stage = _dict(report.get("stage_mapping"))
    if _int(stage.get("selected_key_count")) < 1:
        errors.append("stage_selected_keys_missing")
    if _int(stage.get("selected_file_count")) < 1:
        errors.append("stage_selected_files_missing")
    family_hits = _dict(stage.get("family_hits"))
    for family in ["mla_attention", "moe_router", "shared_experts", "routed_experts", "norms"]:
        if family_hits.get(family) is not True:
            errors.append(f"stage_family_missing:{family}")
    if stage.get("stage_weight_values_loaded") is not False:
        errors.append("stage_weight_values_loaded_overclaim")
    if stage.get("stage_weight_values_public") is not False:
        errors.append("stage_weight_values_public_unsafe")

    headers = _dict(report.get("headers"))
    header_ready = report.get("safetensors_header_ready") is True and report.get("stage_header_shape_ready") is True
    if headers.get("safetensors_header_payload_public") is not False:
        errors.append("safetensors_header_payload_public_unsafe")
    if headers.get("real_weight_tensor_values_loaded") is not False:
        errors.append("real_weight_tensor_values_loaded_overclaim")
    if headers.get("real_weight_tensor_values_public") is not False:
        errors.append("real_weight_tensor_values_public_unsafe")
    if header_ready:
        if _int(headers.get("header_file_count")) != _int(headers.get("selected_file_count")):
            errors.append("header_file_count_mismatch")
        if _int(headers.get("selected_key_count")) != _int(stage.get("selected_key_count")):
            errors.append("header_selected_key_count_mismatch")
        if _int(headers.get("header_fetch_error_count")) != 0:
            errors.append("header_fetch_errors_present")
        if _int(headers.get("missing_header_key_count")) != 0:
            errors.append("missing_header_keys_present")
        if not _dict(headers.get("dtype_counts")):
            errors.append("dtype_counts_missing")
        if _int(headers.get("total_selected_tensor_storage_bytes")) <= 0:
            errors.append("selected_tensor_storage_bytes_missing")
        for entry in _list(headers.get("file_summaries")):
            file_summary = _dict(entry)
            if _int(file_summary.get("header_length_bytes")) <= 0:
                errors.append("file_header_length_missing")
            if _int(file_summary.get("selected_key_count")) <= 0:
                errors.append("file_selected_key_count_missing")
            if _int(file_summary.get("missing_selected_key_count")) != 0:
                errors.append("file_missing_selected_keys_present")
            if _int(file_summary.get("malformed_selected_key_count")) != 0:
                errors.append("file_malformed_selected_keys_present")
    elif require_ready:
        errors.append("safetensors_header_not_ready")
    else:
        if not _list(report.get("blockers")):
            errors.append("not_ready_without_blockers")
        if not str(report.get("failure_stage") or "").strip():
            errors.append("failure_stage_missing")

    safety = _dict(report.get("safety"))
    for flag in [
        "safetensors_header_payload_public",
        "weight_index_payload_public",
        "weight_tensor_values_loaded",
        "weight_tensor_values_public",
        "activation_public",
        "hidden_state_public",
        "logits_public",
        "kv_cache_public",
        "credentials_public",
        "cookies_public",
        "jupyter_proxy_token_public",
        "private_runtime_state_public",
    ]:
        if safety.get(flag) is not False:
            errors.append(f"safety_flag_mismatch:{flag}")
    if safety.get("public_artifact_safe") is not True:
        errors.append("safety_public_artifact_safe_missing")

    artifacts = _dict(report.get("artifacts"))
    for name in ("summary_json", "support_bundle_json"):
        if _dict(artifacts.get(name)).get("present") is not True:
            errors.append(f"artifact_missing:{name}")
    return sorted(set(errors))


def build_check(args: argparse.Namespace) -> dict[str, Any]:
    report = load_json(Path(args.report))
    errors = validate_report(report, require_ready=bool(args.require_ready))
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "report_schema": report.get("schema"),
        "report_path": args.report,
        "safetensors_header_ready": report.get("safetensors_header_ready") is True,
        "stage_header_shape_ready": report.get("stage_header_shape_ready") is True,
        "selected_key_count": _int(_dict(report.get("stage_mapping")).get("selected_key_count")),
        "selected_file_count": _int(_dict(report.get("stage_mapping")).get("selected_file_count")),
        "failure_stage": report.get("failure_stage"),
        "errors": errors,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-ready", action="store_true")
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
