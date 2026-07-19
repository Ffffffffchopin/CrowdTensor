#!/usr/bin/env python3
"""Validate public-safe GLM 5.2 AWQ stage-owned value probe reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "glm52_awq_stage_value_probe_check_v1"
PROBE_SCHEMA = "glm52_awq_stage_value_probe_v1"
MODEL_REPO = "cyankiwi/GLM-5.2-AWQ-INT4"
BASE_MODEL_ID = "zai-org/GLM-5.2"
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Authorization:",
    "Cookie:",
    "Set-Cookie",
    "token=",
    '"weight_tensor_values":',
    '"safetensors_header_payload":',
    '"activation":',
    '"logits":',
    '"kv_cache":',
    '"generated_token_ids":',
)


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


def _hash_ok(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) >= 71


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def validate_report(report: dict[str, Any], *, require_ready: bool = False) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != PROBE_SCHEMA:
        errors.append("schema_mismatch")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))
    if report.get("model_repo") != MODEL_REPO:
        errors.append("model_repo_mismatch")
    if report.get("base_model_id") != BASE_MODEL_ID:
        errors.append("base_model_id_mismatch")
    if report.get("weight_tensor_values_public") is not False:
        errors.append("weight_tensor_values_public_unsafe")
    if report.get("safetensors_header_payload_public") is not False:
        errors.append("safetensors_header_payload_public_unsafe")
    if report.get("stage_runtime_adapter_verified") is True:
        errors.append("stage_runtime_adapter_overclaim")
    if report.get("same_request_route_verified") is True:
        errors.append("same_request_route_overclaim")
    if report.get("same_request_decode_verified") is True:
        errors.append("same_request_decode_overclaim")
    selected = _dict(report.get("selected_tensor"))
    ready = bool(
        report.get("ok") is True
        and report.get("glm52_awq_stage_value_probe_ready") is True
        and report.get("weight_tensor_values_loaded") is True
        and _int(report.get("weight_value_byte_count")) > 0
        and _int(report.get("weight_value_byte_count")) == _int(selected.get("tensor_nbytes"))
        and _hash_ok(report.get("weight_value_sha256"))
        and _hash_ok(selected.get("key_digest"))
        and selected.get("filename")
        and selected.get("dtype")
        and _int(report.get("assigned_weight_key_count")) > 0
        and _int(report.get("header_file_count")) > 0
    )
    if require_ready and not ready:
        errors.append("stage_value_probe_not_ready")
    if ready and report.get("stage_smoke_only") is not True:
        errors.append("stage_value_probe_boundary_missing")
    return sorted(set(errors))


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
        "stage_value_probe_ready": report.get("glm52_awq_stage_value_probe_ready") is True,
        "weight_value_byte_count": _int(report.get("weight_value_byte_count")),
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_awq_stage_value_probe_check: ok={result['ok']} "
            f"errors={len(errors)} ready={result['stage_value_probe_ready']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
