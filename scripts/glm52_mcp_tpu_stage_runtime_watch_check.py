#!/usr/bin/env python3
"""Validate MCP/save-notebook GLM 5.2 TPU stage runtime watch artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "glm52_mcp_tpu_stage_runtime_watch_check_v1"
WATCH_SCHEMA = "glm52_mcp_tpu_stage_runtime_watch_v1"
SENSITIVE_FRAGMENTS = (
    "KAGGLE_KEY",
    "KAGGLE_USERNAME",
    "KAGGLE_API_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "Bearer ",
    "Authorization:",
    "Cookie:",
    "Set-Cookie",
    "token=",
    "runtime_proxy",
    "jupyter-proxy",
    '"raw_prompt":',
    '"generated_text":',
    '"generated_token_ids":',
    '"activation":',
    '"hidden_state":',
    '"logits":',
    '"kv_cache":',
    '"weight_tensor_values":',
    '"safetensors_header_payload":',
)


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def validate_report(report: dict[str, Any], *, require_ready: bool = False) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != WATCH_SCHEMA:
        errors.append("schema_mismatch")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))
    safety = _dict(report.get("safety"))
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
    if not str(report.get("ref") or "").strip():
        errors.append("ref_missing")
    observations = [item for item in _list(report.get("observations")) if isinstance(item, dict)]
    if not observations:
        errors.append("observations_missing")
    status_class = str(report.get("last_status_class") or "")
    if status_class not in {"complete", "failed", "queued", "running", "unknown"}:
        errors.append("last_status_class_invalid")

    ready = report.get("stage_runtime_report_verified") is True
    stage_report = _dict(report.get("stage_runtime_report"))
    stage_check = _dict(report.get("stage_runtime_check"))
    summary = _dict(report.get("stage_runtime_summary"))
    if ready:
        if stage_report.get("present") is not True:
            errors.append("ready_without_stage_report")
        if stage_check.get("ok") is not True or stage_check.get("stage_runtime_verified") is not True:
            errors.append("ready_without_stage_check")
        if summary.get("provider") != "kaggle_jax_tpu":
            errors.append("ready_provider_not_tpu")
        if summary.get("stage_execution_verified") is not True:
            errors.append("ready_stage_execution_missing")
    else:
        if require_ready:
            errors.append("stage_runtime_report_not_verified")
        if report.get("same_request_decode_verified") is True:
            errors.append("not_ready_overclaims_same_request")
    if report.get("same_request_decode_verified") is True:
        errors.append("watch_must_not_claim_same_request_decode")
    if status_class in {"queued", "running"} and report.get("stage_runtime_report_verified") is True:
        errors.append("queued_or_running_overclaims_ready")
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
        "last_status": report.get("last_status"),
        "last_status_class": report.get("last_status_class"),
        "stage_runtime_report_verified": report.get("stage_runtime_report_verified") is True,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_mcp_tpu_stage_runtime_watch_check: ok={result['ok']} "
            f"ready={result['stage_runtime_report_verified']} status={result['last_status']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
