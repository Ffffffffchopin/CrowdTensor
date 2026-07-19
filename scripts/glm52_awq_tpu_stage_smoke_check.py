#!/usr/bin/env python3
"""Validate GLM 5.2 AWQ TPU stage-smoke or queued-watch evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA = "glm52_awq_tpu_stage_smoke_check_v1"
SMOKE_SCHEMA = "glm52_awq_tpu_stage_smoke_v1"
KAGGLE_SMOKE_SCHEMA = "glm52_kaggle_tpu_awq_stage_smoke_v1"
WATCH_SCHEMA = "glm52_kaggle_tpu_awq_stage_smoke_watch_v1"
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
    "kaggle-cookies",
    "kaggle-web-storage-state",
    "token=",
    "runtime_proxy",
    "jupyter-proxy",
    '"prompt":',
    '"raw_prompt":',
    '"generated_text":',
    '"generated_token_ids":',
    '"input_ids":',
    '"activation":',
    '"activations":',
    '"hidden_state":',
    '"hidden_states":',
    '"logits":',
    '"kv_cache":',
    '"past_key_values":',
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


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def _validate_safety(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
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
        if key in safety and safety.get(key) is not False:
            errors.append(f"safety_flag_not_false:{key}")
    return errors


def _validate_watch(report: dict[str, Any], *, require_ready: bool) -> list[str]:
    errors = _validate_safety(report)
    if report.get("ref") != "tpuowner/ct-glm52-awq-tpu-stage-smoke-0704-r1":
        errors.append("watch_ref_mismatch")
    observations = [item for item in _list(report.get("observations")) if isinstance(item, dict)]
    if not observations:
        errors.append("watch_observations_missing")
    status = str(report.get("last_status") or (observations[-1].get("status") if observations else ""))
    if not status:
        errors.append("watch_status_missing")
    output_verified = report.get("notebook_output_verified") is True
    smoke_ready = report.get("stage_runtime_adapter_smoke_ready") is True
    stage_check = _dict(report.get("stage_smoke_check"))
    stage_output = _dict(report.get("stage_smoke_output"))
    if output_verified or smoke_ready:
        if output_verified is not True:
            errors.append("watch_output_verified_missing_for_ready")
        if smoke_ready is not True:
            errors.append("watch_stage_smoke_ready_missing_for_verified_output")
        if stage_check.get("ok") is not True:
            errors.append("watch_stage_smoke_check_not_ok")
        if stage_output.get("present") is not True:
            errors.append("watch_stage_smoke_output_missing")
    if require_ready and not (output_verified and smoke_ready and stage_check.get("ok") is True):
        errors.append("stage_smoke_not_ready")
    return errors


def _ready(report: dict[str, Any]) -> bool:
    if str(report.get("schema") or "") == WATCH_SCHEMA:
        stage_check = _dict(report.get("stage_smoke_check"))
        stage_output = _dict(report.get("stage_smoke_output"))
        return bool(
            report.get("notebook_output_verified") is True
            and report.get("stage_runtime_adapter_smoke_ready") is True
            and stage_check.get("ok") is True
            and stage_output.get("present") is True
        )
    assigned = _int(report.get("assigned_weight_key_count") or report.get("assigned_stage_key_count"))
    present = _int(report.get("present_stage_key_count") or report.get("present_weight_key_count"))
    missing = _int(report.get("missing_stage_key_count") or report.get("missing_weight_key_count"))
    header_ready = bool(
        report.get("glm52_awq_stage_header_ready") is True
        or report.get("stage_header_ready") is True
        or (assigned > 0 and present == assigned and missing == 0)
    )
    jax_shape_ready = bool(
        report.get("jax_shape_smoke_ready") is True
        or report.get("jax_bf16_matmul_smoke_ready") is True
        or report.get("jax_tpu_stage_smoke_ready") is True
        or report.get("jax_tpu_shape_smoke_ready") is True
    )
    tpu_runtime_ready = bool(
        report.get("tpu_runtime_ready") is True
        or _int(report.get("jax_tpu_device_count") or report.get("tpu_device_count")) > 0
    )
    return bool(
        report.get("ok") is True
        and tpu_runtime_ready
        and _int(report.get("jax_tpu_device_count") or report.get("tpu_device_count")) > 0
        and header_ready
        and jax_shape_ready
        and str(report.get("base_model_id") or "") == BASE_MODEL_ID
        and str(report.get("model_repo") or "") == MODEL_REPO
    )


def _validate_smoke(report: dict[str, Any], *, require_ready: bool) -> list[str]:
    errors = _validate_safety(report)
    if report.get("model_repo") != MODEL_REPO:
        errors.append("model_repo_mismatch")
    if report.get("base_model_id") != BASE_MODEL_ID:
        errors.append("base_model_id_mismatch")
    quantization = str(report.get("quantization") or "")
    if quantization not in {"AWQ-INT4", "AWQ", ""}:
        errors.append("quantization_unexpected")
    if report.get("weight_tensor_values_loaded") is not False:
        errors.append("weight_tensor_values_loaded_overclaim")
    if report.get("weight_tensor_values_public") is not False:
        errors.append("weight_tensor_values_public_unsafe")
    if report.get("safetensors_header_payload_public") is True:
        errors.append("safetensors_header_payload_public_unsafe")
    if report.get("same_request_decode_verified") is True:
        errors.append("same_request_decode_overclaim")
    if report.get("same_request_route_verified") is True:
        errors.append("same_request_route_overclaim")

    assigned = _int(report.get("assigned_weight_key_count") or report.get("assigned_stage_key_count"))
    present = _int(report.get("present_stage_key_count") or report.get("present_weight_key_count"))
    missing = _int(report.get("missing_stage_key_count") or report.get("missing_weight_key_count"))
    ready = _ready(report)
    if ready:
        if _int(report.get("jax_tpu_device_count") or report.get("tpu_device_count")) < 1:
            errors.append("jax_tpu_device_count_missing")
        if assigned <= 0:
            errors.append("assigned_weight_key_count_missing")
        if present != assigned:
            errors.append("present_stage_key_count_mismatch")
        if missing != 0:
            errors.append("missing_stage_keys_present")
        header_file_count = _int(report.get("header_file_count") or report.get("assigned_weight_file_count"))
        if header_file_count <= 0:
            errors.append("header_file_count_missing")
        if not _dict(report.get("dtype_counts")):
            errors.append("dtype_counts_missing")
        family_hits = _dict(report.get("stage_family_hits"))
        dtype_counts = _dict(report.get("dtype_counts"))
        diagnosis_codes = set(str(item) for item in _list(report.get("diagnosis_codes")))
        awq_family_ready = bool(
            family_hits.get("awq_quantized_tensors") is True
            or _int(dtype_counts.get("I32")) > 0
            or _int(dtype_counts.get("I64")) > 0
            or "glm52_awq_stage_header_ready_on_kaggle_tpu" in diagnosis_codes
        )
        if not awq_family_ready:
            errors.append("awq_quantized_tensor_family_missing")
    elif require_ready:
        errors.append("stage_smoke_not_ready")
    return errors


def validate_report(report: dict[str, Any], *, require_ready: bool = False) -> list[str]:
    schema = str(report.get("schema") or "")
    if schema == WATCH_SCHEMA:
        return _validate_watch(report, require_ready=require_ready)
    if schema in {SMOKE_SCHEMA, KAGGLE_SMOKE_SCHEMA}:
        return _validate_smoke(report, require_ready=require_ready)
    errors = _validate_safety(report)
    errors.append("schema_mismatch")
    if require_ready:
        errors.append("stage_smoke_not_ready")
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
        "source_schema": report.get("schema"),
        "stage_runtime_adapter_smoke_ready": _ready(report),
        "last_status": report.get("last_status"),
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_awq_tpu_stage_smoke_check: ok={result['ok']} "
            f"errors={len(errors)} ready={result['stage_runtime_adapter_smoke_ready']} "
            f"status={result['last_status']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
