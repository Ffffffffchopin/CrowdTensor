#!/usr/bin/env python3
"""Validate a single public-safe GLM 5.2 Kaggle stage runtime proof."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA = "glm52_kaggle_stage_runtime_check_v1"
STAGE_SCHEMA = "glm52_kaggle_stage_runtime_report_v1"
MODEL_ID = "zai-org/GLM-5.2"
COMPATIBLE_WEIGHT_REPO = "cyankiwi/GLM-5.2-AWQ-INT4"
REQUIRED_PROVIDERS = {"kaggle_cuda", "kaggle_jax_tpu", "kaggle_cpu"}
STAGE_SMOKE_SCHEMAS = {
    "glm52_awq_tpu_stage_smoke_v1",
    "glm52_kaggle_tpu_awq_stage_smoke_watch_v1",
}
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
    '"raw_generated_text":',
    '"generated_token_ids":',
    '"input_ids":',
    '"activation":',
    '"activations":',
    '"hidden_state":',
    '"hidden_states":',
    '"logits":',
    '"kv_cache":',
    '"past_key_values":',
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


def model_id(report: dict[str, Any]) -> str:
    return str(report.get("model_id") or _dict(report.get("model")).get("model_id") or "")


def provider(report: dict[str, Any]) -> str:
    return str(report.get("provider") or report.get("backend") or report.get("stage_provider") or "")


def coordinator_request_hash(report: dict[str, Any]) -> str:
    return str(
        report.get("coordinator_request_id_hash")
        or report.get("request_id_hash")
        or _dict(report.get("same_request")).get("coordinator_request_id_hash")
        or ""
    )


def stage_output_hash(report: dict[str, Any]) -> str:
    return str(
        report.get("stage_output_hash")
        or report.get("output_hash")
        or report.get("activation_handoff_hash")
        or _dict(report.get("stage")).get("stage_output_hash")
        or ""
    )


def stage_weight_value_hash(report: dict[str, Any]) -> str:
    return str(
        report.get("stage_weight_value_hash")
        or report.get("weight_value_sha256")
        or report.get("weight_value_hash")
        or _dict(report.get("stage")).get("stage_weight_value_hash")
        or ""
    )


def stage_weight_values_loaded(report: dict[str, Any]) -> bool:
    return bool(
        (
            report.get("stage_owned_weight_values_loaded") is True
            or report.get("weight_tensor_values_loaded") is True
        )
        and _hash_ok(stage_weight_value_hash(report))
        and _int(report.get("weight_value_byte_count") or report.get("stage_weight_value_byte_count")) > 0
        and report.get("weight_tensor_values_public") is not True
    )


def stage_runtime_verified(report: dict[str, Any]) -> bool:
    return bool(
        (
            report.get("stage_execution_verified") is True
            or report.get("stage_decode_verified") is True
            or (report.get("stage_owned_model_loaded") is True and _hash_ok(stage_output_hash(report)))
        )
        and stage_weight_values_loaded(report)
    )


def validate_report(report: dict[str, Any], *, require_verified: bool = False) -> list[str]:
    errors: list[str] = []
    source_schema = str(report.get("schema") or "")
    if source_schema in STAGE_SMOKE_SCHEMAS or report.get("stage_smoke_only") is True:
        errors.append("stage_smoke_is_not_runtime_proof")
    if source_schema != STAGE_SCHEMA:
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
        if key in safety and safety.get(key) is not False:
            errors.append(f"safety_flag_not_false:{key}")
    for key in [
        "activation_public",
        "kv_cache_public",
        "hidden_state_public",
        "logits_public",
        "weight_tensor_values_public",
        "safetensors_header_payload_public",
    ]:
        if key in report and report.get(key) is not False:
            errors.append(f"stage_flag_not_false:{key}")

    if model_id(report) != MODEL_ID:
        errors.append("model_id_not_glm52")
    if report.get("fallback_model_used") is True or _dict(report.get("model")).get("fallback_model_used") is True:
        errors.append("fallback_model_used")
    if provider(report) not in REQUIRED_PROVIDERS:
        errors.append(f"provider_not_required:{provider(report) or 'missing'}")
    if not _hash_ok(coordinator_request_hash(report)):
        errors.append("coordinator_request_hash_missing")
    if not _hash_ok(stage_output_hash(report)):
        errors.append("stage_output_hash_missing")
    if not stage_weight_values_loaded(report):
        errors.append("stage_weight_values_not_loaded")
    if report.get("live_run_performed") is not True:
        errors.append("live_run_not_performed")

    layer_range = _list(report.get("stage_layer_range"))
    if len(layer_range) != 2 or _int(layer_range[1]) <= _int(layer_range[0]):
        errors.append("stage_layer_range_invalid")
    if _int(report.get("stage_id"), -1) < 0:
        errors.append("stage_id_missing")

    compatible = str(report.get("compatible_weight_repo") or report.get("model_repo") or "")
    if compatible and "GLM-5.2" not in compatible:
        errors.append("compatible_weight_repo_not_glm52")
    if report.get("queue_only_evidence") is True:
        errors.append("queue_evidence_overclaim")
    if report.get("metadata_only") is True:
        errors.append("metadata_only_overclaim")

    verified = stage_runtime_verified(report)
    if require_verified and not verified:
        errors.append("stage_runtime_not_verified")
    if verified and report.get("ok") is False:
        errors.append("verified_but_report_not_ok")
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
        "stage_runtime_verified": stage_runtime_verified(report),
        "model_id": model_id(report),
        "provider": provider(report),
        "stage_id": _int(report.get("stage_id"), -1),
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_kaggle_stage_runtime_check: ok={result['ok']} "
            f"errors={len(errors)} verified={result['stage_runtime_verified']} "
            f"provider={result['provider']} stage_id={result['stage_id']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
