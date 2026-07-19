#!/usr/bin/env python3
"""Validate GLM 5.2 Kaggle CPU/GPU/TPU same-request proof."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA = "glm52_kaggle_same_request_check_v1"
PROOF_SCHEMA = "glm52_kaggle_same_request_probe_v1"
MODEL_ID = "zai-org/GLM-5.2"
REQUIRED_PROVIDERS = {"kaggle_cuda", "kaggle_jax_tpu", "kaggle_cpu"}
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


def model_id(report: dict[str, Any]) -> str:
    success = _dict(report.get("success"))
    same = _dict(report.get("same_request"))
    return str(
        report.get("model_id")
        or _dict(report.get("model")).get("model_id")
        or success.get("model_id")
        or same.get("model_id")
        or ""
    )


def accepted_providers(report: dict[str, Any]) -> set[str]:
    success = _dict(report.get("success"))
    values = (
        _list(success.get("accepted_providers"))
        or _list(report.get("accepted_providers"))
        or _list(report.get("accepted_stage_providers"))
        or _list(report.get("accepted_stage_backends"))
    )
    return {str(item) for item in values}


def generated_token_count(report: dict[str, Any]) -> int:
    success = _dict(report.get("success"))
    return _int(success.get("generated_token_count") or report.get("generated_token_count"))


def same_request_verified(report: dict[str, Any]) -> bool:
    success = _dict(report.get("success"))
    return bool(
        report.get("glm52_kaggle_same_request_verified") is True
        or report.get("same_request_decode_verified") is True
        or success.get("same_request_decode_verified") is True
    )


def _stage_reports(report: dict[str, Any]) -> list[dict[str, Any]]:
    stages = _list(report.get("stage_reports")) or _list(report.get("stages"))
    return [item for item in stages if isinstance(item, dict)]


def _stage_provider(stage: dict[str, Any]) -> str:
    return str(stage.get("provider") or stage.get("backend") or stage.get("stage_provider") or "")


def _hash_ok(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) >= 71


def _stage_weight_value_hash(stage: dict[str, Any]) -> str:
    return str(
        stage.get("stage_weight_value_hash")
        or stage.get("weight_value_sha256")
        or stage.get("weight_value_hash")
        or ""
    )


def _stage_weight_values_loaded(stage: dict[str, Any]) -> bool:
    normalized_hash_present = stage.get("weight_value_hash_present") is True
    return bool(
        (
            stage.get("stage_owned_weight_values_loaded") is True
            or stage.get("weight_tensor_values_loaded") is True
        )
        and (_hash_ok(_stage_weight_value_hash(stage)) or normalized_hash_present)
        and _int(stage.get("weight_value_byte_count") or stage.get("stage_weight_value_byte_count")) > 0
        and stage.get("weight_tensor_values_public") is not True
    )


def _stage_ready(stage: dict[str, Any]) -> bool:
    return bool(
        stage.get("stage_decode_verified") is True
        and _stage_weight_values_loaded(stage)
    )


def validate_report(report: dict[str, Any], *, require_verified: bool = False) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != PROOF_SCHEMA:
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

    verified = same_request_verified(report)
    if require_verified and not verified:
        errors.append("same_request_not_verified")
    if not verified:
        return sorted(set(errors))

    if report.get("ok") is not True:
        errors.append("verified_but_report_not_ok")
    if report.get("live_run_performed") is not True:
        errors.append("live_run_not_performed")
    if model_id(report) != MODEL_ID:
        errors.append("model_id_not_glm52")
    if report.get("fallback_model_used") is True or _dict(report.get("model")).get("fallback_model_used") is True:
        errors.append("fallback_model_used")
    if generated_token_count(report) < 1:
        errors.append("generated_token_missing")
    success = _dict(report.get("success"))
    token_hash = str(success.get("generated_token_hash") or report.get("generated_token_hash") or "")
    if not token_hash.startswith("sha256:"):
        errors.append("generated_token_hash_missing")

    providers = accepted_providers(report)
    missing = sorted(REQUIRED_PROVIDERS - providers)
    for provider in missing:
        errors.append(f"required_provider_missing:{provider}")

    same = _dict(report.get("same_request"))
    request_hash = str(
        same.get("coordinator_request_id_hash")
        or same.get("request_id_hash")
        or report.get("coordinator_request_id_hash")
        or ""
    )
    if same.get("coordinator_request_verified") is not True and not request_hash.startswith("sha256:"):
        errors.append("coordinator_request_evidence_missing")

    stages = _stage_reports(report)
    stage_providers = {_stage_provider(stage) for stage in stages if _stage_provider(stage)}
    for provider in REQUIRED_PROVIDERS:
        provider_stages = [stage for stage in stages if _stage_provider(stage) == provider]
        if not provider_stages:
            errors.append(f"stage_report_missing:{provider}")
            continue
        if not any(_stage_ready(stage) for stage in provider_stages):
            errors.append(f"stage_execution_missing:{provider}")
        if not any(stage.get("stage_decode_verified") is True for stage in provider_stages):
            errors.append(f"stage_decode_missing:{provider}")
        if not any(_stage_weight_values_loaded(stage) for stage in provider_stages):
            errors.append(f"stage_weight_values_not_loaded:{provider}")
        if not any(stage.get("live_run_performed") is True for stage in provider_stages):
            errors.append(f"stage_live_run_not_performed:{provider}")
    if not REQUIRED_PROVIDERS.issubset(stage_providers):
        errors.append("stage_provider_coverage_missing")

    cleanup = _dict(report.get("cleanup") or report.get("cleanup_status"))
    if cleanup.get("public_artifact_safe") is False:
        errors.append("cleanup_public_artifact_unsafe")
    if cleanup.get("temporary_kaggle_kernels_deleted") is not True and cleanup.get("temporary_resources_deleted") is not True:
        errors.append("cleanup_kernel_delete_missing")
    if cleanup.get("temporary_private_packages_removed") is not True and cleanup.get("private_packages_removed") is not True:
        errors.append("cleanup_private_package_removal_missing")
    if cleanup.get("live_resources_left_running") is not False:
        errors.append("cleanup_live_resources_left_unknown")

    if report.get("queue_only_evidence") is True:
        errors.append("queue_evidence_overclaim")
    if report.get("metadata_only") is True:
        errors.append("metadata_only_overclaim")
    if report.get("stage_smoke_only") is True:
        errors.append("stage_smoke_only_overclaim")
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
        "same_request_decode_verified": same_request_verified(report),
        "model_id": model_id(report),
        "accepted_providers": sorted(accepted_providers(report)),
        "generated_token_count": generated_token_count(report),
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_kaggle_same_request_check: ok={result['ok']} "
            f"errors={len(errors)} verified={result['same_request_decode_verified']} "
            f"model={result['model_id']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
