#!/usr/bin/env python3
"""Validate GLM 5.2 Kaggle stage runtime adapter plan evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "glm52_kaggle_stage_runtime_plan_check_v1"
PLAN_SCHEMA = "glm52_kaggle_stage_runtime_plan_v1"
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


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def validate_report(report: dict[str, Any], *, require_verified: bool = False) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != PLAN_SCHEMA:
        errors.append("schema_mismatch")
    if report.get("ok") is not True:
        errors.append("report_not_ok")
    if report.get("glm52_stage_runtime_plan_ready") is not True:
        errors.append("plan_ready_missing")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))

    model = _dict(report.get("model"))
    if model.get("model_id") != MODEL_ID:
        errors.append("model_id_not_glm52")
    if model.get("fallback_model_allowed_for_success") is not False:
        errors.append("fallback_boundary_missing")

    specs = [item for item in _list(report.get("stage_specs")) if isinstance(item, dict)]
    providers = {str(spec.get("provider")) for spec in specs}
    if not REQUIRED_PROVIDERS.issubset(providers):
        errors.append("required_provider_specs_missing")
    stage_ids = [_int(spec.get("stage_id"), -1) for spec in specs]
    if len(stage_ids) != len(set(stage_ids)):
        errors.append("stage_id_not_unique")
    for spec in specs:
        provider = str(spec.get("provider") or "")
        if provider not in REQUIRED_PROVIDERS:
            errors.append(f"stage_provider_not_required:{provider or 'missing'}")
        if spec.get("model_id") != MODEL_ID:
            errors.append(f"stage_model_id_not_glm52:{provider or 'missing'}")
        if _int(spec.get("stage_count"), 0) <= 0:
            errors.append(f"stage_count_missing:{provider or 'missing'}")
        layer_range = _list(spec.get("stage_layer_range"))
        if len(layer_range) != 2 or _int(layer_range[1]) <= _int(layer_range[0]):
            errors.append(f"stage_layer_range_invalid:{provider or 'missing'}")
        if spec.get("expected_stage_report_schema") != "glm52_kaggle_stage_runtime_report_v1":
            errors.append(f"stage_report_schema_missing:{provider or 'missing'}")
        if spec.get("public_artifact_safe") is not True:
            errors.append(f"stage_public_artifact_unsafe:{provider or 'missing'}")

    topology = _dict(report.get("stage_topology"))
    if topology:
        if topology.get("public_artifact_safe") is not True:
            errors.append("topology_public_artifact_unsafe")
        if _int(topology.get("stage_count"), -1) != len(specs):
            errors.append("topology_stage_count_mismatch")
        for spec in specs:
            if _int(spec.get("stage_count"), -2) != _int(topology.get("stage_count"), -1):
                errors.append(f"topology_stage_count_spec_mismatch:{spec.get('provider') or 'missing'}")
        if topology.get("contiguous_full_layer_coverage") is not True:
            errors.append("topology_layer_coverage_not_contiguous")
        provider_counts = _dict(topology.get("provider_counts"))
        for provider in REQUIRED_PROVIDERS:
            if _int(provider_counts.get(provider), 0) <= 0:
                errors.append(f"topology_provider_missing:{provider}")

    launcher = _dict(report.get("launcher_contract"))
    if launcher.get("expected_stage_report_schema") != "glm52_kaggle_stage_runtime_report_v1":
        errors.append("launcher_stage_report_schema_missing")
    if launcher.get("private_kernel_required") is not True:
        errors.append("launcher_private_kernel_boundary_missing")

    boundary = _dict(report.get("completion_boundary"))
    for key in [
        "plan_is_not_runtime_success",
        "stage_runtime_report_required",
        "same_request_probe_required",
        "queue_or_stage_smoke_is_not_success",
    ]:
        if boundary.get(key) is not True:
            errors.append(f"completion_boundary_missing:{key}")

    verified = report.get("stage_runtime_adapter_verified") is True and report.get("same_request_route_verified") is True
    if require_verified and not verified:
        errors.append("stage_runtime_plan_not_verified")
    if verified:
        if report.get("blockers"):
            errors.append("verified_plan_has_blockers")
        for spec in specs:
            if spec.get("stage_runtime_adapter_verified") is not True:
                errors.append(f"verified_plan_stage_not_verified:{spec.get('provider') or 'missing'}")
            if spec.get("same_request_route_verified") is not True:
                errors.append(f"verified_plan_stage_route_not_verified:{spec.get('provider') or 'missing'}")
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
        "stage_runtime_plan_ready": report.get("glm52_stage_runtime_plan_ready") is True,
        "stage_runtime_adapter_verified": report.get("stage_runtime_adapter_verified") is True,
        "same_request_route_verified": report.get("same_request_route_verified") is True,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_kaggle_stage_runtime_plan_check: ok={result['ok']} "
            f"errors={len(errors)} verified={result['stage_runtime_adapter_verified']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
