#!/usr/bin/env python3
"""Validate GLM 5.2 Kaggle Alpha HTTP service smoke reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor import glm52_kaggle_alpha as alpha  # noqa: E402
from scripts import glm52_kaggle_alpha_service_smoke_probe as probe  # noqa: E402


SCHEMA = "glm52_kaggle_alpha_service_smoke_check_v1"


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
    return sorted({fragment for fragment in probe.SENSITIVE_FRAGMENTS if fragment in encoded})


def validate_report(report: dict[str, Any], *, require_verified: bool = False) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != probe.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))
    health = _dict(report.get("health"))
    status = _dict(report.get("status"))
    generate = _dict(report.get("generate"))
    cleanup = _dict(report.get("cleanup"))
    if health.get("http_status") != 200 or health.get("ok") is not True:
        errors.append("health_route_not_ready")
    if status.get("http_status") != 200:
        errors.append("status_route_not_ready")
    if status.get("resume_private_inputs_verified") is not True:
        errors.append("status_resume_private_inputs_missing")
    if generate.get("attempted") is not True:
        errors.append("generate_route_not_attempted")
    if generate.get("raw_prompt_public") is True:
        errors.append("raw_prompt_public")
    if generate.get("raw_generated_text_public") is True:
        errors.append("raw_generated_text_public")
    if generate.get("generated_token_ids_public") is True:
        errors.append("generated_token_ids_public")
    if generate.get("quota_blocker_verified") is not True and generate.get("successful_generate_verified") is not True:
        errors.append("generate_route_not_verified")
    if generate.get("quota_blocker_verified") is True and generate.get("http_status") != 503:
        errors.append("quota_blocker_http_status_mismatch")
    if generate.get("quota_blocker_verified") is True and "kaggle_gpu_quota_unavailable" not in _list(generate.get("blockers")):
        errors.append("quota_blocker_missing")
    if generate.get("quota_blocker_verified") is True and generate.get("resume_private_inputs_verified") is not True:
        errors.append("quota_blocker_resume_private_inputs_missing")
    if report.get("cleanup_route_verified") is not True:
        errors.append("cleanup_route_not_verified")
    if cleanup.get("http_status") != 200 or cleanup.get("ok") is not True:
        errors.append("cleanup_route_not_ready")
    if cleanup.get("temporary_kaggle_kernels_deleted") is not True:
        errors.append("cleanup_kernel_delete_missing")
    if cleanup.get("temporary_private_packages_removed") is not True:
        errors.append("cleanup_private_package_removal_missing")
    if cleanup.get("live_resources_left_running") is True:
        errors.append("cleanup_live_resources_left_running")
    if generate.get("successful_generate_verified") is True:
        if generate.get("same_request_decode_verified") is not True:
            errors.append("successful_generate_without_same_request")
        if _int(generate.get("generated_token_count")) < _int(generate.get("target_generated_token_count"), 1):
            errors.append("successful_generate_token_count_below_target")
        providers = {str(item) for item in _list(generate.get("accepted_providers"))}
        if not set(alpha.live_probe.REQUIRED_PROVIDERS).issubset(providers):
            errors.append("successful_generate_required_provider_missing")
    boundary = _dict(report.get("completion_boundary"))
    for field in [
        "service_smoke_is_not_live_success",
        "quota_blocker_generate_is_not_multitoken_success",
        "strict_alpha_ready_still_requires_live_report",
    ]:
        if boundary.get(field) is not True:
            errors.append(f"boundary_missing:{field}")
    if require_verified and report.get("service_http_smoke_verified") is not True:
        errors.append("service_http_smoke_not_verified")
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
        "service_http_smoke_verified": report.get("service_http_smoke_verified") is True,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            "glm52_kaggle_alpha_service_smoke_check: "
            f"ok={result['ok']} errors={len(errors)}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
