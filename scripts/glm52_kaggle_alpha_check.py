#!/usr/bin/env python3
"""Validate the GLM 5.2 Kaggle service Alpha artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import glm52_kaggle_alpha_pack as pack  # noqa: E402


SCHEMA = "glm52_kaggle_alpha_check_v1"


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
    if report.get("schema") != pack.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = pack.public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))
    safety = _dict(report.get("safety"))
    for key, expected in pack.alpha.safety_flags().items():
        if key in safety and safety.get(key) is not expected:
            errors.append(f"safety_flag_mismatch:{key}")
    if report.get("model_id") != pack.alpha.MODEL_ID:
        errors.append("model_id_not_glm52")
    if report.get("compatible_weight_repo") != pack.alpha.COMPATIBLE_WEIGHT_REPO:
        errors.append("compatible_weight_repo_mismatch")
    if report.get("service_api_ready") is not True:
        errors.append("service_api_not_ready")
    if report.get("generate_routes_to_same_request_live_probe") is not True:
        errors.append("generate_not_routed_to_same_request_pipeline")

    service = _dict(report.get("service_summary"))
    if service.get("cli_generate_command_available") is not True:
        errors.append("cli_generate_command_missing")
    if service.get("cli_generate_artifact_recovery_supported") is not True:
        errors.append("cli_generate_artifact_recovery_missing")
    if service.get("cli_serve_default_matches_deploy") is not True:
        errors.append("cli_serve_default_mismatch")
    if service.get("cli_status_default_matches_deploy") is not True:
        errors.append("cli_status_default_mismatch")
    if service.get("cli_cleanup_default_matches_deploy") is not True:
        errors.append("cli_cleanup_default_mismatch")
    if service.get("model_request_supported") is not True:
        errors.append("model_request_not_supported")
    if service.get("accelerator_request_complete") is not True:
        errors.append("accelerator_request_incomplete")
    requested_accelerators = {str(item) for item in _list(service.get("accelerators"))}
    for accelerator in pack.alpha.REQUIRED_ACCELERATORS:
        if accelerator not in requested_accelerators:
            errors.append(f"accelerator_missing:{accelerator}")
    if service.get("hf_token_env_supported") is not True:
        errors.append("hf_token_env_contract_missing")
    if service.get("hf_token_public") is True:
        errors.append("hf_token_public")
    resume_private_inputs = _dict(service.get("resume_private_inputs"))
    if resume_private_inputs.get("schema") != pack.alpha.RESUME_PRIVATE_INPUTS_SCHEMA:
        errors.append("resume_private_inputs_schema_mismatch")
    if resume_private_inputs.get("required_for_live_resume") is not True:
        errors.append("resume_private_inputs_required_missing")
    if resume_private_inputs.get("resume_command_omits_private_credentials") is not True:
        errors.append("resume_private_inputs_redaction_missing")
    if resume_private_inputs.get("kaggle_credentials_required") is not True:
        errors.append("resume_private_inputs_kaggle_credentials_missing")
    if resume_private_inputs.get("kaggle_credential_values_public") is not False:
        errors.append("resume_private_inputs_kaggle_values_public")
    if resume_private_inputs.get("kaggle_token_file_paths_public") is not False:
        errors.append("resume_private_inputs_token_paths_public")
    if resume_private_inputs.get("hf_env_values_public") is not False:
        errors.append("resume_private_inputs_hf_values_public")
    if resume_private_inputs.get("public_artifact_safe") is not True:
        errors.append("resume_private_inputs_public_safe_missing")
    if service.get("status_loads_existing_alpha_artifacts") is not True:
        errors.append("status_existing_artifact_load_missing")
    if service.get("status_exposes_resume_private_inputs") is not True:
        errors.append("status_resume_private_inputs_missing")
    if service.get("generate_validates_request_schema") is not True:
        errors.append("generate_request_validation_missing")
    if service.get("generate_uses_current_gpu_quota_blocker") is not True:
        errors.append("generate_current_gpu_quota_blocker_missing")
    if service.get("kaggle_runtime_blocker_classification_ready") is not True:
        errors.append("kaggle_runtime_blocker_classification_missing")
    for field in ["health_route_ready", "status_route_ready", "generate_route_ready", "cleanup_route_ready"]:
        if service.get(field) is not True:
            errors.append(f"{field}_missing")
    for field in ["generate_accepts_prompt", "generate_accepts_max_new_tokens", "generate_accepts_timeout"]:
        if service.get(field) is not True:
            errors.append(f"{field}_missing")
    if service.get("raw_prompt_public") is True or service.get("raw_generated_text_public") is True:
        errors.append("private_text_public")

    service_smoke = _dict(report.get("service_smoke_summary"))
    if service_smoke.get("present") is True:
        if service_smoke.get("service_smoke_check_ok") is not True:
            errors.append("service_smoke_check_failed")
        if service_smoke.get("service_http_smoke_verified") is not True:
            errors.append("service_http_smoke_not_verified")
        if service_smoke.get("generate_route_reaches_service") is not True:
            errors.append("service_smoke_generate_route_not_reached")
        if service_smoke.get("status_resume_private_inputs_verified") is not True:
            errors.append("service_smoke_status_resume_private_inputs_missing")
        if service_smoke.get("generate_route_quota_blocker_verified") is True and service_smoke.get("generate_resume_private_inputs_verified") is not True:
            errors.append("service_smoke_generate_resume_private_inputs_missing")
        if service_smoke.get("cleanup_route_verified") is not True:
            errors.append("service_smoke_cleanup_route_not_verified")
        if service_smoke.get("cleanup_temporary_kaggle_kernels_deleted") is not True:
            errors.append("service_smoke_cleanup_kernel_delete_missing")
        if service_smoke.get("cleanup_temporary_private_packages_removed") is not True:
            errors.append("service_smoke_cleanup_private_package_removal_missing")
        if service_smoke.get("cleanup_live_resources_left_running") is True:
            errors.append("service_smoke_cleanup_live_resources_left_running")
        if service_smoke.get("public_artifact_safe") is not True:
            errors.append("service_smoke_public_safe_missing")

    generate_cli = _dict(report.get("generate_cli_summary"))
    if generate_cli.get("present") is True:
        if generate_cli.get("generate_cli_check_ok") is not True:
            errors.append("generate_cli_check_failed")
        if generate_cli.get("artifact_recovery_present") is not True:
            errors.append("generate_cli_artifact_recovery_missing")
        if generate_cli.get("artifact_recovery_resume_command_present") is not True:
            errors.append("generate_cli_resume_command_missing")
        if generate_cli.get("artifact_recovery_resume_private_inputs_verified") is not True:
            errors.append("generate_cli_resume_private_inputs_missing")
        if generate_cli.get("public_artifact_safe") is not True:
            errors.append("generate_cli_public_safe_missing")

    phase_status = _dict(report.get("phase_status"))
    if phase_status.get("schema") != pack.PHASE_STATUS_SCHEMA:
        errors.append("phase_status_schema_mismatch")
    phase_names = {str(item) for item in _list(phase_status.get("phase_names"))}
    for name in [
        "configuration_check",
        "model_source_check",
        "gpu_quota_preflight",
        "kernel_push",
        "gpu_queue_running",
        "tpu_queue_running",
        "cpu_queue_running",
        "stage_completed",
        "decode_completed",
        "cleanup_completed",
    ]:
        if name not in phase_names:
            errors.append(f"phase_status_missing:{name}")
    if phase_status.get("public_artifact_safe") is not True:
        errors.append("phase_status_public_safe_missing")

    live = _dict(report.get("live_summary"))
    min_tokens = max(2, _int(report.get("min_required_generated_tokens"), pack.MIN_TARGET_TOKENS))
    ready = report.get("glm52_kaggle_alpha_ready") is True and report.get("ok") is True
    if not ready and not require_ready:
        if not _list(report.get("blockers")):
            errors.append("blocked_report_missing_blockers")
        blocker = _dict(report.get("blocker_report"))
        if blocker.get("schema") != pack.BLOCKER_SCHEMA:
            errors.append("blocker_report_schema_mismatch")
        if blocker.get("blocked") is not True:
            errors.append("blocker_report_not_marked_blocked")
        blocker_resume = str(blocker.get("next_resume_command") or "")
        report_resume = str(report.get("next_resume_command") or "")
        if not blocker_resume:
            errors.append("blocker_next_resume_command_missing")
        if not report_resume:
            errors.append("next_resume_command_missing")
        if blocker_resume and report_resume and blocker_resume != report_resume:
            errors.append("next_resume_command_mismatch")
        if blocker.get("next_resume_command_redacts_credentials") is not True:
            errors.append("blocker_next_resume_command_redaction_missing")
        blocker_resume_private_inputs = _dict(blocker.get("resume_private_inputs"))
        if blocker_resume_private_inputs.get("schema") != pack.alpha.RESUME_PRIVATE_INPUTS_SCHEMA:
            errors.append("blocker_resume_private_inputs_schema_mismatch")
        if blocker_resume_private_inputs.get("resume_command_omits_private_credentials") is not True:
            errors.append("blocker_resume_private_inputs_redaction_missing")
        if blocker_resume_private_inputs.get("public_artifact_safe") is not True:
            errors.append("blocker_resume_private_inputs_public_safe_missing")
        if report.get("next_resume_command_redacts_credentials") is not True:
            errors.append("next_resume_command_redaction_missing")
        report_resume_private_inputs = _dict(report.get("resume_private_inputs"))
        if report_resume_private_inputs.get("schema") != pack.alpha.RESUME_PRIVATE_INPUTS_SCHEMA:
            errors.append("resume_private_inputs_top_level_schema_mismatch")
        return sorted(set(errors))

    if live.get("source_schema") == "glm52_kaggle_accelerator_deployment_rc_v1":
        errors.append("old_rc_artifact_is_not_alpha_live_report")
    if live.get("source_schema") == pack.live_probe.SCHEMA and _int(live.get("generated_token_count")) == 1:
        errors.append("old_single_token_live_report")
    if live.get("same_request_decode_verified") is not True:
        errors.append("same_request_multitoken_not_verified")
    if live.get("multi_token_alpha_verified") is not True:
        errors.append("multi_token_alpha_not_verified")
    if _int(live.get("generated_token_count")) < min_tokens:
        errors.append("generated_token_count_below_minimum")
    if _int(live.get("generated_token_hash_count")) < min_tokens:
        errors.append("generated_token_hashes_below_minimum")
    providers = {str(item) for item in _list(report.get("accepted_providers"))}
    for provider in pack.live_probe.REQUIRED_PROVIDERS:
        if provider not in providers:
            errors.append(f"required_provider_missing:{provider}")
    if live.get("cleanup_verified") is not True or report.get("cleanup_verified") is not True:
        errors.append("cleanup_not_verified")
    cleanup = _dict(live.get("cleanup_status"))
    if cleanup.get("temporary_kaggle_kernels_deleted") is not True:
        errors.append("cleanup_kernel_delete_missing")
    if cleanup.get("temporary_private_packages_removed") is not True:
        errors.append("cleanup_private_package_removal_missing")
    if cleanup.get("live_resources_left_running") is not False:
        errors.append("cleanup_live_resources_left_running")

    benchmark = _dict(report.get("benchmark"))
    if benchmark.get("schema") != pack.BENCHMARK_SCHEMA:
        errors.append("benchmark_schema_mismatch")
    if _int(benchmark.get("stage_count")) < 2:
        errors.append("benchmark_stage_count_missing")
    if _int(benchmark.get("tokens_generated")) < min_tokens:
        errors.append("benchmark_tokens_below_minimum")

    boundaries = _dict(report.get("boundaries"))
    for field in [
        "old_one_token_rc_is_not_success",
        "mock_only_is_not_success",
        "single_backend_is_not_success",
        "queue_only_is_not_success",
        "non_glm_fallback_is_not_success",
    ]:
        if boundaries.get(field) is not True:
            errors.append(f"boundary_missing:{field}")

    if require_ready and not ready:
        errors.append("glm52_kaggle_alpha_not_ready")
    if ready:
        if _list(report.get("blockers")):
            errors.append("ready_but_blockers_present")
        blocker = _dict(report.get("blocker_report"))
        if blocker.get("blocked") is True:
            errors.append("ready_but_blocker_report_blocked")
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
        "glm52_kaggle_alpha_ready": report.get("glm52_kaggle_alpha_ready") is True,
        "generated_token_count": _int(report.get("generated_token_count")),
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_kaggle_alpha_check: ok={result['ok']} "
            f"errors={len(errors)} ready={result['glm52_kaggle_alpha_ready']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
