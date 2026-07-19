#!/usr/bin/env python3
"""Build the GLM 5.2 Kaggle CPU/GPU/TPU service Alpha artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor import glm52_kaggle_alpha as alpha  # noqa: E402
from scripts import glm52_kaggle_same_request_live_check as live_check  # noqa: E402
from scripts import glm52_kaggle_same_request_live_probe as live_probe  # noqa: E402
from scripts import glm52_kaggle_alpha_service_smoke_check as service_smoke_check  # noqa: E402


SCHEMA = "glm52_kaggle_alpha_v1"
SUPPORT_SCHEMA = "glm52_kaggle_alpha_support_v1"
BENCHMARK_SCHEMA = "glm52_kaggle_alpha_benchmark_v1"
BLOCKER_SCHEMA = "glm52_kaggle_alpha_blocker_v1"
PHASE_STATUS_SCHEMA = "glm52_kaggle_alpha_phase_status_v1"
GENERATE_CLI_SCHEMA = "glm52_kaggle_alpha_cli_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-kaggle-alpha"
MIN_TARGET_TOKENS = 8
SENSITIVE_FRAGMENTS = alpha.SENSITIVE_FRAGMENTS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    loaded = json.loads(p.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


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


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    try:
        rel = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        rel = str(path)
    entry: dict[str, Any] = {"kind": kind, "path": rel, "present": path.is_file()}
    if path.is_file():
        entry["sha256"] = sha256_file(path)
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def build_service_summary(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    routes = _dict(report.get("routes"))
    generate_fields = {str(item) for item in _list(report.get("generate_request_fields"))}
    resume_private_inputs = _dict(report.get("resume_private_inputs"))
    return {
        "schema": "glm52_kaggle_alpha_service_summary_v1",
        "source_path": str(path),
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "ok": report.get("ok") is True,
        "service_api_ready": report.get("service_api_ready") is True,
        "requested_model": str(report.get("requested_model") or ""),
        "model_request_supported": report.get("model_request_supported") is True,
        "model_request": _dict(report.get("model_request")),
        "accelerators": [str(item) for item in _list(report.get("accelerators"))],
        "required_accelerators": [str(item) for item in _list(report.get("required_accelerators"))],
        "accelerator_request_complete": report.get("accelerator_request_complete") is True,
        "accelerator_request": _dict(report.get("accelerator_request")),
        "hf_token_env_supported": report.get("hf_token_env_supported") is True,
        "hf_token_env_count": _int(report.get("hf_token_env_count")),
        "hf_token_env_name_hashes": [str(item) for item in _list(report.get("hf_token_env_name_hashes"))],
        "hf_token_env_configured": report.get("hf_token_env_configured") is True,
        "hf_token_env_configured_count": _int(report.get("hf_token_env_configured_count")),
        "hf_token_public": report.get("hf_token_public") is True,
        "resume_private_inputs": resume_private_inputs,
        "resume_private_inputs_ready": bool(
            resume_private_inputs.get("schema") == alpha.RESUME_PRIVATE_INPUTS_SCHEMA
            and resume_private_inputs.get("required_for_live_resume") is True
            and resume_private_inputs.get("resume_command_omits_private_credentials") is True
            and resume_private_inputs.get("kaggle_credentials_required") is True
            and resume_private_inputs.get("kaggle_credential_values_public") is False
            and resume_private_inputs.get("public_artifact_safe") is True
        ),
        "cli_generate_command_available": report.get("cli_generate_command_available") is True,
        "cli_generate_artifact_recovery_supported": report.get("cli_generate_artifact_recovery_supported") is True,
        "cli_serve_default_matches_deploy": report.get("cli_serve_default_matches_deploy") is True,
        "cli_status_default_matches_deploy": report.get("cli_status_default_matches_deploy") is True,
        "cli_cleanup_default_matches_deploy": report.get("cli_cleanup_default_matches_deploy") is True,
        "status_loads_existing_alpha_artifacts": report.get("status_loads_existing_alpha_artifacts") is True,
        "status_exposes_resume_private_inputs": report.get("status_exposes_resume_private_inputs") is True,
        "generate_validates_request_schema": report.get("generate_validates_request_schema") is True,
        "generate_routes_to_same_request_live_probe": report.get("generate_routes_to_same_request_live_probe") is True,
        "generate_uses_current_gpu_quota_blocker": report.get("generate_uses_current_gpu_quota_blocker") is True,
        "kaggle_runtime_blocker_classification_ready": report.get("kaggle_runtime_blocker_classification_ready") is True,
        "kaggle_runtime_blocker_classes": [str(item) for item in _list(report.get("kaggle_runtime_blocker_classes"))],
        "health_route_ready": routes.get("health") == "GET /health",
        "status_route_ready": routes.get("status") == "GET /status",
        "generate_route_ready": routes.get("generate") == "POST /generate",
        "cleanup_route_ready": routes.get("cleanup") == "POST /cleanup"
        and report.get("cleanup_route_ready") is True,
        "generate_accepts_prompt": "prompt" in generate_fields,
        "generate_accepts_max_new_tokens": "max_new_tokens" in generate_fields,
        "generate_accepts_timeout": bool({"timeout", "timeout_seconds"} & generate_fields),
        "generate_request_fields": sorted(generate_fields),
        "stage_worker_package_report": str(report.get("stage_worker_package_report") or ""),
        "stage_push_parallelism": _int(report.get("stage_push_parallelism")),
        "gpu_accelerator": str(report.get("gpu_accelerator") or ""),
        "tpu_accelerator": str(report.get("tpu_accelerator") or ""),
        "wait_seconds": float(report.get("wait_seconds") or 0.0),
        "poll_interval_seconds": float(report.get("poll_interval_seconds") or 0.0),
        "command_timeout_seconds": float(report.get("command_timeout_seconds") or 0.0),
        "kernel_timeout_seconds": _int(report.get("kernel_timeout_seconds")),
        "coordinator_task_timeout_seconds": float(report.get("coordinator_task_timeout_seconds") or 0.0),
        "coordinator_worker_poll_interval_seconds": float(report.get("coordinator_worker_poll_interval_seconds") or 0.0),
        "runtime_tuning": _dict(report.get("runtime_tuning")),
        "raw_prompt_public": report.get("raw_prompt_public") is True,
        "raw_generated_text_public": report.get("raw_generated_text_public") is True,
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def build_live_summary(path: Path | None, report: dict[str, Any], *, min_tokens: int) -> dict[str, Any]:
    errors = live_check.validate_report(report, require_verified=True) if report else ["live_report_missing"]
    cleanup = _dict(report.get("cleanup_status"))
    providers = {str(item) for item in _list(report.get("accepted_providers"))}
    token_hashes = _list(report.get("generated_token_hashes"))
    return {
        "schema": "glm52_kaggle_alpha_live_summary_v1",
        "source_path": str(path) if path is not None else "",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "live_check_ok": not errors,
        "live_check_errors": errors,
        "same_request_decode_verified": report.get("same_request_decode_verified") is True,
        "model_id": str(report.get("model_id") or ""),
        "compatible_weight_repo": str(report.get("compatible_weight_repo") or ""),
        "target_generated_token_count": _int(report.get("target_generated_token_count")),
        "generated_token_count": _int(report.get("generated_token_count")),
        "generated_token_hash_count": len(token_hashes),
        "generated_token_hashes": token_hashes,
        "multi_token_alpha_verified": bool(
            not errors
            and report.get("same_request_decode_verified") is True
            and _int(report.get("generated_token_count")) >= int(min_tokens)
            and len(token_hashes) >= int(min_tokens)
        ),
        "accepted_providers": sorted(providers),
        "all_required_providers_present": set(live_probe.REQUIRED_PROVIDERS).issubset(providers),
        "stage_count": _int(report.get("stage_count")),
        "expected_stage_task_count": _int(report.get("expected_stage_task_count")),
        "coordinator_stage_reports_collected": _int(report.get("coordinator_stage_reports_collected")),
        "worker_stage_decode_task_count": _int(report.get("worker_stage_decode_task_count")),
        "runtime_tuning": _dict(report.get("runtime_tuning")),
        "cleanup_status": cleanup,
        "cleanup_verified": bool(
            cleanup.get("temporary_kaggle_kernels_deleted") is True
            and cleanup.get("temporary_private_packages_removed") is True
            and cleanup.get("live_resources_left_running") is False
        ),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def build_service_smoke_summary(path: Path | None, report: dict[str, Any]) -> dict[str, Any]:
    errors = service_smoke_check.validate_report(report, require_verified=True) if report else ["service_smoke_report_missing"]
    generate = _dict(report.get("generate"))
    status = _dict(report.get("status"))
    cleanup = _dict(report.get("cleanup"))
    return {
        "schema": "glm52_kaggle_alpha_service_smoke_summary_v1",
        "source_path": str(path) if path is not None else "",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "service_smoke_check_ok": not errors,
        "service_smoke_check_errors": errors,
        "service_http_smoke_verified": report.get("service_http_smoke_verified") is True,
        "health_route_verified": _dict(report.get("health")).get("ok") is True,
        "status_route_verified": _dict(report.get("status")).get("http_status") == 200,
        "generate_route_reaches_service": report.get("generate_route_reaches_service") is True,
        "generate_route_quota_blocker_verified": report.get("generate_route_quota_blocker_verified") is True,
        "generate_route_success_verified": report.get("generate_route_success_verified") is True,
        "status_resume_private_inputs_verified": report.get("status_resume_private_inputs_verified") is True,
        "generate_resume_private_inputs_verified": report.get("generate_resume_private_inputs_verified") is True,
        "cleanup_route_verified": report.get("cleanup_route_verified") is True,
        "cleanup_http_status": _int(cleanup.get("http_status")),
        "cleanup_temporary_kaggle_kernels_deleted": cleanup.get("temporary_kaggle_kernels_deleted") is True,
        "cleanup_temporary_private_packages_removed": cleanup.get("temporary_private_packages_removed") is True,
        "cleanup_live_resources_left_running": cleanup.get("live_resources_left_running") is True,
        "generate_http_status": _int(generate.get("http_status")),
        "status_phase": str(status.get("phase") or ""),
        "generated_token_count": _int(generate.get("generated_token_count")),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def build_generate_cli_summary(path: Path | None, report: dict[str, Any]) -> dict[str, Any]:
    response = _dict(report.get("response"))
    artifact_recovery = _dict(report.get("artifact_recovery"))
    resume_private_inputs = _dict(report.get("resume_private_inputs")) or _dict(
        artifact_recovery.get("resume_private_inputs")
    )
    diagnosis_codes = [str(item) for item in _list(report.get("diagnosis_codes")) if item]
    errors: list[str] = []
    if report:
        if report.get("schema") != GENERATE_CLI_SCHEMA:
            errors.append("schema_mismatch")
        if report.get("command") != "generate" or report.get("target") != "glm52-kaggle":
            errors.append("target_mismatch")
        if report.get("public_artifact_safe") is not True:
            errors.append("public_artifact_safe_missing")
        if report.get("raw_prompt_public") is True or report.get("raw_generated_text_public") is True:
            errors.append("private_text_public")
        if response and response.get("public_artifact_safe") is not True:
            errors.append("response_public_artifact_safe_missing")
        if artifact_recovery.get("present") is True:
            if artifact_recovery.get("next_resume_command") or report.get("next_resume_command"):
                pass
            else:
                errors.append("artifact_recovery_resume_command_missing")
            if resume_private_inputs.get("schema") != alpha.RESUME_PRIVATE_INPUTS_SCHEMA:
                errors.append("artifact_recovery_resume_private_inputs_missing")
            if resume_private_inputs.get("resume_command_omits_private_credentials") is not True:
                errors.append("artifact_recovery_resume_private_inputs_redaction_missing")
    else:
        errors.append("generate_cli_report_missing")
    return {
        "schema": "glm52_kaggle_alpha_generate_cli_summary_v1",
        "source_path": str(path) if path is not None else "",
        "present": bool(report),
        "source_schema": str(report.get("schema") or ""),
        "generate_cli_check_ok": bool(report and not errors),
        "generate_cli_check_errors": errors,
        "http_status": _int(report.get("http_status")),
        "ok": report.get("ok") is True,
        "artifact_recovery_present": artifact_recovery.get("present") is True,
        "artifact_recovery_phase": str(artifact_recovery.get("phase") or ""),
        "artifact_recovery_blocker_count": len(_list(artifact_recovery.get("blockers"))),
        "artifact_recovery_resume_command_present": bool(
            str(artifact_recovery.get("next_resume_command") or report.get("next_resume_command") or "")
        ),
        "artifact_recovery_resume_private_inputs_verified": bool(
            resume_private_inputs.get("schema") == alpha.RESUME_PRIVATE_INPUTS_SCHEMA
            and resume_private_inputs.get("resume_command_omits_private_credentials") is True
            and resume_private_inputs.get("public_artifact_safe") is True
        ),
        "diagnosis_codes": sorted(set(diagnosis_codes)),
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def build_benchmark(live: dict[str, Any]) -> dict[str, Any]:
    status = _dict(live.get("coordinator_status"))
    return {
        "schema": BENCHMARK_SCHEMA,
        "deploy_time_seconds": float(status.get("elapsed_seconds") or live.get("duration_seconds") or 0.0),
        "stage_count": _int(live.get("stage_count")),
        "provider_coverage": _list(live.get("accepted_providers")),
        "first_token_latency_seconds": None,
        "stage_latency_available": True,
        "tokens_generated": _int(live.get("generated_token_count")),
        "runtime_tuning": _dict(live.get("runtime_tuning")),
        "cleanup_status": _dict(live.get("cleanup_status")),
        "public_artifact_safe": True,
    }


def shell_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if str(part) != "")


def phase_entry(name: str, state: str, *, detail: str = "", evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "state": state,
        "detail": detail,
        "evidence": evidence or {},
        "public_artifact_safe": True,
    }


def build_phase_status(
    *,
    service_summary: dict[str, Any],
    live_summary: dict[str, Any],
    gpu_quota_summary: dict[str, Any],
    cleanup_verified: bool,
    blockers: set[str],
    min_tokens: int,
) -> dict[str, Any]:
    service_ready = bool(
        service_summary.get("service_api_ready") is True
        and service_summary.get("model_request_supported") is True
        and service_summary.get("accelerator_request_complete") is True
        and service_summary.get("hf_token_env_supported") is True
        and service_summary.get("hf_token_public") is not True
        and service_summary.get("resume_private_inputs_ready") is True
        and service_summary.get("cli_generate_command_available") is True
        and service_summary.get("cli_generate_artifact_recovery_supported") is True
        and service_summary.get("cli_serve_default_matches_deploy") is True
        and service_summary.get("cli_status_default_matches_deploy") is True
        and service_summary.get("cli_cleanup_default_matches_deploy") is True
        and service_summary.get("status_loads_existing_alpha_artifacts") is True
        and service_summary.get("status_exposes_resume_private_inputs") is True
        and service_summary.get("generate_validates_request_schema") is True
        and service_summary.get("health_route_ready") is True
        and service_summary.get("status_route_ready") is True
        and service_summary.get("generate_route_ready") is True
        and service_summary.get("cleanup_route_ready") is True
        and service_summary.get("generate_accepts_prompt") is True
        and service_summary.get("generate_accepts_max_new_tokens") is True
        and service_summary.get("generate_accepts_timeout") is True
        and service_summary.get("generate_uses_current_gpu_quota_blocker") is True
        and service_summary.get("kaggle_runtime_blocker_classification_ready") is True
    )
    required_providers = set(live_probe.REQUIRED_PROVIDERS)
    accepted_providers = {str(item) for item in _list(live_summary.get("accepted_providers"))}
    quota_blocked = gpu_quota_summary.get("all_auth_ok_accounts_gpu_quota_exhausted") is True
    live_present = live_summary.get("present") is True
    completed_stage_tasks = _int(live_summary.get("coordinator_stage_reports_collected"))
    expected_stage_tasks = _int(live_summary.get("expected_stage_task_count"))
    generated_tokens = _int(live_summary.get("generated_token_count"))
    stage_state = "completed" if live_summary.get("multi_token_alpha_verified") is True else "not_started"
    if completed_stage_tasks > 0 and not live_summary.get("multi_token_alpha_verified"):
        stage_state = "partial"
    decode_state = "completed" if live_summary.get("multi_token_alpha_verified") is True else "blocked"
    if not live_present and quota_blocked:
        decode_state = "not_started"
    kernel_state = "completed" if live_present and expected_stage_tasks > 0 else "not_started"
    if quota_blocked and not live_present:
        kernel_state = "blocked"
    gpu_state = "completed" if "kaggle_cuda" in accepted_providers else ("blocked" if quota_blocked else "not_started")
    tpu_state = "completed" if "kaggle_jax_tpu" in accepted_providers else "not_started"
    cpu_state = "completed" if "kaggle_cpu" in accepted_providers else "not_started"
    phases = [
        phase_entry(
            "configuration_check",
            "completed" if service_ready else "blocked",
            detail="service routes and generate request contract",
            evidence={
                "service_api_ready": service_summary.get("service_api_ready") is True,
                "requested_model": str(service_summary.get("requested_model") or ""),
                "model_request_supported": service_summary.get("model_request_supported") is True,
                "accelerators": _list(service_summary.get("accelerators")),
                "required_accelerators": _list(service_summary.get("required_accelerators")),
                "accelerator_request_complete": service_summary.get("accelerator_request_complete") is True,
                "hf_token_env_supported": service_summary.get("hf_token_env_supported") is True,
                "hf_token_env_configured": service_summary.get("hf_token_env_configured") is True,
                "hf_token_env_configured_count": _int(service_summary.get("hf_token_env_configured_count")),
                "hf_token_public": service_summary.get("hf_token_public") is True,
                "resume_private_inputs_ready": service_summary.get("resume_private_inputs_ready") is True,
                "cli_generate_command_available": service_summary.get("cli_generate_command_available") is True,
                "cli_generate_artifact_recovery_supported": service_summary.get("cli_generate_artifact_recovery_supported") is True,
                "cli_serve_default_matches_deploy": service_summary.get("cli_serve_default_matches_deploy") is True,
                "cli_status_default_matches_deploy": service_summary.get("cli_status_default_matches_deploy") is True,
                "cli_cleanup_default_matches_deploy": service_summary.get("cli_cleanup_default_matches_deploy") is True,
                "status_loads_existing_alpha_artifacts": service_summary.get("status_loads_existing_alpha_artifacts") is True,
                "status_exposes_resume_private_inputs": service_summary.get("status_exposes_resume_private_inputs") is True,
                "generate_validates_request_schema": service_summary.get("generate_validates_request_schema") is True,
                "health_route_ready": service_summary.get("health_route_ready") is True,
                "status_route_ready": service_summary.get("status_route_ready") is True,
                "generate_route_ready": service_summary.get("generate_route_ready") is True,
                "cleanup_route_ready": service_summary.get("cleanup_route_ready") is True,
                "generate_uses_current_gpu_quota_blocker": service_summary.get("generate_uses_current_gpu_quota_blocker") is True,
                "kaggle_runtime_blocker_classification_ready": service_summary.get("kaggle_runtime_blocker_classification_ready") is True,
                "kaggle_runtime_blocker_classes": _list(service_summary.get("kaggle_runtime_blocker_classes")),
                "generate_request_fields": _list(service_summary.get("generate_request_fields")),
            },
        ),
        phase_entry(
            "model_source_check",
            "completed",
            detail="GLM 5.2 compatible quantized source selected",
            evidence={
                "model_id": alpha.MODEL_ID,
                "compatible_weight_repo": alpha.COMPATIBLE_WEIGHT_REPO,
                "non_glm_fallback": False,
            },
        ),
        phase_entry(
            "gpu_quota_preflight",
            "blocked" if quota_blocked else ("completed" if gpu_quota_summary.get("report_count") else "not_checked"),
            detail="Kaggle GPU quota availability",
            evidence={
                "account_count": _int(gpu_quota_summary.get("account_count")),
                "auth_ok_count": _int(gpu_quota_summary.get("auth_ok_count")),
                "gpu_submission_accepted_count": _int(gpu_quota_summary.get("gpu_submission_accepted_count")),
                "weekly_gpu_quota_exhausted_count": _int(gpu_quota_summary.get("weekly_gpu_quota_exhausted_count")),
                "next_quota_refresh_time": str(gpu_quota_summary.get("next_quota_refresh_time") or ""),
            },
        ),
        phase_entry(
            "kernel_push",
            kernel_state,
            detail="temporary Kaggle worker package/kernel push",
            evidence={
                "live_report_present": live_present,
                "stage_count": _int(live_summary.get("stage_count")),
                "quota_blocked_before_live": bool(quota_blocked and not live_present),
            },
        ),
        phase_entry("gpu_queue_running", gpu_state, detail="Kaggle CUDA provider", evidence={"provider": "kaggle_cuda", "accepted": "kaggle_cuda" in accepted_providers}),
        phase_entry("tpu_queue_running", tpu_state, detail="Kaggle JAX TPU provider", evidence={"provider": "kaggle_jax_tpu", "accepted": "kaggle_jax_tpu" in accepted_providers}),
        phase_entry("cpu_queue_running", cpu_state, detail="Kaggle CPU provider", evidence={"provider": "kaggle_cpu", "accepted": "kaggle_cpu" in accepted_providers}),
        phase_entry(
            "stage_completed",
            stage_state,
            detail="Coordinator stage task completion",
            evidence={
                "coordinator_stage_reports_collected": completed_stage_tasks,
                "expected_stage_task_count": expected_stage_tasks,
                "worker_stage_decode_task_count": _int(live_summary.get("worker_stage_decode_task_count")),
            },
        ),
        phase_entry(
            "decode_completed",
            decode_state,
            detail="same-request multi-token decode",
            evidence={
                "generated_token_count": generated_tokens,
                "min_required_generated_tokens": int(min_tokens),
                "same_request_decode_verified": live_summary.get("same_request_decode_verified") is True,
            },
        ),
        phase_entry(
            "cleanup_completed",
            "completed" if cleanup_verified else "blocked",
            detail="temporary kernels/private packages cleanup",
            evidence={"cleanup_verified": bool(cleanup_verified)},
        ),
    ]
    phase_names = [str(item.get("name")) for item in phases]
    return {
        "schema": PHASE_STATUS_SCHEMA,
        "overall_state": "ready" if not blockers else ("blocked" if any(item.get("state") == "blocked" for item in phases) else "incomplete"),
        "phases": phases,
        "phase_names": phase_names,
        "blocked_phase_names": [str(item.get("name")) for item in phases if item.get("state") == "blocked"],
        "completed_phase_names": [str(item.get("name")) for item in phases if item.get("state") == "completed"],
        "public_artifact_safe": True,
    }


def build_gpu_quota_summary(paths: list[str]) -> dict[str, Any]:
    reports = []
    account_summaries: list[dict[str, Any]] = []
    refresh_times: list[str] = []
    cleanup_verified_by_report: list[bool] = []
    for raw_path in paths:
        path = Path(str(raw_path))
        report = load_json(path)
        if not report:
            reports.append({"source_path": str(path), "present": False})
            cleanup_verified_by_report.append(False)
            continue
        accounts = []
        account_cleanup_ok = True
        for account in [item for item in _list(report.get("accounts")) if isinstance(item, dict)]:
            cleanup = _dict(account.get("cleanup"))
            if account.get("push_accepted") is True and cleanup.get("deleted") is not True:
                account_cleanup_ok = False
            quota = _dict(account.get("accelerator_quota"))
            refresh_time = str(quota.get("quota_refresh_time") or "")
            if refresh_time:
                refresh_times.append(refresh_time)
            gpu_quota = _dict(quota.get("gpu_quota"))
            item = {
                "label": str(account.get("label") or ""),
                "owner": str(account.get("owner") or ""),
                "auth_ok": account.get("auth_ok") is True,
                "quota_class": str(account.get("quota_class") or ""),
                "push_accepted": account.get("push_accepted") is True,
                "weekly_gpu_quota_exhausted": account.get("weekly_gpu_quota_exhausted") is True,
                "weekly_gpu_quota_exhausted_by_api": account.get("weekly_gpu_quota_exhausted_by_api") is True,
                "gpu_reserved_exceeds_remaining_by_api": account.get("gpu_reserved_exceeds_remaining_by_api") is True,
                "effective_remaining_after_reserved_seconds": float(gpu_quota.get("effective_remaining_after_reserved_seconds") or 0.0),
                "quota_refresh_time": refresh_time,
            }
            accounts.append(item)
            account_summaries.append(item)
        reports.append(
            {
                "source_path": str(path),
                "present": True,
                "source_schema": str(report.get("schema") or ""),
                "public_artifact_safe": report.get("public_artifact_safe") is True,
                "summary": _dict(report.get("summary")),
                "accounts": accounts,
                "private_kernel_payloads_removed": report.get("private_kernel_payloads_removed") is True,
                "cleanup_verified": bool(report.get("private_kernel_payloads_removed") is True and account_cleanup_ok),
            }
        )
        cleanup_verified_by_report.append(bool(report.get("private_kernel_payloads_removed") is True and account_cleanup_ok))
    accepted_count = sum(1 for item in account_summaries if item.get("push_accepted") is True)
    exhausted_count = sum(1 for item in account_summaries if item.get("weekly_gpu_quota_exhausted") is True)
    auth_ok_count = sum(1 for item in account_summaries if item.get("auth_ok") is True)
    return {
        "schema": "glm52_kaggle_alpha_gpu_quota_summary_v1",
        "report_count": len(reports),
        "reports": reports,
        "account_count": len(account_summaries),
        "auth_ok_count": auth_ok_count,
        "gpu_submission_accepted_count": accepted_count,
        "weekly_gpu_quota_exhausted_count": exhausted_count,
        "all_auth_ok_accounts_gpu_quota_exhausted": bool(auth_ok_count > 0 and exhausted_count >= auth_ok_count and accepted_count == 0),
        "cleanup_verified": bool(reports and all(cleanup_verified_by_report)),
        "quota_refresh_times": sorted(set(refresh_times)),
        "next_quota_refresh_time": sorted(set(refresh_times))[0] if refresh_times else "",
        "public_artifact_safe": True,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    service_path = Path(args.service_report) if str(args.service_report or "") else output_dir / "glm52_kaggle_alpha_service.json"
    live_path = Path(args.live_report) if str(args.live_report or "") else None
    runtime_blocker_path = Path(args.runtime_blocker_report) if str(args.runtime_blocker_report or "") else None
    service_smoke_path = Path(args.service_smoke_report) if str(args.service_smoke_report or "") else None
    generate_cli_path = Path(args.generate_cli_report) if str(getattr(args, "generate_cli_report", "") or "") else None
    service = load_json(service_path)
    live = load_json(live_path) if live_path is not None else {}
    runtime_blocker = load_json(runtime_blocker_path) if runtime_blocker_path is not None else {}
    service_smoke = load_json(service_smoke_path) if service_smoke_path is not None else {}
    generate_cli = load_json(generate_cli_path) if generate_cli_path is not None else {}
    gpu_quota_summary = build_gpu_quota_summary([str(item) for item in _list(getattr(args, "gpu_quota_report", []))])
    min_tokens = max(2, int(args.min_tokens))
    service_summary = build_service_summary(service_path, service)
    live_summary = build_live_summary(live_path, live, min_tokens=min_tokens)
    service_smoke_summary = build_service_smoke_summary(service_smoke_path, service_smoke)
    generate_cli_summary = build_generate_cli_summary(generate_cli_path, generate_cli)
    runtime_tuning = _dict(live.get("runtime_tuning")) or _dict(service_summary.get("runtime_tuning"))
    benchmark = build_benchmark(live)
    if runtime_tuning and not benchmark.get("runtime_tuning"):
        benchmark["runtime_tuning"] = runtime_tuning

    blockers: set[str] = set()
    if service_summary.get("service_api_ready") is not True:
        blockers.add("glm52_alpha_service_api_not_ready")
    if service_summary.get("model_request_supported") is not True:
        blockers.add("glm52_alpha_model_request_not_supported")
    if service_summary.get("accelerator_request_complete") is not True:
        blockers.add("glm52_alpha_accelerator_request_incomplete")
    if service_summary.get("hf_token_env_supported") is not True:
        blockers.add("glm52_alpha_hf_token_env_contract_missing")
    if service_summary.get("hf_token_public") is True:
        blockers.add("glm52_alpha_hf_token_public")
    if service_summary.get("resume_private_inputs_ready") is not True:
        blockers.add("glm52_alpha_resume_private_inputs_missing")
    if service_summary.get("cli_generate_command_available") is not True:
        blockers.add("glm52_alpha_cli_generate_command_missing")
    if service_summary.get("cli_generate_artifact_recovery_supported") is not True:
        blockers.add("glm52_alpha_cli_generate_artifact_recovery_missing")
    if service_summary.get("cli_serve_default_matches_deploy") is not True:
        blockers.add("glm52_alpha_cli_serve_default_mismatch")
    if service_summary.get("cli_status_default_matches_deploy") is not True:
        blockers.add("glm52_alpha_cli_status_default_mismatch")
    if service_summary.get("cli_cleanup_default_matches_deploy") is not True:
        blockers.add("glm52_alpha_cli_cleanup_default_mismatch")
    if service_summary.get("status_loads_existing_alpha_artifacts") is not True:
        blockers.add("glm52_alpha_status_existing_artifact_load_missing")
    if service_summary.get("status_exposes_resume_private_inputs") is not True:
        blockers.add("glm52_alpha_status_resume_private_inputs_missing")
    if service_summary.get("generate_validates_request_schema") is not True:
        blockers.add("glm52_alpha_generate_request_validation_missing")
    if service_summary.get("generate_routes_to_same_request_live_probe") is not True:
        blockers.add("glm52_alpha_generate_not_routed_to_same_request_pipeline")
    if service_summary.get("generate_uses_current_gpu_quota_blocker") is not True:
        blockers.add("glm52_alpha_generate_current_gpu_quota_blocker_missing")
    if service_summary.get("kaggle_runtime_blocker_classification_ready") is not True:
        blockers.add("glm52_alpha_kaggle_runtime_blocker_classification_missing")
    for field in ["health_route_ready", "status_route_ready", "generate_route_ready", "cleanup_route_ready"]:
        if service_summary.get(field) is not True:
            blockers.add(f"glm52_alpha_{field}_missing")
    for field in ["generate_accepts_prompt", "generate_accepts_max_new_tokens", "generate_accepts_timeout"]:
        if service_summary.get(field) is not True:
            blockers.add(f"glm52_alpha_{field}_missing")
    if service_summary.get("raw_prompt_public") is True or service_summary.get("raw_generated_text_public") is True:
        blockers.add("glm52_alpha_private_text_public")
    if live_summary.get("multi_token_alpha_verified") is not True:
        blockers.add("glm52_alpha_multitoken_live_not_verified")
    if live_summary.get("generated_token_count", 0) < min_tokens:
        blockers.add("glm52_alpha_generated_token_count_below_minimum")
    if live_summary.get("generated_token_hash_count", 0) < min_tokens:
        blockers.add("glm52_alpha_generated_token_hashes_below_minimum")
    if live_summary.get("all_required_providers_present") is not True:
        blockers.add("glm52_alpha_required_provider_coverage_missing")
    if live_summary.get("cleanup_verified") is not True:
        blockers.add("glm52_alpha_cleanup_not_verified")
    if live_summary.get("model_id") and live_summary.get("model_id") != alpha.MODEL_ID:
        blockers.add("glm52_alpha_model_id_not_glm52")
    if live_summary.get("source_schema") == "glm52_kaggle_accelerator_deployment_rc_v1":
        blockers.add("glm52_alpha_old_rc_artifact_is_not_alpha_live_report")
    if live_summary.get("source_schema") == live_probe.SCHEMA and live_summary.get("generated_token_count") == 1:
        blockers.add("glm52_alpha_old_single_token_live_report")
    blockers.update(str(item) for item in _list(live.get("blockers")) if item)
    blockers.update(str(item) for item in _list(live_summary.get("live_check_errors")) if item)
    if runtime_blocker:
        blockers.update(str(item) for item in _list(runtime_blocker.get("blockers")) if item)
        failure_stage = str(runtime_blocker.get("failure_stage") or "")
        if failure_stage:
            blockers.add(failure_stage)
    if gpu_quota_summary.get("all_auth_ok_accounts_gpu_quota_exhausted") is True:
        blockers.add("kaggle_gpu_quota_unavailable")
    quota_skip_cleanup_verified = bool(
        not live
        and gpu_quota_summary.get("all_auth_ok_accounts_gpu_quota_exhausted") is True
        and gpu_quota_summary.get("cleanup_verified") is True
    )
    cleanup_verified = live_summary.get("cleanup_verified") is True or quota_skip_cleanup_verified
    if quota_skip_cleanup_verified:
        blockers.discard("glm52_alpha_cleanup_not_verified")

    success = not blockers
    stage_worker_package_report = (
        str(service_summary.get("stage_worker_package_report") or "")
        or str(live.get("stage_worker_package_report") or "")
        or alpha.DEFAULT_STAGE_WORKER_PACKAGE_REPORT
    )
    requested_model = str(service_summary.get("requested_model") or alpha.COMPATIBLE_WEIGHT_REPO)
    requested_accelerators = ",".join(
        str(item) for item in (_list(service_summary.get("accelerators")) or list(alpha.REQUIRED_ACCELERATORS))
    )
    resume_command = [
        "crowdtensor",
        "deploy",
        "glm52-kaggle",
        "--run-live",
        "--gpu-quota-preflight",
        "--output-dir",
        str(output_dir),
        "--model",
        requested_model,
        "--accelerators",
        requested_accelerators,
        "--max-new-tokens",
        str(min_tokens),
        "--stage-worker-package-report",
        stage_worker_package_report,
    ]
    stage_push_parallelism = _int(service_summary.get("stage_push_parallelism"))
    if stage_push_parallelism > 0:
        resume_command.extend(["--stage-push-parallelism", str(stage_push_parallelism)])
    for key, flag in [
        ("gpu_accelerator", "--gpu-accelerator"),
        ("tpu_accelerator", "--tpu-accelerator"),
    ]:
        value = str(service_summary.get(key) or "")
        if value:
            resume_command.extend([flag, value])
    for key, flag in [
        ("wait_seconds", "--wait-seconds"),
        ("poll_interval_seconds", "--poll-interval-seconds"),
        ("command_timeout_seconds", "--command-timeout-seconds"),
        ("kernel_timeout_seconds", "--kernel-timeout-seconds"),
        ("coordinator_task_timeout_seconds", "--coordinator-task-timeout-seconds"),
        ("coordinator_worker_poll_interval_seconds", "--coordinator-worker-poll-interval-seconds"),
    ]:
        value = service_summary.get(key)
        if isinstance(value, (int, float)) and value > 0:
            resume_command.extend([flag, str(value)])
    tuning_arg_map = {
        "full_prefix_prefill_length": "--full-prefix-prefill-length",
        "full_prefix_dsa_mask_topk": "--full-prefix-dsa-mask-topk",
        "full_prefix_executed_expert_count": "--full-prefix-executed-expert-count",
        "full_prefix_top_k": "--full-prefix-top-k",
        "full_prefix_row_block_size": "--full-prefix-row-block-size",
        "full_prefix_max_tensor_bytes": "--full-prefix-max-tensor-bytes",
        "full_prefix_max_block_bytes": "--full-prefix-max-block-bytes",
        "cpu_group_stage_attempt_seconds": "--cpu-group-stage-attempt-seconds",
        "cpu_group_stage_poll_seconds": "--cpu-group-stage-poll-seconds",
    }
    for key, flag in tuning_arg_map.items():
        value = runtime_tuning.get(key)
        if isinstance(value, (int, float)) and value > 0:
            resume_command.extend([flag, str(value)])
    next_resume_command = shell_command(resume_command)
    blocker_report = {
        "schema": BLOCKER_SCHEMA,
        "generated_at": utc_now(),
        "blocked": not success,
        "blockers": [] if success else sorted(blockers),
        "deployment_engineering_complete": service_summary.get("service_api_ready") is True,
        "external_resource_blockers": {
            "kaggle_gpu_quota_unavailable": gpu_quota_summary.get("all_auth_ok_accounts_gpu_quota_exhausted") is True,
            "next_quota_refresh_time": str(gpu_quota_summary.get("next_quota_refresh_time") or ""),
        },
        "next_resume_command": next_resume_command,
        "next_resume_command_redacts_credentials": True,
        "resume_private_inputs": _dict(service_summary.get("resume_private_inputs")),
        "public_artifact_safe": True,
    }
    phase_status = build_phase_status(
        service_summary=service_summary,
        live_summary=live_summary,
        gpu_quota_summary=gpu_quota_summary,
        cleanup_verified=cleanup_verified,
        blockers=blockers,
        min_tokens=min_tokens,
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": success,
        "glm52_kaggle_alpha_ready": success,
        "model_id": alpha.MODEL_ID,
        "compatible_weight_repo": alpha.COMPATIBLE_WEIGHT_REPO,
        "requested_model": service_summary.get("requested_model") or alpha.COMPATIBLE_WEIGHT_REPO,
        "accelerators": service_summary.get("accelerators") or list(alpha.REQUIRED_ACCELERATORS),
        "min_required_generated_tokens": min_tokens,
        "service_api_ready": service_summary.get("service_api_ready") is True,
        "generate_routes_to_same_request_live_probe": service_summary.get("generate_routes_to_same_request_live_probe") is True,
        "same_request_multitoken_verified": live_summary.get("multi_token_alpha_verified") is True,
        "generated_token_count": live_summary.get("generated_token_count"),
        "generated_token_hashes": live_summary.get("generated_token_hashes"),
        "accepted_providers": live_summary.get("accepted_providers"),
        "cleanup_verified": cleanup_verified,
        "benchmark": benchmark,
        "runtime_tuning": runtime_tuning,
        "service_summary": service_summary,
        "service_smoke_summary": service_smoke_summary,
        "generate_cli_summary": generate_cli_summary,
        "live_summary": live_summary,
        "runtime_blocker": runtime_blocker,
        "gpu_quota_summary": gpu_quota_summary,
        "phase_status": phase_status,
        "blocker_report": blocker_report,
        "next_resume_command": "" if success else next_resume_command,
        "next_resume_command_redacts_credentials": True,
        "resume_private_inputs": _dict(service_summary.get("resume_private_inputs")),
        "blockers": [] if success else sorted(blockers),
        "boundaries": {
            "old_one_token_rc_is_not_success": True,
            "mock_only_is_not_success": True,
            "single_backend_is_not_success": True,
            "queue_only_is_not_success": True,
            "non_glm_fallback_is_not_success": True,
            "public_artifact_contains_no_tokens_or_cookies": True,
        },
        "safety": alpha.safety_flags(),
        "public_artifact_safe": True,
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["glm52_kaggle_alpha_ready"] = False
        report["public_artifact_safe"] = False
        report["blockers"] = sorted(set(_list(report.get("blockers")) + ["public_redaction_scan_failed"]))
    write_json(output_dir / "glm52_kaggle_alpha_blocker.json", blocker_report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--service-report", default="")
    parser.add_argument("--live-report", default="")
    parser.add_argument("--runtime-blocker-report", default="")
    parser.add_argument("--service-smoke-report", default="")
    parser.add_argument("--generate-cli-report", default="")
    parser.add_argument("--gpu-quota-report", action="append", default=[])
    parser.add_argument("--min-tokens", type=int, default=MIN_TARGET_TOKENS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.min_tokens < 2:
        raise SystemExit("--min-tokens must be at least 2")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    report = build_report(args)
    support = {
        "schema": SUPPORT_SCHEMA,
        "generated_at": utc_now(),
        "summary": {
            "ok": report.get("ok") is True,
            "generated_token_count": report.get("generated_token_count"),
            "accepted_providers": report.get("accepted_providers"),
            "blockers": report.get("blockers"),
        },
        "public_artifact_safe": True,
    }
    benchmark = dict(_dict(report.get("benchmark")))
    if benchmark:
        benchmark["generated_at"] = utc_now()
        benchmark["source_alpha_report"] = "glm52_kaggle_alpha.json"
        benchmark["alpha_ready"] = report.get("glm52_kaggle_alpha_ready") is True
        benchmark["blockers"] = report.get("blockers") if isinstance(report.get("blockers"), list) else []
        benchmark["public_artifact_safe"] = True
    else:
        benchmark = {
            "schema": BENCHMARK_SCHEMA,
            "generated_at": utc_now(),
            "source_alpha_report": "glm52_kaggle_alpha.json",
            "alpha_ready": False,
            "deploy_time_seconds": 0.0,
            "stage_count": 0,
            "provider_coverage": [],
            "first_token_latency_seconds": None,
            "stage_latency_available": False,
            "tokens_generated": 0,
            "cleanup_status": {},
            "blockers": report.get("blockers") if isinstance(report.get("blockers"), list) else [],
            "public_artifact_safe": True,
        }
    write_json(output_dir / "glm52_kaggle_alpha_support.json", support)
    benchmark_path = output_dir / "glm52_kaggle_alpha_benchmark.json"
    write_json(benchmark_path, benchmark)
    path = output_dir / "glm52_kaggle_alpha.json"
    write_json(path, report)
    report["artifacts"] = {
        "summary_json": artifact_entry(path, output_dir, kind="glm52_kaggle_alpha_summary", schema=SCHEMA, ok=bool(report.get("ok"))),
        "support_json": artifact_entry(output_dir / "glm52_kaggle_alpha_support.json", output_dir, kind="glm52_kaggle_alpha_support", schema=SUPPORT_SCHEMA, ok=True),
        "benchmark_json": artifact_entry(benchmark_path, output_dir, kind="glm52_kaggle_alpha_benchmark", schema=BENCHMARK_SCHEMA, ok=bool(report.get("glm52_kaggle_alpha_ready"))),
        "blocker_json": artifact_entry(output_dir / "glm52_kaggle_alpha_blocker.json", output_dir, kind="glm52_kaggle_alpha_blocker", schema=BLOCKER_SCHEMA, ok=not bool(report.get("blockers"))),
    }
    if str(getattr(args, "service_smoke_report", "") or ""):
        smoke_path = Path(args.service_smoke_report)
        report["artifacts"]["service_smoke_json"] = artifact_entry(
            smoke_path,
            output_dir,
            kind="glm52_kaggle_alpha_service_smoke",
            schema="glm52_kaggle_alpha_service_smoke_probe_v1",
            ok=_dict(report.get("service_smoke_summary")).get("service_http_smoke_verified") is True,
        )
    if str(getattr(args, "generate_cli_report", "") or ""):
        generate_cli_path = Path(args.generate_cli_report)
        report["artifacts"]["generate_cli_json"] = artifact_entry(
            generate_cli_path,
            output_dir,
            kind="glm52_kaggle_alpha_generate_cli",
            schema=GENERATE_CLI_SCHEMA,
            ok=_dict(report.get("generate_cli_summary")).get("generate_cli_check_ok") is True,
        )
    write_json(path, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
        print(f"Alpha ready: {report.get('glm52_kaggle_alpha_ready')}")
        print(f"Generated tokens: {report.get('generated_token_count')}")
    return 0 if report.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
