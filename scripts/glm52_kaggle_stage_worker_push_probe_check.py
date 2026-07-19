#!/usr/bin/env python3
"""Validate GLM 5.2 Kaggle stage worker push probe reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "glm52_kaggle_stage_worker_push_probe_check_v1"
PROBE_SCHEMA = "glm52_kaggle_stage_worker_push_probe_v1"
REQUIRED_PROVIDERS = {"kaggle_cuda", "kaggle_jax_tpu", "kaggle_cpu"}
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
)


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def validate_report(report: dict[str, Any], *, require_live: bool = False) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != PROBE_SCHEMA:
        errors.append("schema_mismatch")
    if report.get("ok") is not True:
        errors.append("report_not_ok")
    if report.get("glm52_stage_worker_push_probe_ready") is not True:
        errors.append("push_probe_ready_missing")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    leaks = public_redaction_errors(report)
    if leaks:
        errors.append("public_redaction_scan_failed:" + ",".join(leaks[:8]))
    mode = str(report.get("mode") or "")
    if mode not in {"preflight", "live", "collect", "import"}:
        errors.append("mode_invalid")
    pushes = [item for item in _list(report.get("pushes")) if isinstance(item, dict)]
    providers = {str(push.get("provider") or "") for push in pushes}
    if require_live and not REQUIRED_PROVIDERS.issubset(providers):
        errors.append("required_provider_pushes_missing")
    for push in pushes:
        provider = str(push.get("provider") or "")
        if provider not in REQUIRED_PROVIDERS:
            errors.append(f"push_provider_not_required:{provider or 'missing'}")
        if push.get("public_artifact_safe") is not True:
            errors.append(f"push_public_artifact_unsafe:{provider or 'missing'}")
        if _int(push.get("stage_id"), -1) < 0:
            errors.append(f"push_stage_id_missing:{provider or 'missing'}")
        if mode == "preflight":
            if push.get("pushed") is not False:
                errors.append(f"preflight_push_overclaim:{provider or 'missing'}")
            if push.get("output_collected") is not False:
                errors.append(f"preflight_output_overclaim:{provider or 'missing'}")
            if push.get("stage_report_present") is True:
                errors.append(f"preflight_stage_report_overclaim:{provider or 'missing'}")
            if push.get("stage_runtime_verified") is True:
                errors.append(f"preflight_stage_runtime_overclaim:{provider or 'missing'}")
        retained_tpu_queue = bool(
            provider == "kaggle_jax_tpu"
            and "kaggle_tpu_kernel_retained_for_queue" in _list(report.get("blockers"))
            and str(push.get("terminal_status") or "").upper() in {"QUEUED", "RUNNING", "PREPARING"}
        )
        retained_gpu_queue = bool(
            provider == "kaggle_cuda"
            and "kaggle_gpu_kernel_retained_for_queue_or_run" in _list(report.get("blockers"))
            and str(push.get("terminal_status") or "").upper() in {"QUEUED", "RUNNING", "PREPARING", "PENDING", "UNKNOWN"}
        )
        retained_cpu_run = bool(
            provider == "kaggle_cpu"
            and "kaggle_cpu_kernel_retained_for_run" in _list(report.get("blockers"))
            and str(push.get("terminal_status") or "").upper() in {"QUEUED", "RUNNING", "PREPARING", "PENDING", "UNKNOWN"}
        )
        if (
            mode in {"live", "collect", "import"}
            and push.get("pushed") is True
            and push.get("cleanup_performed") is not True
            and not retained_tpu_queue
            and not retained_gpu_queue
            and not retained_cpu_run
        ):
            errors.append(f"live_push_cleanup_missing:{provider or 'missing'}")
        if (
            mode == "live"
            and push.get("coordinator_private_runtime_env_uploaded") is True
            and push.get("coordinator_private_runtime_env_inlined") is not True
        ):
            errors.append(f"coordinator_private_runtime_env_not_inlined:{provider or 'missing'}")
        if (
            mode == "live"
            and push.get("coordinator_private_runtime_env_uploaded") is True
            and push.get("private_runtime_env_kernel_restored") is not True
        ):
            errors.append(f"private_runtime_env_kernel_not_restored:{provider or 'missing'}")
        if mode in {"live", "collect", "import"} and push.get("output_collected") is True:
            stage_check = _dict(push.get("stage_report_check"))
            if require_live and push.get("stage_report_present") is not True:
                errors.append(f"stage_report_missing:{provider or 'missing'}")
            if require_live and (stage_check.get("ok") is not True or push.get("stage_runtime_verified") is not True):
                errors.append(f"stage_report_not_verified:{provider or 'missing'}")
            if str(stage_check.get("provider") or "") not in {"", provider}:
                errors.append(f"stage_report_provider_mismatch:{provider or 'missing'}")
            if stage_check.get("stage_id") not in {None, "", push.get("stage_id")} and _int(stage_check.get("stage_id"), -1) != _int(push.get("stage_id"), -2):
                errors.append(f"stage_report_stage_id_mismatch:{provider or 'missing'}")
    boundary = _dict(report.get("completion_boundary"))
    for key in [
        "preflight_is_not_runtime_success",
        "push_required",
        "terminal_kernel_output_required",
        "stage_runtime_check_required",
        "same_request_probe_required",
    ]:
        if boundary.get(key) is not True:
            errors.append(f"completion_boundary_missing:{key}")
    if require_live and report.get("live_run_performed") is not True:
        errors.append("live_run_not_performed")
    if require_live and _int(report.get("stage_runtime_reports_collected")) < len(REQUIRED_PROVIDERS):
        errors.append("stage_runtime_reports_not_collected")
    if require_live and _int(report.get("stage_runtime_reports_verified")) < len(REQUIRED_PROVIDERS):
        errors.append("stage_runtime_reports_not_verified")
    if mode == "import" and report.get("live_run_performed") is True and _int(report.get("stage_runtime_reports_collected")) <= 0:
        errors.append("import_live_without_stage_reports")
    if report.get("stage_runtime_adapter_verified") is True or report.get("same_request_route_verified") is True:
        errors.append("push_probe_must_not_claim_runtime_success")
    return sorted(set(errors))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-live", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = load_json(Path(args.report))
    errors = validate_report(report, require_live=bool(args.require_live))
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors,
        "mode": report.get("mode"),
        "live_run_performed": report.get("live_run_performed") is True,
        "stage_runtime_reports_collected": _int(report.get("stage_runtime_reports_collected")),
        "stage_runtime_reports_verified": _int(report.get("stage_runtime_reports_verified")),
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"glm52_kaggle_stage_worker_push_probe_check: ok={result['ok']} "
            f"errors={len(errors)} mode={result['mode']}"
        )
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
