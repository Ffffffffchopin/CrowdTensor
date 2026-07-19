#!/usr/bin/env python3
"""Validate public-safe Kaggle GPU concurrency probe artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import kaggle_gpu_concurrency_probe as probe  # noqa: E402


SCHEMA = "kaggle_gpu_concurrency_check_v1"


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def validate_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != probe.SCHEMA:
        errors.append("schema_mismatch")
    if report.get("evidence_ready") is not True:
        errors.append("evidence_ready_missing")
    if report.get("public_artifact_safe") is not True:
        errors.append("public_artifact_safe_missing")
    if probe.public_redaction_errors(report):
        errors.append("public_redaction_scan_failed")
    if int(report.get("requested_kernel_count") or 0) < 2:
        errors.append("requested_kernel_count_below_two")
    if report.get("accelerator") != "NvidiaTeslaT4":
        errors.append("accelerator_not_t4")

    cleanup = _dict(report.get("cleanup"))
    if cleanup.get("attempted") is not True and int(report.get("accepted_submission_count") or 0) > 0:
        errors.append("cleanup_not_attempted")
    if _list(cleanup.get("failed_delete_refs")):
        errors.append("cleanup_failed")
    if report.get("private_kernel_payloads_removed") is not True:
        errors.append("private_kernel_payloads_not_removed")

    worker_reports = [item for item in _list(report.get("worker_reports")) if isinstance(item, dict)]
    if report.get("ok") is True:
        if report.get("simultaneous_t4x2_verified") is not True:
            errors.append("ok_without_simultaneous_t4x2_verified")
        if int(report.get("accepted_submission_count") or 0) < int(report.get("requested_kernel_count") or 0):
            errors.append("success_missing_accepted_kernels")
        if len(worker_reports) < int(report.get("requested_kernel_count") or 0):
            errors.append("success_missing_worker_reports")
        if int(report.get("max_observed_running_count") or 0) < 2 and report.get("worker_runtime_overlap_verified") is not True:
            errors.append("success_without_overlap_evidence")
        for item in worker_reports:
            if item.get("ok") is not True:
                errors.append("worker_report_not_ok")
            if item.get("public_artifact_safe") is not True:
                errors.append("worker_report_public_safe_missing")
            if item.get("raw_gpu_names_public") is not False:
                errors.append("worker_raw_gpu_names_public")
            if int(item.get("cuda_device_count") or 0) < 2:
                errors.append("worker_less_than_two_cuda_devices")
    else:
        if not _list(report.get("blockers")):
            errors.append("failed_probe_without_blockers")
        if report.get("simultaneous_t4x2_verified") is True:
            errors.append("failed_probe_claims_verified")

    safety = _dict(report.get("safety"))
    for flag in [
        "raw_gpu_names_public",
        "credentials_public",
        "cookies_public",
        "runtime_proxy_public",
        "activation_public",
        "hidden_state_public",
        "logits_public",
        "kv_cache_public",
    ]:
        if safety.get(flag) is not False:
            errors.append(f"safety_flag_mismatch:{flag}")
    return sorted(set(errors))


def build_check(args: argparse.Namespace) -> dict[str, Any]:
    report = load_json(Path(args.report))
    errors = validate_report(report)
    return {
        "schema": SCHEMA,
        "ok": not errors,
        "report_schema": report.get("schema"),
        "report_path": args.report,
        "probe_ok": report.get("ok"),
        "evidence_ready": report.get("evidence_ready"),
        "simultaneous_t4x2_verified": report.get("simultaneous_t4x2_verified"),
        "accepted_submission_count": report.get("accepted_submission_count"),
        "max_observed_running_count": report.get("max_observed_running_count"),
        "blockers": report.get("blockers") or [],
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = build_check(args)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"kaggle_gpu_concurrency_check: ok={result['ok']} errors={','.join(result['errors']) or 'none'}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
