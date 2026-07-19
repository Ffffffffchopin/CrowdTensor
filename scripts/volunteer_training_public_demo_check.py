#!/usr/bin/env python3
"""Check a retained Volunteer Campaign public preview without rerunning it."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_value
from crowdtensor.training_contract import sha256_file, sha256_json


SCHEMA = "crowdtensor_volunteer_training_public_demo_v1"
CHECK_SCHEMA = "crowdtensor_volunteer_training_public_demo_check_v1"
HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("public demo report must be a JSON object")
    return value


def _safe(value: dict[str, Any], errors: list[str], prefix: str) -> None:
    for field in (
        "credential_values_public",
        "private_paths_public",
        "raw_data_public",
        "tensor_values_public",
    ):
        if value.get(field) is not False:
            errors.append(f"{prefix}_{field}_not_false")
    if value.get("public_artifact_safe") is not True:
        errors.append(f"{prefix}_public_artifact_safe_missing")


def check(report_path: str | Path, *, require_verified: bool = False) -> dict[str, Any]:
    path = Path(report_path).expanduser().resolve()
    errors: list[str] = []
    try:
        report = _read(path)
    except Exception as exc:
        return {
            "schema": CHECK_SCHEMA,
            "ok": False,
            "verified": False,
            "error_count": 1,
            "errors": ["public_demo_report_load_failed:" + type(exc).__name__],
        }
    if report.get("schema") != SCHEMA:
        errors.append("public_demo_schema_mismatch")
    expected = sha256_json(
        {key: value for key, value in report.items() if key != "content_hash"}
    )
    if report.get("content_hash") != expected or not HASH.fullmatch(
        str(report.get("content_hash") or "")
    ):
        errors.append("public_demo_content_hash_invalid")
    _safe(report, errors, "public_demo")
    if report.get("demo_scope") != "same_host_two_independent_cell_processes":
        errors.append("public_demo_scope_invalid")
    if report.get("physical_multi_host_verified") is not False:
        errors.append("public_demo_physical_multi_host_overclaim")
    cells = report.get("cell_processes")
    if (
        not isinstance(cells, list)
        or len(cells) != 2
        or len({str(item.get("cell_id_hash") or "") for item in cells if isinstance(item, dict)}) != 2
    ):
        errors.append("public_demo_two_independent_cells_missing")
    else:
        for item in cells:
            if not isinstance(item, dict):
                errors.append("public_demo_cell_summary_invalid")
                continue
            _safe(item, errors, "public_demo_cell")
            if (
                item.get("ok") is not True
                or item.get("work_completed") is not True
                or int(
                    item.get("process_returncode")
                    if item.get("process_returncode") is not None
                    else -1
                )
                != 0
            ):
                errors.append("public_demo_cell_not_completed")
            if item.get("real_transformers_peft_lora") is not True:
                errors.append("public_demo_cell_real_peft_missing")
    dashboard = report.get("dashboard_routes")
    if not isinstance(dashboard, dict) or not all(
        dashboard.get(field) is True
        for field in ("health", "dashboard", "stylesheet", "script", "content_security_policy")
    ):
        errors.append("public_demo_dashboard_routes_incomplete")
    progress = report.get("progress") if isinstance(report.get("progress"), dict) else {}
    if (
        int(progress.get("completed_rounds") or 0) < 1
        or int(progress.get("accepted_update_count") or 0) < 2
        or int(progress.get("adapter_version") or 0) < 1
    ):
        errors.append("public_demo_progress_incomplete")
    claims = report.get("claims") if isinstance(report.get("claims"), dict) else {}
    for field in (
        "model_quality_improvement_claimed",
        "permissionless_training_claimed",
        "sybil_resistance_claimed",
        "poisoning_resistance_claimed",
        "internet_multi_host_claimed",
    ):
        if claims.get(field) is not False:
            errors.append("public_demo_claim_boundary_missing:" + field)
    cleanup = report.get("cleanup") if isinstance(report.get("cleanup"), dict) else {}
    if (
        cleanup.get("cleanup_verified") is not True
        or cleanup.get("live_resources_left_running") is not False
        or cleanup.get("private_runtime_removed") is not True
    ):
        errors.append("public_demo_cleanup_incomplete")
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    hashes = report.get("artifacts_sha256") if isinstance(report.get("artifacts_sha256"), dict) else {}
    root = path.parent
    for name, relative in artifacts.items():
        artifact = (root / str(relative)).resolve()
        try:
            artifact.relative_to(root)
        except ValueError:
            errors.append("public_demo_artifact_escapes_report:" + str(name))
            continue
        expected_hash = str(hashes.get(name) or "")
        if not artifact.is_file():
            errors.append("public_demo_artifact_missing:" + str(name))
        elif not HASH.fullmatch(expected_hash) or sha256_file(artifact) != expected_hash:
            errors.append("public_demo_artifact_hash_invalid:" + str(name))
    if report.get("public_artifact_scan_ok") is not True:
        errors.append("public_demo_public_scan_failed")
    privacy = scan_public_value(report)
    if privacy.get("ok") is not True:
        errors.append("public_demo_report_privacy_failed")
    verified = not errors
    if require_verified and not verified:
        errors.append("public_demo_required_verification_missing")
    return {
        "schema": CHECK_SCHEMA,
        "ok": not errors if not require_verified else verified,
        "verified": verified,
        "error_count": len(errors),
        "errors": sorted(set(errors)),
        "report_content_hash": report.get("content_hash"),
        "public_artifact_safe": privacy.get("ok") is True,
        "physical_multi_host_verified": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-verified", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = check(args.report, require_verified=args.require_verified)
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else f"verified={result['verified']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
