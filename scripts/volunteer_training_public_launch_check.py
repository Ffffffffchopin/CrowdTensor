#!/usr/bin/env python3
"""Check the public founding-preview bundle and its formal-launch boundary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_files, scan_public_value
from crowdtensor.training_contract import sha256_file, sha256_json
from crowdtensor.volunteer_campaign_proposal import validate_proposal
from scripts.volunteer_training_operator_beta_check import check as check_operator
from scripts.volunteer_training_public_demo_check import check as check_demo


SCHEMA = "crowdtensor_volunteer_training_public_launch_rc_v1"
CHECK_SCHEMA = "crowdtensor_volunteer_training_public_launch_check_v1"
MULTI_HOST_SCHEMA = "crowdtensor_volunteer_training_physical_multihost_evidence_v1"
HASH = re.compile(r"^sha256:[0-9a-f]{64}$")

REQUIRED_ARTIFACTS = {
    "readme",
    "governance",
    "launch_kit",
    "proposal",
    "proposal_schema",
    "demo_report",
    "demo_snapshot",
    "demo_status",
    "operator_rc",
    "visual_report",
    "desktop_screenshot",
    "mobile_screenshot",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("launch report must be a JSON object")
    return value


def _safe(value: dict[str, Any], errors: list[str], prefix: str) -> None:
    for field in (
        "credential_values_public",
        "credential_paths_public",
        "cookies_public",
        "private_urls_public",
        "private_paths_public",
        "raw_training_text_public",
        "raw_data_public",
        "token_ids_public",
        "activation_values_public",
        "gradient_values_public",
        "checkpoint_tensor_values_public",
    ):
        if field in value and value.get(field) is not False:
            errors.append(f"{prefix}_{field}_not_false")
    if value.get("public_artifact_safe") is not True:
        errors.append(f"{prefix}_public_artifact_safe_missing")


def _validate_multi_host(path: Path) -> tuple[bool, list[str], dict[str, Any]]:
    errors: list[str] = []
    try:
        value = _read(path)
    except Exception as exc:
        return False, ["formal_multihost_report_load_failed:" + type(exc).__name__], {}
    if value.get("schema") != MULTI_HOST_SCHEMA:
        errors.append("formal_multihost_schema_mismatch")
    expected = sha256_json({key: item for key, item in value.items() if key != "content_hash"})
    if value.get("content_hash") != expected or not HASH.fullmatch(
        str(value.get("content_hash") or "")
    ):
        errors.append("formal_multihost_content_hash_invalid")
    _safe(value, errors, "formal_multihost")
    required_true = (
        "physical_multi_host_verified",
        "independent_host_identities_verified",
        "independent_admin_domains_verified",
        "real_network_route_verified",
        "cleanup_verified",
    )
    for field in required_true:
        if value.get(field) is not True:
            errors.append("formal_multihost_field_missing:" + field)
    if int(value.get("independent_host_count") or 0) < 2:
        errors.append("formal_multihost_host_count_insufficient")
    privacy = scan_public_value(value)
    if privacy.get("ok") is not True:
        errors.append("formal_multihost_public_safety_failed")
    return not errors, errors, value


def check(report_path: str | Path, *, require_formal: bool = False) -> dict[str, Any]:
    path = Path(report_path).expanduser().resolve()
    errors: list[str] = []
    try:
        report = _read(path)
    except Exception as exc:
        return {
            "schema": CHECK_SCHEMA,
            "ok": False,
            "founding_preview_ready": False,
            "formal_launch_ready": False,
            "error_count": 1,
            "errors": ["launch_report_load_failed:" + type(exc).__name__],
        }
    if report.get("schema") != SCHEMA:
        errors.append("launch_rc_schema_mismatch")
    expected = sha256_json({key: item for key, item in report.items() if key != "content_hash"})
    if report.get("content_hash") != expected or not HASH.fullmatch(
        str(report.get("content_hash") or "")
    ):
        errors.append("launch_rc_content_hash_invalid")
    _safe(report, errors, "launch_rc")
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    hashes = report.get("artifact_hashes") if isinstance(report.get("artifact_hashes"), dict) else {}
    if not REQUIRED_ARTIFACTS.issubset(artifacts):
        errors.append("launch_rc_required_artifacts_missing")
    root = path.parent
    resolved: dict[str, Path] = {}
    for name, relative in artifacts.items():
        artifact = (root / str(relative)).resolve()
        try:
            artifact.relative_to(root)
        except ValueError:
            errors.append("launch_rc_artifact_escapes_bundle:" + str(name))
            continue
        resolved[str(name)] = artifact
        if not artifact.is_file():
            errors.append("launch_rc_artifact_missing:" + str(name))
            continue
        expected_hash = str(hashes.get(name) or "")
        if not HASH.fullmatch(expected_hash) or sha256_file(artifact) != expected_hash:
            errors.append("launch_rc_artifact_hash_invalid:" + str(name))

    proposal_result: dict[str, Any] = {}
    if resolved.get("proposal", Path()).is_file():
        try:
            proposal_result = validate_proposal(_read(resolved["proposal"]))
            if proposal_result.get("ok") is not True:
                errors.append("launch_rc_proposal_not_ready")
        except Exception as exc:
            errors.append("launch_rc_proposal_invalid:" + type(exc).__name__)
    else:
        errors.append("launch_rc_proposal_missing")

    demo_result = (
        check_demo(resolved["demo_report"], require_verified=True)
        if resolved.get("demo_report", Path()).is_file()
        else {"verified": False, "errors": ["demo_missing"]}
    )
    if demo_result.get("verified") is not True:
        errors.append("launch_rc_demo_not_verified")

    operator_result = (
        check_operator(resolved["operator_rc"], require_ready=True)
        if resolved.get("operator_rc", Path()).is_file()
        else {"ok": False, "errors": ["operator_missing"]}
    )
    if operator_result.get("ok") is not True:
        errors.append("launch_rc_operator_beta_not_verified")

    visual_result: dict[str, Any] = {}
    if resolved.get("visual_report", Path()).is_file():
        try:
            visual_result = _read(resolved["visual_report"])
            expected_visual = sha256_json(
                {key: item for key, item in visual_result.items() if key != "content_hash"}
            )
            if visual_result.get("content_hash") != expected_visual:
                errors.append("launch_rc_visual_report_content_hash_invalid")
            if visual_result.get("ok") is not True:
                errors.append("launch_rc_visual_probe_not_ready")
            for viewport in ("desktop", "mobile"):
                item = (visual_result.get("viewports") or {}).get(viewport) or {}
                if (
                    item.get("canvas_nonblank") is not True
                    or item.get("horizontal_overflow") is not False
                    or item.get("vertical_order_coherent") is not True
                ):
                    errors.append("launch_rc_visual_viewport_invalid:" + viewport)
        except Exception as exc:
            errors.append("launch_rc_visual_report_invalid:" + type(exc).__name__)
    else:
        errors.append("launch_rc_visual_report_missing")

    for name, phrase in (
        ("readme", "open campaigns for volunteer model"),
        ("governance", "physical multi-host"),
        ("launch_kit", "LocalLLaMA"),
    ):
        text = resolved.get(name, Path()).read_text(encoding="utf-8") if resolved.get(name, Path()).is_file() else ""
        if phrase not in text:
            errors.append("launch_rc_document_phrase_missing:" + name)

    public_files = [
        resolved[name]
        for name in REQUIRED_ARTIFACTS
        if name != "readme"
        and name in resolved
        and resolved[name].suffix.lower() in {".json", ".md", ".yml"}
    ]
    public_scan = scan_public_files(public_files)
    if public_scan.get("ok") is not True:
        errors.append("launch_rc_public_safety_scan_failed")
    external = report.get("formal_multihost") if isinstance(report.get("formal_multihost"), dict) else {}
    external_ready = False
    if external.get("artifact") and resolved.get("formal_multihost", Path()).is_file():
        external_ready, external_errors, _external_value = _validate_multi_host(
            resolved["formal_multihost"]
        )
        errors.extend(external_errors)
    else:
        external_errors = ["formal_multihost_evidence_missing"]
    founding_ready = not errors
    formal_ready = founding_ready and external_ready
    reported_founding = report.get("founding_preview_ready") is True
    reported_formal = report.get("formal_launch_ready") is True
    if reported_founding != founding_ready:
        errors.append("launch_rc_founding_readiness_mismatch")
    if reported_formal != formal_ready:
        errors.append("launch_rc_formal_readiness_mismatch")
    if require_formal and not formal_ready:
        errors.append("formal_launch_required_external_multihost_evidence_missing")
    # The default check intentionally succeeds for a complete founding preview
    # even while formal launch remains blocked by the external gate.
    ok = not errors if not require_formal else formal_ready and not errors
    return {
        "schema": CHECK_SCHEMA,
        "ok": ok,
        "founding_preview_ready": founding_ready,
        "formal_launch_ready": formal_ready,
        "formal_multihost_evidence_present": external_ready,
        "error_count": len(sorted(set(errors))),
        "errors": sorted(set(errors)),
        "proposal_ready": proposal_result.get("ok") is True,
        "demo_verified": demo_result.get("verified") is True,
        "operator_beta_verified": operator_result.get("ok") is True,
        "visual_probe_verified": visual_result.get("ok") is True,
        "public_artifact_safe": public_scan.get("ok") is True,
        "formal_multihost_blockers": external_errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-formal", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = check(args.report, require_formal=args.require_formal)
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else f"founding_preview_ready={result['founding_preview_ready']} formal_launch_ready={result['formal_launch_ready']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
