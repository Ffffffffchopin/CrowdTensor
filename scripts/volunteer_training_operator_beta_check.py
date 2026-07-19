#!/usr/bin/env python3
"""Strict checker for the Volunteer Campaign Single-Host Operator Beta RC."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_value
from crowdtensor.training_contract import sha256_file, sha256_json
from scripts.volunteer_training_internet_beta_check import check as check_real_peft
from scripts.volunteer_training_operator_beta_pack import (
    PROBE_SCHEMA,
    RELEASE_SCHEMA,
    SCHEMA,
)


CHECK_SCHEMA = "crowdtensor_volunteer_campaign_single_host_operator_beta_check_v1"
HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
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


def _true(
    value: dict[str, Any], fields: tuple[str, ...], errors: list[str], prefix: str
) -> None:
    for field in fields:
        if value.get(field) is not True:
            errors.append(f"{prefix}_{field}_missing")


def _false(
    value: dict[str, Any], fields: tuple[str, ...], errors: list[str], prefix: str
) -> None:
    for field in fields:
        if value.get(field) is not False:
            errors.append(f"{prefix}_{field}_not_false")


def check(report_path: str | Path, *, require_ready: bool = False) -> dict[str, Any]:
    path = Path(report_path).expanduser().resolve()
    errors: list[str] = []
    try:
        report = _read(path)
    except Exception as exc:
        return {
            "schema": CHECK_SCHEMA,
            "ok": False,
            "error_count": 1,
            "errors": ["operator_beta_report_load_failed:" + type(exc).__name__],
            "volunteer_campaign_single_host_operator_beta_ready": False,
            "goal_achieved": False,
        }
    if report.get("schema") != SCHEMA:
        errors.append("operator_beta_rc_schema_mismatch")
    expected_hash = sha256_json(
        {key: value for key, value in report.items() if key != "content_hash"}
    )
    if report.get("content_hash") != expected_hash or not HASH.fullmatch(
        str(report.get("content_hash") or "")
    ):
        errors.append("operator_beta_rc_content_hash_mismatch")
    _safe(report, errors, "rc")
    if report.get("evidence_scope") != (
        "same_host_https_minio_independent_process_operator_beta"
    ):
        errors.append("operator_beta_evidence_scope_invalid")

    root = path.parent.resolve()
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    hashes = report.get("artifact_hashes") if isinstance(
        report.get("artifact_hashes"), dict
    ) else {}
    required_artifacts = {
        "probe",
        "probe_security",
        "probe_deployment",
        "probe_lifecycle",
        "probe_stress",
        "probe_faults",
        "probe_monitoring",
        "probe_real_peft",
        "probe_cleanup",
        "release",
        "wheel",
        "doc_operator_runbook",
        "doc_internet_beta",
        "doc_readme",
    }
    if not required_artifacts.issubset(artifacts):
        errors.append("operator_beta_required_artifacts_missing")
    resolved: dict[str, Path] = {}
    for name, relative in artifacts.items():
        artifact = (root / str(relative)).resolve()
        resolved[str(name)] = artifact
        try:
            artifact.relative_to(root)
        except ValueError:
            errors.append("operator_beta_artifact_escapes_rc:" + str(name))
            continue
        if not artifact.is_file():
            errors.append("operator_beta_artifact_missing:" + str(name))
            continue
        expected = str(hashes.get(name) or "")
        if not HASH.fullmatch(expected) or sha256_file(artifact) != expected:
            errors.append("operator_beta_artifact_hash_mismatch:" + str(name))

    probe: dict[str, Any] = {}
    release: dict[str, Any] = {}
    try:
        probe = _read(resolved["probe"])
    except Exception as exc:
        errors.append("operator_beta_probe_load_failed:" + type(exc).__name__)
    try:
        release = _read(resolved["release"])
    except Exception as exc:
        errors.append("operator_beta_release_load_failed:" + type(exc).__name__)
    if probe.get("schema") != PROBE_SCHEMA:
        errors.append("operator_beta_probe_schema_mismatch")
    if release.get("schema") != RELEASE_SCHEMA:
        errors.append("operator_beta_release_schema_mismatch")
    for value, prefix in ((probe, "probe"), (release, "release")):
        if value:
            expected = sha256_json(
                {key: item for key, item in value.items() if key != "content_hash"}
            )
            if value.get("content_hash") != expected:
                errors.append(prefix + "_content_hash_mismatch")
            _safe(value, errors, prefix)
            if value.get("public_artifact_scan_ok") is not True:
                errors.append(prefix + "_public_artifact_scan_failed")
    if (
        probe.get("ok") is not True
        or probe.get("volunteer_campaign_single_host_operator_beta_verified")
        is not True
    ):
        errors.append("operator_beta_probe_not_verified")
    if release.get("ok") is not True:
        errors.append("operator_beta_release_not_verified")

    summary_fields = {
        "security": "probe_security",
        "deployment": "probe_deployment",
        "lifecycle": "probe_lifecycle",
        "stress": "probe_stress",
        "faults": "probe_faults",
        "monitoring": "probe_monitoring",
        "retained_real_peft": "probe_real_peft",
        "cleanup": "probe_cleanup",
    }
    for field, artifact_name in summary_fields.items():
        if probe.get(field) != report.get(field):
            errors.append("operator_beta_probe_summary_mismatch:" + field)
        artifact = resolved.get(artifact_name)
        if artifact and artifact.is_file():
            try:
                if _read(artifact) != report.get(field):
                    errors.append("operator_beta_section_artifact_mismatch:" + field)
            except Exception as exc:
                errors.append(
                    "operator_beta_section_load_failed:"
                    + field
                    + ":"
                    + type(exc).__name__
                )

    security = report.get("security") if isinstance(report.get("security"), dict) else {}
    _true(
        security,
        (
            "per_cell_short_lived_credential_verified",
            "scope_rejection_verified",
            "revocation_verified",
            "replay_rejection_verified",
            "request_rate_limit_verified",
            "credential_capacity_limit_verified",
            "upload_capacity_limit_verified",
        ),
        errors,
        "operator_beta_security",
    )
    if security.get("credential_values_persisted_publicly") is not False:
        errors.append("operator_beta_security_credential_persistence_overclaim")

    deployment = report.get("deployment") if isinstance(
        report.get("deployment"), dict
    ) else {}
    _true(
        deployment,
        (
            "same_physical_host",
            "coordinator_real_process",
            "https_reverse_proxy_real_container",
            "tls_handshake_and_certificate_verification",
            "direct_backend_http_rejected",
            "forwarded_proxy_identity_enforced",
            "minio_real_container",
            "s3_compatible_real_api_calls",
            "content_addressed_upload_store",
            "coordinator_process_restart_verified",
            "active_lease_preserved_across_restart",
            "https_reverse_proxy_restart_verified",
        ),
        errors,
        "operator_beta_deployment",
    )
    _false(
        deployment,
        ("physical_multi_host_verified", "external_managed_storage_sla_verified"),
        errors,
        "operator_beta_deployment",
    )
    images = deployment.get("container_images") if isinstance(
        deployment.get("container_images"), dict
    ) else {}
    for name in ("caddy", "minio"):
        image = images.get(name) if isinstance(images.get(name), dict) else {}
        if image.get("identity_verified") is not True or not HASH.fullmatch(
            str(image.get("image_id_hash") or "")
        ):
            errors.append("operator_beta_deployment_image_identity_invalid:" + name)

    lifecycle = report.get("lifecycle") if isinstance(report.get("lifecycle"), dict) else {}
    _true(
        lifecycle,
        (
            "validate_verified",
            "start_verified",
            "pause_verified",
            "resume_verified",
            "finalize_verified",
            "evaluate_verified",
            "export_verified",
            "backup_restore_verified",
            "upgrade_migration_verified",
        ),
        errors,
        "operator_beta_lifecycle",
    )
    stress = report.get("stress") if isinstance(report.get("stress"), dict) else {}
    process_count = int(stress.get("requested_process_count") or 0)
    if not 20 <= process_count <= 50:
        errors.append("operator_beta_stress_process_count_invalid")
    if (
        stress.get("ok") is not True
        or int(stress.get("independent_process_report_count") or 0) != process_count
        or stress.get("all_processes_stopped") is not True
        or stress.get("campaign_complete") is not True
        or int(stress.get("completed_round_count") or 0) < 3
        or int(stress.get("accepted_update_count") or 0) < 6
        or int(stress.get("protocol_only_process_count") or 0) != process_count
        or int(stress.get("real_training_process_count", -1)) != 0
        or stress.get("same_physical_host_only") is not True
        or stress.get("physical_multi_host_verified") is not False
        or float(stress.get("elapsed_seconds") or 0.0)
        > float(stress.get("bounded_gate_seconds") or 0.0) + 5.0
    ):
        errors.append("operator_beta_stress_contract_invalid")

    faults = report.get("faults") if isinstance(report.get("faults"), dict) else {}
    _true(
        faults,
        (
            "slow_cell_lease_expiry_verified",
            "upload_interruption_verified",
            "minio_unavailable_failure_verified",
            "minio_restart_verified",
            "upload_resume_without_retraining_verified",
            "duplicate_submission_idempotency_verified",
            "content_addressed_s3_blob_verified",
        ),
        errors,
        "operator_beta_faults",
    )
    monitoring = report.get("monitoring") if isinstance(
        report.get("monitoring"), dict
    ) else {}
    _true(
        monitoring,
        (
            "prometheus_text_endpoint_verified",
            "credential_metrics_present",
            "fault_counters_present",
        ),
        errors,
        "operator_beta_monitoring",
    )
    if monitoring.get("raw_metric_labels_public") is not False:
        errors.append("operator_beta_monitoring_private_labels_present")

    retained = report.get("retained_real_peft") if isinstance(
        report.get("retained_real_peft"), dict
    ) else {}
    if (
        retained.get("retained_real_peft_rc_verified") is not True
        or retained.get("model_id") != "HuggingFaceTB/SmolLM2-135M"
        or retained.get("dataset_id") != "Salesforce/wikitext"
        or int(retained.get("real_training_round_count") or 0) < 3
        or int(retained.get("real_optimizer_step_count") or 0) < 6
        or retained.get("fresh_real_peft_rerun_performed") is not False
    ):
        errors.append("operator_beta_retained_real_peft_invalid")
    real_relative = str(report.get("retained_real_peft_rc_artifact") or "")
    real_path = (root / real_relative).resolve()
    try:
        real_path.relative_to(root)
        real_check = check_real_peft(real_path, require_ready=True)
    except Exception as exc:
        real_check = {"ok": False, "errors": [type(exc).__name__]}
    if real_check.get("ok") is not True or real_check.get("goal_achieved") is not True:
        errors.append("operator_beta_retained_real_peft_strict_check_failed")
    if real_path.is_file() and sha256_file(real_path) != retained.get(
        "retained_real_peft_rc_file_hash"
    ):
        errors.append("operator_beta_retained_real_peft_hash_mismatch")

    release_summary = report.get("release") if isinstance(
        report.get("release"), dict
    ) else {}
    clean = release_summary.get("clean_install") if isinstance(
        release_summary.get("clean_install"), dict
    ) else {}
    container = release_summary.get("container") if isinstance(
        release_summary.get("container"), dict
    ) else {}
    wheel = release_summary.get("wheel") if isinstance(
        release_summary.get("wheel"), dict
    ) else {}
    wheel_contract = wheel.get("contract") if isinstance(wheel.get("contract"), dict) else {}
    _true(
        clean,
        (
            "fresh_isolated_venv",
            "wheel_installed_with_declared_base_dependencies",
            "workspace_pythonpath_removed",
            "installed_module_under_venv",
            "volunteer_contract_command_verified",
            "one_command_operator_help_verified",
            "dependency_check_verified",
            "temporary_venv_removed_after_probe",
        ),
        errors,
        "operator_beta_clean_install",
    )
    _true(
        container,
        (
            "compose_configuration_valid",
            "project_image_built_from_current_source",
            "non_root_configured_user_verified",
            "volunteer_contract_inside_container_verified",
            "image_removed",
        ),
        errors,
        "operator_beta_container",
    )
    if (
        clean.get("ok") is not True
        or container.get("ok") is not True
        or container.get("container_left_running") is not False
        or container.get("container_registry_publish_performed") is not False
        or release_summary.get("external_publish_performed") is not False
        or release_summary.get("public_artifact_scan_ok") is not True
        or not HASH.fullmatch(str(container.get("image_id_hash") or ""))
    ):
        errors.append("operator_beta_release_contract_invalid")
    _true(
        wheel_contract,
        (
            "required_modules_present",
            "volunteer_cli_entry_point_present",
            "storage_extra_declared",
            "hf_extra_declared",
        ),
        errors,
        "operator_beta_wheel",
    )
    wheel_path = resolved.get("wheel")
    if (
        wheel_path is None
        or not wheel_path.is_file()
        or sha256_file(wheel_path) != wheel.get("sha256")
        or wheel_path.stat().st_size != int(wheel.get("byte_count") or -1)
    ):
        errors.append("operator_beta_wheel_identity_invalid")

    cleanup = report.get("cleanup") if isinstance(report.get("cleanup"), dict) else {}
    _true(
        cleanup,
        (
            "coordinator_process_stopped",
            "worker_processes_stopped",
            "caddy_container_removed",
            "minio_container_removed",
            "s3_bucket_deleted",
            "temporary_uploads_removed",
            "private_temporary_workspace_removed",
            "cleanup_verified",
        ),
        errors,
        "operator_beta_cleanup",
    )
    if cleanup.get("live_resources_left_running") is not False:
        errors.append("operator_beta_cleanup_live_resources_present")

    limitations = report.get("limitations") if isinstance(
        report.get("limitations"), dict
    ) else {}
    _false(
        limitations,
        (
            "independent_physical_multi_host_test_performed",
            "sybil_resistance_claimed",
            "semantic_poisoning_safety_claimed",
            "byzantine_consensus_claimed",
            "general_availability_claimed",
            "service_level_agreement_claimed",
            "stress_process_training_is_real_peft",
        ),
        errors,
        "operator_beta_limitations",
    )
    documentation = report.get("documentation") if isinstance(
        report.get("documentation"), dict
    ) else {}
    operator_doc = resolved.get("doc_operator_runbook")
    if (
        len(documentation) != 3
        or operator_doc is None
        or not operator_doc.is_file()
    ):
        errors.append("operator_beta_documentation_missing")
    else:
        text = operator_doc.read_text(encoding="utf-8")
        for marker in (
            "per-Cell credential",
            "HTTPS and MinIO",
            "Backup and Upgrade",
            "Prometheus",
            "independent physical multi-host",
            "Sybil",
            "General Availability",
        ):
            if marker not in text:
                errors.append("operator_beta_documentation_marker_missing:" + marker)

    privacy = scan_public_value(report)
    if privacy.get("ok") is not True or report.get("public_artifact_scan_ok") is not True:
        errors.append("operator_beta_public_safety_invalid")
    ready_claim = report.get("volunteer_campaign_single_host_operator_beta_ready") is True
    goal_claim = report.get("goal_achieved") is True
    if require_ready and not ready_claim:
        errors.append("operator_beta_readiness_required")
    if ready_claim != goal_claim:
        errors.append("operator_beta_goal_readiness_mismatch")
    if errors and (ready_claim or goal_claim):
        errors.append("operator_beta_false_ready_claim")
    ready = not errors and ready_claim and goal_claim
    return {
        "schema": CHECK_SCHEMA,
        "ok": not errors,
        "error_count": len(sorted(set(errors))),
        "errors": sorted(set(errors)),
        "volunteer_campaign_single_host_operator_beta_ready": ready,
        "goal_achieved": ready,
        "physical_multi_host_verified": False,
        "retained_real_peft_strict_check_ok": real_check.get("ok") is True,
        "public_safety": privacy,
        "public_artifact_safe": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check(args.report, require_ready=bool(args.require_ready))
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            f"ok={result['ok']} ready={result['volunteer_campaign_single_host_operator_beta_ready']} errors={result['error_count']}"
        )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
