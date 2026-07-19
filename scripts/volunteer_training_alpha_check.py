#!/usr/bin/env python3
"""Strict checker for Volunteer Training Protocol Alpha public evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from crowdtensor.training_contract import sha256_file, sha256_json


SCHEMA = "crowdtensor_volunteer_training_alpha_rc_v1"
PROBE_SCHEMA = "crowdtensor_volunteer_training_alpha_probe_v1"
HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _true(value: dict[str, Any], fields: tuple[str, ...], errors: list[str], prefix: str) -> None:
    for field in fields:
        if value.get(field) is not True:
            errors.append(f"{prefix}_{field}_missing")


def _public_safety(value: Any, errors: list[str], prefix: str) -> None:
    if not isinstance(value, dict):
        errors.append(f"{prefix}_object_missing")
        return
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


def check(report_path: str | Path, *, require_ready: bool = False) -> dict[str, Any]:
    path = Path(report_path).resolve()
    errors: list[str] = []
    try:
        report = _read(path)
    except Exception as exc:
        return {
            "schema": "crowdtensor_volunteer_training_alpha_check_v1",
            "ok": False,
            "volunteer_training_protocol_alpha_ready": False,
            "goal_achieved": False,
            "errors": ["volunteer_alpha_report_load_failed:" + type(exc).__name__],
        }
    if report.get("schema") != SCHEMA:
        errors.append("volunteer_alpha_rc_schema_mismatch")
    content_hash = str(report.get("content_hash") or "")
    expected_content_hash = sha256_json(
        {key: value for key, value in report.items() if key != "content_hash"}
    )
    if not HASH.fullmatch(content_hash) or content_hash != expected_content_hash:
        errors.append("volunteer_alpha_rc_content_hash_mismatch")
    _public_safety(report, errors, "rc")
    if report.get("evidence_scope") != "local_http_real_peft_protocol_alpha":
        errors.append("volunteer_alpha_evidence_scope_invalid")
    if report.get("protocol_version") != "volunteer_training_v1.0":
        errors.append("volunteer_alpha_protocol_version_invalid")
    if not HASH.fullmatch(str(report.get("campaign_manifest_hash") or "")):
        errors.append("volunteer_alpha_campaign_hash_missing")

    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    hashes = report.get("artifact_hashes") if isinstance(report.get("artifact_hashes"), dict) else {}
    required_artifacts = {
        "probe",
        "campaign",
        "status",
        "audit_ledger",
        "baseline",
        "churn",
        "update_validation",
        "communication",
        "cleanup",
        "ledger_check",
        "http_service",
        "join_workflow",
    }
    if not required_artifacts.issubset(artifacts):
        errors.append("volunteer_alpha_required_artifacts_missing")
    artifact_paths: dict[str, Path] = {}
    for name, relative in artifacts.items():
        artifact = (path.parent / str(relative)).resolve()
        artifact_paths[str(name)] = artifact
        if not artifact.is_file():
            errors.append(f"volunteer_alpha_artifact_{name}_missing")
            continue
        expected_hash = str(hashes.get(name) or "")
        if not HASH.fullmatch(expected_hash) or sha256_file(artifact) != expected_hash:
            errors.append(f"volunteer_alpha_artifact_{name}_hash_mismatch")

    probe: dict[str, Any] = {}
    if artifact_paths.get("probe", Path("/missing")).is_file():
        try:
            probe = _read(artifact_paths["probe"])
        except Exception as exc:
            errors.append("volunteer_alpha_probe_load_failed:" + type(exc).__name__)
    if probe.get("schema") != PROBE_SCHEMA:
        errors.append("volunteer_alpha_probe_schema_mismatch")
    if probe.get("ok") is not True or probe.get("volunteer_training_protocol_alpha_verified") is not True:
        errors.append("volunteer_alpha_probe_not_verified")
    if probe.get("campaign_manifest_hash") != report.get("campaign_manifest_hash"):
        errors.append("volunteer_alpha_probe_campaign_hash_mismatch")
    if probe.get("public_artifact_scan_ok") is not True:
        errors.append("volunteer_alpha_probe_public_scan_failed")
    for field in (
        "real_training",
        "round_progress",
        "churn_proof",
        "update_validation",
        "centralized_baseline",
        "communication",
        "contributor_workflow",
        "http_service",
        "audit_ledger",
        "cleanup",
        "limitations",
    ):
        if probe.get(field) != report.get(field):
            errors.append(f"volunteer_alpha_probe_{field}_summary_mismatch")

    real = report.get("real_training") if isinstance(report.get("real_training"), dict) else {}
    _true(
        real,
        ("pytorch_autograd", "transformers_peft_lora", "base_weights_frozen"),
        errors,
        "real_training",
    )
    if real.get("mock_only") is not False:
        errors.append("volunteer_alpha_mock_only_evidence_rejected")
    if int(real.get("cell_update_count") or 0) < 4 or int(real.get("optimizer_steps") or 0) < 8:
        errors.append("volunteer_alpha_real_update_budget_insufficient")

    rounds = report.get("round_progress") if isinstance(report.get("round_progress"), dict) else {}
    _true(
        rounds,
        ("all_rounds_distinct_cell_quorum", "atomic_version_advance"),
        errors,
        "round_progress",
    )
    if (
        int(rounds.get("adapter_version_before", -1)) != 0
        or int(rounds.get("adapter_version_after") or 0) < 2
        or int(rounds.get("outer_step_after") or 0) < 2
        or int(rounds.get("completed_rounds") or 0) < 2
        or int(rounds.get("accepted_update_count") or 0) < 4
        or int(rounds.get("minimum_quorum") or 0) < 2
    ):
        errors.append("volunteer_alpha_round_progress_incomplete")

    churn = report.get("churn_proof") if isinstance(report.get("churn_proof"), dict) else {}
    _true(
        churn,
        (
            "cell_disappeared_before_submit",
            "same_work_reassigned",
            "generation_advanced",
            "replacement_used_canonical_adapter",
            "duplicate_retry_idempotent",
            "duplicate_retry_accepted",
        ),
        errors,
        "churn",
    )
    if int(churn.get("expired_lease_count") or 0) < 1:
        errors.append("volunteer_alpha_expired_lease_missing")
    stale = churn.get("late_stale_delta_rejection") if isinstance(
        churn.get("late_stale_delta_rejection"), dict
    ) else {}
    if (
        stale.get("rejected") is not True
        or stale.get("expected_code_observed") is not True
        or stale.get("code") != "volunteer_stale_adapter_version_rejected"
    ):
        errors.append("volunteer_alpha_late_stale_fence_missing")

    validation = report.get("update_validation") if isinstance(
        report.get("update_validation"), dict
    ) else {}
    _true(
        validation,
        (
            "tensor_contract_validation",
            "content_hash_validation",
            "finite_value_validation",
            "norm_clipping_policy",
            "hard_norm_rejection_policy",
            "distinct_cell_quorum",
        ),
        errors,
        "update_validation",
    )
    fork = validation.get("forked_base_hash_rejection") if isinstance(
        validation.get("forked_base_hash_rejection"), dict
    ) else {}
    non_finite = validation.get("non_finite_delta_rejection") if isinstance(
        validation.get("non_finite_delta_rejection"), dict
    ) else {}
    if fork.get("code") != "base_adapter_hash_mismatch" or fork.get("rejected") is not True:
        errors.append("volunteer_alpha_fork_fence_missing")
    if non_finite.get("code") != "adapter_delta_non_finite" or non_finite.get("rejected") is not True:
        errors.append("volunteer_alpha_non_finite_rejection_missing")

    baseline = report.get("centralized_baseline") if isinstance(
        report.get("centralized_baseline"), dict
    ) else {}
    _true(
        baseline,
        (
            "real_pytorch_autograd",
            "real_transformers_peft_lora",
            "same_optimizer_step_budget",
            "same_token_budget",
            "same_dataset_snapshot",
            "same_batch_sequence_contract",
            "results_compared_not_quality_equated",
        ),
        errors,
        "baseline",
    )
    if int(baseline.get("distributed_optimizer_steps") or 0) != int(
        baseline.get("centralized_optimizer_steps") or -1
    ):
        errors.append("volunteer_alpha_baseline_compute_budget_mismatch")
    if int(baseline.get("distributed_tokens_seen") or 0) != int(
        baseline.get("centralized_tokens_seen") or -1
    ):
        errors.append("volunteer_alpha_baseline_token_budget_mismatch")
    for field in (
        "initial_validation_loss",
        "distributed_validation_loss",
        "centralized_validation_loss",
        "distributed_loss_progress",
        "centralized_loss_progress",
    ):
        try:
            finite = math.isfinite(float(baseline[field]))
        except (KeyError, TypeError, ValueError):
            finite = False
        if not finite:
            errors.append(f"volunteer_alpha_baseline_{field}_invalid")
    if baseline.get("useful_model_quality_claimed") is not False:
        errors.append("volunteer_alpha_fixture_quality_overclaim")
    if baseline.get("broad_scalability_claimed") is not False:
        errors.append("volunteer_alpha_scalability_overclaim")

    communication = report.get("communication") if isinstance(
        report.get("communication"), dict
    ) else {}
    if communication.get("low_frequency_delta_transport_verified") is not True:
        errors.append("volunteer_alpha_low_frequency_transport_missing")
    if communication.get("per_layer_activation_wan_transport_used") is not False:
        errors.append("volunteer_alpha_per_layer_wan_transport_used")
    if int(communication.get("local_steps_per_delta") or 0) < 2:
        errors.append("volunteer_alpha_aggregation_interval_too_short")
    try:
        ratio = float(communication.get("measured_to_stepwise_upload_ratio"))
    except (TypeError, ValueError):
        ratio = math.inf
    if not math.isfinite(ratio) or not 0.0 < ratio < 1.0:
        errors.append("volunteer_alpha_communication_ratio_invalid")

    workflow = report.get("contributor_workflow") if isinstance(
        report.get("contributor_workflow"), dict
    ) else {}
    _true(
        workflow,
        (
            "one_command_join_verified",
            "hardware_detection",
            "resource_limits",
            "content_addressed_cache",
            "pause_resume_commands",
            "lease_heartbeat",
            "private_invite_required",
        ),
        errors,
        "workflow",
    )
    if workflow.get("command") != "crowdtensor volunteer join <private-invite> --once":
        errors.append("volunteer_alpha_join_command_invalid")
    if int(workflow.get("command_exit_code", -1)) != 0:
        errors.append("volunteer_alpha_join_command_failed")

    service = report.get("http_service") if isinstance(report.get("http_service"), dict) else {}
    _true(
        service,
        (
            "health_route_verified",
            "claim_route_verified",
            "authenticated_artifact_download_verified",
            "binary_safetensors_submission_verified",
            "heartbeat_route_enabled",
            "loopback_http_service_stopped",
        ),
        errors,
        "http_service",
    )
    if service.get("physical_internet_route_verified") is not False:
        errors.append("volunteer_alpha_physical_internet_overclaim")

    ledger = report.get("audit_ledger") if isinstance(report.get("audit_ledger"), dict) else {}
    if ledger.get("ok") is not True or int(ledger.get("event_count") or 0) < 10:
        errors.append("volunteer_alpha_audit_ledger_unverified")
    if ledger.get("errors") != []:
        errors.append("volunteer_alpha_audit_ledger_errors_present")
    cleanup = report.get("cleanup") if isinstance(report.get("cleanup"), dict) else {}
    _true(
        cleanup,
        (
            "http_service_stopped",
            "all_cell_processes_stopped",
            "private_runtime_removed",
            "canonical_public_evidence_preserved",
            "cleanup_verified",
        ),
        errors,
        "cleanup",
    )
    if cleanup.get("live_resources_left_running") is not False:
        errors.append("volunteer_alpha_live_resources_left_running")

    artifact_section_map = {
        "baseline": "centralized_baseline",
        "churn": "churn_proof",
        "update_validation": "update_validation",
        "communication": "communication",
        "cleanup": "cleanup",
        "ledger_check": "audit_ledger",
        "http_service": "http_service",
        "join_workflow": "contributor_workflow",
    }
    for artifact_name, report_field in artifact_section_map.items():
        artifact = artifact_paths.get(artifact_name)
        if artifact is None or not artifact.is_file():
            continue
        try:
            artifact_value = _read(artifact)
        except Exception as exc:
            errors.append(
                f"volunteer_alpha_artifact_{artifact_name}_load_failed:{type(exc).__name__}"
            )
            continue
        if artifact_value != report.get(report_field):
            errors.append(f"volunteer_alpha_artifact_{artifact_name}_summary_mismatch")

    campaign_artifact = artifact_paths.get("campaign")
    if campaign_artifact is not None and campaign_artifact.is_file():
        try:
            campaign = _read(campaign_artifact)
        except Exception as exc:
            errors.append("volunteer_alpha_campaign_load_failed:" + type(exc).__name__)
            campaign = {}
        if campaign.get("schema") != "crowdtensor_volunteer_training_campaign_v1":
            errors.append("volunteer_alpha_campaign_schema_mismatch")
        if campaign.get("protocol_version") != report.get("protocol_version"):
            errors.append("volunteer_alpha_campaign_protocol_mismatch")
        if campaign.get("manifest_hash") != report.get("campaign_manifest_hash"):
            errors.append("volunteer_alpha_campaign_manifest_hash_mismatch")
        policy = campaign.get("round_policy") if isinstance(campaign.get("round_policy"), dict) else {}
        if int(policy.get("minimum_quorum") or 0) != int(rounds.get("minimum_quorum") or -1):
            errors.append("volunteer_alpha_campaign_quorum_mismatch")
        if int(policy.get("target_rounds") or 0) != int(rounds.get("completed_rounds") or -1):
            errors.append("volunteer_alpha_campaign_target_rounds_mismatch")

    status_artifact = artifact_paths.get("status")
    status: dict[str, Any] = {}
    if status_artifact is not None and status_artifact.is_file():
        try:
            status = _read(status_artifact)
        except Exception as exc:
            errors.append("volunteer_alpha_status_load_failed:" + type(exc).__name__)
        if status.get("schema") != "crowdtensor_volunteer_training_status_v1":
            errors.append("volunteer_alpha_status_schema_mismatch")
        if status.get("campaign_complete") is not True:
            errors.append("volunteer_alpha_status_campaign_incomplete")
        status_fields = {
            "adapter_version": "adapter_version_after",
            "outer_step": "outer_step_after",
            "completed_rounds": "completed_rounds",
            "accepted_update_count": "accepted_update_count",
        }
        for status_field, round_field in status_fields.items():
            if int(status.get(status_field) or 0) != int(rounds.get(round_field) or -1):
                errors.append(f"volunteer_alpha_status_{status_field}_mismatch")
        status_rounds = status.get("rounds") if isinstance(status.get("rounds"), list) else []
        if len(status_rounds) != int(rounds.get("completed_rounds") or -1):
            errors.append("volunteer_alpha_status_round_count_mismatch")
        elif any(
            item.get("state") != "completed"
            or int(item.get("distinct_accepted_cell_count") or 0)
            < int(rounds.get("minimum_quorum") or 0)
            for item in status_rounds
            if isinstance(item, dict)
        ):
            errors.append("volunteer_alpha_status_round_quorum_incomplete")

    ledger_artifact = artifact_paths.get("audit_ledger")
    computed_head = "sha256:" + "0" * 64
    computed_count = 0
    if ledger_artifact is not None and ledger_artifact.is_file():
        for line in ledger_artifact.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            computed_count += 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                errors.append("volunteer_alpha_ledger_json_invalid")
                break
            if not isinstance(event, dict) or event.get("previous_event_hash") != computed_head:
                errors.append("volunteer_alpha_ledger_chain_broken")
                break
            event_hash = sha256_json(
                {key: value for key, value in event.items() if key != "event_hash"}
            )
            if event.get("event_hash") != event_hash:
                errors.append("volunteer_alpha_ledger_event_hash_mismatch")
                break
            computed_head = event_hash
    if computed_count != int(ledger.get("event_count") or -1):
        errors.append("volunteer_alpha_ledger_event_count_mismatch")
    if computed_head != ledger.get("head_hash"):
        errors.append("volunteer_alpha_ledger_head_hash_mismatch")
    if status and (
        status.get("ledger_head_hash") != computed_head
        or int(status.get("ledger_sequence") or -1) != computed_count
    ):
        errors.append("volunteer_alpha_status_ledger_mismatch")

    limitations = report.get("limitations") if isinstance(report.get("limitations"), dict) else {}
    for field in (
        "physical_internet_multi_machine_verified",
        "permissionless_byzantine_safety",
        "sybil_resistance",
        "secure_aggregation",
        "useful_model_quality_claimed",
        "broad_scalability_claimed",
        "general_availability",
        "service_level_agreement",
    ):
        if limitations.get(field) is not False:
            errors.append(f"volunteer_alpha_limitation_{field}_not_false")
    if limitations.get("loopback_http_protocol_verified") is not True:
        errors.append("volunteer_alpha_loopback_scope_missing")

    forbidden_patterns = (
        r"Bearer\s+[A-Za-z0-9._~-]+",
        r'"invite_token"\s*:',
        r'"lease_token"\s*:',
        r'"input_ids"\s*:',
        r'"delta_path"\s*:',
        r'"base_model_path"\s*:',
        r'"dataset_path"\s*:',
    )
    public_paths = [path] + [
        artifact
        for name, artifact in artifact_paths.items()
        if name != "probe" and artifact.is_file()
    ]
    public_paths.append(artifact_paths.get("probe", Path("/missing")))
    for public_path in public_paths:
        if not public_path.is_file():
            continue
        text = public_path.read_text(encoding="utf-8", errors="replace")
        for pattern in forbidden_patterns:
            if re.search(pattern, text):
                errors.append(f"volunteer_alpha_public_artifact_private_material:{public_path.name}")
                break

    declared_ready = report.get("volunteer_training_protocol_alpha_ready") is True
    declared_achieved = report.get("goal_achieved") is True
    if declared_ready != declared_achieved:
        errors.append("volunteer_alpha_ready_goal_mismatch")
    ready = not errors and declared_ready and declared_achieved
    if require_ready and not ready and not errors:
        errors.append("volunteer_training_protocol_alpha_not_ready")
    return {
        "schema": "crowdtensor_volunteer_training_alpha_check_v1",
        "ok": not errors and (ready if require_ready else True),
        "volunteer_training_protocol_alpha_ready": ready,
        "goal_achieved": ready,
        "error_count": len(errors),
        "errors": errors,
        "evidence_scope": report.get("evidence_scope"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = check(args.report, require_ready=args.require_ready)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            "volunteer_training_protocol_alpha_ready="
            + str(result["volunteer_training_protocol_alpha_ready"])
        )
        for error in result["errors"]:
            print("error=" + error)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
