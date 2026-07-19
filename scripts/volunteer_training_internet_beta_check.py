#!/usr/bin/env python3
"""Strict checker for the Volunteer Training Internet Beta Engineering RC."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from crowdtensor.training_contract import sha256_file, sha256_json


SCHEMA = "crowdtensor_volunteer_training_internet_beta_engineering_rc_v1"
PROBE_SCHEMA = "crowdtensor_volunteer_training_internet_beta_probe_v1"
CHECK_SCHEMA = "crowdtensor_volunteer_training_internet_beta_engineering_check_v1"
HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
MODEL_WEIGHT_HASH = "sha256:80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
DATASET_ID = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_HASHES = {
    "train": "sha256:e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7",
    "validation": "sha256:204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c",
}
REQUIRED_ARTIFACTS = {
    "probe",
    "campaign",
    "status",
    "audit_ledger",
    "campaign_source",
    "transport_security",
    "process_training",
    "fault_recovery",
    "checkpoint_lineage",
    "baseline",
    "communication",
    "independent_replay",
    "workflow",
    "cleanup",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _true(value: dict[str, Any], fields: tuple[str, ...], errors: list[str], prefix: str) -> None:
    for field in fields:
        if value.get(field) is not True:
            errors.append(f"{prefix}_{field}_missing")


def _false(value: dict[str, Any], fields: tuple[str, ...], errors: list[str], prefix: str) -> None:
    for field in fields:
        if value.get(field) is not False:
            errors.append(f"{prefix}_{field}_not_false")


def _safe_flags(value: dict[str, Any], errors: list[str], prefix: str) -> None:
    _false(
        value,
        (
            "credential_values_public",
            "private_paths_public",
            "raw_data_public",
            "tensor_values_public",
        ),
        errors,
        prefix,
    )
    if value.get("public_artifact_safe") is not True:
        errors.append(f"{prefix}_public_artifact_safe_missing")


def _scan_public_text(text: str, errors: list[str], prefix: str) -> None:
    for marker, code in (
        ("/root/", "absolute_private_path"),
        ("Bearer ", "bearer_material"),
        ('"lease_token"', "lease_token"),
        ('"invite_token"', "invite_token"),
        ('"coordinator_url"', "coordinator_url"),
        ('"input_ids"', "token_ids"),
        ('"tensor_specs"', "tensor_specs"),
        ('"tensor_values":', "tensor_values"),
        ('"gradient_values":', "gradient_values"),
        ("http://127.0.0.1", "loopback_url"),
    ):
        if marker in text:
            errors.append(f"{prefix}_public_{code}_present")


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def check(report_path: str | Path, *, require_ready: bool = False) -> dict[str, Any]:
    path = Path(report_path).expanduser().resolve()
    errors: list[str] = []
    try:
        report = _read(path)
    except Exception as exc:
        return {
            "schema": CHECK_SCHEMA,
            "ok": False,
            "volunteer_training_internet_beta_engineering_rc_ready": False,
            "goal_achieved": False,
            "error_count": 1,
            "errors": ["volunteer_beta_report_load_failed:" + type(exc).__name__],
        }
    if report.get("schema") != SCHEMA:
        errors.append("volunteer_beta_rc_schema_mismatch")
    expected_content_hash = sha256_json(
        {key: value for key, value in report.items() if key != "content_hash"}
    )
    if (
        not HASH.fullmatch(str(report.get("content_hash") or ""))
        or report.get("content_hash") != expected_content_hash
    ):
        errors.append("volunteer_beta_rc_content_hash_mismatch")
    _safe_flags(report, errors, "rc")
    if report.get("evidence_scope") != "local_independent_process_real_peft_engineering_rc":
        errors.append("volunteer_beta_evidence_scope_invalid")
    if not HASH.fullmatch(str(report.get("campaign_manifest_hash") or "")):
        errors.append("volunteer_beta_campaign_hash_missing")

    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    hashes = report.get("artifact_hashes") if isinstance(report.get("artifact_hashes"), dict) else {}
    if not REQUIRED_ARTIFACTS.issubset(artifacts):
        errors.append("volunteer_beta_required_artifacts_missing")
    artifact_paths: dict[str, Path] = {}
    report_root = path.parent.resolve()
    for name, relative in artifacts.items():
        artifact = (report_root / str(relative)).resolve()
        artifact_paths[str(name)] = artifact
        try:
            artifact.relative_to(report_root)
        except ValueError:
            errors.append(f"volunteer_beta_artifact_{name}_escapes_rc")
            continue
        if not artifact.is_file():
            errors.append(f"volunteer_beta_artifact_{name}_missing")
            continue
        expected_hash = str(hashes.get(name) or "")
        if not HASH.fullmatch(expected_hash) or sha256_file(artifact) != expected_hash:
            errors.append(f"volunteer_beta_artifact_{name}_hash_mismatch")
        _scan_public_text(
            artifact.read_text(encoding="utf-8"), errors, f"artifact_{name}"
        )
    _scan_public_text(path.read_text(encoding="utf-8"), errors, "rc")

    probe: dict[str, Any] = {}
    probe_path = artifact_paths.get("probe")
    if probe_path and probe_path.is_file():
        try:
            probe = _read(probe_path)
        except Exception as exc:
            errors.append("volunteer_beta_probe_load_failed:" + type(exc).__name__)
    if probe.get("schema") != PROBE_SCHEMA:
        errors.append("volunteer_beta_probe_schema_mismatch")
    if (
        probe.get("ok") is not True
        or probe.get("volunteer_training_internet_beta_engineering_verified") is not True
        or probe.get("public_artifact_scan_ok") is not True
    ):
        errors.append("volunteer_beta_probe_not_verified")
    if probe.get("campaign_manifest_hash") != report.get("campaign_manifest_hash"):
        errors.append("volunteer_beta_probe_campaign_hash_mismatch")
    for field in (
        "campaign_source",
        "round_progress",
        "real_training",
        "transport_security",
        "fault_recovery",
        "checkpoint_lineage",
        "centralized_baseline",
        "communication",
        "independent_replay",
        "contributor_workflow",
        "cleanup",
        "limitations",
    ):
        if probe.get(field) != report.get(field):
            errors.append(f"volunteer_beta_probe_{field}_summary_mismatch")

    section_artifacts = {
        "campaign_source": "campaign_source",
        "transport_security": "transport_security",
        "process_training": "real_training",
        "fault_recovery": "fault_recovery",
        "checkpoint_lineage": "checkpoint_lineage",
        "baseline": "centralized_baseline",
        "communication": "communication",
        "independent_replay": "independent_replay",
        "workflow": "contributor_workflow",
        "cleanup": "cleanup",
    }
    artifact_values: dict[str, dict[str, Any]] = {}
    for artifact_name, report_field in section_artifacts.items():
        artifact = artifact_paths.get(artifact_name)
        if artifact is None or not artifact.is_file():
            continue
        try:
            artifact_value = _read(artifact)
        except Exception as exc:
            errors.append(
                f"volunteer_beta_artifact_{artifact_name}_load_failed:"
                + type(exc).__name__
            )
            continue
        artifact_values[artifact_name] = artifact_value
        if artifact_value != report.get(report_field):
            errors.append(
                f"volunteer_beta_artifact_{artifact_name}_summary_mismatch"
            )

    campaign_artifact: dict[str, Any] = {}
    campaign_path = artifact_paths.get("campaign")
    if campaign_path and campaign_path.is_file():
        try:
            campaign_artifact = _read(campaign_path)
        except Exception as exc:
            errors.append("volunteer_beta_campaign_load_failed:" + type(exc).__name__)
    status_artifact: dict[str, Any] = {}
    status_path = artifact_paths.get("status")
    if status_path and status_path.is_file():
        try:
            status_artifact = _read(status_path)
        except Exception as exc:
            errors.append("volunteer_beta_status_load_failed:" + type(exc).__name__)

    source = report.get("campaign_source") if isinstance(report.get("campaign_source"), dict) else {}
    model = source.get("model_source") if isinstance(source.get("model_source"), dict) else {}
    dataset = source.get("dataset_source") if isinstance(source.get("dataset_source"), dict) else {}
    if (
        source.get("import_profile") != "smollm2_135m_wikitext2_lora_v1"
        or source.get("model_adapter_id") != "smollm2_lora_v1"
        or source.get("fixture_is_mock") is not False
        or source.get("real_public_weights_imported") is not True
        or source.get("immutable_public_dataset_imported") is not True
    ):
        errors.append("volunteer_beta_campaign_import_unverified")
    if (
        model.get("model_id") != MODEL_ID
        or model.get("revision") != MODEL_REVISION
        or model.get("license") != "apache-2.0"
        or model.get("source_verified") is not True
        or model.get("immutable_revision") is not True
    ):
        errors.append("volunteer_beta_model_source_invalid")
    model_files = model.get("imported_files") if isinstance(model.get("imported_files"), list) else []
    weight = next(
        (item for item in model_files if item.get("relative_name") == "model.safetensors"),
        {},
    )
    if (
        len(model_files) < 3
        or model.get("imported_file_count") != len(model_files)
        or model.get("imported_snapshot_hash") != sha256_json(model_files)
        or weight.get("sha256") != MODEL_WEIGHT_HASH
    ):
        errors.append("volunteer_beta_model_snapshot_invalid")
    if (
        dataset.get("dataset_id") != DATASET_ID
        or dataset.get("revision") != DATASET_REVISION
        or sorted(dataset.get("licenses") or []) != ["cc-by-sa-3.0", "gfdl"]
        or dataset.get("source_verified") is not True
        or dataset.get("immutable_revision") is not True
    ):
        errors.append("volunteer_beta_dataset_source_invalid")
    dataset_files = dataset.get("source_files") if isinstance(dataset.get("source_files"), list) else []
    observed_dataset = {
        str(item.get("split") or ""): item.get("sha256") for item in dataset_files
    }
    if (
        observed_dataset != DATASET_HASHES
        or dataset.get("source_snapshot_hash") != sha256_json(dataset_files)
    ):
        errors.append("volunteer_beta_dataset_snapshot_invalid")
    if (
        campaign_artifact.get("schema")
        != "crowdtensor_volunteer_training_campaign_v1"
        or campaign_artifact.get("manifest_hash")
        != report.get("campaign_manifest_hash")
        or campaign_artifact.get("model_adapter_id") != "smollm2_lora_v1"
        or campaign_artifact.get("model_source") != model
        or campaign_artifact.get("dataset_source") != dataset
        or campaign_artifact.get("campaign_import")
        != source.get("campaign_import")
        or int((campaign_artifact.get("round_policy") or {}).get("target_rounds") or 0)
        != 3
        or int((campaign_artifact.get("round_policy") or {}).get("minimum_quorum") or 0)
        != 2
        or (campaign_artifact.get("transport") or {}).get(
            "content_addressed_object_store"
        )
        is not True
        or (campaign_artifact.get("transport") or {}).get("resumable_chunk_upload")
        is not True
        or (campaign_artifact.get("transport") or {}).get(
            "s3_minio_presigned_download_contract"
        )
        is not True
        or campaign_artifact.get("physical_internet_multi_machine_verified")
        is not False
    ):
        errors.append("volunteer_beta_campaign_artifact_invalid")

    rounds = report.get("round_progress") if isinstance(report.get("round_progress"), dict) else {}
    if any(
        int(rounds.get(field, -1)) != expected
        for field, expected in {
            "target_rounds": 3,
            "completed_rounds": 3,
            "minimum_quorum": 2,
            "accepted_update_count": 6,
            "adapter_version_before": 0,
            "adapter_version_after": 3,
            "outer_step_after": 3,
        }.items()
    ):
        errors.append("volunteer_beta_round_progress_incomplete")
    status_rounds = (
        status_artifact.get("rounds")
        if isinstance(status_artifact.get("rounds"), list)
        else []
    )
    if (
        status_artifact.get("schema") != "crowdtensor_volunteer_training_status_v1"
        or status_artifact.get("campaign_manifest_hash")
        != report.get("campaign_manifest_hash")
        or status_artifact.get("campaign_complete") is not True
        or int(status_artifact.get("adapter_version") or 0) != 3
        or int(status_artifact.get("outer_step") or 0) != 3
        or int(status_artifact.get("completed_rounds") or 0) != 3
        or int(status_artifact.get("accepted_update_count") or 0) != 6
        or len(status_rounds) != 3
        or any(
            item.get("state") != "completed"
            or int(item.get("distinct_accepted_cell_count") or 0) != 2
            or int(item.get("accepted_result_count") or 0) != 2
            for item in status_rounds
        )
    ):
        errors.append("volunteer_beta_status_artifact_invalid")
    real = report.get("real_training") if isinstance(report.get("real_training"), dict) else {}
    _true(
        real,
        (
            "all_accepted_updates_originated_in_independent_cli_processes",
            "real_pytorch_autograd",
            "real_transformers_peft_lora",
            "base_weights_frozen",
        ),
        errors,
        "real_training",
    )
    if (
        int(real.get("accepted_update_count") or 0) != 6
        or int(real.get("real_training_process_count") or 0) != 6
        or int(real.get("distinct_real_training_process_count") or 0) != 6
        or int(real.get("optimizer_steps") or 0) != 6
        or int(real.get("tokens_seen") or 0) != 96
        or real.get("physical_internet_multi_machine_verified") is not False
    ):
        errors.append("volunteer_beta_independent_process_training_invalid")

    security = report.get("transport_security") if isinstance(report.get("transport_security"), dict) else {}
    _true(
        security,
        (
            "tls_termination_contract_verified",
            "direct_http_rejected",
            "untrusted_proxy_rejected",
            "trusted_forwarded_https_accepted",
            "resumable_chunk_upload",
            "content_addressed_upload_completion",
            "upload_state_survives_coordinator_restart",
        ),
        errors,
        "transport",
    )
    if security.get("actual_public_tls_handshake_verified") is not False:
        errors.append("volunteer_beta_public_tls_overclaim")

    faults = report.get("fault_recovery") if isinstance(report.get("fault_recovery"), dict) else {}
    offline = faults.get("cell_offline") if isinstance(faults.get("cell_offline"), dict) else {}
    network = faults.get("network_interruption") if isinstance(faults.get("network_interruption"), dict) else {}
    upload = faults.get("upload_interruption") if isinstance(faults.get("upload_interruption"), dict) else {}
    restart = faults.get("coordinator_restart") if isinstance(faults.get("coordinator_restart"), dict) else {}
    _true(
        offline,
        (
            "cell_disappeared_after_claim",
            "same_work_reassigned",
            "lease_generation_advanced",
            "replacement_completed",
        ),
        errors,
        "offline",
    )
    _true(
        network,
        (
            "interrupted_attempt_failed_publicly",
            "same_lease_generation_preserved",
            "recovery_completed",
        ),
        errors,
        "network",
    )
    _true(
        upload,
        (
            "active_upload_before_restart",
            "resume_completed",
            "pending_submission_recovery_used",
        ),
        errors,
        "upload",
    )
    if (
        upload.get("training_reexecuted_during_resume") is not False
        or int(upload.get("resumed_session_count") or 0) < 1
    ):
        errors.append("volunteer_beta_upload_resume_retrained_or_missing")
    if (
        restart.get("all_recoveries_verified") is not True
        or int(restart.get("restart_count") or 0) < 2
        or not _finite(restart.get("maximum_restart_seconds"))
    ):
        errors.append("volunteer_beta_coordinator_restart_unverified")

    lineage = report.get("checkpoint_lineage") if isinstance(report.get("checkpoint_lineage"), dict) else {}
    entries = lineage.get("entries") if isinstance(lineage.get("entries"), list) else []
    if (
        lineage.get("ok") is not True
        or int(lineage.get("adapter_version") or 0) != 3
        or int(lineage.get("completed_round_count") or 0) != 3
        or len(entries) != 3
        or any(item.get("lineage_link_verified") is not True for item in entries)
        or any(
            int(item.get("adapter_version_before", -1)) != index
            or int(item.get("adapter_version_after", -1)) != index + 1
            or int(item.get("distinct_cell_count") or 0) != 2
            for index, item in enumerate(entries)
        )
        or any(
            entries[index]["canonical_adapter_hash"]
            != entries[index + 1]["base_adapter_hash"]
            for index in range(2)
        )
    ):
        errors.append("volunteer_beta_checkpoint_lineage_invalid")

    baseline = report.get("centralized_baseline") if isinstance(report.get("centralized_baseline"), dict) else {}
    _true(
        baseline,
        (
            "same_optimizer_step_budget",
            "same_token_budget",
            "same_model_snapshot",
            "same_dataset_snapshot",
            "same_batch_sequence_contract",
            "all_losses_finite",
            "results_compared_not_quality_equated",
        ),
        errors,
        "baseline",
    )
    if (
        int(baseline.get("distributed_optimizer_steps") or 0)
        != int(baseline.get("centralized_optimizer_steps") or -1)
        or int(baseline.get("distributed_tokens_seen") or 0)
        != int(baseline.get("centralized_tokens_seen") or -1)
        or any(
            not _finite(baseline.get(field))
            for field in (
                "initial_validation_loss",
                "distributed_validation_loss",
                "centralized_validation_loss",
            )
        )
        or baseline.get("quality_superiority_claimed") is not False
    ):
        errors.append("volunteer_beta_centralized_baseline_invalid")

    replay = report.get("independent_replay") if isinstance(report.get("independent_replay"), dict) else {}
    if (
        replay.get("ok") is not True
        or replay.get("independent_process_replay_verified") is not True
        or replay.get("all_losses_finite") is not True
        or replay.get("distributed_checkpoint_hash_matches_lineage_head") is not True
        or replay.get("quality_equivalence_claimed") is not False
    ):
        errors.append("volunteer_beta_independent_replay_invalid")
    communication = report.get("communication") if isinstance(report.get("communication"), dict) else {}
    if (
        int(communication.get("accepted_delta_upload_count") or 0) != 6
        or int(communication.get("upload_session_count") or 0) != 6
        or int(communication.get("resumed_session_count") or 0) < 1
        or int(communication.get("resumable_completed_upload_bytes") or 0)
        != int(communication.get("accepted_delta_upload_bytes") or -1)
        or communication.get("shared_cache_download_savings_observed") is not True
        or communication.get("low_frequency_delta_only") is not True
        or communication.get("per_layer_activation_wan_transport_used") is not False
    ):
        errors.append("volunteer_beta_communication_metrics_invalid")
    workflow = report.get("contributor_workflow") if isinstance(report.get("contributor_workflow"), dict) else {}
    if (
        workflow.get("one_command_contribution_verified") is not True
        or workflow.get("contributor_command")
        != "crowdtensor volunteer join <private-invite> --once"
        or workflow.get("resumable_upload_default") is not True
    ):
        errors.append("volunteer_beta_contributor_workflow_invalid")
    cleanup = report.get("cleanup") if isinstance(report.get("cleanup"), dict) else {}
    _true(
        cleanup,
        (
            "http_service_stopped",
            "all_cell_subprocesses_reaped",
            "resumable_uploads_removed",
            "private_runtime_removed",
            "canonical_public_evidence_preserved",
            "cleanup_verified",
        ),
        errors,
        "cleanup",
    )
    if cleanup.get("live_resources_left_running") is not False:
        errors.append("volunteer_beta_live_resources_left_running")

    ledger_path = artifact_paths.get("audit_ledger")
    ledger_events: list[dict[str, Any]] = []
    ledger_errors: list[str] = []
    previous = "sha256:" + "0" * 64
    if ledger_path and ledger_path.is_file():
        for line_number, line in enumerate(
            ledger_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                ledger_errors.append(f"json:{line_number}")
                break
            if not isinstance(event, dict):
                ledger_errors.append(f"object:{line_number}")
                break
            expected_hash = sha256_json(
                {key: value for key, value in event.items() if key != "event_hash"}
            )
            if (
                int(event.get("sequence") or 0) != len(ledger_events) + 1
                or event.get("previous_event_hash") != previous
                or event.get("event_hash") != expected_hash
            ):
                ledger_errors.append(f"chain:{line_number}")
                break
            previous = expected_hash
            ledger_events.append(event)
    event_counts: dict[str, int] = {}
    for event in ledger_events:
        event_type = str(event.get("event_type") or "")
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    if (
        ledger_errors
        or len(ledger_events) != int(status_artifact.get("ledger_sequence") or -1)
        or previous != status_artifact.get("ledger_head_hash")
        or event_counts.get("campaign_created") != 1
        or event_counts.get("update_accepted") != 6
        or event_counts.get("round_aggregated") != 3
        or event_counts.get("coordinator_recovered") != 2
        or event_counts.get("lease_expired", 0) < 1
        or event_counts.get("campaign_target_reached") != 1
    ):
        errors.append("volunteer_beta_audit_ledger_invalid")
    limitations = report.get("limitations") if isinstance(report.get("limitations"), dict) else {}
    _false(
        limitations,
        (
            "physical_internet_multi_machine_verified",
            "independent_physical_host_test_performed",
            "permissionless_byzantine_safety",
            "sybil_resistance",
            "poisoning_resistance",
            "secure_aggregation",
            "general_availability",
            "service_level_agreement",
        ),
        errors,
        "limitation",
    )
    if limitations.get("local_independent_processes_verified") is not True:
        errors.append("volunteer_beta_local_process_scope_missing")
    if report_root.joinpath(".private").exists():
        errors.append("volunteer_beta_private_runtime_present_in_rc")

    claimed_ready = bool(
        report.get("volunteer_training_internet_beta_engineering_rc_ready") is True
        and report.get("goal_achieved") is True
    )
    if require_ready and not claimed_ready:
        errors.append("volunteer_beta_engineering_rc_not_ready")
    ready = claimed_ready and not errors
    return {
        "schema": CHECK_SCHEMA,
        "ok": not errors,
        "volunteer_training_internet_beta_engineering_rc_ready": ready,
        "goal_achieved": ready,
        "error_count": len(errors),
        "errors": sorted(set(errors)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = check(args.report, require_ready=bool(args.require_ready))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
