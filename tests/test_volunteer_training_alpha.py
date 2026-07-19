from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from crowdtensor.training_contract import sha256_file, sha256_json
from scripts.volunteer_training_alpha_check import SCHEMA, check


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _safe(value: dict) -> dict:
    return {
        **value,
        "credential_values_public": False,
        "private_paths_public": False,
        "raw_data_public": False,
        "tensor_values_public": False,
        "public_artifact_safe": True,
    }


def _tree(root: Path):
    stale = _safe(
        {
            "rejected": True,
            "expected_code_observed": True,
            "code": "volunteer_stale_adapter_version_rejected",
        }
    )
    report = _safe(
        {
            "schema": SCHEMA,
            "volunteer_training_protocol_alpha_ready": True,
            "goal_achieved": True,
            "evidence_scope": "local_http_real_peft_protocol_alpha",
            "campaign_manifest_hash": "sha256:" + "a" * 64,
            "protocol_version": "volunteer_training_v1.0",
            "real_training": {
                "pytorch_autograd": True,
                "transformers_peft_lora": True,
                "base_weights_frozen": True,
                "mock_only": False,
                "cell_update_count": 4,
                "optimizer_steps": 8,
            },
            "round_progress": {
                "adapter_version_before": 0,
                "adapter_version_after": 2,
                "outer_step_after": 2,
                "completed_rounds": 2,
                "accepted_update_count": 4,
                "minimum_quorum": 2,
                "all_rounds_distinct_cell_quorum": True,
                "atomic_version_advance": True,
            },
            "churn_proof": {
                "cell_disappeared_before_submit": True,
                "same_work_reassigned": True,
                "generation_advanced": True,
                "replacement_used_canonical_adapter": True,
                "duplicate_retry_idempotent": True,
                "duplicate_retry_accepted": True,
                "expired_lease_count": 1,
                "late_stale_delta_rejection": stale,
            },
            "update_validation": {
                "tensor_contract_validation": True,
                "content_hash_validation": True,
                "finite_value_validation": True,
                "norm_clipping_policy": True,
                "hard_norm_rejection_policy": True,
                "distinct_cell_quorum": True,
                "forked_base_hash_rejection": {
                    "code": "base_adapter_hash_mismatch",
                    "rejected": True,
                },
                "non_finite_delta_rejection": {
                    "code": "adapter_delta_non_finite",
                    "rejected": True,
                },
            },
            "centralized_baseline": {
                "real_pytorch_autograd": True,
                "real_transformers_peft_lora": True,
                "same_optimizer_step_budget": True,
                "same_token_budget": True,
                "same_dataset_snapshot": True,
                "same_batch_sequence_contract": True,
                "results_compared_not_quality_equated": True,
                "distributed_optimizer_steps": 8,
                "centralized_optimizer_steps": 8,
                "distributed_tokens_seen": 256,
                "centralized_tokens_seen": 256,
                "initial_validation_loss": 4.0,
                "distributed_validation_loss": 3.8,
                "centralized_validation_loss": 3.7,
                "distributed_loss_progress": 0.2,
                "centralized_loss_progress": 0.3,
                "useful_model_quality_claimed": False,
                "broad_scalability_claimed": False,
            },
            "communication": {
                "low_frequency_delta_transport_verified": True,
                "per_layer_activation_wan_transport_used": False,
                "local_steps_per_delta": 2,
                "measured_to_stepwise_upload_ratio": 0.5,
            },
            "contributor_workflow": {
                "one_command_join_verified": True,
                "hardware_detection": True,
                "resource_limits": True,
                "content_addressed_cache": True,
                "pause_resume_commands": True,
                "lease_heartbeat": True,
                "private_invite_required": True,
                "command": "crowdtensor volunteer join <private-invite> --once",
                "command_exit_code": 0,
            },
            "http_service": {
                "health_route_verified": True,
                "claim_route_verified": True,
                "authenticated_artifact_download_verified": True,
                "binary_safetensors_submission_verified": True,
                "heartbeat_route_enabled": True,
                "loopback_http_service_stopped": True,
                "physical_internet_route_verified": False,
            },
            "audit_ledger": {"ok": True, "event_count": 18, "errors": []},
            "cleanup": {
                "http_service_stopped": True,
                "all_cell_processes_stopped": True,
                "private_runtime_removed": True,
                "canonical_public_evidence_preserved": True,
                "cleanup_verified": True,
                "live_resources_left_running": False,
            },
            "limitations": {
                "physical_internet_multi_machine_verified": False,
                "loopback_http_protocol_verified": True,
                "permissionless_byzantine_safety": False,
                "sybil_resistance": False,
                "secure_aggregation": False,
                "useful_model_quality_claimed": False,
                "broad_scalability_claimed": False,
                "general_availability": False,
                "service_level_agreement": False,
            },
        }
    )
    previous = "sha256:" + "0" * 64
    events = []
    for sequence in range(1, 19):
        event = {
            "schema": "crowdtensor_volunteer_training_ledger_event_v1",
            "sequence": sequence,
            "event_type": "test_event",
            "previous_event_hash": previous,
        }
        event["event_hash"] = sha256_json(
            {key: value for key, value in event.items() if key != "event_hash"}
        )
        previous = event["event_hash"]
        events.append(event)
    report["audit_ledger"]["head_hash"] = previous
    probe = _safe(
        {
            "schema": "crowdtensor_volunteer_training_alpha_probe_v1",
            "ok": True,
            "volunteer_training_protocol_alpha_verified": True,
            "campaign_manifest_hash": report["campaign_manifest_hash"],
            "public_artifact_scan_ok": True,
            **{
                field: copy.deepcopy(report[field])
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
                )
            },
        }
    )
    artifact_names = {
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
    artifact_values = {
        "probe": probe,
        "campaign": {
            "schema": "crowdtensor_volunteer_training_campaign_v1",
            "protocol_version": "volunteer_training_v1.0",
            "manifest_hash": report["campaign_manifest_hash"],
            "round_policy": {"minimum_quorum": 2, "target_rounds": 2},
        },
        "status": {
            "schema": "crowdtensor_volunteer_training_status_v1",
            "campaign_complete": True,
            "adapter_version": 2,
            "outer_step": 2,
            "completed_rounds": 2,
            "accepted_update_count": 4,
            "ledger_sequence": 18,
            "ledger_head_hash": previous,
            "rounds": [
                {"state": "completed", "distinct_accepted_cell_count": 2},
                {"state": "completed", "distinct_accepted_cell_count": 2},
            ],
        },
        "audit_ledger": "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        "baseline": report["centralized_baseline"],
        "churn": report["churn_proof"],
        "update_validation": report["update_validation"],
        "communication": report["communication"],
        "cleanup": report["cleanup"],
        "ledger_check": report["audit_ledger"],
        "http_service": report["http_service"],
        "join_workflow": report["contributor_workflow"],
    }
    artifacts = {}
    hashes = {}
    for name in artifact_names:
        suffix = ".jsonl" if name == "audit_ledger" else ".json"
        artifact = root / f"{name}{suffix}"
        _write(artifact, artifact_values[name])
        artifacts[name] = artifact.name
        hashes[name] = sha256_file(artifact)
    report["artifacts"] = artifacts
    report["artifact_hashes"] = hashes
    report["content_hash"] = sha256_json(report)
    report_path = root / "volunteer_training_alpha_rc.json"
    _write(report_path, report)
    return report_path, report


def test_strict_checker_accepts_complete_scoped_alpha(tmp_path) -> None:
    path, _report = _tree(tmp_path)
    result = check(path, require_ready=True)
    assert result["ok"] is True
    assert result["error_count"] == 0


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("mock", "volunteer_alpha_mock_only_evidence_rejected"),
        ("no_stale", "volunteer_alpha_late_stale_fence_missing"),
        ("budget", "volunteer_alpha_baseline_compute_budget_mismatch"),
        ("internet_overclaim", "volunteer_alpha_limitation_physical_internet_multi_machine_verified_not_false"),
        ("artifact_hash", "volunteer_alpha_artifact_campaign_hash_mismatch"),
    ],
)
def test_checker_rejects_false_or_tampered_evidence(tmp_path, mutation, expected) -> None:
    path, report = _tree(tmp_path)
    if mutation == "mock":
        report["real_training"]["mock_only"] = True
    elif mutation == "no_stale":
        report["churn_proof"]["late_stale_delta_rejection"]["rejected"] = False
    elif mutation == "budget":
        report["centralized_baseline"]["centralized_optimizer_steps"] = 7
    elif mutation == "internet_overclaim":
        report["limitations"]["physical_internet_multi_machine_verified"] = True
    elif mutation == "artifact_hash":
        (tmp_path / "campaign.json").write_text("tampered\n", encoding="utf-8")
    _write(path, report)
    result = check(path, require_ready=True)
    assert result["ok"] is False
    assert expected in result["errors"]
