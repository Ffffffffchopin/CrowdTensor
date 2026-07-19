from __future__ import annotations

import copy
import json

import pytest

from crowdtensor.training_contract import sha256_file, sha256_json
from scripts.volunteer_training_internet_beta_check import check
from scripts.volunteer_training_internet_beta_pack import pack
from scripts.volunteer_training_internet_beta_probe import _compact_cell_payload


def _safe(value):
    return {
        **value,
        "credential_values_public": False,
        "private_paths_public": False,
        "raw_data_public": False,
        "tensor_values_public": False,
        "public_artifact_safe": True,
    }


def _probe_tree(root):
    model_files = [
        {
            "relative_name": "config.json",
            "sha256": "sha256:" + "1" * 64,
            "byte_count": 1,
        },
        {
            "relative_name": "model.safetensors",
            "sha256": "sha256:80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1",
            "byte_count": 269060552,
        },
        {
            "relative_name": "tokenizer.json",
            "sha256": "sha256:" + "2" * 64,
            "byte_count": 1,
        },
    ]
    dataset_files = [
        {
            "split": "train",
            "relative_name": "wikitext-2-raw-v1/train.parquet",
            "sha256": "sha256:e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7",
            "byte_count": 1,
        },
        {
            "split": "validation",
            "relative_name": "wikitext-2-raw-v1/validation.parquet",
            "sha256": "sha256:204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c",
            "byte_count": 1,
        },
    ]
    source = _safe(
        {
            "import_profile": "smollm2_135m_wikitext2_lora_v1",
            "model_adapter_id": "smollm2_lora_v1",
            "fixture_is_mock": False,
            "real_public_weights_imported": True,
            "immutable_public_dataset_imported": True,
            "model_source": {
                "model_id": "HuggingFaceTB/SmolLM2-135M",
                "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
                "license": "apache-2.0",
                "source_verified": True,
                "immutable_revision": True,
                "imported_files": model_files,
                "imported_file_count": len(model_files),
                "imported_snapshot_hash": sha256_json(model_files),
            },
            "dataset_source": {
                "dataset_id": "Salesforce/wikitext",
                "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
                "licenses": ["cc-by-sa-3.0", "gfdl"],
                "source_verified": True,
                "immutable_revision": True,
                "source_files": dataset_files,
                "source_snapshot_hash": sha256_json(dataset_files),
            },
        }
    )
    source["campaign_import"] = {
        "schema": "crowdtensor_volunteer_campaign_import_v1",
        "profile": "smollm2_135m_wikitext2_lora_v1",
        "model_adapter_id": "smollm2_lora_v1",
        "model_source": copy.deepcopy(source["model_source"]),
        "dataset_source": copy.deepcopy(source["dataset_source"]),
        "source_verified": True,
    }
    rounds = {
        "target_rounds": 3,
        "completed_rounds": 3,
        "minimum_quorum": 2,
        "accepted_update_count": 6,
        "adapter_version_before": 0,
        "adapter_version_after": 3,
        "outer_step_after": 3,
    }
    real = _safe(
        {
            "all_accepted_updates_originated_in_independent_cli_processes": True,
            "real_pytorch_autograd": True,
            "real_transformers_peft_lora": True,
            "base_weights_frozen": True,
            "accepted_update_count": 6,
            "real_training_process_count": 6,
            "distinct_real_training_process_count": 6,
            "optimizer_steps": 6,
            "tokens_seen": 96,
            "physical_internet_multi_machine_verified": False,
        }
    )
    security = _safe(
        {
            "tls_termination_contract_verified": True,
            "direct_http_rejected": True,
            "untrusted_proxy_rejected": True,
            "trusted_forwarded_https_accepted": True,
            "resumable_chunk_upload": True,
            "content_addressed_upload_completion": True,
            "upload_state_survives_coordinator_restart": True,
            "actual_public_tls_handshake_verified": False,
        }
    )
    faults = _safe(
        {
            "cell_offline": {
                "cell_disappeared_after_claim": True,
                "same_work_reassigned": True,
                "lease_generation_advanced": True,
                "replacement_completed": True,
            },
            "network_interruption": {
                "interrupted_attempt_failed_publicly": True,
                "same_lease_generation_preserved": True,
                "recovery_completed": True,
            },
            "upload_interruption": {
                "active_upload_before_restart": True,
                "resume_completed": True,
                "pending_submission_recovery_used": True,
                "training_reexecuted_during_resume": False,
                "resumed_session_count": 1,
            },
            "coordinator_restart": {
                "all_recoveries_verified": True,
                "restart_count": 2,
                "maximum_restart_seconds": 1.0,
            },
        }
    )
    entries = []
    hashes = ["sha256:" + character * 64 for character in "abcd"]
    for index in range(3):
        entries.append(
            {
                "adapter_version_before": index,
                "adapter_version_after": index + 1,
                "base_adapter_hash": hashes[index],
                "canonical_adapter_hash": hashes[index + 1],
                "distinct_cell_count": 2,
                "lineage_link_verified": True,
            }
        )
    lineage = _safe(
        {
            "ok": True,
            "adapter_version": 3,
            "completed_round_count": 3,
            "entries": entries,
        }
    )
    baseline = _safe(
        {
            "same_optimizer_step_budget": True,
            "same_token_budget": True,
            "same_model_snapshot": True,
            "same_dataset_snapshot": True,
            "same_batch_sequence_contract": True,
            "all_losses_finite": True,
            "results_compared_not_quality_equated": True,
            "distributed_optimizer_steps": 6,
            "centralized_optimizer_steps": 6,
            "distributed_tokens_seen": 96,
            "centralized_tokens_seen": 96,
            "initial_validation_loss": 4.0,
            "distributed_validation_loss": 3.9,
            "centralized_validation_loss": 3.8,
            "quality_superiority_claimed": False,
        }
    )
    replay = _safe(
        {
            "ok": True,
            "independent_process_replay_verified": True,
            "all_losses_finite": True,
            "distributed_checkpoint_hash_matches_lineage_head": True,
            "quality_equivalence_claimed": False,
        }
    )
    communication = _safe(
        {
            "accepted_delta_upload_count": 6,
            "upload_session_count": 6,
            "resumed_session_count": 1,
            "resumable_completed_upload_bytes": 600,
            "accepted_delta_upload_bytes": 600,
            "shared_cache_download_savings_observed": True,
            "low_frequency_delta_only": True,
            "per_layer_activation_wan_transport_used": False,
        }
    )
    workflow = _safe(
        {
            "one_command_contribution_verified": True,
            "contributor_command": "crowdtensor volunteer join <private-invite> --once",
            "resumable_upload_default": True,
        }
    )
    cleanup = _safe(
        {
            "http_service_stopped": True,
            "all_cell_subprocesses_reaped": True,
            "resumable_uploads_removed": True,
            "private_runtime_removed": True,
            "canonical_public_evidence_preserved": True,
            "cleanup_verified": True,
            "live_resources_left_running": False,
        }
    )
    limitations = {
        "physical_internet_multi_machine_verified": False,
        "independent_physical_host_test_performed": False,
        "local_independent_processes_verified": True,
        "permissionless_byzantine_safety": False,
        "sybil_resistance": False,
        "poisoning_resistance": False,
        "secure_aggregation": False,
        "general_availability": False,
        "service_level_agreement": False,
    }
    probe = _safe(
        {
            "schema": "crowdtensor_volunteer_training_internet_beta_probe_v1",
            "ok": True,
            "volunteer_training_internet_beta_engineering_verified": True,
            "campaign_id": "synthetic-beta",
            "campaign_manifest_hash": "sha256:" + "f" * 64,
            "campaign_source": source,
            "round_progress": rounds,
            "real_training": real,
            "transport_security": security,
            "fault_recovery": faults,
            "checkpoint_lineage": lineage,
            "centralized_baseline": baseline,
            "communication": communication,
            "independent_replay": replay,
            "contributor_workflow": workflow,
            "cleanup": cleanup,
            "public_artifact_scan_ok": True,
            "limitations": limitations,
        }
    )
    event_types = [
        "campaign_created",
        "coordinator_recovered",
        "coordinator_recovered",
        "lease_expired",
        *(["update_accepted"] * 6),
        *(["round_aggregated"] * 3),
        "campaign_target_reached",
    ]
    previous = "sha256:" + "0" * 64
    events = []
    for sequence, event_type in enumerate(event_types, start=1):
        event = {
            "schema": "crowdtensor_volunteer_training_ledger_event_v1",
            "sequence": sequence,
            "event_type": event_type,
            "previous_event_hash": previous,
        }
        event["event_hash"] = sha256_json(
            {key: value for key, value in event.items() if key != "event_hash"}
        )
        previous = event["event_hash"]
        events.append(event)
    campaign = {
        "schema": "crowdtensor_volunteer_training_campaign_v1",
        "campaign_id": "synthetic-beta",
        "manifest_hash": probe["campaign_manifest_hash"],
        "model_adapter_id": "smollm2_lora_v1",
        "model_source": copy.deepcopy(source["model_source"]),
        "dataset_source": copy.deepcopy(source["dataset_source"]),
        "campaign_import": copy.deepcopy(source["campaign_import"]),
        "round_policy": {"target_rounds": 3, "minimum_quorum": 2},
        "transport": {
            "content_addressed_object_store": True,
            "resumable_chunk_upload": True,
            "s3_minio_presigned_download_contract": True,
        },
        "physical_internet_multi_machine_verified": False,
    }
    status = {
        "schema": "crowdtensor_volunteer_training_status_v1",
        "campaign_id": "synthetic-beta",
        "campaign_manifest_hash": probe["campaign_manifest_hash"],
        "campaign_complete": True,
        "adapter_version": 3,
        "outer_step": 3,
        "completed_rounds": 3,
        "accepted_update_count": 6,
        "ledger_sequence": len(events),
        "ledger_head_hash": previous,
        "rounds": [
            {
                "state": "completed",
                "distinct_accepted_cell_count": 2,
                "accepted_result_count": 2,
            }
            for _ in range(3)
        ],
    }
    artifact_values = {
        "campaign": campaign,
        "status": status,
        "audit_ledger": "".join(
            json.dumps(event, sort_keys=True) + "\n" for event in events
        ),
        "campaign_source": source,
        "transport_security": security,
        "process_training": real,
        "fault_recovery": faults,
        "checkpoint_lineage": lineage,
        "baseline": baseline,
        "communication": communication,
        "independent_replay": replay,
        "workflow": workflow,
        "cleanup": cleanup,
    }
    probe["artifacts"] = {}
    for name, value in artifact_values.items():
        suffix = ".jsonl" if name == "audit_ledger" else ".json"
        path = root / f"{name}{suffix}"
        path.write_text(
            value
            if isinstance(value, str)
            else json.dumps(value, sort_keys=True) + "\n"
        )
        probe["artifacts"][name] = path.name
    probe_path = root / "probe.json"
    probe_path.write_text(json.dumps(probe, sort_keys=True) + "\n")
    rc_dir = root / "rc"
    pack(probe_path, rc_dir)
    return rc_dir / "volunteer_training_internet_beta_engineering_rc.json"


def _rewrite(path, report):
    report["content_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "content_hash"}
    )
    path.write_text(json.dumps(report, sort_keys=True) + "\n")


def test_compact_cell_payload_removes_tensor_and_lease_metadata() -> None:
    compact = _compact_cell_payload(
        {
            "ok": True,
            "last_report": {
                "work_completed": True,
                "optimizer_steps": 1,
                "tokens_seen": 16,
                "submission": {
                    "accepted": True,
                    "lease_token": "private",
                    "validation": {"tensor_specs": [{"name": "private"}]},
                },
            },
        }
    )
    serialized = json.dumps(compact)
    assert compact["last_report"]["submission"]["accepted"] is True
    assert "lease_token" not in serialized
    assert "tensor_specs" not in serialized


def test_strict_checker_accepts_scoped_engineering_rc(tmp_path) -> None:
    result = check(_probe_tree(tmp_path), require_ready=True)
    assert result["ok"] is True, result["errors"]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("model_revision", "volunteer_beta_model_source_invalid"),
        ("upload_retrained", "volunteer_beta_upload_resume_retrained_or_missing"),
        ("token_budget", "volunteer_beta_centralized_baseline_invalid"),
        ("multi_host_overclaim", "limitation_physical_internet_multi_machine_verified_not_false"),
        ("artifact_hash", "volunteer_beta_artifact_campaign_hash_mismatch"),
    ],
)
def test_checker_rejects_tampered_or_overclaimed_rc(
    tmp_path, mutation, expected
) -> None:
    path = _probe_tree(tmp_path)
    report = json.loads(path.read_text())
    if mutation == "artifact_hash":
        artifact = path.parent / report["artifacts"]["campaign"]
        artifact.write_text("tampered\n")
    else:
        if mutation == "model_revision":
            report["campaign_source"]["model_source"]["revision"] = "0" * 40
        elif mutation == "upload_retrained":
            report["fault_recovery"]["upload_interruption"][
                "training_reexecuted_during_resume"
            ] = True
        elif mutation == "token_budget":
            report["centralized_baseline"]["centralized_tokens_seen"] = 95
        elif mutation == "multi_host_overclaim":
            report["limitations"]["physical_internet_multi_machine_verified"] = True
        probe_path = path.parent / report["artifacts"]["probe"]
        probe = json.loads(probe_path.read_text())
        for field in (
            "campaign_source",
            "fault_recovery",
            "centralized_baseline",
            "limitations",
        ):
            probe[field] = copy.deepcopy(report[field])
        probe_path.write_text(json.dumps(probe, sort_keys=True) + "\n")
        report["artifact_hashes"]["probe"] = sha256_file(probe_path)
        _rewrite(path, report)
    result = check(path, require_ready=True)
    assert result["ok"] is False
    assert expected in result["errors"]
