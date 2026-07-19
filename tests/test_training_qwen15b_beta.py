from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from crowdtensor.qwen15b_training import MODEL_ID, MODEL_REVISION
from crowdtensor.qwen15b_training import QwenStageSpec, QwenStageTrainer
from crowdtensor.qwen15b_four_gpu_runtime import _stage_error_code
from crowdtensor.training_qwen15b_beta_service import TrainingBetaJobStore
from crowdtensor.training_qwen15b_job import reconcile_qwen15b_training_job_status
from crowdtensor.training_qwen15b_job import cleanup_qwen15b_training_job
from scripts.training_qwen15b_beta_check import (
    AUTHORITATIVE_ALPHA_HASH,
    PHASES,
    SCHEMA,
    check,
)
from scripts.training_qwen15b_beta_pack import AUTHORITATIVE_ALPHA, pack


HASH = "sha256:" + "a" * 64


def _worker(role: str) -> dict:
    stage_ids = [0, 1] if role == "kernel_a" else [2, 3]
    ready = [
        {
            "stage_id": stage,
            "cuda_live": True,
            "device": f"cuda:{index}",
            "pid": 100 + stage,
        }
        for index, stage in enumerate(stage_ids)
    ]
    recoveries = [
        {
            "stage_id": stage,
            "after_step": 4,
            "old_pid": 100 + stage,
            "new_pid": 200 + stage,
            "checkpoint_resume_verified": True,
            "resumed_global_step": 4,
            "resumed_dataset_cursor": 16,
            "loaded_checkpoint_hash": HASH,
        }
        for stage in stage_ids
    ]
    runs = {}
    for run_kind in ("baseline", "resumed"):
        runs[run_kind] = {
            "steps_completed": 8,
            "real_forward": True,
            "real_backward": True,
            "step_reports": [
                {
                    "step": step,
                    "stages": [
                        {
                            "stage_id": stage,
                            "global_step": step,
                            "optimizer_step_applied": True,
                            "lora_gradient_norm": 1.0,
                            "checkpoint_hash": HASH,
                        }
                        for stage in stage_ids
                    ],
                }
                for step in range(1, 9)
            ],
        }
    worker = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "parameter_count": 1_500_000_000,
        "base_weights_frozen": True,
        "positive_lora_gradient_norms": True,
        "coordinator_restart_owned_stages_verified": True,
        "coordinator_restart_stage_recoveries": recoveries,
        "transport_reliability": {
            "bounded_retry_enabled": True,
            "retry_attempt_limit": 12,
            "retry_count": 1,
            "reconnect_registration_count": 1,
        },
        "stage_ready": {"baseline": ready, "resumed": ready},
        "runs": runs,
        "evaluation": {},
        "export": {},
    }
    if role == "kernel_b":
        worker["evaluation"] = {
            "evaluation_verified": True,
            "standard_peft_cpu_load": True,
            "standard_peft_cuda_load": True,
            "adapter_changes_logits": True,
            "validation_loss_reduced": True,
        }
        worker["export"] = {
            "standard_peft_format": True,
            "layer_indexes": list(range(28)),
        }
    return {"role": role, "ok": True, "worker": worker}


def complete_report() -> dict:
    payloads = [
        {
            "run_kind": run_kind,
            "kind": kind,
            "step": step,
            "microbatch": microbatch,
            "producer_role": "kernel_a" if kind == "activation" else "kernel_b",
            "payload_hash": HASH,
            "byte_count": 1024,
            "tensor_count": 1,
        }
        for run_kind in ("baseline", "resumed")
        for kind in ("activation", "gradient")
        for step in range(8)
        for microbatch in range(4)
    ]
    payloads.append(
        {
            "run_kind": "resumed",
            "kind": "stage_adapter",
            "step": 8,
            "microbatch": -1,
            "producer_role": "kernel_a",
            "payload_hash": HASH,
            "byte_count": 2048,
            "tensor_count": 196,
        }
    )
    cleanup = {
        "kernels_deleted": True,
        "only_attempt_kernel_refs_targeted": True,
        "private_packages_removed": True,
        "coordinator_stopped": True,
        "tunnel_stopped": True,
        "private_runtime_removed": True,
        "rendezvous_private_payloads_removed": True,
        "checkpoint_archives_verified_before_cleanup": True,
    }
    checkpoint_bundles = [
        {
            "role": role,
            "verified": True,
            "preserved": True,
            "all_checkpoint_files_hash_verified": True,
            "all_manifest_content_hashes_verified": True,
            "checkpoint_manifest_count": 4,
            "file_hash": HASH,
        }
        for role in ("kernel_a", "kernel_b")
    ]
    live = {
        "schema": "crowdtensor_qwen15b_four_gpu_live_probe_v1",
        "ok": True,
        "started_at": "2026-07-12T12:30:00+00:00",
        "finished_at": "2026-07-12T12:40:00+00:00",
        "beta_mode": True,
        "live_run_performed": True,
        "mock_runtime_used": False,
        "cpu_fallback_used": False,
        "tiny_or_random_model_used": False,
        "training_qwen15b_beta_live_verified": True,
        "coordinator_restart_after_step": 4,
        "requested_model": MODEL_ID,
        "requested_model_revision": MODEL_REVISION,
        "requested_kernel_count": 2,
        "requested_accelerator": "NvidiaTeslaT4",
        "max_observed_running_kernel_count": 2,
        "same_authorized_account": True,
        "multi_account_gate_substitution": False,
        "worker_reports": [_worker("kernel_a"), _worker("kernel_b")],
        "rendezvous": {
            "payloads": payloads,
            "persistent_state_enabled": True,
            "recovered_from_persistent_state": True,
            "coordinator_restart_verified": True,
            "coordinator_restarts": [{"after_step": 4, "duration_seconds": 1.0}],
            "post_restart_registered_roles": ["kernel_a", "kernel_b"],
        },
        "evidence": {
            "beta_recovery_verified": True,
            "optimizer_steps_unique": True,
            "four_stage_compute_overlap_verified": True,
            "activation_payload_count": 64,
            "gradient_payload_count": 64,
            "resume_adapter_equivalence_verified": True,
            "resume_loss_equivalence_verified": True,
            "loss_reduced": True,
        },
        "checkpoint_bundles": checkpoint_bundles,
        "adapter_bundle": {
            "verified": True,
            "preserved": True,
            "standard_peft_layout": True,
            "safetensors_header_verified": True,
            "model_revision_verified": True,
            "file_hash": HASH,
        },
        "cleanup": cleanup,
        "blockers": [],
        "public_artifact_safe": True,
    }
    benchmark = {
        "schema": "crowdtensor_training_qwen15b_beta_benchmark_v1",
        "benchmark_complete": True,
        "completed_within_1800_seconds": True,
        "step_latency_count": 16,
        "step_latencies": [
            {"run_kind": run_kind, "step": step, "latency_ms": 10.0}
            for run_kind in ("baseline", "resumed")
            for step in range(1, 9)
        ],
        "private_network_payload_count": 129,
        "private_network_bytes": 131072,
        "peak_gpu_allocated_bytes": 1024,
        "coordinator_recovery_seconds": 1.0,
    }
    service_flags = {
        key: True
        for key in (
            "health_route_ready",
            "authentication_required",
            "submit_route_ready",
            "submit_idempotent",
            "status_route_ready",
            "resume_route_ready",
            "cancel_route_ready",
            "running_cancel_marker_ready",
            "export_route_ready",
            "cleanup_route_ready",
            "artifacts_route_ready",
            "events_route_ready",
            "persistent_process_restart_recovery_verified",
            "bounded_queue_ready",
            "one_live_gpu_job_enforced",
            "private_inputs_redacted",
        )
    }
    faults = {
        "duplicate_submission_rejected_or_idempotent": True,
        "expired_lease_recovery_verified": True,
        "corrupted_checkpoint_rejected": True,
        "non_finite_tensor_rejected": True,
        "worker_timeout_classified": True,
        "coordinator_unavailable_retry_verified": True,
    }
    artifacts = {
        name: {
            "present": True,
            "file_name": f"{name}.json",
            "file_hash": HASH,
            "byte_count": 100,
        }
        for name in (
            "authoritative_alpha",
            "source_report",
            "dataset_report",
            "service_smoke",
            "test_summary",
            "job_status",
            "user_export",
            "job_cleanup",
            "live_report",
            "benchmark",
            "allocation_ledger",
        )
    }
    report = {
        "schema": SCHEMA,
        "goal_achieved": True,
        "training_qwen15b_beta_ready": True,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "topology": "kaggle-2x-t4x2",
        "steps": 8,
        "authoritative_alpha": {
            "goal_achieved": True,
            "qwen15b_four_gpu_alpha_ready": True,
            "reused_without_rewrite": True,
            "artifact_hash": AUTHORITATIVE_ALPHA_HASH,
        },
        "source": {
            "ok": True,
            "source": {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "parameter_count": 1_500_000_000,
            },
            "ownership": {
                "stages": [
                    {"layer_start": start, "layer_end": end}
                    for start, end in ((0, 7), (7, 14), (14, 21), (21, 28))
                ],
                "all_source_tensors_covered": True,
                "four_distinct_kernel_device_placements": True,
            },
        },
        "dataset": {
            "ok": True,
            "private_payload_present": True,
            "manifest": {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "dataset_id": "Salesforce/wikitext",
                "train_sequence_count": 32,
                "validation_sequence_count": 4,
            },
            "raw_text_public": False,
            "token_ids_public": False,
        },
        "service_smoke": {
            "schema": "crowdtensor_training_qwen15b_beta_service_smoke_v1",
            "ok": True,
            **service_flags,
            "recovered_global_step": 4,
            "live_gpu_run_performed": False,
        },
        "job_store_summary": {
            "persistent_sqlite": True,
            "one_live_gpu_job": True,
            "max_queue_size": 8,
            "event_count": 20,
            "event_ids_unique": True,
            "global_step_monotonic": True,
        },
        "test_summary": {
            "ok": True,
            "passed": 320,
            "failed": 0,
            "existing_313_regressions_included": True,
            "beta_service_regressions_included": True,
            "fault_injection": faults,
        },
        "job_status": {
            "overall_state": "completed",
            "global_step": 8,
            "user_command_path_executed": True,
            "prebuilt_dist_inputs_used": False,
            "input_preparation": {
                "generated_by_user_command": True,
                "prebuilt_dist_inputs_used": False,
            },
            "phases": {phase: {"state": "completed"} for phase in PHASES},
            "credential_values_public": False,
            "credential_paths_public": False,
            "private_paths_public": False,
        },
        "user_export": {
            "schema": "crowdtensor_qwen15b_training_export_v1",
            "ok": True,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "standard_peft_layout": True,
            "adapter_model_hash": HASH,
            "adapter_config_hash": HASH,
            "private_paths_public": False,
            "public_artifact_safe": True,
        },
        "job_cleanup": {
            "schema": "crowdtensor_qwen15b_training_job_cleanup_v1",
            "ok": True,
            "temporary_kaggle_kernels_deleted": True,
            "only_recorded_job_kernel_refs_targeted": True,
            "live_resources_left_running": False,
            "temporary_private_runtime_removed": True,
            "checkpoint_and_evidence_preserved": True,
            "private_paths_public": False,
            "public_artifact_safe": True,
        },
        "live_report": live,
        "benchmark": benchmark,
        "allocation_summary": {
            "beta_goal_authorization": True,
            "attempt_limit": 3,
            "attempt_count": 1,
            "latest_outcome": "verified",
            "attempt_numbers_sequential": True,
            "all_attempts_completed": True,
            "same_authorized_account_per_attempt": True,
            "automatic_retry_loop": False,
        },
        "artifacts": artifacts,
        "blockers": [],
        "raw_training_text_public": False,
        "token_ids_public": False,
        "activation_values_public": False,
        "gradient_values_public": False,
        "adapter_tensor_values_public": False,
        "credential_values_public": False,
        "credential_paths_public": False,
        "coordinator_token_public": False,
        "coordinator_url_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    return report


def test_strict_checker_accepts_complete_beta_contract() -> None:
    result = check(complete_report(), require_ready=True)
    assert result["ok"] is True
    assert result["training_qwen15b_beta_ready"] is True
    assert result["goal_achieved"] is True


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("old_live", "training_beta_fresh_live_evidence_missing"),
        ("mock", "training_beta_fresh_live_evidence_missing"),
        ("cpu_fallback", "training_beta_fresh_live_evidence_missing"),
        ("one_gpu", "training_beta_same_account_two_t4x2_live_invalid"),
        ("retry", "training_beta_worker_retry_reregistration_missing"),
        ("recovery", "training_beta_all_stage_checkpoint_recovery_missing"),
        ("duplicate_step", "training_beta_duplicate_or_missing_optimizer_step"),
        ("payload", "training_beta_cross_kernel_payload_evidence_invalid"),
        ("peft", "training_beta_peft_export_evaluation_invalid"),
        ("archive", "training_beta_archive_integrity_invalid"),
        ("cleanup", "training_beta_cleanup_incomplete"),
        ("user_export", "training_beta_ordinary_user_export_missing"),
        ("user_cleanup", "training_beta_ordinary_user_cleanup_missing"),
        ("store", "training_beta_persistent_job_store_invalid"),
        ("fault", "training_beta_required_regressions_or_fault_matrix_missing"),
        ("alpha", "training_beta_authoritative_alpha_invalid"),
        ("secret", "training_beta_public_root_path"),
    ],
)
def test_strict_checker_rejects_false_beta_evidence(mutation: str, expected: str) -> None:
    report = complete_report()
    if mutation == "old_live":
        report["live_report"]["started_at"] = "2026-07-12T09:00:00+00:00"
    elif mutation == "mock":
        report["live_report"]["mock_runtime_used"] = True
    elif mutation == "cpu_fallback":
        report["live_report"]["cpu_fallback_used"] = True
    elif mutation == "one_gpu":
        report["live_report"]["max_observed_running_kernel_count"] = 1
    elif mutation == "retry":
        report["live_report"]["worker_reports"][0]["worker"]["transport_reliability"]["retry_count"] = 0
    elif mutation == "recovery":
        report["live_report"]["worker_reports"][0]["worker"]["coordinator_restart_stage_recoveries"].pop()
    elif mutation == "duplicate_step":
        run = report["live_report"]["worker_reports"][0]["worker"]["runs"]["baseline"]
        run["step_reports"][-1]["step"] = 7
        for stage in run["step_reports"][-1]["stages"]:
            stage["global_step"] = 7
    elif mutation == "payload":
        report["live_report"]["rendezvous"]["payloads"].pop(0)
    elif mutation == "peft":
        report["live_report"]["worker_reports"][1]["worker"]["evaluation"]["standard_peft_cpu_load"] = False
    elif mutation == "archive":
        report["live_report"]["checkpoint_bundles"][0]["all_checkpoint_files_hash_verified"] = False
    elif mutation == "cleanup":
        report["live_report"]["cleanup"]["kernels_deleted"] = False
    elif mutation == "user_export":
        report["user_export"]["standard_peft_layout"] = False
    elif mutation == "user_cleanup":
        report["job_cleanup"]["live_resources_left_running"] = True
    elif mutation == "store":
        report["job_store_summary"]["persistent_sqlite"] = False
    elif mutation == "fault":
        report["test_summary"]["fault_injection"]["non_finite_tensor_rejected"] = False
    elif mutation == "alpha":
        report["authoritative_alpha"]["artifact_hash"] = HASH
    elif mutation == "secret":
        report["leaked_path"] = "/root/private-token"
    result = check(report, require_ready=True)
    assert result["ok"] is False
    assert expected in result["errors"]
    assert result["training_qwen15b_beta_ready"] is False


def test_default_checker_accepts_well_formed_blocker_without_claiming_ready() -> None:
    report = complete_report()
    report["live_report"] = {}
    report["benchmark"] = {}
    for name in ("live_report", "benchmark"):
        report["artifacts"][name] = {
            "present": False,
            "file_name": "",
            "file_hash": "",
            "byte_count": 0,
        }
    report["goal_achieved"] = False
    report["training_qwen15b_beta_ready"] = False
    result = check(report)
    assert result["ok"] is True
    assert result["training_qwen15b_beta_ready"] is False
    assert "training_beta_fresh_live_evidence_missing" in result["readiness_errors"]
    assert check(report, require_ready=True)["ok"] is False


def test_pack_collects_job_store_and_writes_strict_ready_artifact(tmp_path: Path) -> None:
    complete = complete_report()
    job = tmp_path / "beta-job"
    source_path = job / "inputs/source/training_qwen15b_source_probe.json"
    dataset_path = job / ".private-inputs/dataset/training_qwen15b_dataset_prepare.json"
    live_dir = job / "attempts/qwen15b-beta-1"
    live_path = live_dir / "training_qwen15b_four_gpu_live_probe.json"
    benchmark_path = live_dir / "training_qwen15b_beta_benchmark.json"
    status_path = job / "training_qwen15b_status.json"
    export_path = job / "training_qwen15b_export.json"
    cleanup_path = job / "training_qwen15b_cleanup.json"
    ledger_path = job / "allocation_attempts.json"
    for path, value in (
        (source_path, complete["source"]),
        (dataset_path, complete["dataset"]),
        (live_path, complete["live_report"]),
        (benchmark_path, complete["benchmark"]),
        (status_path, complete["job_status"]),
        (export_path, complete["user_export"]),
        (cleanup_path, complete["job_cleanup"]),
        (
            ledger_path,
            {
                "beta_goal_allocation_authorization": {
                    "authorized": True,
                    "same_authorized_account_only": True,
                    "topology": "kaggle-2x-t4x2",
                    "goal_attempt_limit": 3,
                    "automatic_retry_loop": False,
                },
                "qwen15b_four_gpu_attempts": [
                    {"attempt": 1, "completed": True, "outcome": "verified"}
                ],
            },
        ),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    store = TrainingBetaJobStore(job / ".private-service/training_beta_jobs.sqlite3")
    submitted, _ = store.submit(
        {
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "topology": "kaggle-2x-t4x2",
            "steps": 8,
            "job_dir": str(job),
        },
        idempotency_key="pack-test",
    )
    status = complete["job_status"]
    for index, phase in enumerate(sorted(PHASES), 1):
        store.update_status(
            submitted["job_id"],
            {**status, "current_phase": phase},
            event_id=f"phase-{index}",
        )
    service = tmp_path / "service.json"
    tests = tmp_path / "tests.json"
    service.write_text(json.dumps(complete["service_smoke"]), encoding="utf-8")
    tests.write_text(json.dumps(complete["test_summary"]), encoding="utf-8")
    report = pack(
        tmp_path / "packed",
        job_dir=job,
        test_summary=tests,
        service_smoke=service,
        authoritative_alpha=AUTHORITATIVE_ALPHA,
    )
    assert report["training_qwen15b_beta_ready"] is True
    assert report["strict_checker"]["ok"] is True
    encoded = json.dumps(report, sort_keys=True)
    assert str(tmp_path) not in encoded
    assert (tmp_path / "packed/training_qwen15b_beta.json").is_file()


def test_qwen_worker_fault_codes_are_stable_and_public() -> None:
    assert _stage_error_code(TimeoutError("private worker details")) == "worker_timeout"
    assert (
        _stage_error_code(RuntimeError("qwen15b_non_finite_stage_activation"))
        == "non_finite_stage_activation"
    )
    assert "private worker details" not in _stage_error_code(
        TimeoutError("private worker details")
    )


def test_qwen_stage_fails_closed_on_non_finite_activation(tmp_path: Path) -> None:
    import torch

    class NonFiniteStage(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_parameter = torch.nn.Parameter(torch.ones(1))

        def forward(self, value):
            return value.float().unsqueeze(-1) * self.lora_parameter * torch.tensor(float("nan"))

    trainer = QwenStageTrainer(
        NonFiniteStage(),
        QwenStageSpec(0, "A", 0, 0, 1, owns_embedding=True),
        device="cpu",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    trainer.begin_step()
    with pytest.raises(RuntimeError, match="qwen15b_non_finite_stage_activation"):
        trainer.forward(0, torch.tensor([[1, 2, 3]]))


def test_completed_job_reconciles_stale_allocation_phase_without_new_attempt(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    status = {
        "schema": "crowdtensor_training_qwen15b_beta_job_runtime_status_v1",
        "overall_state": "completed",
        "current_phase": "cleanup",
        "global_step": 8,
        "attempts_used": 1,
        "attempt_limit": 3,
        "latest_attempt": {"attempt": 1, "ok": True},
        "phases": {
            **{phase: {"state": "completed"} for phase in PHASES},
            "allocation": {"state": "running"},
        },
        "events": [],
        "retry_count": 0,
    }
    (job / "training_qwen15b_status.json").write_text(
        json.dumps(status), encoding="utf-8"
    )
    reconciled = reconcile_qwen15b_training_job_status(job)
    assert reconciled["phases"]["allocation"]["state"] == "completed"
    assert reconciled["phases"]["allocation"]["reconciled_from_verified_live_attempt"] is True
    assert reconciled["current_phase"] == "cleanup"
    assert reconciled["attempts_used"] == 1


def test_standalone_cleanup_persists_cleaned_status_idempotently(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    status = {
        "schema": "crowdtensor_training_qwen15b_beta_job_runtime_status_v1",
        "overall_state": "completed",
        "current_phase": "cleanup",
        "global_step": 8,
        "retry_count": 0,
        "phases": {phase: {"state": "completed"} for phase in PHASES},
        "events": [],
    }
    (job / "training_qwen15b_status.json").write_text(
        json.dumps(status), encoding="utf-8"
    )
    first = cleanup_qwen15b_training_job(job)
    persisted = json.loads(
        (job / "training_qwen15b_status.json").read_text(encoding="utf-8")
    )
    assert first["ok"] is True
    assert persisted["overall_state"] == "cleaned"
    assert persisted["cleanup"]["ok"] is True
    revision_events = len(persisted["events"])
    second = cleanup_qwen15b_training_job(job)
    persisted_again = json.loads(
        (job / "training_qwen15b_status.json").read_text(encoding="utf-8")
    )
    assert second == first
    assert len(persisted_again["events"]) == revision_events
