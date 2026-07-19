from __future__ import annotations

import json
import hashlib
from copy import deepcopy

import pytest

from scripts.training_heterogeneous_tpu_beta_check import LIVE_SCHEMA, _stable_hash, check
from scripts.training_heterogeneous_tpu_beta_pack import pack


HASH = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def passing_live_report() -> dict:
    initial = [
        {
            "stage_id": stage_id,
            "device_type": device_type,
            "job_id_hash": HASH,
            "run_id_hash": HASH_B,
            "resource_fit_verified": True,
        }
        for stage_id, device_type in enumerate(
            ["cuda", "cuda", "jax_tpu", "cuda", "cpu"]
        )
    ]
    replacement = deepcopy(initial)
    return {
        "schema": LIVE_SCHEMA,
        "live_run_performed": True,
        "execution_provider": "kaggle",
        "model_id": "Qwen/Qwen2.5-7B",
        "model_revision": "d149729398750b98c0af14eb82c78cfe92750796",
        "training_manifest_schema": "crowdtensor_heterogeneous_training_manifest_v2",
        "training_manifest_hash": HASH,
        "parameter_count": 7_615_616_000,
        "stage_boundaries": [[0, 7], [7, 14], [14, 20], [20, 26], [26, 28]],
        "sequence_length": 8,
        "microbatch_size": 1,
        "target_steps": 6,
        "same_job_training_verified": True,
        "job_id_hash": HASH,
        "run_id_hash": HASH_B,
        "provider_coverage": ["kaggle_cpu", "kaggle_cuda", "kaggle_jax_tpu"],
        "placement_evidence": {
            "initial_assignments": initial,
            "replacement_assignments": replacement,
            "initial_generation": 1,
            "replacement_generation": 3,
            "hbm_reserve_enforced": True,
            "tpu_compile_cost_considered": True,
            "tpu_steady_state_cost_considered": True,
            "network_and_load_cost_considered": True,
        },
        "training_evidence": {
            "committed_steps": [1, 2, 3, 4, 5, 6],
            "committed_steps_contiguous": True,
            "duplicate_committed_steps": [],
            "missing_committed_steps": [],
            "optimizer_commit_count": 6,
            "atomic_global_commit_verified": True,
            "updated_stage_ids": [0, 1, 2, 3, 4],
            "finite_loss_count": 6,
            "non_finite_loss_count": 0,
            "positive_gradient_stage_ids": [0, 1, 2, 3, 4],
            "changed_lora_stage_ids": [0, 1, 2, 3, 4],
            "all_optimizer_steps_real": True,
            "random_or_synthetic_weights_used": False,
            "fake_gradients_used": False,
        },
        "tpu_training_evidence": {
            "execution_provider": "kaggle",
            "runtime_backend": "jax_tpu",
            "accelerator_type": "TPU v5e",
            "stage_id": 2,
            "layer_start": 14,
            "layer_end": 20,
            "jax_tpu_device_count": 8,
            "jax_mesh_shape": [8],
            "all_mesh_devices_used": True,
            "parameter_sharding": "named_mesh_model_axis",
            "stage_selective_real_weights": True,
            "full_model_loaded": False,
            "compute_dtype": "bfloat16",
            "forward_executed": True,
            "backward_executed": True,
            "optimizer_executed": True,
            "committed_steps": [1, 2, 3, 4, 5, 6],
            "positive_lora_gradient_min": 0.01,
            "adapter_hash_before": HASH,
            "adapter_hash_after": HASH_B,
            "compile_latency_ms": 2300.0,
            "steady_profile_sample_count": 5,
        },
        "tensor_transport_evidence": {
            "format": "safetensors",
            "pickle_deserialization_allowed": False,
            "jax_array_conversion_verified": True,
            "forward_activation_count": 24,
            "backward_gradient_count": 24,
            "cuda_to_tpu_activation_count": 6,
            "tpu_to_cuda_activation_count": 6,
            "cuda_to_tpu_gradient_count": 6,
            "tpu_to_cuda_gradient_count": 6,
            "cuda_to_cpu_activation_count": 6,
            "cpu_to_cuda_gradient_count": 6,
            "all_checksums_verified": True,
            "chunking_verified": True,
            "finite_retry_verified": True,
            "idempotent_delivery_verified": True,
            "stale_generation_rejected": True,
        },
        "tpu_recovery_evidence": {
            "tpu_removed_after_committed_step": 3,
            "same_tpu_kernel_runtime_retained": True,
            "old_tpu_miner_id_hash": HASH,
            "replacement_tpu_miner_id_hash": HASH_B,
            "pause_or_incomplete_placement_observed": True,
            "step3_tpu_checkpoint_restored": True,
            "restored_global_step": 3,
            "replacement_committed_steps": [4, 5, 6],
            "old_generation_result_rejected": True,
            "rebalance_verified": True,
        },
        "checkpoint_evidence": {
            "all_five_stage_archives_valid": True,
            "atomic_checkpoint_barrier_verified": True,
            "stage_ids": [0, 1, 2, 3, 4],
            "pytorch_components_complete": True,
            "tpu_runtime_backend": "jax_tpu",
            "tpu_optimizer_state_present": True,
            "tpu_scheduler_state_present": True,
            "tpu_jax_prng_state_present": True,
            "tpu_grad_scaler_applicable": False,
            "tpu_pickle_deserialization_allowed": False,
            "all_component_hashes_verified": True,
        },
        "export_evidence": {
            "standard_peft_format": True,
            "adapter_tensor_count": 392,
            "layer_indexes": list(range(28)),
            "cpu_reload_verified": True,
            "finite_full_stagewise_forward_verified": True,
            "model_binding_verified": True,
            "adapter_file_hash": HASH,
        },
        "regression_summary": {
            "passed": 160,
            "failed": 0,
            "legacy_cpu_cuda_tests_included": True,
            "jax_tpu_tests_included": True,
            "public_safety_tests_included": True,
        },
        "cleanup": {
            "all_remote_kernels_deleted": True,
            "temporary_private_packages_removed": True,
            "coordinator_stopped": True,
            "tunnel_stopped": True,
            "tensor_payloads_removed": True,
            "temporary_credentials_removed": True,
            "live_resources_left_running": False,
        },
        "blockers": [],
        "credential_values_public": False,
        "credential_paths_public": False,
        "coordinator_url_public": False,
        "raw_training_text_public": False,
        "token_ids_public": False,
        "activation_values_public": False,
        "gradient_values_public": False,
        "checkpoint_tensor_values_public": False,
        "adapter_tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def write(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def file_hash(path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_strict_tpu_beta_pack_and_checker_accept_complete_live_evidence(tmp_path) -> None:
    live = tmp_path / "live.json"
    write(live, passing_live_report())

    packed = pack(live, tmp_path / "packed")
    checked = check(
        tmp_path / "packed" / "training_heterogeneous_tpu_beta.json",
        require_ready=True,
    )

    assert packed["heterogeneous_training_tpu_beta_ready"] is True
    assert all(packed["acceptance_gates"].values())
    assert checked["ok"] is True
    assert checked["heterogeneous_training_tpu_beta_ready"] is True
    assert checked["error_count"] == 0


def tpu_compile_kernel_report(manifest_hash: str = HASH) -> dict:
    def worker(role: str, miner_id_hash: str, compile_latency_ms: float) -> dict:
        return {
            "label": role,
            "report": {
                "ok": True,
                "deployment_role": role,
                "miner_id_hash": miner_id_hash,
                "device_policy": "jax_tpu",
                "assigned_stage_ids": [2],
                "training_manifest_hash": manifest_hash,
                "stage_process_statuses": [
                    {
                        "runtime_backend": "jax_tpu",
                        "stage_id": 2,
                        "compile_latency_ms": compile_latency_ms,
                        "jax_mesh_device_count": 8,
                        "jax_mesh_shape": [8],
                        "all_mesh_devices_used": True,
                        "forward_output_sharding_explicit": True,
                        "backward_output_sharding_explicit": True,
                        "boundary_output_replicated": True,
                        "tensor_values_public": False,
                        "public_artifact_safe": True,
                    }
                ],
            },
        }

    return {
        "schema": "crowdtensor_heterogeneous_training_tpu_beta_kaggle_kernel_v1",
        "ok": True,
        "kernel_role": "tpu",
        "worker_results": [
            worker("tpu_old", HASH, 39000.0),
            worker("tpu_replacement", HASH_B, 36000.0),
        ],
        "credential_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def test_pack_recovers_omitted_compile_latency_from_bound_tpu_worker_status(
    tmp_path,
) -> None:
    live = tmp_path / "live.json"
    source = passing_live_report()
    source["tpu_training_evidence"]["compile_latency_ms"] = 0.0
    write(live, source)
    kernel = tmp_path / "private-tpu-kernel-report.json"
    write(kernel, tpu_compile_kernel_report(source["training_manifest_hash"]))

    packed = pack(
        live,
        tmp_path / "packed",
        tpu_kernel_report=kernel,
    )
    checked = check(
        tmp_path / "packed" / "training_heterogeneous_tpu_beta.json",
        require_ready=True,
    )
    encoded = json.dumps(packed, sort_keys=True)

    assert packed["tpu_training_evidence"]["compile_latency_ms"] == 39000.0
    assert packed["tpu_compile_latency_evidence"]["worker_status_count"] == 2
    assert packed["tpu_compile_latency_evidence"]["measurement_recomputed"] is False
    assert packed["source_evidence"]["omitted_live_compile_measurement_recovered"] is True
    assert str(kernel) not in encoded
    assert checked["ok"] is True
    assert checked["heterogeneous_training_tpu_beta_ready"] is True

    malformed = deepcopy(packed)
    malformed["tpu_compile_latency_evidence"]["worker_statuses"][0][
        "compile_latency_ms"
    ] = 1.0
    malformed["content_hash"] = _stable_hash(
        {key: value for key, value in malformed.items() if key != "content_hash"}
    )
    write(tmp_path / "malformed-compile-import.json", malformed)
    result = check(tmp_path / "malformed-compile-import.json")
    assert result["ok"] is False
    assert "heterogeneous_tpu_beta_compile_import_invalid" in result["errors"]


def test_pack_rejects_unbound_or_zero_tpu_compile_status(tmp_path) -> None:
    live = tmp_path / "live.json"
    source = passing_live_report()
    source["tpu_training_evidence"]["compile_latency_ms"] = 0.0
    write(live, source)
    kernel = tmp_path / "kernel.json"
    invalid = tpu_compile_kernel_report("sha256:" + "c" * 64)
    write(kernel, invalid)
    with pytest.raises(
        ValueError,
        match="heterogeneous_training_tpu_compile_worker_binding_invalid",
    ):
        pack(live, tmp_path / "wrong-manifest", tpu_kernel_report=kernel)

    invalid = tpu_compile_kernel_report(source["training_manifest_hash"])
    invalid["worker_results"][0]["report"]["stage_process_statuses"][0][
        "compile_latency_ms"
    ] = 0.0
    write(kernel, invalid)
    with pytest.raises(
        ValueError, match="heterogeneous_training_tpu_compile_status_invalid"
    ):
        pack(live, tmp_path / "zero-latency", tpu_kernel_report=kernel)


def test_blocker_pack_imports_public_safe_bounded_resource_ledgers(tmp_path) -> None:
    live = tmp_path / "live.json"
    blocked = passing_live_report()
    blocked["live_run_performed"] = False
    blocked["blockers"] = ["kaggle_tpu_queue_window_exhausted"]
    write(live, blocked)
    acquisition = tmp_path / "private-acquisitions.json"
    live_attempts = tmp_path / "private-live-attempts.json"
    write(
        acquisition,
        {
            "attempt_limit": 3,
            "limit_extensions": [
                {
                    "schema": "crowdtensor_heterogeneous_training_limit_extension_v1",
                    "old_limit": 2,
                    "new_limit": 3,
                    "authorized_at": "2026-07-14T05:00:00+00:00",
                    "authorization_id_hash": HASH,
                    "authorization_identifier_public": False,
                    "credential_values_public": False,
                    "public_artifact_safe": True,
                }
            ],
            "attempts": [
                {
                    "attempt": 1,
                    "started_at": "2026-07-13T00:00:00+00:00",
                    "finished_at": "2026-07-13T01:00:00+00:00",
                    "completed": True,
                    "outcome": "push_rejected",
                },
                {
                    "attempt": 2,
                    "started_at": "2026-07-13T01:00:00+00:00",
                    "finished_at": "2026-07-13T13:00:00+00:00",
                    "completed": True,
                    "outcome": "kaggle_tpu_queue_window_exhausted",
                    "submission_count": 3,
                    "submission_outcomes": [
                        {
                            "submission": 1,
                            "finished_at": "2026-07-13T03:00:00+00:00",
                            "outcome": "tpu_running",
                        }
                    ],
                },
                {
                    "attempt": 3,
                    "started_at": "2026-07-14T05:00:01+00:00",
                    "finished_at": "2026-07-14T17:00:01+00:00",
                    "completed": True,
                    "outcome": "kaggle_tpu_queue_window_exhausted",
                },
            ],
        },
    )
    write(
        live_attempts,
        {
            "attempt_limit": 3,
            "attempts": [
                {
                    "attempt": 1,
                    "started_at": "2026-07-13T03:00:00+00:00",
                    "finished_at": "2026-07-13T04:00:00+00:00",
                    "completed": True,
                    "outcome": "runtime_failed",
                }
            ],
        },
    )

    packed = pack(
        live,
        tmp_path / "packed",
        acquisition_ledger=acquisition,
        live_attempt_ledger=live_attempts,
    )
    encoded = json.dumps(packed, sort_keys=True)

    assert packed["bounded_resource_summary"]["acquisition_windows_exhausted"] is True
    assert packed["bounded_resource_summary"]["tpu_acquisition"]["attempt_count"] == 3
    assert packed["bounded_resource_summary"]["tpu_acquisition"]["limit_extensions"] == [
        {
            "schema": "crowdtensor_heterogeneous_training_limit_extension_v1",
            "old_limit": 2,
            "new_limit": 3,
            "authorized_at": "2026-07-14T05:00:00+00:00",
            "authorization_id_hash": HASH,
            "authorization_identifier_public": False,
            "credential_values_public": False,
            "public_artifact_safe": True,
        }
    ]
    assert packed["bounded_resource_summary"]["live_gate_attempts_used"] == 1
    assert packed["resume_contract"]["current_goal_acquisition_boundary_exhausted"] is True
    assert packed["resume_contract"]["resume_command_public"] is False
    assert str(acquisition) not in encoded
    assert str(live_attempts) not in encoded
    assert check(tmp_path / "packed" / "training_heterogeneous_tpu_beta.json")["ok"] is True


def test_blocker_pack_imports_unlimited_but_duration_bounded_authorizations(
    tmp_path,
) -> None:
    live = tmp_path / "live.json"
    blocked = passing_live_report()
    blocked["live_run_performed"] = False
    blocked["blockers"] = ["kaggle_tpu_queue_window_exhausted"]
    write(live, blocked)
    acquisition = tmp_path / "private-acquisitions.json"
    live_attempts = tmp_path / "private-live-attempts.json"

    def unlimited_ledger(kind: str, duration: float, count: int) -> dict:
        return {
            "schema": "crowdtensor_heterogeneous_training_attempt_ledger_v1",
            "attempt_limit": 0,
            "attempt_limit_mode": "unlimited_authorized",
            "attempts": [
                {
                    "attempt": number,
                    "started_at": f"2026-07-15T0{number}:00:00+00:00",
                    "finished_at": f"2026-07-15T0{number}:30:00+00:00",
                    "completed": True,
                    "outcome": "diagnostic_failed",
                }
                for number in range(1, count + 1)
            ],
            "attempt_authorizations": [
                {
                    "schema": "crowdtensor_heterogeneous_training_attempt_authorization_v1",
                    "kind": kind,
                    "mode": "unlimited_authorized",
                    "previous_attempt_limit": count,
                    "authorized_at": "2026-07-15T00:00:00+00:00",
                    "authorization_id_hash": HASH,
                    "authorization_identifier_public": False,
                    "max_attempt_duration_seconds": duration,
                    "attempt_duration_remains_bounded": True,
                    "credential_values_public": False,
                    "public_artifact_safe": True,
                }
            ],
            "authorization_identifiers_public": False,
            "credential_values_public": False,
            "public_artifact_safe": True,
        }

    write(acquisition, unlimited_ledger("tpu_acquisition_window", 43200.0, 3))
    write(live_attempts, unlimited_ledger("six_step_live_gate", 21600.0, 4))
    packed = pack(
        live,
        tmp_path / "packed",
        acquisition_ledger=acquisition,
        live_attempt_ledger=live_attempts,
    )
    encoded = json.dumps(packed, sort_keys=True)

    resources = packed["bounded_resource_summary"]
    assert resources["acquisition_windows_exhausted"] is False
    assert resources["tpu_acquisition"]["attempt_limit_mode"] == "unlimited_authorized"
    assert resources["tpu_acquisition"]["unlimited_attempts_authorized"] is True
    assert resources["live_gate"]["unlimited_attempts_authorized"] is True
    assert resources["live_gate"]["attempt_limit_reached"] is False
    assert packed["resume_contract"]["current_goal_acquisition_boundary_exhausted"] is False
    assert packed["resume_contract"]["current_goal_live_gate_boundary_exhausted"] is False
    assert packed["resume_contract"]["next_action"] == (
        "start_next_bounded_tpu_window_then_full_live_gate"
    )
    assert "heterogeneous_tpu_training_live_gate_limit_reached" not in packed["blockers"]
    assert "private-acquisitions.json" not in encoded
    assert "private-live-attempts.json" not in encoded
    assert check(tmp_path / "packed" / "training_heterogeneous_tpu_beta.json")["ok"] is True

    malformed = deepcopy(packed)
    malformed["bounded_resource_summary"]["live_gate"]["attempt_authorizations"][0][
        "max_attempt_duration_seconds"
    ] = 21601.0
    malformed["content_hash"] = _stable_hash(
        {key: value for key, value in malformed.items() if key != "content_hash"}
    )
    write(tmp_path / "malformed.json", malformed)
    result = check(tmp_path / "malformed.json")
    assert result["ok"] is False
    assert (
        "heterogeneous_tpu_beta_live_gate_unlimited_authorization_invalid"
        in result["errors"]
    )


def test_blocker_pack_imports_public_safe_gpu_quota_diagnosis(tmp_path) -> None:
    live = tmp_path / "live.json"
    blocked = passing_live_report()
    blocked["live_run_performed"] = False
    blocked["blockers"] = ["heterogeneous_gpu_a_kernel_push_rejected"]
    write(live, blocked)
    diagnosis = tmp_path / "gpu-quota.json"
    write(
        diagnosis,
        {
            "schema": "crowdtensor_heterogeneous_training_tpu_gpu_quota_diagnosis_v1",
            "diagnosed_at": "2026-07-14T14:29:01+00:00",
            "source_live_report_hash": HASH,
            "failure_phase": "gpu_worker_push_after_tpu_running",
            "failed_role": "gpu_a",
            "tpu_submission_accepted": True,
            "tpu_running_observed": True,
            "failed_gpu_account_quota": {
                "weekly_gpu_quota_exhausted": True,
                "effective_remaining_after_reserved_seconds": 0.0,
            },
            "authorized_alternative_gpu_account_count": 3,
            "authorized_alternative_gpu_accounts_with_positive_quota": 3,
            "authorized_alternative_effective_remaining_min_seconds": 1.0,
            "authorized_alternative_effective_remaining_max_seconds": 2.0,
            "acquisition_window": {"attempt": 3},
            "live_gate": {"attempt_count": 3, "attempt_limit": 3},
            "cleanup_verified": True,
            "live_resources_left_running": False,
            "blockers": [
                "heterogeneous_tpu_training_live_gate_limit_reached",
                "kaggle_gpu_weekly_quota_exhausted",
            ],
            "account_labels_public": False,
            "credential_values_public": False,
            "credential_paths_public": False,
            "coordinator_url_public": False,
            "raw_quota_api_response_public": False,
            "public_artifact_safe": True,
        },
    )

    packed = pack(
        live,
        tmp_path / "packed",
        gpu_quota_diagnosis=diagnosis,
    )

    assert packed["gpu_quota_diagnosis"]["failed_role"] == "gpu_a"
    assert packed["gpu_quota_diagnosis"]["diagnosis_report_hash"].startswith(
        "sha256:"
    )
    assert "kaggle_gpu_weekly_quota_exhausted" in packed["blockers"]
    assert check(tmp_path / "packed" / "training_heterogeneous_tpu_beta.json")["ok"] is True


def test_blocker_pack_imports_integrity_checked_runtime_diagnosis(tmp_path) -> None:
    live = tmp_path / "live.json"
    blocked = passing_live_report()
    blocked["blockers"] = ["heterogeneous_kernel_terminal_before_training_complete"]
    write(live, blocked)
    diagnosis = {
        "schema": "crowdtensor_heterogeneous_training_tpu_runtime_diagnosis_v1",
        "diagnosed_at": "2026-07-14T21:00:00+00:00",
        "source_live_report_hash": file_hash(live),
        "peak_live_miner_count": 6,
        "online_accelerator_counts": {"cpu": 1, "cuda": 4, "tpu": 1},
        "assigned_stage_devices": [
            {"stage_id": 0, "device_type": "cuda"},
            {"stage_id": 1, "device_type": "cuda"},
            {"stage_id": 2, "device_type": "jax_tpu"},
            {"stage_id": 3, "device_type": "cuda"},
            {"stage_id": 4, "device_type": "cpu"},
        ],
        "checkpoint_submitted_stage_ids": [3, 4],
        "missing_step1_checkpoint_stage_ids": [0, 1, 2],
        "committed_steps": [],
        "terminal_worker_report_retrieved": False,
        "root_cause_confirmed": False,
        "cleanup_verified": True,
        "blockers": ["heterogeneous_tpu_terminal_worker_report_unavailable"],
        "credential_values_public": False,
        "credential_paths_public": False,
        "account_labels_public": False,
        "coordinator_url_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    diagnosis["content_hash"] = _stable_hash(diagnosis)
    diagnosis_path = tmp_path / "runtime-diagnosis.json"
    write(diagnosis_path, diagnosis)

    packed = pack(
        live,
        tmp_path / "packed",
        runtime_diagnosis=diagnosis_path,
    )
    checked = check(tmp_path / "packed" / "training_heterogeneous_tpu_beta.json")

    assert packed["runtime_diagnosis"]["peak_live_miner_count"] == 6
    assert packed["runtime_diagnosis"]["diagnosis_report_hash"].startswith(
        "sha256:"
    )
    assert "heterogeneous_tpu_terminal_worker_report_unavailable" in packed[
        "blockers"
    ]
    assert checked["ok"] is True


def test_blocker_pack_imports_non_acceptance_tpu_stage_diagnostic(tmp_path) -> None:
    live = tmp_path / "live.json"
    blocked = passing_live_report()
    blocked["blockers"] = ["heterogeneous_kernel_terminal_before_training_complete"]
    write(live, blocked)
    diagnostic = {
        "schema": "crowdtensor_heterogeneous_training_tpu_stage_diagnostic_live_v1",
        "ok": False,
        "live_probe_performed": True,
        "diagnostic_only": True,
        "full_training_gate_evidence": False,
        "same_job_three_accelerator_evidence": False,
        "live_gate_ledger_modified": False,
        "requested_accelerator": "tpuV5e8",
        "stage_id": 2,
        "terminal_state": "failed",
        "queue_observations": [{"state": "running"}],
        "kernel_output_collection": {"report_found": True},
        "kernel_report": {
            "phase": "jax_stage_loading",
            "failure_phase": "jax_stage_loading",
            "progress": {"phase": "jax_stage_loading"},
            "source_evidence": {"source_verified": True},
            "shard_evidence": {},
            "jax_load_evidence": {},
            "training_step_evidence": {},
            "blockers": ["jax_stage_loading:diagnostic_failure"],
        },
        "blockers": ["heterogeneous_tpu_stage_diagnostic_kernel_incomplete"],
        "cleanup": {
            "remote_kernel_deleted": True,
            "temporary_private_package_removed": True,
            "live_resources_left_running": False,
        },
        "public_artifact_safe": True,
    }
    diagnostic["content_hash"] = _stable_hash(diagnostic)
    diagnostic_path = tmp_path / "stage-diagnostic.json"
    write(diagnostic_path, diagnostic)

    packed = pack(
        live,
        tmp_path / "packed",
        stage_diagnostic=diagnostic_path,
    )
    checked = check(tmp_path / "packed" / "training_heterogeneous_tpu_beta.json")

    summary = packed["stage_diagnostic_summary"]
    assert summary["diagnostic_only"] is True
    assert summary["full_training_gate_evidence"] is False
    assert summary["kernel_phase"] == "jax_stage_loading"
    assert checked["ok"] is True


@pytest.mark.parametrize(
    ("mutate", "gate"),
    [
        (
            lambda report: report["tpu_training_evidence"].update(
                jax_tpu_device_count=0
            ),
            "real_v5e8_jax_training_verified",
        ),
        (
            lambda report: report["training_evidence"].update(
                committed_steps=[1, 2, 3]
            ),
            "six_atomic_steps_verified",
        ),
        (
            lambda report: report["cleanup"].update(
                live_resources_left_running=True
            ),
            "cleanup_verified",
        ),
    ],
)
def test_strict_checker_rejects_partial_or_unclean_evidence(
    tmp_path, mutate, gate
) -> None:
    report = passing_live_report()
    mutate(report)
    report["blockers"] = ["live_gate_incomplete"]
    live = tmp_path / "live.json"
    write(live, report)
    packed = pack(live, tmp_path / "packed")

    default = check(tmp_path / "packed" / "training_heterogeneous_tpu_beta.json")
    strict = check(
        tmp_path / "packed" / "training_heterogeneous_tpu_beta.json",
        require_ready=True,
    )

    assert packed["acceptance_gates"][gate] is False
    assert packed["heterogeneous_training_tpu_beta_ready"] is False
    assert default["ok"] is True
    assert default["heterogeneous_training_tpu_beta_ready"] is False
    assert strict["ok"] is False
    assert any(gate in error for error in strict["errors"])
