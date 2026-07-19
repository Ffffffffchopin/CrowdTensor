from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from crowdtensor.heterogeneous_training_manifest import qwen25_7b_lora_tpu_manifest
from scripts.training_heterogeneous_tpu_beta_check import check
from scripts.training_heterogeneous_tpu_beta_live_probe import (
    TPU_KERNEL_REPORT,
    _finish_attempt,
    _reserve_attempt,
    authorize_unlimited_attempts,
    build_live_evidence,
    classify_training_worker_push,
    classify_tpu_push,
    collect_kernel_output_with_retry,
    extend_acquisition_window_limit,
    extend_live_attempt_limit,
    gpu_quota_preflight_summary,
    reserve_acquisition_window,
    runtime_observation_summary,
    runtime_progress_summary,
    stale_generation_probe_due,
    terminal_before_training_complete,
)
from scripts.training_heterogeneous_tpu_beta_pack import pack


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def test_tpu_push_classifier_preserves_actionable_public_blockers() -> None:
    assert classify_tpu_push(
        {"ok": False, "output_tail": "Maximum batch TPU session count reached"}
    ) == "kaggle_tpu_batch_session_limit_reached"
    assert classify_tpu_push(
        {"ok": False, "output_tail": "HTTP 429 Too Many Requests"}
    ) == "kaggle_tpu_push_rate_limited"
    assert classify_tpu_push(
        {"ok": True, "output_tail": "Kernel version 1 successfully pushed"}
    ) == "tpu_submission_accepted"
    assert terminal_before_training_complete(
        {"tpu": "failed", "gpu": "running"},
        runtime_state="paused_waiting_for_miners",
    ) == ["tpu"]
    assert terminal_before_training_complete(
        {"tpu": "complete", "gpu": "running"}, runtime_state="completed"
    ) == []
    summary = runtime_observation_summary(
        [
            {
                "kernel_states": {"tpu": "running"},
                "committed_step": 0,
                "placement_generation": 1,
            },
            {
                "kernel_states": {"tpu": "failed"},
                "committed_step": 2,
                "placement_generation": 3,
            },
        ]
    )
    assert summary["observation_count"] == 2
    assert summary["max_committed_step"] == 2
    assert summary["terminal_kernel_states_seen"] == ["tpu:failed"]


def test_gpu_quota_preflight_and_worker_push_are_actionable() -> None:
    summary = gpu_quota_preflight_summary(
        {
            "ok": True,
            "quota_refresh_time": "2026-07-18T00:00:00",
            "gpu_quota": {
                "present": True,
                "time_used_seconds": 144000.0,
                "time_reserved_seconds": 0.0,
                "total_time_allowed_seconds": 108000.0,
                "effective_remaining_after_reserved_seconds": 0.0,
                "quota_exhausted_by_used": True,
            },
        },
        phase="before_tpu_acquisition",
    )
    assert summary["weekly_gpu_quota_exhausted"] is True
    assert summary["quota_refresh_time"] == "2026-07-18T00:00:00"
    assert classify_training_worker_push(
        "gpu_a",
        {
            "ok": False,
            "output_tail": "Maximum weekly GPU quota of 30.00 hours reached",
        },
    ) == "kaggle_gpu_weekly_quota_exhausted"
    assert classify_training_worker_push(
        "cpu", {"ok": False, "output_tail": "HTTP 429 Too Many Requests"}
    ) == "kaggle_cpu_push_rate_limited"


def test_runtime_progress_and_terminal_output_collection_are_public_safe(
    tmp_path,
) -> None:
    progress = runtime_progress_summary(
        {
            "runtime_state": "running",
            "committed_steps": [],
            "live_miner_count": 3,
            "miners": [
                {"state": "online", "accelerator": "cuda"},
                {"state": "online", "accelerator": "tpu"},
                {"state": "online", "accelerator": "cpu"},
            ],
            "assignments": [
                {"stage_id": 0, "device_type": "cuda", "state": "active"},
                {"stage_id": 2, "device_type": "jax_tpu", "state": "active"},
            ],
            "events": [
                {
                    "sequence": 1,
                    "operation": "stage_checkpoint_submitted",
                    "stage_id": 4,
                    "target_step": 1,
                },
                {
                    "sequence": 2,
                    "operation": "stage_profile_updated",
                    "stage_id": 3,
                },
            ],
        }
    )
    assert progress["online_accelerator_counts"] == {
        "cpu": 1,
        "cuda": 1,
        "tpu": 1,
    }
    assert progress["checkpoint_submitted_stage_ids"] == [4]
    assert progress["profiled_stage_ids"] == [3]

    calls = []

    def runner(command, *, env, timeout):
        del env, timeout
        calls.append(command)
        if len(calls) == 2:
            (tmp_path / TPU_KERNEL_REPORT).write_text(
                json.dumps({"ok": False, "blockers": ["stage_loader_failed"]}),
                encoding="utf-8",
            )
        return {"returncode": 0, "timed_out": False}

    kernel, collection = collect_kernel_output_with_retry(
        ref="private/ref",
        env={},
        destination=tmp_path,
        filename=TPU_KERNEL_REPORT,
        file_pattern=r"custom-diagnostic\.json",
        timeout_seconds=2.0,
        poll_interval_seconds=0.1,
        runner=runner,
        sleeper=lambda _seconds: None,
    )
    assert kernel["blockers"] == ["stage_loader_failed"]
    assert collection["report_found"] is True
    assert collection["attempt_count"] == 2
    assert collection["kernel_ref_public"] is False
    assert calls[-1][-1] == r"custom-diagnostic\.json"


def test_stale_generation_probe_runs_at_replacement_before_step4_commit() -> None:
    assert stale_generation_probe_due(
        {
            "runtime_state": "running",
            "committed_step": 3,
            "placement_generation": 2,
        },
        initial_generation=1,
        already_verified=False,
    ) is True
    assert stale_generation_probe_due(
        {
            "runtime_state": "completed",
            "committed_step": 6,
            "placement_generation": 2,
        },
        initial_generation=1,
        already_verified=False,
    ) is False
    assert stale_generation_probe_due(
        {
            "runtime_state": "running",
            "committed_step": 3,
            "placement_generation": 2,
        },
        initial_generation=1,
        already_verified=True,
    ) is False


def test_acquisition_retry_reuses_window_without_resetting_limit(tmp_path) -> None:
    ledger = tmp_path / "acquisitions.json"
    first = _reserve_attempt(ledger, limit=2)
    _finish_attempt(ledger, attempt=first, outcome="push_rejected")
    second = _reserve_attempt(ledger, limit=2)
    _finish_attempt(ledger, attempt=second, outcome="tpu_running")

    reused, remaining = reserve_acquisition_window(
        ledger,
        limit=2,
        reuse_attempt=second,
        window_seconds=43200.0,
    )
    value = json.loads(ledger.read_text(encoding="utf-8"))

    assert reused == 2
    assert remaining > 43100.0
    assert len(value["attempts"]) == 2
    assert value["attempts"][-1]["submission_count"] == 2
    assert value["attempts"][-1]["submission_outcomes"][-1]["outcome"] == "tpu_running"
    assert value["attempts"][-1]["completed"] is False


def _completed_two_window_ledger(path) -> None:
    first = _reserve_attempt(path, limit=2)
    _finish_attempt(path, attempt=first, outcome="push_rejected")
    second = _reserve_attempt(path, limit=2)
    _finish_attempt(path, attempt=second, outcome="queue_window_exhausted")


def test_acquisition_limit_extension_is_authorized_auditable_and_idempotent(
    tmp_path,
) -> None:
    ledger = tmp_path / "acquisitions.json"
    _completed_two_window_ledger(ledger)

    first = extend_acquisition_window_limit(
        ledger,
        requested_limit=3,
        extension_authorized=True,
        authorization_id="tpu-training-beta-window-3-20260714",
    )
    encoded_after_first = ledger.read_text(encoding="utf-8")
    second = extend_acquisition_window_limit(
        ledger,
        requested_limit=3,
        extension_authorized=True,
        authorization_id="tpu-training-beta-window-3-20260714",
    )
    value = json.loads(encoded_after_first)

    assert first == second
    assert ledger.read_text(encoding="utf-8") == encoded_after_first
    assert value["attempt_limit"] == 3
    assert len(value["attempts"]) == 2
    assert len(value["limit_extensions"]) == 1
    assert value["limit_extensions"][0]["old_limit"] == 2
    assert value["limit_extensions"][0]["new_limit"] == 3
    assert value["limit_extensions"][0]["authorization_id_hash"].startswith(
        "sha256:"
    )
    assert "tpu-training-beta-window-3-20260714" not in encoded_after_first

    third, _remaining = reserve_acquisition_window(
        ledger,
        limit=3,
        reuse_attempt=0,
        window_seconds=43200.0,
    )
    reserved = json.loads(ledger.read_text(encoding="utf-8"))
    assert third == 3
    assert len(reserved["attempts"]) == 3
    assert reserved["limit_extensions"] == value["limit_extensions"]
    assert reserved["attempts"][-1]["submission_limit"] == 3
    assert reserved["attempts"][-1]["window_seconds"] == 43200.0
    assert reserved["attempts"][-1]["expires_at"]


def test_acquisition_limit_extension_rejects_missing_authorization(tmp_path) -> None:
    ledger = tmp_path / "acquisitions.json"
    _completed_two_window_ledger(ledger)

    with pytest.raises(
        RuntimeError,
        match="heterogeneous_acquisition_window_extension_not_authorized",
    ):
        extend_acquisition_window_limit(
            ledger,
            requested_limit=3,
            extension_authorized=False,
            authorization_id="approval",
        )
    with pytest.raises(
        RuntimeError,
        match="heterogeneous_acquisition_window_extension_authorization_id_missing",
    ):
        extend_acquisition_window_limit(
            ledger,
            requested_limit=3,
            extension_authorized=True,
            authorization_id="",
        )


def test_acquisition_limit_extension_rejects_invalid_jump(tmp_path) -> None:
    ledger = tmp_path / "acquisitions.json"
    _completed_two_window_ledger(ledger)

    with pytest.raises(
        RuntimeError,
        match="heterogeneous_acquisition_window_extension_invalid_jump",
    ):
        extend_acquisition_window_limit(
            ledger,
            requested_limit=4,
            extension_authorized=True,
            authorization_id="approval",
        )


def _completed_three_live_gate_ledger(path) -> None:
    for outcome in ("runtime_failed", "wait_timeout", "gpu_push_rejected"):
        attempt = _reserve_attempt(path, limit=3)
        _finish_attempt(path, attempt=attempt, outcome=outcome)


def test_live_gate_limit_extension_is_authorized_auditable_and_idempotent(
    tmp_path,
) -> None:
    ledger = tmp_path / "live-attempts.json"
    _completed_three_live_gate_ledger(ledger)

    first = extend_live_attempt_limit(
        ledger,
        requested_limit=4,
        extension_authorized=True,
        authorization_id="tpu-training-beta-live-gate-4-20260714",
    )
    encoded = ledger.read_text(encoding="utf-8")
    second = extend_live_attempt_limit(
        ledger,
        requested_limit=4,
        extension_authorized=True,
        authorization_id="tpu-training-beta-live-gate-4-20260714",
    )
    fourth = _reserve_attempt(ledger, limit=4)
    value = json.loads(ledger.read_text(encoding="utf-8"))

    assert first == second
    assert fourth == 4
    assert value["attempt_limit"] == 4
    assert len(value["attempts"]) == 4
    assert value["limit_extensions"] == [first]
    assert "tpu-training-beta-live-gate-4-20260714" not in encoded


def test_live_gate_limit_extension_rejects_missing_or_invalid_authorization(
    tmp_path,
) -> None:
    ledger = tmp_path / "live-attempts.json"
    _completed_three_live_gate_ledger(ledger)

    with pytest.raises(
        RuntimeError, match="heterogeneous_live_gate_extension_not_authorized"
    ):
        extend_live_attempt_limit(
            ledger,
            requested_limit=4,
            extension_authorized=False,
            authorization_id="approval",
        )
    with pytest.raises(
        RuntimeError,
        match="heterogeneous_live_gate_extension_authorization_id_missing",
    ):
        extend_live_attempt_limit(
            ledger,
            requested_limit=4,
            extension_authorized=True,
            authorization_id="",
        )
    with pytest.raises(
        RuntimeError, match="heterogeneous_live_gate_extension_invalid_jump"
    ):
        extend_live_attempt_limit(
            ledger,
            requested_limit=5,
            extension_authorized=True,
            authorization_id="approval",
        )


@pytest.mark.parametrize(
    ("kind", "duration_seconds", "initial_limit", "outcomes"),
    [
        (
            "tpu_acquisition_window",
            43200.0,
            3,
            ["push_rejected", "queue_timeout", "tpu_running"],
        ),
        (
            "six_step_live_gate",
            21600.0,
            4,
            ["runtime_failed", "wait_timeout", "push_rejected", "terminal"],
        ),
    ],
)
def test_unlimited_attempt_authorization_is_auditable_idempotent_and_reservable(
    tmp_path,
    kind,
    duration_seconds,
    initial_limit,
    outcomes,
) -> None:
    ledger = tmp_path / f"{kind}.json"
    for outcome in outcomes:
        attempt = _reserve_attempt(ledger, limit=initial_limit)
        _finish_attempt(ledger, attempt=attempt, outcome=outcome)

    authorization_id = f"private-{kind}-unlimited-authorization"
    first = authorize_unlimited_attempts(
        ledger,
        kind=kind,
        authorization_granted=True,
        authorization_id=authorization_id,
        max_attempt_duration_seconds=duration_seconds,
    )
    encoded = ledger.read_text(encoding="utf-8")
    second = authorize_unlimited_attempts(
        ledger,
        kind=kind,
        authorization_granted=True,
        authorization_id=authorization_id,
        max_attempt_duration_seconds=duration_seconds,
    )
    next_attempt = _reserve_attempt(ledger, limit=0)
    _finish_attempt(ledger, attempt=next_attempt, outcome="diagnostic_failed")
    following_attempt = _reserve_attempt(ledger, limit=0)
    value = json.loads(ledger.read_text(encoding="utf-8"))

    assert first == second
    assert next_attempt == initial_limit + 1
    assert following_attempt == initial_limit + 2
    assert value["attempt_limit"] == 0
    assert value["attempt_limit_mode"] == "unlimited_authorized"
    assert len(value["attempt_authorizations"]) == 1
    assert value["attempt_authorizations"][0]["previous_attempt_limit"] == initial_limit
    assert value["attempt_authorizations"][0]["attempt_duration_remains_bounded"] is True
    assert value["attempt_authorizations"][0]["authorization_id_hash"].startswith(
        "sha256:"
    )
    assert authorization_id not in encoded
    assert value["authorization_identifiers_public"] is False


def test_unlimited_attempt_reservation_requires_authorization(tmp_path) -> None:
    ledger = tmp_path / "attempts.json"
    with pytest.raises(
        RuntimeError,
        match="heterogeneous_live_attempt_unlimited_authorization_missing",
    ):
        _reserve_attempt(ledger, limit=0)

    bounded = _reserve_attempt(ledger, limit=1)
    _finish_attempt(ledger, attempt=bounded, outcome="complete")
    with pytest.raises(
        RuntimeError, match="heterogeneous_unlimited_attempts_not_authorized"
    ):
        authorize_unlimited_attempts(
            ledger,
            kind="six_step_live_gate",
            authorization_granted=False,
            authorization_id="private-approval",
            max_attempt_duration_seconds=21600.0,
        )
    with pytest.raises(
        RuntimeError,
        match="heterogeneous_unlimited_attempts_authorization_id_missing",
    ):
        authorize_unlimited_attempts(
            ledger,
            kind="six_step_live_gate",
            authorization_granted=True,
            authorization_id="",
            max_attempt_duration_seconds=21600.0,
        )
    with pytest.raises(
        RuntimeError, match="heterogeneous_unlimited_attempt_duration_invalid"
    ):
        authorize_unlimited_attempts(
            ledger,
            kind="six_step_live_gate",
            authorization_granted=True,
            authorization_id="private-approval",
            max_attempt_duration_seconds=21601.0,
        )


def stage_result(stage_id: int, step: int) -> dict:
    return {
        "stage_id": stage_id,
        "target_step": step,
        "lora_gradient_norm": 0.1 + stage_id,
        "adapter_tensor_hash": "sha256:" + f"{stage_id + step:064x}"[-64:],
        "optimizer_step_applied": True,
        "scheduler_step_applied": True,
        "compile_latency_ms": 0.0,
        "losses": [1.0 / step] if stage_id == 4 else [],
    }


def worker(role: str, policy: str, stage_id: int, steps: list[int]) -> dict:
    capability = {
        "content_hash": HASH_A,
        "gpus": [{"device_id": "cuda:0"}] if policy == "cuda" else [],
        "tpu_groups": (
            [{"device_count": 8, "accelerator_type": "TPU v5e"}]
            if policy == "jax_tpu"
            else []
        ),
    }
    ready = {
        "stage_id": stage_id,
        "adapter_hash_before": "sha256:" + f"{stage_id:064x}"[-64:],
        "resumed": role == "tpu_replacement",
        "resumed_global_step": 3 if role == "tpu_replacement" else 0,
        "jax_mesh_shape": [8] if policy == "jax_tpu" else [],
        "all_mesh_devices_used": policy == "jax_tpu",
        "load_report": {
            "parameter_sharding": "named_mesh_model_axis",
            "compute_dtype": "bfloat16",
        },
    }
    return {
        "ok": True,
        "deployment_role": role,
        "miner_id_hash": "sha256:" + f"{100 + stage_id + (50 if role == 'tpu_replacement' else 0):064x}"[-64:],
        "device_policy": policy,
        "training_manifest_hash": qwen25_7b_lora_tpu_manifest()["content_hash"],
        "assigned_stage_ids": [stage_id],
        "steps": [
            {"target_step": step, "stages": [stage_result(stage_id, step)]}
            for step in steps
        ],
        "steps_completed": len(steps),
        "central_checkpoint_restore_count": int(role == "tpu_replacement"),
        "stage_process_ready": [ready],
        "stage_process_statuses": [
            {
                "stage_id": stage_id,
                "compile_latency_ms": 2500.0 if policy == "jax_tpu" else 0.0,
                "steady_forward_sample_count": max(1, len(steps) - 1),
            }
        ],
        "shard_reports": [
            {"stage_id": stage_id, "stage_selective_loading": True}
        ],
        "capability": capability,
    }


def kernel_reports() -> list[dict]:
    return [
        {
            "kernel_role": "tpu",
            "kernel_ref_hash": HASH_A,
            "logical_tpu_restart_count": 1,
            "same_tpu_kernel_runtime_hash": HASH_B,
            "pause_observation": {"verified": True},
            "worker_results": [
                {"label": "tpu_old", "report": worker("tpu_old", "jax_tpu", 2, [1, 2, 3])},
                {"label": "tpu_replacement", "report": worker("tpu_replacement", "jax_tpu", 2, [4, 5, 6])},
            ],
        },
        {
            "kernel_role": "gpu_a",
            "kernel_ref_hash": HASH_A,
            "worker_results": [
                {"label": "gpu0", "report": worker("gpu0", "cuda", 0, list(range(1, 7)))},
                {"label": "gpu1", "report": worker("gpu1", "cuda", 1, list(range(1, 7)))},
            ],
        },
        {
            "kernel_role": "gpu_b",
            "kernel_ref_hash": HASH_A,
            "worker_results": [
                {"label": "gpu3", "report": worker("gpu3", "cuda", 3, list(range(1, 7)))},
            ],
        },
        {
            "kernel_role": "cpu",
            "kernel_ref_hash": HASH_A,
            "worker_results": [
                {"label": "cpu", "report": worker("cpu", "cpu", 4, list(range(1, 7)))},
            ],
            "export_reload": {
                "report": {
                    "standard_peft_format": True,
                    "adapter_reload_verified": True,
                    "forward_inference_verified": True,
                    "finite_logits_verified": True,
                    "model_binding_verified": True,
                    "adapter_file_hash": HASH_A,
                }
            },
        },
    ]


def final_status_and_snapshots() -> tuple[dict, list[dict]]:
    epochs = []
    assignments = []
    device_order = ["cuda", "cuda", "jax_tpu", "cuda", "cpu"]
    miner_ids = [
        worker("gpu0", "cuda", 0, [1])["miner_id_hash"],
        worker("gpu1", "cuda", 1, [1])["miner_id_hash"],
        worker("tpu_old", "jax_tpu", 2, [1])["miner_id_hash"],
        worker("gpu3", "cuda", 3, [1])["miner_id_hash"],
        worker("cpu", "cpu", 4, [1])["miner_id_hash"],
    ]
    replacement_tpu = worker("tpu_replacement", "jax_tpu", 2, [4])["miner_id_hash"]
    for step in range(1, 7):
        generation = 1 if step <= 3 else 3
        epochs.append({"epoch_id": step, "target_step": step, "state": "committed"})
        for stage_id, device_type in enumerate(device_order):
            miner = replacement_tpu if stage_id == 2 and step >= 4 else miner_ids[stage_id]
            assignments.append(
                {
                    "epoch_id": step,
                    "stage_id": stage_id,
                    "miner_id_hash": miner,
                    "miner_session_hash": miner,
                    "device_id": "jax_tpu:0" if stage_id == 2 else "cpu" if stage_id == 4 else "cuda:0",
                    "device_type": device_type,
                    "placement_generation": generation,
                }
            )
    snapshots = []
    for generation in (1, 3):
        rows = []
        for stage_id, device_type in enumerate(device_order):
            miner = replacement_tpu if stage_id == 2 and generation == 3 else miner_ids[stage_id]
            row = {
                "stage_id": stage_id,
                "miner_id_hash": miner,
                "device_id": "jax_tpu:0" if stage_id == 2 else "cpu" if stage_id == 4 else "cuda:0",
                "device_type": device_type,
                "resource_estimate": {"estimated_peak_bytes": 100},
                "available_after_reserve_bytes": 1000,
                "compute_latency_ms": 10.0,
                "compute_latency_measured": generation == 3,
                "incoming_transfer_latency_ms": 1.0,
                "incremental_score": 11.0,
                "selection_reason": "measured" if generation == 3 else "estimated",
            }
            if stage_id == 2:
                row["tpu_compile_cost_ms"] = 2500.0
            rows.append(row)
        snapshots.append(
            {
                "placement_generation": generation,
                "placement_plan": {"content_hash": HASH_A if generation == 1 else HASH_B, "assignments": rows},
            }
        )
    return {
        "runtime_state": "completed",
        "committed_steps": list(range(1, 7)),
        "epochs": epochs,
        "assignments": assignments,
    }, snapshots


def transport_messages() -> dict:
    messages = []
    for step in range(1, 7):
        generation = 1 if step <= 3 else 3
        for source in range(4):
            messages.append(
                {
                    "message_id": "sha256:" + f"{step * 100 + source:064x}"[-64:],
                    "global_step": step,
                    "source_stage_id": source,
                    "target_stage_id": source + 1,
                    "direction": "forward_activation",
                    "placement_generation": generation,
                    "complete": True,
                    "chunk_hashes_verified": True,
                    "payload_hash_verified": True,
                }
            )
        for source in range(4, 0, -1):
            messages.append(
                {
                    "message_id": "sha256:" + f"{step * 1000 + source:064x}"[-64:],
                    "global_step": step,
                    "source_stage_id": source,
                    "target_stage_id": source - 1,
                    "direction": "backward_gradient",
                    "placement_generation": generation,
                    "complete": True,
                    "chunk_hashes_verified": True,
                    "payload_hash_verified": True,
                }
            )
    return {"messages": messages}


def test_live_builder_maps_runtime_evidence_into_strict_canonical_gate(tmp_path) -> None:
    manifest = qwen25_7b_lora_tpu_manifest()
    status, snapshots = final_status_and_snapshots()
    controller = SimpleNamespace(job_id="fixture-job", run_id="fixture-run")
    checkpoint = {
        "stage_ids": [0, 1, 2, 3, 4],
        "all_five_stage_archives_valid": True,
        "atomic_checkpoint_barrier_verified": True,
        "pytorch_components_complete": True,
        "tpu_runtime_backend": "jax_tpu",
        "tpu_optimizer_state_present": True,
        "tpu_scheduler_state_present": True,
        "tpu_jax_prng_state_present": True,
        "tpu_grad_scaler_applicable": False,
        "tpu_pickle_deserialization_allowed": False,
        "all_component_hashes_verified": True,
    }
    cleanup = {
        "all_remote_kernels_deleted": True,
        "temporary_private_packages_removed": True,
        "coordinator_stopped": True,
        "tunnel_stopped": True,
        "tensor_payloads_removed": True,
        "temporary_credentials_removed": True,
        "live_resources_left_running": False,
    }
    live = build_live_evidence(
        manifest=manifest,
        controller=controller,
        kernel_reports=kernel_reports(),
        final_status=status,
        snapshots=snapshots,
        retained_transport=transport_messages(),
        transport_contract={
            "chunking_verified": True,
            "finite_retry_verified": True,
            "idempotent_delivery_verified": True,
        },
        checkpoint_evidence=checkpoint,
        local_export={
            "standard_peft_format": True,
            "adapter_tensor_count": 392,
            "layer_indexes": list(range(28)),
            "adapter_file_hash": HASH_A,
        },
        stale_probe={"verified": True},
        cleanup=cleanup,
        blockers=[],
    )
    live_path = tmp_path / "live.json"
    live_path.write_text(json.dumps(live), encoding="utf-8")
    regression_path = tmp_path / "regression.json"
    regression_path.write_text(
        json.dumps(
            {
                "passed": 160,
                "failed": 0,
                "legacy_cpu_cuda_tests_included": True,
                "jax_tpu_tests_included": True,
                "public_safety_tests_included": True,
            }
        ),
        encoding="utf-8",
    )

    packed = pack(
        live_path,
        tmp_path / "packed",
        regression_summary=regression_path,
    )
    checked = check(
        tmp_path / "packed" / "training_heterogeneous_tpu_beta.json",
        require_ready=True,
    )

    assert packed["heterogeneous_training_tpu_beta_ready"] is True
    assert checked["ok"] is True
    assert checked["error_count"] == 0
