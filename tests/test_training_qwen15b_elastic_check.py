import copy

from scripts.training_qwen15b_elastic_check import REQUIRED_EVIDENCE, check_report


def _worker(role: str, steps: list[int], *, restored: bool) -> dict:
    generation = "new" if restored else "old"
    stages = [0, 1] if role == "kernel_a" else [2, 3]
    return {
        "ok": True,
        "kaggle_kernel": True,
        "cuda_available": True,
        "cuda_device_count": 2,
        "worker": {
            "ok": True,
            "role": role,
            "runtime": {"step_reports": [{"step": step} for step in steps]},
            "barrier_commits": [
                {"global_step": step, "barrier_committed": True} for step in steps
            ],
            "central_checkpoint_restore_verified": restored,
            "central_checkpoint_restore": (
                [
                    {
                        "stage_id": stage,
                        "global_step": 4,
                        "archive_hash": f"sha256:step4-stage{stage}",
                    }
                    for stage in stages
                ]
                if restored
                else []
            ),
            "fresh_checkpoint_directory_before_restore": True,
            "old_kernel_local_checkpoint_dependency": False,
            "elastic_client": {
                "miner_id_hash": f"sha256:{generation}-{role}"
            },
            "export": (
                {"standard_peft_format": True}
                if restored and role == "kernel_b"
                else {}
            ),
            "evaluation": (
                {"evaluation_verified": True}
                if restored and role == "kernel_b"
                else {}
            ),
        }
    }


def _ready_report() -> dict:
    return {
        "schema": "crowdtensor_qwen15b_elastic_live_probe_v1",
        "ok": True,
        "elastic_volunteer_training_ready": True,
        "requested_model": "Qwen/Qwen2.5-1.5B",
        "target_steps": 8,
        "mock_runtime_used": False,
        "cpu_fallback_used": False,
        "tiny_or_random_model_used": False,
        "old_generation": {
            "ok": True,
            "worker_reports": [
                _worker("kernel_a", [1, 2, 3, 4], restored=False),
                _worker("kernel_b", [1, 2, 3, 4], restored=False),
            ],
            "kernel_ref_hashes": ["sha256:old-a", "sha256:old-b"],
            "all_kernels_deleted": True,
            "deleted_at_epoch": 100.0,
            "deletions": [
                {"deleted_or_absent": True},
                {"deleted_or_absent": True},
            ],
        },
        "new_generation": {
            "ok": True,
            "worker_reports": [
                _worker("kernel_a", [5, 6, 7, 8], restored=True),
                _worker("kernel_b", [5, 6, 7, 8], restored=True),
            ],
            "kernel_ref_hashes": ["sha256:new-a", "sha256:new-b"],
            "all_kernels_deleted": True,
            "launched_at_epoch": 110.0,
            "deletions": [
                {"deleted_or_absent": True},
                {"deleted_or_absent": True},
            ],
        },
        "midpoint_status": {
            "committed_step": 4,
            "runtime_state": "paused_waiting_for_miners",
            "zero_live_miners": True,
            "live_miner_count": 0,
            "paused_waiting_for_miners": True,
            "miners": [
                {
                    "miner_id_hash": f"sha256:old-{role}",
                    "miner_session_hash": f"sha256:old-{role}-session",
                }
                for role in ("kernel_a", "kernel_b")
            ],
        },
        "final_status": {
            "committed_step": 8,
            "runtime_state": "completed",
            "optimizer_commit_count": 8,
            "committed_steps": list(range(1, 9)),
            "committed_steps_contiguous": True,
            "epochs": [
                {"epoch_id": 5, "target_step": 5, "state": "aborted"},
                {"epoch_id": 6, "target_step": 5, "state": "committed"},
            ],
            "assignments": [
                {
                    "epoch_id": 6,
                    "stage_id": stage,
                    "state": "completed",
                    "miner_id_hash": (
                        "sha256:new-kernel_a" if stage < 2 else "sha256:new-kernel_b"
                    ),
                }
                for stage in range(4)
            ],
            "miners": [
                {
                    "miner_id_hash": f"sha256:{generation}-{role}",
                    "miner_session_hash": f"sha256:{generation}-{role}-session",
                }
                for generation in ("old", "new")
                for role in ("kernel_a", "kernel_b")
            ],
            "events": [
                {
                    "operation": "stage_checkpoint_submitted",
                    "target_step": 4,
                    "stage_id": stage,
                    "archive_hash": f"sha256:step4-stage{stage}",
                }
                for stage in range(4)
            ]
            + [
                {"operation": "training_paused"},
                {"operation": "training_auto_woke"},
                {"operation": "training_auto_woke"},
            ],
        },
        "full_offline_pause": {
            "observed_seconds": 10.0,
            "new_kernel_launched_during_pause": False,
            "observations": [
                {
                    "runtime_state": "paused_waiting_for_miners",
                    "committed_step": 4,
                    "live_miner_count": 0,
                },
                {
                    "runtime_state": "paused_waiting_for_miners",
                    "committed_step": 4,
                    "live_miner_count": 0,
                },
            ],
        },
        "rendezvous": {
            "events": [
                {"run_kind": "elastic", "operation": "optimizer_step"}
                for _index in range(32)
            ]
        },
        "evidence": {**{key: True for key in REQUIRED_EVIDENCE}, "verified": True},
        "cleanup": {
            "all_four_kernels_deleted": True,
            "coordinator_stopped": True,
            "tunnel_stopped": True,
            "private_runtime_removed": True,
            "rendezvous_payloads_removed": True,
            "uncommitted_checkpoint_blobs_removed": True,
            "live_resources_left_running": False,
        },
        "blockers": [],
        "credentials_public": False,
        "credential_paths_public": False,
        "coordinator_url_public": False,
        "session_tokens_public": False,
        "assignment_tokens_public": False,
        "private_paths_public": False,
        "activation_values_public": False,
        "gradient_values_public": False,
        "checkpoint_tensor_values_public": False,
        "adapter_tensor_values_public": False,
        "token_ids_public": False,
        "raw_training_text_public": False,
        "public_artifact_safe": True,
    }


def test_strict_checker_accepts_only_full_offline_replacement_evidence() -> None:
    result = check_report(_ready_report(), require_ready=True)
    assert result["ok"] is True
    assert result["elastic_volunteer_training_ready"] is True


def test_strict_checker_rejects_restart_from_step_zero_and_identity_leak() -> None:
    wrong_steps = _ready_report()
    wrong_steps["new_generation"]["worker_reports"][0]["worker"]["runtime"][
        "step_reports"
    ][0]["step"] = 1
    result = check_report(wrong_steps, require_ready=True)
    assert result["ok"] is False
    assert "elastic_live_gate_failed:new_steps_valid" in result["errors"]

    leaked = copy.deepcopy(_ready_report())
    leaked["debug_session"] = "elastic-miner-private-session-id"
    result = check_report(leaked, require_ready=True)
    assert result["ok"] is False
    assert any("elastic-miner-" in error for error in result["errors"])
