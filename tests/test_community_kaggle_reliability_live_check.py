import json

from scripts.community_kaggle_reliability_live_check import check
from scripts.community_kaggle_reliability_live_probe import SCHEMA
from scripts.community_live_gate_ledger_amend import AMENDMENT_SCOPE


def valid_report() -> dict:
    return {
        "schema": SCHEMA,
        "ok": True,
        "live_run_performed": True,
        "live_gate_attempt": 3,
        "maximum_full_live_gates": 3,
        "live_gate_amendment": {
            "verified": True,
            "approval_statement_hash": "sha256:" + "c" * 64,
            "amended_at": "2026-07-17T00:00:00Z",
            "old_maximum_full_live_gates": 2,
            "new_maximum_full_live_gates": 3,
            "scope": AMENDMENT_SCOPE,
        },
        "maximum_gate_seconds": 2700,
        "node_scope": "Kaggle logical multi-node",
        "physical_multi_machine_verified": False,
        "real_open_model_weights": True,
        "random_or_synthetic_weights_used": False,
        "providers": ["kaggle_cpu", "kaggle_cuda"],
        "logical_kernel_count": 2,
        "logical_miner_count": 3,
        "clean_install": {"verified": True, "fresh_install_root_per_kernel": True, "fresh_install_kind": "pip_target", "workspace_import_used": False},
        "committed_step_ids": list(range(1, 101)),
        "duration_seconds": 1200,
        "worker_replacement_verified": True,
        "coordinator_restart_verified": True,
        "checkpoint_recovery_verified": True,
        "ledger_exactly_once_verified": True,
        "finite_update_verified": True,
        "adapter_reload_verified": True,
        "monitoring_verified": True,
        "cleanup_verified": True,
        "worker_replacement": {
            "verified": True,
            "replacement_after_step": 30,
            "restored_checkpoint_step": 30,
            "optimizer_state_restored": True,
            "old_worker_id_hash": "sha256:" + "a" * 64,
            "replacement_worker_id_hash": "sha256:" + "b" * 64,
        },
        "coordinator_restart": {
            "verified": True,
            "restart_barrier_verified": True,
            "restart_at_committed_step": 50,
            "generation_before": 1,
            "generation_after": 2,
            "same_committed_step_after_restart": True,
        },
        "kernel_evidence": [
            {"kernel_role": "stage0", "backend": "cuda", "ok": True, "wheel_clean_install": True, "model_stack_import_verified": True},
            {"kernel_role": "stage1", "backend": "cpu", "ok": True, "wheel_clean_install": True, "model_stack_import_verified": True},
        ],
        "second_model_live": {
            "verified": True,
            "logical_stage_count": 2,
            "devices": ["cuda", "cuda"],
            "adapter_reload_verified": True,
        },
        "export": {"standard_peft_format": True, "adapter_tensor_count": 420},
        "reload": {"adapter_reload_verified": True, "independent_process_reload": True},
        "benchmark": {
            "steps_per_second": 0.1,
            "p50_step_seconds": 10.0,
            "p95_step_seconds": 12.0,
            "checkpoint_count": 2,
            "checkpoint_write_count": 6,
            "checkpoint_bytes": 1024,
            "forward_payload_count": 100,
            "forward_payload_bytes": 2048,
            "backward_payload_count": 100,
            "backward_payload_bytes": 2048,
            "transfer_payloads_private": True,
            "resource_scope": "one Kaggle GPU Kernel plus one Kaggle CPU Kernel",
        },
        "tpu": {"required": False, "acquisition_windows_used": 0},
        "cleanup": {
            "all_remote_kernels_deleted": True,
            "coordinator_stopped": True,
            "tunnel_stopped": True,
            "private_runtime_removed": True,
            "live_resources_left_running": False,
        },
        "acceptance": {"ok": True},
        "credential_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def test_checker_accepts_bounded_complete_gate(tmp_path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(valid_report()), encoding="utf-8")
    assert check(path)["ok"] is True


def test_checker_rejects_gaps_cleanup_and_overclaim(tmp_path) -> None:
    value = valid_report()
    value["committed_step_ids"].remove(50)
    value["physical_multi_machine_verified"] = True
    value["cleanup"]["live_resources_left_running"] = True
    value["worker_replacement"]["optimizer_state_restored"] = False
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    result = check(path)
    assert result["ok"] is False
    assert len(result["errors"]) >= 4


def test_checker_rejects_old_gate_limit_or_missing_authorization(tmp_path) -> None:
    value = valid_report()
    value["live_gate_attempt"] = 2
    value["maximum_full_live_gates"] = 2
    value.pop("live_gate_amendment")
    path = tmp_path / "old-bound.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    errors = check(path)["errors"]
    assert "community_reliability_live_attempt_bound_invalid" in errors
    assert "community_reliability_live_authorization_invalid" in errors
