from __future__ import annotations

import copy
import json

from scripts.training_elastic_beta_check import REQUIRED_GATES, _stable_hash, check
from scripts.training_elastic_beta_pack import pack


def _worker(role: str, start: int, end: int, *, final: bool = False) -> dict:
    return {
        "schema": "crowdtensor_elastic_training_beta_miner_join_v1",
        "ok": True,
        "role": role,
        "expected_start_step": start,
        "segment_end_step": end,
        "barrier_commit_count": 4,
        "all_completed_barriers_committed": True,
        "base_weights_frozen": True,
        "positive_lora_gradient_norms": True,
        "graceful_drain_applied": not final,
        "central_checkpoint_restore_verified": True,
        "standard_peft_export_verified": final and role == "kernel_b",
        "evaluation_verified": final and role == "kernel_b",
    }


def _generation(start: int, end: int) -> dict:
    final = end == 8
    return {
        "ok": True,
        "maximum_running_kernel_count": 2,
        "terminal_states": ["complete", "complete"],
        "all_kernels_deleted": True,
        "worker_reports": [
            {"worker": _worker("kernel_a", start, end, final=final)},
            {"worker": _worker("kernel_b", start, end, final=final)},
        ],
    }


def _complete_report() -> dict:
    report = {
        "schema": "crowdtensor_elastic_training_beta_live_probe_v1",
        "ok": True,
        "elastic_training_beta_ready": True,
        "live_run_performed": True,
        "model_id": "Qwen/Qwen2.5-1.5B",
        "target_steps": 8,
        "ordinary_user_cli": {"create": True, "status": True, "export": True},
        "acceptance_gates": {key: True for key in REQUIRED_GATES},
        "old_generation": _generation(0, 4),
        "replacement_generation": _generation(4, 8),
        "midpoint": {
            "committed_step": 4,
            "zero_live_miners": True,
            "all_observations_paused": True,
        },
        "service_restart": {
            "old_service_stopped": True,
            "same_job_id": True,
            "committed_step_recovered": True,
            "runtime_paused_recovered": True,
            "rendezvous_recovered": True,
            "restart_recorded": True,
        },
        "final_status": {
            "overall_state": "completed",
            "global_step": 8,
            "runtime": {
                "committed_steps": list(range(1, 9)),
                "optimizer_commit_count": 8,
                "checkpoint_signatures_required": True,
                "checkpoint_tensor_validation_required": True,
            },
        },
        "export_cli_report": {
            "ok": True,
            "standard_peft_format": True,
            "adapter_tensor_count": 392,
            "layer_indexes": list(range(28)),
        },
        "cleanup_verified": True,
        "cleanup": {
            "all_kernels_deleted": True,
            "service_stopped": True,
            "tunnel_stopped": True,
            "rendezvous_payloads_removed": True,
            "private_runtime_removed": True,
            "live_resources_left_running": False,
        },
        "repack": {
            "runtime_measurements_changed": False,
            "generation_cleanup_verified": True,
            "post_cleanup_account_audit_verified": True,
            "selected_account_active_kernel_count": 0,
        },
        "credentials_public": False,
        "credential_paths_public": False,
        "coordinator_url_public": False,
        "raw_training_text_public": False,
        "token_ids_public": False,
        "activation_values_public": False,
        "gradient_values_public": False,
        "checkpoint_tensor_values_public": False,
        "adapter_tensor_values_public": False,
        "private_paths_public": False,
        "public_safety_errors": [],
        "blockers": [],
    }
    report["content_hash"] = _stable_hash(report)
    return report


def _write(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_elastic_beta_checker_accepts_complete_product_evidence(tmp_path) -> None:
    path = tmp_path / "report.json"
    _write(path, _complete_report())
    result = check(path, require_ready=True)
    assert result["ok"] is True
    assert result["elastic_training_beta_ready"] is True
    assert result["errors"] == []


def test_elastic_beta_checker_rejects_missing_restore_and_cleanup(tmp_path) -> None:
    report = _complete_report()
    report["replacement_generation"]["worker_reports"][0]["worker"][
        "central_checkpoint_restore_verified"
    ] = False
    report["cleanup"]["all_kernels_deleted"] = False
    report["content_hash"] = _stable_hash(
        {key: value for key, value in report.items() if key != "content_hash"}
    )
    path = tmp_path / "invalid.json"
    _write(path, report)
    result = check(path, require_ready=True)
    assert result["ok"] is False
    assert "elastic_beta_replacement_restore_missing" in result["errors"]
    assert "elastic_beta_cleanup_invalid" in result["errors"]


def test_repack_uses_generation_deletes_and_post_cleanup_audit(tmp_path) -> None:
    source = _complete_report()
    source.pop("repack")
    source["ok"] = False
    source["elastic_training_beta_ready"] = False
    source["cleanup_verified"] = False
    source["cleanup"]["all_kernels_deleted"] = False
    source["cleanup"]["live_resources_left_running"] = True
    source["blockers"] = ["elastic_training_beta_cleanup_incomplete"]
    source.pop("content_hash")
    source_path = tmp_path / "source.json"
    _write(source_path, source)
    owner_hash = "sha256:" + "a" * 64
    source["selected_account"] = {"owner_hash": owner_hash}
    _write(source_path, source)
    audit_path = tmp_path / "audit.json"
    _write(
        audit_path,
        {
            "account_preflight": [
                {
                    "owner_hash": owner_hash,
                    "authenticated": True,
                    "active_kernel_count": 0,
                }
            ]
        },
    )
    packed = pack(source_path, audit_path, tmp_path / "packed")
    assert packed["ok"] is True
    assert packed["cleanup_verified"] is True
    assert packed["cleanup"]["all_kernels_deleted"] is True
    assert packed["repack"]["runtime_measurements_changed"] is False
    assert packed["blockers"] == []
