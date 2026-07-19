from crowdtensor.community_security import scan_public_value
from crowdtensor.community_live_training import CommunityLiveCoordinator
from crowdtensor.model_adapter import stable_hash
from scripts.mistral_kaggle_live_check import check_report
from scripts.mistral_kaggle_live_probe import _checkpoint_summary, _success_report


def test_checkpoint_summary_requires_both_mistral_gate_steps(tmp_path) -> None:
    coordinator = CommunityLiveCoordinator(
        tmp_path / "state.json",
        run_id="mistral-summary",
        target_steps=8,
        checkpoint_steps=(4, 8),
    )
    coordinator.state["events"].extend(
        [
            {"operation": "checkpoint_committed", "role": role, "step": step}
            for step in (4, 8)
            for role in ("stage0", "stage1")
        ]
    )
    coordinator.state["checkpoints"] = {
        role: {"step": 8} for role in ("stage0", "stage1")
    }
    summary = _checkpoint_summary(coordinator)
    assert summary["steps_by_role"] == {"stage0": [4, 8], "stage1": [4, 8]}
    assert summary["final_stage_checkpoints_present"] is True


def test_failure_report_cannot_pass_strict_checker() -> None:
    report = {
        "schema": "crowdtensor_mistral_kaggle_heterogeneous_live_v1",
        "ok": False,
        "live_run_performed": False,
        "blockers": ["mistral_kaggle_gpu_quota_unavailable"],
    }
    assert check_report(report)["ok"] is False


def test_success_report_builder_satisfies_strict_checker() -> None:
    hash_a = "sha256:" + "a" * 64
    hash_b = "sha256:" + "b" * 64

    class Coordinator:
        state = {
            "events": [
                {"operation": "checkpoint_committed", "role": role, "step": step}
                for step in (4, 8)
                for role in ("stage0", "stage1")
            ],
            "ledger": [
                {
                    "activation_hash": hash_a,
                    "gradient_hash": hash_b,
                    "activation_bytes": 10,
                    "gradient_bytes": 10,
                }
                for _ in range(8)
            ],
        }

        @staticmethod
        def private_checkpoint(role):
            return {"step": 8}

        @staticmethod
        def public_status():
            return {
                "model_adapter_id": "mistral_lora_v1",
                "model_id": "Locutusque/TinyMistral-248M-v2",
                "target_steps": 8,
                "committed_step_ids": list(range(1, 9)),
                "strictly_contiguous_steps": True,
                "finite_losses": True,
                "ledger_entry_count": 8,
                "completed": True,
            }

    def worker(role, backend, identity, *, steps=None, **extra):
        selected_steps = steps or list(range(1, 9))
        return {
            "ok": True,
            "role": role,
            "backend": backend,
            "worker_id_hash": identity,
            "model_adapter_id": "mistral_lora_v1",
            "model_id": "Locutusque/TinyMistral-248M-v2",
            "model_revision": "0f57b17cb317bb322c7c1466b669c681f80c058f",
            "real_model_weights_loaded": True,
            "adapter_updated": True,
            "stage_runtime": {
                "family": "mistral",
                "architecture": "MistralForCausalLM",
            },
            "step_events": [
                {"step": step, "phase": "commit", "accepted": True}
                for step in selected_steps
            ],
            "last_committed_step": max(selected_steps),
            **extra,
        }

    kernel_common = {
        "ok": True,
        "node_scope": "Kaggle logical multi-node",
        "core_wheel_hash_verified": True,
        "adapter_wheel_hash_verified": True,
        "both_wheels_installed_in_fresh_environment": True,
        "adapter_plugin_discovered": True,
        "workspace_import_used": False,
    }
    kernels = [
        {
            **kernel_common,
            "kernel_role": "stage0",
            "backend": "cuda",
            "cuda_device_count": 2,
            "worker_replacement_verified": True,
            "worker_reports": [
                worker("stage0", "cuda", hash_a, steps=list(range(1, 5))),
                worker(
                    "stage0",
                    "cuda",
                    hash_b,
                    steps=list(range(5, 9)),
                    checkpoint_restored=True,
                    restored_checkpoint_step=4,
                    optimizer_state_restored=True,
                ),
            ],
        },
        {
            **kernel_common,
            "kernel_role": "stage1",
            "backend": "cpu",
            "cuda_device_count": 0,
            "worker_reports": [
                worker("stage1", "cpu", "sha256:" + "c" * 64)
            ],
        },
    ]
    report = _success_report(
        coordinator=Coordinator(),
        kernel_reports=kernels,
        package_report={"ok": True, "public_artifact_safe": True},
        host_plugin={
            "registration_kind": "entry_point_plugin",
            "distribution_name": "crowdtensor-mistral-adapter",
            "distribution_version": "0.1.0b1",
            "core_wheel_hash": hash_a,
            "adapter_wheel_hash": hash_b,
        },
        exported={
            "export": {
                "adapter_id": "mistral_lora_v1",
                "standard_peft_format": True,
                "stage_adapter_key_overlap": False,
                "adapter_tensor_count": 4,
                "adapter_file_hash": hash_a,
            },
            "reload": {
                "adapter_id": "mistral_lora_v1",
                "independent_process_reload": True,
                "adapter_reload_verified": True,
                "reload_logits_finite": True,
            },
        },
        attempt_number=1,
        duration_seconds=10.0,
    )
    report["cleanup"] = {
        "all_remote_kernels_deleted": True,
        "coordinator_stopped": True,
        "tunnel_stopped": True,
        "private_runtime_removed": True,
        "live_resources_left_running": False,
    }
    report["cleanup_verified"] = True
    report["public_safety"] = scan_public_value(report)
    assert report["public_safety"]["ok"] is True
    report["content_hash"] = stable_hash(report)
    assert check_report(report)["ok"] is True
