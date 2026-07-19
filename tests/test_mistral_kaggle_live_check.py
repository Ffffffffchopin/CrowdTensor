import copy

from crowdtensor.model_adapter import stable_hash
from scripts.mistral_kaggle_live_check import check_report


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
MODEL_ID = "Locutusque/TinyMistral-248M-v2"
REVISION = "0f57b17cb317bb322c7c1466b669c681f80c058f"


def worker(role, backend, identity, *, steps=None):
    return {
        "ok": True,
        "role": role,
        "backend": backend,
        "worker_id_hash": identity,
        "model_adapter_id": "mistral_lora_v1",
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "real_model_weights_loaded": True,
        "adapter_updated": True,
        "stage_runtime": {
            "family": "mistral",
            "architecture": "MistralForCausalLM",
        },
        "step_events": [
            {"step": step, "phase": "commit", "accepted": True}
            for step in (steps or list(range(1, 9)))
        ],
        "last_committed_step": max(steps or list(range(1, 9))),
    }


def valid_report():
    stage0_old = worker("stage0", "cuda", HASH_A, steps=list(range(1, 5)))
    stage0_new = {
        **worker("stage0", "cuda", HASH_B, steps=list(range(5, 9))),
        "checkpoint_restored": True,
        "restored_checkpoint_step": 4,
        "optimizer_state_restored": True,
    }
    stage1 = worker("stage1", "cpu", "sha256:" + "c" * 64)
    value = {
        "schema": "crowdtensor_mistral_kaggle_heterogeneous_live_v1",
        "ok": True,
        "live_run_performed": True,
        "node_scope": "Kaggle logical multi-node",
        "accepted_providers": ["kaggle_cpu", "kaggle_cuda"],
        "model": {
            "adapter_id": "mistral_lora_v1",
            "family": "mistral",
            "architecture": "MistralForCausalLM",
            "model_id": MODEL_ID,
            "model_revision": REVISION,
            "license": "apache-2.0",
            "parameter_count": 248_024_064,
            "real_trained_weights": True,
            "random_or_synthetic_weights_used": False,
        },
        "plugin_installation": {
            "core_wheel_hash_verified": True,
            "adapter_wheel_hash_verified": True,
            "both_wheels_installed_in_fresh_environment": True,
            "entry_point_plugin_discovered": True,
            "workspace_import_not_used": True,
            "registration_kind": "entry_point_plugin",
        },
        "final_status": {
            "model_adapter_id": "mistral_lora_v1",
            "model_id": MODEL_ID,
            "target_steps": 8,
            "committed_step_ids": list(range(1, 9)),
            "strictly_contiguous_steps": True,
            "finite_losses": True,
            "ledger_entry_count": 8,
            "completed": True,
        },
        "kernel_evidence": [
            {
                "ok": True,
                "kernel_role": "stage0",
                "backend": "cuda",
                "node_scope": "Kaggle logical multi-node",
                "both_wheels_installed_in_fresh_environment": True,
                "adapter_plugin_discovered": True,
                "cuda_device_count": 2,
                "worker_reports": [stage0_old, stage0_new],
            },
            {
                "ok": True,
                "kernel_role": "stage1",
                "backend": "cpu",
                "node_scope": "Kaggle logical multi-node",
                "both_wheels_installed_in_fresh_environment": True,
                "adapter_plugin_discovered": True,
                "cuda_device_count": 0,
                "worker_reports": [stage1],
            },
        ],
        "gpu_worker_replacement": {
            "verified": True,
            "after_step": 4,
            "old_worker_id_hash": HASH_A,
            "new_worker_id_hash": HASH_B,
            "checkpoint_restored": True,
            "restored_checkpoint_step": 4,
            "optimizer_state_restored": True,
        },
        "checkpoints": {
            "steps_by_role": {"stage0": [4, 8], "stage1": [4, 8]},
            "adapter_state_saved": True,
            "adam_state_saved": True,
            "hash_integrity_verified": True,
            "final_stage_checkpoints_present": True,
        },
        "cross_device_transfer": {
            "activation_gradient_transfer_verified": True,
            "forward_activation_count": 8,
            "backward_gradient_count": 8,
            "safetensors_serialization": True,
            "all_payload_hashes_verified": True,
            "payload_values_public": False,
        },
        "export": {
            "adapter_id": "mistral_lora_v1",
            "standard_peft_format": True,
            "stage_adapter_key_overlap": False,
            "adapter_tensor_count": 1,
            "adapter_file_hash": HASH_A,
        },
        "reload": {
            "adapter_id": "mistral_lora_v1",
            "independent_process_reload": True,
            "adapter_reload_verified": True,
            "reload_logits_finite": True,
        },
        "cleanup": {
            "all_remote_kernels_deleted": True,
            "coordinator_stopped": True,
            "tunnel_stopped": True,
            "private_runtime_removed": True,
            "live_resources_left_running": False,
        },
        "cleanup_verified": True,
        "attempt_ledger": {
            "schema": "crowdtensor_mistral_kaggle_live_gate_ledger_v1",
            "attempt": 1,
            "maximum_attempts": 2,
            "community_maturity_ledger_modified": False,
        },
        "unsupported_claims": {
            "arbitrary_mistral_models_supported": False,
            "full_parameter_training_verified": False,
            "mistral_7b_live_verified": False,
            "physical_multi_machine_verified": False,
            "production_sla_verified": False,
        },
        "public_artifact_safe": True,
    }
    value["content_hash"] = stable_hash(value)
    return value


def test_strict_checker_accepts_complete_live_evidence() -> None:
    result = check_report(valid_report())
    assert result["ok"] is True
    assert result["mistral_live_verified"] is True


def test_strict_checker_rejects_missing_replacement_or_cleanup() -> None:
    value = valid_report()
    value["gpu_worker_replacement"]["verified"] = False
    value["cleanup"]["live_resources_left_running"] = True
    value["content_hash"] = stable_hash(
        {key: item for key, item in value.items() if key != "content_hash"}
    )
    result = check_report(value)
    assert result["ok"] is False
    assert "mistral_live_gpu_worker_replacement_invalid" in result["errors"]
    assert "mistral_live_cleanup_invalid" in result["errors"]


def test_strict_checker_rejects_model_overclaim() -> None:
    value = copy.deepcopy(valid_report())
    value["unsupported_claims"]["mistral_7b_live_verified"] = True
    value["content_hash"] = stable_hash(
        {key: item for key, item in value.items() if key != "content_hash"}
    )
    assert "mistral_live_unsupported_claim_invalid" in check_report(value)["errors"]
