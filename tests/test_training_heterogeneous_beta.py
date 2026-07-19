import json

from scripts.training_heterogeneous_beta_check import check
from scripts.training_heterogeneous_beta_pack import LIVE_SCHEMA, pack


def digest(seed: str) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(seed.encode()).hexdigest()


def live_report() -> dict:
    old = digest("gpu-old")
    replacement = digest("gpu-replacement")
    stable = [digest(f"gpu-stable-{index}") for index in range(3)]
    cpu = digest("cpu")

    def assignment(stage: int, miner: str, generation: int) -> dict:
        return {
            "stage_id": stage,
            "miner_id_hash": miner,
            "device_type": "cpu" if stage == 4 else "cuda",
            "placement_generation": generation,
            "resource_fit_verified": True,
        }

    def worker(
        role: str,
        miner: str,
        steps: list[int],
        *,
        gpu_count: int,
        restore_count: int = 0,
    ) -> dict:
        return {
            "role": role,
            "miner_id_hash": miner,
            "kernel_ref_hash": digest(f"kernel:{role}"),
            "device_policy": "cuda" if gpu_count else "cpu",
            "gpu_count": gpu_count,
            "single_gpu_miner": gpu_count == 1,
            "pure_cpu_miner": gpu_count == 0,
            "committed_steps": steps,
            "steps_completed": len(steps),
            "central_checkpoint_restore_count": restore_count,
            "positive_lora_gradient_norms": True,
            "optimizer_and_scheduler_steps_applied": True,
            "worker_report_hash": digest(f"report:{role}"),
        }

    return {
        "schema": LIVE_SCHEMA,
        "live_run_performed": True,
        "execution_provider": "kaggle",
        "model_id": "Qwen/Qwen2.5-7B",
        "model_revision": "d149729398750b98c0af14eb82c78cfe92750796",
        "parameter_count": 7_615_616_000,
        "training_manifest_hash": digest("manifest"),
        "stage_count": 5,
        "target_steps": 6,
        "job_id_hash": digest("job"),
        "run_id_hash": digest("run"),
        "same_job_training_verified": True,
        "kernel_topology": {
            "gpu_kernel_count": 2,
            "cpu_kernel_count": 1,
            "physical_gpu_count": 4,
            "initial_single_gpu_miner_count": 4,
            "pure_cpu_miner_count": 1,
        },
        "placement_evidence": {
            "initial_generation": 5,
            "replacement_generation": 7,
            "initial_assignments": [
                assignment(0, old, 5),
                assignment(1, stable[0], 5),
                assignment(2, stable[1], 5),
                assignment(3, stable[2], 5),
                assignment(4, cpu, 5),
            ],
            "replacement_assignments": [
                assignment(0, replacement, 7),
                assignment(1, stable[0], 7),
                assignment(2, stable[1], 7),
                assignment(3, stable[2], 7),
                assignment(4, cpu, 7),
            ],
            "auditable_scores_present": True,
            "memory_reserve_enforced": True,
            "performance_and_network_cost_used": True,
        },
        "worker_evidence": [
            worker("gpu_old", old, [1, 2, 3], gpu_count=1),
            worker(
                "gpu_replacement",
                replacement,
                [4, 5, 6],
                gpu_count=1,
                restore_count=1,
            ),
            *[
                worker(
                    "gpu_stable",
                    miner,
                    [1, 2, 3, 4, 5, 6],
                    gpu_count=1,
                    restore_count=1,
                )
                for miner in stable
            ],
            worker(
                "cpu",
                cpu,
                [1, 2, 3, 4, 5, 6],
                gpu_count=0,
                restore_count=1,
            ),
        ],
        "replacement_evidence": {
            "old_miner_id_hash": old,
            "replacement_miner_id_hash": replacement,
            "removed_after_committed_step": 3,
            "trainable_stage_removed": True,
            "pause_or_incomplete_placement_observed": True,
            "rebalance_verified": True,
            "replacement_checkpoint_restore_verified": True,
            "replacement_steps_completed": 3,
        },
        "training_evidence": {
            "committed_steps": [1, 2, 3, 4, 5, 6],
            "committed_steps_contiguous": True,
            "optimizer_commit_count": 6,
            "duplicate_committed_steps": [],
            "missing_committed_steps": [],
            "atomic_global_commit_verified": True,
            "checkpoint_components": [
                "adapter",
                "optimizer",
                "lr_scheduler",
                "grad_scaler",
                "rng",
                "manifest",
            ],
            "finite_loss_count": 6,
            "non_finite_loss_count": 0,
            "positive_gradient_stage_ids": [0, 1, 2, 3, 4],
            "changed_lora_stage_ids": [0, 1, 2, 3, 4],
        },
        "tensor_transport_evidence": {
            "format": "safetensors",
            "pickle_deserialization_allowed": False,
            "forward_activation_count": 24,
            "backward_gradient_count": 24,
            "cuda_to_cpu_activation_count": 6,
            "cpu_to_cuda_gradient_count": 6,
            "all_checksums_verified": True,
            "chunking_verified": True,
            "finite_retry_verified": True,
            "idempotent_delivery_verified": True,
            "stale_generation_rejected": True,
            "duplicate_message_deduplicated": True,
        },
        "export_evidence": {
            "standard_peft_format": True,
            "all_five_stages_present": True,
            "adapter_reload_verified": True,
            "forward_inference_verified": True,
            "finite_logits_verified": True,
            "model_binding_verified": True,
            "adapter_file_hash": digest("adapter"),
        },
        "regression_summary": {
            "passed": 76,
            "failed": 0,
            "legacy_training_regression_included": True,
            "heterogeneous_training_tests_included": True,
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


def test_pack_and_strict_checker_accept_complete_live_gate(tmp_path) -> None:
    source = tmp_path / "live.json"
    write(source, live_report())

    packed = pack(source, tmp_path / "canonical")
    result = check(
        tmp_path / "canonical" / "training_heterogeneous_beta.json",
        require_ready=True,
    )

    assert packed["heterogeneous_training_beta_ready"] is True
    assert all(packed["acceptance_gates"].values())
    assert result["ok"] is True
    assert result["heterogeneous_training_beta_ready"] is True


def test_valid_blocker_passes_default_and_fails_strict(tmp_path) -> None:
    value = live_report()
    value["live_run_performed"] = False
    value["blockers"] = ["kaggle_gpu_allocation_unavailable"]
    source = tmp_path / "live.json"
    write(source, value)
    packed = pack(source, tmp_path / "canonical")
    report = tmp_path / "canonical" / "training_heterogeneous_beta.json"

    default = check(report)
    strict = check(report, require_ready=True)

    assert packed["heterogeneous_training_beta_ready"] is False
    assert default["ok"] is True
    assert strict["ok"] is False
    assert "heterogeneous_training_beta_not_ready" in strict["errors"]


def test_checker_rejects_forged_ready_and_private_path(tmp_path) -> None:
    value = live_report()
    value["tensor_transport_evidence"]["cpu_to_cuda_gradient_count"] = 0
    value["diagnostic"] = "/root/private/credential.json"
    source = tmp_path / "live.json"
    write(source, value)
    pack(source, tmp_path / "canonical")
    report_path = tmp_path / "canonical" / "training_heterogeneous_beta.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["heterogeneous_training_beta_ready"] = True
    report.pop("content_hash")
    from scripts.training_heterogeneous_beta_check import _stable_hash

    report["content_hash"] = _stable_hash(report)
    write(report_path, report)

    result = check(report_path)

    assert result["ok"] is False
    assert "heterogeneous_beta_public_safety_scan_failed" in result["errors"]
    assert "heterogeneous_beta_ready_claim_invalid" in result["errors"]
