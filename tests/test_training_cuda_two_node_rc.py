from __future__ import annotations

import copy
import json

import pytest

from crowdtensor.training_allocation_budget import allocation_budget_summary
from scripts.training_cuda_two_node_probe import _build_embedded_single_gate
from scripts.training_cuda_two_node_rc_check import REQUIRED_REJECTION_CODES, SCHEMA, check
from scripts.training_cuda_two_node_rc_pack import pack


def _rejections() -> dict:
    return {
        "ok": True,
        "public_artifact_safe": True,
        "private_tensors_removed": True,
        "checks": {
            name: {"code": code, "rejected_as_expected": True}
            for name, code in REQUIRED_REJECTION_CODES.items()
        },
    }


def _single() -> dict:
    def pipeline(*, resumed: bool) -> dict:
        return {
            "total_steps": 4,
            "real_cuda_forward": True,
            "real_cuda_backward": True,
            "real_activation_transport": True,
            "real_backward_gradient_transport": True,
            "loss_reduced": True,
            "base_weights_frozen": True,
            "positive_lora_gradient_norms": True,
            "positive_cuda_memory": True,
            "distinct_stage_pids": True,
            "distinct_cuda_devices": True,
            "no_stage_loaded_full_model": True,
            "stage_records": {
                str(stage_id): [
                    {
                        "step": step,
                        "pid": 100 + stage_id + (10 if resumed and stage_id == 1 and step >= 2 else 0),
                        "cuda_device": f"cuda:{stage_id}",
                        "forward_hash": f"sha256:{stage_id}-forward-{step}",
                        "backward_gradient_hash": f"sha256:{stage_id}-gradient-{step}",
                        "checkpoint_hash": f"sha256:{stage_id}-checkpoint-{step}",
                        "lora_gradient_norm": 1.0,
                        "peak_allocated_bytes": 1024,
                        "peak_reserved_bytes": 2048,
                    }
                    for step in range(4)
                ]
                for stage_id in (0, 1)
            },
            "final_checkpoint": {
                "complete": True,
                "stage_count": 2,
                "global_step": 4,
                "outer_step": 4,
                "content_hash": "sha256:global-checkpoint",
                "stages": [
                    {
                        "content_hash": f"sha256:stage{stage_id}-checkpoint",
                        "grad_scaler_state_present": True,
                        "cuda_placement": f"cuda:{stage_id}",
                    }
                    for stage_id in (0, 1)
                ],
            },
            "cleanup": {"all_worker_processes_stopped": True},
            "interruption": (
                {
                    "performed": True,
                    "checkpoint_loaded": True,
                    "worker_restarted": True,
                    "old_pid": 101,
                    "new_pid": 111,
                    "resumed_optimizer_step": 2,
                }
                if resumed
                else {"performed": False}
            ),
        }

    baseline = pipeline(resumed=False)
    resumed = pipeline(resumed=True)
    return {
        "ok": True,
        "public_artifact_safe": True,
        "single_kernel_t4x2_verified": True,
        "cleanup": {
            "kernel_deleted": True,
            "private_package_removed": True,
            "checkpoint_preserved": True,
            "private_cleanup_state_removed": True,
        },
        "checkpoint_bundle": {
            "preserved": True,
            "worker_hash_match": True,
            "file_hash": "sha256:single-checkpoint",
            "byte_count": 1024,
        },
        "worker_report": {
            "ok": True,
            "kaggle_kernel": True,
            "gpu_live_verified": True,
            "cuda_device_count": 2,
            "two_distinct_processes": True,
            "two_distinct_cuda_devices": True,
            "real_activation_transport": True,
            "real_backward_gradient_transport": True,
            "real_cuda_backward": True,
            "no_stage_loaded_full_model": True,
            "base_weights_frozen": True,
            "positive_lora_gradient_norms": True,
            "positive_cuda_memory": True,
            "loss_reduced": True,
            "controlled_stage_restart": True,
            "checkpoint_resume_verified": True,
            "checkpoint_bundle": {
                "present": True,
                "file_hash": "sha256:single-checkpoint",
                "file_count": 12,
                "contains_baseline_and_resumed_checkpoints": True,
            },
            "resume_equivalence": {"checkpoint_resume_verified": True},
            "baseline": baseline,
            "resumed": resumed,
        },
    }


def _worker(role: str) -> dict:
    shard_index = 0 if role == "stage0" else 1
    return {
        "ok": True,
        "public_artifact_safe": True,
        "role": role,
        "kaggle_kernel": True,
        "gpu_live_verified": True,
        "cuda_device_count": 2,
        "pipeline": {
            "role": role,
            "steps_completed": 4,
            "real_cuda_forward": True,
            "real_cuda_backward": True,
            "real_activation_transport": True,
            "real_backward_gradient_transport": True,
            "positive_lora_gradient_norms": True,
            "base_weights_frozen": True,
            "no_full_model_loaded": True,
            "checkpoint_hash": f"sha256:{role}-pipeline-checkpoint",
            "checkpoint_grad_scaler_state_present": True,
        },
        "miner": {
            "base_model_version": 1,
            "adapter_version": 0,
            "model_manifest_hash": "sha256:model-manifest",
            "base_model_hash": "sha256:base-model",
            "base_adapter_hash": "sha256:base-adapter",
            "dataset_shard_index": shard_index,
            "coordinator_accepted": True,
            "base_weights_frozen": True,
            "only_lora_trainable": True,
            "real_backward": True,
            "loss_reduced": True,
            "adapter_delta_tensor_count": 2,
            "adapter_delta_file_hash": f"sha256:{role}-delta",
            "adapter_delta_tensor_specs_hash": f"sha256:{role}-specs",
            "adapter_delta_format": "named_safetensors",
            "adapter_delta_named_tensors": True,
            "optimizer_steps": 4,
            "tokens_seen": 128,
            "elapsed_seconds": 1.5,
            "checkpoint_hash": f"sha256:{role}-miner-checkpoint",
            "peak_allocated_bytes": 1024,
            "peak_reserved_bytes": 2048,
            "runtime": {
                "cuda_used": True,
                "gpu_live_verified": True,
                "device_index": 0,
                "device_name_hash": f"sha256:{role}-device",
            },
        },
        "evaluation": {
            "standard_peft_cuda_load": True,
            "adapter_changes_logits": True,
            "validation_loss_reduced": True,
        },
        "global_adapter": {"adapter_version": 1, "outer_step": 1},
        "checkpoint_bundle": {
            "present": True,
            "file_hash": f"sha256:{role}-checkpoint-bundle",
            "file_count": 8,
            "contains_pipeline_and_miner_checkpoints": True,
        },
        "cleanup": {"private_runtime_removed": True},
    }


def _two_node() -> dict:
    payloads = [
        {
            "kind": kind,
            "step": step,
            "payload_hash": f"sha256:{kind}-{step}",
            "byte_count": 128,
            "shape": [2, 12, 32],
            "dtype": "float16",
        }
        for step in range(4)
        for kind in ("activation", "gradient")
    ]
    return {
        "ok": True,
        "public_artifact_safe": True,
        "two_node_cuda_verified": True,
        "same_authorized_account": True,
        "multi_account_used": False,
        "tpu_used": False,
        "requested_kernel_count": 2,
        "used_gpu_per_kernel": 1,
        "all_four_t4_used_claimed": False,
        "max_observed_running_kernel_count": 2,
        "kernel_ref_hashes": ["sha256:kernel-stage0", "sha256:kernel-stage1"],
        "worker_reports": [_worker("stage0"), _worker("stage1")],
        "checkpoint_bundles": [
            {
                "role": role,
                "preserved": True,
                "worker_hash_match": True,
                "file_hash": f"sha256:{role}-checkpoint-bundle",
                "byte_count": 1024,
                "contains_pipeline_and_miner_checkpoints": True,
            }
            for role in ("stage0", "stage1")
        ],
        "rendezvous": {
            "registered_roles": ["stage0", "stage1"],
            "registrations": [
                {
                    "role": role,
                    "pid": 100 + index,
                    "cuda_live": True,
                    "cuda_device_index": 0,
                    "cuda_device_name_hash": f"sha256:{role}-device",
                }
                for index, role in enumerate(("stage0", "stage1"))
            ],
            "completions": [{"role": "stage0"}, {"role": "stage1"}],
            "payloads": payloads,
        },
        "training_state": {
            "round_status": "aggregated",
            "adapter_version": 1,
            "outer_step": 1,
            "accepted_result_count": 2,
            "accepted_shard_indexes": [0, 1],
            "dense_diloco_aggregation": True,
        },
        "error_feedback": {
            "error_feedback": True,
            "dense_reconstruction_with_residual_verified": True,
        },
        "evaluation_export": {
            "validation_loss_reduced": True,
            "cpu_adapter_changes_logits": True,
            "cpu_cuda_logits_close": True,
            "standard_peft_cpu_load": True,
            "standard_peft_cuda_load": True,
        },
        "cleanup": {
            "kernels_deleted": True,
            "private_packages_removed": True,
            "coordinator_stopped": True,
            "tunnel_stopped": True,
            "private_runtime_removed": True,
            "checkpoint_bundles_preserved": True,
            "private_cleanup_state_removed": True,
        },
        "rendezvous_cleanup": {"private_payloads_removed": True},
    }


def _two_node_with_embedded_single() -> dict:
    two_node = _two_node()
    two_node["attempt"] = 3
    two_node["kernel_ref_hashes_by_role"] = {
        "stage0": "sha256:kernel-stage0",
        "stage1": "sha256:kernel-stage1",
    }
    stage0_worker = two_node["worker_reports"][0]
    embedded_worker = copy.deepcopy(_single()["worker_report"])
    embedded_worker.update(
        {
            "single_kernel_t4x2_verified": True,
            "source_role": "stage0",
            "execution_order": "before_cross_node_stage0",
            "coallocated_with_two_node_attempt": True,
        }
    )
    embedded_worker["checkpoint_bundle"].update(
        {
            "file_hash": "sha256:stage0-checkpoint-bundle",
            "contains_baseline_and_resumed_checkpoints": True,
        }
    )
    stage0_worker["embedded_single_kernel_gate"] = embedded_worker
    stage0_worker["embedded_single_kernel_gate_verified"] = True
    two_node["checkpoint_bundles"][0][
        "contains_baseline_and_resumed_checkpoints"
    ] = True
    gate = _build_embedded_single_gate(
        stage0_worker=stage0_worker,
        stage0_bundle=two_node["checkpoint_bundles"][0],
        stage0_kernel_ref_hash="sha256:kernel-stage0",
        attempt=3,
    )
    gate["cleanup"] = {
        "kernel_deleted": True,
        "private_package_removed": True,
        "checkpoint_preserved": True,
        "private_cleanup_state_removed": True,
    }
    gate["ok"] = True
    two_node["embedded_single_kernel_gate"] = gate
    return two_node


def _report(*, ready: bool) -> dict:
    return {
        "schema": SCHEMA,
        "training_cuda_two_node_rc_ready": ready,
        "goal_achieved": ready,
        "gpu_success_claimed": ready,
        "cpu_foundation_baseline": {"goal_achieved": True, "training_foundation_rc_ready": True},
        "runtime_contracts": {
            "cuda_lora_runtime_implemented": True,
            "cuda_stage_runtime_implemented": True,
            "fp16_autocast_supported": True,
            "grad_scaler_supported": True,
            "gradient_clipping_supported": True,
            "cuda_oom_classification_supported": True,
            "cpu_checkpoint_delta_compatibility_preserved": True,
            "authenticated_private_rendezvous_implemented": True,
            "remote_delta_materialization_implemented": True,
            "checkpoint_bundle_preservation_supported": True,
            "crash_recoverable_private_cleanup_ledger_supported": True,
        },
        "allocation_attempts": {"single_kernel_attempts": [], "two_node_attempts": []},
        "single_kernel_gate": _single() if ready else {},
        "two_node_gate": _two_node() if ready else {},
        "rejection_matrix": _rejections(),
        "test_summary": {
            "ok": True,
            "cuda_training_tests_passed": True,
            "cpu_training_regressions_passed": True,
            "state_store_miner_coordinator_regressions_passed": True,
        },
        "cleanup_summary": {
            "all_kaggle_kernels_deleted": True,
            "all_private_packages_removed": True,
            "all_local_runtime_stopped": True,
            "live_resources_left_running": False,
        },
        "public_artifact_safe": True,
    }


def _check(tmp_path, report: dict, *, strict: bool = True) -> dict:
    path = tmp_path / "rc.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return check(path, require_ready=strict)


def test_checker_accepts_complete_live_cuda_evidence(tmp_path) -> None:
    result = _check(tmp_path, _report(ready=True))
    assert result["ok"] is True
    assert result["goal_achieved"] is True


def test_checker_accepts_record_level_embedded_single_gate_from_live_stage0(tmp_path) -> None:
    report = _report(ready=True)
    two_node = _two_node_with_embedded_single()
    report["two_node_gate"] = two_node
    report["single_kernel_gate"] = two_node["embedded_single_kernel_gate"]
    report["single_kernel_gate_source"] = "two_node_stage0_embedded"
    result = _check(tmp_path, report)
    assert result["ok"] is True
    assert result["single_kernel_gate_verified"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["single_kernel_gate"].update(
            source_kernel_ref_hash="sha256:unbound-kernel"
        ),
        lambda value: value["single_kernel_gate"].update(
            execution_order="after_cross_node_stage0"
        ),
        lambda value: value["two_node_gate"]["worker_reports"][0][
            "embedded_single_kernel_gate"
        ]["baseline"]["stage_records"]["0"][0].update(forward_hash="sha256:changed"),
    ],
)
def test_checker_rejects_unbound_or_metadata_only_embedded_single_gate(
    tmp_path, mutation
) -> None:
    report = _report(ready=True)
    two_node = _two_node_with_embedded_single()
    report["two_node_gate"] = two_node
    report["single_kernel_gate"] = copy.deepcopy(two_node["embedded_single_kernel_gate"])
    report["single_kernel_gate_source"] = "two_node_stage0_embedded"
    mutation(report)
    result = _check(tmp_path, report)
    assert result["ok"] is False
    assert result["single_kernel_gate_verified"] is False


def test_rc_pack_selects_bound_embedded_gate_and_preserves_failed_history(tmp_path) -> None:
    two_node = _two_node_with_embedded_single()
    standalone = {
        "ok": False,
        "single_kernel_t4x2_verified": False,
        "blockers": ["historical_single_failure"],
        "cleanup": {
            "kernel_deleted": True,
            "private_package_removed": True,
            "checkpoint_preserved": True,
            "private_cleanup_state_removed": True,
        },
        "public_artifact_safe": True,
    }
    sources = {
        "cpu": {
            "goal_achieved": True,
            "training_foundation_rc_ready": True,
            "backend": "cpu",
        },
        "single": standalone,
        "two": two_node,
        "ledger": {"single_kernel_attempts": [], "two_node_attempts": []},
        "rejections": _rejections(),
        "tests": {
            "ok": True,
            "cuda_training_tests_passed": True,
            "cpu_training_regressions_passed": True,
            "state_store_miner_coordinator_regressions_passed": True,
        },
    }
    paths = {}
    for name, value in sources.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    result = pack(
        output_dir=tmp_path / "rc",
        cpu_baseline=paths["cpu"],
        single_attempt_reports=[paths["single"]],
        two_node_attempt_reports=[paths["two"]],
        attempt_ledger=paths["ledger"],
        rejection_matrix=paths["rejections"],
        test_summary=paths["tests"],
    )
    assert result["training_cuda_two_node_rc_ready"] is True
    assert result["single_kernel_gate_source"] == "two_node_stage0_embedded"
    assert result["single_kernel_attempt_history"] == [standalone]
    assert result["single_kernel_gate"]["ok"] is True


def test_checker_accepts_honest_blocker_only_without_require_ready(tmp_path) -> None:
    report = _report(ready=False)
    assert _check(tmp_path, report, strict=False)["ok"] is True
    strict = _check(tmp_path, report, strict=True)
    assert strict["ok"] is False
    assert "single_kernel_t4x2_live_gate_missing" in strict["errors"]
    assert "two_kernel_cross_machine_live_gate_missing" in strict["errors"]


def test_checker_accepts_non_live_authenticated_route_preflight(tmp_path) -> None:
    report = _report(ready=False)
    report["coordinator_route_preflight"] = {
        "schema": "crowdtensor_cuda_training_coordinator_route_preflight_summary_v1",
        "verified": True,
        "authenticated_status_verified": True,
        "miner_auth_required_verified": True,
        "run_id_hash_verified": True,
        "stable_successes_observed": 2,
        "stable_successes_required": 2,
        "allocation_started": False,
        "kernel_push_attempted": False,
        "live_gate_claimed": False,
        "cleanup_verified": True,
        "url_public": False,
        "credentials_public": False,
        "public_artifact_safe": True,
    }
    result = _check(tmp_path, report, strict=False)
    assert result["ok"] is True
    assert result["coordinator_route_preflight_verified"] is True
    assert result["training_cuda_two_node_rc_ready"] is False


def test_checker_rejects_route_preflight_promoted_to_live_gate(tmp_path) -> None:
    report = _report(ready=False)
    report["coordinator_route_preflight"] = {
        "schema": "crowdtensor_cuda_training_coordinator_route_preflight_summary_v1",
        "verified": True,
        "authenticated_status_verified": True,
        "miner_auth_required_verified": True,
        "run_id_hash_verified": True,
        "stable_successes_observed": 2,
        "stable_successes_required": 2,
        "allocation_started": False,
        "kernel_push_attempted": False,
        "live_gate_claimed": True,
        "cleanup_verified": True,
        "url_public": False,
        "credentials_public": False,
        "public_artifact_safe": True,
    }
    result = _check(tmp_path, report, strict=False)
    assert result["ok"] is False
    assert "coordinator_route_preflight_invalid" in result["errors"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["single_kernel_gate"]["worker_report"].update(cuda_device_count=1),
        lambda value: value["single_kernel_gate"]["worker_report"].update(two_distinct_processes=False),
        lambda value: value["single_kernel_gate"]["worker_report"].update(real_cuda_backward=False),
        lambda value: value["single_kernel_gate"]["worker_report"].update(no_stage_loaded_full_model=False),
        lambda value: value["single_kernel_gate"]["worker_report"].update(base_weights_frozen=False),
        lambda value: value["single_kernel_gate"]["worker_report"].update(loss_reduced=False),
        lambda value: value["single_kernel_gate"]["worker_report"].update(checkpoint_resume_verified=False),
        lambda value: value["single_kernel_gate"]["worker_report"]["baseline"]["stage_records"]["0"][0].update(
            forward_hash=""
        ),
        lambda value: value["single_kernel_gate"]["worker_report"]["resumed"]["interruption"].update(
            new_pid=101
        ),
        lambda value: value["single_kernel_gate"]["worker_report"]["resumed"]["final_checkpoint"].update(
            stage_count=1
        ),
        lambda value: value["single_kernel_gate"]["checkpoint_bundle"].update(preserved=False),
        lambda value: value["two_node_gate"]["worker_reports"][0]["miner"].update(adapter_delta_tensor_count=0),
        lambda value: value["two_node_gate"]["worker_reports"][0]["miner"].update(
            adapter_delta_format="raw_tensor"
        ),
        lambda value: value["two_node_gate"]["worker_reports"][1]["miner"].update(
            dataset_shard_index=0
        ),
        lambda value: value["two_node_gate"]["worker_reports"][1]["miner"].update(
            base_model_version=2
        ),
        lambda value: value["two_node_gate"].update(
            kernel_ref_hashes=["sha256:kernel-stage0"]
        ),
        lambda value: value["two_node_gate"]["rendezvous"]["payloads"][0].update(
            byte_count=0
        ),
        lambda value: value["two_node_gate"]["rendezvous"]["registrations"][0].update(
            cuda_live=False
        ),
        lambda value: value["two_node_gate"]["training_state"].update(dense_diloco_aggregation=False),
        lambda value: value["two_node_gate"]["evaluation_export"].update(standard_peft_cuda_load=False),
        lambda value: value["two_node_gate"]["cleanup"].update(kernels_deleted=False),
        lambda value: value["two_node_gate"]["checkpoint_bundles"][0].update(preserved=False),
    ],
)
def test_checker_rejects_false_cuda_success_claims(tmp_path, mutation) -> None:
    report = copy.deepcopy(_report(ready=True))
    mutation(report)
    result = _check(tmp_path, report)
    assert result["ok"] is False
    assert result["training_cuda_two_node_rc_ready"] is False


def test_checker_rejects_private_tensor_and_path_material(tmp_path) -> None:
    report = _report(ready=True)
    report["private_payload"] = {"payload_b64": "secret", "checkpoint_path": "/root/private.pt"}
    result = _check(tmp_path, report)
    assert result["ok"] is False
    assert result["public_artifact_safe"] is False


def test_checker_accepts_exact_one_time_attempt_budget_amendment(tmp_path) -> None:
    report = _report(ready=False)
    amendment = {
        "schema": "crowdtensor_cuda_training_allocation_budget_amendment_v1",
        "authorized": True,
        "authorized_at": "2026-07-11T18:04:09Z",
        "authorization_hash": "sha256:" + "a" * 64,
        "authorization_text_public": False,
        "same_authorized_account_only": True,
        "original_single_kernel_attempt_limit": 2,
        "original_two_node_attempt_limit": 2,
        "additional_single_kernel_attempts": 1,
        "additional_two_node_attempts": 1,
        "revised_single_kernel_attempt_limit": 3,
        "revised_two_node_attempt_limit": 3,
        "allocation_timeout_seconds": 1800,
    }
    report["allocation_attempts"] = {
        "allocation_budget_amendment": amendment,
        "single_kernel_attempts": [{"attempt": value} for value in (1, 2, 3)],
        "two_node_attempts": [{"attempt": value} for value in (1, 2, 3)],
    }
    report["allocation_budget"] = allocation_budget_summary(report["allocation_attempts"])
    result = _check(tmp_path, report, strict=False)
    assert result["ok"] is True


def test_checker_rejects_fourth_attempt_after_one_time_amendment(tmp_path) -> None:
    report = _report(ready=False)
    report["allocation_attempts"] = {
        "allocation_budget_amendment": {
            "schema": "crowdtensor_cuda_training_allocation_budget_amendment_v1",
            "authorized": True,
            "authorized_at": "2026-07-11T18:04:09Z",
            "authorization_hash": "sha256:" + "a" * 64,
            "authorization_text_public": False,
            "same_authorized_account_only": True,
            "original_single_kernel_attempt_limit": 2,
            "original_two_node_attempt_limit": 2,
            "additional_single_kernel_attempts": 1,
            "additional_two_node_attempts": 1,
            "revised_single_kernel_attempt_limit": 3,
            "revised_two_node_attempt_limit": 3,
            "allocation_timeout_seconds": 1800,
        },
        "single_kernel_attempts": [{"attempt": value} for value in (1, 2, 3, 4)],
        "two_node_attempts": [],
    }
    result = _check(tmp_path, report, strict=False)
    assert result["ok"] is False
    assert "single_kernel_attempt_limit_exceeded" in result["errors"]
