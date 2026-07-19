from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from crowdtensor.training_contract import sha256_file
from scripts.training_foundation_rc_check import SCHEMA, check


PHASES = [
    "configuration",
    "dataset",
    "worker_assignment",
    "forward",
    "backward",
    "local_step",
    "outer_aggregation",
    "checkpoint",
    "evaluation",
    "cleanup",
]


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pipeline(*, interrupted: bool) -> dict:
    stage_records = {
        str(stage_id): [
            {
                "step": step,
                "forward_hash": "sha256:" + "1" * 64,
                "backward_gradient_hash": "sha256:" + "2" * 64,
                "lora_gradient_norm": 0.5,
                "optimizer_step": step + 1,
                "checkpoint_hash": "sha256:" + "3" * 64,
            }
            for step in range(2)
        ]
        for stage_id in (0, 1)
    }
    checkpoint_stages = [
        {
            "stage_id": stage_id,
            "optimizer_step": 2,
            "global_step": 2,
            "dataset_cursor": 4,
            "adapter_file_hash": "sha256:" + "4" * 64,
            "adapter_tensor_hash": "sha256:" + "5" * 64,
            "optimizer_file_hash": "sha256:" + "6" * 64,
            "content_hash": "sha256:" + "7" * 64,
            "base_weights_frozen": True,
        }
        for stage_id in (0, 1)
    ]
    return {
        "process_count": 2,
        "independent_worker_processes": True,
        "no_stage_loaded_full_model": True,
        "real_activation_transport": True,
        "real_backward_gradient_transport": True,
        "positive_lora_gradient_norms": True,
        "base_weights_frozen": True,
        "loss_start": 4.0,
        "loss_end": 3.0,
        "loss_reduced": True,
        "total_steps": 2,
        "stage_records": stage_records,
        "final_checkpoint": {
            "schema": "crowdtensor_pipeline_global_checkpoint_v1",
            "global_step": 2,
            "outer_step": 2,
            "dataset_cursor": 4,
            "stage_count": 2,
            "content_hash": "sha256:" + "8" * 64,
            "stages": checkpoint_stages,
        },
        "cleanup": {"all_worker_processes_stopped": True},
        "interruption": {
            "performed": interrupted,
            "worker_restarted": interrupted,
            "checkpoint_loaded": interrupted,
        },
    }


def _local_result(index: int) -> dict:
    return {
        "result_id": f"result-{index}",
        "delta_tensor_count": 2,
        "delta_file_hash": "sha256:" + str(index) * 64,
        "loss_start": 4.0,
        "loss_end": 3.5,
        "loss_reduced": True,
        "real_backward": True,
        "base_weights_frozen": True,
        "only_lora_trainable": True,
        "runtime": {"device": "cpu", "cuda_used": False},
    }


def _valid_tree(root: Path) -> tuple[Path, dict, dict]:
    adapter = {"layer.lora_A.weight": torch.ones(2, 2)}
    model_path = root / "exported_adapter" / "adapter_model.safetensors"
    model_path.parent.mkdir(parents=True)
    save_file(adapter, str(model_path))
    config_path = root / "exported_adapter" / "adapter_config.json"
    _write(config_path, {"peft_type": "LORA", "r": 2})
    stage_artifacts = []
    for stage_id in (0, 1):
        baseline = root / "pipeline" / f"baseline-{stage_id}.safetensors"
        resumed = root / "pipeline" / f"resumed-{stage_id}.safetensors"
        baseline.parent.mkdir(parents=True, exist_ok=True)
        save_file(adapter, str(baseline))
        save_file(adapter, str(resumed))
        stage_artifacts.append(
            {
                "stage_id": stage_id,
                "baseline": str(baseline.relative_to(root)),
                "resumed": str(resumed.relative_to(root)),
                "baseline_hash": sha256_file(baseline),
                "resumed_hash": sha256_file(resumed),
            }
        )
    gpu = {
        "schema": "crowdtensor_gpu_training_continuation_v1",
        "cpu_backend_verified": True,
        "gpu_live_verified": False,
        "gpu_success_claimed": False,
        "cuda_runtime_dry_run": {
            "dry_run_only": True,
            "cuda_initialized": False,
        },
        "stage_placement": [{"stage_id": 0}, {"stage_id": 1}],
        "two_machine_gpu_live_command": "future-gpu-command",
        "unverified_gpu_conditions": ["CUDA devices available"],
        "protocol_changes_required_for_gpu": False,
    }
    gpu_path = root / "gpu.json"
    _write(gpu_path, gpu)
    cleanup_path = root / "cleanup.json"
    _write(cleanup_path, {"cleanup_verified": True})
    evaluation = {
        "before": {"mean_loss": 4.0},
        "after": {"mean_loss": 3.0},
        "validation_loss_reduced": True,
        "adapter_changes_logits": True,
        "standard_peft_load_verified": True,
        "cpu_inference_verified": True,
    }
    evaluation_path = root / "evaluation.json"
    _write(evaluation_path, evaluation)
    job = {
        "schema": "crowdtensor_training_foundation_job_v1",
        "ok": True,
        "real_training": {
            "pytorch_autograd": True,
            "transformers_causal_lm": True,
            "peft_lora": True,
            "real_backward": True,
            "base_weights_frozen": True,
            "only_lora_trainable": True,
            "model_under_200m": True,
            "mock_only": False,
        },
        "local_training_results": [_local_result(0), _local_result(1)],
        "workers": {
            "worker_count": 2,
            "distinct_local_miners": True,
            "distinct_dataset_shards": True,
            "same_base_and_adapter_version": True,
            "assignments": [{"dataset_shard_index": 0}, {"dataset_shard_index": 1}],
        },
        "coordinator": {
            "existing_state_store_used": True,
            "http_coordinator_used": True,
            "task_lease_used": True,
            "result_ledger_used": True,
            "accepted_results": 2,
        },
        "outer_aggregation": {
            "input_delta_count": 2,
            "outer_step_before": 0,
            "outer_step_after": 1,
            "adapter_version_before": 0,
            "adapter_version_after": 1,
            "base_adapter_updated": True,
        },
        "compressed_transport": {
            "error_feedback": True,
            "dense_reconstruction_with_residual_verified": True,
        },
        "trusted_replay": {"accepted": True, "adapter_delta_tensors_exact": True},
        "pipeline_baseline": _pipeline(interrupted=False),
        "pipeline_interrupted_resume": _pipeline(interrupted=True),
        "checkpoint_resume_equivalence": {
            "checkpoint_resume_verified": True,
            "adapter_tensors_close": True,
            "final_loss_close": True,
        },
        "evaluation": evaluation,
        "phase_status": {phase: {"state": "completed"} for phase in PHASES},
        "cleanup": {"cleanup_verified": True, "live_resources_left_running": False},
        "security_boundary": {
            "trusted_workers_only": True,
            "open_public_malicious_training_solved": False,
        },
        "private_paths_public": False,
        "raw_dataset_public": False,
        "gpu_live_verified": False,
    }
    job_path = root / "job.json"
    _write(job_path, job)
    report = {
        "schema": SCHEMA,
        "training_foundation_rc_ready": True,
        "goal_achieved": True,
        "artifacts": {
            "job_report": "job.json",
            "gpu_continuation_manifest": "gpu.json",
            "exported_adapter_model": str(model_path.relative_to(root)),
            "exported_adapter_config": str(config_path.relative_to(root)),
            "cleanup": "cleanup.json",
            "evaluation": "evaluation.json",
        },
        "artifact_hashes": {
            "job_report": sha256_file(job_path),
            "gpu_continuation_manifest": sha256_file(gpu_path),
            "exported_adapter_model": sha256_file(model_path),
            "exported_adapter_config": sha256_file(config_path),
            "cleanup": sha256_file(cleanup_path),
            "evaluation": sha256_file(evaluation_path),
        },
        "stage_adapter_artifacts": stage_artifacts,
        "gpu_live_verified": False,
        "private_paths_public": False,
        "raw_dataset_public": False,
    }
    report_path = root / "training_foundation_rc.json"
    _write(report_path, report)
    return report_path, report, job


def _rewrite_job(root: Path, report: dict, job: dict) -> None:
    job_path = root / "job.json"
    _write(job_path, job)
    report["artifact_hashes"]["job_report"] = sha256_file(job_path)
    _write(root / "training_foundation_rc.json", report)


def test_checker_accepts_complete_real_training_contract(tmp_path) -> None:
    report_path, _report, _job = _valid_tree(tmp_path)
    result = check(report_path, require_ready=True)
    assert result["ok"] is True
    assert result["training_foundation_rc_ready"] is True


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("mock", "mock_only_training_rejected"),
        ("no_backward", "real_backward_missing"),
        ("single_process", "baseline_single_process_fake_sharding"),
        ("base_changed", "base_weights_changed"),
        ("no_loss_reduction", "validation_loss_not_reduced"),
        ("no_adapter_tensors", "local_result_0_real_adapter_tensors_missing"),
        ("no_outer", "outer_aggregation_missing"),
        ("no_resume", "checkpoint_resume_equivalence_missing"),
        ("gpu_overclaim", "gpu_dry_run_misrepresented_as_success"),
    ],
)
def test_checker_rejects_false_training_evidence(tmp_path, mutation: str, expected: str) -> None:
    report_path, report, job = _valid_tree(tmp_path)
    if mutation == "mock":
        job["real_training"]["mock_only"] = True
    elif mutation == "no_backward":
        job["real_training"]["real_backward"] = False
    elif mutation == "single_process":
        job["pipeline_baseline"]["process_count"] = 1
    elif mutation == "base_changed":
        job["real_training"]["base_weights_frozen"] = False
    elif mutation == "no_loss_reduction":
        job["evaluation"]["validation_loss_reduced"] = False
        job["evaluation"]["after"]["mean_loss"] = 5.0
    elif mutation == "no_adapter_tensors":
        job["local_training_results"][0]["delta_tensor_count"] = 0
        job["local_training_results"][0]["delta_file_hash"] = ""
    elif mutation == "no_outer":
        job["outer_aggregation"]["input_delta_count"] = 0
    elif mutation == "no_resume":
        job["checkpoint_resume_equivalence"]["checkpoint_resume_verified"] = False
    elif mutation == "gpu_overclaim":
        gpu_path = tmp_path / "gpu.json"
        gpu = json.loads(gpu_path.read_text(encoding="utf-8"))
        gpu["gpu_live_verified"] = True
        gpu["gpu_success_claimed"] = True
        _write(gpu_path, gpu)
        report["artifact_hashes"]["gpu_continuation_manifest"] = sha256_file(gpu_path)
    _rewrite_job(tmp_path, report, job)
    result = check(report_path, require_ready=True)
    assert result["ok"] is False
    assert expected in result["errors"]


def test_checker_rejects_missing_exported_adapter(tmp_path) -> None:
    report_path, _report, _job = _valid_tree(tmp_path)
    (tmp_path / "exported_adapter" / "adapter_model.safetensors").unlink()
    result = check(report_path, require_ready=True)
    assert result["ok"] is False
    assert "exported_adapter_model_missing" in result["errors"]
