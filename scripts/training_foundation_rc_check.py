#!/usr/bin/env python3
"""Strict checker for a CrowdTensor Training Foundation RC artifact."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor.training_contract import sha256_file  # noqa: E402


SCHEMA = "crowdtensor_training_foundation_rc_v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _artifact(report_path: Path, report: dict[str, Any], name: str) -> Path:
    value = (report.get("artifacts") or {}).get(name)
    if not isinstance(value, str) or not value:
        return report_path.parent / "<missing>"
    return report_path.parent / value


def check(report_path: str | Path, *, require_ready: bool = False) -> dict[str, Any]:
    path = Path(report_path).resolve()
    errors: list[str] = []
    try:
        report = _load(path)
    except Exception as exc:
        return {
            "schema": "crowdtensor_training_foundation_rc_check_v1",
            "ok": False,
            "training_foundation_rc_ready": False,
            "errors": [f"report_load_failed:{type(exc).__name__}:{exc}"],
            "error_count": 1,
        }
    if report.get("schema") != SCHEMA:
        errors.append("training_foundation_rc_schema_mismatch")

    job_path = _artifact(path, report, "job_report")
    gpu_path = _artifact(path, report, "gpu_continuation_manifest")
    export_model_path = _artifact(path, report, "exported_adapter_model")
    export_config_path = _artifact(path, report, "exported_adapter_config")
    cleanup_path = _artifact(path, report, "cleanup")
    evaluation_path = _artifact(path, report, "evaluation")
    for name, artifact_path in {
        "job_report": job_path,
        "gpu_continuation_manifest": gpu_path,
        "exported_adapter_model": export_model_path,
        "exported_adapter_config": export_config_path,
        "cleanup": cleanup_path,
        "evaluation": evaluation_path,
    }.items():
        if not artifact_path.is_file():
            errors.append(f"{name}_missing")
    if job_path.is_file() and sha256_file(job_path) != (report.get("artifact_hashes") or {}).get("job_report"):
        errors.append("job_report_hash_mismatch")
    if gpu_path.is_file() and sha256_file(gpu_path) != (report.get("artifact_hashes") or {}).get(
        "gpu_continuation_manifest"
    ):
        errors.append("gpu_continuation_manifest_hash_mismatch")
    if export_model_path.is_file() and sha256_file(export_model_path) != (report.get("artifact_hashes") or {}).get(
        "exported_adapter_model"
    ):
        errors.append("exported_adapter_model_hash_mismatch")
    if export_config_path.is_file() and sha256_file(export_config_path) != (report.get("artifact_hashes") or {}).get(
        "exported_adapter_config"
    ):
        errors.append("exported_adapter_config_hash_mismatch")
    for name, artifact_path in (("cleanup", cleanup_path), ("evaluation", evaluation_path)):
        if artifact_path.is_file() and sha256_file(artifact_path) != (report.get("artifact_hashes") or {}).get(name):
            errors.append(f"{name}_hash_mismatch")

    job: dict[str, Any] = {}
    if job_path.is_file():
        try:
            job = _load(job_path)
        except Exception as exc:
            errors.append(f"job_report_load_failed:{type(exc).__name__}")
    if job.get("schema") != "crowdtensor_training_foundation_job_v1" or job.get("ok") is not True:
        errors.append("real_training_job_not_successful")
    real = job.get("real_training") if isinstance(job.get("real_training"), dict) else {}
    required_real = {
        "pytorch_autograd": "real_pytorch_autograd_missing",
        "transformers_causal_lm": "real_transformers_causal_lm_missing",
        "peft_lora": "real_peft_lora_missing",
        "real_backward": "real_backward_missing",
        "base_weights_frozen": "base_weights_changed",
        "only_lora_trainable": "non_lora_parameter_was_trainable",
        "model_under_200m": "fixture_model_exceeds_200m",
    }
    for field, code in required_real.items():
        if real.get(field) is not True:
            errors.append(code)
    if real.get("mock_only") is not False:
        errors.append("mock_only_training_rejected")

    local_results = job.get("local_training_results") if isinstance(job.get("local_training_results"), list) else []
    if len(local_results) != 2:
        errors.append("two_real_local_training_results_required")
    for index, result in enumerate(local_results):
        if not isinstance(result, dict):
            errors.append(f"local_result_{index}_invalid")
            continue
        if not all(result.get(field) is True for field in ("real_backward", "base_weights_frozen", "only_lora_trainable")):
            errors.append(f"local_result_{index}_training_invariant_failed")
        if int(result.get("delta_tensor_count") or 0) < 1 or not str(result.get("delta_file_hash") or "").startswith(
            "sha256:"
        ):
            errors.append(f"local_result_{index}_real_adapter_tensors_missing")
        if result.get("loss_reduced") is not True or not (
            float(result.get("loss_end", math.inf)) < float(result.get("loss_start", -math.inf))
        ):
            errors.append(f"local_result_{index}_loss_not_reduced")
        runtime = result.get("runtime") if isinstance(result.get("runtime"), dict) else {}
        if runtime.get("device") != "cpu" or runtime.get("cuda_used") is not False:
            errors.append(f"local_result_{index}_cpu_boundary_failed")

    workers = job.get("workers") if isinstance(job.get("workers"), dict) else {}
    if int(workers.get("worker_count") or 0) != 2:
        errors.append("two_local_miners_required")
    for field, code in {
        "distinct_local_miners": "distinct_local_miners_missing",
        "distinct_dataset_shards": "distinct_dataset_shards_missing",
        "same_base_and_adapter_version": "miner_base_adapter_version_mismatch",
    }.items():
        if workers.get(field) is not True:
            errors.append(code)
    assignments = workers.get("assignments") if isinstance(workers.get("assignments"), list) else []
    if {int(item.get("dataset_shard_index", -1)) for item in assignments if isinstance(item, dict)} != {0, 1}:
        errors.append("worker_shard_assignment_invalid")

    coordinator = job.get("coordinator") if isinstance(job.get("coordinator"), dict) else {}
    for field in ("existing_state_store_used", "http_coordinator_used", "task_lease_used", "result_ledger_used"):
        if coordinator.get(field) is not True:
            errors.append(f"coordinator_{field}_missing")
    if int(coordinator.get("accepted_results") or 0) != 2:
        errors.append("coordinator_two_results_not_accepted")

    outer = job.get("outer_aggregation") if isinstance(job.get("outer_aggregation"), dict) else {}
    if int(outer.get("input_delta_count") or 0) != 2:
        errors.append("outer_aggregation_missing")
    if int(outer.get("outer_step_before", -1)) != 0 or int(outer.get("outer_step_after", -1)) != 1:
        errors.append("outer_optimizer_step_not_advanced")
    if int(outer.get("adapter_version_before", -1)) != 0 or int(outer.get("adapter_version_after", -1)) != 1:
        errors.append("global_adapter_version_not_advanced")
    if outer.get("base_adapter_updated") is not True:
        errors.append("global_adapter_not_updated")

    compression = job.get("compressed_transport") if isinstance(job.get("compressed_transport"), dict) else {}
    if compression.get("error_feedback") is not True:
        errors.append("error_feedback_transport_missing")
    if compression.get("dense_reconstruction_with_residual_verified") is not True:
        errors.append("error_feedback_reconstruction_failed")
    replay = job.get("trusted_replay") if isinstance(job.get("trusted_replay"), dict) else {}
    if replay.get("accepted") is not True or replay.get("adapter_delta_tensors_exact") is not True:
        errors.append("trusted_worker_replay_missing")

    baseline = job.get("pipeline_baseline") if isinstance(job.get("pipeline_baseline"), dict) else {}
    resumed = job.get("pipeline_interrupted_resume") if isinstance(job.get("pipeline_interrupted_resume"), dict) else {}
    for label, pipeline in (("baseline", baseline), ("resumed", resumed)):
        if int(pipeline.get("process_count") or 0) != 2 or pipeline.get("independent_worker_processes") is not True:
            errors.append(f"{label}_single_process_fake_sharding")
        if pipeline.get("no_stage_loaded_full_model") is not True:
            errors.append(f"{label}_full_model_loaded_by_stage")
        if pipeline.get("real_activation_transport") is not True:
            errors.append(f"{label}_forward_transport_missing")
        if pipeline.get("real_backward_gradient_transport") is not True:
            errors.append(f"{label}_backward_transport_missing")
        if pipeline.get("positive_lora_gradient_norms") is not True:
            errors.append(f"{label}_lora_gradients_missing")
        if pipeline.get("base_weights_frozen") is not True:
            errors.append(f"{label}_base_weights_changed")
        if pipeline.get("loss_reduced") is not True or not (
            float(pipeline.get("loss_end", math.inf)) < float(pipeline.get("loss_start", -math.inf))
        ):
            errors.append(f"{label}_loss_not_reduced")
        if (pipeline.get("cleanup") or {}).get("all_worker_processes_stopped") is not True:
            errors.append(f"{label}_worker_cleanup_missing")
        total_steps = int(pipeline.get("total_steps") or 0)
        stage_records = pipeline.get("stage_records") if isinstance(pipeline.get("stage_records"), dict) else {}
        if total_steps < 2 or set(stage_records) != {"0", "1"}:
            errors.append(f"{label}_per_stage_records_missing")
        else:
            for stage_id in ("0", "1"):
                records = stage_records.get(stage_id) if isinstance(stage_records.get(stage_id), list) else []
                if len(records) != total_steps:
                    errors.append(f"{label}_stage_{stage_id}_record_count_mismatch")
                    continue
                for record in records:
                    if not str(record.get("forward_hash") or "").startswith("sha256:"):
                        errors.append(f"{label}_stage_{stage_id}_forward_hash_missing")
                        break
                    if not str(record.get("backward_gradient_hash") or "").startswith("sha256:"):
                        errors.append(f"{label}_stage_{stage_id}_backward_hash_missing")
                        break
                    if float(record.get("lora_gradient_norm") or 0.0) <= 0.0:
                        errors.append(f"{label}_stage_{stage_id}_gradient_norm_missing")
                        break
                    if int(record.get("optimizer_step") or 0) < 1:
                        errors.append(f"{label}_stage_{stage_id}_optimizer_step_missing")
                        break
                    if not str(record.get("checkpoint_hash") or "").startswith("sha256:"):
                        errors.append(f"{label}_stage_{stage_id}_checkpoint_hash_missing")
                        break
        checkpoint = pipeline.get("final_checkpoint") if isinstance(pipeline.get("final_checkpoint"), dict) else {}
        checkpoint_stages = checkpoint.get("stages") if isinstance(checkpoint.get("stages"), list) else []
        if (
            checkpoint.get("schema") != "crowdtensor_pipeline_global_checkpoint_v1"
            or int(checkpoint.get("stage_count") or 0) != 2
            or len(checkpoint_stages) != 2
            or int(checkpoint.get("global_step") or 0) != total_steps
            or int(checkpoint.get("outer_step") or 0) != total_steps
            or not str(checkpoint.get("content_hash") or "").startswith("sha256:")
        ):
            errors.append(f"{label}_global_checkpoint_incomplete")
        for stage in checkpoint_stages:
            if not all(str(stage.get(field) or "").startswith("sha256:") for field in (
                "adapter_file_hash",
                "adapter_tensor_hash",
                "optimizer_file_hash",
                "content_hash",
            )):
                errors.append(f"{label}_stage_checkpoint_hashes_incomplete")
            if stage.get("base_weights_frozen") is not True:
                errors.append(f"{label}_stage_checkpoint_base_changed")
            if int(stage.get("dataset_cursor", -1)) < 0 or int(stage.get("optimizer_step") or 0) != total_steps:
                errors.append(f"{label}_stage_checkpoint_cursor_or_step_invalid")
    interruption = resumed.get("interruption") if isinstance(resumed.get("interruption"), dict) else {}
    if not all(interruption.get(field) is True for field in ("performed", "worker_restarted", "checkpoint_loaded")):
        errors.append("controlled_worker_interruption_resume_missing")
    equivalence = job.get("checkpoint_resume_equivalence") if isinstance(
        job.get("checkpoint_resume_equivalence"), dict
    ) else {}
    if equivalence.get("checkpoint_resume_verified") is not True:
        errors.append("checkpoint_resume_equivalence_missing")
    if equivalence.get("adapter_tensors_close") is not True or equivalence.get("final_loss_close") is not True:
        errors.append("checkpoint_resume_numeric_mismatch")

    stage_artifacts = report.get("stage_adapter_artifacts") if isinstance(
        report.get("stage_adapter_artifacts"), list
    ) else []
    if len(stage_artifacts) != 2:
        errors.append("pipeline_stage_adapter_artifacts_missing")
    else:
        try:
            import torch
            from safetensors.torch import load_file

            for entry in stage_artifacts:
                baseline_path = path.parent / str(entry["baseline"])
                resumed_path = path.parent / str(entry["resumed"])
                left = load_file(str(baseline_path), device="cpu")
                right = load_file(str(resumed_path), device="cpu")
                if set(left) != set(right) or not all(torch.allclose(left[name], right[name]) for name in left):
                    errors.append(f"stage_{entry.get('stage_id')}_checkpoint_adapter_mismatch")
        except Exception as exc:
            errors.append(f"pipeline_stage_adapter_load_failed:{type(exc).__name__}")

    evaluation = job.get("evaluation") if isinstance(job.get("evaluation"), dict) else {}
    if evaluation_path.is_file():
        try:
            if _load(evaluation_path) != evaluation:
                errors.append("evaluation_artifact_does_not_match_job_report")
        except Exception as exc:
            errors.append(f"evaluation_artifact_load_failed:{type(exc).__name__}")
    before = evaluation.get("before") if isinstance(evaluation.get("before"), dict) else {}
    after = evaluation.get("after") if isinstance(evaluation.get("after"), dict) else {}
    if evaluation.get("validation_loss_reduced") is not True or not (
        float(after.get("mean_loss", math.inf)) < float(before.get("mean_loss", -math.inf))
    ):
        errors.append("validation_loss_not_reduced")
    if evaluation.get("adapter_changes_logits") is not True:
        errors.append("exported_adapter_does_not_affect_model")
    if evaluation.get("standard_peft_load_verified") is not True or evaluation.get("cpu_inference_verified") is not True:
        errors.append("standard_peft_export_load_not_verified")
    if export_model_path.is_file():
        try:
            from safetensors.torch import load_file

            if not load_file(str(export_model_path), device="cpu"):
                errors.append("exported_adapter_has_no_tensors")
        except Exception as exc:
            errors.append(f"exported_adapter_load_failed:{type(exc).__name__}")
    if export_config_path.is_file():
        try:
            config = _load(export_config_path)
            if str(config.get("peft_type") or "").upper() != "LORA":
                errors.append("exported_adapter_config_not_lora")
        except Exception as exc:
            errors.append(f"exported_adapter_config_load_failed:{type(exc).__name__}")

    gpu: dict[str, Any] = {}
    if gpu_path.is_file():
        try:
            gpu = _load(gpu_path)
        except Exception as exc:
            errors.append(f"gpu_continuation_manifest_load_failed:{type(exc).__name__}")
    if gpu.get("schema") != "crowdtensor_gpu_training_continuation_v1":
        errors.append("gpu_continuation_manifest_schema_mismatch")
    if gpu.get("cpu_backend_verified") is not True:
        errors.append("gpu_handoff_cpu_backend_not_verified")
    if gpu.get("gpu_live_verified") is not False or gpu.get("gpu_success_claimed") is not False:
        errors.append("gpu_dry_run_misrepresented_as_success")
    dry_run = gpu.get("cuda_runtime_dry_run") if isinstance(gpu.get("cuda_runtime_dry_run"), dict) else {}
    if dry_run.get("dry_run_only") is not True or dry_run.get("cuda_initialized") is not False:
        errors.append("cuda_dry_run_contract_missing")
    if len(gpu.get("stage_placement") or []) != 2 or not gpu.get("two_machine_gpu_live_command"):
        errors.append("gpu_stage_placement_or_live_command_missing")
    if not gpu.get("unverified_gpu_conditions"):
        errors.append("gpu_unverified_conditions_missing")
    if gpu.get("protocol_changes_required_for_gpu") is not False:
        errors.append("gpu_handoff_requires_protocol_changes")

    phases = job.get("phase_status") if isinstance(job.get("phase_status"), dict) else {}
    for phase in (
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
    ):
        if (phases.get(phase) or {}).get("state") != "completed":
            errors.append(f"phase_{phase}_not_completed")
    cleanup = job.get("cleanup") if isinstance(job.get("cleanup"), dict) else {}
    if cleanup.get("cleanup_verified") is not True or cleanup.get("live_resources_left_running") is not False:
        errors.append("training_cleanup_not_verified")
    security = job.get("security_boundary") if isinstance(job.get("security_boundary"), dict) else {}
    if security.get("trusted_workers_only") is not True:
        errors.append("permissioned_worker_boundary_missing")
    if security.get("open_public_malicious_training_solved") is not False:
        errors.append("open_miner_security_overclaimed")
    if any(job.get(field) is not False for field in ("private_paths_public", "raw_dataset_public", "gpu_live_verified")):
        errors.append("public_safety_or_gpu_boundary_failed")

    serialized = json.dumps(report, sort_keys=True).lower()
    for forbidden in ("authorization: bearer", "kaggle_token", "api_token", "private_dataset.jsonl"):
        if forbidden in serialized:
            errors.append(f"public_report_contains_forbidden_material:{forbidden}")

    ready = not errors
    if report.get("training_foundation_rc_ready") is not ready:
        errors.append("training_foundation_rc_ready_flag_not_truthful")
    if report.get("goal_achieved") is not ready:
        errors.append("goal_achieved_flag_not_truthful")
    if require_ready and not ready:
        errors.append("training_foundation_rc_required_but_not_ready")
    return {
        "schema": "crowdtensor_training_foundation_rc_check_v1",
        "ok": not errors,
        "training_foundation_rc_ready": ready and not errors,
        "goal_achieved": ready and not errors,
        "errors": errors,
        "error_count": len(errors),
        "report": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check(args.report, require_ready=args.require_ready)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"ok={result['ok']} ready={result['training_foundation_rc_ready']}")
        for error in result["errors"]:
            print(f"error: {error}")
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
