#!/usr/bin/env python3
"""Strict checker for the CrowdTensor two-node CUDA Training RC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from crowdtensor.training_allocation_budget import allocation_budget_summary
from crowdtensor.training_contract import sha256_json


SCHEMA = "crowdtensor_training_cuda_two_node_rc_v1"
REQUIRED_REJECTION_CODES = {
    "duplicate_result": "duplicate_result",
    "wrong_shard": "dataset_shard_hash_mismatch",
    "stale_adapter_version": "adapter_version_mismatch",
    "stale_model_version": "base_model_version_mismatch",
    "shape_mismatch": "adapter_delta_shape_mismatch",
    "dtype_mismatch": "adapter_delta_dtype_mismatch",
    "nan": "adapter_delta_non_finite",
    "infinity": "adapter_delta_non_finite",
    "excessive_norm": "adapter_delta_norm_too_large",
    "loss_spike": "training_loss_spike",
}


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("CUDA Training RC report must be a JSON object")
    return value


def _single_pipeline_verified(report: dict[str, Any], *, resumed: bool) -> bool:
    records = dict(report.get("stage_records") or {})
    checkpoint = dict(report.get("final_checkpoint") or {})
    checkpoint_stages = list(checkpoint.get("stages") or [])
    stage_records = {
        stage_id: list(records.get(str(stage_id)) or [])
        for stage_id in (0, 1)
    }
    interruption = dict(report.get("interruption") or {})
    record_contract = all(
        len(stage_records[stage_id]) >= 4
        and {int(item.get("step", -1)) for item in stage_records[stage_id]} >= set(range(4))
        and all(
            int(item.get("pid") or 0) > 0
            and item.get("cuda_device") == f"cuda:{stage_id}"
            and str(item.get("forward_hash") or "").startswith("sha256:")
            and str(item.get("backward_gradient_hash") or "").startswith("sha256:")
            and str(item.get("checkpoint_hash") or "").startswith("sha256:")
            and float(item.get("lora_gradient_norm") or 0.0) > 0.0
            and int(item.get("peak_allocated_bytes") or 0) > 0
            and int(item.get("peak_reserved_bytes") or 0) > 0
            for item in stage_records[stage_id]
        )
        for stage_id in (0, 1)
    )
    interruption_contract = bool(
        (
            interruption.get("performed") is True
            and interruption.get("checkpoint_loaded") is True
            and interruption.get("worker_restarted") is True
            and int(interruption.get("old_pid") or 0) > 0
            and int(interruption.get("new_pid") or 0) > 0
            and int(interruption.get("old_pid")) != int(interruption.get("new_pid"))
            and int(interruption.get("resumed_optimizer_step") or 0) > 0
        )
        if resumed
        else interruption.get("performed") is not True
    )
    return bool(
        int(report.get("total_steps") or 0) >= 4
        and report.get("real_cuda_forward") is True
        and report.get("real_cuda_backward") is True
        and report.get("real_activation_transport") is True
        and report.get("real_backward_gradient_transport") is True
        and report.get("loss_reduced") is True
        and report.get("base_weights_frozen") is True
        and report.get("positive_lora_gradient_norms") is True
        and report.get("positive_cuda_memory") is True
        and report.get("distinct_stage_pids") is True
        and report.get("distinct_cuda_devices") is True
        and report.get("no_stage_loaded_full_model") is True
        and record_contract
        and checkpoint.get("complete") is True
        and int(checkpoint.get("stage_count") or 0) == 2
        and int(checkpoint.get("global_step") or 0) >= 4
        and int(checkpoint.get("outer_step") or 0) >= 4
        and str(checkpoint.get("content_hash") or "").startswith("sha256:")
        and len(checkpoint_stages) == 2
        and all(
            str(item.get("content_hash") or "").startswith("sha256:")
            and item.get("grad_scaler_state_present") is True
            and item.get("cuda_placement") == f"cuda:{index}"
            for index, item in enumerate(checkpoint_stages)
        )
        and (report.get("cleanup") or {}).get("all_worker_processes_stopped") is True
        and interruption_contract
    )


def _single_gate_verified(report: dict[str, Any]) -> bool:
    worker = dict(report.get("worker_report") or {})
    baseline = dict(worker.get("baseline") or {})
    resumed = dict(worker.get("resumed") or {})
    equivalence = dict(worker.get("resume_equivalence") or {})
    cleanup = dict(report.get("cleanup") or {})
    worker_bundle = dict(worker.get("checkpoint_bundle") or {})
    checkpoint_bundle = dict(report.get("checkpoint_bundle") or {})
    return bool(
        report.get("ok") is True
        and report.get("public_artifact_safe") is True
        and report.get("single_kernel_t4x2_verified") is True
        and worker.get("ok") is True
        and worker.get("kaggle_kernel") is True
        and worker.get("gpu_live_verified") is True
        and int(worker.get("cuda_device_count") or 0) >= 2
        and worker.get("two_distinct_processes") is True
        and worker.get("two_distinct_cuda_devices") is True
        and worker.get("real_activation_transport") is True
        and worker.get("real_backward_gradient_transport") is True
        and worker.get("real_cuda_backward") is True
        and worker.get("no_stage_loaded_full_model") is True
        and worker.get("base_weights_frozen") is True
        and worker.get("positive_lora_gradient_norms") is True
        and worker.get("positive_cuda_memory") is True
        and worker.get("loss_reduced") is True
        and worker.get("controlled_stage_restart") is True
        and worker.get("checkpoint_resume_verified") is True
        and equivalence.get("checkpoint_resume_verified") is True
        and _single_pipeline_verified(baseline, resumed=False)
        and _single_pipeline_verified(resumed, resumed=True)
        and worker_bundle.get("present") is True
        and worker_bundle.get("contains_baseline_and_resumed_checkpoints") is True
        and int(worker_bundle.get("file_count") or 0) > 0
        and checkpoint_bundle.get("preserved") is True
        and checkpoint_bundle.get("worker_hash_match") is True
        and checkpoint_bundle.get("file_hash") == worker_bundle.get("file_hash")
        and int(checkpoint_bundle.get("byte_count") or 0) > 0
        and (resumed.get("interruption") or {}).get("checkpoint_loaded") is True
        and cleanup.get("kernel_deleted") is True
        and cleanup.get("private_package_removed") is True
        and cleanup.get("checkpoint_preserved") is True
        and cleanup.get("private_cleanup_state_removed") is True
    )


def _embedded_single_gate_binding_verified(
    single_gate: dict[str, Any],
    two_node_gate: dict[str, Any],
) -> bool:
    workers = {
        str(worker.get("role")): worker
        for worker in two_node_gate.get("worker_reports") or []
        if isinstance(worker, dict)
    }
    bundles = {
        str(bundle.get("role")): bundle
        for bundle in two_node_gate.get("checkpoint_bundles") or []
        if isinstance(bundle, dict)
    }
    stage0_worker = dict(workers.get("stage0") or {})
    embedded_worker = dict(stage0_worker.get("embedded_single_kernel_gate") or {})
    selected_worker = dict(single_gate.get("worker_report") or {})
    selected_bundle = dict(single_gate.get("checkpoint_bundle") or {})
    stage0_bundle = dict(bundles.get("stage0") or {})
    kernel_hashes_by_role = dict(two_node_gate.get("kernel_ref_hashes_by_role") or {})
    source_kernel_hash = str(single_gate.get("source_kernel_ref_hash") or "")
    two_cleanup = dict(two_node_gate.get("cleanup") or {})
    single_cleanup = dict(single_gate.get("cleanup") or {})
    return bool(
        single_gate.get("evidence_source")
        == "two_node_stage0_kernel_embedded_single_gate"
        and single_gate.get("coallocated_with_two_node_attempt") is True
        and single_gate.get("source_role") == "stage0"
        and single_gate.get("execution_order") == "before_cross_node_stage0"
        and int(single_gate.get("source_two_node_attempt") or 0)
        == int(two_node_gate.get("attempt") or 0)
        and int(two_node_gate.get("attempt") or 0) > 0
        and single_gate.get("source_binding_verified") is True
        and stage0_worker.get("embedded_single_kernel_gate_verified") is True
        and selected_worker == embedded_worker
        and str(single_gate.get("source_worker_report_hash") or "")
        == sha256_json(selected_worker)
        and source_kernel_hash.startswith("sha256:")
        and source_kernel_hash == kernel_hashes_by_role.get("stage0")
        and source_kernel_hash in set(two_node_gate.get("kernel_ref_hashes") or [])
        and selected_bundle.get("preserved") is True
        and selected_bundle.get("worker_hash_match") is True
        and selected_bundle.get("contains_baseline_and_resumed_checkpoints") is True
        and selected_bundle.get("file_hash") == stage0_bundle.get("file_hash")
        and selected_bundle.get("file_hash")
        == (selected_worker.get("checkpoint_bundle") or {}).get("file_hash")
        and stage0_bundle.get("preserved") is True
        and stage0_bundle.get("worker_hash_match") is True
        and stage0_bundle.get("contains_baseline_and_resumed_checkpoints") is True
        and single_cleanup.get("kernel_deleted") is True
        and single_cleanup.get("private_package_removed") is True
        and single_cleanup.get("private_cleanup_state_removed") is True
        and single_cleanup.get("kernel_deleted") == two_cleanup.get("kernels_deleted")
        and single_cleanup.get("private_package_removed")
        == two_cleanup.get("private_packages_removed")
        and single_cleanup.get("private_cleanup_state_removed")
        == two_cleanup.get("private_cleanup_state_removed")
    )


def _worker_verified(worker: dict[str, Any], role: str) -> bool:
    pipeline = dict(worker.get("pipeline") or {})
    miner = dict(worker.get("miner") or {})
    runtime = dict(miner.get("runtime") or {})
    evaluation = dict(worker.get("evaluation") or {})
    global_adapter = dict(worker.get("global_adapter") or {})
    checkpoint_bundle = dict(worker.get("checkpoint_bundle") or {})
    return bool(
        worker.get("ok") is True
        and worker.get("public_artifact_safe") is True
        and worker.get("role") == role
        and worker.get("kaggle_kernel") is True
        and worker.get("gpu_live_verified") is True
        and int(worker.get("cuda_device_count") or 0) >= 1
        and pipeline.get("role") == role
        and int(pipeline.get("steps_completed") or 0) >= 4
        and pipeline.get("real_cuda_forward") is True
        and pipeline.get("real_cuda_backward") is True
        and pipeline.get("real_activation_transport") is True
        and pipeline.get("real_backward_gradient_transport") is True
        and pipeline.get("positive_lora_gradient_norms") is True
        and pipeline.get("base_weights_frozen") is True
        and pipeline.get("no_full_model_loaded") is True
        and bool(str(pipeline.get("checkpoint_hash") or ""))
        and pipeline.get("checkpoint_grad_scaler_state_present") is True
        and miner.get("coordinator_accepted") is True
        and int(miner.get("base_model_version", -1)) >= 0
        and int(miner.get("adapter_version", -1)) >= 0
        and int(miner.get("dataset_shard_index", -1)) in {0, 1}
        and miner.get("base_weights_frozen") is True
        and miner.get("only_lora_trainable") is True
        and miner.get("real_backward") is True
        and miner.get("loss_reduced") is True
        and int(miner.get("adapter_delta_tensor_count") or 0) > 0
        and str(miner.get("adapter_delta_file_hash") or "").startswith("sha256:")
        and str(miner.get("adapter_delta_tensor_specs_hash") or "").startswith("sha256:")
        and miner.get("adapter_delta_format") == "named_safetensors"
        and miner.get("adapter_delta_named_tensors") is True
        and int(miner.get("optimizer_steps") or 0) > 0
        and int(miner.get("tokens_seen") or 0) > 0
        and float(miner.get("elapsed_seconds") or 0.0) > 0.0
        and str(miner.get("checkpoint_hash") or "").startswith("sha256:")
        and int(miner.get("peak_allocated_bytes") or 0) > 0
        and int(miner.get("peak_reserved_bytes") or 0) > 0
        and runtime.get("cuda_used") is True
        and runtime.get("gpu_live_verified") is True
        and int(runtime.get("device_index", -1)) == 0
        and bool(str(runtime.get("device_name_hash") or ""))
        and evaluation.get("standard_peft_cuda_load") is True
        and evaluation.get("adapter_changes_logits") is True
        and evaluation.get("validation_loss_reduced") is True
        and int(global_adapter.get("adapter_version") or 0) == 1
        and int(global_adapter.get("outer_step") or 0) == 1
        and checkpoint_bundle.get("present") is True
        and checkpoint_bundle.get("contains_pipeline_and_miner_checkpoints") is True
        and int(checkpoint_bundle.get("file_count") or 0) > 0
        and (worker.get("cleanup") or {}).get("private_runtime_removed") is True
    )


def _two_node_gate_verified(report: dict[str, Any]) -> bool:
    workers = {
        str(worker.get("role")): worker
        for worker in report.get("worker_reports") or []
        if isinstance(worker, dict)
    }
    rendezvous = dict(report.get("rendezvous") or {})
    payloads = list(rendezvous.get("payloads") or [])
    state = dict(report.get("training_state") or {})
    evaluation = dict(report.get("evaluation_export") or {})
    feedback = dict(report.get("error_feedback") or {})
    cleanup = dict(report.get("cleanup") or {})
    checkpoint_bundles = {
        str(item.get("role")): item
        for item in report.get("checkpoint_bundles") or []
        if isinstance(item, dict)
    }
    registrations = {
        str(item.get("role")): item
        for item in rendezvous.get("registrations") or []
        if isinstance(item, dict)
    }
    miners = {
        role: dict((workers.get(role) or {}).get("miner") or {})
        for role in ("stage0", "stage1")
    }
    payload_contract = bool(
        {
            (str(item.get("kind")), int(item.get("step", -1)))
            for item in payloads
        }
        == {(kind, step) for kind in ("activation", "gradient") for step in range(4)}
        and all(
            str(item.get("payload_hash") or "").startswith("sha256:")
            and int(item.get("byte_count") or 0) > 0
            and bool(item.get("shape"))
            and bool(str(item.get("dtype") or ""))
            for item in payloads
        )
    )
    return bool(
        report.get("ok") is True
        and report.get("public_artifact_safe") is True
        and report.get("two_node_cuda_verified") is True
        and report.get("same_authorized_account") is True
        and report.get("multi_account_used") is False
        and report.get("tpu_used") is False
        and int(report.get("requested_kernel_count") or 0) == 2
        and int(report.get("used_gpu_per_kernel") or 0) == 1
        and report.get("all_four_t4_used_claimed") is False
        and int(report.get("max_observed_running_kernel_count") or 0) >= 2
        and len(set(report.get("kernel_ref_hashes") or [])) == 2
        and all(str(value).startswith("sha256:") for value in report.get("kernel_ref_hashes") or [])
        and set(workers) == {"stage0", "stage1"}
        and _worker_verified(workers.get("stage0") or {}, "stage0")
        and _worker_verified(workers.get("stage1") or {}, "stage1")
        and set(rendezvous.get("registered_roles") or []) == {"stage0", "stage1"}
        and set(registrations) == {"stage0", "stage1"}
        and all(
            int(registrations[role].get("pid") or 0) > 0
            and registrations[role].get("cuda_live") is True
            and int(registrations[role].get("cuda_device_index", -1)) == 0
            and bool(str(registrations[role].get("cuda_device_name_hash") or ""))
            for role in ("stage0", "stage1")
        )
        and len(rendezvous.get("completions") or []) == 2
        and len([item for item in payloads if item.get("kind") == "activation"]) == 4
        and len([item for item in payloads if item.get("kind") == "gradient"]) == 4
        and payload_contract
        and {int(miners[role].get("dataset_shard_index", -1)) for role in miners} == {0, 1}
        and len({int(miners[role].get("base_model_version", -1)) for role in miners}) == 1
        and len({int(miners[role].get("adapter_version", -1)) for role in miners}) == 1
        and all(
            len({str(miners[role].get(field) or "") for role in miners}) == 1
            and all(bool(str(miners[role].get(field) or "")) for role in miners)
            for field in (
                "model_manifest_hash",
                "base_model_hash",
                "base_adapter_hash",
            )
        )
        and state.get("round_status") == "aggregated"
        and int(state.get("adapter_version") or 0) == 1
        and int(state.get("outer_step") or 0) == 1
        and int(state.get("accepted_result_count") or 0) == 2
        and set(state.get("accepted_shard_indexes") or []) == {0, 1}
        and state.get("dense_diloco_aggregation") is True
        and feedback.get("error_feedback") is True
        and feedback.get("dense_reconstruction_with_residual_verified") is True
        and evaluation.get("validation_loss_reduced") is True
        and evaluation.get("cpu_adapter_changes_logits") is True
        and evaluation.get("cpu_cuda_logits_close") is True
        and evaluation.get("standard_peft_cpu_load") is True
        and evaluation.get("standard_peft_cuda_load") is True
        and set(checkpoint_bundles) == {"stage0", "stage1"}
        and all(
            checkpoint_bundles[role].get("preserved") is True
            and checkpoint_bundles[role].get("worker_hash_match") is True
            and checkpoint_bundles[role].get("contains_pipeline_and_miner_checkpoints") is True
            and int(checkpoint_bundles[role].get("byte_count") or 0) > 0
            and checkpoint_bundles[role].get("file_hash")
            == (workers[role].get("checkpoint_bundle") or {}).get("file_hash")
            for role in ("stage0", "stage1")
        )
        and all(
            cleanup.get(key) is True
            for key in (
                "kernels_deleted",
                "private_packages_removed",
                "coordinator_stopped",
                "tunnel_stopped",
                "private_runtime_removed",
                "checkpoint_bundles_preserved",
                "private_cleanup_state_removed",
            )
        )
        and (report.get("rendezvous_cleanup") or {}).get("private_payloads_removed") is True
    )


def _rejection_matrix_verified(report: dict[str, Any]) -> bool:
    checks = dict(report.get("checks") or {})
    return bool(
        report.get("ok") is True
        and report.get("private_tensors_removed") is True
        and all(
            (checks.get(name) or {}).get("rejected_as_expected") is True
            and (checks.get(name) or {}).get("code") == code
            for name, code in REQUIRED_REJECTION_CODES.items()
        )
    )


def _public_safety_errors(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    forbidden_keys = {
        "payload_b64",
        "logits_b64",
        "training_adapter_delta_b64",
        "coordinator_token",
        "kaggle_key",
        "kaggle_api_token",
        "raw_text",
        "activation_gradient",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            key_lower = str(key).lower()
            item_present = item is not None and item is not False and item != "" and item != [] and item != {}
            if key_lower in forbidden_keys and item_present:
                errors.append(f"public_sensitive_field:{path}.{key}")
            if key_lower.endswith("_path") and item not in {None, "", False}:
                errors.append(f"public_private_path_field:{path}.{key}")
            errors.extend(_public_safety_errors(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_public_safety_errors(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for fragment in ("kaggle_key", "kaggle_api_token", "bearer ", "authorization:", "cookie:"):
            if fragment in lowered:
                errors.append(f"public_sensitive_string:{path}:{fragment}")
        if any(fragment in value for fragment in ("/root/", "/tmp/", "/home/")):
            errors.append(f"public_absolute_private_path:{path}")
        if ".trycloudflare.com" in lowered:
            errors.append(f"public_private_tunnel_url:{path}")
    return errors


def check(report_path: str | Path, *, require_ready: bool = False) -> dict[str, Any]:
    report = _load(report_path)
    errors: list[str] = []
    if report.get("schema") != SCHEMA:
        errors.append("schema_mismatch")
    baseline = dict(report.get("cpu_foundation_baseline") or {})
    if baseline.get("goal_achieved") is not True or baseline.get("training_foundation_rc_ready") is not True:
        errors.append("cpu_training_foundation_baseline_missing")
    contracts = dict(report.get("runtime_contracts") or {})
    for field in (
        "cuda_lora_runtime_implemented",
        "cuda_stage_runtime_implemented",
        "fp16_autocast_supported",
        "grad_scaler_supported",
        "gradient_clipping_supported",
        "cuda_oom_classification_supported",
        "cpu_checkpoint_delta_compatibility_preserved",
        "authenticated_private_rendezvous_implemented",
        "remote_delta_materialization_implemented",
        "checkpoint_bundle_preservation_supported",
        "crash_recoverable_private_cleanup_ledger_supported",
    ):
        if contracts.get(field) is not True:
            errors.append(f"runtime_contract_missing:{field}")

    attempts = dict(report.get("allocation_attempts") or {})
    allocation_budget = allocation_budget_summary(attempts)
    reported_budget = dict(report.get("allocation_budget") or {})
    if allocation_budget.get("amendment_present") and not allocation_budget.get("amendment_valid"):
        errors.append("allocation_budget_amendment_invalid")
    if reported_budget and reported_budget != allocation_budget:
        errors.append("allocation_budget_summary_mismatch")
    single_attempts = list(attempts.get("single_kernel_attempts") or [])
    two_node_attempts = list(attempts.get("two_node_attempts") or [])
    if len(single_attempts) > int(allocation_budget["single_kernel_attempt_limit"]):
        errors.append("single_kernel_attempt_limit_exceeded")
    if len(two_node_attempts) > int(allocation_budget["two_node_attempt_limit"]):
        errors.append("two_node_attempt_limit_exceeded")

    route_preflight = dict(report.get("coordinator_route_preflight") or {})
    route_preflight_verified = False
    if route_preflight:
        route_preflight_verified = bool(
            route_preflight.get("schema")
            == "crowdtensor_cuda_training_coordinator_route_preflight_summary_v1"
            and route_preflight.get("verified") is True
            and route_preflight.get("authenticated_status_verified") is True
            and route_preflight.get("miner_auth_required_verified") is True
            and route_preflight.get("run_id_hash_verified") is True
            and int(route_preflight.get("stable_successes_observed") or 0)
            >= int(route_preflight.get("stable_successes_required") or 1)
            and route_preflight.get("allocation_started") is False
            and route_preflight.get("kernel_push_attempted") is False
            and route_preflight.get("live_gate_claimed") is False
            and route_preflight.get("cleanup_verified") is True
            and route_preflight.get("url_public") is False
            and route_preflight.get("credentials_public") is False
            and route_preflight.get("public_artifact_safe") is True
        )
        if not route_preflight_verified:
            errors.append("coordinator_route_preflight_invalid")

    single_gate = dict(report.get("single_kernel_gate") or {})
    two_node_gate = dict(report.get("two_node_gate") or {})
    single_source = str(report.get("single_kernel_gate_source") or "standalone_attempt")
    single_verified = _single_gate_verified(single_gate)
    if single_source == "two_node_stage0_embedded":
        single_verified = bool(
            single_verified
            and _embedded_single_gate_binding_verified(single_gate, two_node_gate)
        )
    elif single_source != "standalone_attempt":
        single_verified = False
        errors.append("single_kernel_gate_source_invalid")
    elif single_gate.get("evidence_source") == "two_node_stage0_kernel_embedded_single_gate":
        single_verified = False
        errors.append("embedded_single_kernel_gate_source_not_declared")
    two_node_verified = _two_node_gate_verified(two_node_gate)
    rejection_verified = _rejection_matrix_verified(dict(report.get("rejection_matrix") or {}))
    if not rejection_verified:
        errors.append("adapter_delta_rejection_matrix_missing")
    tests = dict(report.get("test_summary") or {})
    tests_verified = bool(
        tests.get("ok") is True
        and tests.get("cuda_training_tests_passed") is True
        and tests.get("cpu_training_regressions_passed") is True
        and tests.get("state_store_miner_coordinator_regressions_passed") is True
    )
    if not tests_verified:
        errors.append("required_test_summary_missing")
    public_safety = report.get("public_artifact_safe") is True and not _public_safety_errors(report)
    if not public_safety:
        errors.extend(_public_safety_errors(report))
    cleanup = dict(report.get("cleanup_summary") or {})
    cleanup_verified = bool(
        cleanup.get("all_kaggle_kernels_deleted") is True
        and cleanup.get("all_private_packages_removed") is True
        and cleanup.get("all_local_runtime_stopped") is True
        and cleanup.get("live_resources_left_running") is False
    )
    if not cleanup_verified:
        errors.append("cleanup_not_verified")

    ready = bool(
        single_verified
        and two_node_verified
        and rejection_verified
        and tests_verified
        and cleanup_verified
        and public_safety
    )
    if bool(report.get("training_cuda_two_node_rc_ready")) != ready:
        errors.append("readiness_claim_mismatch")
    if bool(report.get("goal_achieved")) != ready:
        errors.append("goal_achieved_claim_mismatch")
    if report.get("gpu_success_claimed") is not ready:
        errors.append("gpu_success_claim_mismatch")
    if require_ready and not ready:
        if not single_verified:
            errors.append("single_kernel_t4x2_live_gate_missing")
        if not two_node_verified:
            errors.append("two_kernel_cross_machine_live_gate_missing")
        errors.append("training_cuda_two_node_rc_not_ready")
    return {
        "schema": "crowdtensor_training_cuda_two_node_rc_check_v1",
        "ok": not errors,
        "error_count": len(errors),
        "errors": sorted(set(errors)),
        "training_cuda_two_node_rc_ready": ready,
        "goal_achieved": ready,
        "single_kernel_gate_verified": single_verified,
        "two_node_gate_verified": two_node_verified,
        "rejection_matrix_verified": rejection_verified,
        "tests_verified": tests_verified,
        "cleanup_verified": cleanup_verified,
        "public_artifact_safe": public_safety,
        "coordinator_route_preflight_verified": route_preflight_verified,
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
        print(
            f"training_cuda_two_node_rc_check ok={result['ok']} "
            f"ready={result['training_cuda_two_node_rc_ready']} errors={result['error_count']}"
        )
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
