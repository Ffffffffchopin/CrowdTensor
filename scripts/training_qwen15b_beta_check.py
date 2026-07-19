#!/usr/bin/env python3
"""Check the Qwen 1.5B four-GPU Training Service Beta RC artifact."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA = "crowdtensor_training_qwen15b_beta_v1"
MODEL_ID = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
ALPHA_FINISHED_AT = "2026-07-12T09:12:07+00:00"
AUTHORITATIVE_ALPHA_HASH = "sha256:7dca6b5d84d7bdaebd647e1efe488ca881d5b96b03c3ad916b5fd9b6da8587c1"
SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
PHASES = {
    "model_resolution",
    "dataset",
    "account_preflight",
    "allocation",
    "kernel_launch",
    "stage_loading",
    "forward",
    "backward",
    "checkpoint",
    "recovery",
    "evaluation",
    "export",
    "cleanup",
}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def _source_errors(report: dict[str, Any]) -> list[str]:
    errors = []
    source = _dict(report.get("source"))
    manifest = _dict(source.get("source"))
    ownership = _dict(source.get("ownership"))
    stages = _list(ownership.get("stages"))
    if (
        source.get("ok") is not True
        or manifest.get("model_id") != MODEL_ID
        or manifest.get("model_revision") != MODEL_REVISION
        or int(manifest.get("parameter_count") or 0) < 1_000_000_000
    ):
        errors.append("training_beta_pinned_real_source_invalid")
    if (
        len(stages) != 4
        or [(int(item.get("layer_start", -1)), int(item.get("layer_end", -1))) for item in stages]
        != [(0, 7), (7, 14), (14, 21), (21, 28)]
        or ownership.get("all_source_tensors_covered") is not True
        or ownership.get("four_distinct_kernel_device_placements") is not True
    ):
        errors.append("training_beta_four_stage_source_ownership_invalid")
    dataset = _dict(report.get("dataset"))
    dataset_manifest = _dict(dataset.get("manifest"))
    if (
        dataset.get("ok") is not True
        or dataset.get("private_payload_present") is not True
        or dataset_manifest.get("model_id") != MODEL_ID
        or dataset_manifest.get("model_revision") != MODEL_REVISION
        or dataset_manifest.get("dataset_id") != "Salesforce/wikitext"
        or int(dataset_manifest.get("train_sequence_count") or 0) < 32
        or int(dataset_manifest.get("validation_sequence_count") or 0) < 4
        or dataset.get("raw_text_public") is not False
        or dataset.get("token_ids_public") is not False
    ):
        errors.append("training_beta_private_dataset_invalid")
    return errors


def _service_errors(report: dict[str, Any]) -> list[str]:
    smoke = _dict(report.get("service_smoke"))
    required = (
        "health_route_ready",
        "authentication_required",
        "submit_route_ready",
        "submit_idempotent",
        "status_route_ready",
        "resume_route_ready",
        "cancel_route_ready",
        "running_cancel_marker_ready",
        "export_route_ready",
        "cleanup_route_ready",
        "artifacts_route_ready",
        "events_route_ready",
        "persistent_process_restart_recovery_verified",
        "bounded_queue_ready",
        "one_live_gpu_job_enforced",
        "private_inputs_redacted",
    )
    errors = []
    if (
        smoke.get("schema") != "crowdtensor_training_qwen15b_beta_service_smoke_v1"
        or smoke.get("ok") is not True
        or any(smoke.get(key) is not True for key in required)
        or int(smoke.get("recovered_global_step") or 0) != 4
        or smoke.get("live_gpu_run_performed") is not False
    ):
        errors.append("training_beta_authenticated_service_contract_invalid")
    store = _dict(report.get("job_store_summary"))
    if (
        store.get("persistent_sqlite") is not True
        or store.get("one_live_gpu_job") is not True
        or int(store.get("max_queue_size") or 0) < 1
        or int(store.get("event_count") or 0) < len(PHASES)
        or store.get("event_ids_unique") is not True
        or store.get("global_step_monotonic") is not True
    ):
        errors.append("training_beta_persistent_job_store_invalid")
    return errors


def _test_errors(report: dict[str, Any]) -> list[str]:
    tests = _dict(report.get("test_summary"))
    faults = _dict(tests.get("fault_injection"))
    required_faults = (
        "duplicate_submission_rejected_or_idempotent",
        "expired_lease_recovery_verified",
        "corrupted_checkpoint_rejected",
        "non_finite_tensor_rejected",
        "worker_timeout_classified",
        "coordinator_unavailable_retry_verified",
    )
    if (
        tests.get("ok") is not True
        or int(tests.get("failed") or 0) != 0
        or int(tests.get("passed") or 0) < 313
        or tests.get("existing_313_regressions_included") is not True
        or tests.get("beta_service_regressions_included") is not True
        or any(faults.get(key) is not True for key in required_faults)
    ):
        return ["training_beta_required_regressions_or_fault_matrix_missing"]
    return []


def _job_errors(report: dict[str, Any]) -> list[str]:
    status = _dict(report.get("job_status"))
    phases = _dict(status.get("phases"))
    errors = []
    if (
        status.get("overall_state") not in {"completed", "cleaned"}
        or int(status.get("global_step") or 0) != 8
        or status.get("user_command_path_executed") is not True
        or status.get("prebuilt_dist_inputs_used") is not False
        or _dict(status.get("input_preparation")).get("generated_by_user_command") is not True
        or _dict(status.get("input_preparation")).get("prebuilt_dist_inputs_used") is not False
        or not PHASES.issubset(phases)
        or any(_dict(phases.get(phase)).get("state") != "completed" for phase in PHASES)
    ):
        errors.append("training_beta_ordinary_user_job_path_incomplete")
    if (
        status.get("credential_values_public") is True
        or status.get("credential_paths_public") is True
        or status.get("private_paths_public") is not False
    ):
        errors.append("training_beta_job_status_public_safety_invalid")
    user_export = _dict(report.get("user_export"))
    if (
        user_export.get("schema") != "crowdtensor_qwen15b_training_export_v1"
        or user_export.get("ok") is not True
        or user_export.get("model") != MODEL_ID
        or user_export.get("model_revision") != MODEL_REVISION
        or user_export.get("standard_peft_layout") is not True
        or not _sha(user_export.get("adapter_model_hash"))
        or not _sha(user_export.get("adapter_config_hash"))
    ):
        errors.append("training_beta_ordinary_user_export_missing")
    job_cleanup = _dict(report.get("job_cleanup"))
    if (
        job_cleanup.get("schema") != "crowdtensor_qwen15b_training_job_cleanup_v1"
        or job_cleanup.get("ok") is not True
        or job_cleanup.get("temporary_kaggle_kernels_deleted") is not True
        or job_cleanup.get("only_recorded_job_kernel_refs_targeted") is not True
        or job_cleanup.get("live_resources_left_running") is not False
        or job_cleanup.get("temporary_private_runtime_removed") is not True
        or job_cleanup.get("checkpoint_and_evidence_preserved") is not True
    ):
        errors.append("training_beta_ordinary_user_cleanup_missing")
    return errors


def _live_errors(report: dict[str, Any]) -> list[str]:
    live = _dict(report.get("live_report"))
    evidence = _dict(live.get("evidence"))
    rendezvous = _dict(live.get("rendezvous"))
    workers = _list(live.get("worker_reports"))
    by_role = {str(item.get("role") or ""): item for item in workers}
    errors = []
    try:
        fresh = datetime.fromisoformat(str(live.get("started_at"))) > datetime.fromisoformat(
            ALPHA_FINISHED_AT
        )
    except (TypeError, ValueError):
        fresh = False
    if (
        not fresh
        or live.get("beta_mode") is not True
        or live.get("live_run_performed") is not True
        or live.get("mock_runtime_used") is not False
        or live.get("cpu_fallback_used") is not False
        or live.get("tiny_or_random_model_used") is not False
        or live.get("training_qwen15b_beta_live_verified") is not True
        or live.get("ok") is not True
        or int(live.get("coordinator_restart_after_step") or 0) != 4
        or live.get("requested_model") != MODEL_ID
        or live.get("requested_model_revision") != MODEL_REVISION
    ):
        errors.append("training_beta_fresh_live_evidence_missing")
    if (
        live.get("same_authorized_account") is not True
        or live.get("multi_account_gate_substitution") is not False
        or int(live.get("requested_kernel_count") or 0) != 2
        or live.get("requested_accelerator") != "NvidiaTeslaT4"
        or int(live.get("max_observed_running_kernel_count") or 0) < 2
        or set(by_role) != {"kernel_a", "kernel_b"}
        or any(item.get("ok") is not True for item in workers)
    ):
        errors.append("training_beta_same_account_two_t4x2_live_invalid")

    optimizer_identities = []
    recovery_records = []
    stage_ids = set()
    for role, outer in by_role.items():
        worker = _dict(outer.get("worker"))
        expected = {0, 1} if role == "kernel_a" else {2, 3}
        if (
            worker.get("model_id") != MODEL_ID
            or worker.get("model_revision") != MODEL_REVISION
            or int(worker.get("parameter_count") or 0) < 1_000_000_000
            or worker.get("base_weights_frozen") is not True
            or worker.get("positive_lora_gradient_norms") is not True
            or worker.get("coordinator_restart_owned_stages_verified") is not True
        ):
            errors.append("training_beta_real_worker_contract_invalid")
        reliability = _dict(worker.get("transport_reliability"))
        if (
            reliability.get("bounded_retry_enabled") is not True
            or int(reliability.get("retry_attempt_limit") or 0) < 2
            or int(reliability.get("retry_count") or 0) < 1
            or int(reliability.get("reconnect_registration_count") or 0) < 1
        ):
            errors.append("training_beta_worker_retry_reregistration_missing")
        role_recoveries = _list(worker.get("coordinator_restart_stage_recoveries"))
        recovery_records.extend(role_recoveries)
        if {int(item.get("stage_id", -1)) for item in role_recoveries} != expected:
            errors.append("training_beta_all_stage_checkpoint_recovery_missing")
        for item in role_recoveries:
            if (
                int(item.get("after_step") or 0) != 4
                or int(item.get("old_pid") or 0) <= 0
                or int(item.get("new_pid") or 0) <= 0
                or item.get("old_pid") == item.get("new_pid")
                or item.get("checkpoint_resume_verified") is not True
                or int(item.get("resumed_global_step") or 0) != 4
                or int(item.get("resumed_dataset_cursor") or 0) != 16
                or not _sha(item.get("loaded_checkpoint_hash"))
            ):
                errors.append("training_beta_checkpoint_recovery_record_invalid")
        ready_by_run = _dict(worker.get("stage_ready"))
        runs = _dict(worker.get("runs"))
        for run_kind in ("baseline", "resumed"):
            ready = _list(ready_by_run.get(run_kind))
            run = _dict(runs.get(run_kind))
            stage_ids.update(int(item.get("stage_id", -1)) for item in ready)
            if (
                len(ready) != 2
                or {int(item.get("stage_id", -1)) for item in ready} != expected
                or any(item.get("cuda_live") is not True for item in ready)
                or int(run.get("steps_completed") or 0) != 8
                or run.get("real_forward") is not True
                or run.get("real_backward") is not True
            ):
                errors.append("training_beta_four_cuda_stage_runtime_incomplete")
            steps = _list(run.get("step_reports"))
            if len(steps) != 8:
                errors.append("training_beta_eight_step_records_incomplete")
            for step in steps:
                number = int(step.get("step") or 0)
                stages = _list(step.get("stages"))
                if len(stages) != 2 or {int(item.get("stage_id", -1)) for item in stages} != expected:
                    errors.append("training_beta_per_step_stage_records_invalid")
                for stage in stages:
                    optimizer_identities.append(
                        (run_kind, int(stage.get("stage_id", -1)), number)
                    )
                    if (
                        int(stage.get("global_step") or 0) != number
                        or stage.get("optimizer_step_applied") is not True
                        or not _finite_positive(stage.get("lora_gradient_norm"))
                        or not _sha(stage.get("checkpoint_hash"))
                    ):
                        errors.append("training_beta_optimizer_checkpoint_record_invalid")
    if stage_ids != {0, 1, 2, 3}:
        errors.append("training_beta_four_live_stage_ids_missing")
    expected_optimizer_identities = {
        (run_kind, stage_id, step)
        for run_kind in ("baseline", "resumed")
        for stage_id in range(4)
        for step in range(1, 9)
    }
    if set(optimizer_identities) != expected_optimizer_identities or len(optimizer_identities) != 64:
        errors.append("training_beta_duplicate_or_missing_optimizer_step")
    if {int(item.get("stage_id", -1)) for item in recovery_records} != {0, 1, 2, 3}:
        errors.append("training_beta_all_stage_checkpoint_recovery_missing")

    payloads = _list(rendezvous.get("payloads"))
    payload_identities = {
        (
            str(item.get("run_kind") or ""),
            str(item.get("kind") or ""),
            int(item.get("step", -1)),
            int(item.get("microbatch", -1)),
        )
        for item in payloads
        if item.get("kind") in {"activation", "gradient"}
    }
    expected_payload_identities = {
        (run_kind, kind, step, microbatch)
        for run_kind in ("baseline", "resumed")
        for kind in ("activation", "gradient")
        for step in range(8)
        for microbatch in range(4)
    }
    if (
        sum(item.get("kind") == "activation" for item in payloads) != 64
        or sum(item.get("kind") == "gradient" for item in payloads) != 64
        or sum(item.get("kind") == "stage_adapter" for item in payloads) != 1
        or payload_identities != expected_payload_identities
        or any(not _sha(item.get("payload_hash")) for item in payloads)
        or any(int(item.get("byte_count") or 0) <= 0 for item in payloads)
        or any(int(item.get("tensor_count") or 0) <= 0 for item in payloads)
        or any(
            item.get("producer_role")
            != ("kernel_a" if item.get("kind") in {"activation", "stage_adapter"} else "kernel_b")
            for item in payloads
        )
    ):
        errors.append("training_beta_cross_kernel_payload_evidence_invalid")
    restart = _dict((list(rendezvous.get("coordinator_restarts") or [{}]))[-1])
    if (
        rendezvous.get("persistent_state_enabled") is not True
        or rendezvous.get("recovered_from_persistent_state") is not True
        or rendezvous.get("coordinator_restart_verified") is not True
        or int(restart.get("after_step") or 0) != 4
        or rendezvous.get("post_restart_registered_roles") != ["kernel_a", "kernel_b"]
    ):
        errors.append("training_beta_persistent_coordinator_restart_missing")
    if (
        evidence.get("beta_recovery_verified") is not True
        or evidence.get("optimizer_steps_unique") is not True
        or evidence.get("four_stage_compute_overlap_verified") is not True
        or int(evidence.get("activation_payload_count") or 0) != 64
        or int(evidence.get("gradient_payload_count") or 0) != 64
        or evidence.get("resume_adapter_equivalence_verified") is not True
        or evidence.get("resume_loss_equivalence_verified") is not True
        or evidence.get("loss_reduced") is not True
    ):
        errors.append("training_beta_derived_live_evidence_invalid")
    kernel_b = _dict(_dict(by_role.get("kernel_b")).get("worker"))
    evaluation = _dict(kernel_b.get("evaluation"))
    export = _dict(kernel_b.get("export"))
    if (
        evaluation.get("evaluation_verified") is not True
        or evaluation.get("standard_peft_cpu_load") is not True
        or evaluation.get("standard_peft_cuda_load") is not True
        or evaluation.get("adapter_changes_logits") is not True
        or evaluation.get("validation_loss_reduced") is not True
        or export.get("standard_peft_format") is not True
        or _list(export.get("layer_indexes")) != list(range(28))
    ):
        errors.append("training_beta_peft_export_evaluation_invalid")
    checkpoint_bundles = _list(live.get("checkpoint_bundles"))
    adapter_bundle = _dict(live.get("adapter_bundle"))
    if (
        len(checkpoint_bundles) != 2
        or {str(item.get("role") or "") for item in checkpoint_bundles}
        != {"kernel_a", "kernel_b"}
        or any(
            item.get("verified") is not True
            or item.get("preserved") is not True
            or item.get("all_checkpoint_files_hash_verified") is not True
            or item.get("all_manifest_content_hashes_verified") is not True
            or int(item.get("checkpoint_manifest_count") or 0) != 4
            or not _sha(item.get("file_hash"))
            for item in checkpoint_bundles
        )
        or adapter_bundle.get("verified") is not True
        or adapter_bundle.get("preserved") is not True
        or adapter_bundle.get("standard_peft_layout") is not True
        or adapter_bundle.get("safetensors_header_verified") is not True
        or adapter_bundle.get("model_revision_verified") is not True
        or not _sha(adapter_bundle.get("file_hash"))
    ):
        errors.append("training_beta_archive_integrity_invalid")
    cleanup = _dict(live.get("cleanup"))
    for key in (
        "kernels_deleted",
        "only_attempt_kernel_refs_targeted",
        "private_packages_removed",
        "coordinator_stopped",
        "tunnel_stopped",
        "private_runtime_removed",
        "rendezvous_private_payloads_removed",
        "checkpoint_archives_verified_before_cleanup",
    ):
        if cleanup.get(key) is not True:
            errors.append("training_beta_cleanup_incomplete")
            break
    benchmark = _dict(report.get("benchmark"))
    step_latencies = _list(benchmark.get("step_latencies"))
    latency_identities = {
        (str(item.get("run_kind") or ""), int(item.get("step") or 0))
        for item in step_latencies
    }
    if (
        benchmark.get("benchmark_complete") is not True
        or benchmark.get("completed_within_1800_seconds") is not True
        or int(benchmark.get("step_latency_count") or 0) != 16
        or int(benchmark.get("private_network_payload_count") or 0) != 129
        or int(benchmark.get("private_network_bytes") or 0) <= 0
        or int(benchmark.get("peak_gpu_allocated_bytes") or 0) <= 0
        or not _finite_positive(benchmark.get("coordinator_recovery_seconds"))
        or latency_identities
        != {(run_kind, step) for run_kind in ("baseline", "resumed") for step in range(1, 9)}
        or any(not _finite_positive(item.get("latency_ms")) for item in step_latencies)
    ):
        errors.append("training_beta_benchmark_incomplete")
    artifacts = _dict(report.get("artifacts"))
    if any(
        _dict(artifacts.get(name)).get("present") is not True
        for name in ("live_report", "benchmark", "user_export", "job_cleanup")
    ):
        errors.append("training_beta_fresh_live_artifacts_missing")
    allocation = _dict(report.get("allocation_summary"))
    if (
        allocation.get("beta_goal_authorization") is not True
        or int(allocation.get("attempt_limit") or 0) != 3
        or int(allocation.get("attempt_count") or 0) < 1
        or int(allocation.get("attempt_count") or 0) > 3
        or allocation.get("latest_outcome") != "verified"
        or allocation.get("attempt_numbers_sequential") is not True
        or allocation.get("all_attempts_completed") is not True
        or allocation.get("same_authorized_account_per_attempt") is not True
        or allocation.get("automatic_retry_loop") is not False
    ):
        errors.append("training_beta_goal_allocation_ledger_invalid")
    return sorted(set(errors))


def _public_safety_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, default=str)
    lowered = encoded.lower()
    errors = []
    forbidden = {
        "training_beta_public_payload_b64": '"payload_b64"',
        "training_beta_public_token_ids": '"token_ids":',
        "training_beta_public_activation": '"activation":',
        "training_beta_public_gradient": '"activation_gradient":',
        "training_beta_public_raw_text": '"raw_training_text":',
        "training_beta_public_root_path": "/root/",
        "training_beta_public_tmp_path": "/tmp/",
        "training_beta_public_kaggle_path": "/kaggle/",
        "training_beta_public_bearer": "bearer ",
        "training_beta_public_cookie": "cookie:",
        "training_beta_public_tunnel": "trycloudflare.com",
    }
    for code, fragment in forbidden.items():
        if fragment in lowered:
            errors.append(code)
    if re.search(r"KGA[A-Za-z0-9_-]{8,}", encoded):
        errors.append("training_beta_public_kaggle_token")
    return errors


def check(report: dict[str, Any], *, require_ready: bool = False) -> dict[str, Any]:
    structural = []
    if report.get("schema") != SCHEMA:
        structural.append("training_beta_schema_invalid")
    alpha = _dict(report.get("authoritative_alpha"))
    if (
        alpha.get("goal_achieved") is not True
        or alpha.get("qwen15b_four_gpu_alpha_ready") is not True
        or alpha.get("reused_without_rewrite") is not True
        or alpha.get("artifact_hash") != AUTHORITATIVE_ALPHA_HASH
    ):
        structural.append("training_beta_authoritative_alpha_invalid")
    if (
        report.get("model_id") != MODEL_ID
        or report.get("model_revision") != MODEL_REVISION
        or report.get("topology") != "kaggle-2x-t4x2"
        or int(report.get("steps") or 0) != 8
    ):
        structural.append("training_beta_requested_contract_invalid")
    artifacts = _dict(report.get("artifacts"))
    required_artifacts = {
        "authoritative_alpha",
        "source_report",
        "dataset_report",
        "service_smoke",
        "test_summary",
        "job_status",
        "user_export",
        "job_cleanup",
        "live_report",
        "benchmark",
        "allocation_ledger",
    }
    required_present_artifacts = required_artifacts - {
        "live_report",
        "benchmark",
        "user_export",
        "job_cleanup",
    }
    if set(artifacts) != required_artifacts or any(
        _dict(artifacts.get(name)).get("present") is not True
        or not _sha(_dict(artifacts.get(name)).get("file_hash"))
        or int(_dict(artifacts.get(name)).get("byte_count") or 0) <= 0
        for name in required_present_artifacts
    ) or any(
        _dict(artifacts.get(name)).get("present") is True
        and (
            not _sha(_dict(artifacts.get(name)).get("file_hash"))
            or int(_dict(artifacts.get(name)).get("byte_count") or 0) <= 0
        )
        for name in ("live_report", "benchmark", "user_export", "job_cleanup")
    ):
        structural.append("training_beta_artifact_manifest_invalid")
    for key in (
        "raw_training_text_public",
        "token_ids_public",
        "activation_values_public",
        "gradient_values_public",
        "adapter_tensor_values_public",
        "credential_values_public",
        "credential_paths_public",
        "coordinator_token_public",
        "coordinator_url_public",
        "private_paths_public",
    ):
        if report.get(key) is not False:
            structural.append("training_beta_public_safety_contract_invalid")
            break
    structural.extend(_source_errors(report))
    structural.extend(_service_errors(report))
    structural.extend(_test_errors(report))
    structural.extend(_public_safety_errors(report))
    readiness = _job_errors(report) + _live_errors(report)
    ready = not structural and not readiness
    if bool(report.get("goal_achieved")) != ready:
        structural.append("training_beta_goal_achieved_flag_incoherent")
    if bool(report.get("training_qwen15b_beta_ready")) != ready:
        structural.append("training_beta_ready_flag_incoherent")
    errors = sorted(set(structural + (readiness if require_ready else [])))
    return {
        "schema": "crowdtensor_training_qwen15b_beta_check_v1",
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors,
        "readiness_error_count": len(set(readiness)),
        "readiness_errors": sorted(set(readiness)),
        "training_qwen15b_beta_ready": ready,
        "goal_achieved": ready,
        "require_ready": bool(require_ready),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check(json.loads(Path(args.report).read_text(encoding="utf-8")), require_ready=args.require_ready)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            f"training_qwen15b_beta_check ok={result['ok']} "
            f"ready={result['training_qwen15b_beta_ready']} errors={result['error_count']}"
        )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
