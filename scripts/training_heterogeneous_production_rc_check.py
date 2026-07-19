#!/usr/bin/env python3
"""Validate a public-safe heterogeneous Training Production RC artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from crowdtensor.heterogeneous_training_manifest import stable_hash
from scripts.training_cuda_kaggle_common import public_safety_errors


SCHEMA = "crowdtensor_heterogeneous_training_production_rc_v1"
CHECK_SCHEMA = "crowdtensor_heterogeneous_training_production_rc_check_v1"
LIVE_SCHEMA = "crowdtensor_heterogeneous_training_production_live_probe_v1"
REQUIRED_PROVIDERS = ["kaggle_cpu", "kaggle_cuda", "kaggle_jax_tpu"]
REQUIRED_REPLACEMENTS = ["cpu", "cuda", "jax_tpu"]
REQUIRED_KERNEL_ROLES = ["cpu", "gpu_a", "gpu_b", "tpu"]
RESTORE_EVIDENCE_SOURCES = {
    "stage_ready_history",
    "generation_fenced_contiguous_checkpoint_handoff",
}


def _read(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("training_production_rc_report_invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("training_production_rc_report_invalid")
    return value


def _hash_valid(value: str) -> bool:
    text = str(value)
    return bool(
        text.startswith("sha256:")
        and len(text) == 71
        and all(item in "0123456789abcdef" for item in text.split(":", 1)[1])
    )


def _content_hash_valid(report: dict[str, Any]) -> bool:
    supplied = str(report.get("content_hash") or "")
    return _hash_valid(supplied) and supplied == stable_hash(
        {key: value for key, value in report.items() if key != "content_hash"}
    )


def _finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def replacement_evidence_ready(value: dict[str, Any]) -> bool:
    row = dict(value or {})
    removed = int(row.get("removed_after_step") or 0)
    first = int(row.get("replacement_first_step") or 0)
    restored = int(row.get("restored_checkpoint_step") or 0)
    previous_generation = int(row.get("previous_generation") or 0)
    replacement_generation = int(row.get("replacement_generation") or 0)
    old_stage_ids = [int(item) for item in row.get("old_stage_ids") or []]
    replacement_stage_ids = [
        int(item) for item in row.get("replacement_stage_ids") or []
    ]
    source_hashes = list(row.get("source_checkpoint_archive_hashes") or [])
    replacement_hashes = list(
        row.get("replacement_checkpoint_archive_hashes") or []
    )
    source = str(row.get("restore_evidence_source") or "")
    selection = str(row.get("replacement_selection") or "")
    return bool(
        row.get("verified") is True
        and row.get("identity_changed") is True
        and row.get("old_worker_drained") is True
        and row.get("replacement_worker_accepted") is True
        and row.get("same_stage_handoff_verified") is True
        and row.get("contiguous_step_handoff_verified") is True
        and row.get("generation_fencing_verified") is True
        and row.get("checkpoint_restore_verified") is True
        and removed >= 1
        and first == removed + 1
        and restored == removed
        and replacement_generation > previous_generation > 0
        and int(row.get("checkpoint_download_count") or 0) >= 1
        and _hash_valid(row.get("old_identity_hash") or "")
        and _hash_valid(row.get("replacement_identity_hash") or "")
        and row.get("old_identity_hash") != row.get("replacement_identity_hash")
        and old_stage_ids
        and old_stage_ids == replacement_stage_ids
        and source_hashes
        and replacement_hashes
        and all(_hash_valid(item) for item in source_hashes + replacement_hashes)
        and source in RESTORE_EVIDENCE_SOURCES
        and (
            source != "stage_ready_history"
            or row.get("checkpoint_ready_event_matched") is True
        )
        and selection
        in {"designated_replacement", "cross_kernel_dynamic_reassignment"}
        and bool(row.get("old_kernel_role"))
        and bool(row.get("replacement_kernel_role"))
        and (
            selection != "cross_kernel_dynamic_reassignment"
            or row.get("old_kernel_role") != row.get("replacement_kernel_role")
        )
    )


def check_report(report: dict[str, Any], *, require_ready: bool = False) -> dict[str, Any]:
    errors = []

    def require(condition: bool, code: str) -> None:
        if not condition:
            errors.append(code)

    require(report.get("schema") == SCHEMA, "training_production_schema_invalid")
    require(_content_hash_valid(report), "training_production_content_hash_invalid")
    require(
        report.get("public_artifact_safe") is True,
        "training_production_public_safety_flag_missing",
    )
    safety = public_safety_errors(report)
    if safety:
        errors.extend(f"training_production_public_safety:{item}" for item in safety)
    require(
        report.get("model_id") == "Qwen/Qwen2.5-7B",
        "training_production_model_invalid",
    )
    require(
        int(report.get("minimum_required_steps") or 0) >= 100,
        "training_production_minimum_steps_invalid",
    )
    require(
        int(report.get("minimum_required_duration_seconds") or 0) >= 3600,
        "training_production_minimum_duration_invalid",
    )
    require(
        float(report.get("minimum_performance_improvement_fraction") or 0.0)
        >= 0.15,
        "training_production_performance_threshold_invalid",
    )
    workflow = dict(report.get("workflow_summary") or {})
    require(
        workflow.get("source_schema")
        == "crowdtensor_heterogeneous_training_production_workflow_probe_v1",
        "training_production_workflow_source_invalid",
    )
    require(
        workflow.get("workflow_verified") is True,
        "training_production_workflow_missing",
    )
    require(
        workflow.get("monitoring_contract_verified") is True,
        "training_production_monitoring_contract_missing",
    )
    require(
        workflow.get("cleanup_verified") is True,
        "training_production_workflow_cleanup_missing",
    )
    require(
        workflow.get("next_resume_command_uses_public_placeholder") is True,
        "training_production_workflow_resume_path_redaction_missing",
    )
    fault = dict(report.get("fault_summary") or {})
    require(
        fault.get("source_schema")
        == "crowdtensor_heterogeneous_training_production_fault_probe_v1",
        "training_production_fault_source_invalid",
    )
    for field in (
        "fault_injection_suite_ready",
        "generation_fencing_verified",
        "lease_reclaim_verified",
        "circuit_breaker_verified",
        "checkpoint_fallback_verified",
        "coordinator_journal_recovery_verified",
        "bounded_retry_verified",
        "cleanup_verified",
    ):
        require(
            fault.get(field) is True,
            f"training_production_fault_gate_missing:{field}",
        )
    live = dict(report.get("live_summary") or {})
    if live:
        require(
            live.get("source_schema") == LIVE_SCHEMA,
            "training_production_live_source_invalid",
        )
        require(
            live.get("source_content_hash_valid") is True,
            "training_production_live_source_hash_invalid",
        )
    ready = report.get("training_production_rc_ready") is True
    if ready or require_ready:
        require(ready, "training_production_rc_not_ready")
        require(
            live.get("live_run_performed") is True,
            "training_production_live_run_missing",
        )
        require(
            live.get("external_runtime_verified") is True,
            "training_production_external_runtime_missing",
        )
        require(
            sorted(live.get("accepted_providers") or []) == REQUIRED_PROVIDERS,
            "training_production_provider_coverage_missing",
        )
        steps = [int(item) for item in live.get("committed_steps") or []]
        require(
            len(steps) >= 100
            and steps == list(range(1, len(steps) + 1)),
            "training_production_commit_ledger_invalid",
        )
        require(
            int(live.get("committed_step_count") or 0) == len(steps),
            "training_production_commit_count_mismatch",
        )
        require(
            float(live.get("soak_duration_seconds") or 0.0) >= 3600.0,
            "training_production_soak_duration_short",
        )
        require(
            float(live.get("full_live_gate_elapsed_seconds") or 0.0) <= 21600.0,
            "training_production_live_gate_unbounded",
        )
        require(
            int(live.get("maximum_checkpoint_interval_steps") or 0) <= 10
            and int(live.get("maximum_checkpoint_interval_steps") or 0) >= 1,
            "training_production_checkpoint_interval_invalid",
        )
        for field in (
            "finite_updates_all_stages",
            "changed_lora_hashes_all_stages",
            "atomic_ledger_verified",
            "checkpoint_integrity_verified",
            "adapter_cpu_reload_verified",
            "activation_gradient_transfer_verified",
            "monitoring_live_verified",
            "coordinator_restart_live_verified",
            "stale_result_rejected",
        ):
            require(
                live.get(field) is True,
                f"training_production_live_gate_missing:{field}",
            )
        replacements = dict(live.get("worker_replacements") or {})
        for provider in REQUIRED_REPLACEMENTS:
            row = dict(replacements.get(provider) or {})
            require(
                replacement_evidence_ready(row),
                f"training_production_live_replacement_missing:{provider}",
            )
        kernel_evidence = [
            dict(item) for item in live.get("kernel_evidence") or []
        ]
        require(
            live.get("effective_kernel_evidence_verified") is True,
            "training_production_effective_kernel_evidence_missing",
        )
        require(
            sorted(item.get("kernel_role") for item in kernel_evidence)
            == REQUIRED_KERNEL_ROLES,
            "training_production_kernel_role_coverage_missing",
        )
        for item in kernel_evidence:
            require(
                item.get("effective_ok") is True
                and item.get("worker_results_valid") is True
                and item.get("cleanup_verified") is True
                and _hash_valid(item.get("kernel_report_hash") or "")
                and (
                    item.get("raw_ok") is True
                    or (
                        item.get("raw_failure_reclassified") is True
                        and item.get("automatic_cross_kernel_takeover_observed")
                        is True
                    )
                ),
                "training_production_kernel_evidence_invalid:"
                + str(item.get("kernel_role") or "unknown"),
            )
        replay = dict(live.get("evidence_replay") or {})
        if replay:
            require(
                replay.get("schema")
                == "crowdtensor_heterogeneous_training_production_live_replay_v1"
                and replay.get("source_schema") == LIVE_SCHEMA
                and replay.get("source_content_hash_valid") is True
                and _hash_valid(replay.get("source_content_hash") or "")
                and int(replay.get("raw_kernel_report_count") or 0) == 4
                and replay.get("source_live_run_reused") is True
                and replay.get("live_run_reexecuted") is False,
                "training_production_live_replay_invalid",
            )
        performance = dict(live.get("performance") or {})
        require(
            performance.get("schema")
            == "crowdtensor_heterogeneous_training_performance_comparison_v1",
            "training_production_performance_schema_invalid",
        )
        require(
            performance.get("performance_gate_passed") is True,
            "training_production_performance_gate_failed",
        )
        require(
            int(performance.get("baseline_window_count") or 0) >= 5
            and int(performance.get("candidate_window_count") or 0) >= 5,
            "training_production_performance_windows_insufficient",
        )
        require(
            performance.get("same_workload_verified") is True
            and performance.get("same_topology_verified") is True
            and performance.get("workload_or_topology_reduction_used") is False,
            "training_production_performance_identity_invalid",
        )
        require(
            max(
                float(performance.get("throughput_improvement_fraction") or 0.0),
                float(performance.get("p50_latency_improvement_fraction") or 0.0),
            )
            >= 0.15,
            "training_production_performance_improvement_short",
        )
        require(
            float(performance.get("p95_regression_fraction") or 0.0)
            <= float(
                performance.get("maximum_p95_regression_fraction") or 0.0
            ),
            "training_production_performance_p95_regressed",
        )
        optimization = dict(live.get("optimization_summary") or {})
        require(
            int(optimization.get("performance_window_count_per_phase") or 0)
            >= 5,
            "training_production_optimization_window_contract_missing",
        )
        require(
            int(optimization.get("inline_tensor_message_upload_count") or 0) > 0
            and int(optimization.get("inline_tensor_message_download_count") or 0)
            > 0,
            "training_production_inline_tensor_transport_missing",
        )
        require(
            int(
                optimization.get("large_payload_connection_isolation_count")
                or 0
            )
            > 0
            and 0
            < int(optimization.get("persistent_http_max_body_bytes") or 0)
            <= 4 * 1024 * 1024,
            "training_production_large_payload_isolation_missing",
        )
        benchmark = dict(live.get("benchmark") or {})
        for field in (
            "step_throughput_per_second",
            "p50_step_latency_seconds",
            "p95_step_latency_seconds",
            "checkpoint_overhead_seconds",
            "transfer_bytes",
        ):
            require(
                _finite_positive(benchmark.get(field)),
                f"training_production_benchmark_metric_invalid:{field}",
            )
        cleanup = dict(live.get("cleanup") or {})
        require(
            cleanup.get("all_remote_kernels_deleted") is True
            and cleanup.get("temporary_private_packages_removed") is True
            and cleanup.get("coordinator_stopped") is True
            and cleanup.get("tunnel_stopped") is True
            and cleanup.get("tensor_payloads_removed") is True
            and cleanup.get("live_resources_left_running") is False,
            "training_production_live_cleanup_invalid",
        )
        quality = dict(report.get("quality_summary") or {})
        require(
            int(quality.get("passed") or 0) >= 1
            and int(quality.get("failed", -1)) == 0,
            "training_production_regression_missing",
        )
        require(
            not report.get("blockers"),
            "training_production_ready_report_has_blockers",
        )
    else:
        require(
            bool(report.get("blockers")),
            "training_production_blocker_report_missing_blockers",
        )
    result = {
        "schema": CHECK_SCHEMA,
        "ok": not errors,
        "error_count": len(errors),
        "errors": errors,
        "training_production_rc_ready": ready,
        "public_artifact_safe": not safety,
        "public_safety_errors": safety,
        "report_content_hash": str(report.get("content_hash") or ""),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = check_report(_read(args.report), require_ready=args.require_ready)
    except ValueError as exc:
        result = {
            "schema": CHECK_SCHEMA,
            "ok": False,
            "error_count": 1,
            "errors": [str(exc)],
            "training_production_rc_ready": False,
            "public_artifact_safe": False,
            "public_safety_errors": [],
        }
    print(json.dumps(result, sort_keys=True) if args.json else result)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
