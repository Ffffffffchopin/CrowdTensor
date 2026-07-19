#!/usr/bin/env python3
"""Pack Training Production engineering and live evidence into one RC report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from crowdtensor.heterogeneous_training_manifest import stable_hash
from scripts.training_cuda_kaggle_common import public_safety_errors
from scripts.training_heterogeneous_production_rc_check import (
    LIVE_SCHEMA,
    REQUIRED_PROVIDERS,
    SCHEMA,
    check_report,
    replacement_evidence_ready,
)


def _read(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("training_production_pack_source_invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("training_production_pack_source_invalid")
    return value


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _content_hash_valid(value: dict[str, Any]) -> bool:
    supplied = str(value.get("content_hash") or "")
    return bool(
        supplied
        and supplied
        == stable_hash(
            {key: item for key, item in value.items() if key != "content_hash"}
        )
    )


def _artifact(path: str | Path, value: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": Path(path).name,
        "file_sha256": _sha(path),
        "content_hash": str(value.get("content_hash") or ""),
        "content_hash_valid": _content_hash_valid(value),
    }


def _workflow_summary(value: dict[str, Any]) -> dict[str, Any]:
    cleanup = dict(value.get("cleanup") or {})
    return {
        "source_schema": str(value.get("schema") or ""),
        "source_content_hash": str(value.get("content_hash") or ""),
        "source_content_hash_valid": _content_hash_valid(value),
        "workflow_verified": value.get("ok") is True,
        "dry_run_verified": value.get("dry_run_verified") is True,
        "idempotent_start_verified": value.get("idempotent_start_verified") is True,
        "cli_complete_lifecycle_verified": value.get(
            "cli_complete_lifecycle_verified"
        )
        is True,
        "monitoring_contract_verified": value.get(
            "monitoring_contract_verified"
        )
        is True,
        "next_resume_command_redacts_credentials": value.get(
            "next_resume_command_redacts_credentials"
        )
        is True,
        "next_resume_command_uses_public_placeholder": value.get(
            "next_resume_command_uses_public_placeholder"
        )
        is True,
        "cleanup_verified": bool(
            cleanup.get("active_miner_leases_revoked") is True
            and cleanup.get("live_resources_left_running") is False
            and cleanup.get("temporary_private_runtime_removed") is True
        ),
        "public_artifact_safe": value.get("public_artifact_safe") is True,
    }


def _fault_summary(value: dict[str, Any]) -> dict[str, Any]:
    cleanup = dict(value.get("cleanup") or {})
    return {
        "source_schema": str(value.get("schema") or ""),
        "source_content_hash": str(value.get("content_hash") or ""),
        "source_content_hash_valid": _content_hash_valid(value),
        "fault_injection_suite_ready": value.get("fault_injection_suite_ready")
        is True,
        "generation_fencing_verified": value.get("generation_fencing_verified")
        is True,
        "lease_reclaim_verified": value.get("lease_reclaim_verified") is True,
        "circuit_breaker_verified": value.get("circuit_breaker_verified") is True,
        "checkpoint_fallback_verified": value.get("checkpoint_fallback_verified")
        is True,
        "coordinator_journal_recovery_verified": value.get(
            "coordinator_journal_recovery_verified"
        )
        is True,
        "bounded_retry_verified": value.get("bounded_retry_verified") is True,
        "cleanup_verified": bool(
            cleanup.get("runtime_cleaned") is True
            and cleanup.get("active_miner_leases_revoked") is True
            and cleanup.get("live_resources_left_running") is False
            and cleanup.get("temporary_private_runtime_removed") is True
        ),
        "public_artifact_safe": value.get("public_artifact_safe") is True,
    }


def _live_summary(value: dict[str, Any]) -> dict[str, Any]:
    if not value:
        return {}
    training = dict(value.get("training_evidence") or {})
    return {
        "source_schema": str(value.get("schema") or ""),
        "source_content_hash": str(value.get("content_hash") or ""),
        "source_content_hash_valid": _content_hash_valid(value),
        "live_run_performed": value.get("live_run_performed") is True,
        "external_runtime_verified": value.get("external_runtime_verified")
        is True,
        "accepted_providers": sorted(value.get("accepted_providers") or []),
        "committed_steps": [int(item) for item in training.get("committed_steps") or []],
        "committed_step_count": int(training.get("committed_step_count") or 0),
        "soak_duration_seconds": float(value.get("soak_duration_seconds") or 0.0),
        "full_live_gate_elapsed_seconds": float(
            value.get("full_live_gate_elapsed_seconds") or 0.0
        ),
        "maximum_checkpoint_interval_steps": int(
            training.get("maximum_checkpoint_interval_steps") or 0
        ),
        "finite_updates_all_stages": training.get("finite_updates_all_stages")
        is True,
        "changed_lora_hashes_all_stages": training.get(
            "changed_lora_hashes_all_stages"
        )
        is True,
        "atomic_ledger_verified": training.get("atomic_ledger_verified") is True,
        "checkpoint_integrity_verified": training.get(
            "checkpoint_integrity_verified"
        )
        is True,
        "adapter_cpu_reload_verified": value.get("adapter_cpu_reload_verified")
        is True,
        "activation_gradient_transfer_verified": value.get(
            "activation_gradient_transfer_verified"
        )
        is True,
        "monitoring_live_verified": value.get("monitoring_live_verified") is True,
        "coordinator_restart_live_verified": value.get(
            "coordinator_restart_live_verified"
        )
        is True,
        "stale_result_rejected": value.get("stale_result_rejected") is True,
        "worker_replacements": dict(value.get("worker_replacements") or {}),
        "kernel_evidence": list(value.get("kernel_evidence") or []),
        "effective_kernel_evidence_verified": value.get(
            "effective_kernel_evidence_verified"
        )
        is True,
        "evidence_replay": dict(value.get("evidence_replay") or {}),
        "performance": dict(value.get("performance") or {}),
        "optimization_summary": dict(value.get("optimization_summary") or {}),
        "benchmark": dict(value.get("benchmark") or {}),
        "cleanup": dict(value.get("cleanup") or {}),
        "blockers": sorted(value.get("blockers") or []),
        "public_artifact_safe": value.get("public_artifact_safe") is True,
    }


def pack(
    *,
    workflow_path: str | Path,
    fault_path: str | Path,
    output_dir: str | Path,
    live_path: str | Path = "",
    regression_path: str | Path = "",
) -> dict[str, Any]:
    workflow = _read(workflow_path)
    fault = _read(fault_path)
    live = _read(live_path) if live_path else {}
    regression = _read(regression_path) if regression_path else {}
    workflow_summary = _workflow_summary(workflow)
    fault_summary = _fault_summary(fault)
    live_summary = _live_summary(live)
    quality = {
        "passed": int(regression.get("passed") or 0),
        "failed": int(regression.get("failed") or 0),
        "skipped": int(regression.get("skipped") or 0),
        "duration_seconds": float(regression.get("duration_seconds") or 0.0),
        "source_content_hash": str(regression.get("content_hash") or ""),
        "source_content_hash_valid": _content_hash_valid(regression)
        if regression
        else False,
    }
    local_ready = bool(
        workflow_summary["workflow_verified"]
        and workflow_summary["monitoring_contract_verified"]
        and workflow_summary["cleanup_verified"]
        and workflow_summary["next_resume_command_uses_public_placeholder"]
        and fault_summary["fault_injection_suite_ready"]
        and fault_summary["generation_fencing_verified"]
        and fault_summary["lease_reclaim_verified"]
        and fault_summary["circuit_breaker_verified"]
        and fault_summary["checkpoint_fallback_verified"]
        and fault_summary["coordinator_journal_recovery_verified"]
        and fault_summary["bounded_retry_verified"]
        and fault_summary["cleanup_verified"]
    )
    steps = list(live_summary.get("committed_steps") or [])
    replacements = dict(live_summary.get("worker_replacements") or {})
    performance = dict(live_summary.get("performance") or {})
    optimization = dict(live_summary.get("optimization_summary") or {})
    cleanup = dict(live_summary.get("cleanup") or {})
    live_ready = bool(
        live_summary.get("source_schema") == LIVE_SCHEMA
        and live_summary.get("source_content_hash_valid") is True
        and live_summary.get("live_run_performed") is True
        and live_summary.get("external_runtime_verified") is True
        and live_summary.get("accepted_providers") == REQUIRED_PROVIDERS
        and len(steps) >= 100
        and steps == list(range(1, len(steps) + 1))
        and float(live_summary.get("soak_duration_seconds") or 0.0) >= 3600.0
        and float(live_summary.get("full_live_gate_elapsed_seconds") or 0.0)
        <= 21600.0
        and all(
            live_summary.get(field) is True
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
            )
        )
        and all(
            replacement_evidence_ready(dict(replacements.get(kind) or {}))
            for kind in ("cpu", "cuda", "jax_tpu")
        )
        and live_summary.get("effective_kernel_evidence_verified") is True
        and sorted(
            str(item.get("kernel_role") or "")
            for item in live_summary.get("kernel_evidence") or []
        )
        == ["cpu", "gpu_a", "gpu_b", "tpu"]
        and all(
            item.get("effective_ok") is True
            for item in live_summary.get("kernel_evidence") or []
        )
        and performance.get("performance_gate_passed") is True
        and int(performance.get("baseline_window_count") or 0) >= 5
        and int(performance.get("candidate_window_count") or 0) >= 5
        and float(performance.get("p95_regression_fraction") or 0.0)
        <= float(performance.get("maximum_p95_regression_fraction") or 0.0)
        and int(optimization.get("performance_window_count_per_phase") or 0)
        >= 5
        and int(optimization.get("inline_tensor_message_upload_count") or 0) > 0
        and int(optimization.get("inline_tensor_message_download_count") or 0) > 0
        and int(
            optimization.get("large_payload_connection_isolation_count") or 0
        )
        > 0
        and 0
        < int(optimization.get("persistent_http_max_body_bytes") or 0)
        <= 4 * 1024 * 1024
        and cleanup.get("all_remote_kernels_deleted") is True
        and cleanup.get("temporary_private_packages_removed") is True
        and cleanup.get("coordinator_stopped") is True
        and cleanup.get("tunnel_stopped") is True
        and cleanup.get("tensor_payloads_removed") is True
        and cleanup.get("live_resources_left_running") is False
        and not live_summary.get("blockers")
    )
    quality_ready = bool(quality["passed"] >= 1 and quality["failed"] == 0)
    ready = bool(local_ready and live_ready and quality_ready)
    blockers = []
    if not local_ready:
        blockers.append("training_production_local_engineering_gate_incomplete")
    if not live:
        blockers.append("training_production_live_soak_not_run")
    elif not live_ready:
        blockers.extend(live_summary.get("blockers") or [])
        blockers.append("training_production_live_soak_gate_incomplete")
    if not quality_ready:
        blockers.append("training_production_regression_gate_incomplete")
    artifacts = {
        "workflow": _artifact(workflow_path, workflow),
        "fault": _artifact(fault_path, fault),
    }
    if live_path:
        artifacts["live"] = _artifact(live_path, live)
    if regression_path:
        artifacts["regression"] = _artifact(regression_path, regression)
    report = {
        "schema": SCHEMA,
        "model_id": "Qwen/Qwen2.5-7B",
        "training_mode": "peft_lora",
        "minimum_required_steps": 100,
        "minimum_required_duration_seconds": 3600,
        "minimum_performance_improvement_fraction": 0.15,
        "maximum_acquisition_window_seconds": 43200,
        "maximum_full_live_gate_seconds": 21600,
        "training_production_rc_ready": ready,
        "local_engineering_ready": local_ready,
        "live_soak_ready": live_ready,
        "quality_ready": quality_ready,
        "workflow_summary": workflow_summary,
        "fault_summary": fault_summary,
        "live_summary": live_summary,
        "quality_summary": quality,
        "artifacts": artifacts,
        "blockers": sorted(set(blockers)),
        "credential_values_public": False,
        "credential_paths_public": False,
        "cookies_public": False,
        "coordinator_url_public": False,
        "session_tokens_public": False,
        "assignment_tokens_public": False,
        "activation_values_public": False,
        "gradient_values_public": False,
        "checkpoint_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    safety = public_safety_errors(report)
    if safety:
        report["public_artifact_safe"] = False
        report["training_production_rc_ready"] = False
        report["blockers"] = sorted(
            set(report["blockers"]) | {"training_production_public_safety_failed"}
        )
    report["public_safety_errors"] = safety
    report["content_hash"] = stable_hash(report)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "training_heterogeneous_production_rc.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checked = check_report(report)
    if not checked["ok"]:
        raise RuntimeError(
            "training_production_pack_checker_failed:" + ",".join(checked["errors"])
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-report", required=True)
    parser.add_argument("--fault-report", required=True)
    parser.add_argument("--live-report", default="")
    parser.add_argument("--regression-report", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = pack(
        workflow_path=args.workflow_report,
        fault_path=args.fault_report,
        live_path=args.live_report,
        regression_path=args.regression_report,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
