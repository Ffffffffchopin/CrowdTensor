#!/usr/bin/env python3
"""Build the canonical public-safe CPU/CUDA/JAX-TPU Training Beta artifact."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.training_heterogeneous_tpu_beta_check import (
    LIVE_SCHEMA,
    SCHEMA,
    build_acceptance_gates,
)


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("heterogeneous_training_tpu_beta_source_invalid")
    return value


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _tpu_compile_latency_import(
    path: str | Path, *, source: dict[str, Any]
) -> dict[str, Any]:
    source_path = Path(path).resolve()
    kernel = _load(source_path)
    if (
        kernel.get("schema")
        != "crowdtensor_heterogeneous_training_tpu_beta_kaggle_kernel_v1"
        or kernel.get("kernel_role") != "tpu"
        or kernel.get("ok") is not True
    ):
        raise ValueError("heterogeneous_training_tpu_compile_kernel_invalid")
    if (
        kernel.get("public_artifact_safe") is not True
        or kernel.get("credential_values_public") is not False
        or kernel.get("private_paths_public") is not False
    ):
        raise ValueError(
            "heterogeneous_training_tpu_compile_kernel_not_public_safe"
        )

    expected_manifest = str(source.get("training_manifest_hash") or "")
    recovery = dict(source.get("tpu_recovery_evidence") or {})
    expected_ids = {
        "tpu_old": str(recovery.get("old_tpu_miner_id_hash") or ""),
        "tpu_replacement": str(
            recovery.get("replacement_tpu_miner_id_hash") or ""
        ),
    }
    worker_rows: list[dict[str, Any]] = []
    for raw in kernel.get("worker_results") or []:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("label") or "")
        if role not in expected_ids:
            continue
        worker = dict(raw.get("report") or {})
        if (
            worker.get("ok") is not True
            or worker.get("device_policy") != "jax_tpu"
            or worker.get("assigned_stage_ids") != [2]
            or worker.get("training_manifest_hash") != expected_manifest
            or worker.get("miner_id_hash") != expected_ids[role]
        ):
            raise ValueError(
                "heterogeneous_training_tpu_compile_worker_binding_invalid"
            )
        statuses = [
            dict(item)
            for item in worker.get("stage_process_statuses") or []
            if isinstance(item, dict)
            and int(item.get("stage_id") or -1) == 2
            and item.get("runtime_backend") == "jax_tpu"
        ]
        if len(statuses) != 1:
            raise ValueError(
                "heterogeneous_training_tpu_compile_status_missing"
            )
        status = statuses[0]
        compile_latency = float(status.get("compile_latency_ms") or 0.0)
        if (
            compile_latency <= 0.0
            or int(status.get("jax_mesh_device_count") or 0) != 8
            or status.get("jax_mesh_shape") != [8]
            or status.get("all_mesh_devices_used") is not True
            or status.get("forward_output_sharding_explicit") is not True
            or status.get("backward_output_sharding_explicit") is not True
            or status.get("boundary_output_replicated") is not True
            or status.get("public_artifact_safe") is not True
            or status.get("tensor_values_public") is not False
        ):
            raise ValueError(
                "heterogeneous_training_tpu_compile_status_invalid"
            )
        worker_rows.append(
            {
                "deployment_role": role,
                "miner_id_hash": expected_ids[role],
                "runtime_backend": "jax_tpu",
                "stage_id": 2,
                "compile_latency_ms": compile_latency,
                "jax_mesh_device_count": 8,
                "jax_mesh_shape": [8],
                "all_mesh_devices_used": True,
                "forward_output_sharding_explicit": True,
                "backward_output_sharding_explicit": True,
                "boundary_output_replicated": True,
                "tensor_values_public": False,
                "public_artifact_safe": True,
            }
        )
    worker_rows.sort(key=lambda item: item["deployment_role"])
    if [item["deployment_role"] for item in worker_rows] != [
        "tpu_old",
        "tpu_replacement",
    ]:
        raise ValueError("heterogeneous_training_tpu_compile_workers_incomplete")
    return {
        "schema": "crowdtensor_heterogeneous_training_tpu_compile_latency_import_v1",
        "source_kernel_report_hash": _file_hash(source_path),
        "training_manifest_hash": expected_manifest,
        "worker_statuses": worker_rows,
        "worker_status_count": len(worker_rows),
        "compile_latency_ms": max(
            float(item["compile_latency_ms"]) for item in worker_rows
        ),
        "measurement_recomputed": False,
        "measurement_recovered_from_retained_worker_status": True,
        "credential_values_public": False,
        "private_paths_public": False,
        "tensor_values_public": False,
        "public_artifact_safe": True,
    }


def _ledger_summary(path: str | Path, *, kind: str) -> dict[str, Any]:
    source = Path(path).resolve()
    ledger = _load(source)
    attempts = []
    for raw in ledger.get("attempts") or []:
        item = {
            "attempt": int(raw.get("attempt") or 0),
            "started_at": str(raw.get("started_at") or ""),
            "finished_at": str(raw.get("finished_at") or ""),
            "completed": raw.get("completed") is True,
            "outcome": str(raw.get("outcome") or ""),
        }
        if int(raw.get("submission_count") or 0) > 0:
            item["submission_count"] = int(raw["submission_count"])
            item["submission_limit"] = int(raw.get("submission_limit") or 3)
            item["window_seconds"] = float(raw.get("window_seconds") or 0.0)
            item["expires_at"] = str(raw.get("expires_at") or "")
            item["submission_outcomes"] = [
                {
                    "submission": int(entry.get("submission") or 0),
                    "finished_at": str(entry.get("finished_at") or ""),
                    "outcome": str(entry.get("outcome") or ""),
                }
                for entry in raw.get("submission_outcomes") or []
            ]
        attempts.append(item)
    attempt_limit = int(ledger.get("attempt_limit") or 0)
    attempt_limit_mode = str(ledger.get("attempt_limit_mode") or "bounded")
    reusable_attempt = 0
    reusable_until = ""
    if attempts:
        latest = attempts[-1]
        expires_at = str(latest.get("expires_at") or "")
        try:
            expires = datetime.fromisoformat(expires_at)
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
        except ValueError:
            expires = None
        if (
            latest.get("completed") is True
            and expires is not None
            and expires > datetime.now(timezone.utc)
            and int(latest.get("submission_count") or 0)
            < int(latest.get("submission_limit") or 0)
        ):
            reusable_attempt = int(latest.get("attempt") or 0)
            reusable_until = expires.isoformat()
    limit_extensions = [
        {
            "schema": str(raw.get("schema") or ""),
            "old_limit": int(raw.get("old_limit") or 0),
            "new_limit": int(raw.get("new_limit") or 0),
            "authorized_at": str(raw.get("authorized_at") or ""),
            "authorization_id_hash": str(raw.get("authorization_id_hash") or ""),
            "authorization_identifier_public": False,
            "credential_values_public": False,
            "public_artifact_safe": True,
        }
        for raw in ledger.get("limit_extensions") or []
        if isinstance(raw, dict)
    ]
    attempt_authorizations = [
        {
            "schema": str(raw.get("schema") or ""),
            "kind": str(raw.get("kind") or ""),
            "mode": str(raw.get("mode") or ""),
            "previous_attempt_limit": int(raw.get("previous_attempt_limit") or 0),
            "authorized_at": str(raw.get("authorized_at") or ""),
            "authorization_id_hash": str(raw.get("authorization_id_hash") or ""),
            "authorization_identifier_public": False,
            "max_attempt_duration_seconds": float(
                raw.get("max_attempt_duration_seconds") or 0.0
            ),
            "attempt_duration_remains_bounded": (
                raw.get("attempt_duration_remains_bounded") is True
            ),
            "credential_values_public": False,
            "public_artifact_safe": True,
        }
        for raw in ledger.get("attempt_authorizations") or []
        if isinstance(raw, dict)
    ]
    unlimited_authorized = bool(
        attempt_limit == 0
        and attempt_limit_mode == "unlimited_authorized"
        and attempt_authorizations
        and all(
            item["mode"] == "unlimited_authorized"
            and item["attempt_duration_remains_bounded"] is True
            and item["authorization_id_hash"].startswith("sha256:")
            for item in attempt_authorizations
        )
    )
    return {
        "schema": "crowdtensor_heterogeneous_training_bounded_ledger_summary_v1",
        "kind": str(kind),
        "attempt_limit": attempt_limit,
        "attempt_limit_mode": attempt_limit_mode,
        "unlimited_attempts_authorized": unlimited_authorized,
        "attempt_count": len(attempts),
        "completed_attempt_count": len(
            [item for item in attempts if item["completed"]]
        ),
        "all_attempts_completed": bool(attempts)
        and all(item["completed"] for item in attempts),
        "attempt_limit_reached": bool(
            attempt_limit_mode == "bounded"
            and attempt_limit
            and len(attempts) >= attempt_limit
        ),
        "reusable_window_present": reusable_attempt > 0,
        "reusable_attempt": reusable_attempt,
        "reusable_until": reusable_until,
        "attempts": attempts,
        "limit_extensions": limit_extensions,
        "attempt_authorizations": attempt_authorizations,
        "authorization_identifiers_public": False,
        "source_report_hash": _file_hash(source),
        "credential_values_public": False,
        "credential_paths_public": False,
        "account_labels_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def pack(
    live_report: str | Path,
    output_dir: str | Path,
    *,
    regression_summary: str | Path | None = None,
    acquisition_ledger: str | Path | None = None,
    live_attempt_ledger: str | Path | None = None,
    gpu_quota_diagnosis: str | Path | None = None,
    runtime_diagnosis: str | Path | None = None,
    stage_diagnostic: str | Path | None = None,
    tpu_kernel_report: str | Path | None = None,
) -> dict[str, Any]:
    source_path = Path(live_report).resolve()
    source = _load(source_path)
    if source.get("schema") != LIVE_SCHEMA:
        raise ValueError("heterogeneous_training_tpu_beta_live_schema_invalid")
    report = copy.deepcopy(source)
    superseded_resource_blockers: set[str] = set()
    report["schema"] = SCHEMA
    report["source_evidence"] = {
        "live_report_hash": _file_hash(source_path),
        "runtime_measurements_changed": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    if tpu_kernel_report is not None:
        compile_import = _tpu_compile_latency_import(
            tpu_kernel_report, source=source
        )
        report["tpu_compile_latency_evidence"] = compile_import
        report["tpu_training_evidence"]["compile_latency_ms"] = float(
            compile_import["compile_latency_ms"]
        )
        report["source_evidence"].update(
            {
                "tpu_kernel_report_hash": compile_import[
                    "source_kernel_report_hash"
                ],
                "omitted_live_compile_measurement_recovered": True,
                "compile_measurement_recomputed": False,
            }
        )
    if regression_summary is not None:
        regression_path = Path(regression_summary).resolve()
        regression = _load(regression_path)
        report["regression_summary"] = {
            **regression,
            "source_report_hash": _file_hash(regression_path),
        }
    if acquisition_ledger is not None or live_attempt_ledger is not None:
        if acquisition_ledger is None or live_attempt_ledger is None:
            raise ValueError("heterogeneous_training_tpu_beta_both_ledgers_required")
        acquisition = _ledger_summary(
            acquisition_ledger, kind="tpu_acquisition_window"
        )
        live_attempts = _ledger_summary(
            live_attempt_ledger, kind="six_step_live_gate"
        )
        report["bounded_resource_summary"] = {
            "schema": "crowdtensor_heterogeneous_training_tpu_resource_summary_v1",
            "tpu_acquisition": acquisition,
            "live_gate": live_attempts,
            "acquisition_windows_exhausted": bool(
                acquisition["attempt_limit_reached"]
                and acquisition["all_attempts_completed"]
                and not acquisition["reusable_window_present"]
            ),
            "live_gate_attempts_used": int(live_attempts["attempt_count"]),
            "public_artifact_safe": True,
        }
        acquisition_exhausted = bool(
            acquisition["attempt_limit_reached"]
            and acquisition["all_attempts_completed"]
            and not acquisition["reusable_window_present"]
        )
        live_gate_exhausted = bool(
            live_attempts["attempt_limit_reached"]
            and live_attempts["all_attempts_completed"]
        )
        acquisition_unlimited = bool(acquisition["unlimited_attempts_authorized"])
        live_gate_unlimited = bool(live_attempts["unlimited_attempts_authorized"])
        if acquisition_unlimited:
            superseded_resource_blockers.update(
                {
                    "heterogeneous_acquisition_window_limit_reached",
                    "heterogeneous_live_attempt_limit_reached",
                }
            )
        if live_gate_unlimited:
            superseded_resource_blockers.update(
                {
                    "heterogeneous_tpu_training_live_gate_limit_reached",
                    "heterogeneous_live_attempt_limit_reached",
                }
            )
        if acquisition_unlimited and live_gate_unlimited:
            next_action = "start_next_bounded_tpu_window_then_full_live_gate"
        elif acquisition_exhausted and live_gate_exhausted:
            next_action = (
                "explicit_new_bounded_tpu_acquisition_window_and_live_gate_authorization_required"
            )
        elif acquisition_exhausted:
            next_action = "explicit_new_bounded_tpu_acquisition_window_required"
        elif live_gate_exhausted:
            next_action = (
                "reuse_current_tpu_acquisition_window_with_explicit_live_gate_authorization"
            )
        else:
            next_action = "reuse_current_tpu_acquisition_window"
        report["resume_contract"] = {
            "schema": "crowdtensor_heterogeneous_training_tpu_resume_v1",
            "resumable": True,
            "current_goal_acquisition_boundary_exhausted": acquisition_exhausted,
            "current_goal_live_gate_boundary_exhausted": live_gate_exhausted,
            "next_action": next_action,
            "resume_requires_private_kaggle_credentials": True,
            "resume_command_public": False,
            "credential_values_public": False,
            "credential_paths_public": False,
            "account_labels_public": False,
            "public_artifact_safe": True,
        }
    if gpu_quota_diagnosis is not None:
        diagnosis_path = Path(gpu_quota_diagnosis).resolve()
        diagnosis = _load(diagnosis_path)
        if diagnosis.get("schema") != (
            "crowdtensor_heterogeneous_training_tpu_gpu_quota_diagnosis_v1"
        ):
            raise ValueError(
                "heterogeneous_training_tpu_gpu_quota_diagnosis_schema_invalid"
            )
        if diagnosis.get("public_artifact_safe") is not True:
            raise ValueError(
                "heterogeneous_training_tpu_gpu_quota_diagnosis_not_public_safe"
            )
        report["gpu_quota_diagnosis"] = {
            key: copy.deepcopy(diagnosis.get(key))
            for key in (
                "schema",
                "diagnosed_at",
                "source_live_report_hash",
                "failure_phase",
                "failed_role",
                "tpu_submission_accepted",
                "tpu_running_observed",
                "failed_gpu_account_quota",
                "authorized_alternative_gpu_account_count",
                "authorized_alternative_gpu_accounts_with_positive_quota",
                "authorized_alternative_effective_remaining_min_seconds",
                "authorized_alternative_effective_remaining_max_seconds",
                "acquisition_window",
                "live_gate",
                "cleanup_verified",
                "cleanup_addendum",
                "live_resources_left_running",
                "account_labels_public",
                "credential_values_public",
                "credential_paths_public",
                "coordinator_url_public",
                "raw_quota_api_response_public",
                "public_artifact_safe",
            )
        }
        report["gpu_quota_diagnosis"]["diagnosis_report_hash"] = _file_hash(
            diagnosis_path
        )
        report["blockers"] = list(report.get("blockers") or []) + list(
            diagnosis.get("blockers") or []
        )
    if stage_diagnostic is not None:
        diagnostic_path = Path(stage_diagnostic).resolve()
        diagnostic = _load(diagnostic_path)
        if diagnostic.get("schema") != (
            "crowdtensor_heterogeneous_training_tpu_stage_diagnostic_live_v1"
        ):
            raise ValueError(
                "heterogeneous_training_tpu_stage_diagnostic_schema_invalid"
            )
        if diagnostic.get("public_artifact_safe") is not True:
            raise ValueError(
                "heterogeneous_training_tpu_stage_diagnostic_not_public_safe"
            )
        stored_diagnostic_hash = str(diagnostic.get("content_hash") or "")
        if stored_diagnostic_hash != _stable_hash(
            {
                key: value
                for key, value in diagnostic.items()
                if key != "content_hash"
            }
        ):
            raise ValueError(
                "heterogeneous_training_tpu_stage_diagnostic_content_hash_invalid"
            )
        kernel = dict(diagnostic.get("kernel_report") or {})
        report["stage_diagnostic_summary"] = {
            "schema": "crowdtensor_heterogeneous_training_tpu_stage_diagnostic_summary_v1",
            "diagnostic_report_hash": _file_hash(diagnostic_path),
            "live_probe_performed": diagnostic.get("live_probe_performed") is True,
            "diagnostic_ok": diagnostic.get("ok") is True,
            "diagnostic_only": diagnostic.get("diagnostic_only") is True,
            "full_training_gate_evidence": diagnostic.get("full_training_gate_evidence")
            is True,
            "same_job_three_accelerator_evidence": diagnostic.get(
                "same_job_three_accelerator_evidence"
            )
            is True,
            "live_gate_ledger_modified": diagnostic.get("live_gate_ledger_modified")
            is True,
            "requested_accelerator": str(
                diagnostic.get("requested_accelerator") or ""
            ),
            "stage_id": int(diagnostic.get("stage_id") or -1),
            "terminal_state": str(diagnostic.get("terminal_state") or ""),
            "queue_observation_count": len(
                diagnostic.get("queue_observations") or []
            ),
            "kernel_phase": str(kernel.get("phase") or ""),
            "kernel_failure_phase": str(kernel.get("failure_phase") or ""),
            "kernel_progress": copy.deepcopy(kernel.get("progress") or {}),
            "source_evidence": copy.deepcopy(kernel.get("source_evidence") or {}),
            "shard_evidence": copy.deepcopy(kernel.get("shard_evidence") or {}),
            "jax_load_evidence": copy.deepcopy(kernel.get("jax_load_evidence") or {}),
            "training_step_evidence": copy.deepcopy(
                kernel.get("training_step_evidence") or {}
            ),
            "kernel_blockers": list(kernel.get("blockers") or []),
            "output_collection": copy.deepcopy(
                diagnostic.get("kernel_output_collection") or {}
            ),
            "cleanup": copy.deepcopy(diagnostic.get("cleanup") or {}),
            "credential_values_public": False,
            "credential_paths_public": False,
            "account_labels_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        if diagnostic.get("ok") is not True:
            report["blockers"] = list(report.get("blockers") or []) + list(
                diagnostic.get("blockers") or []
            )
    if runtime_diagnosis is not None:
        diagnosis_path = Path(runtime_diagnosis).resolve()
        diagnosis = _load(diagnosis_path)
        if diagnosis.get("schema") != (
            "crowdtensor_heterogeneous_training_tpu_runtime_diagnosis_v1"
        ):
            raise ValueError(
                "heterogeneous_training_tpu_runtime_diagnosis_schema_invalid"
            )
        if diagnosis.get("public_artifact_safe") is not True:
            raise ValueError(
                "heterogeneous_training_tpu_runtime_diagnosis_not_public_safe"
            )
        if diagnosis.get("source_live_report_hash") != _file_hash(source_path):
            raise ValueError(
                "heterogeneous_training_tpu_runtime_diagnosis_source_mismatch"
            )
        report["runtime_diagnosis"] = copy.deepcopy(diagnosis)
        report["runtime_diagnosis"]["diagnosis_report_hash"] = _file_hash(
            diagnosis_path
        )
        report["blockers"] = list(report.get("blockers") or []) + list(
            diagnosis.get("blockers") or []
        )
    current_blockers = {
        str(item) for item in report.get("blockers") or [] if str(item)
    }
    superseded = sorted(current_blockers & superseded_resource_blockers)
    if superseded:
        report["superseded_resource_blockers"] = [
            {
                "blocker": blocker,
                "reason": "unlimited_attempt_count_authorized_with_bounded_duration",
                "public_artifact_safe": True,
            }
            for blocker in superseded
        ]
    report["blockers"] = sorted(current_blockers - superseded_resource_blockers)
    report.pop("content_hash", None)
    report["acceptance_gates"] = build_acceptance_gates(report)
    report["heterogeneous_training_tpu_beta_ready"] = bool(
        report.get("live_run_performed") is True
        and all(report["acceptance_gates"].values())
        and not report["blockers"]
    )
    report["content_hash"] = _stable_hash(report)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "training_heterogeneous_tpu_beta.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-report", required=True)
    parser.add_argument("--regression-summary", default="")
    parser.add_argument("--acquisition-ledger", default="")
    parser.add_argument("--live-attempt-ledger", default="")
    parser.add_argument("--gpu-quota-diagnosis", default="")
    parser.add_argument("--runtime-diagnosis", default="")
    parser.add_argument("--stage-diagnostic", default="")
    parser.add_argument("--tpu-kernel-report", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = pack(
        args.live_report,
        args.output_dir,
        regression_summary=args.regression_summary or None,
        acquisition_ledger=args.acquisition_ledger or None,
        live_attempt_ledger=args.live_attempt_ledger or None,
        gpu_quota_diagnosis=args.gpu_quota_diagnosis or None,
        runtime_diagnosis=args.runtime_diagnosis or None,
        stage_diagnostic=args.stage_diagnostic or None,
        tpu_kernel_report=args.tpu_kernel_report or None,
    )
    print(json.dumps(report, sort_keys=True) if args.json else report)


if __name__ == "__main__":
    main()
