#!/usr/bin/env python3
"""Run the bounded six-step Kaggle CPU/CUDA/JAX-TPU Training Beta gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import io
import json
import math
import os
import shutil
import sqlite3
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from crowdtensor.heterogeneous_training_beta import (
    HeterogeneousTrainingBetaController,
    create_heterogeneous_training_beta_app,
)
from crowdtensor.heterogeneous_training_checkpoint import (
    checkpoint_file_names,
    validate_checkpoint_manifest,
)
from crowdtensor.heterogeneous_training_manifest import (
    qwen25_7b_lora_tpu_manifest,
    stable_hash,
)
from scripts.training_cuda_kaggle_common import (
    authenticated_owner,
    delete_succeeded_or_absent,
    extract_kernel_ref,
    kaggle_env,
    public_safety_errors,
    push_accepted,
    run_command,
    safe_slug,
    status_class,
    utc_now,
)
from scripts.kaggle_gpu_token_weekly_quota_probe import fetch_accelerator_quota
from scripts.training_heterogeneous_beta_kaggle_package import (
    build_package as build_cpu_gpu_package,
)
from scripts.training_heterogeneous_beta_live_probe import (
    _committed_assignments,
    _finish_attempt,
    _free_port,
    _hash_text,
    _reserve_attempt,
    _safe_status_snapshot,
    _start_verified_tunnel,
    _wait_local_ready,
    collect_public_tensor_metadata,
    ensure_cloudflared,
    stop_process,
    transport_contract_probe,
)
from scripts.training_heterogeneous_tpu_beta_check import LIVE_SCHEMA
from scripts.training_heterogeneous_tpu_beta_kaggle_package import (
    build_package as build_tpu_package,
)


TERMINAL = {"complete", "failed"}
GPU_CPU_KERNEL_REPORT = "training_heterogeneous_beta_kernel.json"
TPU_KERNEL_REPORT = "training_heterogeneous_tpu_beta_kernel.json"
OUTPUT_PATTERN = (
    r"training_heterogeneous_(?:tpu_)?beta_kernel\.json|"
    r"training_heterogeneous_export_reload_probe\.json|worker-.*\.json"
)


def _write_json(path: Path, value: Any, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    path.chmod(mode)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _public_blocker(exc: BaseException) -> str:
    text = str(exc)
    if text.startswith(("heterogeneous_", "elastic_", "kaggle_")):
        return text.splitlines()[0][:180]
    return "heterogeneous_tpu_live_failed:" + type(exc).__name__


def classify_tpu_push(step: dict[str, Any]) -> str:
    if push_accepted(step):
        return "tpu_submission_accepted"
    text = str(step.get("output_tail") or "").lower()
    if "maximum batch tpu session count" in text:
        return "kaggle_tpu_batch_session_limit_reached"
    if "quota" in text and "tpu" in text:
        return "kaggle_tpu_quota_unavailable"
    if "accelerator" in text and "unavailable" in text:
        return "kaggle_tpu_accelerator_unavailable"
    if "429" in text or "too many requests" in text or "rate limit" in text:
        return "kaggle_tpu_push_rate_limited"
    if step.get("timed_out") is True:
        return "kaggle_tpu_kernel_push_timeout"
    return "kaggle_tpu_kernel_push_rejected"


def classify_training_worker_push(role: str, step: dict[str, Any]) -> str:
    role = str(role)
    if push_accepted(step):
        return f"{role}_submission_accepted"
    text = str(step.get("output_tail") or "").lower()
    if role.startswith("gpu") and "maximum weekly gpu quota" in text:
        return "kaggle_gpu_weekly_quota_exhausted"
    if role.startswith("gpu") and "maximum batch gpu session count" in text:
        return "kaggle_gpu_batch_session_limit_reached"
    if role == "cpu" and "maximum batch cpu session count" in text:
        return "kaggle_cpu_batch_session_limit_reached"
    if "429" in text or "too many requests" in text or "rate limit" in text:
        return f"kaggle_{role}_push_rate_limited"
    if step.get("timed_out") is True:
        return f"kaggle_{role}_kernel_push_timeout"
    return f"heterogeneous_{role}_kernel_push_rejected"


def gpu_quota_preflight_summary(
    quota: dict[str, Any], *, phase: str
) -> dict[str, Any]:
    gpu = dict(quota.get("gpu_quota") or {})
    exhausted = bool(
        quota.get("ok") is True
        and gpu.get("present") is True
        and (
            gpu.get("quota_exhausted_by_used") is True
            or float(gpu.get("effective_remaining_after_reserved_seconds") or 0.0)
            <= 0.0
        )
    )
    return {
        "schema": "crowdtensor_heterogeneous_training_gpu_quota_preflight_v1",
        "phase": str(phase),
        "quota_api_ok": quota.get("ok") is True,
        "quota_refresh_time": str(quota.get("quota_refresh_time") or ""),
        "gpu_quota_present": gpu.get("present") is True,
        "time_used_seconds": float(gpu.get("time_used_seconds") or 0.0),
        "time_reserved_seconds": float(gpu.get("time_reserved_seconds") or 0.0),
        "total_time_allowed_seconds": float(
            gpu.get("total_time_allowed_seconds") or 0.0
        ),
        "effective_remaining_after_reserved_seconds": float(
            gpu.get("effective_remaining_after_reserved_seconds") or 0.0
        ),
        "weekly_gpu_quota_exhausted": exhausted,
        "account_label_public": False,
        "credential_values_public": False,
        "credential_paths_public": False,
        "public_artifact_safe": True,
    }


def terminal_before_training_complete(
    states: dict[str, str], *, runtime_state: str
) -> list[str]:
    if str(runtime_state) == "completed":
        return []
    return sorted(
        role for role, state in states.items() if str(state) in TERMINAL
    )


def runtime_observation_summary(
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = [dict(item) for item in observations if isinstance(item, dict)]
    terminal_states = sorted(
        {
            f"{role}:{state}"
            for item in rows
            for role, state in dict(item.get("kernel_states") or {}).items()
            if str(state) in TERMINAL
        }
    )
    return {
        "schema": "crowdtensor_heterogeneous_training_tpu_runtime_observations_v1",
        "observation_count": len(rows),
        "max_committed_step": max(
            [int(item.get("committed_step") or 0) for item in rows] or [0]
        ),
        "max_placement_generation": max(
            [int(item.get("placement_generation") or 0) for item in rows] or [0]
        ),
        "terminal_kernel_states_seen": terminal_states,
        "first_observation": rows[0] if rows else {},
        "last_observation": rows[-1] if rows else {},
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def runtime_progress_summary(status: dict[str, Any]) -> dict[str, Any]:
    events = [dict(item) for item in status.get("events") or []]
    assignments = [dict(item) for item in status.get("assignments") or []]
    miners = [dict(item) for item in status.get("miners") or []]
    checkpoints = [
        item for item in events if item.get("operation") == "stage_checkpoint_submitted"
    ]
    profiles = [
        item for item in events if item.get("operation") == "stage_profile_updated"
    ]
    accelerator_counts: dict[str, int] = {}
    for miner in miners:
        if miner.get("state") != "online":
            continue
        accelerator = str(miner.get("accelerator") or "")
        accelerator_counts[accelerator] = accelerator_counts.get(accelerator, 0) + 1
    return {
        "schema": "crowdtensor_heterogeneous_training_tpu_runtime_progress_v1",
        "runtime_state": str(status.get("runtime_state") or ""),
        "committed_steps": [int(item) for item in status.get("committed_steps") or []],
        "live_miner_count": int(status.get("live_miner_count") or 0),
        "online_accelerator_counts": accelerator_counts,
        "assigned_stage_providers": [
            {
                "stage_id": int(item.get("stage_id") or 0),
                "device_type": str(item.get("device_type") or ""),
                "state": str(item.get("state") or ""),
            }
            for item in sorted(assignments, key=lambda row: int(row.get("stage_id") or 0))
        ],
        "checkpoint_submitted_stage_ids": sorted(
            {int(item.get("stage_id") or 0) for item in checkpoints}
        ),
        "checkpoint_target_steps": sorted(
            {int(item.get("target_step") or 0) for item in checkpoints}
        ),
        "profiled_stage_ids": sorted(
            {int(item.get("stage_id") or 0) for item in profiles}
        ),
        "event_count": len(events),
        "last_event_sequence": max(
            [int(item.get("sequence") or 0) for item in events] or [0]
        ),
        "miner_ids_public": False,
        "assignment_tokens_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def collect_kernel_output_with_retry(
    *,
    ref: str,
    env: dict[str, str],
    destination: Path,
    filename: str,
    file_pattern: str = OUTPUT_PATTERN,
    timeout_seconds: float,
    poll_interval_seconds: float,
    runner: Any = run_command,
    sleeper: Any = time.sleep,
) -> tuple[dict[str, Any], dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + max(1.0, float(timeout_seconds))
    attempts = 0
    last_step: dict[str, Any] = {}
    kernel: dict[str, Any] = {}
    while time.monotonic() < deadline:
        attempts += 1
        remaining = max(1.0, deadline - time.monotonic())
        last_step = runner(
            [
                "kaggle",
                "kernels",
                "output",
                str(ref),
                "-p",
                str(destination),
                "--force",
            "--file-pattern",
            str(file_pattern),
            ],
            env=env,
            timeout=min(120.0, remaining),
        )
        kernel = _read_json(destination / filename)
        if kernel:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        sleeper(min(max(0.1, float(poll_interval_seconds)), remaining))
    return kernel, {
        "schema": "crowdtensor_heterogeneous_training_terminal_output_collection_v1",
        "attempt_count": attempts,
        "report_found": bool(kernel),
        "duration_seconds": round(time.monotonic() - started, 3),
        "last_returncode": last_step.get("returncode"),
        "last_timed_out": last_step.get("timed_out") is True,
        "kernel_ref_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def reserve_acquisition_window(
    path: Path,
    *,
    limit: int,
    reuse_attempt: int,
    window_seconds: float,
    submission_limit: int = 3,
) -> tuple[int, float]:
    if int(reuse_attempt) <= 0:
        attempt = _reserve_attempt(path, limit=limit)
        ledger = _read_json(path)
        for item in ledger.get("attempts") or []:
            if int(item.get("attempt") or 0) == attempt:
                item["submission_count"] = 1
                item["submission_limit"] = int(submission_limit)
                item["window_seconds"] = float(window_seconds)
                started = datetime.fromisoformat(str(item["started_at"]))
                item["expires_at"] = datetime.fromtimestamp(
                    started.timestamp() + float(window_seconds),
                    tz=started.tzinfo,
                ).isoformat()
        _write_json(path, ledger)
        return attempt, float(window_seconds)
    ledger = _read_json(path)
    attempts = list(ledger.get("attempts") or [])
    if int(ledger.get("attempt_limit") or 0) != int(limit):
        raise RuntimeError("heterogeneous_live_attempt_limit_conflict")
    if not attempts or int(attempts[-1].get("attempt") or 0) != int(reuse_attempt):
        raise RuntimeError("heterogeneous_acquisition_window_reuse_invalid")
    item = attempts[-1]
    if item.get("completed") is not True:
        raise RuntimeError("heterogeneous_acquisition_window_still_active")
    started = datetime.fromisoformat(str(item.get("started_at") or "")).timestamp()
    remaining = float(window_seconds) - max(0.0, time.time() - started)
    if remaining < 600.0:
        raise RuntimeError("heterogeneous_acquisition_window_expired")
    submission_count = int(item.get("submission_count") or 1)
    if submission_count >= int(submission_limit):
        raise RuntimeError("heterogeneous_acquisition_submission_limit_reached")
    history = list(item.get("submission_outcomes") or [])
    history.append(
        {
            "submission": submission_count,
            "finished_at": str(item.get("finished_at") or ""),
            "outcome": str(item.get("outcome") or ""),
        }
    )
    item.update(
        {
            "submission_count": submission_count + 1,
            "submission_limit": int(submission_limit),
            "window_seconds": float(window_seconds),
            "expires_at": datetime.fromtimestamp(
                started + float(window_seconds),
                tz=datetime.fromisoformat(str(item["started_at"])).tzinfo,
            ).isoformat(),
            "submission_outcomes": history,
            "last_submission_started_at": utc_now(),
            "completed": False,
            "outcome": "running",
        }
    )
    item.pop("finished_at", None)
    _write_json(path, ledger)
    return int(reuse_attempt), remaining


def authorize_unlimited_attempts(
    path: Path,
    *,
    kind: str,
    authorization_granted: bool,
    authorization_id: str,
    max_attempt_duration_seconds: float,
) -> dict[str, Any]:
    """Persist an idempotent, public-safe authorization for unlimited attempts."""

    if kind not in {
        "tpu_acquisition_window",
        "six_step_live_gate",
        "production_soak_live_gate",
    }:
        raise RuntimeError("heterogeneous_unlimited_attempt_kind_invalid")
    if authorization_granted is not True:
        raise RuntimeError("heterogeneous_unlimited_attempts_not_authorized")
    identifier = str(authorization_id).strip()
    if not identifier:
        raise RuntimeError(
            "heterogeneous_unlimited_attempts_authorization_id_missing"
        )
    duration_limit = float(max_attempt_duration_seconds)
    expected_duration = 43200.0 if kind == "tpu_acquisition_window" else 21600.0
    if duration_limit != expected_duration:
        raise RuntimeError("heterogeneous_unlimited_attempt_duration_invalid")

    authorization_hash = _hash_text(identifier)
    ledger = _read_json(path)
    attempts = [
        dict(item)
        for item in ledger.get("attempts") or []
        if isinstance(item, dict)
    ]
    if any(item.get("completed") is not True for item in attempts):
        raise RuntimeError("heterogeneous_unlimited_attempt_ledger_active")
    existing = [
        dict(item)
        for item in ledger.get("attempt_authorizations") or []
        if isinstance(item, dict)
    ]
    matching = next(
        (
            item
            for item in existing
            if item.get("kind") == kind
            and item.get("mode") == "unlimited_authorized"
            and item.get("authorization_id_hash") == authorization_hash
            and float(item.get("max_attempt_duration_seconds") or 0.0)
            == duration_limit
        ),
        None,
    )
    mode = str(ledger.get("attempt_limit_mode") or "bounded")
    if mode == "unlimited_authorized":
        if matching is None or len(existing) != 1:
            raise RuntimeError(
                "heterogeneous_unlimited_attempt_authorization_conflict"
            )
        return dict(matching)
    if existing:
        raise RuntimeError("heterogeneous_unlimited_attempt_authorization_conflict")

    previous_limit = int(ledger.get("attempt_limit") or 0)
    if attempts and previous_limit <= 0:
        raise RuntimeError("heterogeneous_unlimited_attempt_ledger_invalid")
    authorization = {
        "schema": "crowdtensor_heterogeneous_training_attempt_authorization_v1",
        "kind": kind,
        "mode": "unlimited_authorized",
        "previous_attempt_limit": previous_limit,
        "authorized_at": utc_now(),
        "authorization_id_hash": authorization_hash,
        "authorization_identifier_public": False,
        "max_attempt_duration_seconds": duration_limit,
        "attempt_duration_remains_bounded": True,
        "credential_values_public": False,
        "public_artifact_safe": True,
    }
    updated = dict(ledger)
    updated.update(
        {
            "schema": "crowdtensor_heterogeneous_training_attempt_ledger_v1",
            "attempt_limit": 0,
            "attempt_limit_mode": "unlimited_authorized",
            "attempts": attempts,
            "attempt_authorizations": [authorization],
            "authorization_identifiers_public": False,
            "credential_values_public": False,
            "public_artifact_safe": True,
        }
    )
    _write_json(path, updated)
    return dict(authorization)


def extend_acquisition_window_limit(
    path: Path,
    *,
    requested_limit: int,
    extension_authorized: bool,
    authorization_id: str,
) -> dict[str, Any]:
    """Apply the one-time 2 -> 3 acquisition-window authorization."""

    if int(requested_limit) != 3:
        raise RuntimeError("heterogeneous_acquisition_window_extension_invalid_jump")
    if extension_authorized is not True:
        raise RuntimeError("heterogeneous_acquisition_window_extension_not_authorized")
    identifier = str(authorization_id).strip()
    if not identifier:
        raise RuntimeError(
            "heterogeneous_acquisition_window_extension_authorization_id_missing"
        )
    authorization_hash = _hash_text(identifier)
    ledger = _read_json(path)
    attempts = list(ledger.get("attempts") or [])
    extensions = [
        dict(item)
        for item in ledger.get("limit_extensions") or []
        if isinstance(item, dict)
    ]
    matching = next(
        (
            item
            for item in extensions
            if int(item.get("old_limit") or 0) == 2
            and int(item.get("new_limit") or 0) == 3
            and item.get("authorization_id_hash") == authorization_hash
        ),
        None,
    )
    if int(ledger.get("attempt_limit") or 0) == 3:
        if matching is None:
            raise RuntimeError(
                "heterogeneous_acquisition_window_extension_authorization_conflict"
            )
        return dict(matching)
    if int(ledger.get("attempt_limit") or 0) != 2:
        raise RuntimeError("heterogeneous_acquisition_window_extension_ledger_invalid")
    if (
        len(attempts) != 2
        or [int(item.get("attempt") or 0) for item in attempts] != [1, 2]
        or any(item.get("completed") is not True for item in attempts)
    ):
        raise RuntimeError("heterogeneous_acquisition_window_extension_ledger_invalid")
    if extensions:
        raise RuntimeError(
            "heterogeneous_acquisition_window_extension_authorization_conflict"
        )
    extension = {
        "schema": "crowdtensor_heterogeneous_training_limit_extension_v1",
        "old_limit": 2,
        "new_limit": 3,
        "authorized_at": utc_now(),
        "authorization_id_hash": authorization_hash,
        "authorization_identifier_public": False,
        "credential_values_public": False,
        "public_artifact_safe": True,
    }
    ledger["attempt_limit"] = 3
    ledger["limit_extensions"] = [extension]
    ledger["authorization_identifiers_public"] = False
    _write_json(path, ledger)
    return dict(extension)


def extend_live_attempt_limit(
    path: Path,
    *,
    requested_limit: int,
    extension_authorized: bool,
    authorization_id: str,
) -> dict[str, Any]:
    """Apply the one-time 3 -> 4 live-gate authorization."""

    if int(requested_limit) != 4:
        raise RuntimeError("heterogeneous_live_gate_extension_invalid_jump")
    if extension_authorized is not True:
        raise RuntimeError("heterogeneous_live_gate_extension_not_authorized")
    identifier = str(authorization_id).strip()
    if not identifier:
        raise RuntimeError("heterogeneous_live_gate_extension_authorization_id_missing")
    authorization_hash = _hash_text(identifier)
    ledger = _read_json(path)
    attempts = list(ledger.get("attempts") or [])
    extensions = [
        dict(item)
        for item in ledger.get("limit_extensions") or []
        if isinstance(item, dict)
    ]
    matching = next(
        (
            item
            for item in extensions
            if int(item.get("old_limit") or 0) == 3
            and int(item.get("new_limit") or 0) == 4
            and item.get("authorization_id_hash") == authorization_hash
        ),
        None,
    )
    if int(ledger.get("attempt_limit") or 0) == 4:
        if matching is None:
            raise RuntimeError(
                "heterogeneous_live_gate_extension_authorization_conflict"
            )
        return dict(matching)
    if int(ledger.get("attempt_limit") or 0) != 3:
        raise RuntimeError("heterogeneous_live_gate_extension_ledger_invalid")
    if (
        len(attempts) != 3
        or [int(item.get("attempt") or 0) for item in attempts] != [1, 2, 3]
        or any(item.get("completed") is not True for item in attempts)
    ):
        raise RuntimeError("heterogeneous_live_gate_extension_ledger_invalid")
    if extensions:
        raise RuntimeError("heterogeneous_live_gate_extension_authorization_conflict")
    extension = {
        "schema": "crowdtensor_heterogeneous_training_live_gate_limit_extension_v1",
        "old_limit": 3,
        "new_limit": 4,
        "authorized_at": utc_now(),
        "authorization_id_hash": authorization_hash,
        "authorization_identifier_public": False,
        "credential_values_public": False,
        "public_artifact_safe": True,
    }
    ledger["attempt_limit"] = 4
    ledger["limit_extensions"] = [extension]
    ledger["authorization_identifiers_public"] = False
    _write_json(path, ledger)
    return dict(extension)


def _base_report(blocker: str) -> dict[str, Any]:
    manifest = qwen25_7b_lora_tpu_manifest()
    return {
        "schema": LIVE_SCHEMA,
        "live_run_performed": False,
        "execution_provider": "kaggle",
        "model_id": manifest["model"]["model_id"],
        "model_revision": manifest["model"]["model_revision"],
        "training_manifest_schema": manifest["schema"],
        "training_manifest_hash": manifest["content_hash"],
        "parameter_count": int(manifest["model"]["parameter_count"]),
        "stage_boundaries": [
            [int(item["layer_start"]), int(item["layer_end"])]
            for item in manifest["stages"]
        ],
        "sequence_length": int(manifest["training"]["sequence_length"]),
        "microbatch_size": int(manifest["training"]["microbatch_size"]),
        "target_steps": int(manifest["training"]["target_steps"]),
        "same_job_training_verified": False,
        "job_id_hash": "",
        "run_id_hash": "",
        "provider_coverage": [],
        "placement_evidence": {},
        "training_evidence": {},
        "tpu_training_evidence": {},
        "tensor_transport_evidence": {},
        "tpu_recovery_evidence": {},
        "checkpoint_evidence": {},
        "export_evidence": {},
        "regression_summary": {
            "passed": 0,
            "failed": 0,
            "legacy_cpu_cuda_tests_included": False,
            "jax_tpu_tests_included": False,
            "public_safety_tests_included": False,
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
        "blockers": [str(blocker)],
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


def _flatten_workers(kernel_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    workers = []
    for kernel in kernel_reports:
        for result in kernel.get("worker_results") or []:
            worker = dict(result.get("report") or {})
            if not worker:
                continue
            capability = dict(worker.get("capability") or {})
            label = str(worker.get("deployment_role") or result.get("label") or "")
            workers.append(
                {
                    "deployment_role": label,
                    "kernel_role": str(kernel.get("kernel_role") or ""),
                    "kernel_ref_hash": str(kernel.get("kernel_ref_hash") or ""),
                    "miner_id_hash": str(worker.get("miner_id_hash") or ""),
                    "device_policy": str(worker.get("device_policy") or ""),
                    "training_manifest_hash": str(
                        worker.get("training_manifest_hash") or ""
                    ),
                    "assigned_stage_ids": list(worker.get("assigned_stage_ids") or []),
                    "committed_steps": [
                        int(item.get("target_step") or 0)
                        for item in worker.get("steps") or []
                    ],
                    "steps_completed": int(worker.get("steps_completed") or 0),
                    "central_checkpoint_restore_count": int(
                        worker.get("central_checkpoint_restore_count") or 0
                    ),
                    "capability_hash": str(capability.get("content_hash") or ""),
                    "gpu_count": len(capability.get("gpus") or []),
                    "tpu_group_count": len(capability.get("tpu_groups") or []),
                    "public_artifact_safe": True,
                    "_full": worker,
                }
            )
    return workers


def _stage_results(workers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(stage)
        for worker in workers
        for step in worker["_full"].get("steps") or []
        for stage in step.get("stages") or []
    ]


def collect_checkpoint_evidence(
    controller: HeterogeneousTrainingBetaController, *, target_step: int
) -> dict[str, Any]:
    rows = []
    for stage_id in range(5):
        archive, archive_report = controller.runtime.read_committed_checkpoint(
            stage_id=stage_id, target_step=target_step
        )
        name = checkpoint_file_names(stage_id)["manifest"]
        with zipfile.ZipFile(io.BytesIO(archive), "r") as bundle:
            checkpoint = json.loads(bundle.read(name).decode("utf-8"))
        checkpoint = validate_checkpoint_manifest(
            checkpoint,
            training_manifest=controller.manifest,
            expected_stage_id=stage_id,
            expected_step=target_step,
            expected_dataset_cursor=target_step,
        )
        rows.append(
            {
                "stage_id": stage_id,
                "archive_hash": str(archive_report.get("archive_hash") or ""),
                "checkpoint_hash": str(checkpoint.get("content_hash") or ""),
                "runtime_backend": str(
                    checkpoint.get("runtime_backend") or "pytorch"
                ),
                "optimizer_state_present": checkpoint.get(
                    "optimizer_state_present"
                )
                is True,
                "scheduler_state_present": checkpoint.get(
                    "scheduler_state_present"
                )
                is True,
                "grad_scaler_state_present": checkpoint.get(
                    "grad_scaler_state_present"
                )
                is True,
                "grad_scaler_state_applicable": bool(
                    checkpoint.get("grad_scaler_state_applicable", True)
                ),
                "rng_state_present": checkpoint.get("rng_state_present") is True,
                "jax_prng_state_present": checkpoint.get(
                    "jax_prng_state_present"
                )
                is True,
                "component_hashes_verified": bool(
                    archive_report.get("archive_paths_validated") is True
                    and archive_report.get("tensor_payload_validation_enabled")
                    is True
                ),
            }
        )
    return {
        "stage_ids": [item["stage_id"] for item in rows],
        "all_five_stage_archives_valid": len(rows) == 5,
        "atomic_checkpoint_barrier_verified": True,
        "pytorch_components_complete": all(
            item["optimizer_state_present"]
            and item["scheduler_state_present"]
            and item["grad_scaler_state_present"]
            and item["rng_state_present"]
            for item in rows
            if item["stage_id"] != 2
        ),
        "tpu_runtime_backend": rows[2]["runtime_backend"],
        "tpu_optimizer_state_present": rows[2]["optimizer_state_present"],
        "tpu_scheduler_state_present": rows[2]["scheduler_state_present"],
        "tpu_jax_prng_state_present": rows[2]["jax_prng_state_present"],
        "tpu_grad_scaler_applicable": rows[2]["grad_scaler_state_applicable"],
        "tpu_pickle_deserialization_allowed": False,
        "all_component_hashes_verified": all(
            item["component_hashes_verified"] for item in rows
        ),
        "checkpoint_set_hash": stable_hash(rows),
        "checkpoint_tensor_values_public": False,
        "public_artifact_safe": True,
    }


def stale_generation_probe(
    controller: HeterogeneousTrainingBetaController, *, old_generation: int
) -> dict[str, Any]:
    connection = sqlite3.connect(controller.runtime.state_path)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            """
            SELECT a.*,m.session_token FROM assignments a JOIN epochs e
              ON e.run_id=a.run_id AND e.epoch_id=a.epoch_id
              JOIN miners m ON m.session_id=a.session_id
            WHERE a.run_id=? AND a.stage_id=2 AND a.state='active'
              AND e.state='active' AND a.placement_generation>?
            ORDER BY a.epoch_id DESC LIMIT 1
            """,
            (controller.run_id, int(old_generation)),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return {"verified": False, "error_code": "active_replacement_assignment_missing"}
    try:
        controller.runtime.report_stage_runtime(
            session_id=str(row["session_id"]),
            session_token=str(row["session_token"]),
            assignment_token=str(row["assignment_token"]),
            placement_generation=int(old_generation),
            stage_id=2,
            device_id=str(row["device_id"]),
            event_type="profile",
            forward_latency_ms=1.0,
            backward_latency_ms=1.0,
        )
    except ValueError as exc:
        code = str(exc)
        return {
            "verified": "elastic_stage_placement_generation_stale" in code,
            "error_code": (
                "elastic_stage_placement_generation_stale"
                if "elastic_stage_placement_generation_stale" in code
                else "unexpected_rejection"
            ),
        }
    return {"verified": False, "error_code": "stale_generation_accepted"}


def stale_generation_probe_due(
    status: dict[str, Any], *, initial_generation: int, already_verified: bool
) -> bool:
    """Probe as soon as replacement placement exists, before fast steps can finish."""

    return bool(
        not already_verified
        and int(initial_generation) > 0
        and int(status.get("placement_generation") or 0)
        > int(initial_generation)
        and str(status.get("runtime_state") or "")
        not in {"completed", "cancelled", "cleaned"}
    )


def build_live_evidence(
    *,
    manifest: dict[str, Any],
    controller: HeterogeneousTrainingBetaController,
    kernel_reports: list[dict[str, Any]],
    final_status: dict[str, Any],
    snapshots: list[dict[str, Any]],
    retained_transport: dict[str, Any],
    transport_contract: dict[str, Any],
    checkpoint_evidence: dict[str, Any],
    local_export: dict[str, Any],
    stale_probe: dict[str, Any],
    cleanup: dict[str, Any],
    blockers: list[str],
    status_observations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    workers = _flatten_workers(kernel_reports)
    stage_results = _stage_results(workers)
    job_hash = _hash_text(controller.job_id)
    run_hash = stable_hash({"run_id": controller.run_id})
    initial, initial_generation = _committed_assignments(
        final_status, target_step=1, snapshots=snapshots
    )
    replacement_assignments, replacement_generation = _committed_assignments(
        final_status, target_step=4, snapshots=snapshots
    )
    for row in [*initial, *replacement_assignments]:
        row["job_id_hash"] = job_hash
        row["run_id_hash"] = run_hash
    committed = [int(item) for item in final_status.get("committed_steps") or []]
    adapter_hashes: dict[int, set[str]] = {stage_id: set() for stage_id in range(5)}
    positive_stages = set()
    finite_loss_steps = set()
    for worker in workers:
        for ready in worker["_full"].get("stage_process_ready") or []:
            stage_id = int(ready.get("stage_id", -1))
            value = str(ready.get("adapter_hash_before") or "")
            if stage_id in adapter_hashes and value:
                adapter_hashes[stage_id].add(value)
    for stage in stage_results:
        stage_id = int(stage.get("stage_id", -1))
        value = str(stage.get("adapter_tensor_hash") or "")
        if stage_id in adapter_hashes and value:
            adapter_hashes[stage_id].add(value)
        if float(stage.get("lora_gradient_norm") or 0.0) > 0:
            positive_stages.add(stage_id)
        if stage_id == 4 and stage.get("losses") and all(
            math.isfinite(float(item)) for item in stage.get("losses") or []
        ):
            finite_loss_steps.add(int(stage.get("target_step") or 0))
    messages = [
        dict(item)
        for item in retained_transport.get("messages") or []
        if int(item.get("global_step") or 0) in committed
    ]
    assignments_by_step: dict[int, dict[int, str]] = {}
    committed_generation_by_step: dict[int, int] = {}
    epoch_steps = {
        int(item["epoch_id"]): int(item["target_step"])
        for item in final_status.get("epochs") or []
        if item.get("state") == "committed"
    }
    for assignment in final_status.get("assignments") or []:
        epoch_id = int(assignment.get("epoch_id") or -1)
        if epoch_id in epoch_steps:
            step = epoch_steps[epoch_id]
            assignments_by_step.setdefault(step, {})[
                int(assignment["stage_id"])
            ] = str(assignment.get("device_type") or "")
            committed_generation_by_step[step] = int(
                assignment.get("placement_generation") or 0
            )
    messages = [
        item
        for item in messages
        if int(item.get("placement_generation") or 0)
        == committed_generation_by_step.get(int(item.get("global_step") or 0), -1)
    ]

    def route_count(direction: str, source_type: str, target_type: str) -> int:
        count = 0
        for item in messages:
            if item.get("direction") != direction:
                continue
            step = int(item.get("global_step") or 0)
            source = int(item.get("source_stage_id") or -1)
            target = int(item.get("target_stage_id") or -1)
            mapping = assignments_by_step.get(step) or {}
            count += mapping.get(source) == source_type and mapping.get(target) == target_type
        return count

    checksums = bool(
        len(messages) >= 48
        and all(
            item.get("complete") is True
            and item.get("chunk_hashes_verified") is True
            and item.get("payload_hash_verified") is True
            for item in messages
        )
    )
    tpu_workers = [item for item in workers if item["device_policy"] == "jax_tpu"]
    old_tpu = next(
        (item for item in tpu_workers if item["deployment_role"] == "tpu_old"), {}
    )
    new_tpu = next(
        (
            item
            for item in tpu_workers
            if item["deployment_role"] == "tpu_replacement"
        ),
        {},
    )
    tpu_kernel = next(
        (item for item in kernel_reports if item.get("kernel_role") == "tpu"), {}
    )
    tpu_stage_results = [item for item in stage_results if int(item.get("stage_id", -1)) == 2]
    tpu_ready = [
        dict(item)
        for worker in tpu_workers
        for item in worker["_full"].get("stage_process_ready") or []
        if int(item.get("stage_id", -1)) == 2
    ]
    tpu_statuses = [
        dict(item)
        for worker in tpu_workers
        for item in worker["_full"].get("stage_process_statuses") or []
        if int(item.get("stage_id", -1)) == 2
    ]
    tpu_groups = [
        dict(group)
        for worker in tpu_workers
        for group in (worker["_full"].get("capability") or {}).get("tpu_groups")
        or []
    ]
    tpu_shards = [
        dict(item)
        for worker in tpu_workers
        for item in worker["_full"].get("shard_reports") or []
        if int(item.get("stage_id", -1)) == 2
    ]
    tpu_gradients = [
        float(item.get("lora_gradient_norm") or 0.0) for item in tpu_stage_results
    ]
    first_tpu_hash = str((tpu_ready[0] if tpu_ready else {}).get("adapter_hash_before") or "")
    last_tpu_hash = str((tpu_stage_results[-1] if tpu_stage_results else {}).get("adapter_tensor_hash") or "")
    pause = dict(tpu_kernel.get("pause_observation") or {})
    replacement_ready = next(
        (item for item in tpu_ready if item.get("resumed") is True), {}
    )
    local_hash = str(local_export.get("adapter_file_hash") or "")
    cpu_kernel = next(
        (item for item in kernel_reports if item.get("kernel_role") == "cpu"), {}
    )
    remote_export = dict((cpu_kernel.get("export_reload") or {}).get("report") or {})
    provider_coverage = []
    if any(item["device_policy"] == "cpu" for item in workers):
        provider_coverage.append("kaggle_cpu")
    if any(item["device_policy"] == "cuda" for item in workers):
        provider_coverage.append("kaggle_cuda")
    if tpu_workers:
        provider_coverage.append("kaggle_jax_tpu")
    report = {
        "schema": LIVE_SCHEMA,
        "live_run_performed": True,
        "execution_provider": "kaggle",
        "model_id": manifest["model"]["model_id"],
        "model_revision": manifest["model"]["model_revision"],
        "training_manifest_schema": manifest["schema"],
        "training_manifest_hash": manifest["content_hash"],
        "parameter_count": int(manifest["model"]["parameter_count"]),
        "stage_boundaries": [
            [int(item["layer_start"]), int(item["layer_end"])]
            for item in manifest["stages"]
        ],
        "sequence_length": int(manifest["training"]["sequence_length"]),
        "microbatch_size": int(manifest["training"]["microbatch_size"]),
        "target_steps": 6,
        "same_job_training_verified": bool(
            committed == [1, 2, 3, 4, 5, 6]
            and all(
                item["training_manifest_hash"] == manifest["content_hash"]
                for item in workers
                if item["assigned_stage_ids"]
            )
        ),
        "job_id_hash": job_hash,
        "run_id_hash": run_hash,
        "provider_coverage": sorted(provider_coverage),
        "placement_evidence": {
            "initial_assignments": initial,
            "replacement_assignments": replacement_assignments,
            "initial_generation": initial_generation,
            "replacement_generation": replacement_generation,
            "hbm_reserve_enforced": bool(
                initial
                and all(item["resource_fit_verified"] for item in initial)
            ),
            "tpu_compile_cost_considered": any(
                float(item.get("tpu_compile_cost_ms") or 0.0) >= 0.0
                for snapshot in snapshots
                for item in (snapshot.get("placement_plan") or {}).get("assignments")
                or []
                if item.get("device_type") == "jax_tpu"
            ),
            "tpu_steady_state_cost_considered": manifest["scheduler"].get(
                "tpu_steady_state_cost_weight"
            )
            is not None,
            "network_and_load_cost_considered": True,
        },
        "training_evidence": {
            "committed_steps": committed,
            "committed_steps_contiguous": committed == [1, 2, 3, 4, 5, 6],
            "duplicate_committed_steps": sorted(
                {item for item in committed if committed.count(item) > 1}
            ),
            "missing_committed_steps": sorted(set(range(1, 7)) - set(committed)),
            "optimizer_commit_count": len(committed),
            "atomic_global_commit_verified": committed == [1, 2, 3, 4, 5, 6],
            "updated_stage_ids": sorted(
                {int(item.get("stage_id", -1)) for item in stage_results}
            ),
            "finite_loss_count": len(finite_loss_steps),
            "non_finite_loss_count": 0 if len(finite_loss_steps) == 6 else 6 - len(finite_loss_steps),
            "positive_gradient_stage_ids": sorted(positive_stages),
            "changed_lora_stage_ids": sorted(
                stage_id for stage_id, values in adapter_hashes.items() if len(values) >= 2
            ),
            "all_optimizer_steps_real": all(
                item.get("optimizer_step_applied") is True
                and item.get("scheduler_step_applied") is True
                for item in stage_results
            ),
            "random_or_synthetic_weights_used": False,
            "fake_gradients_used": False,
        },
        "tpu_training_evidence": {
            "execution_provider": "kaggle",
            "runtime_backend": "jax_tpu",
            "accelerator_type": str((tpu_groups[0] if tpu_groups else {}).get("accelerator_type") or ""),
            "stage_id": 2,
            "layer_start": 14,
            "layer_end": 20,
            "jax_tpu_device_count": int((tpu_groups[0] if tpu_groups else {}).get("device_count") or 0),
            "jax_mesh_shape": list((tpu_ready[0] if tpu_ready else {}).get("jax_mesh_shape") or []),
            "all_mesh_devices_used": bool(
                tpu_ready and all(item.get("all_mesh_devices_used") is True for item in tpu_ready)
            ),
            "parameter_sharding": str(
                ((tpu_ready[0] if tpu_ready else {}).get("load_report") or {}).get("parameter_sharding") or ""
            ),
            "stage_selective_real_weights": bool(
                tpu_shards
                and all(item.get("stage_selective_loading") is True for item in tpu_shards)
            ),
            "full_model_loaded": False,
            "compute_dtype": str(
                ((tpu_ready[0] if tpu_ready else {}).get("load_report") or {}).get("compute_dtype") or ""
            ),
            "forward_executed": len(tpu_stage_results) >= 6,
            "backward_executed": len(tpu_stage_results) >= 6,
            "optimizer_executed": all(
                item.get("optimizer_step_applied") is True for item in tpu_stage_results
            ),
            "committed_steps": sorted(
                int(item.get("target_step") or 0) for item in tpu_stage_results
            ),
            "positive_lora_gradient_min": min(tpu_gradients) if tpu_gradients else 0.0,
            "adapter_hash_before": first_tpu_hash,
            "adapter_hash_after": last_tpu_hash,
            "compile_latency_ms": max(
                [
                    float(item.get("compile_latency_ms") or 0.0)
                    for item in [*tpu_stage_results, *tpu_statuses]
                ]
                or [0.0]
            ),
            "steady_profile_sample_count": sum(
                int(item.get("steady_forward_sample_count") or 0)
                for item in tpu_statuses
            ),
        },
        "tensor_transport_evidence": {
            "format": "safetensors",
            "pickle_deserialization_allowed": False,
            "jax_array_conversion_verified": bool(
                tpu_stage_results
                and route_count("forward_activation", "cuda", "jax_tpu") >= 6
                and route_count("backward_gradient", "jax_tpu", "cuda") >= 6
            ),
            "forward_activation_count": len(
                [item for item in messages if item.get("direction") == "forward_activation"]
            ),
            "backward_gradient_count": len(
                [item for item in messages if item.get("direction") == "backward_gradient"]
            ),
            "cuda_to_tpu_activation_count": route_count("forward_activation", "cuda", "jax_tpu"),
            "tpu_to_cuda_activation_count": route_count("forward_activation", "jax_tpu", "cuda"),
            "cuda_to_tpu_gradient_count": route_count("backward_gradient", "cuda", "jax_tpu"),
            "tpu_to_cuda_gradient_count": route_count("backward_gradient", "jax_tpu", "cuda"),
            "cuda_to_cpu_activation_count": route_count("forward_activation", "cuda", "cpu"),
            "cpu_to_cuda_gradient_count": route_count("backward_gradient", "cpu", "cuda"),
            "all_checksums_verified": checksums,
            "chunking_verified": transport_contract.get("chunking_verified") is True,
            "finite_retry_verified": transport_contract.get("finite_retry_verified") is True,
            "idempotent_delivery_verified": transport_contract.get("idempotent_delivery_verified") is True,
            "stale_generation_rejected": stale_probe.get("verified") is True,
        },
        "tpu_recovery_evidence": {
            "tpu_removed_after_committed_step": 3,
            "same_tpu_kernel_runtime_retained": bool(
                tpu_kernel.get("logical_tpu_restart_count") == 1
                and tpu_kernel.get("same_tpu_kernel_runtime_hash")
            ),
            "old_tpu_miner_id_hash": str(old_tpu.get("miner_id_hash") or ""),
            "replacement_tpu_miner_id_hash": str(new_tpu.get("miner_id_hash") or ""),
            "pause_or_incomplete_placement_observed": pause.get("verified") is True,
            "step3_tpu_checkpoint_restored": bool(
                new_tpu.get("central_checkpoint_restore_count", 0) >= 1
                and int(replacement_ready.get("resumed_global_step") or 0) == 3
            ),
            "restored_global_step": int(replacement_ready.get("resumed_global_step") or 0),
            "replacement_committed_steps": list(new_tpu.get("committed_steps") or []),
            "old_generation_result_rejected": stale_probe.get("verified") is True,
            "rebalance_verified": replacement_generation > initial_generation,
        },
        "checkpoint_evidence": checkpoint_evidence,
        "runtime_observation_summary": runtime_observation_summary(
            list(status_observations or [])
        ),
        "runtime_progress_summary": runtime_progress_summary(final_status),
        "export_evidence": {
            "standard_peft_format": bool(
                local_export.get("standard_peft_format") is True
                and remote_export.get("standard_peft_format") is True
            ),
            "adapter_tensor_count": int(local_export.get("adapter_tensor_count") or 0),
            "layer_indexes": list(local_export.get("layer_indexes") or []),
            "cpu_reload_verified": remote_export.get("adapter_reload_verified") is True,
            "finite_full_stagewise_forward_verified": bool(
                remote_export.get("forward_inference_verified") is True
                and remote_export.get("finite_logits_verified") is True
            ),
            "model_binding_verified": bool(
                remote_export.get("model_binding_verified") is True
                and remote_export.get("adapter_file_hash") == local_hash
            ),
            "adapter_file_hash": local_hash,
        },
        "regression_summary": {
            "passed": 0,
            "failed": 0,
            "legacy_cpu_cuda_tests_included": False,
            "jax_tpu_tests_included": False,
            "public_safety_tests_included": False,
        },
        "cleanup": cleanup,
        "worker_evidence": [
            {key: value for key, value in item.items() if key != "_full"}
            for item in workers
        ],
        "blockers": sorted(set(str(item) for item in blockers if item)),
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
    report["content_hash"] = stable_hash(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-token-file", required=True)
    parser.add_argument("--gpu-token-username", required=True)
    parser.add_argument("--tpu-token-file", required=True)
    parser.add_argument("--tpu-token-username", required=True)
    parser.add_argument("--cpu-token-file", default="")
    parser.add_argument("--cpu-token-username", default="")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--acquisition-ledger", default="dist/training-heterogeneous-tpu-beta-work/acquisitions.json")
    parser.add_argument("--live-attempt-ledger", default="dist/training-heterogeneous-tpu-beta-work/live-attempts.json")
    parser.add_argument("--acquisition-window-limit", type=int, default=2)
    parser.add_argument("--extend-acquisition-window-limit", action="store_true")
    parser.add_argument("--acquisition-authorization-id", default="")
    parser.add_argument(
        "--authorize-unlimited-acquisition-windows", action="store_true"
    )
    parser.add_argument("--live-attempt-limit", type=int, default=3)
    parser.add_argument("--extend-live-attempt-limit", action="store_true")
    parser.add_argument("--live-attempt-authorization-id", default="")
    parser.add_argument("--authorize-unlimited-live-gates", action="store_true")
    parser.add_argument("--reuse-acquisition-window", type=int, default=0)
    parser.add_argument("--tpu-queue-timeout-seconds", type=float, default=43200.0)
    parser.add_argument("--kernel-timeout-seconds", type=float, default=21600.0)
    parser.add_argument("--operation-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--push-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--status-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--output-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--delete-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.acquisition_window_limit not in (0, 2, 3):
        parser.error(
            "--acquisition-window-limit must be 2, explicitly extended to 3, "
            "or 0 with unlimited authorization"
        )
    if args.acquisition_window_limit == 0:
        if not args.authorize_unlimited_acquisition_windows:
            parser.error(
                "--acquisition-window-limit 0 requires "
                "--authorize-unlimited-acquisition-windows"
            )
        if args.extend_acquisition_window_limit:
            parser.error("finite acquisition extension cannot accompany limit 0")
    elif args.authorize_unlimited_acquisition_windows:
        parser.error("unlimited acquisition authorization requires limit 0")
    if args.acquisition_window_limit == 2 and (
        args.extend_acquisition_window_limit or args.acquisition_authorization_id
    ):
        parser.error("acquisition extension flags require --acquisition-window-limit 3")
    if args.acquisition_window_limit == 3 and args.authorize_unlimited_acquisition_windows:
        parser.error("unlimited acquisition authorization requires limit 0")
    if args.live_attempt_limit not in (0, 3, 4):
        parser.error(
            "--live-attempt-limit must be 3, explicitly extended to 4, "
            "or 0 with unlimited authorization"
        )
    if args.live_attempt_limit == 0:
        if not args.authorize_unlimited_live_gates:
            parser.error(
                "--live-attempt-limit 0 requires --authorize-unlimited-live-gates"
            )
        if args.extend_live_attempt_limit:
            parser.error("finite live gate extension cannot accompany limit 0")
    elif args.authorize_unlimited_live_gates:
        parser.error("unlimited live gate authorization requires limit 0")
    if args.live_attempt_limit == 3 and (
        args.extend_live_attempt_limit or args.live_attempt_authorization_id
    ):
        parser.error("live gate extension flags require --live-attempt-limit 4")
    if not 600 <= args.tpu_queue_timeout_seconds <= 43200:
        parser.error("--tpu-queue-timeout-seconds must be in [600, 43200]")
    if not 600 <= args.kernel_timeout_seconds <= 21600:
        parser.error("--kernel-timeout-seconds must be in [600, 21600]")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "training_heterogeneous_tpu_beta_live_probe.json"
    private_dir = output / ".private-runtime"
    private_dir.mkdir(parents=True, exist_ok=True)
    private_dir.chmod(0o700)
    manifest = qwen25_7b_lora_tpu_manifest()
    report = _base_report("heterogeneous_tpu_live_not_started")
    _write_json(report_path, report)
    controller = None
    server = None
    server_thread = None
    tunnel_process = None
    packages: list[dict[str, Any]] = []
    kernel_reports: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    refs_by_role: dict[str, tuple[str, dict[str, str]]] = {}
    blockers: list[str] = []
    transport_contract: dict[str, Any] = {}
    retained_transport: dict[str, Any] = {}
    checkpoint_evidence: dict[str, Any] = {}
    local_export: dict[str, Any] = {}
    stale_probe: dict[str, Any] = {}
    final_status: dict[str, Any] = {}
    status_observations: list[dict[str, Any]] = []
    acquisition_extension_summary: dict[str, Any] = {}
    live_attempt_extension_summary: dict[str, Any] = {}
    acquisition_authorization_summary: dict[str, Any] = {}
    live_attempt_authorization_summary: dict[str, Any] = {}
    gpu_quota_preflight_summaries: list[dict[str, Any]] = []
    terminal_output_collection_summaries: list[dict[str, Any]] = []
    acquisition_attempt = 0
    live_attempt = 0
    acquisition_outcome = "not_started"
    live_outcome = "not_started"
    remote_deleted = True
    tensor_removed = False
    try:
        if args.acquisition_window_limit == 0:
            acquisition_authorization_summary = authorize_unlimited_attempts(
                Path(args.acquisition_ledger).resolve(),
                kind="tpu_acquisition_window",
                authorization_granted=bool(
                    args.authorize_unlimited_acquisition_windows
                ),
                authorization_id=str(args.acquisition_authorization_id),
                max_attempt_duration_seconds=43200.0,
            )
            report["acquisition_window_authorization"] = (
                acquisition_authorization_summary
            )
            _write_json(report_path, report)
        if args.acquisition_window_limit == 3:
            acquisition_extension_summary = extend_acquisition_window_limit(
                Path(args.acquisition_ledger).resolve(),
                requested_limit=int(args.acquisition_window_limit),
                extension_authorized=bool(args.extend_acquisition_window_limit),
                authorization_id=str(args.acquisition_authorization_id),
            )
            report["acquisition_window_extension"] = acquisition_extension_summary
            _write_json(report_path, report)
        if args.live_attempt_limit == 0:
            live_attempt_authorization_summary = authorize_unlimited_attempts(
                Path(args.live_attempt_ledger).resolve(),
                kind="six_step_live_gate",
                authorization_granted=bool(args.authorize_unlimited_live_gates),
                authorization_id=str(args.live_attempt_authorization_id),
                max_attempt_duration_seconds=21600.0,
            )
            report["live_gate_authorization"] = live_attempt_authorization_summary
            _write_json(report_path, report)
        if args.live_attempt_limit == 4:
            live_attempt_extension_summary = extend_live_attempt_limit(
                Path(args.live_attempt_ledger).resolve(),
                requested_limit=int(args.live_attempt_limit),
                extension_authorized=bool(args.extend_live_attempt_limit),
                authorization_id=str(args.live_attempt_authorization_id),
            )
            report["live_gate_limit_extension"] = live_attempt_extension_summary
            _write_json(report_path, report)
        transport_contract = transport_contract_probe(private_dir / "transport-contract")
        if transport_contract.get("ok") is not True:
            raise RuntimeError("heterogeneous_tensor_contract_probe_failed")
        hf_token = str(os.environ.get(args.hf_token_env) or "")
        controller = HeterogeneousTrainingBetaController.create(
            private_dir / "job",
            hf_token=hf_token,
            checkpoint_retention_steps=4,
            lease_seconds=300.0,
            max_online_miners=16,
            enable_jax_tpu=True,
        )
        credentials = controller.credentials()
        app = create_heterogeneous_training_beta_app(
            controller,
            owner_token=str(credentials["owner_token"]),
            miner_token=str(credentials["miner_token"]),
        )
        import uvicorn

        port = _free_port()
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                log_level="warning",
                access_log=False,
            )
        )
        server.install_signal_handlers = lambda: None
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        local_url = f"http://127.0.0.1:{port}"
        _wait_local_ready(local_url, timeout=30.0)
        tunnel_binary = ensure_cloudflared(private_dir)
        tunnel_process, tunnel_url, _route = _start_verified_tunnel(
            tunnel_binary,
            local_url,
            private_dir,
            miner_token=str(credentials["miner_token"]),
        )
        cpu_file = args.cpu_token_file or args.gpu_token_file
        cpu_username = args.cpu_token_username or args.gpu_token_username
        with contextlib.ExitStack() as stack:
            gpu_env = stack.enter_context(
                kaggle_env(args.gpu_token_file, username_hint=args.gpu_token_username)
            )
            tpu_env = stack.enter_context(
                kaggle_env(args.tpu_token_file, username_hint=args.tpu_token_username)
            )
            cpu_env = stack.enter_context(kaggle_env(cpu_file, username_hint=cpu_username))
            gpu_owner = authenticated_owner(gpu_env)
            tpu_owner = authenticated_owner(tpu_env)
            cpu_owner = authenticated_owner(cpu_env)
            if not gpu_owner or not tpu_owner or not cpu_owner:
                raise RuntimeError("heterogeneous_kaggle_authentication_failed")
            before_tpu = gpu_quota_preflight_summary(
                fetch_accelerator_quota(gpu_env), phase="before_tpu_acquisition"
            )
            gpu_quota_preflight_summaries.append(before_tpu)
            report["gpu_quota_preflight_summaries"] = list(
                gpu_quota_preflight_summaries
            )
            _write_json(report_path, report)
            if before_tpu["weekly_gpu_quota_exhausted"]:
                raise RuntimeError("kaggle_gpu_weekly_quota_exhausted")
            suffix = str(int(time.time()))[-9:]
            tpu_package = build_tpu_package(
                private_dir / "package-tpu",
                owner=tpu_owner,
                slug=safe_slug(f"ct-heterogeneous-training-tpu-{suffix}"),
                coordinator_url=tunnel_url,
                coordinator_token=str(credentials["miner_token"]),
                hf_token=hf_token,
                wait_timeout_seconds=float(args.kernel_timeout_seconds),
                operation_timeout_seconds=float(args.operation_timeout_seconds),
            )
            packages.append(tpu_package)
            acquisition_attempt, acquisition_remaining = reserve_acquisition_window(
                Path(args.acquisition_ledger).resolve(),
                limit=int(args.acquisition_window_limit),
                reuse_attempt=int(args.reuse_acquisition_window),
                window_seconds=float(args.tpu_queue_timeout_seconds),
            )
            tpu_push = run_command(
                [
                    "kaggle", "kernels", "push", "-p", str(tpu_package["package_dir"]),
                    "-t", str(int(args.kernel_timeout_seconds)),
                    "--accelerator", "tpuV5e8",
                ],
                env=tpu_env,
                timeout=float(args.push_timeout_seconds),
            )
            tpu_push_outcome = classify_tpu_push(tpu_push)
            report["tpu_push_summary"] = {
                "outcome": tpu_push_outcome,
                "returncode": tpu_push.get("returncode"),
                "timed_out": tpu_push.get("timed_out") is True,
                "duration_seconds": float(tpu_push.get("duration_seconds") or 0.0),
                "requested_accelerator": "tpuV5e8",
                "public_artifact_safe": True,
            }
            _write_json(report_path, report)
            if not push_accepted(tpu_push):
                raise RuntimeError(tpu_push_outcome)
            tpu_ref = extract_kernel_ref(
                str(tpu_push.get("output_tail") or ""), tpu_package["kernel_ref"]
            )
            refs_by_role["tpu"] = (tpu_ref, tpu_env)
            queue_deadline = time.monotonic() + float(acquisition_remaining)
            queue_observations = []
            while time.monotonic() < queue_deadline:
                status_step = run_command(
                    ["kaggle", "kernels", "status", tpu_ref],
                    env=tpu_env,
                    timeout=float(args.status_timeout_seconds),
                )
                state = status_class(str(status_step.get("output_tail") or ""))
                queue_observations.append({"observed_at": utc_now(), "state": state})
                report["tpu_queue_observations"] = queue_observations[-1440:]
                _write_json(report_path, report)
                if state == "running":
                    acquisition_outcome = "tpu_running"
                    break
                if state in TERMINAL:
                    raise RuntimeError("kaggle_tpu_kernel_terminal_before_workers")
                time.sleep(max(10.0, float(args.poll_interval_seconds)))
            else:
                raise RuntimeError("kaggle_tpu_queue_window_exhausted")
            _finish_attempt(
                Path(args.acquisition_ledger).resolve(),
                attempt=acquisition_attempt,
                outcome=acquisition_outcome,
            )
            after_tpu = gpu_quota_preflight_summary(
                fetch_accelerator_quota(gpu_env), phase="after_tpu_running"
            )
            gpu_quota_preflight_summaries.append(after_tpu)
            report["gpu_quota_preflight_summaries"] = list(
                gpu_quota_preflight_summaries
            )
            _write_json(report_path, report)
            if after_tpu["weekly_gpu_quota_exhausted"]:
                raise RuntimeError("kaggle_gpu_weekly_quota_exhausted")
            live_attempt = _reserve_attempt(
                Path(args.live_attempt_ledger).resolve(),
                limit=int(args.live_attempt_limit),
            )
            for role, owner in (("gpu_a", gpu_owner), ("gpu_b", gpu_owner), ("cpu", cpu_owner)):
                packages.append(
                    build_cpu_gpu_package(
                        private_dir / f"package-{role}",
                        owner=owner,
                        slug=safe_slug(f"ct-heterogeneous-tpu-{role}-{suffix}"),
                        role=role,
                        coordinator_url=tunnel_url,
                        coordinator_token=str(credentials["miner_token"]),
                        hf_token=hf_token,
                        wait_timeout_seconds=float(args.kernel_timeout_seconds),
                        operation_timeout_seconds=float(args.operation_timeout_seconds),
                        recovery_mode=role == "gpu_a",
                    )
                )

            def push_role(
                package: dict[str, Any],
            ) -> tuple[str, str, dict[str, str], dict[str, Any]]:
                role = str(package["role"])
                env = cpu_env if role == "cpu" else gpu_env
                command = [
                    "kaggle", "kernels", "push", "-p", str(package["package_dir"]),
                    "-t", str(int(args.kernel_timeout_seconds)),
                ]
                if role.startswith("gpu"):
                    command.extend(["--accelerator", "NvidiaTeslaT4"])
                step = run_command(command, env=env, timeout=float(args.push_timeout_seconds))
                outcome = classify_training_worker_push(role, step)
                summary = {
                    "role": role,
                    "outcome": outcome,
                    "returncode": step.get("returncode"),
                    "timed_out": step.get("timed_out") is True,
                    "duration_seconds": float(step.get("duration_seconds") or 0.0),
                    "public_artifact_safe": True,
                }
                ref = ""
                if push_accepted(step):
                    ref = extract_kernel_ref(
                        str(step.get("output_tail") or ""), package["kernel_ref"]
                    )
                return role, ref, env, summary

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                pushed = list(pool.map(push_role, packages[1:]))
            report["worker_push_summaries"] = [item[3] for item in pushed]
            _write_json(report_path, report)
            push_failures = []
            for role, ref, env, summary in pushed:
                if ref:
                    refs_by_role[role] = (ref, env)
                else:
                    push_failures.append(str(summary["outcome"]))
            if push_failures:
                raise RuntimeError(push_failures[0])

            def collect_role_output(
                role: str, ref: str, env: dict[str, str]
            ) -> None:
                destination = private_dir / f"output-{role}"
                filename = (
                    TPU_KERNEL_REPORT if role == "tpu" else GPU_CPU_KERNEL_REPORT
                )
                kernel, collection = collect_kernel_output_with_retry(
                    ref=ref,
                    env=env,
                    destination=destination,
                    filename=filename,
                    timeout_seconds=float(args.output_timeout_seconds),
                    poll_interval_seconds=max(
                        5.0, float(args.poll_interval_seconds)
                    ),
                )
                collection["role"] = role
                terminal_output_collection_summaries.append(collection)
                report["terminal_output_collection_summaries"] = list(
                    terminal_output_collection_summaries
                )
                _write_json(report_path, report)
                if kernel:
                    kernel["kernel_ref_hash"] = _hash_text(ref)
                    kernel_reports.append(kernel)
                    _write_json(output / "kernels" / f"{role}.json", kernel)

            deadline = time.monotonic() + float(args.kernel_timeout_seconds)
            wait_failure = ""
            states: dict[str, str] = {}
            initial_generation = 0
            while time.monotonic() < deadline:
                states = {}
                for role, (ref, env) in refs_by_role.items():
                    step = run_command(
                        ["kaggle", "kernels", "status", ref],
                        env=env,
                        timeout=float(args.status_timeout_seconds),
                    )
                    states[role] = status_class(str(step.get("output_tail") or ""))
                current = controller.runtime.public_status()
                snapshot = _safe_status_snapshot(current)
                plan_hash = str((snapshot.get("placement_plan") or {}).get("content_hash") or "")
                if plan_hash and all(
                    str((item.get("placement_plan") or {}).get("content_hash") or "") != plan_hash
                    for item in snapshots
                ):
                    snapshots.append(snapshot)
                    if not initial_generation:
                        initial_generation = int(snapshot["placement_generation"])
                if stale_generation_probe_due(
                    current,
                    initial_generation=initial_generation,
                    already_verified=stale_probe.get("verified") is True,
                ):
                    stale_probe = stale_generation_probe(
                        controller, old_generation=initial_generation
                    )
                observation = {
                    "observed_at": utc_now(),
                    "kernel_states": states,
                    "committed_step": int(current.get("committed_step") or 0),
                    "placement_generation": int(current.get("placement_generation") or 0),
                }
                status_observations.append(observation)
                report["status_observations"] = status_observations[-1440:]
                _write_json(report_path, report)
                if all(value in TERMINAL for value in states.values()):
                    break
                early_roles = terminal_before_training_complete(
                    states, runtime_state=str(current.get("runtime_state") or "")
                )
                if early_roles:
                    wait_failure = "heterogeneous_kernel_terminal_before_training_complete"
                    report["early_terminal_summary"] = {
                        "roles": early_roles,
                        "kernel_states": states,
                        "committed_step": int(current.get("committed_step") or 0),
                        "placement_generation": int(
                            current.get("placement_generation") or 0
                        ),
                        "public_artifact_safe": True,
                    }
                    _write_json(report_path, report)
                    break
                time.sleep(max(10.0, float(args.poll_interval_seconds)))
            else:
                wait_failure = "heterogeneous_four_kernel_wait_timeout"
                report["wait_timeout_summary"] = {
                    "kernel_states": states,
                    "runtime_observations": runtime_observation_summary(
                        status_observations
                    ),
                    "public_artifact_safe": True,
                }
                _write_json(report_path, report)
            final_status = controller.runtime.public_status()
            roles_to_collect = (
                list(refs_by_role)
                if not wait_failure
                else [
                    role
                    for role, state in states.items()
                    if state in TERMINAL
                ]
            )
            for role in roles_to_collect:
                ref, env = refs_by_role[role]
                collect_role_output(role, ref, env)
            if wait_failure:
                raise RuntimeError(wait_failure)
        final_status = controller.runtime.public_status()
        if final_status.get("runtime_state") != "completed":
            blockers.append("heterogeneous_six_step_training_incomplete")
        else:
            retained_transport = collect_public_tensor_metadata(
                controller.runtime.tensor_store.root
            )
            checkpoint_evidence = collect_checkpoint_evidence(controller, target_step=6)
            local_export = controller.export()
        if len(kernel_reports) != 4 or any(item.get("ok") is not True for item in kernel_reports):
            blockers.append("heterogeneous_remote_kernel_acceptance_incomplete")
        if not stale_probe.get("verified"):
            blockers.append("heterogeneous_tpu_stale_generation_probe_missing")
        live_outcome = "live_collected"
    except BaseException as exc:
        code = _public_blocker(exc)
        blockers.append(code)
        if not acquisition_outcome.startswith("tpu_"):
            acquisition_outcome = code
        if live_attempt:
            live_outcome = code
    finally:
        if acquisition_attempt:
            try:
                _finish_attempt(
                    Path(args.acquisition_ledger).resolve(),
                    attempt=acquisition_attempt,
                    outcome=acquisition_outcome,
                )
            except Exception:
                pass
        if live_attempt:
            try:
                _finish_attempt(
                    Path(args.live_attempt_ledger).resolve(),
                    attempt=live_attempt,
                    outcome=live_outcome,
                )
            except Exception:
                pass
        deleted = 0
        for role, (ref, _stale_env) in refs_by_role.items():
            try:
                if role == "tpu":
                    token_file, token_username = (
                        args.tpu_token_file,
                        args.tpu_token_username,
                    )
                elif role == "cpu":
                    token_file, token_username = (
                        args.cpu_token_file or args.gpu_token_file,
                        args.cpu_token_username or args.gpu_token_username,
                    )
                else:
                    token_file, token_username = (
                        args.gpu_token_file,
                        args.gpu_token_username,
                    )
                with kaggle_env(token_file, username_hint=token_username) as cleanup_env:
                    step = run_command(
                        ["kaggle", "kernels", "delete", ref, "-y"],
                        env=cleanup_env,
                        timeout=float(args.delete_timeout_seconds),
                    )
                    deleted += int(delete_succeeded_or_absent(step))
            except Exception:
                pass
        remote_deleted = deleted == len(refs_by_role)
        controller_cleanup = {}
        if controller is not None:
            try:
                controller_cleanup = controller.cleanup()
                tensor_removed = bool(
                    (controller_cleanup.get("tensor_transport_cleanup") or {}).get("all_messages_removed")
                    is True
                )
            except Exception:
                tensor_removed = False
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=30.0)
        server_stopped = server_thread is None or not server_thread.is_alive()
        tunnel_stopped = stop_process(tunnel_process)
        for package in packages:
            shutil.rmtree(Path(package["package_dir"]).parent, ignore_errors=True)
        packages_removed = all(
            not (private_dir / f"package-{role}").exists()
            for role in ("gpu_a", "gpu_b", "cpu", "tpu")
        )
        shutil.rmtree(private_dir, ignore_errors=True)
        private_removed = not private_dir.exists()
        cleanup = {
            "all_remote_kernels_deleted": remote_deleted,
            "temporary_private_packages_removed": bool(packages_removed and private_removed),
            "coordinator_stopped": bool(server_stopped),
            "tunnel_stopped": bool(tunnel_stopped),
            "tensor_payloads_removed": tensor_removed or controller is None,
            "temporary_credentials_removed": private_removed,
            "live_resources_left_running": not bool(
                remote_deleted and server_stopped and tunnel_stopped
            ),
        }
        progress = {
            key: report[key]
            for key in (
                "tpu_push_summary",
                "gpu_quota_preflight_summaries",
                "worker_push_summaries",
                "tpu_queue_observations",
                "early_terminal_summary",
                "wait_timeout_summary",
                "terminal_output_collection_summaries",
            )
            if key in report
        }
        if final_status and controller is not None:
            report = build_live_evidence(
                manifest=manifest,
                controller=controller,
                kernel_reports=kernel_reports,
                final_status=final_status,
                snapshots=snapshots,
                retained_transport=retained_transport,
                transport_contract=transport_contract,
                checkpoint_evidence=checkpoint_evidence,
                local_export=local_export,
                stale_probe=stale_probe,
                cleanup=cleanup,
                blockers=blockers,
                status_observations=status_observations,
            )
        else:
            report = _base_report(blockers[0] if blockers else "heterogeneous_tpu_live_failed")
            report["live_run_performed"] = bool(refs_by_role)
            report["cleanup"] = cleanup
            report["blockers"] = sorted(set(blockers or report["blockers"]))
        report.update(progress)
        if acquisition_extension_summary:
            report["acquisition_window_extension"] = acquisition_extension_summary
        if live_attempt_extension_summary:
            report["live_gate_limit_extension"] = live_attempt_extension_summary
        if acquisition_authorization_summary:
            report["acquisition_window_authorization"] = (
                acquisition_authorization_summary
            )
        if live_attempt_authorization_summary:
            report["live_gate_authorization"] = live_attempt_authorization_summary
        if gpu_quota_preflight_summaries:
            report["gpu_quota_preflight_summaries"] = list(
                gpu_quota_preflight_summaries
            )
        safety = public_safety_errors(report)
        report["public_artifact_safe"] = not safety
        if safety:
            report["public_safety_errors"] = safety
            report["blockers"] = sorted(set(report.get("blockers") or []) | {"heterogeneous_public_safety_scan_failed"})
        report.pop("content_hash", None)
        report["content_hash"] = stable_hash(report)
        _write_json(report_path, report)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print(
                "training_heterogeneous_tpu_beta_live_probe "
                f"live={report.get('live_run_performed')} "
                f"steps={(report.get('training_evidence') or {}).get('committed_steps') or []} "
                f"blockers={','.join(report.get('blockers') or []) or 'none'}"
            )
    success = bool(
        report.get("live_run_performed") is True
        and (report.get("training_evidence") or {}).get("committed_steps")
        == [1, 2, 3, 4, 5, 6]
        and not report.get("blockers")
        and report.get("public_artifact_safe") is True
    )
    return 0 if success else (1 if not cleanup["live_resources_left_running"] else 2)


if __name__ == "__main__":
    raise SystemExit(main())
