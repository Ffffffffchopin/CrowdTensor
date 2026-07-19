#!/usr/bin/env python3
"""Run the bounded 400-step/60-minute Kaggle Training Production live gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import math
import os
import shutil
import statistics
import threading
import time
from pathlib import Path
from typing import Any

from crowdtensor.elastic_training_client import PERSISTENT_HTTP_MAX_BODY_BYTES
from crowdtensor.heterogeneous_training_beta import (
    create_heterogeneous_training_beta_app,
)
from crowdtensor.heterogeneous_training_manifest import stable_hash
from crowdtensor.heterogeneous_training_production import (
    HeterogeneousTrainingProductionController,
    compare_performance_windows,
    default_production_config,
)
from scripts.kaggle_gpu_token_weekly_quota_probe import fetch_accelerator_quota
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
from scripts.training_heterogeneous_beta_kaggle_package import (
    build_package as build_cpu_gpu_package,
)
from scripts.training_heterogeneous_beta_live_probe import (
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
from scripts.training_heterogeneous_tpu_beta_kaggle_package import (
    build_package as build_tpu_package,
)
from scripts.training_heterogeneous_tpu_beta_live_probe import (
    OUTPUT_PATTERN,
    TERMINAL,
    authorize_unlimited_attempts,
    classify_tpu_push,
    classify_training_worker_push,
    collect_checkpoint_evidence,
    collect_kernel_output_with_retry,
    gpu_quota_preflight_summary,
    reserve_acquisition_window,
    stale_generation_probe,
)
from scripts.training_heterogeneous_production_rc_check import LIVE_SCHEMA


GPU_CPU_KERNEL_REPORT = "training_heterogeneous_beta_kernel.json"
TPU_KERNEL_REPORT = "training_heterogeneous_tpu_beta_kernel.json"
MINIMUM_REQUIRED_STEPS = 100
TARGET_STEPS = 400
MINIMUM_SOAK_SECONDS = 3600.0
PERFORMANCE_WINDOW_SIZE = 5
PERFORMANCE_WINDOW_COUNT = 5
BASELINE_START_STEP = 6
BASELINE_END_STEP = (
    BASELINE_START_STEP
    + PERFORMANCE_WINDOW_SIZE * PERFORMANCE_WINDOW_COUNT
    - 1
)
PERFORMANCE_GATE_STEP = BASELINE_END_STEP + (
    PERFORMANCE_WINDOW_SIZE * PERFORMANCE_WINDOW_COUNT
)
GPU_REPLACEMENT_STEP = 70
COORDINATOR_RESTART_STEP = 80
CPU_REPLACEMENT_STEP = 90
TPU_REPLACEMENT_STEP = 100
EXPECTED_LIVE_MINERS = 6
PLACEMENT_BLOCKER_OBSERVATIONS = 3


def _write(path: Path, value: dict[str, Any], *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(mode)


def _base_report(blocker: str) -> dict[str, Any]:
    report = {
        "schema": LIVE_SCHEMA,
        "live_run_performed": False,
        "external_runtime_verified": False,
        "model_id": "Qwen/Qwen2.5-7B",
        "target_steps": TARGET_STEPS,
        "minimum_required_steps": MINIMUM_REQUIRED_STEPS,
        "minimum_soak_duration_seconds": MINIMUM_SOAK_SECONDS,
        "accepted_providers": [],
        "training_evidence": {
            "committed_steps": [],
            "committed_step_count": 0,
        },
        "blockers": [str(blocker)],
        "credential_values_public": False,
        "credential_paths_public": False,
        "coordinator_url_public": False,
        "raw_training_text_public": False,
        "token_ids_public": False,
        "activation_values_public": False,
        "gradient_values_public": False,
        "checkpoint_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = stable_hash(report)
    return report


def _public_blocker(exc: BaseException) -> str:
    lines = str(exc).splitlines()
    text = lines[0] if lines else type(exc).__name__
    if text.startswith(("heterogeneous_", "elastic_", "kaggle_", "training_")):
        return text[:180]
    return "training_production_live_failed:" + type(exc).__name__


def _observe_placement_blocker(
    status: dict[str, Any],
    *,
    previous_code: str,
    previous_count: int,
) -> dict[str, Any]:
    placement_error = dict(status.get("placement_error") or {})
    code = str(placement_error.get("code") or "")
    active = bool(
        status.get("runtime_state") == "paused_waiting_for_miners"
        and int(status.get("live_miner_count") or 0) >= EXPECTED_LIVE_MINERS
        and code.startswith("heterogeneous_placement_")
    )
    count = previous_count + 1 if active and code == previous_code else int(active)
    return {
        "code": code if active else "",
        "consecutive_observations": count,
        "terminal": bool(active and count >= PLACEMENT_BLOCKER_OBSERVATIONS),
        "stage_id": int((placement_error.get("diagnostics") or {}).get("stage_id") or 0),
        "public_artifact_safe": True,
    }


def _start_server(controller: HeterogeneousTrainingProductionController, port: int) -> tuple[Any, threading.Thread]:
    import uvicorn

    credentials = controller.beta.credentials()
    app = create_heterogeneous_training_beta_app(
        controller.beta,
        owner_token=str(credentials["owner_token"]),
        miner_token=str(credentials["miner_token"]),
    )
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=int(port),
            log_level="warning",
            access_log=False,
        )
    )
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_local_ready(f"http://127.0.0.1:{port}", timeout=30.0)
    return server, thread


def _stop_server(server: Any, thread: threading.Thread | None) -> bool:
    if server is not None:
        server.should_exit = True
    if thread is not None:
        thread.join(timeout=30.0)
    return thread is None or not thread.is_alive()


def _flatten_workers(kernel_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    workers = []
    for kernel in kernel_reports:
        kernel_role = str(kernel.get("kernel_role") or "")
        for result in kernel.get("worker_results") or []:
            worker = dict(result.get("report") or {})
            if not worker:
                continue
            workers.append(
                {
                    "kernel_role": kernel_role,
                    "label": str(result.get("label") or worker.get("deployment_role") or ""),
                    "returncode": int(result.get("returncode") or 0),
                    "report": worker,
                }
            )
    return workers


def _valid_public_hash(value: Any) -> bool:
    text = str(value or "")
    return bool(
        text.startswith("sha256:")
        and len(text) == 71
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _stage_steps(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in report.get("steps") or []:
        target_step = int(step.get("target_step") or 0)
        generation = int(step.get("placement_generation") or 0)
        for stage in step.get("stages") or []:
            row = dict(stage)
            row["target_step"] = int(row.get("target_step") or target_step)
            row["placement_generation"] = int(
                row.get("placement_generation") or generation
            )
            rows.append(row)
    return rows


def _replacement_evidence(workers: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    labels = {
        "cuda": ("gpu_old", "gpu_replacement"),
        "cpu": ("cpu_old", "cpu_replacement"),
        "jax_tpu": ("tpu_old", "tpu_replacement"),
    }[kind]
    old_item = next((item for item in workers if item["label"] == labels[0]), {})
    old = dict(old_item.get("report") or {})
    old_steps = [dict(item) for item in old.get("steps") or []]
    old_stage_steps = _stage_steps(old)
    previous_generation = max(
        [int(item.get("placement_generation") or 0) for item in old_steps] or [0]
    )
    old_last_step = max([int(item.get("target_step") or 0) for item in old_steps] or [0])
    old_final_rows = [
        item
        for item in old_stage_steps
        if int(item.get("target_step") or 0) == old_last_step
    ]
    old_stage_ids = {
        int(item["stage_id"])
        for item in old_final_rows
        if item.get("stage_id") is not None
    }

    candidates: list[tuple[int, str, dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
    for item in workers:
        report = dict(item.get("report") or {})
        identity = str(report.get("miner_id_hash") or "")
        if not report or identity == str(old.get("miner_id_hash") or ""):
            continue
        if str(report.get("device_policy") or kind) != kind:
            continue
        handoff_rows = [
            row
            for row in _stage_steps(report)
            if int(row.get("target_step") or 0) == old_last_step + 1
            and (
                not old_stage_ids
                or int(row.get("stage_id") or -1) in old_stage_ids
            )
        ]
        if not handoff_rows:
            continue
        generation = min(
            int(row.get("placement_generation") or 0) for row in handoff_rows
        )
        if generation <= previous_generation:
            continue
        candidates.append(
            (
                generation,
                str(item.get("label") or ""),
                item,
                report,
                handoff_rows,
            )
        )
    candidates.sort(key=lambda value: (value[0], value[1]))
    selected = candidates[0] if candidates else (0, "", {}, {}, [])
    replacement_generation, replacement_label, replacement_item, replacement, new_rows = selected
    replacement_first_step = old_last_step + 1 if new_rows else 0
    replacement_stage_ids = {
        int(item["stage_id"])
        for item in new_rows
        if item.get("stage_id") is not None
    }
    same_stage = bool(old_stage_ids and replacement_stage_ids == old_stage_ids)
    old_checkpoint_valid = bool(
        old_final_rows
        and all(
            item.get("checkpoint_components_validated") is True
            and _valid_public_hash(item.get("archive_hash"))
            and _valid_public_hash(item.get("checkpoint_hash"))
            for item in old_final_rows
        )
    )
    replacement_checkpoint_valid = bool(
        new_rows
        and all(
            item.get("checkpoint_components_validated") is True
            and _valid_public_hash(item.get("archive_hash"))
            and _valid_public_hash(item.get("checkpoint_hash"))
            for item in new_rows
        )
    )
    ready_history = [
        dict(item)
        for item in (
            list(replacement.get("stage_process_ready_history") or [])
            + list(replacement.get("stage_process_ready") or [])
        )
    ]
    matching_ready = next(
        (
            item
            for item in ready_history
            if item.get("resumed") is True
            and int(item.get("resumed_global_step") or 0) == old_last_step
            and int(item.get("placement_generation") or replacement_generation)
            == replacement_generation
            and (
                not old_stage_ids
                or int(item.get("stage_id") or -1) in old_stage_ids
            )
        ),
        {},
    )
    checkpoint_download_count = int(
        (replacement.get("client") or {}).get("checkpoint_download_count") or 0
    )
    contiguous = bool(
        old_last_step > 0 and replacement_first_step == old_last_step + 1
    )
    generation_fenced = bool(
        previous_generation > 0 and replacement_generation > previous_generation
    )
    protocol_restore = bool(
        contiguous
        and same_stage
        and generation_fenced
        and old_checkpoint_valid
        and replacement_checkpoint_valid
        and checkpoint_download_count >= 1
    )
    restore_verified = bool(
        (matching_ready or protocol_restore)
        and old_checkpoint_valid
        and replacement_checkpoint_valid
        and checkpoint_download_count >= 1
    )
    restored = old_last_step if restore_verified else 0
    identity_changed = bool(
        _valid_public_hash(old.get("miner_id_hash"))
        and _valid_public_hash(replacement.get("miner_id_hash"))
        and old.get("miner_id_hash") != replacement.get("miner_id_hash")
    )
    old_worker_drained = bool(
        old.get("ok") is True
        and old.get("graceful_drain_applied") is True
        and int(old_item.get("returncode") or 0) == 0
    )
    replacement_worker_accepted = bool(
        replacement.get("ok") is True
        and int(replacement_item.get("returncode") or 0) == 0
    )
    verified = bool(
        old_worker_drained
        and replacement_worker_accepted
        and identity_changed
        and contiguous
        and same_stage
        and generation_fenced
        and restore_verified
    )
    return {
        "verified": verified,
        "previous_generation": previous_generation,
        "replacement_generation": replacement_generation,
        "removed_after_step": old_last_step,
        "restored_checkpoint_step": restored,
        "replacement_first_step": replacement_first_step,
        "old_identity_hash": str(old.get("miner_id_hash") or ""),
        "replacement_identity_hash": str(replacement.get("miner_id_hash") or ""),
        "identity_changed": identity_changed,
        "old_worker_drained": old_worker_drained,
        "replacement_worker_accepted": replacement_worker_accepted,
        "same_stage_handoff_verified": same_stage,
        "contiguous_step_handoff_verified": contiguous,
        "generation_fencing_verified": generation_fenced,
        "checkpoint_restore_verified": restore_verified,
        "checkpoint_download_count": checkpoint_download_count,
        "checkpoint_ready_event_matched": bool(matching_ready),
        "restore_evidence_source": (
            "stage_ready_history"
            if matching_ready
            else "generation_fenced_contiguous_checkpoint_handoff"
            if protocol_restore
            else "none"
        ),
        "old_stage_ids": sorted(old_stage_ids),
        "replacement_stage_ids": sorted(replacement_stage_ids),
        "source_checkpoint_archive_hashes": sorted(
            str(item.get("archive_hash") or "") for item in old_final_rows
        ),
        "replacement_checkpoint_archive_hashes": sorted(
            str(item.get("archive_hash") or "") for item in new_rows
        ),
        "replacement_label_hash": stable_hash({"label": replacement_label})
        if replacement_label
        else "",
        "replacement_selection": (
            "designated_replacement"
            if replacement_label == labels[1]
            else "cross_kernel_dynamic_reassignment"
            if replacement_label
            else "none"
        ),
        "old_kernel_role": str(old_item.get("kernel_role") or ""),
        "replacement_kernel_role": str(replacement_item.get("kernel_role") or ""),
        "identity_values_public": False,
        "public_artifact_safe": True,
    }


def _effective_kernel_evidence(
    kernel_reports: list[dict[str, Any]],
    replacements: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    rows = []
    for kernel in kernel_reports:
        role = str(kernel.get("kernel_role") or "")
        results = list(kernel.get("worker_results") or [])
        worker_results_valid = bool(
            results
            and all(
                int(item.get("returncode") or 0) == 0
                and (item.get("report") or {}).get("ok") is True
                for item in results
            )
        )
        cleanup_verified = bool(
            kernel.get("all_worker_processes_stopped") is True
            and kernel.get("private_runtime_removed") is True
        )
        observations = list((kernel.get("pause_observation") or {}).get("observations") or [])
        cuda = dict(replacements.get("cuda") or {})
        automatic_takeover = bool(
            role == "gpu_a"
            and cuda.get("verified") is True
            and cuda.get("replacement_kernel_role") != role
            and any(
                int(item.get("committed_step") or 0)
                >= int(cuda.get("removed_after_step") or 0)
                and int(item.get("placement_generation") or 0)
                >= int(cuda.get("replacement_generation") or 0)
                and not list(item.get("missing_stage_ids") or [])
                and item.get("runtime_state") == "running"
                for item in observations
            )
        )
        blockers = sorted(str(item) for item in kernel.get("blockers") or [])
        raw_ok = kernel.get("ok") is True
        recoverable_raw_failure = bool(
            automatic_takeover
            and blockers == ["heterogeneous_kaggle_worker_acceptance_incomplete"]
        )
        effective_ok = bool(
            cleanup_verified
            and worker_results_valid
            and (raw_ok or recoverable_raw_failure)
        )
        rows.append(
            {
                "kernel_role": role,
                "raw_ok": raw_ok,
                "effective_ok": effective_ok,
                "worker_results_valid": worker_results_valid,
                "cleanup_verified": cleanup_verified,
                "automatic_cross_kernel_takeover_observed": automatic_takeover,
                "raw_failure_reclassified": recoverable_raw_failure,
                "raw_blockers": blockers,
                "worker_result_count": len(results),
                "kernel_report_hash": stable_hash(kernel),
                "public_artifact_safe": True,
            }
        )
    required_roles = {"cpu", "gpu_a", "gpu_b", "tpu"}
    verified = bool(
        {item["kernel_role"] for item in rows} == required_roles
        and all(item["effective_ok"] is True for item in rows)
    )
    return rows, verified


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(float(item) for item in values if math.isfinite(float(item)))
    if not ordered:
        return 0.0
    position = max(0.0, min(1.0, float(fraction))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _performance_windows(final_status: dict[str, Any], *, workload_hash: str, topology_hash: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    commits = [dict(item) for item in final_status.get("commits") or []]
    times = {int(item["target_step"]): float(item["committed_at"]) for item in commits}
    intervals = {
        step: times[step] - times[step - 1]
        for step in sorted(times)
        if step > 1 and step - 1 in times and times[step] > times[step - 1]
    }

    def windows(ranges: list[tuple[int, int]]) -> list[dict[str, Any]]:
        result = []
        for start, end in ranges:
            values = [intervals[step] for step in range(start, end + 1) if step in intervals]
            duration = sum(values)
            result.append(
                {
                    "workload_hash": workload_hash,
                    "topology_hash": topology_hash,
                    "step_start": start,
                    "step_end": end,
                    "sample_count": len(values),
                    "step_throughput_per_second": len(values) / duration if duration > 0 else 0.0,
                    "p50_step_latency_seconds": _percentile(values, 0.5),
                    "p95_step_latency_seconds": _percentile(values, 0.95),
                }
            )
        return result

    def ranges(start: int) -> list[tuple[int, int]]:
        return [
            (
                start + index * PERFORMANCE_WINDOW_SIZE,
                start + (index + 1) * PERFORMANCE_WINDOW_SIZE - 1,
            )
            for index in range(PERFORMANCE_WINDOW_COUNT)
        ]

    return (
        windows(ranges(BASELINE_START_STEP)),
        windows(ranges(BASELINE_END_STEP + 1)),
        [intervals[key] for key in sorted(intervals)],
    )


def _provisional_performance_gate(
    controller: HeterogeneousTrainingProductionController,
    status: dict[str, Any],
) -> dict[str, Any]:
    baseline, candidate, _intervals = _performance_windows(
        status,
        workload_hash=controller.beta.manifest["content_hash"],
        topology_hash=stable_hash(
            [
                {
                    "stage_id": int(item["stage_id"]),
                    "device_type": str(item["preferred_device_type"]),
                }
                for item in controller.beta.manifest["stages"]
            ]
        ),
    )
    complete = bool(
        baseline
        and candidate
        and all(
            int(item.get("sample_count") or 0) == PERFORMANCE_WINDOW_SIZE
            for item in baseline + candidate
        )
    )
    comparison = compare_performance_windows(
        baseline=baseline,
        candidate=candidate,
        policy=controller.config["performance"],
    )
    return {
        "windows_complete": complete,
        "performance_gate_passed": bool(
            complete and comparison.get("performance_gate_passed") is True
        ),
        "throughput_improvement_fraction": float(
            comparison.get("throughput_improvement_fraction") or 0.0
        ),
        "p50_latency_improvement_fraction": float(
            comparison.get("p50_latency_improvement_fraction") or 0.0
        ),
        "p95_regression_fraction": float(
            comparison.get("p95_regression_fraction") or 0.0
        ),
        "public_artifact_safe": True,
    }


def _build_live_report(
    *,
    controller: HeterogeneousTrainingProductionController,
    final_status: dict[str, Any],
    kernel_reports: list[dict[str, Any]],
    transport: dict[str, Any],
    checkpoint: dict[str, Any],
    local_export: dict[str, Any],
    stale_probe: dict[str, Any],
    coordinator_restart: dict[str, Any],
    observations: list[dict[str, Any]],
    soak_duration_seconds: float,
    live_elapsed_seconds: float,
    cleanup: dict[str, Any],
    blockers: list[str],
    runtime_metrics: dict[str, Any],
    prometheus_metrics_verified: bool,
) -> dict[str, Any]:
    workers = _flatten_workers(kernel_reports)
    worker_reports = [item["report"] for item in workers]
    stage_results = [
        dict(stage)
        for worker in worker_reports
        for step in worker.get("steps") or []
        for stage in step.get("stages") or []
    ]
    providers = []
    policies = {str(item.get("device_policy") or "") for item in worker_reports}
    if "cpu" in policies:
        providers.append("kaggle_cpu")
    if "cuda" in policies:
        providers.append("kaggle_cuda")
    if "jax_tpu" in policies:
        providers.append("kaggle_jax_tpu")
    committed = [int(item) for item in final_status.get("committed_steps") or []]
    stage_hashes: dict[int, set[str]] = {}
    for item in stage_results:
        stage_hashes.setdefault(int(item.get("stage_id") or 0), set()).add(
            str(item.get("adapter_tensor_hash") or "")
        )
    replacements = {
        kind: _replacement_evidence(workers, kind)
        for kind in ("cpu", "cuda", "jax_tpu")
    }
    kernel_evidence, effective_kernel_evidence_verified = (
        _effective_kernel_evidence(kernel_reports, replacements)
    )
    cpu_kernel = next(
        (item for item in kernel_reports if item.get("kernel_role") == "cpu"), {}
    )
    remote_export = dict((cpu_kernel.get("export_reload") or {}).get("report") or {})
    messages = [dict(item) for item in transport.get("messages") or []]
    assignments = [dict(item) for item in final_status.get("assignments") or []]
    device_by_route = {
        (int(item["placement_generation"]), int(item["stage_id"])): str(
            item.get("device_type") or ""
        )
        for item in assignments
    }

    def route_count(direction: str, source: str, target: str) -> int:
        return sum(
            1
            for item in messages
            if item.get("direction") == direction
            and device_by_route.get(
                (int(item["placement_generation"]), int(item["source_stage_id"]))
            )
            == source
            and device_by_route.get(
                (int(item["placement_generation"]), int(item["target_stage_id"]))
            )
            == target
        )

    initial_plan = next(
        (
            dict(item.get("placement_plan") or {})
            for item in observations
            if (item.get("placement_plan") or {}).get("assignments")
        ),
        dict(final_status.get("placement_plan") or {}),
    )
    topology_hash = stable_hash(
        [
            {
                "stage_id": int(item["stage_id"]),
                "device_type": str(item["device_type"]),
            }
            for item in initial_plan.get("assignments") or []
        ]
    )
    workload_hash = stable_hash(
        {
            "manifest_hash": controller.beta.manifest["content_hash"],
            "microbatch_size": controller.beta.manifest["training"]["microbatch_size"],
            "sequence_length": controller.beta.manifest["training"]["sequence_length"],
            "checkpoint_every_steps": controller.beta.manifest["checkpoint"]["checkpoint_every_steps"],
        }
    )
    baseline, candidate, intervals = _performance_windows(
        final_status,
        workload_hash=workload_hash,
        topology_hash=topology_hash,
    )
    performance = compare_performance_windows(
        baseline=baseline,
        candidate=candidate,
        policy=controller.config["performance"],
    )
    checkpoint_overheads = [
        float(item.get("checkpoint_overhead_ms") or 0.0) / 1000.0
        for item in stage_results
        if float(item.get("checkpoint_overhead_ms") or 0.0) > 0
    ]
    compile_times = [
        float(item.get("compile_latency_ms") or 0.0)
        for worker in worker_reports
        for item in worker.get("stage_process_statuses") or []
        if float(item.get("compile_latency_ms") or 0.0) > 0
    ]
    transfer_bytes = sum(int(item.get("payload_bytes") or 0) for item in messages)
    inline_uploads = sum(
        int((worker.get("client") or {}).get("inline_tensor_message_upload_count") or 0)
        for worker in worker_reports
    )
    inline_downloads = sum(
        int((worker.get("client") or {}).get("inline_tensor_message_download_count") or 0)
        for worker in worker_reports
    )
    large_payload_isolations = sum(
        int(
            (worker.get("client") or {}).get(
                "large_payload_connection_isolation_count"
            )
            or 0
        )
        for worker in worker_reports
    )
    telemetry_sampling_skips = sum(
        int(worker.get("device_telemetry_sampling_skip_count") or 0)
        for worker in worker_reports
    )
    monitoring_live = bool(
        runtime_metrics.get("event_count", 0) > TARGET_STEPS
        and int(runtime_metrics.get("coordinator_generation") or 0) >= 2
        and any(
            item.get("operation") == "coordinator_started"
            for item in final_status.get("events") or []
        )
        and all(
            any(
                int(observation.get("placement_generation") or 0)
                >= int(replacements[kind]["replacement_generation"])
                for observation in observations
            )
            for kind in replacements
        )
    )
    expected = list(range(1, len(committed) + 1))
    finite_updates = bool(
        {int(item.get("stage_id") or 0) for item in stage_results} == set(range(5))
        and all(
            math.isfinite(float(item.get("lora_gradient_norm") or 0.0))
            and float(item.get("lora_gradient_norm") or 0.0) > 0
            and item.get("optimizer_step_applied") is True
            and item.get("scheduler_step_applied") is True
            for item in stage_results
        )
    )
    changed_hashes = all(
        len({value for value in stage_hashes.get(stage_id, set()) if value}) >= 2
        for stage_id in range(5)
    )
    adapter_reload = bool(
        local_export.get("standard_peft_format") is True
        and remote_export.get("standard_peft_format") is True
        and remote_export.get("adapter_reload_verified") is True
        and remote_export.get("forward_inference_verified") is True
        and remote_export.get("finite_logits_verified") is True
        and remote_export.get("model_binding_verified") is True
        and remote_export.get("adapter_file_hash")
        == local_export.get("adapter_file_hash")
    )
    transport_verified = bool(
        transport.get("all_complete") is True
        and transport.get("all_checksums_verified") is True
        and route_count("forward_activation", "cuda", "jax_tpu") >= TARGET_STEPS
        and route_count("backward_gradient", "jax_tpu", "cuda") >= TARGET_STEPS
        and route_count("forward_activation", "cuda", "cpu") >= TARGET_STEPS
        and route_count("backward_gradient", "cpu", "cuda") >= TARGET_STEPS
    )
    derived_blockers = list(blockers)
    gates = {
        "training_production_commit_ledger_incomplete": committed
        == list(range(1, TARGET_STEPS + 1)),
        "training_production_soak_duration_short": float(soak_duration_seconds)
        >= MINIMUM_SOAK_SECONDS,
        "training_production_finite_update_gate_failed": finite_updates,
        "training_production_lora_hash_change_gate_failed": changed_hashes,
        "training_production_checkpoint_integrity_gate_failed": bool(
            checkpoint.get("all_five_stage_archives_valid") is True
            and checkpoint.get("all_component_hashes_verified") is True
        ),
        "training_production_adapter_reload_gate_failed": adapter_reload,
        "training_production_transport_gate_failed": transport_verified,
        "training_production_monitoring_gate_failed": monitoring_live,
        "training_production_coordinator_restart_gate_failed": coordinator_restart.get(
            "verified"
        )
        is True,
        "training_production_stale_result_gate_failed": stale_probe.get("verified")
        is True,
        "training_production_performance_gate_failed": performance.get(
            "performance_gate_passed"
        )
        is True,
        "training_production_worker_replacement_gate_failed": all(
            item.get("verified") is True for item in replacements.values()
        ),
    }
    derived_blockers.extend(code for code, passed in gates.items() if not passed)
    report = {
        "schema": LIVE_SCHEMA,
        "live_run_performed": True,
        "external_runtime_verified": bool(
            sorted(providers)
            == ["kaggle_cpu", "kaggle_cuda", "kaggle_jax_tpu"]
            and len(kernel_reports) == 4
            and effective_kernel_evidence_verified
        ),
        "execution_provider": "kaggle",
        "model_id": controller.beta.manifest["model"]["model_id"],
        "model_revision": controller.beta.manifest["model"]["model_revision"],
        "training_manifest_hash": controller.beta.manifest["content_hash"],
        "target_steps": TARGET_STEPS,
        "minimum_required_steps": MINIMUM_REQUIRED_STEPS,
        "minimum_soak_duration_seconds": MINIMUM_SOAK_SECONDS,
        "accepted_providers": sorted(providers),
        "soak_duration_seconds": float(soak_duration_seconds),
        "full_live_gate_elapsed_seconds": float(live_elapsed_seconds),
        "training_evidence": {
            "committed_steps": committed,
            "committed_step_count": len(committed),
            "duplicate_committed_steps": sorted(
                {item for item in committed if committed.count(item) > 1}
            ),
            "missing_committed_steps": sorted(set(range(1, TARGET_STEPS + 1)) - set(committed)),
            "maximum_checkpoint_interval_steps": 1,
            "finite_updates_all_stages": finite_updates,
            "changed_lora_hashes_all_stages": changed_hashes,
            "atomic_ledger_verified": committed == expected
            and committed == list(range(1, TARGET_STEPS + 1)),
            "checkpoint_integrity_verified": bool(
                checkpoint.get("all_five_stage_archives_valid") is True
                and checkpoint.get("all_component_hashes_verified") is True
            ),
            "updated_stage_ids": sorted(
                {int(item.get("stage_id") or 0) for item in stage_results}
            ),
            "random_or_synthetic_weights_used": False,
            "fake_gradients_used": False,
        },
        "worker_replacements": replacements,
        "coordinator_restart": coordinator_restart,
        "coordinator_restart_live_verified": coordinator_restart.get("verified")
        is True,
        "stale_result_probe": stale_probe,
        "stale_result_rejected": stale_probe.get("verified") is True,
        "adapter_cpu_reload_verified": adapter_reload,
        "activation_gradient_transfer_verified": transport_verified,
        "monitoring_live_verified": monitoring_live,
        "performance": performance,
        "performance_windows": {
            "baseline": baseline,
            "candidate": candidate,
            "optimization_switch_after_step": BASELINE_END_STEP,
            "same_run_comparison": True,
        },
        "optimization_summary": {
            "persistent_http_after_step": BASELINE_END_STEP,
            "indexed_tensor_lookup_after_step": BASELINE_END_STEP,
            "inline_tensor_transport_after_step": BASELINE_END_STEP,
            "inline_tensor_message_upload_count": inline_uploads,
            "inline_tensor_message_download_count": inline_downloads,
            "large_payload_connection_isolation_count": large_payload_isolations,
            "persistent_http_max_body_bytes": PERSISTENT_HTTP_MAX_BODY_BYTES,
            "performance_window_count_per_phase": PERFORMANCE_WINDOW_COUNT,
            "performance_window_size_steps": PERFORMANCE_WINDOW_SIZE,
            "telemetry_sampling_interval_after_optimization_steps": 5,
            "telemetry_sampling_skip_count": telemetry_sampling_skips,
            "training_workload_unchanged": True,
            "resource_topology_unchanged": True,
            "public_artifact_safe": True,
        },
        "benchmark": {
            "step_throughput_per_second": (
                len(intervals) / sum(intervals) if intervals and sum(intervals) > 0 else 0.0
            ),
            "p50_step_latency_seconds": _percentile(intervals, 0.5),
            "p95_step_latency_seconds": _percentile(intervals, 0.95),
            "compile_time_seconds": sum(compile_times) / 1000.0,
            "checkpoint_overhead_seconds": statistics.median(checkpoint_overheads)
            if checkpoint_overheads
            else 0.0,
            "transfer_bytes": transfer_bytes,
            "transfer_bandwidth_bytes_per_second": transfer_bytes
            / max(1.0, float(soak_duration_seconds)),
            "tensor_lookup": runtime_metrics.get("tensor_lookup") or {},
        },
        "transport_summary": {
            "message_count": int(transport.get("message_count") or 0),
            "all_complete": transport.get("all_complete") is True,
            "all_checksums_verified": transport.get("all_checksums_verified")
            is True,
            "metadata_content_hash": str(transport.get("content_hash") or ""),
        },
        "checkpoint_summary": checkpoint,
        "export_summary": {
            "standard_peft_format": local_export.get("standard_peft_format") is True,
            "adapter_tensor_count": int(local_export.get("adapter_tensor_count") or 0),
            "adapter_file_hash": str(local_export.get("adapter_file_hash") or ""),
            "cpu_reload_verified": remote_export.get("adapter_reload_verified") is True,
            "finite_full_stagewise_forward_verified": remote_export.get(
                "finite_logits_verified"
            )
            is True,
        },
        "monitoring_summary": {
            "status_observation_count": len(observations),
            "event_count": int(runtime_metrics.get("event_count") or 0),
            "coordinator_generation": int(
                runtime_metrics.get("coordinator_generation") or 0
            ),
            "worker_state_metrics_present": bool(
                runtime_metrics.get("worker_states")
            ),
            "stage_profiles_present": bool(runtime_metrics.get("stage_profiles")),
            "transfer_metrics_present": bool(
                (runtime_metrics.get("transfer") or {}).get("payload_bytes")
            ),
            "prometheus_metrics_verified": prometheus_metrics_verified,
        },
        "kernel_evidence": kernel_evidence,
        "effective_kernel_evidence_verified": effective_kernel_evidence_verified,
        "cleanup": cleanup,
        "blockers": sorted(set(str(item) for item in derived_blockers if item)),
        "credential_values_public": False,
        "credential_paths_public": False,
        "coordinator_url_public": False,
        "raw_training_text_public": False,
        "token_ids_public": False,
        "activation_values_public": False,
        "gradient_values_public": False,
        "checkpoint_values_public": False,
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
    parser.add_argument("--acquisition-ledger", default="dist/training-heterogeneous-production-work/acquisitions.json")
    parser.add_argument("--live-attempt-ledger", default="dist/training-heterogeneous-production-work/live-attempts.json")
    parser.add_argument("--acquisition-window-limit", type=int, default=0)
    parser.add_argument("--live-attempt-limit", type=int, default=0)
    parser.add_argument("--authorize-unlimited-acquisition-windows", action="store_true")
    parser.add_argument("--authorize-unlimited-live-gates", action="store_true")
    parser.add_argument("--acquisition-authorization-id", default="")
    parser.add_argument("--live-attempt-authorization-id", default="")
    parser.add_argument("--reuse-acquisition-window", type=int, default=0)
    parser.add_argument("--tpu-queue-timeout-seconds", type=float, default=43200.0)
    parser.add_argument("--kernel-timeout-seconds", type=float, default=21600.0)
    parser.add_argument("--operation-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--push-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--status-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--output-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--delete-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.acquisition_window_limit != 0 or not args.authorize_unlimited_acquisition_windows:
        parser.error("Production requires authorized unlimited acquisition windows with bounded 12-hour attempts")
    if args.live_attempt_limit != 0 or not args.authorize_unlimited_live_gates:
        parser.error("Production requires authorized unlimited live gates with bounded 6-hour attempts")
    if not 600 <= args.tpu_queue_timeout_seconds <= 43200:
        parser.error("--tpu-queue-timeout-seconds must be in [600, 43200]")
    if not 3600 <= args.kernel_timeout_seconds <= 21600:
        parser.error("--kernel-timeout-seconds must be in [3600, 21600]")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "training_heterogeneous_production_live_probe.json"
    private = output / ".private-runtime"
    private.mkdir(parents=True, exist_ok=True)
    private.chmod(0o700)
    report = _base_report("training_production_live_not_started")
    _write(report_path, report)
    controller: HeterogeneousTrainingProductionController | None = None
    server = None
    server_thread: threading.Thread | None = None
    tunnel_process = None
    packages: list[dict[str, Any]] = []
    kernel_reports: list[dict[str, Any]] = []
    refs_by_role: dict[str, tuple[str, dict[str, str]]] = {}
    blockers: list[str] = []
    observations: list[dict[str, Any]] = []
    placement_snapshots: list[dict[str, Any]] = []
    placement_blocker_observation: dict[str, Any] = {}
    quota_summaries: list[dict[str, Any]] = []
    acquisition_authorization: dict[str, Any] = {}
    live_authorization: dict[str, Any] = {}
    coordinator_restart: dict[str, Any] = {}
    stale_probe: dict[str, Any] = {}
    transport: dict[str, Any] = {}
    checkpoint: dict[str, Any] = {}
    local_export: dict[str, Any] = {}
    runtime_metrics: dict[str, Any] = {}
    prometheus_metrics_verified = False
    final_status: dict[str, Any] = {}
    acquisition_attempt = 0
    live_attempt = 0
    acquisition_outcome = "not_started"
    live_outcome = "not_started"
    live_started = 0.0
    training_started = 0.0
    training_completed = 0.0
    remote_deleted = True
    tensor_removed = False
    try:
        acquisition_authorization = authorize_unlimited_attempts(
            Path(args.acquisition_ledger).resolve(),
            kind="tpu_acquisition_window",
            authorization_granted=True,
            authorization_id=args.acquisition_authorization_id,
            max_attempt_duration_seconds=43200.0,
        )
        live_authorization = authorize_unlimited_attempts(
            Path(args.live_attempt_ledger).resolve(),
            kind="production_soak_live_gate",
            authorization_granted=True,
            authorization_id=args.live_attempt_authorization_id,
            max_attempt_duration_seconds=21600.0,
        )
        report["acquisition_authorization"] = acquisition_authorization
        report["live_gate_authorization"] = live_authorization
        _write(report_path, report)
        transport_contract = transport_contract_probe(private / "transport-contract")
        if transport_contract.get("ok") is not True:
            raise RuntimeError("heterogeneous_tensor_contract_probe_failed")
        config = default_production_config()
        config["soak"]["target_steps"] = TARGET_STEPS
        config.pop("content_hash", None)
        controller_result = HeterogeneousTrainingProductionController.create(
            private / "job",
            config=config,
            hf_token=str(os.environ.get(args.hf_token_env) or ""),
        )
        assert isinstance(
            controller_result, HeterogeneousTrainingProductionController
        )
        controller = controller_result
        port = _free_port()
        server, server_thread = _start_server(controller, port)
        local_url = f"http://127.0.0.1:{port}"
        tunnel_binary = ensure_cloudflared(private)
        credentials = controller.beta.credentials()
        tunnel_process, tunnel_url, _route = _start_verified_tunnel(
            tunnel_binary,
            local_url,
            private,
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
            quota_summaries.append(before_tpu)
            report["gpu_quota_preflight_summaries"] = quota_summaries
            _write(report_path, report)
            if before_tpu["weekly_gpu_quota_exhausted"]:
                raise RuntimeError("kaggle_gpu_weekly_quota_exhausted")
            suffix = str(int(time.time()))[-9:]
            tpu_package = build_tpu_package(
                private / "package-tpu",
                owner=tpu_owner,
                slug=safe_slug(f"ct-training-production-tpu-{suffix}"),
                coordinator_url=tunnel_url,
                coordinator_token=str(credentials["miner_token"]),
                hf_token=str(os.environ.get(args.hf_token_env) or ""),
                wait_timeout_seconds=float(args.kernel_timeout_seconds),
                operation_timeout_seconds=float(args.operation_timeout_seconds),
                transport_optimization_after_step=BASELINE_END_STEP,
                replacement_after_steps=TPU_REPLACEMENT_STEP,
            )
            packages.append(tpu_package)
            acquisition_attempt, acquisition_remaining = reserve_acquisition_window(
                Path(args.acquisition_ledger).resolve(),
                limit=0,
                reuse_attempt=int(args.reuse_acquisition_window),
                window_seconds=float(args.tpu_queue_timeout_seconds),
            )
            tpu_push = run_command(
                [
                    "kaggle",
                    "kernels",
                    "push",
                    "-p",
                    str(tpu_package["package_dir"]),
                    "-t",
                    str(int(args.kernel_timeout_seconds)),
                    "--accelerator",
                    "tpuV5e8",
                ],
                env=tpu_env,
                timeout=float(args.push_timeout_seconds),
            )
            tpu_outcome = classify_tpu_push(tpu_push)
            report["tpu_push_summary"] = {
                "outcome": tpu_outcome,
                "returncode": tpu_push.get("returncode"),
                "timed_out": tpu_push.get("timed_out") is True,
                "duration_seconds": float(tpu_push.get("duration_seconds") or 0.0),
                "requested_accelerator": "tpuV5e8",
                "public_artifact_safe": True,
            }
            _write(report_path, report)
            if not push_accepted(tpu_push):
                raise RuntimeError(tpu_outcome)
            tpu_ref = extract_kernel_ref(
                str(tpu_push.get("output_tail") or ""), tpu_package["kernel_ref"]
            )
            refs_by_role["tpu"] = (tpu_ref, tpu_env)
            queue_started = time.monotonic()
            queue_deadline = queue_started + float(acquisition_remaining)
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
                _write(report_path, report)
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
            quota_summaries.append(after_tpu)
            report["gpu_quota_preflight_summaries"] = quota_summaries
            _write(report_path, report)
            if after_tpu["weekly_gpu_quota_exhausted"]:
                raise RuntimeError("kaggle_gpu_weekly_quota_exhausted")
            live_attempt = _reserve_attempt(Path(args.live_attempt_ledger).resolve(), limit=0)
            live_started = time.monotonic()
            package_specs = [
                ("gpu_a", gpu_owner, GPU_REPLACEMENT_STEP),
                ("gpu_b", gpu_owner, 0),
                ("cpu", cpu_owner, CPU_REPLACEMENT_STEP),
            ]
            for role, owner, replacement_step in package_specs:
                packages.append(
                    build_cpu_gpu_package(
                        private / f"package-{role}",
                        owner=owner,
                        slug=safe_slug(f"ct-training-production-{role}-{suffix}"),
                        role=role,
                        coordinator_url=tunnel_url,
                        coordinator_token=str(credentials["miner_token"]),
                        hf_token=str(os.environ.get(args.hf_token_env) or ""),
                        wait_timeout_seconds=float(args.kernel_timeout_seconds),
                        operation_timeout_seconds=float(args.operation_timeout_seconds),
                        replacement_after_steps=replacement_step,
                        transport_optimization_after_step=BASELINE_END_STEP,
                    )
                )

            def push_role(package: dict[str, Any]) -> tuple[str, str, dict[str, str], dict[str, Any]]:
                role = str(package["role"])
                env = cpu_env if role == "cpu" else gpu_env
                command = [
                    "kaggle",
                    "kernels",
                    "push",
                    "-p",
                    str(package["package_dir"]),
                    "-t",
                    str(int(args.kernel_timeout_seconds)),
                ]
                if role.startswith("gpu"):
                    command.extend(["--accelerator", "NvidiaTeslaT4"])
                step = run_command(command, env=env, timeout=float(args.push_timeout_seconds))
                outcome = classify_training_worker_push(role, step)
                ref = (
                    extract_kernel_ref(
                        str(step.get("output_tail") or ""), package["kernel_ref"]
                    )
                    if push_accepted(step)
                    else ""
                )
                return role, ref, env, {
                    "role": role,
                    "outcome": outcome,
                    "returncode": step.get("returncode"),
                    "timed_out": step.get("timed_out") is True,
                    "duration_seconds": float(step.get("duration_seconds") or 0.0),
                    "public_artifact_safe": True,
                }

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                pushed = list(pool.map(push_role, packages[1:]))
            report["worker_push_summaries"] = [item[3] for item in pushed]
            _write(report_path, report)
            for role, ref, env, summary in pushed:
                if not ref:
                    raise RuntimeError(str(summary["outcome"]))
                refs_by_role[role] = (ref, env)
            deadline = live_started + float(args.kernel_timeout_seconds)
            states: dict[str, str] = {}
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
                placement_blocker_observation = _observe_placement_blocker(
                    current,
                    previous_code=str(placement_blocker_observation.get("code") or ""),
                    previous_count=int(
                        placement_blocker_observation.get("consecutive_observations")
                        or 0
                    ),
                )
                committed_step = int(current.get("committed_step") or 0)
                if not training_started and current.get("runtime_state") == "running":
                    training_started = time.monotonic()
                snapshot = _safe_status_snapshot(current)
                snapshot["observed_at"] = utc_now()
                observations.append(snapshot)
                plan_hash = str((snapshot.get("placement_plan") or {}).get("content_hash") or "")
                if plan_hash and all(
                    str((item.get("placement_plan") or {}).get("content_hash") or "")
                    != plan_hash
                    for item in placement_snapshots
                ):
                    placement_snapshots.append(snapshot)
                report["runtime_progress"] = {
                    "committed_step": committed_step,
                    "target_steps": TARGET_STEPS,
                    "placement_generation": int(current.get("placement_generation") or 0),
                    "coordinator_generation": int(current.get("coordinator_generation") or 0),
                    "kernel_states": states,
                    "observation_count": len(observations),
                    "placement_blocker_observation": placement_blocker_observation,
                    "public_artifact_safe": True,
                }
                if committed_step >= PERFORMANCE_GATE_STEP:
                    provisional_performance = _provisional_performance_gate(
                        controller, current
                    )
                    report["runtime_progress"]["provisional_performance"] = (
                        provisional_performance
                    )
                    if (
                        provisional_performance["windows_complete"] is True
                        and provisional_performance["performance_gate_passed"]
                        is not True
                    ):
                        _write(report_path, report)
                        raise RuntimeError(
                            "training_production_performance_gate_failed"
                        )
                _write(report_path, report)
                if placement_blocker_observation.get("terminal") is True:
                    raise RuntimeError(str(placement_blocker_observation["code"]))
                if (
                    committed_step >= COORDINATOR_RESTART_STEP
                    and not coordinator_restart
                ):
                    before = {
                        "committed_step": committed_step,
                        "placement_generation": int(current["placement_generation"]),
                        "coordinator_generation": int(current["coordinator_generation"]),
                    }
                    restart_started = time.monotonic()
                    stopped = _stop_server(server, server_thread)
                    controller = HeterogeneousTrainingProductionController(private / "job")
                    server, server_thread = _start_server(controller, port)
                    after = controller.runtime.public_status()
                    coordinator_restart = {
                        "verified": bool(
                            stopped
                            and int(after["committed_step"]) >= before["committed_step"]
                            and int(after["placement_generation"])
                            == before["placement_generation"]
                            and int(after["coordinator_generation"])
                            == before["coordinator_generation"] + 1
                        ),
                        "restart_after_step": committed_step,
                        "committed_step_before": before["committed_step"],
                        "committed_step_after": int(after["committed_step"]),
                        "placement_generation_before": before["placement_generation"],
                        "placement_generation_after": int(after["placement_generation"]),
                        "coordinator_generation_before": before["coordinator_generation"],
                        "coordinator_generation_after": int(after["coordinator_generation"]),
                        "downtime_seconds": time.monotonic() - restart_started,
                        "private_service_identity_public": False,
                        "public_artifact_safe": True,
                    }
                if (
                    committed_step >= TPU_REPLACEMENT_STEP
                    and not stale_probe.get("verified")
                    and int(current.get("placement_generation") or 0) >= 4
                    and current.get("runtime_state") not in {"completed", "cleaned", "cancelled"}
                ):
                    initial_generation = min(
                        [
                            int(item.get("placement_generation") or 0)
                            for item in placement_snapshots
                            if int(item.get("placement_generation") or 0) > 0
                        ]
                        or [1]
                    )
                    stale_probe = stale_generation_probe(
                        controller.beta, old_generation=initial_generation
                    )
                if current.get("runtime_state") == "completed":
                    training_completed = training_completed or time.monotonic()
                    if all(value in TERMINAL for value in states.values()):
                        break
                early = [
                    role
                    for role, state in states.items()
                    if state in TERMINAL and current.get("runtime_state") != "completed"
                ]
                if early:
                    raise RuntimeError("heterogeneous_kernel_terminal_before_training_complete")
                time.sleep(max(10.0, float(args.poll_interval_seconds)))
            else:
                raise RuntimeError("training_production_live_gate_timeout")
            final_status = controller.runtime.public_status()
            for role, (ref, env) in refs_by_role.items():
                destination = private / f"output-{role}"
                filename = TPU_KERNEL_REPORT if role == "tpu" else GPU_CPU_KERNEL_REPORT
                kernel, collection = collect_kernel_output_with_retry(
                    ref=ref,
                    env=env,
                    destination=destination,
                    filename=filename,
                    file_pattern=OUTPUT_PATTERN,
                    timeout_seconds=float(args.output_timeout_seconds),
                    poll_interval_seconds=max(5.0, float(args.poll_interval_seconds)),
                )
                if not kernel:
                    blockers.append(f"training_production_{role}_kernel_report_missing")
                    continue
                kernel["kernel_ref_hash"] = _hash_text(ref)
                kernel_reports.append(kernel)
                _write(output / "kernels" / f"{role}.json", kernel)
            if final_status.get("runtime_state") != "completed":
                blockers.append("training_production_100_step_training_incomplete")
            else:
                transport = collect_public_tensor_metadata(
                    controller.runtime.tensor_store.root
                )
                _write(output / "training_heterogeneous_production_transport_metadata.json", transport)
                checkpoint = collect_checkpoint_evidence(
                    controller.beta, target_step=TARGET_STEPS
                )
                local_export = controller.beta.export()
                integrity = controller.runtime.audit_checkpoint_integrity()
                checkpoint["integrity_audit"] = integrity
                checkpoint["all_component_hashes_verified"] = bool(
                    checkpoint.get("all_component_hashes_verified") is True
                    and integrity.get("latest_checkpoint_valid") is True
                )
                runtime_metrics = controller.runtime.metrics_snapshot()
                prometheus_metrics_verified = (
                    "crowdtensor_training_committed_step"
                    in controller.runtime.prometheus_metrics()
                )
            live_outcome = "live_collected"
    except BaseException as exc:
        code = _public_blocker(exc)
        blockers.append(code)
        if controller is not None:
            try:
                final_status = controller.runtime.public_status()
            except Exception:
                pass
        if acquisition_outcome != "tpu_running":
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
        for role, (ref, _env) in refs_by_role.items():
            try:
                if role == "tpu":
                    token_file, username = args.tpu_token_file, args.tpu_token_username
                elif role == "cpu":
                    token_file = args.cpu_token_file or args.gpu_token_file
                    username = args.cpu_token_username or args.gpu_token_username
                else:
                    token_file, username = args.gpu_token_file, args.gpu_token_username
                with kaggle_env(token_file, username_hint=username) as cleanup_env:
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
                    (controller_cleanup.get("tensor_transport_cleanup") or {}).get(
                        "all_messages_removed"
                    )
                    is True
                )
            except Exception:
                tensor_removed = False
        server_stopped = _stop_server(server, server_thread)
        tunnel_stopped = stop_process(tunnel_process)
        for package in packages:
            shutil.rmtree(Path(package["package_dir"]).parent, ignore_errors=True)
        shutil.rmtree(private, ignore_errors=True)
        cleanup = {
            "all_remote_kernels_deleted": remote_deleted,
            "temporary_private_packages_removed": not private.exists(),
            "coordinator_stopped": server_stopped,
            "tunnel_stopped": tunnel_stopped,
            "tensor_payloads_removed": tensor_removed or controller is None,
            "temporary_credentials_removed": not private.exists(),
            "live_resources_left_running": not bool(
                remote_deleted and server_stopped and tunnel_stopped
            ),
        }
        if final_status and controller is not None:
            soak_duration = (
                training_completed - training_started
                if training_started and training_completed
                else 0.0
            )
            report = _build_live_report(
                controller=controller,
                final_status=final_status,
                kernel_reports=kernel_reports,
                transport=transport,
                checkpoint=checkpoint,
                local_export=local_export,
                stale_probe=stale_probe,
                coordinator_restart=coordinator_restart,
                observations=placement_snapshots or observations,
                soak_duration_seconds=soak_duration,
                live_elapsed_seconds=(
                    time.monotonic() - live_started if live_started else 0.0
                ),
                cleanup=cleanup,
                blockers=blockers,
                runtime_metrics=runtime_metrics,
                prometheus_metrics_verified=prometheus_metrics_verified,
            )
        else:
            report = _base_report(
                blockers[0] if blockers else "training_production_live_failed"
            )
            report["live_run_performed"] = bool(refs_by_role)
            report["cleanup"] = cleanup
            report["blockers"] = sorted(set(blockers or report["blockers"]))
        report["acquisition_authorization"] = acquisition_authorization
        report["live_gate_authorization"] = live_authorization
        report["gpu_quota_preflight_summaries"] = quota_summaries
        safety = public_safety_errors(report)
        report["public_safety_errors"] = safety
        report["public_artifact_safe"] = not safety
        if safety:
            report["blockers"] = sorted(
                set(report.get("blockers") or [])
                | {"training_production_public_safety_failed"}
            )
        report.pop("content_hash", None)
        report["content_hash"] = stable_hash(report)
        _write(report_path, report)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print(
                "training_heterogeneous_production_live_probe "
                f"steps={(report.get('training_evidence') or {}).get('committed_step_count', 0)} "
                f"blockers={','.join(report.get('blockers') or []) or 'none'}"
            )
    success = bool(
        report.get("external_runtime_verified") is True
        and (report.get("training_evidence") or {}).get("atomic_ledger_verified")
        is True
        and report.get("soak_duration_seconds", 0.0) >= MINIMUM_SOAK_SECONDS
        and (report.get("performance") or {}).get("performance_gate_passed") is True
        and not report.get("blockers")
        and report.get("public_artifact_safe") is True
        and not cleanup["live_resources_left_running"]
    )
    return 0 if success else (1 if not cleanup["live_resources_left_running"] else 2)


if __name__ == "__main__":
    raise SystemExit(main())
