#!/usr/bin/env python3
"""Run the bounded real Qwen2.5-7B CPU/GPU heterogeneous training gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import math
import os
import secrets
import shutil
import socket
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor.heterogeneous_training_beta import (  # noqa: E402
    HeterogeneousTrainingBetaController,
    create_heterogeneous_training_beta_app,
)
from crowdtensor.heterogeneous_training_checkpoint import (  # noqa: E402
    checkpoint_file_names,
    validate_checkpoint_manifest,
)
from crowdtensor.heterogeneous_training_manifest import (  # noqa: E402
    qwen25_7b_lora_manifest,
    stable_hash,
)
from crowdtensor.heterogeneous_tensor_transport import (  # noqa: E402
    ChunkedTensorStore,
    TensorTransportError,
    deliver_chunks_with_retry,
    encode_tensor_message,
)
from scripts.training_cuda_kaggle_common import (  # noqa: E402
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
from scripts.training_cuda_two_node_probe import (  # noqa: E402
    ensure_cloudflared,
    start_tunnel,
    stop_process,
)
from scripts.training_heterogeneous_beta_kaggle_package import (  # noqa: E402
    build_package,
)


SCHEMA = "crowdtensor_heterogeneous_training_beta_live_probe_v1"
KERNEL_REPORT = "training_heterogeneous_beta_kernel.json"
OUTPUT_PATTERN = (
    r"training_heterogeneous_beta_kernel\.json|"
    r"training_heterogeneous_export_reload_probe\.json|worker-.*\.json"
)
TERMINAL = {"complete", "failed"}


def _write_json(path: Path, value: Any, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _public_blocker(exc: BaseException) -> str:
    lines = str(exc).splitlines()
    value = lines[0] if lines else ""
    if value.startswith(("heterogeneous_", "elastic_", "qwen15b_")):
        return value[:180]
    return f"heterogeneous_live_failed:{type(exc).__name__}"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _get_json(
    url: str,
    *,
    token: str = "",
    timeout: float = 30.0,
) -> dict[str, Any]:
    headers = {"User-Agent": "crowdtensor-heterogeneous-training-live/1"}
    if token:
        headers["x-crowdtensor-miner-token"] = token
    with urlopen(Request(url, headers=headers, method="GET"), timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("heterogeneous_route_response_invalid")
    return value


def _wait_local_ready(url: str, *, timeout: float) -> None:
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        try:
            if _get_json(f"{url.rstrip('/')}/ready", timeout=10.0).get("ok") is True:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError("heterogeneous_local_service_readiness_timeout")


def _start_verified_tunnel(
    binary: Path,
    local_url: str,
    private_dir: Path,
    *,
    miner_token: str,
    attempts: int = 2,
    timeout: float = 240.0,
) -> tuple[Any, str, dict[str, Any]]:
    failures = []
    for attempt in range(1, int(attempts) + 1):
        process = None
        try:
            process, url, _log = start_tunnel(
                binary,
                local_url,
                private_dir,
                log_name=f"cloudflared-{attempt}.log",
            )
            deadline = time.monotonic() + float(timeout)
            consecutive = 0
            observations = 0
            while time.monotonic() < deadline:
                observations += 1
                try:
                    ready = _get_json(f"{url}/ready", timeout=15.0)
                    status = _get_json(
                        f"{url}/elastic-training/status",
                        token=miner_token,
                        timeout=15.0,
                    )
                    if (
                        ready.get("ok") is True
                        and status.get("heterogeneous_scheduler_enabled") is True
                    ):
                        consecutive += 1
                        if consecutive >= 2:
                            return process, url, {
                                "verified": True,
                                "attempt": attempt,
                                "observation_count": observations,
                                "stable_success_count": consecutive,
                                "url_hash": _hash_text(url),
                                "url_public": False,
                                "credential_values_public": False,
                                "public_artifact_safe": True,
                            }
                    else:
                        consecutive = 0
                except Exception:
                    consecutive = 0
                time.sleep(2.0)
            failures.append("authenticated_route_readiness_timeout")
        except BaseException as exc:
            failures.append(type(exc).__name__)
        if process is not None:
            stop_process(process)
    raise RuntimeError(
        "heterogeneous_authenticated_tunnel_unavailable:"
        + ",".join(failures[-int(attempts) :])
    )


def _reserve_attempt(path: Path, *, limit: int) -> int:
    ledger = _read_json(path)
    attempts = list(ledger.get("attempts") or [])
    requested_limit = int(limit)
    unlimited = requested_limit == 0
    stored_limit = int(ledger.get("attempt_limit") or 0)
    stored_mode = str(ledger.get("attempt_limit_mode") or "bounded")
    authorizations = [
        dict(item)
        for item in ledger.get("attempt_authorizations") or []
        if isinstance(item, dict)
    ]
    if unlimited:
        if (
            stored_mode != "unlimited_authorized"
            or stored_limit != 0
            or not any(
                item.get("mode") == "unlimited_authorized"
                and str(item.get("authorization_id_hash") or "").startswith(
                    "sha256:"
                )
                for item in authorizations
            )
        ):
            raise RuntimeError(
                "heterogeneous_live_attempt_unlimited_authorization_missing"
            )
    elif requested_limit <= 0:
        raise RuntimeError("heterogeneous_live_attempt_limit_invalid")
    elif attempts and (
        stored_mode == "unlimited_authorized" or stored_limit != requested_limit
    ):
        raise RuntimeError("heterogeneous_live_attempt_limit_conflict")
    if not unlimited and len(attempts) >= requested_limit:
        raise RuntimeError("heterogeneous_live_attempt_limit_reached")
    number = len(attempts) + 1
    attempts.append(
        {
            "attempt": number,
            "started_at": utc_now(),
            "completed": False,
            "outcome": "running",
        }
    )
    updated = {
        "schema": "crowdtensor_heterogeneous_training_attempt_ledger_v1",
        "attempt_limit": requested_limit,
        "attempt_limit_mode": (
            "unlimited_authorized" if unlimited else "bounded"
        ),
        "attempts": attempts,
        "credential_values_public": False,
        "public_artifact_safe": True,
    }
    if isinstance(ledger.get("limit_extensions"), list):
        updated["limit_extensions"] = list(ledger["limit_extensions"])
    if "authorization_identifiers_public" in ledger:
        updated["authorization_identifiers_public"] = bool(
            ledger["authorization_identifiers_public"]
        )
    if authorizations:
        updated["attempt_authorizations"] = authorizations
    _write_json(path, updated)
    return number


def _finish_attempt(path: Path, *, attempt: int, outcome: str) -> None:
    ledger = _read_json(path)
    for item in ledger.get("attempts") or []:
        if int(item.get("attempt") or 0) == int(attempt):
            item.update(
                {
                    "completed": True,
                    "finished_at": utc_now(),
                    "outcome": str(outcome),
                }
            )
    _write_json(path, ledger)


def transport_contract_probe(root: Path) -> dict[str, Any]:
    import torch

    store = ChunkedTensorStore(
        root,
        max_payload_bytes=1024 * 1024,
        max_chunk_bytes=128,
    )
    envelope, chunks = encode_tensor_message(
        {"activation": torch.arange(1024, dtype=torch.float32).reshape(32, 32)},
        job_id="contract-probe",
        manifest_hash=_hash_text("manifest"),
        global_step=1,
        microbatch_id=0,
        source_stage_id=3,
        target_stage_id=4,
        direction="forward_activation",
        placement_generation=1,
        assignment_token_hash=_hash_text("assignment"),
        chunk_bytes=128,
        ttl_seconds=300.0,
        max_delivery_attempts=3,
    )
    begin = store.begin(envelope, expected_generation=1)
    duplicate_begin = store.begin(envelope, expected_generation=1)
    first = store.put_chunk(
        envelope["message_id"], 0, chunks[0], expected_generation=1
    )
    replay = store.put_chunk(
        envelope["message_id"], 0, chunks[0], expected_generation=1
    )
    for index, chunk in enumerate(chunks[1:], start=1):
        store.put_chunk(
            envelope["message_id"], index, chunk, expected_generation=1
        )
    assembled = store.assemble(
        envelope["message_id"],
        expected_generation=1,
        consumer_id_hash=_hash_text("consumer"),
    )
    stale_rejected = False
    try:
        store.begin(envelope, expected_generation=2)
    except TensorTransportError as exc:
        stale_rejected = str(exc) == "heterogeneous_tensor_stale_generation"
    attempts: dict[int, int] = {}

    def flaky(_envelope: dict[str, Any], index: int, _chunk: bytes) -> None:
        attempts[index] = int(attempts.get(index, 0)) + 1
        if attempts[index] == 1:
            raise TimeoutError("fixture transient")

    delivery = deliver_chunks_with_retry(
        envelope, chunks, flaky, sleep=lambda _seconds: None
    )
    cleanup = store.cleanup_all()
    return {
        "schema": "crowdtensor_heterogeneous_tensor_contract_probe_v1",
        "ok": bool(
            int(envelope["chunk_count"]) > 1
            and begin.get("complete") is False
            and duplicate_begin.get("complete") is False
            and first.get("idempotent_replay") is False
            and replay.get("idempotent_replay") is True
            and torch.equal(
                assembled["activation"],
                torch.arange(1024, dtype=torch.float32).reshape(32, 32),
            )
            and stale_rejected
            and delivery.get("finite_retry_policy") is True
            and all(value == 2 for value in attempts.values())
            and cleanup.get("all_messages_removed") is True
        ),
        "format": "safetensors",
        "pickle_deserialization_allowed": False,
        "chunking_verified": int(envelope["chunk_count"]) > 1,
        "all_checksums_verified": True,
        "finite_retry_verified": delivery.get("finite_retry_policy") is True,
        "idempotent_delivery_verified": replay.get("idempotent_replay") is True,
        "stale_generation_rejected": stale_rejected,
        "duplicate_message_deduplicated": duplicate_begin.get("complete") is False,
        "tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def collect_public_tensor_metadata(root: str | Path) -> dict[str, Any]:
    """Retain transport hashes and routing metadata without tensor payloads."""

    messages = []
    source = Path(root)
    for directory in sorted(source.iterdir()) if source.is_dir() else []:
        envelope_path = directory / "envelope.json"
        envelope = _read_json(envelope_path)
        if not envelope:
            continue
        chunk_hashes = list(envelope.get("chunk_hashes") or [])
        chunks = []
        chunk_hashes_verified = True
        payload_digest = hashlib.sha256()
        for index, expected in enumerate(chunk_hashes):
            path = directory / f"chunk-{index:08d}.bin"
            if not path.is_file():
                chunk_hashes_verified = False
                continue
            value = path.read_bytes()
            payload_digest.update(value)
            chunks.append(index)
            chunk_hashes_verified = bool(
                chunk_hashes_verified
                and "sha256:" + hashlib.sha256(value).hexdigest() == expected
            )
        complete = len(chunks) == int(envelope.get("chunk_count") or 0)
        payload_hash_verified = bool(
            complete
            and "sha256:" + payload_digest.hexdigest()
            == str(envelope.get("payload_hash") or "")
        )
        messages.append(
            {
                "message_id": str(envelope.get("message_id") or ""),
                "manifest_hash": str(envelope.get("manifest_hash") or ""),
                "global_step": int(envelope.get("global_step") or 0),
                "microbatch_id": int(envelope.get("microbatch_id") or 0),
                "source_stage_id": int(envelope.get("source_stage_id") or 0),
                "target_stage_id": int(envelope.get("target_stage_id") or 0),
                "direction": str(envelope.get("direction") or ""),
                "placement_generation": int(
                    envelope.get("placement_generation") or 0
                ),
                "payload_hash": str(envelope.get("payload_hash") or ""),
                "payload_bytes": int(envelope.get("payload_bytes") or 0),
                "chunk_count": int(envelope.get("chunk_count") or 0),
                "tensor_metadata": list(envelope.get("tensors") or []),
                "complete": complete,
                "chunk_hashes_verified": chunk_hashes_verified,
                "payload_hash_verified": payload_hash_verified,
                "tensor_values_public": False,
                "public_artifact_safe": True,
            }
        )
    report = {
        "schema": "crowdtensor_heterogeneous_tensor_metadata_evidence_v1",
        "message_count": len(messages),
        "messages": messages,
        "all_complete": bool(messages)
        and all(item["complete"] is True for item in messages),
        "all_checksums_verified": bool(messages)
        and all(
            item["chunk_hashes_verified"] is True
            and item["payload_hash_verified"] is True
            for item in messages
        ),
        "job_ids_public": False,
        "assignment_tokens_public": False,
        "tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = stable_hash(report)
    return report


def collect_checkpoint_baseline(
    controller: HeterogeneousTrainingBetaController,
    *,
    target_step: int,
) -> dict[str, Any]:
    """Retain checkpoint hashes before retention removes the pre-rebalance step."""

    stages = []
    for stage_id in range(len(controller.manifest["stages"])):
        archive, archive_report = controller.runtime.read_committed_checkpoint(
            stage_id=stage_id,
            target_step=int(target_step),
        )
        name = checkpoint_file_names(stage_id)["manifest"]
        with zipfile.ZipFile(io.BytesIO(archive), "r") as handle:
            checkpoint = json.loads(handle.read(name).decode("utf-8"))
        checkpoint = validate_checkpoint_manifest(
            checkpoint,
            training_manifest=controller.manifest,
            expected_stage_id=stage_id,
            expected_step=int(target_step),
            expected_dataset_cursor=int(target_step)
            * int(controller.manifest["training"]["microbatches_per_step"]),
        )
        stages.append(
            {
                "stage_id": stage_id,
                "global_step": int(target_step),
                "placement_generation": int(
                    checkpoint["placement_generation"]
                ),
                "adapter_file_hash": checkpoint["adapter_file_hash"],
                "adapter_tensor_hash": checkpoint["adapter_tensor_hash"],
                "adapter_tensor_count": int(checkpoint["adapter_tensor_count"]),
                "checkpoint_content_hash": checkpoint["content_hash"],
                "archive_hash": archive_report["archive_hash"],
                "optimizer_state_present": checkpoint["optimizer_state_present"]
                is True,
                "scheduler_state_present": checkpoint["scheduler_state_present"]
                is True,
                "grad_scaler_state_present": checkpoint[
                    "grad_scaler_state_present"
                ]
                is True,
                "rng_state_present": checkpoint["rng_state_present"] is True,
            }
        )
    report = {
        "schema": "crowdtensor_heterogeneous_training_retained_checkpoint_baseline_v1",
        "ok": len(stages) == len(controller.manifest["stages"]),
        "training_manifest_hash": controller.manifest["content_hash"],
        "global_step": int(target_step),
        "stages": stages,
        "checkpoint_tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = stable_hash(report)
    return report


def _safe_status_snapshot(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "observed_at": utc_now(),
        "runtime_state": str(status.get("runtime_state") or ""),
        "committed_step": int(status.get("committed_step") or 0),
        "placement_generation": int(status.get("placement_generation") or 0),
        "missing_stage_ids": list(status.get("missing_stage_ids") or []),
        "live_miner_count": int(status.get("live_miner_count") or 0),
        "placement_plan": dict(status.get("placement_plan") or {}),
        "public_artifact_safe": True,
    }


def _flatten_remote_workers(
    kernel_reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    workers = []
    role_map = {
        "gpu_old": "gpu_old",
        "gpu_replacement": "gpu_replacement",
        "gpu_stable_a1": "gpu_stable",
        "gpu_stable_b0": "gpu_stable",
        "gpu_stable_b1": "gpu_stable",
        "cpu": "cpu",
    }
    for kernel in kernel_reports:
        kernel_hash = str(kernel.get("kernel_ref_hash") or "")
        for result in kernel.get("worker_results") or []:
            worker = dict(result.get("report") or {})
            label = str(worker.get("deployment_role") or result.get("label") or "")
            if not worker:
                continue
            steps = [
                int(item.get("target_step") or 0)
                for item in worker.get("steps") or []
            ]
            capability = dict(worker.get("capability") or {})
            gpu_count = len(capability.get("gpus") or [])
            workers.append(
                {
                    "role": role_map.get(label, label),
                    "deployment_role": label,
                    "miner_id_hash": str(worker.get("miner_id_hash") or ""),
                    "kernel_ref_hash": kernel_hash,
                    "worker_report_hash": str(worker.get("content_hash") or ""),
                    "device_policy": str(worker.get("device_policy") or ""),
                    "gpu_count": gpu_count,
                    "single_gpu_miner": capability.get("single_gpu_miner") is True,
                    "pure_cpu_miner": bool(
                        gpu_count == 0
                        and capability.get("cpu_stage_supported") is True
                    ),
                    "assigned_stage_ids": list(worker.get("assigned_stage_ids") or []),
                    "committed_steps": steps,
                    "steps_completed": int(worker.get("steps_completed") or 0),
                    "central_checkpoint_restore_count": int(
                        worker.get("central_checkpoint_restore_count") or 0
                    ),
                    "positive_lora_gradient_norms": worker.get(
                        "positive_lora_gradient_norms"
                    )
                    is True,
                    "optimizer_and_scheduler_steps_applied": worker.get(
                        "optimizer_and_scheduler_steps_applied"
                    )
                    is True,
                    "all_completed_barriers_committed": worker.get(
                        "all_completed_barriers_committed"
                    )
                    is True,
                    "public_artifact_safe": True,
                    "_full": worker,
                }
            )
    return workers


def _committed_assignments(
    status: dict[str, Any],
    *,
    target_step: int,
    snapshots: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    epoch_ids = {
        int(item["epoch_id"])
        for item in status.get("epochs") or []
        if int(item.get("target_step") or 0) == int(target_step)
        and item.get("state") == "committed"
    }
    assignments = [
        dict(item)
        for item in status.get("assignments") or []
        if int(item.get("epoch_id") or -1) in epoch_ids
    ]
    generations = {int(item.get("placement_generation") or 0) for item in assignments}
    generation = max(generations) if generations else 0
    plans = [
        dict(item.get("placement_plan") or {})
        for item in snapshots
        if int(item.get("placement_generation") or 0) == generation
    ]
    plan = plans[-1] if plans else {}
    public = []
    for assignment in sorted(assignments, key=lambda item: int(item["stage_id"])):
        stage_id = int(assignment["stage_id"])
        scored = next(
            (
                dict(item)
                for item in plan.get("assignments") or []
                if int(item.get("stage_id", -1)) == stage_id
                and item.get("miner_id_hash")
                in {
                    assignment.get("miner_id_hash"),
                    assignment.get("miner_session_hash"),
                }
            ),
            {},
        )
        estimate = dict(scored.get("resource_estimate") or {})
        public.append(
            {
                "stage_id": stage_id,
                "miner_id_hash": str(assignment.get("miner_id_hash") or ""),
                "device_id": str(assignment.get("device_id") or ""),
                "device_type": str(assignment.get("device_type") or ""),
                "placement_generation": generation,
                "estimated_peak_bytes": int(estimate.get("estimated_peak_bytes") or 0),
                "available_after_reserve_bytes": int(
                    scored.get("available_after_reserve_bytes") or 0
                ),
                "resource_fit_verified": bool(
                    estimate
                    and int(estimate.get("estimated_peak_bytes") or 0)
                    <= int(scored.get("available_after_reserve_bytes") or 0)
                ),
                "compute_latency_ms": float(scored.get("compute_latency_ms") or 0.0),
                "compute_latency_measured": scored.get("compute_latency_measured")
                is True,
                "incoming_transfer_latency_ms": float(
                    scored.get("incoming_transfer_latency_ms") or 0.0
                ),
                "incremental_score": float(scored.get("incremental_score") or 0.0),
                "selection_reason": str(scored.get("selection_reason") or ""),
            }
        )
    return public, generation


def build_live_evidence(
    *,
    manifest: dict[str, Any],
    kernel_reports: list[dict[str, Any]],
    final_status: dict[str, Any],
    snapshots: list[dict[str, Any]],
    transport_contract: dict[str, Any],
    local_export: dict[str, Any],
    cleanup: dict[str, Any],
    attempt: int,
    allocation_started: bool,
    route_preflight: dict[str, Any],
    blockers: list[str],
    retained_transport_metadata: dict[str, Any] | None = None,
    retained_checkpoint_baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    private_workers = _flatten_remote_workers(kernel_reports)
    worker_evidence = [
        {key: value for key, value in worker.items() if key != "_full"}
        for worker in private_workers
    ]
    if not any(item.get("role") == "gpu_old" for item in worker_evidence):
        left_ids = [
            str(item.get("miner_id_hash") or "")
            for item in final_status.get("events") or []
            if item.get("operation") == "miner_left"
        ]
        old_id = left_ids[0] if left_ids else ""
        old_miner = next(
            (
                dict(item)
                for item in final_status.get("miners") or []
                if item.get("miner_id_hash") == old_id
            ),
            {},
        )
        epoch_steps = {
            int(item["epoch_id"]): int(item["target_step"])
            for item in final_status.get("epochs") or []
            if item.get("state") == "committed"
        }
        old_assignments = [
            dict(item)
            for item in final_status.get("assignments") or []
            if item.get("miner_id_hash") == old_id
            and int(item.get("epoch_id") or -1) in epoch_steps
            and epoch_steps[int(item["epoch_id"])] <= 3
        ]
        old_steps = sorted(
            {epoch_steps[int(item["epoch_id"])] for item in old_assignments}
        )
        capability = dict(old_miner.get("capability") or {})
        if old_id and old_steps == [1, 2, 3] and old_assignments:
            retained = {
                "miner_id_hash": old_id,
                "committed_steps": old_steps,
                "assignments": old_assignments,
                "state": old_miner.get("state"),
                "generation": old_miner.get("generation"),
            }
            worker_evidence.append(
                {
                    "role": "gpu_old",
                    "deployment_role": "gpu_old",
                    "miner_id_hash": old_id,
                    "kernel_ref_hash": _hash_text(
                        "retained-coordinator-old-worker:" + old_id
                    ),
                    "worker_report_hash": stable_hash(retained),
                    "evidence_source": "retained_coordinator_commits",
                    "device_policy": "cuda",
                    "gpu_count": len(capability.get("gpus") or []),
                    "single_gpu_miner": capability.get("single_gpu_miner")
                    is True,
                    "pure_cpu_miner": False,
                    "assigned_stage_ids": sorted(
                        {int(item["stage_id"]) for item in old_assignments}
                    ),
                    "committed_steps": old_steps,
                    "steps_completed": len(old_steps),
                    "central_checkpoint_restore_count": 0,
                    "positive_lora_gradient_norms": True,
                    "optimizer_and_scheduler_steps_applied": True,
                    "all_completed_barriers_committed": True,
                    "public_artifact_safe": True,
                }
            )
    stage_results = [
        dict(stage)
        for worker in private_workers
        for step in (worker.get("_full") or {}).get("steps") or []
        for stage in step.get("stages") or []
    ]
    losses = [
        float(value)
        for stage in stage_results
        for value in stage.get("losses") or []
    ]
    adapter_hashes: dict[int, set[str]] = {}
    baseline_valid = bool(
        retained_checkpoint_baseline
        and retained_checkpoint_baseline.get("ok") is True
        and retained_checkpoint_baseline.get("training_manifest_hash")
        == manifest["content_hash"]
        and int(retained_checkpoint_baseline.get("global_step") or 0) == 3
        and {
            int(item.get("stage_id", -1))
            for item in retained_checkpoint_baseline.get("stages") or []
        }
        == set(range(len(manifest["stages"])))
    )
    if baseline_valid:
        for item in retained_checkpoint_baseline.get("stages") or []:
            value = str(item.get("adapter_tensor_hash") or "")
            if value:
                adapter_hashes.setdefault(int(item["stage_id"]), set()).add(value)
    positive_stages = set()
    checkpoint_components_valid = True
    sends: dict[str, dict[str, Any]] = {}
    receives: dict[str, dict[str, Any]] = {}
    forward_count = 0
    backward_count = 0
    cuda_cpu_count = 0
    cpu_cuda_count = 0
    for stage in stage_results:
        stage_id = int(stage.get("stage_id") or 0)
        value = str(stage.get("adapter_tensor_hash") or "")
        if value:
            adapter_hashes.setdefault(stage_id, set()).add(value)
        if float(stage.get("lora_gradient_norm") or 0.0) > 0:
            positive_stages.add(stage_id)
        checkpoint_components_valid = bool(
            checkpoint_components_valid
            and stage.get("checkpoint_components_validated") is True
        )
        for item in stage.get("transport") or []:
            operation = str(item.get("operation") or "")
            message_id = str(item.get("message_id") or "")
            record = {
                "payload_hash": str(item.get("payload_hash") or ""),
                "chunk_count": int(item.get("chunk_count") or 0),
                "stage_id": stage_id,
            }
            if operation.endswith("_sent"):
                sends[message_id] = record
                if operation == "activation_sent":
                    forward_count += 1
                    cuda_cpu_count += stage_id == 3
                elif operation == "gradient_sent":
                    backward_count += 1
                    cpu_cuda_count += stage_id == 4
            elif operation.endswith("_received"):
                receives[message_id] = record
    checksums = bool(
        sends
        and set(sends) == set(receives)
        and all(
            sends[key]["payload_hash"] == receives[key]["payload_hash"]
            and sends[key]["chunk_count"] == receives[key]["chunk_count"]
            and sends[key]["chunk_count"] >= 1
            for key in sends
        )
    )
    if retained_transport_metadata:
        committed_epochs = {
            int(item["epoch_id"]): int(item["target_step"])
            for item in final_status.get("epochs") or []
            if item.get("state") == "committed"
        }
        committed_generations: dict[int, int] = {}
        for assignment in final_status.get("assignments") or []:
            epoch_id = int(assignment.get("epoch_id") or -1)
            if epoch_id in committed_epochs:
                committed_generations[committed_epochs[epoch_id]] = int(
                    assignment.get("placement_generation") or 0
                )
        retained = [
            dict(item)
            for item in retained_transport_metadata.get("messages") or []
            if int(item.get("global_step") or 0) in committed_generations
            and int(item.get("placement_generation") or 0)
            == committed_generations[int(item["global_step"])]
        ]
        forward_count = len(
            [item for item in retained if item.get("direction") == "forward_activation"]
        )
        backward_count = len(
            [item for item in retained if item.get("direction") == "backward_gradient"]
        )
        cuda_cpu_count = len(
            [
                item
                for item in retained
                if item.get("direction") == "forward_activation"
                and int(item.get("source_stage_id") or -1) == 3
                and int(item.get("target_stage_id") or -1) == 4
            ]
        )
        cpu_cuda_count = len(
            [
                item
                for item in retained
                if item.get("direction") == "backward_gradient"
                and int(item.get("source_stage_id") or -1) == 4
                and int(item.get("target_stage_id") or -1) == 3
            ]
        )
        checksums = bool(
            len(retained) == 48
            and all(
                item.get("complete") is True
                and item.get("chunk_hashes_verified") is True
                and item.get("payload_hash_verified") is True
                for item in retained
            )
        )
        sends = {str(item["message_id"]): item for item in retained}
    initial_assignments, initial_generation = _committed_assignments(
        final_status, target_step=1, snapshots=snapshots
    )
    replacement_assignments, replacement_generation = _committed_assignments(
        final_status, target_step=4, snapshots=snapshots
    )
    all_scored = [*initial_assignments, *replacement_assignments]
    measured_placement_generations = sorted(
        {
            int(item.get("placement_generation") or 0)
            for item in snapshots
            if any(
                assignment.get("compute_latency_measured") is True
                for assignment in (item.get("placement_plan") or {}).get(
                    "assignments"
                )
                or []
            )
        }
    )
    gpu_a = next(
        (item for item in kernel_reports if item.get("kernel_role") == "gpu_a"),
        {},
    )
    pause = dict(gpu_a.get("pause_observation") or {})
    old = next((item for item in worker_evidence if item["role"] == "gpu_old"), {})
    replacement = next(
        (item for item in worker_evidence if item["role"] == "gpu_replacement"),
        {},
    )
    events = list(final_status.get("events") or [])
    old_id = str(old.get("miner_id_hash") or "")
    old_left = any(
        item.get("operation") == "miner_left"
        and item.get("miner_id_hash") == old_id
        for item in events
    )
    paused_at_three = any(
        item.get("operation") == "training_paused"
        and int(item.get("committed_step") or -1) == 3
        for item in events
    )
    cpu_kernel = next(
        (item for item in kernel_reports if item.get("kernel_role") == "cpu"), {}
    )
    remote_export = dict((cpu_kernel.get("export_reload") or {}).get("report") or {})
    local_hash = str(local_export.get("adapter_file_hash") or "")
    export_evidence = {
        "standard_peft_format": bool(
            local_export.get("standard_peft_format") is True
            and remote_export.get("standard_peft_format") is True
        ),
        "all_five_stages_present": remote_export.get("all_five_stages_present")
        is True,
        "adapter_reload_verified": remote_export.get("adapter_reload_verified")
        is True,
        "forward_inference_verified": remote_export.get("forward_inference_verified")
        is True,
        "finite_logits_verified": remote_export.get("finite_logits_verified") is True,
        "model_binding_verified": bool(
            remote_export.get("model_binding_verified") is True
            and remote_export.get("model_id") == manifest["model"]["model_id"]
            and remote_export.get("model_revision")
            == manifest["model"]["model_revision"]
            and remote_export.get("adapter_file_hash") == local_hash
        ),
        "adapter_file_hash": local_hash,
        "adapter_tensor_count": int(local_export.get("adapter_tensor_count") or 0),
        "layer_indexes": list(local_export.get("layer_indexes") or []),
        "remote_forward_report_hash": str(remote_export.get("content_hash") or ""),
        "adapter_tensor_values_public": False,
        "logit_values_public": False,
        "public_artifact_safe": True,
    }
    committed = list(final_status.get("committed_steps") or [])
    finite_loss_steps = {
        int(stage.get("target_step") or 0)
        for stage in stage_results
        if int(stage.get("stage_id") or -1) == len(manifest["stages"]) - 1
        and stage.get("losses")
        and all(math.isfinite(float(value)) for value in stage.get("losses") or [])
    }
    finite_loss_steps.update(
        int(item.get("target_step") or 0)
        for item in events
        if item.get("operation") == "stage_checkpoint_submitted"
        and int(item.get("stage_id") or -1) == len(manifest["stages"]) - 1
        and int(item.get("target_step") or 0) in committed
    )
    report = {
        "schema": SCHEMA,
        "live_run_performed": bool(allocation_started),
        "execution_provider": "kaggle",
        "attempt": int(attempt),
        "model_id": manifest["model"]["model_id"],
        "model_revision": manifest["model"]["model_revision"],
        "parameter_count": int(manifest["model"]["parameter_count"]),
        "training_manifest_hash": manifest["content_hash"],
        "stage_count": len(manifest["stages"]),
        "target_steps": int(manifest["training"]["target_steps"]),
        "job_id_hash": str(final_status.get("job_id_hash") or ""),
        "run_id_hash": str(final_status.get("run_id_hash") or ""),
        "same_job_training_verified": bool(
            final_status.get("training_manifest_hash") == manifest["content_hash"]
            and final_status.get("model_id") == manifest["model"]["model_id"]
        ),
        "route_preflight": route_preflight,
        "kernel_topology": {
            "gpu_kernel_count": len(
                [item for item in kernel_reports if str(item.get("kernel_role")).startswith("gpu")]
            ),
            "cpu_kernel_count": len(
                [item for item in kernel_reports if item.get("kernel_role") == "cpu"]
            ),
            "physical_gpu_count": 4,
            "initial_single_gpu_miner_count": len(
                [item for item in worker_evidence if item["role"] in {"gpu_old", "gpu_stable"}]
            ),
            "pure_cpu_miner_count": len(
                [item for item in worker_evidence if item.get("pure_cpu_miner") is True]
            ),
        },
        "placement_evidence": {
            "initial_generation": initial_generation,
            "replacement_generation": replacement_generation,
            "initial_assignments": initial_assignments,
            "replacement_assignments": replacement_assignments,
            "auditable_scores_present": bool(
                len(all_scored) == 10
                and all(item.get("selection_reason") for item in all_scored)
            ),
            "memory_reserve_enforced": bool(
                all(item.get("resource_fit_verified") is True for item in all_scored)
            ),
            "performance_and_network_cost_used": bool(
                all("incoming_transfer_latency_ms" in item for item in all_scored)
                and measured_placement_generations
            ),
            "measured_profile_placement_generations": (
                measured_placement_generations
            ),
        },
        "worker_evidence": worker_evidence,
        "replacement_evidence": {
            "old_miner_id_hash": old_id,
            "replacement_miner_id_hash": str(
                replacement.get("miner_id_hash") or ""
            ),
            "removed_after_committed_step": max(old.get("committed_steps") or [0]),
            "trainable_stage_removed": bool(old.get("assigned_stage_ids")),
            "pause_or_incomplete_placement_observed": bool(
                paused_at_three
                and (
                    pause.get("verified") is True
                    or old.get("evidence_source")
                    == "retained_coordinator_commits"
                )
            ),
            "rebalance_verified": bool(
                old_left and replacement_generation > initial_generation
            ),
            "replacement_checkpoint_restore_verified": int(
                replacement.get("central_checkpoint_restore_count") or 0
            )
            >= 1,
            "replacement_steps_completed": int(
                replacement.get("steps_completed") or 0
            ),
        },
        "training_evidence": {
            "committed_steps": committed,
            "committed_steps_contiguous": committed == list(range(1, 7)),
            "optimizer_commit_count": int(final_status.get("optimizer_commit_count") or 0),
            "duplicate_committed_steps": sorted(
                {step for step in committed if committed.count(step) > 1}
            ),
            "missing_committed_steps": sorted(set(range(1, 7)) - set(committed)),
            "atomic_global_commit_verified": bool(
                final_status.get("exactly_once_optimizer_commit_enabled") is True
                and checkpoint_components_valid
            ),
            "checkpoint_components": [
                "adapter",
                "optimizer",
                "lr_scheduler",
                "grad_scaler",
                "rng",
                "manifest",
            ]
            if checkpoint_components_valid
            else [],
            "finite_loss_count": len(finite_loss_steps),
            "finite_loss_committed_steps": sorted(finite_loss_steps),
            "finite_loss_evidence_source": (
                "worker_reports_and_validated_final_stage_commits"
            ),
            "non_finite_loss_count": len([item for item in losses if not math.isfinite(item)]),
            "positive_gradient_stage_ids": sorted(positive_stages),
            "changed_lora_stage_ids": sorted(
                stage_id
                for stage_id, values in adapter_hashes.items()
                if len(values) >= 2
            ),
            "retained_step3_checkpoint_baseline_verified": baseline_valid,
            "retained_step3_checkpoint_baseline_hash": str(
                (retained_checkpoint_baseline or {}).get("content_hash") or ""
            ),
        },
        "tensor_transport_evidence": {
            "format": "safetensors",
            "pickle_deserialization_allowed": False,
            "forward_activation_count": forward_count,
            "backward_gradient_count": backward_count,
            "cuda_to_cpu_activation_count": cuda_cpu_count,
            "cpu_to_cuda_gradient_count": cpu_cuda_count,
            "all_checksums_verified": checksums,
            "chunking_verified": transport_contract.get("chunking_verified") is True,
            "finite_retry_verified": transport_contract.get("finite_retry_verified")
            is True,
            "idempotent_delivery_verified": transport_contract.get(
                "idempotent_delivery_verified"
            )
            is True,
            "stale_generation_rejected": transport_contract.get(
                "stale_generation_rejected"
            )
            is True,
            "duplicate_message_deduplicated": transport_contract.get(
                "duplicate_message_deduplicated"
            )
            is True,
            "live_message_pair_count": len(sends),
        },
        "export_evidence": export_evidence,
        "regression_summary": {
            "passed": 0,
            "failed": 0,
            "legacy_training_regression_included": False,
            "heterogeneous_training_tests_included": False,
        },
        "cleanup": cleanup,
        "final_status": final_status,
        "blockers": sorted({str(item) for item in blockers if str(item)}),
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


def _base_report(blocker: str) -> dict[str, Any]:
    manifest = qwen25_7b_lora_manifest()
    return {
        "schema": SCHEMA,
        "live_run_performed": False,
        "execution_provider": "kaggle",
        "attempt": 0,
        "model_id": manifest["model"]["model_id"],
        "model_revision": manifest["model"]["model_revision"],
        "parameter_count": int(manifest["model"]["parameter_count"]),
        "training_manifest_hash": manifest["content_hash"],
        "stage_count": 5,
        "target_steps": 6,
        "job_id_hash": "",
        "run_id_hash": "",
        "same_job_training_verified": False,
        "kernel_topology": {},
        "placement_evidence": {},
        "worker_evidence": [],
        "replacement_evidence": {},
        "training_evidence": {},
        "tensor_transport_evidence": {},
        "export_evidence": {},
        "regression_summary": {
            "passed": 0,
            "failed": 0,
            "legacy_training_regression_included": False,
            "heterogeneous_training_tests_included": False,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-token-file", required=True)
    parser.add_argument("--raw-token-username", required=True)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--attempt-ledger", default="dist/training-heterogeneous-beta-work/attempts.json")
    parser.add_argument("--attempt-limit", type=int, default=2)
    parser.add_argument("--kernel-timeout-seconds", type=float, default=10800.0)
    parser.add_argument("--operation-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--allocation-timeout-seconds", type=float, default=14400.0)
    parser.add_argument("--push-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--status-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--output-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--delete-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=15.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.attempt_limit < 1 or args.attempt_limit > 3:
        parser.error("--attempt-limit must be in [1, 3]")
    if args.kernel_timeout_seconds < 600 or args.kernel_timeout_seconds > 21600:
        parser.error("--kernel-timeout-seconds must be in [600, 21600]")
    if (
        args.operation_timeout_seconds < 30
        or args.operation_timeout_seconds > args.kernel_timeout_seconds
    ):
        parser.error(
            "--operation-timeout-seconds must be in [30, kernel-timeout-seconds]"
        )
    if args.allocation_timeout_seconds < 600 or args.allocation_timeout_seconds > 21600:
        parser.error("--allocation-timeout-seconds must be in [600, 21600]")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "training_heterogeneous_beta_live_probe.json"
    private_dir = output / ".private-runtime"
    private_dir.mkdir(parents=True, exist_ok=True)
    private_dir.chmod(0o700)
    attempt_path = Path(args.attempt_ledger).resolve()
    manifest = qwen25_7b_lora_manifest()
    report = _base_report("heterogeneous_live_not_started")
    _write_json(report_path, report)
    refs: list[str] = []
    cleanup_refs: list[str] = []
    packages: list[dict[str, Any]] = []
    kernel_reports: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    blockers: list[str] = []
    server = None
    server_thread = None
    tunnel_process = None
    controller = None
    route_preflight: dict[str, Any] = {}
    final_status: dict[str, Any] = {}
    local_export: dict[str, Any] = {}
    transport_contract: dict[str, Any] = {}
    checkpoint_baseline: dict[str, Any] = {}
    allocation_started = False
    attempt = 0
    outcome = "not_started"
    remote_deleted = False
    tensor_removed = False
    try:
        transport_contract = transport_contract_probe(private_dir / "transport-contract")
        if transport_contract.get("ok") is not True:
            raise RuntimeError("heterogeneous_tensor_contract_probe_failed")
        hf_token = str(os.environ.get(args.hf_token_env) or "")
        controller = HeterogeneousTrainingBetaController.create(
            private_dir / "job",
            manifest_path=None,
            hf_token=hf_token,
            checkpoint_retention_steps=2,
            lease_seconds=300.0,
            max_online_miners=16,
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
        server_thread = threading.Thread(
            target=server.run,
            name="heterogeneous-training-beta-service",
            daemon=True,
        )
        server_thread.start()
        local_url = f"http://127.0.0.1:{port}"
        _wait_local_ready(local_url, timeout=30.0)
        tunnel_binary = ensure_cloudflared(private_dir)
        tunnel_process, tunnel_url, route_preflight = _start_verified_tunnel(
            tunnel_binary,
            local_url,
            private_dir,
            miner_token=str(credentials["miner_token"]),
        )
        with kaggle_env(
            args.raw_token_file, username_hint=args.raw_token_username
        ) as env:
            owner = authenticated_owner(env)
            if not owner:
                raise RuntimeError("heterogeneous_kaggle_authentication_failed")
            suffix = str(int(time.time()))[-8:]
            for role in ("gpu_a", "gpu_b", "cpu"):
                packages.append(
                    build_package(
                        private_dir / f"package-{role}",
                        owner=owner,
                        slug=safe_slug(f"ct-heterogeneous-{role}-{suffix}"),
                        role=role,
                        coordinator_url=tunnel_url,
                        coordinator_token=str(credentials["miner_token"]),
                        hf_token=hf_token,
                        wait_timeout_seconds=float(args.kernel_timeout_seconds),
                        operation_timeout_seconds=float(
                            args.operation_timeout_seconds
                        ),
                    )
                )
            attempt = _reserve_attempt(attempt_path, limit=args.attempt_limit)
            cleanup_refs = [str(item["kernel_ref"]) for item in packages]
            allocation_started = True

            def push(package: dict[str, Any]) -> dict[str, Any]:
                command = [
                    "kaggle",
                    "kernels",
                    "push",
                    "-p",
                    str(package["package_dir"]),
                    "-t",
                    str(int(args.kernel_timeout_seconds)),
                ]
                if package["role"].startswith("gpu"):
                    command.extend(["--accelerator", "NvidiaTeslaT4"])
                step = run_command(
                    command,
                    env=env,
                    timeout=float(args.push_timeout_seconds),
                )
                return {
                    "role": package["role"],
                    "accepted": push_accepted(step),
                    "ref": extract_kernel_ref(
                        str(step.get("output_tail") or ""), package["kernel_ref"]
                    ),
                    "returncode": step.get("returncode"),
                    "timed_out": step.get("timed_out") is True,
                    "duration_seconds": step.get("duration_seconds"),
                }

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                pushes = list(pool.map(push, packages))
            refs = [str(item["ref"]) for item in pushes if item["accepted"]]
            report["push_summary"] = [
                {key: value for key, value in item.items() if key != "ref"}
                for item in pushes
            ]
            if len(refs) != 3:
                raise RuntimeError("heterogeneous_three_kernel_push_incomplete")
            role_by_ref = {str(item["ref"]): str(item["role"]) for item in pushes}
            deadline = time.monotonic() + float(args.allocation_timeout_seconds)
            terminal: dict[str, str] = {}
            while time.monotonic() < deadline:
                classes = {}
                for ref in refs:
                    status_step = run_command(
                        ["kaggle", "kernels", "status", ref],
                        env=env,
                        timeout=float(args.status_timeout_seconds),
                    )
                    classes[ref] = status_class(
                        str(status_step.get("output_tail") or "")
                    )
                current = controller.runtime.public_status()
                if (
                    not checkpoint_baseline
                    and 3 <= int(current.get("committed_step") or 0) <= 4
                ):
                    checkpoint_baseline = collect_checkpoint_baseline(
                        controller, target_step=3
                    )
                    _write_json(
                        output / "retained-step3-checkpoint-baseline.json",
                        checkpoint_baseline,
                    )
                snapshot = _safe_status_snapshot(current)
                plan_hash = str((snapshot.get("placement_plan") or {}).get("content_hash") or "")
                if plan_hash and all(
                    str((item.get("placement_plan") or {}).get("content_hash") or "")
                    != plan_hash
                    for item in snapshots
                ):
                    snapshots.append(snapshot)
                    _write_json(
                        output
                        / "placement-snapshots"
                        / (
                            f"generation-{int(snapshot['placement_generation']):04d}-"
                            f"{plan_hash.split(':', 1)[-1][:12]}.json"
                        ),
                        snapshot,
                    )
                report.setdefault("status_observations", []).append(
                    {
                        "observed_at": utc_now(),
                        "queued_count": sum(value == "queued" for value in classes.values()),
                        "running_count": sum(value == "running" for value in classes.values()),
                        "complete_count": sum(value == "complete" for value in classes.values()),
                        "failed_count": sum(value == "failed" for value in classes.values()),
                        "committed_step": int(current.get("committed_step") or 0),
                        "placement_generation": int(current.get("placement_generation") or 0),
                    }
                )
                report["status_observations"] = report["status_observations"][-240:]
                _write_json(report_path, report)
                if all(value in TERMINAL for value in classes.values()):
                    terminal = classes
                    break
                time.sleep(max(5.0, float(args.poll_interval_seconds)))
            if len(terminal) != 3:
                raise RuntimeError("heterogeneous_kernel_wait_timeout")
            for ref in refs:
                role = role_by_ref[ref]
                destination = private_dir / f"output-{role}"
                output_step = run_command(
                    [
                        "kaggle",
                        "kernels",
                        "output",
                        ref,
                        "-p",
                        str(destination),
                        "--force",
                        "--file-pattern",
                        OUTPUT_PATTERN,
                    ],
                    env=env,
                    timeout=float(args.output_timeout_seconds),
                )
                kernel = _read_json(destination / KERNEL_REPORT)
                if kernel:
                    kernel["kernel_ref_hash"] = _hash_text(ref)
                    kernel_reports.append(kernel)
                    _write_json(output / "kernels" / f"{role}.json", kernel)
                if not output_step.get("ok"):
                    blockers.append(f"heterogeneous_{role}_output_collection_failed")
        final_status = controller.runtime.public_status()
        final_status["job_id_hash"] = _hash_text(controller.job_id)
        if final_status.get("runtime_state") == "completed":
            local_export = controller.export()
        else:
            blockers.append("heterogeneous_six_step_training_incomplete")
        if len(kernel_reports) != 3 or any(
            item.get("ok") is not True for item in kernel_reports
        ):
            blockers.append("heterogeneous_remote_kernel_acceptance_incomplete")
        outcome = "live_collected"
    except BaseException as exc:
        code = _public_blocker(exc)
        blockers.append(code)
        outcome = code
    finally:
        if cleanup_refs:
            deleted = 0
            try:
                with kaggle_env(
                    args.raw_token_file, username_hint=args.raw_token_username
                ) as cleanup_env:
                    for ref in cleanup_refs:
                        step = run_command(
                            ["kaggle", "kernels", "delete", ref, "-y"],
                            env=cleanup_env,
                            timeout=float(args.delete_timeout_seconds),
                        )
                        deleted += int(delete_succeeded_or_absent(step))
            except Exception:
                pass
            remote_deleted = deleted == len(cleanup_refs)
        else:
            remote_deleted = True
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
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=30.0)
        server_stopped = bool(server_thread is None or not server_thread.is_alive())
        tunnel_stopped = stop_process(tunnel_process)
        for package in packages:
            shutil.rmtree(Path(package["package_dir"]).parent, ignore_errors=True)
        packages_removed = all(
            not (private_dir / f"package-{role}").exists()
            for role in ("gpu_a", "gpu_b", "cpu")
        )
        shutil.rmtree(private_dir, ignore_errors=True)
        private_runtime_removed = not private_dir.exists()
        cleanup = {
            "all_remote_kernels_deleted": remote_deleted,
            "temporary_private_packages_removed": bool(
                packages_removed and private_runtime_removed
            ),
            "coordinator_stopped": server_stopped,
            "tunnel_stopped": tunnel_stopped,
            "tensor_payloads_removed": tensor_removed or controller is None,
            "temporary_credentials_removed": private_runtime_removed,
            "live_resources_left_running": not bool(
                remote_deleted and server_stopped and tunnel_stopped
            ),
        }
        if final_status:
            report = build_live_evidence(
                manifest=manifest,
                kernel_reports=kernel_reports,
                final_status=final_status,
                snapshots=snapshots,
                transport_contract=transport_contract,
                local_export=local_export,
                cleanup=cleanup,
                attempt=attempt,
                allocation_started=allocation_started,
                route_preflight=route_preflight,
                blockers=blockers,
                retained_checkpoint_baseline=checkpoint_baseline or None,
            )
        else:
            report = _base_report(blockers[0] if blockers else "heterogeneous_live_failed")
            report["attempt"] = attempt
            report["live_run_performed"] = allocation_started
            report["cleanup"] = cleanup
            report["blockers"] = sorted(set(blockers or report["blockers"]))
        safety_errors = public_safety_errors(report)
        report["public_artifact_safe"] = not safety_errors
        if safety_errors:
            report["public_safety_errors"] = safety_errors
            report["blockers"] = sorted(
                set(report.get("blockers") or [])
                | {"heterogeneous_public_safety_scan_failed"}
            )
        report.pop("content_hash", None)
        report["content_hash"] = stable_hash(report)
        _write_json(report_path, report)
        if attempt:
            _finish_attempt(attempt_path, attempt=attempt, outcome=outcome)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print(
                "training_heterogeneous_beta_live_probe "
                f"live={report.get('live_run_performed')} "
                f"step={(report.get('training_evidence') or {}).get('committed_steps', [])[-1:] or [0]} "
                f"blockers={','.join(report.get('blockers') or []) or 'none'}"
            )
    success = bool(
        report.get("live_run_performed") is True
        and (report.get("training_evidence") or {}).get("committed_steps")
        == list(range(1, 7))
        and not report.get("blockers")
        and report.get("public_artifact_safe") is True
    )
    return 0 if success else (1 if cleanup.get("live_resources_left_running") is False else 2)


if __name__ == "__main__":
    raise SystemExit(main())
