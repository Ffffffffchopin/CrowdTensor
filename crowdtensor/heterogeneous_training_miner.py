"""Ordinary Miner loop for manifest-driven heterogeneous pipeline training."""

from __future__ import annotations

import concurrent.futures
import gc
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from .elastic_training_client import ElasticTrainingHTTPClient
from .heterogeneous_qwen_source import materialize_qwen_stage_shard
from .heterogeneous_qwen_training import (
    HeterogeneousStageProcessClient,
    _spawn_process_with_pipe,
)
from .heterogeneous_training_manifest import stable_hash, validate_training_manifest
from .heterogeneous_training_scheduler import (
    discover_heterogeneous_training_capability,
    validate_miner_capability,
)


TELEMETRY_INTERVAL_AFTER_OPTIMIZATION = 5


MINER_REPORT_SCHEMA = "crowdtensor_heterogeneous_training_miner_v1"
RECOVERABLE_EPOCH_ERRORS = {
    "elastic_barrier_epoch_aborted",
    "elastic_barrier_wait_timeout",
    "elastic_stage_assignment_stale",
    "elastic_stage_placement_generation_stale",
    "elastic_tensor_receive_timeout",
    "elastic_tensor_global_step_stale",
    "heterogeneous_tensor_stale_generation",
}


def _capability_probe_main(connection: Any, settings: dict[str, Any]) -> None:
    try:
        capability = discover_heterogeneous_training_capability(**settings)
        connection.send({"ok": True, "capability": capability})
    except BaseException as exc:
        connection.send(
            {
                "ok": False,
                "error_class": type(exc).__name__,
                "public_artifact_safe": True,
            }
        )
    finally:
        connection.close()


def _discover_capability_isolated(
    *,
    miner_id_hash: str,
    max_stage_count: int,
    run_microbenchmark: bool,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Probe JAX TPU capacity in a process that exits before stage startup."""

    parent, child, process = _spawn_process_with_pipe(
        _capability_probe_main,
        (
            {
                "miner_id_hash": miner_id_hash,
                "max_stage_count": max_stage_count,
                "run_microbenchmark": run_microbenchmark,
                "include_jax_tpu": True,
            },
        ),
    )
    child.close()
    try:
        if not parent.poll(float(timeout)):
            process.terminate()
            process.join(timeout=30.0)
            raise TimeoutError("heterogeneous_miner_tpu_capability_probe_timeout")
        result = parent.recv()
    finally:
        parent.close()
    process.join(timeout=60.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=30.0)
        raise RuntimeError("heterogeneous_miner_tpu_capability_probe_did_not_exit")
    if result.get("ok") is not True or process.exitcode != 0:
        raise RuntimeError(
            "heterogeneous_miner_tpu_capability_probe_failed:"
            + str(result.get("error_class") or "unknown")
        )
    return validate_miner_capability(result["capability"])


def _recoverable_epoch_error(exc: BaseException) -> str:
    value = str(exc)
    return next((code for code in RECOVERABLE_EPOCH_ERRORS if code in value), "")


def miner_capability(
    *,
    miner_id_hash: str,
    device_policy: str,
    cuda_devices: list[int] | None = None,
    max_stage_count: int = 0,
    run_microbenchmark: bool = True,
) -> dict[str, Any]:
    policy = str(device_policy)
    if policy not in {"cpu", "cuda", "jax_tpu", "mixed", "auto"}:
        raise ValueError("heterogeneous_miner_device_policy_invalid")
    if policy in {"jax_tpu", "auto"}:
        value = _discover_capability_isolated(
            miner_id_hash=miner_id_hash,
            max_stage_count=max_stage_count,
            run_microbenchmark=run_microbenchmark,
        )
    else:
        value = discover_heterogeneous_training_capability(
            miner_id_hash=miner_id_hash,
            max_stage_count=max_stage_count,
            run_microbenchmark=run_microbenchmark,
            include_jax_tpu=False,
        )
    value.pop("content_hash", None)
    if policy == "auto":
        policy = (
            "jax_tpu"
            if value.get("tpu_groups")
            else "cuda"
            if value["gpus"]
            else "cpu"
        )
    selected = None if cuda_devices is None else {int(item) for item in cuda_devices}
    if selected is not None:
        value["gpus"] = [
            item for item in value["gpus"] if int(item["device_index"]) in selected
        ]
    if policy == "cpu":
        value["gpus"] = []
        if "tpu_groups" in value:
            value["tpu_groups"] = []
            value["jax_tpu_stage_supported"] = False
        value["cpu_stage_supported"] = True
    elif policy == "cuda":
        if not value["gpus"]:
            raise RuntimeError("heterogeneous_miner_cuda_device_unavailable")
        value["cpu_stage_supported"] = False
        if "tpu_groups" in value:
            value["tpu_groups"] = []
            value["jax_tpu_stage_supported"] = False
    elif policy == "jax_tpu":
        if not value.get("tpu_groups"):
            raise RuntimeError("heterogeneous_miner_jax_tpu_device_unavailable")
        value["gpus"] = []
        value["cpu_stage_supported"] = False
    else:
        if not value["gpus"]:
            raise RuntimeError("heterogeneous_miner_mixed_requires_cuda")
        value["cpu_stage_supported"] = True
    value["max_stage_count"] = int(
        max_stage_count
        or (
            1
            if policy == "jax_tpu"
            else len(value["gpus"])
            if policy == "cuda"
            else max(1, len(value["gpus"]) + 1)
        )
    )
    value["single_gpu_miner"] = len(value["gpus"]) == 1
    value["multi_gpu_miner"] = len(value["gpus"]) > 1
    return validate_miner_capability(value)


def _assignment_compute_dtype(manifest: dict[str, Any], device_type: str) -> str:
    return manifest["precision"][
        {
            "cpu": "cpu_compute_dtype",
            "cuda": "cuda_compute_dtype",
            "jax_tpu": "jax_tpu_compute_dtype",
        }[str(device_type)]
    ]


def _batch(
    tokenized: dict[str, Any],
    assignment: dict[str, Any],
    *,
    microbatch_id: int,
    microbatch_size: int,
) -> list[list[int]]:
    rows = list(tokenized.get("train") or [])
    if not rows:
        raise RuntimeError("heterogeneous_miner_training_rows_missing")
    start = (
        int(assignment["base_dataset_cursor"])
        + int(microbatch_id) * int(microbatch_size)
    )
    return [
        list(rows[(start + offset) % len(rows)])
        for offset in range(int(microbatch_size))
    ]


def _interval_ms(value: dict[str, Any]) -> float:
    return max(
        0.0,
        (int(value.get("ended_ns") or 0) - int(value.get("started_ns") or 0))
        / 1_000_000.0,
    )


def _capability_device(capability: dict[str, Any], device_id: str) -> dict[str, Any]:
    if str(device_id) == "cpu":
        return dict(capability.get("cpu") or {})
    for key in ("gpus", "tpu_groups"):
        for value in capability.get(key) or []:
            if str(value.get("device_id") or "") == str(device_id):
                return dict(value)
    return {}


def _report_committed_telemetry(
    *,
    client: ElasticTrainingHTTPClient,
    capability: dict[str, Any],
    manifest: dict[str, Any],
    assignments: list[dict[str, Any]],
    stage_results: list[dict[str, Any]],
    committed_step: int,
) -> list[dict[str, Any]]:
    reports = []
    results = {int(item["stage_id"]): item for item in stage_results}
    network = dict(capability.get("network") or {})
    stages = {int(item["stage_id"]): item for item in manifest["stages"]}
    for assignment in assignments:
        stage_id = int(assignment["stage_id"])
        device_id = str(assignment["device_id"])
        device = _capability_device(capability, device_id)
        result = results[stage_id]
        total_memory = int(
            device.get("total_memory_bytes")
            or device.get("total_hbm_bytes")
            or 0
        )
        reported_free = int(
            device.get("free_memory_bytes")
            or device.get("free_hbm_bytes")
            or total_memory
        )
        peak = max(
            int(
                profile.get("peak_memory_bytes") or 0
            )
            for profile in [result]
        )
        elapsed_ms = max(
            0.001,
            float(result.get("forward_latency_ms") or 0.0)
            + float(result.get("backward_latency_ms") or 0.0),
        )
        throughput = (
            float(stages[stage_id].get("estimated_compute_units") or 0.0)
            / (elapsed_ms / 1000.0)
        )
        try:
            reports.append(
                client.report_device_telemetry(
                    device_id=device_id,
                    free_memory_bytes=max(0, reported_free - peak),
                    utilization_fraction=(
                        min(1.0, float(peak) / max(1.0, float(total_memory)))
                        if total_memory
                        else 0.0
                    ),
                    throughput_units_per_second=throughput,
                    network_bandwidth_bytes_per_second=float(
                        network.get("measured_bandwidth_bytes_per_second") or 0.0
                    ),
                    network_latency_ms=float(
                        network.get("measured_round_trip_latency_ms") or 0.0
                    ),
                    checkpoint_step=int(committed_step),
                    health_score=1.0,
                )
            )
        except BaseException as exc:
            reports.append(
                {
                    "ok": False,
                    "error_code": "elastic_device_telemetry_report_failed:"
                    + type(exc).__name__,
                    "device_id": device_id,
                    "public_artifact_safe": True,
                }
            )
    return reports


def _committed_telemetry_sampling_interval(
    *, committed_step: int, optimization_after_step: int
) -> int:
    if (
        int(optimization_after_step) >= 0
        and int(committed_step) > int(optimization_after_step)
    ):
        return TELEMETRY_INTERVAL_AFTER_OPTIMIZATION
    return 1


def _execute_stage_epoch(
    *,
    client: ElasticTrainingHTTPClient,
    process: HeterogeneousStageProcessClient,
    assignment: dict[str, Any],
    manifest: dict[str, Any],
    tokenized: dict[str, Any],
    checkpoint_dir: Path,
    wait_timeout: float,
) -> dict[str, Any]:
    stage_id = int(assignment["stage_id"])
    stage_count = len(manifest["stages"])
    microbatch_count = int(manifest["training"]["microbatches_per_step"])
    microbatch_size = int(manifest["training"]["microbatch_size"])
    forward_ms = 0.0
    backward_ms = 0.0
    transport = []
    losses = []
    process.call("begin_step", timeout=wait_timeout)
    try:
        for microbatch_id in range(microbatch_count):
            labels = _batch(
                tokenized,
                assignment,
                microbatch_id=microbatch_id,
                microbatch_size=microbatch_size,
            )
            if stage_id == 0:
                forward = process.call(
                    "forward",
                    timeout=wait_timeout,
                    microbatch_id=microbatch_id,
                    value=labels,
                )
            else:
                tensors, received = client.receive_tensors(
                    assignment,
                    source_stage_id=stage_id - 1,
                    direction="forward_activation",
                    microbatch_id=microbatch_id,
                    timeout=wait_timeout,
                    target_device="cpu",
                    target_dtype=_assignment_compute_dtype(
                        manifest, assignment["device_type"]
                    ),
                )
                transport.append({"operation": "activation_received", **received})
                activation = tensors["activation"]
                if stage_id == stage_count - 1:
                    final = process.call(
                        "loss_backward",
                        timeout=wait_timeout,
                        microbatch_id=microbatch_id,
                        hidden_states=activation,
                        labels=labels,
                        microbatch_count=microbatch_count,
                    )
                    losses.append(float(final["loss"]))
                    forward_ms += _interval_ms(final["compute_interval"])
                    sent = client.send_tensors(
                        assignment,
                        {"gradient": final["activation_gradient"]},
                        target_stage_id=stage_id - 1,
                        direction="backward_gradient",
                        microbatch_id=microbatch_id,
                        manifest_hash=manifest["content_hash"],
                    )
                    transport.append({"operation": "gradient_sent", **sent})
                    continue
                forward = process.call(
                    "forward",
                    timeout=wait_timeout,
                    microbatch_id=microbatch_id,
                    value=activation,
                )
            forward_ms += _interval_ms(forward["compute_interval"])
            sent = client.send_tensors(
                assignment,
                {"activation": forward["activation"]},
                target_stage_id=stage_id + 1,
                direction="forward_activation",
                microbatch_id=microbatch_id,
                manifest_hash=manifest["content_hash"],
            )
            transport.append({"operation": "activation_sent", **sent})
            tensors, received = client.receive_tensors(
                assignment,
                source_stage_id=stage_id + 1,
                direction="backward_gradient",
                microbatch_id=microbatch_id,
                timeout=wait_timeout,
                target_device="cpu",
                target_dtype=_assignment_compute_dtype(
                    manifest, assignment["device_type"]
                ),
            )
            transport.append({"operation": "gradient_received", **received})
            backward = process.call(
                "backward",
                timeout=wait_timeout,
                microbatch_id=microbatch_id,
                activation_gradient=tensors["gradient"],
            )
            backward_ms += _interval_ms(backward["compute_interval"])
            if stage_id > 0:
                sent = client.send_tensors(
                    assignment,
                    {"gradient": backward["activation_gradient"]},
                    target_stage_id=stage_id - 1,
                    direction="backward_gradient",
                    microbatch_id=microbatch_id,
                    manifest_hash=manifest["content_hash"],
                )
                transport.append({"operation": "gradient_sent", **sent})
        finish = process.call(
            "finish_step",
            timeout=wait_timeout,
            global_step=int(assignment["target_step"]),
            dataset_cursor=int(assignment["dataset_cursor"]),
        )
        client.report_stage_runtime(
            assignment,
            event_type="profile",
            forward_latency_ms=forward_ms,
            backward_latency_ms=backward_ms,
            peak_memory_bytes=max(
                int(finish.get("peak_allocated_bytes") or 0),
                int(finish.get("peak_reserved_bytes") or 0),
            ),
            sample_count=microbatch_count,
            compile_latency_ms=float(finish.get("compile_latency_ms") or 0.0),
            steady_forward_latency_ms=float(
                finish.get("steady_forward_latency_ms") or forward_ms
            ),
            steady_backward_latency_ms=float(
                finish.get("steady_backward_latency_ms") or backward_ms
            ),
        )
        checkpoint_started = time.perf_counter()
        submission, archive = client.submit_checkpoint(
            assignment,
            checkpoint_dir=checkpoint_dir,
            training_manifest=manifest,
        )
        checkpoint_overhead_ms = (
            time.perf_counter() - checkpoint_started
        ) * 1000.0
        return {
            "stage_id": stage_id,
            "device_id": assignment["device_id"],
            "device_type": assignment["device_type"],
            "placement_generation": int(assignment["placement_generation"]),
            "target_step": int(assignment["target_step"]),
            "forward_latency_ms": forward_ms,
            "backward_latency_ms": backward_ms,
            "peak_memory_bytes": max(
                int(finish.get("peak_allocated_bytes") or 0),
                int(finish.get("peak_reserved_bytes") or 0),
            ),
            "losses": losses,
            "lora_gradient_norm": float(finish["lora_gradient_norm"]),
            "optimizer_step_applied": finish["optimizer_step_applied"] is True,
            "scheduler_step_applied": finish["scheduler_step_applied"] is True,
            "checkpoint_hash": finish["checkpoint_hash"],
            "adapter_tensor_hash": finish["adapter_tensor_hash"],
            "archive_hash": archive["archive_hash"],
            "checkpoint_overhead_ms": checkpoint_overhead_ms,
            "transport_bytes": sum(
                int(item.get("payload_bytes") or 0) for item in transport
            ),
            "indexed_transport_lookup_count": sum(
                int(item.get("indexed_lookup_enabled") is True)
                for item in transport
            ),
            "checkpoint_components_validated": bool(
                archive.get("optimizer_state_present") is True
                and archive.get("scheduler_state_present") is True
                and (
                    archive.get("grad_scaler_state_present") is True
                    or (
                        archive.get("runtime_backend") == "jax_tpu"
                        and archive.get("grad_scaler_state_applicable") is False
                        and archive.get("jax_prng_state_present") is True
                    )
                )
                and archive.get("rng_state_present") is True
                and archive.get("tensor_payload_validation_enabled") is True
                and archive.get("archive_paths_validated") is True
            ),
            "checkpoint_component_hashes_hash": archive.get(
                "component_hashes_hash"
            ),
            "submission_idempotent": submission.get("idempotent") is True,
            "global_commit_created": submission.get("global_commit_created") is True,
            "transport": transport,
            "activation_values_public": False,
            "gradient_values_public": False,
            "public_artifact_safe": True,
        }
    except BaseException:
        try:
            process.call("abort_step", timeout=30.0)
        except BaseException:
            pass
        raise


def run_heterogeneous_miner(
    *,
    coordinator_url: str,
    coordinator_token: str,
    run_id: str,
    miner_id_hash: str,
    registration_nonce: str,
    training_manifest: dict[str, Any],
    config: dict[str, Any],
    tokenized_payload: dict[str, Any],
    private_root: str | Path,
    device_policy: str = "auto",
    cuda_devices: list[int] | None = None,
    max_stage_count: int = 0,
    max_steps_per_session: int = 0,
    wait_timeout: float = 1800.0,
    operation_timeout: float = 1800.0,
    heartbeat_interval_seconds: float = 5.0,
    drain_requested: Callable[[], bool] | None = None,
    hf_token: str = "",
    attached_model_root: str | Path | None = None,
    run_microbenchmark: bool = True,
    transport_optimization_after_step: int = -1,
) -> dict[str, Any]:
    """Join, execute assigned stages and drain only after an atomic barrier."""

    started = time.time()
    manifest = validate_training_manifest(training_manifest)
    capability = miner_capability(
        miner_id_hash=miner_id_hash,
        device_policy=device_policy,
        cuda_devices=cuda_devices,
        max_stage_count=max_stage_count,
        run_microbenchmark=run_microbenchmark,
    )
    policy = str(device_policy)
    if policy == "auto":
        policy = "cuda" if capability["gpus"] else "cpu"
    accelerator = (
        "mixed" if policy == "mixed" else "tpu" if policy == "jax_tpu" else policy
    )
    root = Path(private_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    client = ElasticTrainingHTTPClient(
        coordinator_url=coordinator_url,
        coordinator_token=coordinator_token,
        run_id=run_id,
        miner_id_hash=miner_id_hash,
        registration_nonce=registration_nonce,
        supported_stage_ids=[int(item["stage_id"]) for item in manifest["stages"]],
        slot_count=int(capability["max_stage_count"]),
        accelerator=accelerator,
        capability=capability,
        timeout=min(120.0, float(wait_timeout)),
        retry_attempts=8,
        heartbeat_interval_seconds=float(heartbeat_interval_seconds),
        persistent_http_after_step=int(transport_optimization_after_step),
    )
    registration = client.register()
    client.start_heartbeat()
    processes: dict[int, HeterogeneousStageProcessClient] = {}
    process_state: dict[int, dict[str, Any]] = {}
    stage_process_ready_history: list[dict[str, Any]] = []
    shard_reports: dict[int, dict[str, Any]] = {}
    steps = []
    barriers = []
    blockers = []
    recoverable_epoch_events = []
    telemetry_reports = []
    telemetry_sampling_skips = []
    graceful_drain = False
    bounded_operation_timeout = min(
        float(wait_timeout), max(30.0, float(operation_timeout))
    )
    try:
        response = registration
        while True:
            if response.get("runtime_state") == "completed":
                break
            current_assignments = [dict(item) for item in response.get("assignments") or []]
            if not current_assignments:
                if drain_requested is not None and drain_requested():
                    graceful_drain = True
                    break
                time.sleep(0.5)
                response = client.heartbeat()
                continue
            client.set_current_step(
                int(current_assignments[0].get("target_step") or 0)
            )
            assigned_ids = {int(item["stage_id"]) for item in current_assignments}
            for stage_id in list(processes):
                if stage_id not in assigned_ids:
                    processes.pop(stage_id).stop()
                    process_state.pop(stage_id, None)
            for assignment in sorted(
                current_assignments, key=lambda item: int(item["stage_id"])
            ):
                stage_id = int(assignment["stage_id"])
                desired = {
                    "base_step": int(assignment["base_step"]),
                    "placement_generation": int(assignment["placement_generation"]),
                    "device_id": str(assignment["device_id"]),
                }
                state = process_state.get(stage_id)
                needs_restart = (
                    stage_id not in processes
                    or state is None
                    or int(state["local_step"]) != desired["base_step"]
                    or int(state["placement_generation"])
                    != desired["placement_generation"]
                    or str(state["device_id"]) != desired["device_id"]
                )
                if not needs_restart:
                    continue
                if stage_id in processes:
                    processes.pop(stage_id).stop()
                checkpoint_dir = root / "checkpoints" / f"stage-{stage_id}"
                client.heartbeat()
                if int(assignment["base_step"]) == 0:
                    shutil.rmtree(checkpoint_dir, ignore_errors=True)
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    resume = False
                else:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    client.download_checkpoint(
                        assignment,
                        checkpoint_dir=checkpoint_dir,
                        training_manifest=manifest,
                    )
                    resume = True
                shard = root / "stage-shards" / f"stage{stage_id}.safetensors"
                if not shard.is_file():
                    shard_report = materialize_qwen_stage_shard(
                        manifest,
                        stage_id=stage_id,
                        output_path=shard,
                        token=hf_token,
                        source_root=attached_model_root,
                    )
                    shard_reports[stage_id] = {
                        key: value
                        for key, value in shard_report.items()
                        if key != "shard_path"
                    }
                client.heartbeat()
                process = HeterogeneousStageProcessClient(
                    training_manifest=manifest,
                    config=config,
                    stage_id=stage_id,
                    shard_path=shard,
                    checkpoint_dir=checkpoint_dir,
                    device=assignment["device_id"],
                    placement_generation=assignment["placement_generation"],
                    resume=resume,
                    ready_timeout=wait_timeout,
                    keepalive=client.heartbeat,
                    keepalive_interval_seconds=max(
                        1.0, float(heartbeat_interval_seconds)
                    ),
                    require_tpu=assignment["device_type"] == "jax_tpu",
                    expected_tpu_devices=int(
                        next(
                            (
                                item.get("device_count")
                                for item in capability.get("tpu_groups") or []
                                if item.get("device_id") == assignment["device_id"]
                            ),
                            8,
                        )
                    ),
                )
                processes[stage_id] = process
                ready = process.public_ready()
                stage_process_ready_history.append(
                    {
                        **ready,
                        "assignment_base_step": int(assignment["base_step"]),
                        "assignment_target_step": int(assignment["target_step"]),
                    }
                )
                process_state[stage_id] = {
                    **desired,
                    "local_step": int(assignment["base_step"]),
                    "ready": ready,
                    "checkpoint_dir": checkpoint_dir,
                }
            epoch_id = int(current_assignments[0]["epoch_id"])
            try:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(current_assignments)
                ) as executor:
                    futures = [
                        executor.submit(
                            _execute_stage_epoch,
                            client=client,
                            process=processes[int(assignment["stage_id"])],
                            assignment=assignment,
                            manifest=manifest,
                            tokenized=tokenized_payload,
                            checkpoint_dir=process_state[int(assignment["stage_id"])][
                                "checkpoint_dir"
                            ],
                            wait_timeout=bounded_operation_timeout,
                        )
                        for assignment in current_assignments
                    ]
                    stage_results = [future.result() for future in futures]
            except BaseException as exc:
                code = str(exc)
                failure_class = (
                    "device_oom"
                    if "out of memory" in code.lower() or "cuda_oom" in code.lower()
                    else "network_timeout"
                    if "timeout" in code.lower() or "transport" in code.lower()
                    else "runtime_error"
                )
                for assignment in current_assignments:
                    try:
                        client.report_device_failure(
                            device_id=str(assignment["device_id"]),
                            failure_class=failure_class,
                        )
                    except BaseException:
                        pass
                if "out of memory" in code.lower() or "cuda_oom" in code.lower():
                    for assignment in current_assignments:
                        try:
                            client.report_stage_runtime(
                                assignment, event_type="oom"
                            )
                            break
                        except BaseException:
                            continue
                recoverable_code = _recoverable_epoch_error(exc)
                if recoverable_code:
                    recoverable_epoch_events.append(
                        {
                            "error_code": recoverable_code,
                            "epoch_id": epoch_id,
                            "placement_generation": int(
                                current_assignments[0]["placement_generation"]
                            ),
                            "stage_ids": sorted(
                                int(item["stage_id"])
                                for item in current_assignments
                            ),
                            "recovered_by_assignment_refresh": True,
                        }
                    )
                else:
                    blockers.append(
                        "heterogeneous_stage_epoch_failed:" + type(exc).__name__
                    )
                response = client.assignments()
                continue
            barrier = client.wait_barrier(
                epoch_id=epoch_id, timeout=bounded_operation_timeout
            )
            barriers.append(
                {
                    "epoch_id": epoch_id,
                    "target_step": int(current_assignments[0]["target_step"]),
                    "barrier_state": barrier["state"],
                    "committed_step": int(barrier["committed_step"]),
                }
            )
            if barrier["state"] != "committed":
                response = client.assignments()
                continue
            for assignment in current_assignments:
                process_state[int(assignment["stage_id"])]["local_step"] = int(
                    assignment["target_step"]
                )
            steps.append(
                {
                    "target_step": int(current_assignments[0]["target_step"]),
                    "placement_generation": int(
                        current_assignments[0]["placement_generation"]
                    ),
                    "stages": sorted(stage_results, key=lambda item: item["stage_id"]),
                }
            )
            committed_step = int(barrier["committed_step"])
            telemetry_interval = _committed_telemetry_sampling_interval(
                committed_step=committed_step,
                optimization_after_step=int(transport_optimization_after_step),
            )
            if committed_step % telemetry_interval == 0:
                telemetry_reports.extend(
                    _report_committed_telemetry(
                        client=client,
                        capability=capability,
                        manifest=manifest,
                        assignments=current_assignments,
                        stage_results=stage_results,
                        committed_step=committed_step,
                    )
                )
            else:
                telemetry_sampling_skips.append(
                    {
                        "schema": "crowdtensor_heterogeneous_telemetry_sampling_v1",
                        "committed_step": committed_step,
                        "sampling_interval_steps": telemetry_interval,
                        "heartbeat_and_lease_renewal_continued": True,
                        "public_artifact_safe": True,
                    }
                )
            should_drain = bool(
                (drain_requested is not None and drain_requested())
                or (
                    int(max_steps_per_session) > 0
                    and len(steps) >= int(max_steps_per_session)
                )
            )
            if should_drain:
                graceful_drain = True
                break
            response = client.assignments()
        try:
            offline = client.offline()
        except BaseException:
            offline = {"offline_transition_applied": False}
        statuses = []
        for process in processes.values():
            try:
                statuses.append(
                    process.call("status", timeout=bounded_operation_timeout)
                )
            except BaseException:
                pass
        report = {
            "schema": MINER_REPORT_SCHEMA,
            "ok": not blockers,
            "run_id_hash": stable_hash({"run_id": run_id}),
            "miner_id_hash": miner_id_hash,
            "device_policy": policy,
            "capability": capability,
            "training_manifest_hash": manifest["content_hash"],
            "assigned_stage_ids": sorted(process_state),
            "steps": steps,
            "steps_completed": len(steps),
            "barriers": barriers,
            "all_completed_barriers_committed": all(
                item["barrier_state"] == "committed" for item in barriers
            ),
            "positive_lora_gradient_norms": all(
                float(stage["lora_gradient_norm"]) > 0
                for step in steps
                for stage in step["stages"]
            ),
            "optimizer_and_scheduler_steps_applied": all(
                stage["optimizer_step_applied"]
                and stage["scheduler_step_applied"]
                for step in steps
                for stage in step["stages"]
            ),
            "central_checkpoint_restore_count": sum(
                int(item.get("resumed") is True)
                for item in stage_process_ready_history
            ),
            "stage_process_ready": [
                dict(process_state[stage_id].get("ready") or {})
                for stage_id in sorted(process_state)
            ],
            "stage_process_ready_history": stage_process_ready_history,
            "graceful_drain_applied": graceful_drain,
            "offline_transition_applied": offline.get("offline_transition_applied")
            is True,
            "stage_process_statuses": statuses,
            "shard_reports": [shard_reports[key] for key in sorted(shard_reports)],
            "operation_timeout_seconds": bounded_operation_timeout,
            "recoverable_epoch_events": recoverable_epoch_events,
            "recoverable_epoch_event_count": len(recoverable_epoch_events),
            "device_telemetry_reports": telemetry_reports,
            "device_telemetry_report_count": len(telemetry_reports),
            "device_telemetry_sampling_skips": telemetry_sampling_skips,
            "device_telemetry_sampling_skip_count": len(
                telemetry_sampling_skips
            ),
            "telemetry_interval_after_optimization_steps": (
                TELEMETRY_INTERVAL_AFTER_OPTIMIZATION
            ),
            "telemetry_sampling_optimization_enabled": bool(
                int(transport_optimization_after_step) >= 0
            ),
            "all_device_telemetry_reports_accepted": bool(telemetry_reports)
            and all(item.get("ok") is True for item in telemetry_reports),
            "blockers": blockers,
            "client": client.public_report(),
            "elapsed_seconds": time.time() - started,
            "credential_values_public": False,
            "coordinator_url_public": False,
            "raw_training_text_public": False,
            "token_ids_public": False,
            "activation_values_public": False,
            "gradient_values_public": False,
            "checkpoint_tensor_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        report["content_hash"] = stable_hash(report)
        return report
    finally:
        client.stop_heartbeat()
        client.close()
        for process in processes.values():
            try:
                process.stop()
            except BaseException:
                process.force_stop()
        gc.collect()
