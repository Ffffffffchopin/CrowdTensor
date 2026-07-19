#!/usr/bin/env python3
"""Run deterministic local fault-governance checks for Training Production."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from crowdtensor.elastic_checkpoint_storage import (
    MirroredCheckpointBlobStore,
    checkpoint_blob_store_from_configuration,
)
from crowdtensor.elastic_training_runtime import ElasticTrainingRuntime
from crowdtensor.heterogeneous_training_manifest import (
    qwen25_7b_lora_tpu_manifest,
    stable_hash,
)
from crowdtensor.heterogeneous_training_production import (
    default_production_config,
    execute_bounded_operation,
)
from crowdtensor.heterogeneous_training_scheduler import (
    CAPABILITY_SCHEMA,
    TPU_CAPABILITY_SCHEMA,
    validate_miner_capability,
)
from scripts.training_cuda_kaggle_common import public_safety_errors


SCHEMA = "crowdtensor_heterogeneous_training_production_fault_probe_v1"
GIB = 1024**3


def _digest(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _capability(name: str, *, kind: str, max_stages: int = 1) -> dict[str, Any]:
    identity = _digest(name)
    value = {
        "schema": TPU_CAPABILITY_SCHEMA if kind == "tpu" else CAPABILITY_SCHEMA,
        "miner_id_hash": identity,
        "cpu": {
            "device_id": "cpu",
            "physical_core_count": 8,
            "logical_core_count": 16,
            "total_memory_bytes": 64 * GIB,
            "free_memory_bytes": 60 * GIB,
            "supported_dtypes": ["bfloat16", "float32"],
            "throughput_units_per_second": 100_000_000,
            "microbenchmark_latency_ms": 1.0,
            "utilization_fraction": 0.05,
        },
        "gpus": (
            [
                {
                    "device_id": "cuda:0",
                    "device_index": 0,
                    "device_name_hash": _digest(name + ":cuda"),
                    "total_memory_bytes": 16 * GIB,
                    "free_memory_bytes": 16 * GIB,
                    "compute_capability": "7.5",
                    "supported_dtypes": ["float16", "float32"],
                    "throughput_units_per_second": 1_500_000_000,
                    "utilization_fraction": 0.05,
                    "raw_device_name_public": False,
                }
            ]
            if kind == "cuda"
            else []
        ),
        "network": {
            "measured_bandwidth_bytes_per_second": 100 * 1024**2,
            "measured_round_trip_latency_ms": 10.0,
            "measurement_count": 3,
        },
        "stage_profiles": [],
        "current_load_fraction": 0.05,
        "max_stage_count": int(max_stages),
        "single_gpu_miner": kind == "cuda",
        "multi_gpu_miner": False,
        "cpu_stage_supported": kind == "cpu",
        "raw_device_names_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    if kind == "tpu":
        value["tpu_groups"] = [
            {
                "device_id": "jax_tpu:0",
                "runtime_backend": "jax",
                "accelerator_type": "TPU v5e",
                "device_kind_hash": _digest("TPU v5e"),
                "device_count": 8,
                "mesh_axis_names": ["data"],
                "mesh_shape": [8],
                "total_hbm_bytes": 128 * GIB,
                "free_hbm_bytes": 120 * GIB,
                "per_device_hbm_bytes": 16 * GIB,
                "hbm_measurement_source": "fixture_capacity",
                "supported_dtypes": ["bfloat16", "float32"],
                "throughput_units_per_second": 8_000_000_000,
                "compile_microbenchmark_latency_ms": 2500.0,
                "steady_microbenchmark_latency_ms": 3.0,
                "utilization_fraction": 0.05,
                "all_devices_addressable": True,
                "raw_device_names_public": False,
            }
        ]
        value["jax_tpu_stage_supported"] = True
    return validate_miner_capability(value)


def _register(runtime: ElasticTrainingRuntime, name: str, kind: str) -> dict[str, Any]:
    capability = _capability(name, kind=kind, max_stages=2 if kind == "cpu" else 1)
    return runtime.register_miner(
        miner_id_hash=capability["miner_id_hash"],
        registration_nonce=name + ":nonce",
        supported_stage_ids=[0, 1, 2, 3, 4],
        slot_count=2 if kind == "cpu" else 1,
        accelerator=kind,
        capability=capability,
    )


def _assignments(runtime: ElasticTrainingRuntime, sessions: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows = []
    for session in sessions:
        try:
            response = runtime.assignments(
                session_id=session["session_id"],
                session_token=session["session_token"],
            )
        except ValueError:
            continue
        rows.extend((session, dict(item)) for item in response.get("assignments") or [])
    return sorted(rows, key=lambda item: int(item[1]["stage_id"]))


def run_probe(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    private = output / ".private-runtime"
    shutil.rmtree(private, ignore_errors=True)
    private.mkdir(parents=True, exist_ok=True)
    manifest = qwen25_7b_lora_tpu_manifest(target_steps=100)
    runtime = ElasticTrainingRuntime(
        private / "runtime.sqlite3",
        run_id="training-production-fault-probe",
        target_steps=100,
        microbatches_per_step=1,
        lease_seconds=60.0,
        training_manifest=manifest,
        tensor_lookup_optimization_after_step=20,
    )
    sessions = [
        _register(runtime, f"cuda-{index}", "cuda") for index in range(4)
    ]
    tpu = _register(runtime, "tpu", "tpu")
    cpu = _register(runtime, "cpu-old", "cpu")
    sessions.extend([tpu, cpu])
    initial = runtime.public_status()
    faults: dict[str, Any] = {}

    network_calls = 0

    def network_operation() -> str:
        nonlocal network_calls
        network_calls += 1
        if network_calls < 3:
            raise TimeoutError("private detail")
        return "ok"

    network_result, retry_report = execute_bounded_operation(
        network_operation,
        operation_name="coordinator_transport",
        policy=default_production_config()["fault_governance"],
        sleep=lambda _seconds: None,
    )
    faults["network_timeout"] = {
        "verified": network_result == "ok" and retry_report["retry_count"] == 2,
        "classification": "network_timeout_recovered_by_bounded_retry",
        "retry": retry_report,
    }

    before = _assignments(runtime, sessions)
    old_session, old_assignment = before[0]
    previous_generation = int(old_assignment["placement_generation"])
    runtime.request_rebalance(reason="owner_requested")
    stale_error = ""
    try:
        runtime.report_stage_runtime(
            session_id=old_session["session_id"],
            session_token=old_session["session_token"],
            assignment_token=old_assignment["assignment_token"],
            placement_generation=previous_generation,
            stage_id=int(old_assignment["stage_id"]),
            device_id=str(old_assignment["device_id"]),
            event_type="profile",
            forward_latency_ms=1.0,
            backward_latency_ms=1.0,
        )
    except ValueError as exc:
        stale_error = str(exc)
    faults["duplicate_or_stale_result"] = {
        "verified": stale_error == "elastic_stage_assignment_stale",
        "classification": stale_error,
        "previous_placement_generation": previous_generation,
        "current_placement_generation": int(
            runtime.public_status()["placement_generation"]
        ),
    }

    cpu_offline = runtime.mark_offline(
        session_id=cpu["session_id"], session_token=cpu["session_token"]
    )
    paused = runtime.public_status()
    cpu_replacement = _register(runtime, "cpu-replacement", "cpu")
    sessions.append(cpu_replacement)
    recovered = runtime.public_status()
    faults["worker_crash"] = {
        "verified": bool(
            cpu_offline["offline_transition_applied"]
            and paused["runtime_state"] == "paused_waiting_for_miners"
            and recovered["runtime_state"] == "running"
            and int(recovered["placement_generation"])
            > int(initial["placement_generation"])
        ),
        "classification": "worker_crash_replaced_from_durable_boundary",
        "committed_step_unchanged": int(recovered["committed_step"])
        == int(initial["committed_step"]),
    }

    active = _assignments(runtime, sessions)
    failed_session, failed_assignment = next(
        item for item in active if item[1]["device_type"] == "cuda"
    )
    circuit = {}
    for _ in range(3):
        circuit = runtime.record_device_failure(
            session_id=failed_session["session_id"],
            session_token=failed_session["session_token"],
            device_id=str(failed_assignment["device_id"]),
            failure_class="worker_crash",
            quarantine_threshold=3,
            quarantine_seconds=60.0,
        )
    after_quarantine = runtime.public_status()
    faults["circuit_breaker"] = {
        "verified": bool(
            circuit.get("device_quarantined") is True
            and any(
                item.get("state") == "quarantined"
                for item in after_quarantine["device_health"]
            )
            and after_quarantine["runtime_state"] == "running"
        ),
        "classification": "failing_worker_quarantined_and_replaced",
        "consecutive_failure_threshold": 3,
    }

    store = checkpoint_blob_store_from_configuration(
        {
            "backend": "mirrored",
            "primary": {"backend": "local"},
            "mirror_root": str(private / "checkpoint-mirror"),
        },
        default_root=private / "checkpoint-store",
    )
    assert isinstance(store, MirroredCheckpointBlobStore)
    checkpoint = b"public-safe-content-addressed-checkpoint"
    checkpoint_hash = _digest(checkpoint)
    store.put(checkpoint_hash, checkpoint)
    primary_path = store.primary.path_for_hash(checkpoint_hash)
    primary_path.write_bytes(b"corrupt")
    restored = store.get(checkpoint_hash)
    storage_report = store.public_report()
    faults["checkpoint_corrupt"] = {
        "verified": bool(
            restored == checkpoint
            and store.primary.get(checkpoint_hash) == checkpoint
            and storage_report["fallback_read_count"] == 1
            and storage_report["primary_repair_count"] == 1
        ),
        "classification": "checkpoint_primary_corrupt_mirror_repair_verified",
        "archive_hash": checkpoint_hash,
        "storage": storage_report,
    }

    first_start = runtime.record_coordinator_start(instance_id_hash=_digest("first"))
    before_restart = runtime.public_status()
    reopened = ElasticTrainingRuntime.open_existing(
        private / "runtime.sqlite3",
        run_id=runtime.run_id,
        lease_seconds=60.0,
    )
    second_start = reopened.record_coordinator_start(instance_id_hash=_digest("second"))
    after_restart = reopened.public_status()
    faults["coordinator_restart"] = {
        "verified": bool(
            first_start["coordinator_generation"] == 1
            and second_start["coordinator_generation"] == 2
            and second_start["persistent_journal_reopened"] is True
            and before_restart["committed_steps"] == after_restart["committed_steps"]
            and before_restart["placement_generation"]
            == after_restart["placement_generation"]
        ),
        "classification": "coordinator_journal_reopened_without_progress_loss",
        "coordinator_generation": int(after_restart["coordinator_generation"]),
    }

    cleanup_calls = 0

    def cleanup_operation() -> bool:
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 1:
            raise OSError("private detail")
        return True

    cleanup_result, cleanup_retry = execute_bounded_operation(
        cleanup_operation,
        operation_name="resource_cleanup",
        policy=default_production_config()["fault_governance"],
        sleep=lambda _seconds: None,
    )
    faults["cleanup_retry"] = {
        "verified": cleanup_result is True and cleanup_retry["retry_count"] == 1,
        "classification": "cleanup_retry_succeeded_within_bound",
        "retry": cleanup_retry,
    }

    runtime_cleanup = reopened.cleanup()
    all_verified = all(
        item.get("verified") is True for item in faults.values()
    )
    report = {
        "schema": SCHEMA,
        "ok": all_verified,
        "fault_injection_suite_ready": all_verified,
        "faults": faults,
        "fault_classifications": sorted(
            str(item["classification"]) for item in faults.values()
        ),
        "initial_provider_coverage": sorted(
            set(initial["placement_plan"].get("accepted_device_types") or [])
        ),
        "generation_fencing_verified": faults["duplicate_or_stale_result"][
            "verified"
        ],
        "lease_reclaim_verified": faults["worker_crash"]["verified"],
        "circuit_breaker_verified": faults["circuit_breaker"]["verified"],
        "checkpoint_fallback_verified": faults["checkpoint_corrupt"]["verified"],
        "coordinator_journal_recovery_verified": faults["coordinator_restart"][
            "verified"
        ],
        "bounded_retry_verified": faults["network_timeout"]["verified"],
        "cleanup": {
            "runtime_cleaned": runtime_cleanup["runtime_state"] == "cleaned",
            "active_miner_leases_revoked": runtime_cleanup["live_miner_count"] == 0,
            "temporary_private_runtime_removed": True,
            "live_resources_left_running": False,
        },
        "credential_values_public": False,
        "session_tokens_public": False,
        "assignment_tokens_public": False,
        "checkpoint_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    shutil.rmtree(private, ignore_errors=True)
    report["cleanup"]["temporary_private_runtime_removed"] = not private.exists()
    safety = public_safety_errors(report)
    report["public_artifact_safe"] = not safety
    report["public_safety_errors"] = safety
    report["ok"] = bool(report["ok"] and not safety)
    report["fault_injection_suite_ready"] = report["ok"]
    report["content_hash"] = stable_hash(report)
    _write(output / "training_heterogeneous_production_fault_probe.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_probe(args.output_dir)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            "training_heterogeneous_production_fault_probe "
            f"ready={report['fault_injection_suite_ready']}"
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
