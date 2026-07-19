import hashlib

import pytest
import torch

from crowdtensor.elastic_training_runtime import ElasticTrainingRuntime
from crowdtensor.heterogeneous_tensor_transport import (
    decode_tensor_payload,
    encode_tensor_message,
)
from crowdtensor.heterogeneous_training_manifest import (
    qwen25_7b_lora_manifest,
    qwen25_7b_lora_tpu_manifest,
)
from crowdtensor.heterogeneous_training_scheduler import (
    CAPABILITY_SCHEMA,
    TPU_CAPABILITY_SCHEMA,
    validate_miner_capability,
)


GIB = 1024**3


def digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def capability(name: str, *, gpu: bool, max_stages: int = 1) -> dict:
    identity = digest(name)
    value = {
        "schema": CAPABILITY_SCHEMA,
        "miner_id_hash": identity,
        "cpu": {
            "device_id": "cpu",
            "physical_core_count": 8,
            "logical_core_count": 16,
            "total_memory_bytes": 32 * GIB,
            "free_memory_bytes": 30 * GIB,
            "supported_dtypes": ["bfloat16", "float32"],
            "throughput_units_per_second": 80_000_000,
            "microbenchmark_latency_ms": 2.0,
            "utilization_fraction": 0.0,
        },
        "gpus": (
            [
                {
                    "device_id": "cuda:0",
                    "device_index": 0,
                    "device_name_hash": digest(name + ":gpu"),
                    "total_memory_bytes": 16 * GIB,
                    "free_memory_bytes": 16 * GIB,
                    "compute_capability": "7.5",
                    "supported_dtypes": ["float16", "float32"],
                    "throughput_units_per_second": 1_500_000_000,
                    "utilization_fraction": 0.0,
                    "raw_device_name_public": False,
                }
            ]
            if gpu
            else []
        ),
        "network": {
            "measured_bandwidth_bytes_per_second": 100 * 1024**2,
            "measured_round_trip_latency_ms": 10.0,
            "measurement_count": 3,
        },
        "stage_profiles": [],
        "current_load_fraction": 0.0,
        "max_stage_count": max_stages,
        "single_gpu_miner": gpu,
        "multi_gpu_miner": False,
        "cpu_stage_supported": not gpu,
        "raw_device_names_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    return validate_miner_capability(value)


def register(
    runtime: ElasticTrainingRuntime,
    name: str,
    *,
    gpu: bool,
    max_stages: int = 1,
) -> dict:
    return runtime.register_miner(
        miner_id_hash=digest(name),
        registration_nonce=name + ":nonce",
        supported_stage_ids=[0, 1, 2, 3, 4],
        slot_count=max_stages,
        accelerator="cuda" if gpu else "cpu",
        capability=capability(name, gpu=gpu, max_stages=max_stages),
    )


def tpu_capability(name: str) -> dict:
    value = capability(name, gpu=False)
    value.pop("content_hash")
    value["schema"] = TPU_CAPABILITY_SCHEMA
    value["cpu_stage_supported"] = False
    value["tpu_groups"] = [
        {
            "device_id": "jax_tpu:0",
            "runtime_backend": "jax",
            "accelerator_type": "TPU v5e",
            "device_kind_hash": digest("TPU v5e"),
            "device_count": 8,
            "mesh_axis_names": ["model"],
            "mesh_shape": [8],
            "total_hbm_bytes": 128 * GIB,
            "free_hbm_bytes": 120 * GIB,
            "per_device_hbm_bytes": 16 * GIB,
            "hbm_measurement_source": "jax_memory_stats",
            "supported_dtypes": ["bfloat16", "float32"],
            "throughput_units_per_second": 8_000_000_000,
            "compile_microbenchmark_latency_ms": 2500.0,
            "steady_microbenchmark_latency_ms": 4.0,
            "utilization_fraction": 0.05,
            "all_devices_addressable": True,
            "raw_device_names_public": False,
        }
    ]
    value["jax_tpu_stage_supported"] = True
    return validate_miner_capability(value)


def register_tpu(runtime: ElasticTrainingRuntime, name: str) -> dict:
    return runtime.register_miner(
        miner_id_hash=digest(name),
        registration_nonce=name + ":nonce",
        supported_stage_ids=[0, 1, 2, 3, 4],
        slot_count=1,
        accelerator="tpu",
        capability=tpu_capability(name),
    )


def assignments(runtime: ElasticTrainingRuntime, sessions: list[dict]):
    result = []
    for session in sessions:
        response = runtime.assignments(
            session_id=session["session_id"],
            session_token=session["session_token"],
        )
        result.extend((session, item) for item in response["assignments"])
    return sorted(result, key=lambda item: item[1]["stage_id"])


def test_idempotent_registration_refreshes_dynamic_capability(tmp_path) -> None:
    runtime = ElasticTrainingRuntime(
        tmp_path / "heterogeneous.sqlite3",
        run_id="capability-refresh-run",
        target_steps=6,
        microbatches_per_step=1,
        training_manifest=qwen25_7b_lora_manifest(),
        lease_seconds=60.0,
    )
    original = capability("refreshing-gpu", gpu=True)
    first = runtime.register_miner(
        miner_id_hash=digest("refreshing-gpu"),
        registration_nonce="stable-recovery-nonce",
        supported_stage_ids=[0, 1, 2, 3, 4],
        slot_count=1,
        accelerator="cuda",
        capability=original,
    )
    refreshed = dict(original)
    refreshed.pop("content_hash", None)
    refreshed["cpu"] = dict(refreshed["cpu"])
    refreshed["cpu"]["free_memory_bytes"] -= GIB
    refreshed["current_load_fraction"] = 0.25
    refreshed = validate_miner_capability(refreshed)

    second = runtime.register_miner(
        miner_id_hash=digest("refreshing-gpu"),
        registration_nonce="stable-recovery-nonce",
        supported_stage_ids=[0, 1, 2, 3, 4],
        slot_count=1,
        accelerator="cuda",
        capability=refreshed,
    )

    assert second["session_id"] == first["session_id"]
    assert second["session_token"] == first["session_token"]
    assert second["registration_idempotent"] is True
    assert second["capability_refreshed"] is True
    status = runtime.public_status()
    online = [item for item in status["miners"] if item["state"] == "online"]
    assert len(online) == 1
    assert online[0]["capability"]["content_hash"] == refreshed["content_hash"]
    assert any(
        item["operation"] == "miner_capability_refreshed"
        for item in status["events"]
    )


def test_same_job_mixed_placement_tensor_transport_and_oom_reassignment(tmp_path) -> None:
    manifest = qwen25_7b_lora_manifest()
    runtime = ElasticTrainingRuntime(
        tmp_path / "heterogeneous.sqlite3",
        run_id="heterogeneous-run",
        target_steps=6,
        microbatches_per_step=1,
        training_manifest=manifest,
        lease_seconds=60.0,
    )
    sessions = [
        register(runtime, f"gpu-{index}", gpu=True) for index in range(4)
    ]
    sessions.append(register(runtime, "cpu", gpu=False, max_stages=2))

    current = assignments(runtime, sessions)
    assert [item["stage_id"] for _session, item in current] == [0, 1, 2, 3, 4]
    assert [item["device_type"] for _session, item in current] == [
        "cuda",
        "cuda",
        "cuda",
        "cuda",
        "cpu",
    ]
    assert {item["placement_generation"] for _session, item in current} == {1}
    initial_status = runtime.public_status()
    assert initial_status["placement_plan"]["single_gpu_miner_participating"] is True
    assert initial_status["placement_plan"]["cpu_miner_participating"] is True
    assert initial_status["training_manifest_hash"] == manifest["content_hash"]

    source_session, source = current[0]
    target_session, target = current[1]
    envelope, chunks = encode_tensor_message(
        {"activation": torch.arange(64, dtype=torch.float32).reshape(1, 4, 16)},
        job_id=runtime.run_id,
        manifest_hash=manifest["content_hash"],
        global_step=1,
        microbatch_id=0,
        source_stage_id=0,
        target_stage_id=1,
        direction="forward_activation",
        placement_generation=1,
        assignment_token_hash=source["assignment_token_hash"],
        chunk_bytes=128,
    )
    runtime.begin_tensor_message(
        session_id=source_session["session_id"],
        session_token=source_session["session_token"],
        assignment_token=source["assignment_token"],
        envelope=envelope,
    )
    for index, chunk in enumerate(chunks):
        runtime.put_tensor_chunk(
            session_id=source_session["session_id"],
            session_token=source_session["session_token"],
            assignment_token=source["assignment_token"],
            message_id=envelope["message_id"],
            chunk_index=index,
            value=chunk,
        )
    lookup = runtime.find_tensor_message(
        session_id=target_session["session_id"],
        session_token=target_session["session_token"],
        assignment_token=target["assignment_token"],
        global_step=1,
        microbatch_id=0,
        source_stage_id=0,
        target_stage_id=1,
        direction="forward_activation",
        placement_generation=1,
    )
    assert lookup["found"] is True
    assert lookup["status"]["complete"] is True
    downloaded = []
    for index in range(envelope["chunk_count"]):
        chunk, _metadata = runtime.read_tensor_chunk(
            session_id=target_session["session_id"],
            session_token=target_session["session_token"],
            assignment_token=target["assignment_token"],
            message_id=envelope["message_id"],
            chunk_index=index,
        )
        downloaded.append(chunk)
    decoded = decode_tensor_payload(b"".join(downloaded), envelope)
    assert torch.equal(
        decoded["activation"],
        torch.arange(64, dtype=torch.float32).reshape(1, 4, 16),
    )

    profile = runtime.report_stage_runtime(
        session_id=source_session["session_id"],
        session_token=source_session["session_token"],
        assignment_token=source["assignment_token"],
        placement_generation=1,
        stage_id=0,
        device_id="cuda:0",
        event_type="profile",
        forward_latency_ms=12.0,
        backward_latency_ms=18.0,
        peak_memory_bytes=10 * GIB,
        sample_count=3,
    )
    assert profile["profile_accepted"] is True
    oom = runtime.report_stage_runtime(
        session_id=source_session["session_id"],
        session_token=source_session["session_token"],
        assignment_token=source["assignment_token"],
        placement_generation=1,
        stage_id=0,
        device_id="cuda:0",
        event_type="oom",
    )
    assert oom["rebalance_triggered"] is True
    assert oom["placement_generation"] == 2

    replacement = assignments(runtime, sessions)
    assert {item["placement_generation"] for _session, item in replacement} == {2}
    assert sum(item["device_type"] == "cpu" for _session, item in replacement) == 2
    assert sum(item["device_type"] == "cuda" for _session, item in replacement) == 3
    status = runtime.public_status()
    assert status["epochs"][0]["state"] == "aborted"
    assert status["epochs"][0]["abort_reason"] == "device_oom"
    assert status["placement_plan"]["rebalance_reason"] == "device_oom"
    assert status["committed_step"] == 0

    with pytest.raises(ValueError, match="stage_assignment_stale"):
        runtime.find_tensor_message(
            session_id=target_session["session_id"],
            session_token=target_session["session_token"],
            assignment_token=target["assignment_token"],
            global_step=1,
            microbatch_id=0,
            source_stage_id=0,
            target_stage_id=1,
            direction="forward_activation",
            placement_generation=1,
        )

    reopened = ElasticTrainingRuntime.open_existing(
        tmp_path / "heterogeneous.sqlite3",
        run_id="heterogeneous-run",
        lease_seconds=60.0,
    )
    reopened_status = reopened.public_status()
    assert reopened_status["placement_generation"] == 2
    assert reopened_status["training_manifest_hash"] == manifest["content_hash"]


def test_tpu_profile_oom_pause_and_capability_refresh_replacement(tmp_path) -> None:
    manifest = qwen25_7b_lora_tpu_manifest()
    runtime = ElasticTrainingRuntime(
        tmp_path / "heterogeneous-tpu.sqlite3",
        run_id="heterogeneous-tpu-run",
        target_steps=6,
        microbatches_per_step=1,
        training_manifest=manifest,
        lease_seconds=60.0,
    )
    sessions = [register(runtime, f"gpu-{index}", gpu=True) for index in range(3)]
    sessions.append(register_tpu(runtime, "tpu"))
    sessions.append(register(runtime, "cpu", gpu=False))
    current = assignments(runtime, sessions)

    assert [item["device_type"] for _session, item in current] == [
        "cuda",
        "cuda",
        "jax_tpu",
        "cuda",
        "cpu",
    ]
    tpu_session, tpu_assignment = current[2]
    profile = runtime.report_stage_runtime(
        session_id=tpu_session["session_id"],
        session_token=tpu_session["session_token"],
        assignment_token=tpu_assignment["assignment_token"],
        placement_generation=1,
        stage_id=2,
        device_id="jax_tpu:0",
        event_type="profile",
        forward_latency_ms=20.0,
        backward_latency_ms=40.0,
        compile_latency_ms=2600.0,
        steady_forward_latency_ms=18.0,
        steady_backward_latency_ms=35.0,
        peak_memory_bytes=8 * GIB,
        sample_count=3,
    )
    assert profile["profile_accepted"] is True
    status = runtime.public_status()
    tpu_public = next(
        item for item in status["miners"] if item["accelerator"] == "tpu"
    )
    measured = tpu_public["capability"]["stage_profiles"][0]
    assert measured["compile_latency_ms"] == pytest.approx(2600.0)
    assert measured["steady_forward_latency_ms"] == pytest.approx(18.0)
    assert status["placement_plan"]["tpu_miner_participating"] is True

    oom = runtime.report_stage_runtime(
        session_id=tpu_session["session_id"],
        session_token=tpu_session["session_token"],
        assignment_token=tpu_assignment["assignment_token"],
        placement_generation=1,
        stage_id=2,
        device_id="jax_tpu:0",
        event_type="oom",
    )
    assert oom["rebalance_triggered"] is True
    assert runtime.public_status()["runtime_state"] == "paused_waiting_for_miners"

    refreshed = runtime.register_miner(
        miner_id_hash=digest("tpu"),
        registration_nonce="tpu:nonce",
        supported_stage_ids=[0, 1, 2, 3, 4],
        slot_count=1,
        accelerator="tpu",
        capability=tpu_capability("tpu"),
    )
    assert refreshed["registration_idempotent"] is True
    assert refreshed["capability_refreshed"] is True
    replacement = assignments(runtime, sessions)
    assert {item["placement_generation"] for _session, item in replacement} == {2}
    assert replacement[2][1]["device_type"] == "jax_tpu"
    with pytest.raises(ValueError, match="stage_assignment_stale"):
        runtime.report_stage_runtime(
            session_id=tpu_session["session_id"],
            session_token=tpu_session["session_token"],
            assignment_token=tpu_assignment["assignment_token"],
            placement_generation=1,
            stage_id=2,
            device_id="jax_tpu:0",
            event_type="profile",
            forward_latency_ms=1.0,
            backward_latency_ms=1.0,
        )
