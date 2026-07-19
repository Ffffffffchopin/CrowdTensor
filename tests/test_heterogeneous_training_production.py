import hashlib
import json
from copy import deepcopy

import pytest

from crowdtensor.cli import main, parse_args
from crowdtensor.elastic_training_runtime import ElasticTrainingRuntime
from crowdtensor.elastic_checkpoint_storage import (
    MirroredCheckpointBlobStore,
    checkpoint_blob_store_from_configuration,
)
from crowdtensor.heterogeneous_training_manifest import (
    qwen25_7b_lora_tpu_manifest,
)
from crowdtensor.heterogeneous_training_production import (
    HeterogeneousTrainingProductionController,
    build_production_plan,
    compare_performance_windows,
    default_production_config,
    production_manifest,
    retry_delay_seconds,
    validate_production_config,
)
from crowdtensor.heterogeneous_training_scheduler import (
    CAPABILITY_SCHEMA,
    build_placement_plan,
    validate_miner_capability,
)


GIB = 1024**3


def digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def capability(name: str, *, gpu: bool, max_stages: int = 1) -> dict:
    value = {
        "schema": CAPABILITY_SCHEMA,
        "miner_id_hash": digest(name),
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


def small_manifest() -> dict:
    manifest = qwen25_7b_lora_tpu_manifest(target_steps=100)
    manifest.pop("content_hash")
    manifest["scheduler"]["required_device_types"] = ["cpu", "cuda"]
    manifest["scheduler"]["max_stages_per_miner"] = 5
    for stage in manifest["stages"]:
        stage["estimated_weight_bytes"] = 64 * 1024**2
        stage["estimated_compute_units"] = 10_000_000.0
        stage["allowed_device_types"] = ["cpu", "cuda"]
        stage["preferred_device_type"] = "cuda"
    manifest["stages"][-1]["preferred_device_type"] = "cpu"
    manifest["precision"].pop("jax_tpu_compute_dtype")
    manifest["scheduler"].pop("tpu_memory_reserve_bytes")
    manifest["scheduler"].pop("tpu_compile_cost_weight")
    manifest["scheduler"].pop("tpu_steady_state_cost_weight")
    manifest["schema"] = "crowdtensor_heterogeneous_training_manifest_v1"
    from crowdtensor.heterogeneous_training_manifest import validate_training_manifest

    return validate_training_manifest(manifest)


def test_production_config_is_bounded_and_pins_the_soak_contract() -> None:
    config = validate_production_config(default_production_config())
    manifest = production_manifest(config)
    plan = build_production_plan(config)

    assert config["soak"]["target_steps"] == 100
    assert config["soak"]["minimum_duration_seconds"] == 3600
    assert config["acquisition"]["maximum_window_seconds"] == 43200
    assert config["acquisition"]["maximum_full_live_gate_seconds"] == 21600
    assert manifest["training"]["target_steps"] == 100
    assert plan["minimum_topology"] == {
        "single_cuda_miners": 3,
        "jax_tpu_v5e8_miners": 1,
        "cpu_miners": 1,
    }


def test_production_config_rejects_weakened_acceptance() -> None:
    value = default_production_config()
    value.pop("content_hash")
    value["soak"]["target_steps"] = 99

    with pytest.raises(ValueError, match="target_steps_invalid"):
        validate_production_config(value)


def test_performance_gate_requires_same_identity_and_fifteen_percent() -> None:
    policy = default_production_config()["performance"]
    baseline = [
        {
            "workload_hash": digest("workload"),
            "topology_hash": digest("topology"),
            "step_throughput_per_second": value,
            "p50_step_latency_seconds": 10.0,
            "p95_step_latency_seconds": 12.0,
        }
        for value in (1.0, 1.01, 0.99)
    ]
    candidate = [
        {
            "workload_hash": digest("workload"),
            "topology_hash": digest("topology"),
            "step_throughput_per_second": value,
            "p50_step_latency_seconds": 8.0,
            "p95_step_latency_seconds": 12.2,
        }
        for value in (1.2, 1.21, 1.19)
    ]

    passed = compare_performance_windows(
        baseline=baseline, candidate=candidate, policy=policy
    )
    candidate[0]["workload_hash"] = digest("smaller-workload")
    rejected = compare_performance_windows(
        baseline=baseline, candidate=candidate, policy=policy
    )

    assert passed["performance_gate_passed"] is True
    assert passed["throughput_improvement_fraction"] >= 0.15
    assert rejected["performance_gate_passed"] is False
    assert rejected["same_workload_verified"] is False


def test_retry_backoff_is_deterministic_bounded_and_jittered() -> None:
    policy = default_production_config()["fault_governance"]
    first = [
        retry_delay_seconds(policy, operation="checkpoint", attempt=index)
        for index in range(policy["retry_attempts"])
    ]
    second = [
        retry_delay_seconds(policy, operation="checkpoint", attempt=index)
        for index in range(policy["retry_attempts"])
    ]

    assert first == second
    assert max(first) <= policy["retry_cap_seconds"]
    assert len(set(first[:4])) == 4
    with pytest.raises(ValueError, match="retry_attempt_invalid"):
        retry_delay_seconds(
            policy, operation="checkpoint", attempt=policy["retry_attempts"]
        )


def test_mirrored_checkpoint_recovers_and_repairs_corrupt_primary(tmp_path) -> None:
    store = checkpoint_blob_store_from_configuration(
        {
            "backend": "mirrored",
            "primary": {"backend": "local"},
            "mirror_root": str(tmp_path / "mirror"),
        },
        default_root=tmp_path / "store",
    )
    assert isinstance(store, MirroredCheckpointBlobStore)
    payload = b"validated-stage-checkpoint"
    archive_hash = "sha256:" + hashlib.sha256(payload).hexdigest()
    store.put(archive_hash, payload)
    primary_path = store.primary.path_for_hash(archive_hash)
    primary_path.write_bytes(b"corrupt")

    recovered = store.get(archive_hash)
    report = store.public_report()

    assert recovered == payload
    assert store.primary.get(archive_hash) == payload
    assert report["fallback_read_count"] == 1
    assert report["primary_repair_count"] == 1
    assert report["automatic_primary_repair"] is True


def test_scheduler_uses_health_memory_performance_and_checkpoint_telemetry() -> None:
    manifest = small_manifest()
    gpu_a = capability("gpu-a", gpu=True, max_stages=5)
    gpu_b = capability("gpu-b", gpu=True, max_stages=5)
    cpu = capability("cpu", gpu=False, max_stages=5)
    initial = build_placement_plan(manifest, [gpu_a, gpu_b, cpu])
    initial_owner = initial["assignments"][0]["miner_id_hash"]
    other = gpu_b if initial_owner == gpu_a["miner_id_hash"] else gpu_a
    telemetry = [
        {
            "miner_id_hash": initial_owner,
            "device_id": "cuda:0",
            "health_score": 0.1,
            "checkpoint_step": 0,
            "free_memory_bytes": 8 * GIB,
            "throughput_units_per_second": 10_000_000,
            "utilization_fraction": 0.9,
            "reported_at": 10.0,
            "consecutive_failures": 2,
        },
        {
            "miner_id_hash": other["miner_id_hash"],
            "device_id": "cuda:0",
            "health_score": 1.0,
            "checkpoint_step": 50,
            "free_memory_bytes": 16 * GIB,
            "throughput_units_per_second": 2_000_000_000,
            "utilization_fraction": 0.1,
            "reported_at": 10.0,
            "consecutive_failures": 0,
        },
    ]

    rebalanced = build_placement_plan(
        manifest,
        [gpu_a, gpu_b, cpu],
        previous_plan=initial,
        reason="performance_rebalance",
        runtime_telemetry=telemetry,
        current_checkpoint_step=50,
    )

    assert rebalanced["runtime_telemetry_considered"] is True
    assert rebalanced["assignments"][0]["miner_id_hash"] == other["miner_id_hash"]
    assert rebalanced["assignments"][0]["checkpoint_lag_steps"] == 0
    assert rebalanced["assignments"][0]["health_penalty"] == 0


def register(runtime: ElasticTrainingRuntime, name: str, *, gpu: bool) -> dict:
    return runtime.register_miner(
        miner_id_hash=digest(name),
        registration_nonce=name + ":nonce",
        supported_stage_ids=[0, 1, 2, 3, 4],
        slot_count=5,
        accelerator="cuda" if gpu else "cpu",
        capability=capability(name, gpu=gpu, max_stages=5),
    )


def test_runtime_persists_control_journal_telemetry_metrics_and_circuit_breaker(
    tmp_path,
) -> None:
    manifest = small_manifest()
    runtime = ElasticTrainingRuntime(
        tmp_path / "production.sqlite3",
        run_id="production-runtime",
        target_steps=100,
        microbatches_per_step=1,
        training_manifest=manifest,
        lease_seconds=60.0,
    )
    gpu = register(runtime, "gpu", gpu=True)
    register(runtime, "cpu", gpu=False)
    first_start = runtime.record_coordinator_start(instance_id_hash=digest("first"))
    second_start = runtime.record_coordinator_start(instance_id_hash=digest("second"))
    telemetry = runtime.report_device_telemetry(
        session_id=gpu["session_id"],
        session_token=gpu["session_token"],
        device_id="cuda:0",
        free_memory_bytes=14 * GIB,
        utilization_fraction=0.2,
        throughput_units_per_second=2_000_000_000,
        checkpoint_step=0,
    )
    paused = runtime.pause()
    paused_again = runtime.pause()
    resumed = runtime.resume()
    for _ in range(3):
        failure = runtime.record_device_failure(
            session_id=gpu["session_id"],
            session_token=gpu["session_token"],
            device_id="cuda:0",
            failure_class="worker_crash",
            quarantine_threshold=3,
            quarantine_seconds=60.0,
        )
    status = runtime.public_status()
    metrics = runtime.metrics_snapshot()
    prometheus = runtime.prometheus_metrics()
    events = runtime.event_tail(limit=100)
    reopened = ElasticTrainingRuntime.open_existing(
        tmp_path / "production.sqlite3",
        run_id="production-runtime",
        lease_seconds=60.0,
    )

    assert first_start["coordinator_generation"] == 1
    assert second_start["coordinator_generation"] == 2
    assert telemetry["telemetry_accepted"] is True
    assert paused["pause_transition_applied"] is True
    assert paused_again["pause_transition_applied"] is False
    assert resumed["resume_transition_applied"] is True
    assert failure["device_quarantined"] is True
    assert any(item["state"] == "quarantined" for item in status["device_health"])
    assert status["coordinator_generation"] == 2
    assert metrics["low_cardinality_labels"] is True
    assert "crowdtensor_training_committed_step" in prometheus
    assert events["event_count"] >= 7
    assert reopened.public_status()["coordinator_generation"] == 2


def test_production_controller_create_is_idempotent_and_private_safe(tmp_path) -> None:
    config = default_production_config()
    manifest = production_manifest(config)
    model_config = {
        "model_type": "qwen2",
        "num_hidden_layers": 28,
        "hidden_size": 3584,
    }
    tokenized = {
        "schema": "crowdtensor_heterogeneous_tokenized_private_v1",
        "training_manifest_hash": manifest["content_hash"],
        "model_id": manifest["model"]["model_id"],
        "model_revision": manifest["model"]["model_revision"],
        "sequence_length": manifest["training"]["sequence_length"],
        "train": [[1] * manifest["training"]["sequence_length"] for _ in range(100)],
        "validation": [[1] * manifest["training"]["sequence_length"]],
    }
    config_path = tmp_path / "config.json"
    tokenized_path = tmp_path / "tokenized.json"
    config_path.write_text(json.dumps(model_config), encoding="utf-8")
    tokenized_path.write_text(json.dumps(tokenized), encoding="utf-8")
    job = tmp_path / "job"

    controller = HeterogeneousTrainingProductionController.create(
        job,
        config=config,
        model_config_path=config_path,
        tokenized_payload_path=tokenized_path,
    )
    repeated = HeterogeneousTrainingProductionController.create(job, config=config)
    status = controller.status()
    cleanup = controller.cleanup()

    assert isinstance(controller, HeterogeneousTrainingProductionController)
    assert isinstance(repeated, HeterogeneousTrainingProductionController)
    assert status["target_steps"] == 100
    assert status["public_artifact_safe"] is True
    assert status["credential_values_public"] is False
    assert status["next_resume_command"] == "crowdtensor train resume <job-dir>"
    assert status["next_resume_command_uses_public_placeholder"] is True
    assert str(job.resolve()) not in json.dumps(status)
    assert cleanup["ok"] is True
    assert cleanup["live_resources_left_running"] is False


def test_production_cli_parses_complete_owner_workflow() -> None:
    assert parse_args(["train", "validate"]).train_action == "validate"
    assert parse_args(["train", "plan"]).train_action == "plan"
    assert parse_args(["train", "start", "job"]).train_action == "start"
    assert parse_args(["train", "run", "job"]).train_action == "run"
    assert parse_args(["train", "pause", "job"]).train_action == "pause"
    assert parse_args(["train", "resume", "job"]).train_action == "resume"
    assert parse_args(["train", "stop", "job"]).train_action == "stop"
    assert parse_args(["train", "cleanup", "job"]).train_action == "cleanup"


def test_production_cli_validate_and_plan_are_public_safe(capsys) -> None:
    with pytest.raises(SystemExit) as validation_exit:
        main(["train", "validate", "--json"])
    validation = json.loads(capsys.readouterr().out)
    with pytest.raises(SystemExit) as plan_exit:
        main(["train", "plan", "--json"])
    plan = json.loads(capsys.readouterr().out)

    assert validation_exit.value.code == 0
    assert validation["configuration_valid"] is True
    assert validation["credential_values_public"] is False
    assert plan_exit.value.code == 0
    assert plan["target_steps"] == 100
    assert plan["public_artifact_safe"] is True
