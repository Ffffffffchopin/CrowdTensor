from copy import deepcopy
from pathlib import Path
import sys

import pytest

from crowdtensor.heterogeneous_training_manifest import (
    qwen25_7b_lora_manifest,
    qwen25_7b_lora_tpu_manifest,
)
from crowdtensor.heterogeneous_training_scheduler import (
    CAPABILITY_SCHEMA,
    TPU_CAPABILITY_SCHEMA,
    TPU_PLACEMENT_SCHEMA,
    PlacementError,
    build_placement_plan,
    detect_stragglers,
    estimate_stage_resources,
    validate_miner_capability,
)


GIB = 1024**3


def capability(
    name: str,
    *,
    gpu_memory: list[int],
    cpu_memory: int = 32 * GIB,
    max_stages: int = 2,
    profiles: list[dict] | None = None,
    load: float = 0.0,
) -> dict:
    value = {
        "schema": CAPABILITY_SCHEMA,
        "miner_id_hash": "sha256:" + name.encode().hex().ljust(64, "0")[:64],
        "cpu": {
            "device_id": "cpu",
            "physical_core_count": 4,
            "logical_core_count": 8,
            "total_memory_bytes": cpu_memory,
            "free_memory_bytes": cpu_memory,
            "supported_dtypes": ["bfloat16", "float32"],
            "throughput_units_per_second": 80_000_000,
            "microbenchmark_latency_ms": 1.0,
            "utilization_fraction": load,
        },
        "gpus": [
            {
                "device_id": f"cuda:{index}",
                "device_index": index,
                "device_name_hash": "sha256:" + f"{name}-{index}".encode().hex().ljust(64, "0")[:64],
                "total_memory_bytes": memory,
                "free_memory_bytes": memory,
                "compute_capability": "7.5",
                "supported_dtypes": ["float16", "float32"],
                "throughput_units_per_second": 1_500_000_000,
                "utilization_fraction": load,
                "raw_device_name_public": False,
            }
            for index, memory in enumerate(gpu_memory)
        ],
        "network": {
            "measured_bandwidth_bytes_per_second": 100 * 1024**2,
            "measured_round_trip_latency_ms": 10.0,
            "measurement_count": 3,
        },
        "stage_profiles": profiles or [],
        "current_load_fraction": load,
        "max_stage_count": max_stages,
        "single_gpu_miner": len(gpu_memory) == 1,
        "multi_gpu_miner": len(gpu_memory) > 1,
        "cpu_stage_supported": True,
        "raw_device_names_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    return validate_miner_capability(value)


def tpu_capability(
    name: str,
    *,
    compile_ms: float = 2500.0,
    forward_ms: float = 20.0,
    backward_ms: float = 40.0,
) -> dict:
    value = capability(name, gpu_memory=[], max_stages=1)
    value.pop("content_hash")
    value["schema"] = TPU_CAPABILITY_SCHEMA
    value["cpu_stage_supported"] = False
    value["tpu_groups"] = [
        {
            "device_id": "jax_tpu:0",
            "runtime_backend": "jax",
            "accelerator_type": "TPU v5e",
            "device_kind_hash": "sha256:" + "a" * 64,
            "device_count": 8,
            "mesh_axis_names": ["data"],
            "mesh_shape": [8],
            "total_hbm_bytes": 128 * GIB,
            "free_hbm_bytes": 120 * GIB,
            "per_device_hbm_bytes": 16 * GIB,
            "hbm_measurement_source": "jax_memory_stats",
            "supported_dtypes": ["bfloat16", "float32"],
            "throughput_units_per_second": 8_000_000_000,
            "compile_microbenchmark_latency_ms": compile_ms,
            "steady_microbenchmark_latency_ms": 2.0,
            "utilization_fraction": 0.05,
            "all_devices_addressable": True,
            "raw_device_names_public": False,
        }
    ]
    value["jax_tpu_stage_supported"] = True
    value["stage_profiles"] = [
        {
            "stage_id": 2,
            "device_id": "jax_tpu:0",
            "forward_latency_ms": forward_ms,
            "backward_latency_ms": backward_ms,
            "steady_forward_latency_ms": forward_ms,
            "steady_backward_latency_ms": backward_ms,
            "compile_latency_ms": compile_ms,
            "peak_memory_bytes": 4 * GIB,
            "sample_count": 3,
            "measured_at": 10.0,
        }
    ]
    return validate_miner_capability(value)


def test_resource_estimate_covers_all_required_peak_components() -> None:
    manifest = qwen25_7b_lora_manifest()
    cuda = estimate_stage_resources(manifest, 0, device_type="cuda")
    cpu = estimate_stage_resources(manifest, 4, device_type="cpu")

    for estimate in (cuda, cpu):
        assert estimate["resident_weight_bytes"] > 0
        assert estimate["lora_parameter_bytes"] > 0
        assert estimate["optimizer_state_bytes"] > 0
        assert estimate["lora_gradient_bytes"] > 0
        assert estimate["activation_bytes"] > 0
        assert estimate["activation_gradient_bytes"] > 0
        assert estimate["workspace_bytes"] > 0
        assert estimate["estimated_peak_bytes"] == sum(
            estimate[key]
            for key in (
                "resident_weight_bytes",
                "lora_parameter_bytes",
                "lora_gradient_bytes",
                "optimizer_state_bytes",
                "activation_bytes",
                "activation_gradient_bytes",
                "workspace_bytes",
            )
        )


def test_mixed_placement_uses_four_single_gpu_miners_and_cpu_stage() -> None:
    manifest = qwen25_7b_lora_manifest()
    miners = [
        capability(f"gpu-{index}", gpu_memory=[16 * GIB], max_stages=1)
        for index in range(4)
    ]
    miners.append(capability("cpu-tail", gpu_memory=[], max_stages=1))

    plan = build_placement_plan(manifest, miners)

    assert plan["complete_stage_coverage"] is True
    assert plan["required_device_coverage_complete"] is True
    assert plan["accepted_device_types"] == ["cpu", "cuda"]
    assert plan["single_gpu_miner_participating"] is True
    assert plan["cpu_miner_participating"] is True
    assert [item["device_type"] for item in plan["assignments"]] == [
        "cuda",
        "cuda",
        "cuda",
        "cuda",
        "cpu",
    ]
    assert all(
        item["device_used_after_assignment_bytes"]
        <= item["available_after_reserve_bytes"]
        for item in plan["assignments"]
    )


def test_tpu_placement_uses_three_cuda_one_v5e_group_and_one_cpu() -> None:
    manifest = qwen25_7b_lora_tpu_manifest()
    miners = [
        capability(f"gpu-{index}", gpu_memory=[16 * GIB], max_stages=1)
        for index in range(3)
    ]
    miners.extend(
        [
            tpu_capability("tpu"),
            capability("cpu-tail", gpu_memory=[], max_stages=1),
        ]
    )

    plan = build_placement_plan(manifest, miners)

    assert plan["schema"] == TPU_PLACEMENT_SCHEMA
    assert plan["accepted_device_types"] == ["cpu", "cuda", "jax_tpu"]
    assert [item["device_type"] for item in plan["assignments"]] == [
        "cuda",
        "cuda",
        "jax_tpu",
        "cuda",
        "cpu",
    ]
    tpu = plan["assignments"][2]
    assert tpu["compute_latency_measured"] is True
    assert tpu["compile_latency_ms"] == pytest.approx(2500.0)
    assert tpu["tpu_compile_cost_ms"] == pytest.approx(2500.0)
    assert tpu["resource_estimate"]["parameter_sharding_required"] is True
    assert plan["tpu_miner_participating"] is True
    assert plan["tpu_compile_cost_considered"] is True
    tpu_capacity = next(
        item for item in plan["device_capacity"] if item["device_type"] == "jax_tpu"
    )
    assert tpu_capacity["group_device_count"] == 8
    assert tpu_capacity["mesh_shape"] == [8]


def test_rebalance_reserves_the_only_cpu_slot_for_the_cpu_only_tail() -> None:
    manifest = qwen25_7b_lora_tpu_manifest(target_steps=100)
    gpu_miners = []
    for index in range(4):
        value = capability(
            f"production-gpu-{index}", gpu_memory=[16 * GIB], max_stages=1
        )
        value.pop("content_hash")
        value["cpu_stage_supported"] = False
        gpu_miners.append(validate_miner_capability(value))
    cpu = capability(
        "production-cpu", gpu_memory=[], cpu_memory=32 * GIB, max_stages=1
    )
    tpu = tpu_capability("production-tpu")
    initial = build_placement_plan(manifest, [*gpu_miners, cpu, tpu])
    telemetry = [
        {
            "miner_id_hash": cpu["miner_id_hash"],
            "device_id": "cpu",
            "health_score": 1.0,
            "checkpoint_step": 4,
            "free_memory_bytes": 30 * GIB,
            "throughput_units_per_second": 1_000_000_000_000.0,
            "utilization_fraction": 0.0,
            "reported_at": 10.0,
            "consecutive_failures": 0,
        }
    ]

    rebalanced = build_placement_plan(
        manifest,
        [*gpu_miners, cpu, tpu],
        previous_plan=initial,
        reason="performance_rebalance",
        runtime_telemetry=telemetry,
        current_checkpoint_step=4,
    )

    cpu_assignments = [
        item for item in rebalanced["assignments"] if item["device_type"] == "cpu"
    ]
    assert [item["stage_id"] for item in cpu_assignments] == [4]
    assert rebalanced["complete_stage_coverage"] is True
    assert rebalanced["forward_feasibility_checked"] is True
    assert rebalanced["forward_feasibility_rejection_count"] > 0


def test_rebalance_does_not_double_charge_resident_gpu_memory() -> None:
    manifest = qwen25_7b_lora_tpu_manifest(target_steps=100)
    gpu_miners = []
    for index in range(4):
        value = capability(
            f"resident-gpu-{index}", gpu_memory=[16 * GIB], max_stages=1
        )
        value.pop("content_hash")
        value["cpu_stage_supported"] = False
        gpu_miners.append(validate_miner_capability(value))
    cpu = capability(
        "resident-cpu", gpu_memory=[], cpu_memory=32 * GIB, max_stages=1
    )
    tpu = tpu_capability("resident-tpu")
    capabilities = [*gpu_miners, cpu, tpu]
    initial = build_placement_plan(manifest, capabilities)
    telemetry = [
        {
            "miner_id_hash": item["miner_id_hash"],
            "device_id": "cuda:0",
            "health_score": 1.0,
            "checkpoint_step": 4,
            "free_memory_bytes": 5 * GIB,
            "throughput_units_per_second": 1_500_000_000.0,
            "utilization_fraction": 0.5,
            "reported_at": 20.0,
            "consecutive_failures": 0,
        }
        for item in gpu_miners
    ]

    rebalanced = build_placement_plan(
        manifest,
        capabilities,
        previous_plan=initial,
        reason="performance_rebalance",
        runtime_telemetry=telemetry,
        current_checkpoint_step=4,
    )

    assert rebalanced["complete_stage_coverage"] is True
    assert rebalanced["resident_assignment_memory_reclaimed"] is True
    assert [item["device_type"] for item in rebalanced["assignments"]] == [
        "cuda",
        "cuda",
        "jax_tpu",
        "cuda",
        "cpu",
    ]
    resident_gpu_budgets = [
        item
        for item in rebalanced["device_capacity"]
        if item["device_type"] == "cuda"
        and item["reclaimable_previous_assignment_bytes"] > 0
    ]
    assert len(resident_gpu_budgets) == 3
    assert all(
        item["available_after_reserve_bytes"] > 5 * GIB
        for item in resident_gpu_budgets
    )


def test_auto_miner_policy_detects_tpu_and_registers_one_resource_group(
    monkeypatch,
) -> None:
    import crowdtensor.heterogeneous_training_miner as miner_module

    identity = "sha256:" + "9" * 64

    def discover(**kwargs):
        value = tpu_capability("auto")
        value.pop("content_hash")
        value["miner_id_hash"] = kwargs["miner_id_hash"]
        return validate_miner_capability(value)

    monkeypatch.setattr(
        miner_module, "_discover_capability_isolated", discover
    )

    detected = miner_module.miner_capability(
        miner_id_hash=identity,
        device_policy="auto",
        run_microbenchmark=False,
    )

    assert detected["schema"] == TPU_CAPABILITY_SCHEMA
    assert detected["jax_tpu_stage_supported"] is True
    assert len(detected["tpu_groups"]) == 1
    assert detected["max_stage_count"] == 1
    assert detected["cpu_stage_supported"] is False
    assert detected["gpus"] == []


def test_isolated_tpu_capability_probe_returns_after_child_exit(
    monkeypatch, tmp_path
) -> None:
    import crowdtensor.heterogeneous_training_miner as miner_module

    package_parent = Path(miner_module.__file__).resolve().parent.parent
    monkeypatch.setattr(
        sys,
        "path",
        [
            value
            for value in sys.path
            if value and Path(value).resolve() != package_parent
        ],
    )
    monkeypatch.chdir(tmp_path)
    identity = "sha256:" + "8" * 64
    detected = miner_module._discover_capability_isolated(
        miner_id_hash=identity,
        max_stage_count=1,
        run_microbenchmark=False,
        timeout=120.0,
    )

    assert detected["schema"] == TPU_CAPABILITY_SCHEMA
    assert detected["miner_id_hash"] == identity
    assert detected["public_artifact_safe"] is True
    assert detected["private_paths_public"] is False


def test_multi_gpu_miner_and_single_gpu_miner_share_same_plan() -> None:
    manifest = qwen25_7b_lora_manifest()
    miners = [
        capability("multi", gpu_memory=[16 * GIB, 16 * GIB], max_stages=2),
        capability("single-a", gpu_memory=[16 * GIB], max_stages=1),
        capability("single-b", gpu_memory=[16 * GIB], max_stages=1),
        capability("cpu", gpu_memory=[], max_stages=1),
    ]

    plan = build_placement_plan(manifest, miners)

    assert plan["multi_gpu_miner_participating"] is True
    assert plan["single_gpu_miner_participating"] is True
    assert plan["cpu_miner_participating"] is True


def test_insufficient_gpu_memory_fails_closed_with_candidate_audit() -> None:
    manifest = qwen25_7b_lora_manifest()
    miners = [
        capability("small", gpu_memory=[8 * GIB], cpu_memory=5 * GIB, max_stages=5)
    ]

    with pytest.raises(PlacementError) as captured:
        build_placement_plan(manifest, miners)

    assert captured.value.code == "heterogeneous_placement_stage_has_no_eligible_device"
    assert captured.value.diagnostics["stage_id"] == 0
    assert all(
        not item["eligible"] for item in captured.value.diagnostics["candidates"]
    )


def _small_manifest() -> dict:
    manifest = qwen25_7b_lora_manifest()
    manifest.pop("content_hash")
    manifest["scheduler"]["max_stages_per_miner"] = 4
    for stage in manifest["stages"]:
        stage["estimated_weight_bytes"] = 64 * 1024**2
        stage["estimated_compute_units"] = 10_000_000.0
        stage["allowed_device_types"] = ["cpu", "cuda"]
        stage["preferred_device_type"] = "cuda"
    manifest["stages"][-1]["preferred_device_type"] = "cpu"
    from crowdtensor.heterogeneous_training_manifest import validate_training_manifest

    return validate_training_manifest(manifest)


def test_measured_throughput_changes_deterministic_placement() -> None:
    manifest = _small_manifest()
    slow = capability(
        "slow",
        gpu_memory=[16 * GIB],
        max_stages=4,
        profiles=[
            {
                "stage_id": 0,
                "device_id": "cuda:0",
                "forward_latency_ms": 50.0,
                "backward_latency_ms": 50.0,
                "peak_memory_bytes": GIB,
                "sample_count": 5,
                "measured_at": 1.0,
            }
        ],
    )
    fast = capability(
        "fast",
        gpu_memory=[16 * GIB],
        max_stages=4,
        profiles=[
            {
                "stage_id": 0,
                "device_id": "cuda:0",
                "forward_latency_ms": 1.0,
                "backward_latency_ms": 1.0,
                "peak_memory_bytes": GIB,
                "sample_count": 5,
                "measured_at": 1.0,
            }
        ],
    )
    cpu = capability("cpu", gpu_memory=[], max_stages=4)

    first = build_placement_plan(manifest, [slow, fast, cpu])
    second = build_placement_plan(manifest, [slow, fast, cpu])

    assert first["content_hash"] == second["content_hash"]
    assert first["assignments"][0]["miner_id_hash"] == fast["miner_id_hash"]
    assert first["assignments"][0]["compute_latency_measured"] is True


def test_migration_cost_prevents_profile_churn_but_allows_straggler_rebalance() -> None:
    manifest = _small_manifest()
    gpu_a = capability("migration-a", gpu_memory=[16 * GIB], max_stages=4)
    gpu_b = capability("migration-b", gpu_memory=[16 * GIB], max_stages=4)
    cpu = capability("migration-cpu", gpu_memory=[], max_stages=4)
    initial = build_placement_plan(manifest, [gpu_a, gpu_b, cpu])
    initial_stage0 = initial["assignments"][0]["miner_id_hash"]
    alternate_stage0 = next(
        item["miner_id_hash"]
        for item in (gpu_a, gpu_b)
        if item["miner_id_hash"] != initial_stage0
    )

    refreshed = []
    for item in (gpu_a, gpu_b, cpu):
        value = deepcopy(item)
        value.pop("content_hash", None)
        if value["gpus"]:
            latency = 200.0 if value["miner_id_hash"] == initial_stage0 else 1.0
            value["stage_profiles"] = [
                {
                    "stage_id": 0,
                    "device_id": "cuda:0",
                    "forward_latency_ms": latency / 2,
                    "backward_latency_ms": latency / 2,
                    "peak_memory_bytes": GIB,
                    "sample_count": 5,
                    "measured_at": 2.0,
                }
            ]
        refreshed.append(validate_miner_capability(value))

    stable = build_placement_plan(
        manifest,
        refreshed,
        previous_plan=initial,
        reason="initial_placement",
    )
    forced = build_placement_plan(
        manifest,
        refreshed,
        previous_plan=initial,
        reason="straggler_detected",
    )

    assert stable["assignments"][0]["miner_id_hash"] == initial_stage0
    assert stable["assignments"][0]["migration_required"] is False
    assert stable["stage_migration_cost_considered"] is True
    assert forced["assignments"][0]["miner_id_hash"] == alternate_stage0
    assert forced["assignments"][0]["migration_penalty_ms"] == 0.0


def test_cpu_fallback_and_generation_increment_on_gpu_oom() -> None:
    manifest = _small_manifest()
    gpu = capability("gpu", gpu_memory=[16 * GIB], max_stages=4)
    surviving_gpu = capability("surviving-gpu", gpu_memory=[16 * GIB], max_stages=1)
    cpu = capability("cpu", gpu_memory=[], max_stages=4)
    initial = build_placement_plan(manifest, [gpu, surviving_gpu, cpu])
    gpu_key = f"{gpu['miner_id_hash']}/cuda:0"

    fallback = build_placement_plan(
        manifest,
        [gpu, surviving_gpu, cpu],
        previous_plan=initial,
        reason="device_oom",
        excluded_devices=[gpu_key],
    )

    assert fallback["placement_generation"] == initial["placement_generation"] + 1
    assert fallback["rebalance_reason"] == "device_oom"
    assert sum(item["device_type"] == "cuda" for item in fallback["assignments"]) == 1
    assert sum(item["device_type"] == "cpu" for item in fallback["assignments"]) == 4


def test_straggler_detection_uses_persistent_sample_threshold() -> None:
    plan = {
        "assignments": [
            {"stage_id": 0, "miner_id_hash": "sha256:a", "device_id": "cuda:0"},
            {"stage_id": 1, "miner_id_hash": "sha256:b", "device_id": "cpu"},
            {"stage_id": 2, "miner_id_hash": "sha256:c", "device_id": "cuda:0"},
        ]
    }
    metrics = [
        {"stage_id": 0, "total_latency_ms": 10.0, "sample_count": 5},
        {"stage_id": 1, "total_latency_ms": 100.0, "sample_count": 5},
        {"stage_id": 2, "total_latency_ms": 12.0, "sample_count": 5},
    ]

    result = detect_stragglers(plan, metrics, ratio=2.0)

    assert [item["stage_id"] for item in result] == [1]
    assert result[0]["straggler_ratio"] > 8.0
