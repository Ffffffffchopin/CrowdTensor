from __future__ import annotations

import copy

import pytest

from crowdtensor.core import (
    ContractError,
    ProviderSnapshot,
    ResourceAvailability,
    StableShardedLaunchSpec,
    TrainingExecutionPlan,
    TrainingMode,
    stable_hash,
)


def _resource(
    resource_id: str,
    *,
    machine: str = "machine-a",
    availability: ResourceAvailability = ResourceAvailability.STABLE_WINDOW,
) -> ProviderSnapshot:
    return ProviderSnapshot(
        provider_id="test-provider",
        resource_id=resource_id,
        machine_id_hash=stable_hash(machine),
        device_type="cuda",
        device_count=1,
        total_memory_bytes=24 * 1024**3,
        free_memory_bytes=20 * 1024**3,
        availability=availability,
        source_hash=stable_hash({"resource": resource_id}),
        capabilities=("cuda", "distributed_collective", "stable_sharded"),
        supported_dtypes=("bfloat16", "float16", "float32"),
        performance_score=1.0,
        stable_group_id=("stable-test" if availability is ResourceAvailability.STABLE_WINDOW else None),
    )


def _plan() -> TrainingExecutionPlan:
    return TrainingExecutionPlan(
        project_hash=stable_hash("project"),
        mode=TrainingMode.STABLE_SHARDED,
        backend_id="accelerate_fsdp2",
        selected_resources=(_resource("gpu-1"), _resource("gpu-0")),
        required_capabilities=("cuda", "distributed_collective", "stable_sharded"),
        restart_semantics="restart_rank_group_from_committed_checkpoint",
        runtime_name="accelerate_fsdp2",
        runtime_available=True,
        runtime_version="accelerate-test+torch-test",
    )


def test_provider_snapshot_and_execution_plan_round_trip() -> None:
    resource = _resource("gpu-0")
    assert ProviderSnapshot.from_dict(resource.to_dict()) == resource
    plan = _plan()
    assert plan.execution_ready is True
    assert [item.resource_id for item in plan.selected_resources] == ["gpu-0", "gpu-1"]
    assert TrainingExecutionPlan.from_dict(plan.to_dict()) == plan


def test_stable_plan_rejects_intermittent_rank() -> None:
    with pytest.raises(ContractError, match="intermittent"):
        TrainingExecutionPlan(
            project_hash=stable_hash("project"),
            mode=TrainingMode.STABLE_SHARDED,
            backend_id="accelerate_fsdp2",
            selected_resources=(
                _resource(
                    "gpu-0",
                    availability=ResourceAvailability.INTERMITTENT,
                ),
            ),
            required_capabilities=("cuda",),
            restart_semantics="restart_rank_group_from_committed_checkpoint",
            runtime_name="accelerate_fsdp2",
            runtime_available=True,
            runtime_version="test",
        )


def test_execution_contract_mutation_is_rejected() -> None:
    payload = _plan().to_dict()
    payload["num_processes"] = 99
    with pytest.raises(ContractError, match="unknown_fields"):
        TrainingExecutionPlan.from_dict(payload)
    payload = copy.deepcopy(_plan().to_dict())
    payload["runtime_version"] = "changed"
    with pytest.raises(ContractError, match="content_hash"):
        TrainingExecutionPlan.from_dict(payload)


def test_stable_launch_is_relative_secret_free_and_round_trips() -> None:
    plan = _plan()
    spec = StableShardedLaunchSpec(
        plan_hash=plan.content_hash,
        project_hash=plan.project_hash,
        backend_id="accelerate_fsdp2",
        distributed_type="fsdp2",
        num_machines=1,
        num_processes=2,
        mixed_precision="bf16",
        config_path=".crowdtensor/state/accelerate-fsdp2.yaml",
        checkpoint_path=".crowdtensor/checkpoints/stable-sharded",
        trainer_entrypoint="train.py",
        transformer_layer_class="LlamaDecoderLayer",
        command_template=(
            "accelerate",
            "launch",
            "--machine_rank",
            "${CROWDTENSOR_MACHINE_RANK}",
            "train.py",
        ),
        environment_names=("CROWDTENSOR_MACHINE_RANK",),
        max_restarts=1,
    )
    assert spec.execution_ready is True
    assert StableShardedLaunchSpec.from_dict(spec.to_dict()) == spec
    assert "/root" not in str(spec.to_dict())
    with pytest.raises(ContractError, match="must_be_relative"):
        StableShardedLaunchSpec(
            **{**spec.__dict__, "config_path": "/tmp/config.yaml"}
        )
