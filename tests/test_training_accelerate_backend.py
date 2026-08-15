from __future__ import annotations

from crowdtensor.backends.accelerate import (
    AccelerateFSDP2Backend,
    accelerate_config_text,
)
from crowdtensor.core import (
    ArtifactRef,
    ProviderSnapshot,
    ResourceAvailability,
    TrainingMode,
    TrainingProject,
    stable_hash,
)
from crowdtensor.core.workspace import init_project


def _project() -> TrainingProject:
    return TrainingProject(
        project_id="stable-demo",
        mode=TrainingMode.STABLE_SHARDED,
        model=ArtifactRef("org/model", "model-revision"),
        dataset=ArtifactRef("org/dataset", "dataset-revision"),
        model_adapter="qwen2_lora_v1",
        training_backend="accelerate_fsdp2",
        target_steps=10,
    )


def _gpu(index: int, machine: str) -> ProviderSnapshot:
    return ProviderSnapshot(
        provider_id="fixture",
        resource_id=f"fixture.gpu-{index}",
        machine_id_hash=stable_hash(machine),
        device_type="cuda",
        device_count=1,
        total_memory_bytes=24 * 1024**3,
        free_memory_bytes=20 * 1024**3,
        availability=ResourceAvailability.STABLE_WINDOW,
        source_hash=stable_hash({"gpu": index, "machine": machine}),
        capabilities=("cuda", "distributed_collective", "stable_sharded"),
        supported_dtypes=("bfloat16", "float16", "float32"),
        stable_group_id="window-a",
    )


def test_accelerate_backend_builds_stable_group_only() -> None:
    backend = AccelerateFSDP2Backend()
    plan = backend.build_plan(
        _project(),
        (_gpu(0, "machine-a"), _gpu(1, "machine-b")),
        runtime_probe={"available": True, "version": "accelerate-test+torch-test"},
    )
    assert plan.execution_ready is True
    assert len(plan.selected_resources) == 2
    spec = backend.build_launch_spec(
        plan,
        trainer_entrypoint="train.py",
        trainer_contract_verified=True,
        transformer_layer_class="Qwen2DecoderLayer",
    )
    assert spec.execution_ready is True
    assert spec.num_machines == 2
    assert spec.num_processes == 2
    assert "fsdp_version: 2" in accelerate_config_text(spec)
    assert "SHARDED_STATE_DICT" in accelerate_config_text(spec)
    assert "${CROWDTENSOR_MAIN_PROCESS_IP}" in spec.command_template


def test_accelerate_materialization_is_preview_only_and_path_independent(tmp_path) -> None:
    project = _project()
    init_project(
        tmp_path,
        model=project.model.uri,
        model_revision=project.model.revision,
        dataset=project.dataset.uri,
        dataset_revision=project.dataset.revision,
        model_adapter=project.model_adapter,
        training_backend=project.training_backend,
        mode=project.mode,
        target_steps=project.target_steps,
        project_id=project.project_id,
    )
    backend = AccelerateFSDP2Backend()
    plan = backend.build_plan(
        project,
        (_gpu(0, "machine-a"), _gpu(1, "machine-a")),
        runtime_probe={"available": True, "version": "accelerate-test+torch-test"},
    )
    spec = backend.build_launch_spec(
        plan,
        trainer_entrypoint="train.py",
        trainer_contract_verified=True,
        transformer_layer_class="Qwen2DecoderLayer",
    )
    report = backend.materialize_launch(spec, workspace=tmp_path)
    assert report["execution_ready"] is True
    assert report["command_executed"] is False
    assert str(tmp_path) not in str(report)
    assert (tmp_path / report["config_path"]).is_file()
