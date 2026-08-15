from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from crowdtensor.backends.accelerate import AccelerateFSDP2Backend
from crowdtensor.backends.stable_session import (
    StableShardedSessionError,
    validate_stable_trainer_result,
)
from crowdtensor.core import (
    ProviderSnapshot,
    ResourceAvailability,
    SessionController,
    TrainingMode,
    export_workspace,
    init_project,
    inspect_workspace,
    load_project,
    record_plan,
    stable_hash,
)


def _prepare_stable_workspace(tmp_path: Path):
    workspace = tmp_path / "stable-workspace"
    project = init_project(
        workspace,
        model="fixture/tiny-fsdp2",
        model_revision="fixture-revision",
        dataset="fixture/deterministic",
        dataset_revision="fixture-revision",
        model_adapter="fixture_full_v1",
        training_backend="accelerate_fsdp2",
        mode=TrainingMode.STABLE_SHARDED,
        target_steps=4,
    )["status"]
    trainer = workspace / "trainer.py"
    shutil.copy2(
        Path(__file__).parent / "fixtures" / "stable_fsdp2_trainer.py",
        trainer,
    )
    resource = ProviderSnapshot(
        provider_id="local-test",
        resource_id="local-test.cpu-ranks",
        machine_id_hash=stable_hash("local-test-machine"),
        device_type="cpu",
        device_count=2,
        total_memory_bytes=2 * 1024**3,
        free_memory_bytes=1024**3,
        availability=ResourceAvailability.STABLE_WINDOW,
        source_hash=stable_hash("local-test-source"),
        capabilities=("distributed_collective", "stable_sharded"),
        supported_dtypes=("float32",),
        stable_group_id="local-test-window",
    )
    backend = AccelerateFSDP2Backend()
    loaded_project = load_project(workspace)
    plan = backend.build_plan(
        loaded_project,
        (resource,),
        allow_cpu_validation=True,
        runtime_probe={"available": True, "version": "torch-2.11.0-cpu-test"},
    )
    spec = backend.build_launch_spec(
        plan,
        trainer_entrypoint="trainer.py",
        trainer_contract_verified=True,
        mixed_precision="no",
        max_restarts=1,
        allow_cpu_validation=True,
    )
    report = {
        "schema": "crowdtensor_training_plan_command_v2",
        "command_ok": True,
        "project_hash": project["project_hash"],
        "mode": TrainingMode.STABLE_SHARDED.value,
        "backend_id": backend.backend_id,
        "provider_snapshot_count": 1,
        "runtime_probe_performed": True,
        "plan": plan.to_dict(),
        "launch": spec.to_dict(),
        "materialization": None,
        "execution_ready": True,
        "command_executed": False,
        "credential_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = stable_hash(report)
    record_plan(workspace, report)
    return workspace, backend


def test_real_two_rank_cpu_fsdp2_restarts_from_committed_checkpoint(tmp_path) -> None:
    workspace, backend = _prepare_stable_workspace(tmp_path)
    fail_once = workspace / ".crowdtensor" / "fail-once"
    fail_once.touch()

    def command(spec, bindings):
        return (
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(spec.num_processes),
            "trainer.py",
            "--crowdtensor-project",
            ".crowdtensor/project.json",
            "--crowdtensor-checkpoint-dir",
            spec.checkpoint_path,
            "--crowdtensor-work-unit",
            bindings["CROWDTENSOR_WORK_UNIT_PATH"],
            "--crowdtensor-base-checkpoint",
            bindings["CROWDTENSOR_BASE_CHECKPOINT_PATH"],
            "--crowdtensor-base-payload",
            bindings["CROWDTENSOR_BASE_PAYLOAD_PATH"],
            "--crowdtensor-output-checkpoint",
            bindings["CROWDTENSOR_OUTPUT_CHECKPOINT_PATH"],
            "--crowdtensor-result",
            bindings["CROWDTENSOR_TRAINER_RESULT_PATH"],
            "--fail-once-marker",
            str(fail_once),
        )

    result = backend.run_session(
        workspace,
        steps_per_work_unit=2,
        max_work_units=2,
        timeout_seconds=120.0,
        command_factory=command,
        launcher_label="torch-distributed-cpu",
    )

    assert result["command_ok"] is True
    assert result["state"] == "completed"
    assert result["initial_step"] == 0
    assert result["final_step"] == 4
    assert result["work_units_completed"] == 2
    assert result["failed_attempts"] == 1
    assert result["restart_count"] == 1
    assert result["rank_count"] == 2
    assert result["device_types"] == ["cpu"]
    assert result["fsdp2_trainer_result_verified"] is True
    assert result["cpu_execution_reported"] is True
    assert result["gpu_execution_reported"] is False
    assert result["hardware_attestation_verified"] is False
    assert result["physical_multi_host_verified"] is False
    assert len(result["process_output_digests"]) == 3
    assert not fail_once.exists()

    controller = SessionController(workspace)
    assert [item.step for item in controller.lineage().checkpoints] == [0, 2, 4]
    failed = controller.terminal("stable-step-2-4", 1)
    accepted = controller.terminal("stable-step-2-4", 2)
    assert failed is not None and failed[1].outcome.value == "rejected"
    assert accepted is not None and accepted[1].outcome.value == "accepted"
    assert accepted[2].step == 2
    assert accepted[3] is not None and accepted[3].step == 4
    trainer_result = json.loads(
        (
            workspace
            / ".crowdtensor/runtime/stable-sharded/results/stable-step-2-4-g2.json"
        ).read_text(encoding="utf-8")
    )
    trainer_result["steps_completed"] = 3
    with pytest.raises(StableShardedSessionError, match="hash_invalid"):
        validate_stable_trainer_result(
            trainer_result,
            work=accepted[0],
            base_checkpoint=accepted[2],
            expected_rank_count=2,
        )
    assert inspect_workspace(workspace)["lifecycle_state"] == "completed"

    payloads = workspace / ".crowdtensor" / "checkpoints" / "stable-sharded" / "payloads"
    assert len(list(payloads.rglob("*.distcp"))) >= 2
    exported = tmp_path / "public-export"
    export_workspace(workspace, exported)
    assert not list(exported.rglob("*.distcp"))
    assert "contributor_id_hash" not in (
        exported / "session-controller.json"
    ).read_text(encoding="utf-8")


def test_cpu_validation_plan_cannot_use_production_accelerate_launcher(tmp_path) -> None:
    workspace, backend = _prepare_stable_workspace(tmp_path)
    with pytest.raises(
        StableShardedSessionError, match="stable_cpu_validation_launcher_required"
    ):
        backend.run_session(workspace, dry_run=True)


def test_missing_trainer_fails_before_work_is_issued(tmp_path) -> None:
    workspace, backend = _prepare_stable_workspace(tmp_path)
    (workspace / "trainer.py").unlink()

    with pytest.raises(
        StableShardedSessionError, match="stable_trainer_entrypoint_missing"
    ):
        backend.run_session(
            workspace,
            command_factory=lambda _spec, _bindings: (sys.executable, "trainer.py"),
            launcher_label="missing-trainer-test",
            dry_run=True,
        )

    assert not (workspace / ".crowdtensor/state/session-controller.json").exists()


def test_rank_group_start_failure_is_bounded_and_terminal(tmp_path) -> None:
    workspace, backend = _prepare_stable_workspace(tmp_path)

    result = backend.run_session(
        workspace,
        command_factory=lambda _spec, _bindings: ("/missing/crowdtensor-launcher",),
        launcher_label="missing-test-launcher",
        timeout_seconds=10.0,
    )

    assert result["command_ok"] is False
    assert result["blocker"] == "stable_rank_group_start_failed"
    assert result["failed_attempts"] == 2
    assert result["restart_count"] == 1
    assert result["command_attempted"] is True
    assert result["command_executed"] is False
    assert result["final_step"] == 0
    status = SessionController(workspace).status()
    assert status["active_work_count"] == 0
    assert status["terminal_count"] == 2
