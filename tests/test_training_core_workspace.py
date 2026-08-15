from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from crowdtensor.core import stable_hash
from crowdtensor.core.workspace import (
    WorkspaceError,
    export_workspace,
    init_project,
    inspect_workspace,
    join_workspace,
    load_project,
    pause_workspace,
    record_plan,
    resume_workspace,
    run_workspace,
)


def _kwargs() -> dict[str, str]:
    return {
        "model": "org/model",
        "model_revision": "model-revision",
        "dataset": "org/data",
        "dataset_revision": "data-revision",
        "model_adapter": "qwen2_lora_v1",
    }


def test_workspace_init_is_idempotent_and_path_independent(tmp_path) -> None:
    first = init_project(tmp_path, **_kwargs())
    second = init_project(tmp_path, **_kwargs())
    assert first["created"] is True
    assert second["created"] is False
    assert first["status"]["project_hash"] == second["status"]["project_hash"]
    assert str(tmp_path) not in json.dumps(second, sort_keys=True)
    assert (tmp_path / ".crowdtensor" / "project.json").is_file()
    assert load_project(tmp_path).project_id == tmp_path.name


def test_workspace_conflicting_intent_fails_closed(tmp_path) -> None:
    init_project(tmp_path, **_kwargs())
    changed = dict(_kwargs(), target_steps=7)
    with pytest.raises(WorkspaceError, match="manifest_conflict"):
        init_project(tmp_path, **changed)


def test_workspace_tampering_is_detected(tmp_path) -> None:
    init_project(tmp_path, **_kwargs())
    path = tmp_path / ".crowdtensor" / "project.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["target_steps"] = 500
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WorkspaceError, match="manifest_invalid"):
        inspect_workspace(tmp_path)


def test_workspace_control_tampering_is_detected(tmp_path) -> None:
    init_project(tmp_path, **_kwargs())
    path = tmp_path / ".crowdtensor" / "state" / "workspace-control.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["generation"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WorkspaceError, match="control_state_invalid"):
        inspect_workspace(tmp_path)


def test_repeated_init_repairs_missing_layout_without_changing_project(tmp_path) -> None:
    first = init_project(tmp_path, **_kwargs())
    receipts = tmp_path / ".crowdtensor" / "receipts"
    receipts.rmdir()
    assert inspect_workspace(tmp_path)["workspace_layout_ready"] is False
    repaired = init_project(tmp_path, **_kwargs())
    assert repaired["created"] is False
    assert repaired["status"]["workspace_layout_ready"] is True
    assert repaired["status"]["project_hash"] == first["status"]["project_hash"]


def test_concurrent_identical_initializers_publish_one_manifest(tmp_path) -> None:
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(lambda _index: init_project(tmp_path, **_kwargs()), range(4))
        )
    assert sum(item["created"] is True for item in results) == 1
    assert len({item["status"]["project_hash"] for item in results}) == 1
    assert inspect_workspace(tmp_path)["workspace_layout_ready"] is True


def test_workspace_lifecycle_is_durable_and_generation_fenced(tmp_path) -> None:
    init_project(tmp_path, **_kwargs())
    initial = inspect_workspace(tmp_path)
    assert initial["lifecycle_state"] == "initialized"
    assert initial["generation"] == 0

    paused = pause_workspace(tmp_path, reason="owner_window_closed")
    assert paused["command_ok"] is True
    assert paused["state"] == "paused"
    assert paused["generation"] == 1
    assert paused["pause_reason"] == "owner_window_closed"

    repeated = pause_workspace(tmp_path, reason="still_closed")
    assert repeated["generation"] == 1
    assert repeated["state"] == "paused"

    resumed = resume_workspace(tmp_path)
    assert resumed["command_ok"] is True
    assert resumed["state"] == "initialized"
    assert resumed["generation"] == 2
    assert inspect_workspace(tmp_path)["pause_reason"] is None

    resumed_again = resume_workspace(tmp_path)
    assert resumed_again["generation"] == 2


def test_workspace_run_and_join_are_explicit_controller_boundaries(tmp_path) -> None:
    init_project(tmp_path, **_kwargs())
    run_report = run_workspace(tmp_path)
    assert run_report["command_ok"] is False
    assert run_report["state"] == "blocked"
    assert run_report["blockers"] == ["execution_plan_required"]

    join_report = join_workspace(tmp_path)
    assert join_report["command_ok"] is False
    assert join_report["blockers"] == ["v2_controller_join_pending"]
    assert inspect_workspace(tmp_path)["last_action"] == "join"


def test_workspace_paused_state_is_not_cleared_by_run_or_join(tmp_path) -> None:
    init_project(tmp_path, **_kwargs())
    pause_workspace(tmp_path, reason="maintenance")

    run_report = run_workspace(tmp_path)
    assert run_report["command_ok"] is False
    assert run_report["state"] == "paused"
    assert run_report["blockers"] == ["workspace_paused"]
    assert inspect_workspace(tmp_path)["pause_reason"] == "maintenance"

    join_report = join_workspace(tmp_path)
    assert join_report["command_ok"] is False
    assert join_report["state"] == "paused"
    assert join_report["blockers"] == ["workspace_paused"]
    assert inspect_workspace(tmp_path)["pause_reason"] == "maintenance"


def test_workspace_plan_does_not_clear_paused_state(tmp_path) -> None:
    initialized = init_project(tmp_path, **_kwargs())
    pause_workspace(tmp_path, reason="maintenance")
    plan = {
        "schema": "crowdtensor_training_plan_command_v2",
        "command_ok": True,
        "project_hash": initialized["status"]["project_hash"],
        "plan": {"blockers": ["runtime_unavailable"]},
        "execution_ready": False,
        "credential_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    plan["content_hash"] = stable_hash(plan)

    report = record_plan(tmp_path, plan)
    assert report["state"] == "paused"
    status = inspect_workspace(tmp_path)
    assert status["pause_reason"] == "maintenance"
    assert status["last_plan_hash"] == plan["content_hash"]


def test_workspace_plan_tampering_is_detected(tmp_path) -> None:
    initialized = init_project(tmp_path, **_kwargs())
    plan = {
        "schema": "crowdtensor_training_plan_command_v2",
        "command_ok": True,
        "project_hash": initialized["status"]["project_hash"],
        "plan": {"blockers": ["runtime_unavailable"]},
        "execution_ready": False,
        "credential_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    plan["content_hash"] = stable_hash(plan)
    record_plan(tmp_path, plan)
    path = tmp_path / ".crowdtensor" / "state" / "execution-plan.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["execution_ready"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(WorkspaceError, match="execution_plan_invalid"):
        inspect_workspace(tmp_path)


def test_workspace_export_contains_public_state_but_no_private_payload(tmp_path) -> None:
    init_project(tmp_path, **_kwargs())
    (tmp_path / ".crowdtensor" / "private.bin").write_bytes(b"secret")
    output = tmp_path.parent / "export"
    report = export_workspace(tmp_path, output)
    assert report["command_ok"] is True
    assert "project.json" in report["artifacts"]
    assert "workspace-control.json" in report["artifacts"]
    assert "status.json" in report["artifacts"]
    assert not (output / "private.bin").exists()
    assert not (output / "model.safetensors").exists()
    exported_status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert exported_status["public_artifact_safe"] is True
    assert str(tmp_path) not in json.dumps(exported_status, sort_keys=True)


def test_workspace_export_rejects_destination_inside_workspace(tmp_path) -> None:
    init_project(tmp_path, **_kwargs())
    with pytest.raises(WorkspaceError, match="destination_inside"):
        export_workspace(tmp_path, tmp_path / "nested-export")


def test_workspace_export_rejects_nonempty_destination(tmp_path) -> None:
    init_project(tmp_path, **_kwargs())
    output = tmp_path.parent / "existing-export"
    output.mkdir()
    (output / "unrelated.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(WorkspaceError, match="destination_not_empty"):
        export_workspace(tmp_path, output)
    assert (output / "unrelated.txt").read_text(encoding="utf-8") == "keep"
