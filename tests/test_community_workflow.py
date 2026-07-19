import json

from crowdtensor.community_cli import run
from crowdtensor.community_workflow import CommunityWorkflow


def test_workspace_lifecycle_is_idempotent_public_safe_and_dry_runnable(tmp_path) -> None:
    workspace = tmp_path / "project"
    initialized = CommunityWorkflow.initialize(workspace, target_steps=100)
    replay = CommunityWorkflow.initialize(workspace, target_steps=100)
    workflow = CommunityWorkflow(workspace)

    assert initialized["ok"] is True
    assert initialized["idempotent_replay"] is False
    assert replay["idempotent_replay"] is True
    assert initialized["run_id"] == replay["run_id"]
    assert workflow.validate()["ok"] is True
    plan = workflow.plan()
    assert len(plan["stage_specs"]) == 5
    assert plan["physical_multi_machine_verified"] is False
    assert workflow.coordinator_up(dry_run=True)["ok"] is True
    assert workflow.miner_join(dry_run=True)["ok"] is True
    assert workflow.train(dry_run=True)["ok"] is True
    assert workflow.status()["runtime_state"] == "initialized"
    assert workflow.pause(dry_run=True)["ok"] is True
    assert workflow.resume(dry_run=True)["ok"] is True
    assert workflow.rebalance(dry_run=True)["ok"] is True
    assert workflow.export(dry_run=True)["ok"] is True
    assert workflow.stop(dry_run=True)["ok"] is True
    assert workflow.cleanup(dry_run=True)["ok"] is True
    contract = workflow.contract()
    assert contract["workflow_actions"] == [
        "init", "validate", "plan", "coordinator up", "miner join", "train",
        "status", "pause", "resume", "rebalance", "export", "stop", "cleanup",
    ]
    assert contract["exit_codes"] == {
        "success": 0, "validation": 2, "state": 3, "protocol": 4, "runtime": 5
    }
    for path in (workspace / "artifacts").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload.get("public_artifact_safe") is True
        assert str(tmp_path) not in path.read_text(encoding="utf-8")


def test_cli_exposes_full_golden_path_with_explicit_exit_codes(tmp_path) -> None:
    workspace = tmp_path / "cli-project"
    assert run(["init", str(workspace), "--json"]) == 0
    assert run(["validate", str(workspace), "--json"]) == 0
    assert run(["plan", str(workspace), "--json"]) == 0
    assert run(["coordinator", "up", str(workspace), "--dry-run", "--json"]) == 0
    assert run(["miner", "join", str(workspace), "--dry-run", "--json"]) == 0
    for action in ("train", "pause", "resume", "export", "stop", "cleanup"):
        assert run([action, str(workspace), "--dry-run", "--json"]) == 0
    assert run(["status", str(workspace), "--json"]) == 0
    assert run(["contract", str(workspace), "--json"]) == 0


def test_smollm_workspace_plans_two_stages_and_rejects_production_start(tmp_path) -> None:
    workspace = tmp_path / "smol"
    CommunityWorkflow.initialize(
        workspace,
        adapter_id="smollm2_lora_v1",
        target_steps=2,
    )
    workflow = CommunityWorkflow(workspace)
    assert len(workflow.plan()["stage_specs"]) == 2
    assert workflow.train(dry_run=True)["ok"] is True
