from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from crowdtensor.core import (
    ArtifactRef,
    CheckpointLineage,
    CheckpointRef,
    ContributionReceipt,
    ReceiptOutcome,
    SessionController,
    SessionControllerError,
    TrainingMode,
    WorkUnit,
    export_workspace,
    init_project,
    inspect_workspace,
    pause_workspace,
    stable_hash,
)


def _project(tmp_path, *, mode: str = "elastic_delta"):
    initialized = init_project(
        tmp_path,
        model="org/model",
        model_revision="model-revision",
        dataset="org/data",
        dataset_revision="data-revision",
        model_adapter="fixture_lora_v1",
        training_backend=(
            "volunteer_peft" if mode == "elastic_delta" else "accelerate_fsdp2"
        ),
        mode=mode,
        target_steps=4,
        optimization_plugins=("peft_lora_v1",),
    )
    return initialized["status"]["project_hash"]


def _genesis(project_hash: str) -> CheckpointRef:
    return CheckpointRef(
        checkpoint_id="adapter-v0",
        project_hash=project_hash,
        step=0,
        generation=0,
        artifact=ArtifactRef(
            "crowdtensor://fixture/adapter/v0",
            "adapter-v0",
            stable_hash("adapter-v0"),
        ),
    )


def _work(
    project_hash: str,
    base: CheckpointRef,
    *,
    work_id: str = "work-1",
    generation: int = 1,
    step_start: int = 0,
) -> WorkUnit:
    return WorkUnit(
        work_id=work_id,
        project_hash=project_hash,
        mode=TrainingMode.ELASTIC_DELTA,
        generation=generation,
        backend="volunteer_peft",
        base_checkpoint_hash=base.content_hash,
        data_shard_hash=stable_hash(f"shard-{work_id}"),
        step_start=step_start,
        step_count=2,
        required_capabilities=("peft_lora",),
    )


def _stable_work(
    project_hash: str,
    base: CheckpointRef,
    *,
    work_id: str = "stable-work-1",
    generation: int = 1,
) -> WorkUnit:
    return WorkUnit(
        work_id=work_id,
        project_hash=project_hash,
        mode=TrainingMode.STABLE_SHARDED,
        generation=generation,
        backend="accelerate_fsdp2",
        base_checkpoint_hash=base.content_hash,
        data_shard_hash=stable_hash("stable-data-window"),
        step_start=base.step,
        step_count=2,
        required_capabilities=("distributed_collective", "stable_sharded"),
    )


def _receipt(
    work: WorkUnit,
    *,
    outcome: ReceiptOutcome = ReceiptOutcome.ACCEPTED,
    output: CheckpointRef | None = None,
    receipt_id: str | None = None,
    contributor_id_hash: str | None = None,
) -> ContributionReceipt:
    accepted = outcome is ReceiptOutcome.ACCEPTED
    return ContributionReceipt(
        receipt_id=receipt_id or f"receipt-{work.work_id}-{work.generation}",
        project_hash=work.project_hash,
        work_id=work.work_id,
        work_generation=work.generation,
        contributor_id_hash=contributor_id_hash or stable_hash("contributor"),
        base_checkpoint_hash=work.base_checkpoint_hash,
        submitted_artifact_hash=stable_hash(
            f"delta-{work.work_id}-{work.generation}"
        ),
        outcome=outcome,
        completed_at=datetime(2026, 8, 14, tzinfo=timezone.utc).isoformat(),
        steps=work.step_count if accepted else 0,
        samples=4 if accepted else 0,
        tokens=32 if accepted else 0,
        checkpoint_committed=output is not None,
        output_checkpoint_hash=output.content_hash if output else None,
        rejection_code=None if accepted else "validation_rejected",
    )


def _output(work: WorkUnit, base: CheckpointRef, *, version: int = 1) -> CheckpointRef:
    return CheckpointRef(
        checkpoint_id=f"adapter-v{version}",
        project_hash=work.project_hash,
        step=version,
        generation=work.generation,
        artifact=ArtifactRef(
            f"crowdtensor://fixture/adapter/v{version}",
            f"adapter-v{version}",
            stable_hash(f"adapter-v{version}"),
        ),
        parent_hash=base.content_hash,
        created_by_work_id=work.work_id,
    )


def test_controller_commits_checkpoint_and_receipt_exactly_once(tmp_path) -> None:
    project_hash = _project(tmp_path)
    base = _genesis(project_hash)
    lineage = CheckpointLineage(project_hash, (base,))
    controller = SessionController(tmp_path)
    initialized = controller.initialize(lineage)
    assert initialized["checkpoint_count"] == 1

    work = _work(project_hash, base)
    assert controller.issue(work)["state"] == "work_active"
    assert controller.issue(work)["idempotent_replay"] is True
    output = _output(work, base)
    receipt = _receipt(work, output=output)
    committed = controller.commit(
        work, receipt, base_checkpoint=base, output_checkpoint=output
    )
    assert committed["state"] == "idle"
    assert committed["terminal_count"] == 1
    assert committed["checkpoint_count"] == 2

    replay = controller.commit(
        work, receipt, base_checkpoint=base, output_checkpoint=output
    )
    assert replay["idempotent_replay"] is True
    status = SessionController(tmp_path).status()
    assert status["head_checkpoint_hash"] == output.content_hash
    workspace = inspect_workspace(tmp_path)
    assert workspace["checkpoint_count"] == 2
    assert workspace["receipt_count"] == 1
    assert workspace["session_controller"]["terminal_count"] == 1


def test_controller_reassignment_fences_stale_generation(tmp_path) -> None:
    project_hash = _project(tmp_path)
    base = _genesis(project_hash)
    controller = SessionController(tmp_path)
    controller.initialize(CheckpointLineage(project_hash, (base,)))
    first = _work(project_hash, base, generation=1)
    second = _work(project_hash, base, generation=2)
    controller.issue(first)
    controller.issue(second, replace_active=True)

    with pytest.raises(SessionControllerError, match="generation_stale"):
        controller.commit(first, _receipt(first), base_checkpoint=base)
    accepted = controller.commit(second, _receipt(second), base_checkpoint=base)
    assert accepted["terminal_count"] == 1
    assert accepted["fenced_work_count"] == 1
    with pytest.raises(SessionControllerError, match="generation_stale"):
        controller.commit(first, _receipt(first), base_checkpoint=base)


def test_controller_tracks_concurrent_leases_and_monotonic_renewal(tmp_path) -> None:
    project_hash = _project(tmp_path)
    base = _genesis(project_hash)
    controller = SessionController(tmp_path)
    controller.initialize(CheckpointLineage(project_hash, (base,)))
    first = _work(project_hash, base, work_id="work-a")
    second = _work(project_hash, base, work_id="work-b")
    first_owner = stable_hash("owner-a")
    second_owner = stable_hash("owner-b")

    controller.issue(
        first, contributor_id_hash=first_owner, lease_expires_at=100.0
    )
    controller.issue(
        second, contributor_id_hash=second_owner, lease_expires_at=110.0
    )
    with pytest.raises(SessionControllerError, match="active_owner_mismatch"):
        controller.issue(first, lease_expires_at=130.0)
    status = controller.status()
    assert status["active_work_count"] == 2
    assert status["active_work"] is None
    with pytest.raises(SessionControllerError, match="active_work_ambiguous"):
        controller.active_work()

    renewed = controller.renew(
        "work-a",
        1,
        contributor_id_hash=first_owner,
        lease_expires_at=120.0,
    )
    assert renewed["idempotent_replay"] is False
    replay = controller.renew(
        "work-a",
        1,
        contributor_id_hash=first_owner,
        lease_expires_at=120.0,
    )
    assert replay["idempotent_replay"] is True
    with pytest.raises(SessionControllerError, match="lease_expiry_regressed"):
        controller.renew(
            "work-a",
            1,
            contributor_id_hash=first_owner,
            lease_expires_at=119.0,
        )
    with pytest.raises(SessionControllerError, match="active_owner_mismatch"):
        controller.renew(
            "work-b",
            1,
            contributor_id_hash=first_owner,
            lease_expires_at=130.0,
        )


def test_controller_explicitly_fences_only_expired_leases(tmp_path) -> None:
    project_hash = _project(tmp_path)
    base = _genesis(project_hash)
    controller = SessionController(tmp_path)
    controller.initialize(CheckpointLineage(project_hash, (base,)))
    expired = _work(project_hash, base, work_id="expired")
    live = _work(project_hash, base, work_id="live")
    controller.issue(expired, lease_expires_at=100.0)
    controller.issue(live, lease_expires_at=200.0)

    fenced = controller.fence_expired(now=150.0)
    assert fenced["active_work_count"] == 1
    assert fenced["fenced_work_count"] == 1
    assert controller.active_lease("expired") is None
    assert controller.active_lease("live") is not None
    replay = controller.fence_expired(now=150.0)
    assert replay["idempotent_replay"] is True
    with pytest.raises(SessionControllerError, match="generation_stale"):
        controller.commit(expired, _receipt(expired), base_checkpoint=base)

    with pytest.raises(SessionControllerError, match="generation_not_next"):
        controller.issue(
            _work(project_hash, base, work_id="expired", generation=3)
        )
    changed = _work(
        project_hash, base, work_id="expired", generation=2, step_start=2
    )
    with pytest.raises(SessionControllerError, match="reassignment_mismatch"):
        controller.issue(changed)
    reassigned = _work(project_hash, base, work_id="expired", generation=2)
    assert controller.issue(reassigned)["active_work_count"] == 2


def test_checkpoint_advance_fences_other_old_base_work(tmp_path) -> None:
    project_hash = _project(tmp_path)
    base = _genesis(project_hash)
    controller = SessionController(tmp_path)
    controller.initialize(CheckpointLineage(project_hash, (base,)))
    accepted = _work(project_hash, base, work_id="accepted")
    aggregating = _work(project_hash, base, work_id="aggregating")
    stale = _work(project_hash, base, work_id="stale")
    for work in (accepted, aggregating, stale):
        controller.issue(work)

    first = controller.commit(
        accepted, _receipt(accepted), base_checkpoint=base
    )
    assert first["active_work_count"] == 2
    output = _output(aggregating, base)
    advanced = controller.commit(
        aggregating,
        _receipt(aggregating, output=output),
        base_checkpoint=base,
        output_checkpoint=output,
    )
    assert advanced["active_work_count"] == 0
    assert advanced["terminal_count"] == 2
    assert advanced["fenced_work_count"] == 1
    with pytest.raises(SessionControllerError, match="generation_stale"):
        controller.commit(stale, _receipt(stale), base_checkpoint=base)


def test_controller_upgrades_legacy_single_active_state_on_mutation(tmp_path) -> None:
    project_hash = _project(tmp_path)
    base = _genesis(project_hash)
    controller = SessionController(tmp_path)
    controller.initialize(CheckpointLineage(project_hash, (base,)))
    work = _work(project_hash, base)
    owner = stable_hash("legacy-owner")
    controller.issue(work, contributor_id_hash=owner)

    state_path = tmp_path / ".crowdtensor" / "state" / "session-controller.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload.pop("active_work_units")
    payload.pop("controller_revision")
    payload.pop("fenced_work_units")
    payload.pop("content_hash")
    payload["content_hash"] = stable_hash(payload)
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    restarted = SessionController(tmp_path)
    assert restarted.status()["active_work_count"] == 1
    restarted.renew(
        work.work_id,
        work.generation,
        contributor_id_hash=owner,
        lease_expires_at=200.0,
    )
    upgraded = json.loads(state_path.read_text(encoding="utf-8"))
    assert upgraded["controller_revision"] == 2
    assert len(upgraded["active_work_units"]) == 1
    assert upgraded["fenced_work_units"] == []


def test_controller_persists_active_owner_and_validates_receipt_owner(tmp_path) -> None:
    project_hash = _project(tmp_path)
    base = _genesis(project_hash)
    controller = SessionController(tmp_path)
    controller.initialize(CheckpointLineage(project_hash, (base,)))
    work = _work(project_hash, base)
    owner = stable_hash("owner")
    controller.issue(work, contributor_id_hash=owner)
    assert SessionController(tmp_path).active_contributor_id_hash() == owner

    wrong_receipt = _receipt(work)
    with pytest.raises(SessionControllerError, match="receipt_owner_mismatch"):
        controller.commit(work, wrong_receipt, base_checkpoint=base)


def test_controller_rejected_work_can_retry_only_as_next_identical_generation(
    tmp_path,
) -> None:
    project_hash = _project(tmp_path)
    base = _genesis(project_hash)
    controller = SessionController(tmp_path)
    controller.initialize(CheckpointLineage(project_hash, (base,)))
    first = _work(project_hash, base, generation=1)
    controller.issue(first)
    rejected = _receipt(first, outcome=ReceiptOutcome.REJECTED)
    controller.commit(first, rejected, base_checkpoint=base)

    with pytest.raises(SessionControllerError, match="generation_not_next"):
        controller.issue(_work(project_hash, base, generation=3))
    changed = _work(project_hash, base, generation=2, step_start=2)
    with pytest.raises(SessionControllerError, match="reassignment_mismatch"):
        controller.issue(changed)
    retry = _work(project_hash, base, generation=2)
    assert controller.issue(retry)["state"] == "work_active"


def test_controller_uncommitted_acceptance_does_not_advance_lineage(tmp_path) -> None:
    project_hash = _project(tmp_path)
    base = _genesis(project_hash)
    controller = SessionController(tmp_path)
    controller.initialize(CheckpointLineage(project_hash, (base,)))
    work = _work(project_hash, base)
    controller.issue(work)
    report = controller.commit(work, _receipt(work), base_checkpoint=base)
    assert report["checkpoint_count"] == 1
    assert report["head_checkpoint_hash"] == base.content_hash
    with pytest.raises(SessionControllerError, match="accepted_work_terminal"):
        controller.issue(_work(project_hash, base, generation=2))


def test_controller_restores_projection_and_accepts_genesis_prefix(tmp_path) -> None:
    project_hash = _project(tmp_path)
    base = _genesis(project_hash)
    lineage = CheckpointLineage(project_hash, (base,))
    controller = SessionController(tmp_path)
    controller.initialize(lineage)
    work = _work(project_hash, base)
    output = _output(work, base)
    receipt = _receipt(work, output=output)
    controller.issue(work)
    controller.commit(work, receipt, base_checkpoint=base, output_checkpoint=output)
    for path in (tmp_path / ".crowdtensor" / "receipts").glob("*.json"):
        path.unlink()
    for path in (tmp_path / ".crowdtensor" / "checkpoints").glob("*.json"):
        path.unlink()

    restored = SessionController(tmp_path).initialize(lineage)
    assert restored["idempotent_replay"] is True
    assert inspect_workspace(tmp_path)["checkpoint_count"] == 2
    assert inspect_workspace(tmp_path)["receipt_count"] == 1


def test_controller_rejects_pause_and_tampering(tmp_path) -> None:
    project_hash = _project(tmp_path)
    base = _genesis(project_hash)
    controller = SessionController(tmp_path)
    controller.initialize(CheckpointLineage(project_hash, (base,)))
    pause_workspace(tmp_path, reason="maintenance")
    with pytest.raises(SessionControllerError, match="workspace_paused"):
        controller.issue(_work(project_hash, base))

    state_path = tmp_path / ".crowdtensor" / "state" / "session-controller.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["terminals"] = [{"forged": True}]
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SessionControllerError, match="state_invalid"):
        SessionController(tmp_path).status()



def test_controller_runs_stable_mode_with_one_rank_group(tmp_path) -> None:
    project_hash = _project(tmp_path, mode="stable_sharded")
    base = _genesis(project_hash)
    controller = SessionController(tmp_path)
    initialized = controller.initialize(CheckpointLineage(project_hash, (base,)))
    assert initialized["concurrent_elastic_work_supported"] is False
    assert initialized["stable_rank_group_restart_supported"] is True

    work = _stable_work(project_hash, base)
    owner = stable_hash("stable-rank-group")
    controller.issue(work, contributor_id_hash=owner)
    with pytest.raises(SessionControllerError, match="stable_rank_group_active"):
        controller.issue(
            _stable_work(project_hash, base, work_id="stable-work-2"),
            contributor_id_hash=owner,
        )

    output = CheckpointRef(
        checkpoint_id="full-v1",
        project_hash=project_hash,
        step=2,
        generation=1,
        artifact=ArtifactRef(
            "crowdtensor://fixture/full/v1",
            "full-v1",
            stable_hash("full-v1"),
        ),
        parent_hash=base.content_hash,
        created_by_work_id=work.work_id,
        adapter_only=False,
    )
    receipt = _receipt(work, output=output, contributor_id_hash=owner)
    committed = controller.commit(
        work, receipt, base_checkpoint=base, output_checkpoint=output
    )
    assert committed["checkpoint_count"] == 2
    assert committed["terminal_count"] == 1


def test_controller_workspace_export_contains_only_public_projections(tmp_path) -> None:
    project_hash = _project(tmp_path)
    base = _genesis(project_hash)
    controller = SessionController(tmp_path)
    controller.initialize(CheckpointLineage(project_hash, (base,)))
    work = _work(project_hash, base)
    controller.issue(work, contributor_id_hash=stable_hash("owner"))
    output = _output(work, base)
    receipt = _receipt(
        work,
        output=output,
        contributor_id_hash=stable_hash("owner"),
    )
    controller.commit(
        work, receipt, base_checkpoint=base, output_checkpoint=output
    )

    destination = tmp_path.parent / "controller-export"
    report = export_workspace(tmp_path, destination)
    assert "session-controller.json" in report["artifacts"]
    exported_controller = json.loads(
        (destination / "session-controller.json").read_text(encoding="utf-8")
    )
    assert exported_controller["schema"] == "crowdtensor_training_session_action_v2"
    assert "contributor_id_hash" not in json.dumps(exported_controller, sort_keys=True)
    assert "terminals" not in exported_controller
    assert "fenced_work_units" not in exported_controller
    assert len(list((destination / "checkpoints").glob("*.json"))) == 2
    assert len(list((destination / "receipts").glob("*.json"))) == 1
    assert not (destination / "session-controller.lock").exists()
    assert not any(destination.rglob("*.safetensors"))
