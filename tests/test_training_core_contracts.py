from __future__ import annotations

import copy

import pytest

from crowdtensor.core.contracts import (
    ArtifactRef,
    CheckpointLineage,
    CheckpointRef,
    ContractError,
    ContributionReceipt,
    ReceiptOutcome,
    TrainingMode,
    TrainingProject,
    WorkUnit,
    stable_hash,
    validate_receipt_binding,
)


def _project() -> TrainingProject:
    return TrainingProject(
        project_id="demo-project",
        mode=TrainingMode.ELASTIC_DELTA,
        model=ArtifactRef("org/model", "revision-model"),
        dataset=ArtifactRef("org/dataset", "revision-data"),
        model_adapter="qwen2_lora_v1",
        training_backend="transformers_accelerate",
        target_steps=12,
        optimization_plugins=("peft_lora_v1",),
    )


def _chain(project: TrainingProject) -> tuple[CheckpointRef, CheckpointRef, WorkUnit]:
    genesis = CheckpointRef(
        checkpoint_id="genesis",
        project_hash=project.content_hash,
        step=0,
        generation=0,
        artifact=ArtifactRef(
            "checkpoints/genesis.safetensors",
            "checkpoint-0",
            stable_hash("genesis-bytes"),
        ),
    )
    work = WorkUnit(
        work_id="work-1",
        project_hash=project.content_hash,
        mode=project.mode,
        generation=1,
        backend=project.training_backend,
        base_checkpoint_hash=genesis.content_hash,
        data_shard_hash=stable_hash("shard-1"),
        step_start=0,
        step_count=2,
        required_capabilities=("cpu",),
    )
    output = CheckpointRef(
        checkpoint_id="checkpoint-1",
        project_hash=project.content_hash,
        step=2,
        generation=1,
        artifact=ArtifactRef(
            "checkpoints/checkpoint-1.safetensors",
            "checkpoint-1",
            stable_hash("checkpoint-1-bytes"),
        ),
        parent_hash=genesis.content_hash,
        created_by_work_id=work.work_id,
    )
    return genesis, output, work


def test_project_and_work_unit_round_trip_with_deterministic_hashes() -> None:
    project = _project()
    assert project.content_hash == project.to_dict()["content_hash"]
    assert TrainingProject.from_dict(project.to_dict()) == project
    _genesis, _output, work = _chain(project)
    assert WorkUnit.from_dict(work.to_dict()) == work
    assert work.to_dict() == WorkUnit.from_dict(work.to_dict()).to_dict()


def test_stable_sharded_mode_remains_explicit_in_project_and_work() -> None:
    project = TrainingProject(
        project_id="stable-project",
        mode=TrainingMode.STABLE_SHARDED,
        model=ArtifactRef("org/model", "revision-model"),
        dataset=ArtifactRef("org/dataset", "revision-data"),
        model_adapter="qwen2_lora_v1",
        training_backend="accelerate_fsdp2",
        target_steps=4,
    )
    checkpoint = CheckpointRef(
        checkpoint_id="stable-genesis",
        project_hash=project.content_hash,
        step=0,
        generation=0,
        artifact=ArtifactRef("checkpoint", "r0", stable_hash("stable")),
        adapter_only=False,
    )
    work = WorkUnit(
        work_id="stable-window-1",
        project_hash=project.content_hash,
        mode=project.mode,
        generation=1,
        backend=project.training_backend,
        base_checkpoint_hash=checkpoint.content_hash,
        data_shard_hash=stable_hash("stable-shard"),
        step_start=0,
        step_count=4,
        required_capabilities=("cuda", "stable_group"),
    )
    assert project.to_dict()["mode"] == "stable_sharded"
    assert WorkUnit.from_dict(work.to_dict()).mode is TrainingMode.STABLE_SHARDED


def test_contract_mutation_is_rejected_even_when_schema_is_unchanged() -> None:
    payload = _project().to_dict()
    payload["target_steps"] = 99
    with pytest.raises(ContractError, match="content_hash"):
        TrainingProject.from_dict(payload)


def test_lineage_requires_parent_chain_and_strict_progress() -> None:
    project = _project()
    genesis, output, _work = _chain(project)
    lineage = CheckpointLineage(project.content_hash, (genesis, output))
    assert CheckpointLineage.from_dict(lineage.to_dict()) == lineage
    fork = CheckpointRef(
        checkpoint_id="fork",
        project_hash=project.content_hash,
        step=3,
        generation=1,
        artifact=ArtifactRef("checkpoints/fork", "fork", stable_hash("fork")),
        parent_hash=stable_hash("wrong-parent"),
    )
    with pytest.raises(ContractError, match="parent_mismatch"):
        lineage.append(fork)


def test_receipt_binds_work_generation_and_output_checkpoint() -> None:
    project = _project()
    genesis, output, work = _chain(project)
    receipt = ContributionReceipt(
        receipt_id="receipt-1",
        project_hash=project.content_hash,
        work_id=work.work_id,
        work_generation=work.generation,
        contributor_id_hash=stable_hash("contributor-1"),
        base_checkpoint_hash=genesis.content_hash,
        submitted_artifact_hash=output.artifact.digest or stable_hash("missing"),
        outcome=ReceiptOutcome.ACCEPTED,
        completed_at="2026-08-11T00:00:00+00:00",
        steps=2,
        samples=4,
        tokens=32,
        checkpoint_committed=True,
        output_checkpoint_hash=output.content_hash,
        metrics=(("loss_end", 0.5),),
    )
    validate_receipt_binding(
        receipt, work=work, base_checkpoint=genesis, output_checkpoint=output
    )
    assert ContributionReceipt.from_dict(receipt.to_dict()) == receipt
    replay = copy.copy(receipt)
    object.__setattr__(replay, "work_generation", 2)
    with pytest.raises(ContractError, match="generation_mismatch"):
        validate_receipt_binding(
            replay, work=work, base_checkpoint=genesis, output_checkpoint=output
        )


def test_receipt_rejects_non_finite_metrics_and_public_identity() -> None:
    project = _project()
    genesis, _output, work = _chain(project)
    with pytest.raises(ContractError, match="non_finite"):
        ContributionReceipt(
            receipt_id="receipt-bad",
            project_hash=project.content_hash,
            work_id=work.work_id,
            work_generation=1,
            contributor_id_hash=stable_hash("contributor"),
            base_checkpoint_hash=genesis.content_hash,
            submitted_artifact_hash=stable_hash("delta"),
            outcome=ReceiptOutcome.REJECTED,
            completed_at="2026-08-11T00:00:00+00:00",
            steps=0,
            samples=0,
            tokens=0,
            rejection_code="delta_invalid",
            metrics=(("loss", float("nan")),),
        )


def test_accepted_delta_receipt_does_not_require_a_quorum_checkpoint() -> None:
    project = _project()
    genesis, _output, work = _chain(project)
    receipt = ContributionReceipt(
        receipt_id="receipt-uncommitted",
        project_hash=project.content_hash,
        work_id=work.work_id,
        work_generation=work.generation,
        contributor_id_hash=stable_hash("contributor-2"),
        base_checkpoint_hash=genesis.content_hash,
        submitted_artifact_hash=stable_hash("accepted-delta"),
        outcome=ReceiptOutcome.ACCEPTED,
        completed_at="2026-08-11T00:00:00+00:00",
        steps=2,
        samples=4,
        tokens=32,
        checkpoint_committed=False,
    )
    validate_receipt_binding(receipt, work=work, base_checkpoint=genesis)
    assert receipt.output_checkpoint_hash is None
    assert ContributionReceipt.from_dict(receipt.to_dict()) == receipt
