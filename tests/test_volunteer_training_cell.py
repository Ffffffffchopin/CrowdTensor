from __future__ import annotations

import pytest

from crowdtensor.hf_lora_training import (
    CPULoRATrainingRuntime,
    create_local_training_fixture,
)
from crowdtensor.volunteer_training_cell import (
    LocalVolunteerTransport,
    VolunteerTrainingCell,
    VolunteerUploadInterrupted,
)
from crowdtensor.volunteer_training_coordinator import VolunteerTrainingCoordinator


class InterruptingLocalTransport(LocalVolunteerTransport):
    def submit(self, **_kwargs):
        raise VolunteerUploadInterrupted("a" * 64)


def test_new_cell_process_resumes_pending_submission_without_retraining(
    tmp_path, monkeypatch
) -> None:
    fixture = create_local_training_fixture(
        tmp_path / "fixture", row_count=8, local_steps=1
    )
    coordinator = VolunteerTrainingCoordinator.create_from_fixture(
        tmp_path / "campaign", fixture, target_rounds=1, lease_seconds=120
    )
    token = coordinator.private_invite()["invite_token"]
    original_run = CPULoRATrainingRuntime.run
    calls = []

    def counted_run(self, spec, *, output_dir):
        calls.append(spec["task_id"])
        return original_run(self, spec, output_dir=output_dir)

    monkeypatch.setattr(CPULoRATrainingRuntime, "run", counted_run)
    workspace = tmp_path / "cell"
    first = VolunteerTrainingCell(
        InterruptingLocalTransport(coordinator, token),
        workspace,
        cell_id="restart-cell",
        device="cpu",
    )
    with pytest.raises(VolunteerUploadInterrupted):
        first.join_once()
    assert len(calls) == 1
    assert "pending_submission" in first._private_state()

    resumed = VolunteerTrainingCell(
        LocalVolunteerTransport(coordinator, token),
        workspace,
        cell_id="ignored-new-process-id",
        device="cpu",
    ).join_once()
    assert resumed["work_completed"] is True
    assert resumed["training_reexecuted_for_submission_resume"] is False
    assert len(calls) == 1
    assert "pending_submission" not in first._private_state()
