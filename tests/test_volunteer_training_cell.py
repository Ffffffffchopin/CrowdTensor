from __future__ import annotations

import pytest

from crowdtensor import volunteer_training_cell as cell_module
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
from crowdtensor.training_contract import sha256_file, sha256_json


class InterruptingLocalTransport(LocalVolunteerTransport):
    def submit(self, **_kwargs):
        raise VolunteerUploadInterrupted("a" * 64)


class ObservingLocalTransport(LocalVolunteerTransport):
    def __init__(self, *args, heartbeat_scope, **kwargs):
        super().__init__(*args, **kwargs)
        self.heartbeat_scope = heartbeat_scope
        self.submit_observations: list[bool] = []

    def submit(self, **kwargs):
        self.submit_observations.append(bool(self.heartbeat_scope["active"]))
        return super().submit(**kwargs)


def _track_heartbeat_scope(monkeypatch):
    scope = {"active": False}
    original = cell_module._Heartbeat

    class TrackingHeartbeat(original):
        def __enter__(self):
            scope["active"] = True
            return super().__enter__()

        def __exit__(self, kind, value, traceback):
            try:
                return super().__exit__(kind, value, traceback)
            finally:
                scope["active"] = False

    monkeypatch.setattr(cell_module, "_Heartbeat", TrackingHeartbeat)
    return scope


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

    heartbeat_scope = _track_heartbeat_scope(monkeypatch)
    resumed_transport = ObservingLocalTransport(
        coordinator, token, heartbeat_scope=heartbeat_scope
    )
    resumed = VolunteerTrainingCell(
        resumed_transport,
        workspace,
        cell_id="ignored-new-process-id",
        device="cpu",
    ).join_once()
    assert resumed["work_completed"] is True
    assert resumed["training_reexecuted_for_submission_resume"] is False
    assert len(calls) == 1
    assert "pending_submission" not in first._private_state()
    assert resumed_transport.submit_observations == [True]


def test_cell_keeps_heartbeat_active_through_initial_submission(
    tmp_path, monkeypatch
) -> None:
    fixture = create_local_training_fixture(
        tmp_path / "fixture", row_count=8, local_steps=1
    )
    coordinator = VolunteerTrainingCoordinator.create_from_fixture(
        tmp_path / "campaign", fixture, target_rounds=1, lease_seconds=120
    )
    token = coordinator.private_invite()["invite_token"]
    heartbeat_scope = _track_heartbeat_scope(monkeypatch)
    transport = ObservingLocalTransport(
        coordinator, token, heartbeat_scope=heartbeat_scope
    )

    result = VolunteerTrainingCell(
        transport,
        tmp_path / "cell",
        cell_id="submission-scope-cell",
        device="cpu",
    ).join_once()

    assert result["work_completed"] is True
    assert transport.submit_observations == [True]


def test_cell_verifies_and_reuses_explicit_huggingface_model_snapshot(
    tmp_path, monkeypatch
) -> None:
    fixture = create_local_training_fixture(
        tmp_path / "fixture", row_count=8, local_steps=1
    )
    coordinator = VolunteerTrainingCoordinator.create_from_fixture(
        tmp_path / "campaign", fixture, target_rounds=1, lease_seconds=120
    )
    token = coordinator.private_invite()["invite_token"]
    model_dir = tmp_path / "fixture" / "base_model"
    records = [
        {
            "relative_name": path.relative_to(model_dir).as_posix(),
            "sha256": sha256_file(path),
            "byte_count": path.stat().st_size,
        }
        for path in sorted(item for item in model_dir.rglob("*") if item.is_file())
    ]
    source_hash = sha256_json(records)
    campaign = coordinator.campaign_manifest()
    campaign["model_source"] = {
        "imported_files": records,
        "imported_snapshot_hash": source_hash,
        "runtime_fetch": {
            "provider": "huggingface_hub",
            "repo_id": "example/tiny-model",
            "revision": "a" * 40,
            "allow_patterns": sorted(item["relative_name"] for item in records),
            "file_manifest_hash": source_hash,
        },
    }
    calls = []

    def downloaded(**kwargs):
        calls.append(kwargs)
        return model_dir

    monkeypatch.setattr(VolunteerTrainingCell, "_download_huggingface_snapshot", staticmethod(downloaded))
    cell = VolunteerTrainingCell(
        LocalVolunteerTransport(coordinator, token),
        tmp_path / "cell",
        device="cpu",
        cache_dir=tmp_path / "shared-cache",
    )
    first = cell._verified_external_model_snapshot(campaign)
    second = cell._verified_external_model_snapshot(campaign)

    assert first is not None and second is not None
    assert first[0] == model_dir.resolve()
    assert first[1] == sum(item["byte_count"] for item in records)
    assert first[2]["model_source_cache_hit"] is False
    assert second[1] == 0
    assert second[2]["model_source_cache_hit"] is True
    assert len(calls) == 1
