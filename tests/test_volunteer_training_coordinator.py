from __future__ import annotations

import copy
import json
import math
import threading

import pytest
import torch

from crowdtensor.hf_lora_training import create_local_training_fixture
from crowdtensor.named_tensor_optimizer import load_tensors, save_tensors
from crowdtensor.training_contract import delta_manifest, sha256_json
from crowdtensor.volunteer_training_coordinator import VolunteerTrainingCoordinator
from crowdtensor.volunteer_training_protocol import VolunteerProtocolError


class Clock:
    def __init__(self) -> None:
        self.value = 1000.0
        self.lock = threading.Lock()

    def __call__(self) -> float:
        with self.lock:
            return self.value

    def advance(self, seconds: float) -> None:
        with self.lock:
            self.value += seconds


def _manifest(tmp_path, coordinator, campaign, work, cell_id, case, *, non_finite=False):
    token = coordinator.private_invite()["invite_token"]
    adapter = coordinator.artifact_path(
        work["artifact_refs"]["base_adapter"]["artifact_id"], invite_token=token
    )
    base = load_tensors(adapter)
    values = {name: torch.ones_like(value) for name, value in base.items()}
    if non_finite:
        values[sorted(values)[0]].view(-1)[0] = float("nan")
    path = save_tensors(values, tmp_path / f"{case}.safetensors")
    return delta_manifest(
        delta_path=path,
        job_id=campaign["campaign_id"],
        round_id=work["round_id"],
        result_id=sha256_json({"case": case}),
        miner_id=cell_id,
        model_manifest_hash=campaign["model_manifest_hash"],
        base_model_hash=campaign["base_model_hash"],
        base_adapter_hash=work["base_adapter_hash"],
        base_model_version=campaign["model_revision"],
        adapter_version=work["adapter_version"],
        dataset_shard_index=work["dataset_shard_index"],
        dataset_shard_hash=work["dataset_shard_hash"],
        loss_start=4.0,
        loss_end=3.9,
        samples_seen=2,
        tokens_seen=16,
    )


def _submit(coordinator, token, cell_id, work, manifest):
    return coordinator.submit(
        cell_id=cell_id,
        invite_token=token,
        work_id=work["work_id"],
        lease_generation=work["lease_generation"],
        lease_token=work["lease_token"],
        delta_manifest=manifest,
    )


def test_churn_fencing_clipping_quorum_and_idempotency(tmp_path) -> None:
    clock = Clock()
    fixture = create_local_training_fixture(
        tmp_path / "fixture", row_count=8, local_steps=1
    )
    coordinator = VolunteerTrainingCoordinator.create_from_fixture(
        tmp_path / "campaign",
        fixture,
        target_rounds=1,
        lease_seconds=10,
        clip_delta_norm=1.0,
        hard_max_delta_norm=100.0,
        clock=clock,
    )
    token = coordinator.private_invite()["invite_token"]
    campaign = coordinator.campaign_manifest()
    with pytest.raises(VolunteerProtocolError, match="authentication_failed"):
        coordinator.claim(cell_id="unauthorized", invite_token="wrong")

    offline = coordinator.claim(cell_id="offline", invite_token=token)["work_unit"]
    survivor = coordinator.claim(cell_id="survivor", invite_token=token)["work_unit"]
    invalid = _manifest(
        tmp_path, coordinator, campaign, offline, "offline", "nan", non_finite=True
    )
    with pytest.raises(VolunteerProtocolError, match="adapter_delta_non_finite"):
        _submit(coordinator, token, "offline", offline, invalid)

    survivor_delta = _manifest(
        tmp_path, coordinator, campaign, survivor, "survivor", "survivor"
    )
    fork = copy.deepcopy(survivor_delta)
    fork["result_id"] = sha256_json({"case": "fork"})
    fork["base_adapter_hash"] = "sha256:" + "f" * 64
    with pytest.raises(VolunteerProtocolError, match="base_adapter_hash_mismatch"):
        _submit(coordinator, token, "survivor", survivor, fork)
    first = _submit(coordinator, token, "survivor", survivor, survivor_delta)
    assert first["accepted"] is True
    assert first["delta_clipped"] is True

    clock.advance(11)
    expired = coordinator.expire_leases(invite_token=token)
    assert expired["expired_lease_count"] == 1
    replacement = coordinator.claim(cell_id="replacement", invite_token=token)["work_unit"]
    assert replacement["work_id"] == offline["work_id"]
    assert replacement["lease_generation"] == offline["lease_generation"] + 1
    replacement_delta = _manifest(
        tmp_path, coordinator, campaign, replacement, "replacement", "replacement"
    )
    second = _submit(coordinator, token, "replacement", replacement, replacement_delta)
    assert second["round_aggregated"] is True
    assert second["adapter_version_after"] == 1
    assert second["accepted_at"] == clock.value

    replay = _submit(
        coordinator,
        token,
        "replacement",
        {"work_id": replacement["work_id"], "lease_generation": 0, "lease_token": ""},
        replacement_delta,
    )
    assert replay["idempotent_replay"] is True
    assert replay["round_aggregated"] is True
    assert replay["adapter_version_after"] == second["adapter_version_after"]
    assert replay["accepted_at"] == second["accepted_at"]
    assert replay["delta_clipped"] == second["delta_clipped"]
    assert replay["delta_norm_before_clip"] == second["delta_norm_before_clip"]
    assert replay["delta_norm_after_clip"] == second["delta_norm_after_clip"]
    first_replay = _submit(
        coordinator,
        token,
        "survivor",
        {"work_id": survivor["work_id"], "lease_generation": 0, "lease_token": ""},
        survivor_delta,
    )
    assert first_replay["idempotent_replay"] is True
    assert first_replay["round_aggregated"] is False
    assert first_replay["adapter_version_after"] == 0
    stale = copy.deepcopy(invalid)
    stale["result_id"] = sha256_json({"case": "late"})
    with pytest.raises(VolunteerProtocolError, match="stale_adapter_version"):
        _submit(coordinator, token, "offline", offline, stale)

    status = coordinator.status()
    assert status["campaign_complete"] is True
    assert status["accepted_update_count"] == 2
    assert status["rejected_update_count"] == 3
    assert coordinator.verify_ledger()["ok"] is True
    lineage = coordinator.checkpoint_lineage()
    assert lineage["ok"] is True
    assert lineage["completed_round_count"] == 1
    assert lineage["entries"][0]["adapter_version_before"] == 0
    assert lineage["entries"][0]["adapter_version_after"] == 1
    evaluation = coordinator.evaluate_campaign(heldout_quality=True)
    assert evaluation["held_out_quality_benchmark_performed"] is True
    assert evaluation["quality"]["candidate_adapter_version"] == 1
    assert math.isfinite(evaluation["quality"]["baseline_mean_loss"])
    assert math.isfinite(evaluation["quality"]["candidate_mean_loss"])
    assert evaluation["statistical_significance_claimed"] is False
    assert str(tmp_path) not in json.dumps(evaluation, sort_keys=True)
    serialized = (coordinator.status_path.read_text(encoding="utf-8") + coordinator.ledger_path.read_text(encoding="utf-8"))
    assert token not in serialized
    assert '"lease_token"' not in serialized


def test_restart_recovery_preserves_active_lease(tmp_path) -> None:
    clock = Clock()
    fixture = create_local_training_fixture(
        tmp_path / "fixture-restart", row_count=8, local_steps=1
    )
    root = tmp_path / "campaign-restart"
    original = VolunteerTrainingCoordinator.create_from_fixture(
        root,
        fixture,
        target_rounds=1,
        lease_seconds=30,
        clock=clock,
    )
    token = original.private_invite()["invite_token"]
    work = original.claim(cell_id="restart-cell", invite_token=token)["work_unit"]

    recovered = VolunteerTrainingCoordinator(root, clock=clock)
    report = recovered.recover_after_restart()
    assert report["ok"] is True
    assert report["active_lease_count_preserved"] == 1
    heartbeat = recovered.heartbeat(
        cell_id="restart-cell",
        invite_token=token,
        work_id=work["work_id"],
        lease_generation=work["lease_generation"],
        lease_token=work["lease_token"],
    )
    assert heartbeat["ok"] is True
    assert heartbeat["work_unit_hash"] == work["work_unit_hash"]
    assert recovered.verify_ledger()["ok"] is True
