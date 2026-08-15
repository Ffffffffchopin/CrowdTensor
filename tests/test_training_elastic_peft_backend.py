from __future__ import annotations

from crowdtensor.backends.elastic_peft import (
    VolunteerControllerTransport,
    VolunteerPEFTBackend,
    checkpoint_lineage_from_campaign,
    project_from_campaign,
    receipt_from_submission,
    work_from_claim,
)
import pytest
import threading
import time

from crowdtensor.core import (
    ArtifactRef,
    CheckpointRef,
    ContractError,
    ProviderSnapshot,
    ResourceAvailability,
    SessionController,
    SessionControllerError,
    stable_hash,
)
from crowdtensor.core.plugins import ElasticDeltaBackend
from crowdtensor.hf_lora_training import create_local_training_fixture
from crowdtensor.volunteer_training_cell import (
    LocalVolunteerTransport,
    VolunteerTrainingCell,
)
from crowdtensor.volunteer_training_coordinator import VolunteerTrainingCoordinator
from crowdtensor.volunteer_training_protocol import (
    CAMPAIGN_SCHEMA,
    PROTOCOL_VERSION,
    WORK_UNIT_SCHEMA,
    campaign_content_hash,
    with_public_safety,
    work_unit_content_hash,
)
from crowdtensor.training_contract import sha256_json


def _campaign() -> dict:
    campaign = {
        "schema": CAMPAIGN_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "campaign_id": "bridge-test",
        "campaign_revision": 1,
        "model_id": "org/model",
        "model_revision": 1,
        "model_manifest_hash": stable_hash("model-manifest"),
        "base_model_hash": stable_hash("base-model"),
        "model_parameter_count": 100,
        "model_artifact_hash": stable_hash("model-artifact"),
        "dataset_id": "org/dataset",
        "dataset_revision": 1,
        "dataset_snapshot_hash": stable_hash("dataset"),
        "dataset_shards": [
            {
                "shard_index": index,
                "shard_hash": stable_hash(f"shard-{index}"),
                "sample_count": 4,
                "token_count": 32,
                "artifact_hash": stable_hash(f"shard-artifact-{index}"),
                "artifact_byte_count": 10,
            }
            for index in range(2)
        ],
        "initial_adapter_hash": stable_hash("adapter-v0"),
        "adapter_tensor_contract_hash": stable_hash("tensor-contract"),
        "adapter_tensor_count": 2,
        "local_training": {
            "local_steps": 2,
            "max_local_steps": 8,
            "learning_rate": 0.001,
            "batch_size": 1,
            "sequence_length": 8,
            "gradient_accumulation": 1,
            "optimizer_contract": "adamw_lora_v1",
        },
        "round_policy": {
            "minimum_quorum": 2,
            "target_rounds": 3,
            "lease_seconds": 60.0,
            "aggregation_interval": "quorum_round",
            "distinct_cells_required": True,
            "stale_updates_rejected": True,
            "late_updates_rejected": True,
        },
        "outer_optimizer": {
            "optimizer_type": "local_sgd_mean",
            "outer_lr": 1.0,
            "momentum": 0.0,
            "named_tensor_aggregation": True,
        },
        "update_admission": {
            "tensor_name_shape_dtype_validation": True,
            "content_hash_validation": True,
            "finite_values_required": True,
            "clip_delta_norm": 1.0,
            "hard_max_delta_norm": 2.0,
            "max_loss_increase": 1.0,
            "max_delta_bytes": 1024,
        },
    }
    campaign = with_public_safety(campaign)
    campaign["manifest_hash"] = campaign_content_hash(campaign)
    return campaign


def _claim(
    campaign: dict,
    *,
    work_id: str = "work-000001",
    shard_index: int = 0,
    generation: int = 1,
) -> dict:
    claim = {
        "schema": WORK_UNIT_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "campaign_id": campaign["campaign_id"],
        "campaign_manifest_hash": campaign["manifest_hash"],
        "round_id": "round-000001",
        "round_index": 0,
        "work_id": work_id,
        "adapter_version": 0,
        "base_adapter_hash": campaign["initial_adapter_hash"],
        "dataset_shard_hash": campaign["dataset_shards"][shard_index]["shard_hash"],
        "local_steps": 2,
        "step_start": 0,
        "lease_generation": generation,
        "lease_expires_at": 9999999999.0,
        "lease_token": "private-token",
    }
    claim["work_unit_hash"] = work_unit_content_hash(claim)
    return claim


def _resource() -> ProviderSnapshot:
    return ProviderSnapshot(
        provider_id="fixture",
        resource_id="fixture.cpu",
        machine_id_hash=stable_hash("machine"),
        device_type="cpu",
        device_count=4,
        total_memory_bytes=8 * 1024**3,
        free_memory_bytes=6 * 1024**3,
        availability=ResourceAvailability.INTERMITTENT,
        source_hash=stable_hash("source"),
        capabilities=("cpu", "elastic_delta", "peft_lora"),
        supported_dtypes=("float32",),
    )


def _lineage_report(campaign: dict, hashes: list[str]) -> dict:
    entries = [
        {
            "round_id": f"round-{index + 1:06d}",
            "round_index": index,
            "adapter_version_before": index,
            "adapter_version_after": index + 1,
            "base_adapter_hash": hashes[index],
            "canonical_adapter_hash": hashes[index + 1],
            "input_delta_hashes": [stable_hash(f"input-{index}")],
            "distinct_cell_count": 2,
            "lineage_link_verified": True,
        }
        for index in range(len(hashes) - 1)
    ]
    report = with_public_safety(
        {
            "schema": "crowdtensor_volunteer_checkpoint_lineage_v1",
            "ok": True,
            "campaign_id": campaign["campaign_id"],
            "campaign_manifest_hash": campaign["manifest_hash"],
            "initial_adapter_hash": hashes[0],
            "canonical_adapter_hash": hashes[-1],
            "adapter_version": len(entries),
            "outer_step": len(entries),
            "completed_round_count": len(entries),
            "entries": entries,
            "errors": [],
            "append_only_ledger_head_hash": stable_hash("ledger"),
        }
    )
    report["content_hash"] = sha256_json(report)
    return report


class _ControllerTransportFixture:
    def __init__(self, *, aggregate: bool) -> None:
        self.campaign_value = _campaign()
        self.claim_value = _claim(self.campaign_value)
        self.aggregate = aggregate
        self.output_hash = stable_hash("adapter-v1-aggregated")
        self.lineage_hashes = [self.campaign_value["initial_adapter_hash"]]
        self.submission_count = 0
        self.first_response: dict | None = None

    def campaign(self):
        return self.campaign_value

    def checkpoint_lineage(self):
        return _lineage_report(self.campaign_value, self.lineage_hashes)

    def status(self):
        return {"ok": True}

    def claim(self, *, cell_id, capability):
        return {"ok": True, "state": "leased", "work_unit": self.claim_value}

    def heartbeat(self, *, cell_id, work):
        return {"ok": True}

    def download_artifact(self, ref, destination, *, max_bytes):
        return 0

    def submit(self, *, cell_id, work, delta_manifest):
        self.submission_count += 1
        if self.first_response is not None:
            return {**self.first_response, "idempotent_replay": True}
        if self.aggregate:
            self.lineage_hashes.append(self.output_hash)
        self.first_response = {
            "accepted": True,
            "idempotent_replay": False,
            "result_id": delta_manifest["result_id"],
            "round_aggregated": self.aggregate,
            "adapter_version_after": 1 if self.aggregate else 0,
            "accepted_at": 1786665600.0,
        }
        return dict(self.first_response)


class _ConcurrentControllerTransportFixture:
    def __init__(self) -> None:
        self.campaign_value = _campaign()
        self.claims = {
            "cell-one": _claim(
                self.campaign_value, work_id="work-000001", shard_index=0
            ),
            "cell-two": _claim(
                self.campaign_value, work_id="work-000002", shard_index=1
            ),
        }
        self.output_hash = stable_hash("adapter-v1-concurrent")
        self.lineage_hashes = [self.campaign_value["initial_adapter_hash"]]
        self.responses: dict[str, dict] = {}
        self.claim_count = 0
        self.lock = threading.Lock()

    def campaign(self):
        return self.campaign_value

    def checkpoint_lineage(self):
        return _lineage_report(self.campaign_value, self.lineage_hashes)

    def status(self):
        return {"ok": True}

    def claim(self, *, cell_id, capability):
        with self.lock:
            self.claim_count += 1
        return {
            "ok": True,
            "state": "leased",
            "work_unit": self.claims[cell_id],
        }

    def heartbeat(self, *, cell_id, work):
        return {
            "ok": True,
            "lease_expires_at": float(work["lease_expires_at"]) + 60.0,
        }

    def download_artifact(self, ref, destination, *, max_bytes):
        return 0

    def submit(self, *, cell_id, work, delta_manifest):
        result_id = delta_manifest["result_id"]
        with self.lock:
            existing = self.responses.get(result_id)
            if existing is not None:
                return {**existing, "idempotent_replay": True}
            aggregated = len(self.responses) == 1
            if aggregated:
                self.lineage_hashes.append(self.output_hash)
            response = {
                "accepted": True,
                "idempotent_replay": False,
                "result_id": result_id,
                "round_aggregated": aggregated,
                "adapter_version_after": 1 if aggregated else 0,
                "accepted_at": 1786665600.0 + len(self.responses),
            }
            self.responses[result_id] = response
            return dict(response)


def _delta(result: str) -> dict:
    return {
        "result_id": stable_hash(result),
        "delta_file_hash": stable_hash("delta-" + result),
        "samples_seen": 4,
        "tokens_seen": 32,
        "loss_start": 1.0,
        "loss_end": 0.8,
    }


def test_volunteer_campaign_claim_and_uncommitted_receipt_bridge() -> None:
    campaign = _campaign()
    project = project_from_campaign(campaign)
    lineage_report = with_public_safety(
        {
            "schema": "crowdtensor_volunteer_checkpoint_lineage_v1",
            "ok": True,
            "campaign_id": campaign["campaign_id"],
            "campaign_manifest_hash": campaign["manifest_hash"],
            "initial_adapter_hash": campaign["initial_adapter_hash"],
            "canonical_adapter_hash": campaign["initial_adapter_hash"],
            "entries": [],
        }
    )
    lineage_report["content_hash"] = sha256_json(lineage_report)
    lineage = checkpoint_lineage_from_campaign(
        project,
        campaign,
        lineage_report,
    )
    base = lineage.checkpoints[-1]
    work = work_from_claim(project, campaign, _claim(campaign), base_checkpoint=base)
    delta = {
        "result_id": stable_hash("result-1"),
        "delta_file_hash": stable_hash("delta-1"),
        "samples_seen": 4,
        "tokens_seen": 32,
        "loss_start": 1.0,
        "loss_end": 0.8,
    }
    receipt = receipt_from_submission(
        project,
        work,
        base,
        contributor_id_hash=stable_hash("contributor"),
        delta_manifest=delta,
        response={
            "accepted": True,
            "result_id": delta["result_id"],
            "round_aggregated": False,
            "delta_norm_before_clip": 0.5,
            "delta_norm_after_clip": 0.5,
        },
        completed_at="2026-08-11T00:00:00+00:00",
    )
    assert project.training_backend == "volunteer_peft"
    assert receipt.checkpoint_committed is False
    assert receipt.output_checkpoint_hash is None
    assert "lease_token" not in str(work.to_dict())


def test_volunteer_backend_builds_elastic_plan_and_delegates_cell() -> None:
    backend = VolunteerPEFTBackend()
    assert isinstance(backend, ElasticDeltaBackend)
    project = project_from_campaign(_campaign())
    plan = backend.build_plan(
        project,
        (_resource(),),
        runtime_probe={"available": True, "version": "fixture-runtime"},
    )
    assert plan.execution_ready is True
    assert plan.mode.value == "elastic_delta"

    class Cell:
        def join_once(self):
            return {"ok": True, "state": "submitted"}

    assert backend.run_volunteer_cell_once(Cell())["state"] == "submitted"


def test_quorum_receipt_binds_aggregated_checkpoint_and_replay_is_not_reissued() -> None:
    campaign = _campaign()
    project = project_from_campaign(campaign)
    base = CheckpointRef(
        checkpoint_id="adapter-v0",
        project_hash=project.content_hash,
        step=0,
        generation=0,
        artifact=ArtifactRef(
            "crowdtensor://campaign/bridge-test/adapter/v0",
            "adapter-v0",
            campaign["initial_adapter_hash"],
        ),
    )
    work = work_from_claim(project, campaign, _claim(campaign), base_checkpoint=base)
    output = CheckpointRef(
        checkpoint_id="adapter-v1",
        project_hash=project.content_hash,
        step=1,
        generation=1,
        artifact=ArtifactRef(
            "crowdtensor://campaign/bridge-test/adapter/v1",
            "adapter-v1",
            stable_hash("adapter-v1-aggregated"),
        ),
        parent_hash=base.content_hash,
    )
    delta = {
        "result_id": stable_hash("result-quorum"),
        "delta_file_hash": stable_hash("delta-quorum"),
        "samples_seen": 4,
        "tokens_seen": 32,
    }
    response = {
        "accepted": True,
        "result_id": delta["result_id"],
        "round_aggregated": True,
    }
    receipt = receipt_from_submission(
        project,
        work,
        base,
        contributor_id_hash=stable_hash("contributor-quorum"),
        delta_manifest=delta,
        response=response,
        completed_at=1786396800.0,
        output_checkpoint=output,
    )
    assert receipt.checkpoint_committed is True
    assert receipt.output_checkpoint_hash == output.content_hash
    with pytest.raises(ContractError, match="existing_receipt"):
        receipt_from_submission(
            project,
            work,
            base,
            contributor_id_hash=stable_hash("contributor-quorum"),
            delta_manifest=delta,
            response={**response, "idempotent_replay": True},
            completed_at=1786396800.0,
            output_checkpoint=output,
        )


def test_controller_transport_commits_unaggregated_delta_without_checkpoint(
    tmp_path,
) -> None:
    wrapped = _ControllerTransportFixture(aggregate=False)
    transport = VolunteerControllerTransport(wrapped, tmp_path)
    claim = transport.claim(cell_id="cell-one", capability={})
    assert claim["v2_session"]["state"] == "work_active"
    response = transport.submit(
        cell_id="cell-one",
        work=claim["work_unit"],
        delta_manifest=_delta("unaggregated"),
    )
    assert response["v2_session"]["terminal_count"] == 1
    assert response["v2_session"]["checkpoint_count"] == 1
    status = SessionController(tmp_path).status()
    assert status["state"] == "idle"
    assert status["checkpoint_count"] == 1


def test_controller_transport_commits_quorum_lineage_and_replays_exactly_once(
    tmp_path,
) -> None:
    wrapped = _ControllerTransportFixture(aggregate=True)
    transport = VolunteerControllerTransport(wrapped, tmp_path)
    claim = transport.claim(cell_id="cell-two", capability={})
    delta = _delta("aggregated")
    first = transport.submit(
        cell_id="cell-two", work=claim["work_unit"], delta_manifest=delta
    )
    assert first["round_aggregated"] is True
    assert first["v2_session"]["checkpoint_count"] == 2

    replay = transport.submit(
        cell_id="cell-two", work=claim["work_unit"], delta_manifest=delta
    )
    assert replay["idempotent_replay"] is True
    assert replay["v2_session"]["idempotent_replay"] is True
    assert SessionController(tmp_path).status()["terminal_count"] == 1


def test_controller_transport_rejects_forged_quorum_without_lineage_advance(
    tmp_path,
) -> None:
    wrapped = _ControllerTransportFixture(aggregate=False)
    transport = VolunteerControllerTransport(wrapped, tmp_path)
    claim = transport.claim(cell_id="cell-three", capability={})

    original_submit = wrapped.submit

    def forged_submit(**kwargs):
        response = original_submit(**kwargs)
        return {**response, "round_aggregated": True, "adapter_version_after": 1}

    wrapped.submit = forged_submit
    with pytest.raises(
        ValueError, match="aggregated_lineage_advance_invalid"
    ):
        transport.submit(
            cell_id="cell-three",
            work=claim["work_unit"],
            delta_manifest=_delta("forged"),
        )


def test_controller_transport_persists_two_concurrent_cell_leases(tmp_path) -> None:
    wrapped = _ConcurrentControllerTransportFixture()
    transport = VolunteerControllerTransport(wrapped, tmp_path)
    first = transport.claim(cell_id="cell-one", capability={})
    second = transport.claim(cell_id="cell-two", capability={})
    assert first["work_unit"]["work_id"] != second["work_unit"]["work_id"]
    status = SessionController(tmp_path).status()
    assert status["active_work_count"] == 2
    assert status["concurrent_elastic_work_supported"] is True
    assert wrapped.claim_count == 2


def test_controller_transport_recovers_active_cell_owner_after_restart(tmp_path) -> None:
    wrapped = _ConcurrentControllerTransportFixture()
    transport = VolunteerControllerTransport(wrapped, tmp_path)
    first = transport.claim(cell_id="cell-one", capability={})
    second = transport.claim(cell_id="cell-two", capability={})

    restarted = VolunteerControllerTransport(wrapped, tmp_path)
    first_replay = restarted.claim(cell_id="cell-one", capability={})
    second_replay = restarted.claim(cell_id="cell-two", capability={})
    assert first_replay["work_unit"]["work_unit_hash"] == first["work_unit"]["work_unit_hash"]
    assert second_replay["work_unit"]["work_unit_hash"] == second["work_unit"]["work_unit_hash"]
    assert first_replay["v2_session"]["idempotent_replay"] is True
    assert second_replay["v2_session"]["idempotent_replay"] is True
    assert restarted.controller.status()["active_work_count"] == 2


def test_controller_transport_serializes_and_persists_concurrent_remote_claims(tmp_path) -> None:
    wrapped = _ConcurrentControllerTransportFixture()
    original_claim = wrapped.claim

    def slow_claim(**kwargs):
        time.sleep(0.05)
        return original_claim(**kwargs)

    wrapped.claim = slow_claim
    first = VolunteerControllerTransport(wrapped, tmp_path)
    second = VolunteerControllerTransport(wrapped, tmp_path)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def claim(transport, cell_id):
        barrier.wait()
        response = transport.claim(cell_id=cell_id, capability={})
        outcomes.append(response["work_unit"]["work_id"])

    threads = [
        threading.Thread(target=claim, args=(first, "cell-one")),
        threading.Thread(target=claim, args=(second, "cell-two")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    assert all(not thread.is_alive() for thread in threads)
    assert sorted(outcomes) == ["work-000001", "work-000002"]
    assert wrapped.claim_count == 2
    assert SessionController(tmp_path).status()["active_work_count"] == 2


def test_controller_transport_concurrent_submissions_commit_two_receipts_once(
    tmp_path,
) -> None:
    wrapped = _ConcurrentControllerTransportFixture()
    first = VolunteerControllerTransport(wrapped, tmp_path)
    second = VolunteerControllerTransport(wrapped, tmp_path)
    claims = {
        "cell-one": first.claim(cell_id="cell-one", capability={})["work_unit"],
        "cell-two": second.claim(cell_id="cell-two", capability={})["work_unit"],
    }
    barrier = threading.Barrier(2)
    responses: list[dict] = []

    def submit(transport, cell_id):
        barrier.wait()
        responses.append(
            transport.submit(
                cell_id=cell_id,
                work=claims[cell_id],
                delta_manifest=_delta(cell_id),
            )
        )

    threads = [
        threading.Thread(target=submit, args=(first, "cell-one")),
        threading.Thread(target=submit, args=(second, "cell-two")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    assert all(not thread.is_alive() for thread in threads)
    assert len(responses) == 2
    assert sum(item["round_aggregated"] is True for item in responses) == 1
    status = SessionController(tmp_path).status()
    assert status["active_work_count"] == 0
    assert status["terminal_count"] == 2
    assert status["checkpoint_count"] == 2


def test_controller_transport_recovers_remote_commit_before_local_commit(tmp_path) -> None:
    wrapped = _ControllerTransportFixture(aggregate=True)
    transport = VolunteerControllerTransport(wrapped, tmp_path)
    claim = transport.claim(cell_id="cell-one", capability={})
    delta = _delta("crash-window")
    remote = wrapped.submit(
        cell_id="cell-one", work=claim["work_unit"], delta_manifest=delta
    )
    assert remote["round_aggregated"] is True
    assert SessionController(tmp_path).status()["checkpoint_count"] == 1

    restarted = VolunteerControllerTransport(wrapped, tmp_path)
    recovered = restarted.submit(
        cell_id="cell-one", work=claim["work_unit"], delta_manifest=delta
    )
    assert recovered["idempotent_replay"] is True
    assert recovered["v2_session"]["checkpoint_count"] == 2
    assert recovered["v2_session"]["terminal_count"] == 1


def test_real_cpu_peft_cells_advance_v2_controller_lineage(tmp_path) -> None:
    fixture = create_local_training_fixture(
        tmp_path / "fixture", row_count=8, local_steps=1
    )
    coordinator = VolunteerTrainingCoordinator.create_from_fixture(
        tmp_path / "campaign",
        fixture,
        target_rounds=1,
        minimum_quorum=2,
        lease_seconds=120,
    )
    token = coordinator.private_invite()["invite_token"]
    transport = VolunteerControllerTransport(
        LocalVolunteerTransport(coordinator, token), tmp_path / "v2-workspace"
    )

    capability = {
        "selected_device": "cpu",
        "max_local_steps": 1,
        "real_peft_lora": True,
    }
    transport.claim(cell_id="cell-one", capability=capability)
    transport.claim(cell_id="cell-two", capability=capability)
    assert SessionController(tmp_path / "v2-workspace").status()[
        "active_work_count"
    ] == 2

    first = VolunteerTrainingCell(
        transport,
        tmp_path / "cell-one",
        cell_id="cell-one",
        device="cpu",
        max_local_steps=1,
    ).join_once()
    assert first["work_completed"] is True
    assert first["submission"]["round_aggregated"] is False
    assert first["submission"]["v2_session"]["checkpoint_count"] == 1

    second = VolunteerTrainingCell(
        transport,
        tmp_path / "cell-two",
        cell_id="cell-two",
        device="cpu",
        max_local_steps=1,
    ).join_once()
    assert second["work_completed"] is True
    assert second["submission"]["round_aggregated"] is True
    assert second["submission"]["v2_session"]["checkpoint_count"] == 2
    status = SessionController(tmp_path / "v2-workspace").status()
    assert status["terminal_count"] == 2
    assert status["checkpoint_count"] == 2
    assert coordinator.checkpoint_lineage()["completed_round_count"] == 1
