import base64

import pytest
import torch
from fastapi.testclient import TestClient

from crowdtensor.community_live_training import (
    CommunityLiveCoordinator,
    _encode_tensor,
    create_live_app,
)
from crowdtensor.community_security import scan_public_value


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def complete_one_step(coordinator: CommunityLiveCoordinator) -> None:
    stage0 = coordinator.register(worker_id_hash=HASH_A, role="stage0", backend="cuda")
    stage1 = coordinator.register(worker_id_hash=HASH_B, role="stage1", backend="cpu")

    forward = coordinator.claim(worker_id_hash=HASH_A, role="stage0", generation=stage0["worker_generation"])
    activation_b64, activation_hash = _encode_tensor(torch.ones((1, 4, 8)))
    coordinator.submit(
        worker_id_hash=HASH_A,
        lease=forward["lease"],
        value={"activation_b64": activation_b64, "activation_hash": activation_hash},
    )
    backward = coordinator.claim(worker_id_hash=HASH_B, role="stage1", generation=stage1["worker_generation"])
    gradient_b64, gradient_hash = _encode_tensor(torch.full((1, 4, 8), 0.5))
    coordinator.submit(
        worker_id_hash=HASH_B,
        lease=backward["lease"],
        value={"gradient_b64": gradient_b64, "gradient_hash": gradient_hash, "loss": 1.25},
    )
    stage0_backward = coordinator.claim(worker_id_hash=HASH_A, role="stage0", generation=stage0["worker_generation"])
    coordinator.submit(
        worker_id_hash=HASH_A,
        lease=stage0_backward["lease"],
        value={"gradient_ready": True},
    )
    checkpoint = b"safe-stage-checkpoint"
    checkpoint_value = {
        "payload_b64": base64.b64encode(checkpoint).decode(),
        "payload_hash": "sha256:" + __import__("hashlib").sha256(checkpoint).hexdigest(),
        "adapter_hash": HASH_A,
    }
    commit0 = coordinator.claim(worker_id_hash=HASH_A, role="stage0", generation=stage0["worker_generation"])
    coordinator.submit(
        worker_id_hash=HASH_A,
        lease=commit0["lease"],
        value={"adapter_hash": HASH_A, "checkpoint": checkpoint_value},
    )
    commit1 = coordinator.claim(worker_id_hash=HASH_B, role="stage1", generation=stage1["worker_generation"])
    coordinator.submit(
        worker_id_hash=HASH_B,
        lease=commit1["lease"],
        value={"adapter_hash": HASH_B, "checkpoint": {**checkpoint_value, "adapter_hash": HASH_B}},
    )


def test_atomic_state_machine_persists_contiguous_step_and_private_payloads(tmp_path) -> None:
    path = tmp_path / "private" / "state.json"
    coordinator = CommunityLiveCoordinator(
        path,
        run_id="run-one",
        target_steps=1,
        sequence_length=4,
        checkpoint_steps=(1,),
    )
    complete_one_step(coordinator)
    status = coordinator.public_status()
    assert status["completed"] is True
    assert status["committed_step_ids"] == [1]
    assert status["strictly_contiguous_steps"] is True
    assert status["finite_losses"] is True
    assert set(status["checkpoint_summary"]) == {"stage0", "stage1"}
    assert coordinator.state["ledger"][0]["activation_bytes"] > 0
    assert coordinator.state["ledger"][0]["gradient_bytes"] > 0
    checkpoint_events = [
        item
        for item in coordinator.state["events"]
        if item["operation"] == "checkpoint_committed"
    ]
    assert len(checkpoint_events) == 2
    assert all(item["payload_bytes"] > 0 for item in checkpoint_events)
    assert "payload_b64" not in str(status)
    assert scan_public_value(status)["ok"] is True

    reopened = CommunityLiveCoordinator(path, run_id="run-one")
    restart = reopened.record_restart()
    assert restart["journal_recovered"] is True
    assert restart["committed_step"] == 1
    assert reopened.public_status()["coordinator_generation"] == 2


def test_stale_lease_and_worker_replacement_are_recorded(tmp_path) -> None:
    coordinator = CommunityLiveCoordinator(tmp_path / "state.json", run_id="run", target_steps=2)
    first = coordinator.register(worker_id_hash=HASH_A, role="stage0", backend="cuda")
    task = coordinator.claim(worker_id_hash=HASH_A, role="stage0", generation=first["worker_generation"])
    coordinator.record_restart()
    with pytest.raises(RuntimeError, match="duplicate_or_stale"):
        coordinator.submit(worker_id_hash=HASH_A, lease=task["lease"], value={})
    replacement = coordinator.register(
        worker_id_hash="sha256:" + "c" * 64,
        role="stage0",
        backend="cuda",
    )
    assert replacement["worker_generation"] == 2
    worker = next(item for item in coordinator.public_status()["workers"] if item["role"] == "stage0")
    assert worker["replacement"] is True


def test_restart_barrier_pauses_claims_at_atomic_step_boundary(tmp_path) -> None:
    coordinator = CommunityLiveCoordinator(
        tmp_path / "state.json",
        run_id="restart-barrier",
        target_steps=2,
        checkpoint_steps=(1, 2),
    )
    complete_one_step(coordinator)
    barrier = coordinator.request_restart_barrier(after_step=1)
    assert barrier == {
        "requested_after_step": 1,
        "ready": True,
        "committed_step": 1,
        "phase": "stage0_forward",
    }
    stage0 = next(item for item in coordinator.public_status()["workers"] if item["role"] == "stage0")
    blocked = coordinator.claim(
        worker_id_hash=stage0["worker_id_hash"],
        role="stage0",
        generation=stage0["generation"],
    )
    assert blocked["task_available"] is False
    assert blocked["restart_barrier"] is True

    reopened = CommunityLiveCoordinator(tmp_path / "state.json", run_id="restart-barrier")
    reopened.record_restart()
    resumed = reopened.claim(
        worker_id_hash=stage0["worker_id_hash"],
        role="stage0",
        generation=stage0["generation"],
    )
    assert resumed["task_available"] is True
    assert resumed["step"] == 2


def test_worker_step_limit_stops_before_leasing_next_step(tmp_path) -> None:
    coordinator = CommunityLiveCoordinator(
        tmp_path / "state.json",
        run_id="worker-limit",
        target_steps=2,
        checkpoint_steps=(1, 2),
    )
    complete_one_step(coordinator)
    stage0 = next(
        item for item in coordinator.public_status()["workers"] if item["role"] == "stage0"
    )
    stopped = coordinator.claim(
        worker_id_hash=stage0["worker_id_hash"],
        role="stage0",
        generation=stage0["generation"],
        max_committed_step=1,
    )
    assert stopped["worker_stop_requested"] is True
    assert stopped["committed_step"] == 1
    assert stopped["task_available"] is False
    assert coordinator.state["leases"] == {}


def test_live_api_authenticates_status_and_wheel_download(tmp_path) -> None:
    wheel = tmp_path / "crowdtensor.whl"
    wheel.write_bytes(b"wheel-bytes")
    coordinator = CommunityLiveCoordinator(tmp_path / "state.json", run_id="api", target_steps=1)
    client = TestClient(create_live_app(coordinator, miner_token="private-token", wheel_path=wheel))
    assert client.get("/health").status_code == 200
    assert client.get("/v1/community-live/status").status_code == 401
    response = client.get(
        "/v1/community-live/wheel",
        headers={"x-crowdtensor-miner-token": "private-token"},
    )
    assert response.status_code == 200
    assert response.content == b"wheel-bytes"
    assert response.headers["x-crowdtensor-wheel-sha256"].startswith("sha256:")
    assert response.headers["x-crowdtensor-wheel-filename"] == "crowdtensor.whl"
    assert (
        client.get(
            "/v1/community-live/adapter-wheel",
            headers={"x-crowdtensor-miner-token": "private-token"},
        ).status_code
        == 503
    )


def test_live_api_authenticates_adapter_plugin_wheel_download(tmp_path) -> None:
    wheel = tmp_path / "crowdtensord-1-py3-none-any.whl"
    adapter_wheel = tmp_path / "crowdtensor_mistral_adapter-1-py3-none-any.whl"
    wheel.write_bytes(b"core-wheel")
    adapter_wheel.write_bytes(b"adapter-wheel")
    coordinator = CommunityLiveCoordinator(
        tmp_path / "state.json", run_id="adapter-api", target_steps=1
    )
    client = TestClient(
        create_live_app(
            coordinator,
            miner_token="private-token",
            wheel_path=wheel,
            adapter_wheel_path=adapter_wheel,
        )
    )
    assert client.get("/v1/community-live/adapter-wheel").status_code == 401
    response = client.get(
        "/v1/community-live/adapter-wheel",
        headers={"x-crowdtensor-miner-token": "private-token"},
    )
    assert response.status_code == 200
    assert response.content == b"adapter-wheel"
    assert response.headers["x-crowdtensor-wheel-sha256"].startswith("sha256:")
    assert response.headers["x-crowdtensor-wheel-filename"] == adapter_wheel.name
    assert response.headers["x-crowdtensor-wheel-kind"] == "model-adapter-plugin"
