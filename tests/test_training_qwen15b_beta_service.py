from __future__ import annotations

import json
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient

from crowdtensor.qwen15b_training import MODEL_ID, MODEL_REVISION
from crowdtensor.training_qwen15b_beta_service import (
    TrainingBetaController,
    TrainingBetaJobStore,
    create_training_beta_app,
)


def _request(tmp_path, name: str = "job") -> dict:
    return {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "topology": "kaggle-2x-t4x2",
        "steps": 8,
        "job_dir": str(tmp_path / name),
        "kaggle_token_files": [str(tmp_path / "private-token")],
        "allocation_timeout_seconds": 1800,
    }


def test_job_store_submit_is_idempotent_and_rejects_conflict(tmp_path) -> None:
    store = TrainingBetaJobStore(tmp_path / "jobs.sqlite3")
    first, created = store.submit(_request(tmp_path), idempotency_key="same")
    second, created_again = store.submit(_request(tmp_path), idempotency_key="same")
    assert created is True
    assert created_again is False
    assert first["job_id"] == second["job_id"]
    assert len(store.events(first["job_id"])) == 1
    changed = _request(tmp_path, "other")
    with pytest.raises(ValueError, match="idempotency_conflict"):
        store.submit(changed, idempotency_key="same")
    encoded = json.dumps(second, sort_keys=True)
    assert str(tmp_path) not in encoded
    assert "private-token" not in encoded


def test_job_store_allows_one_live_job_and_bounded_queue(tmp_path) -> None:
    store = TrainingBetaJobStore(tmp_path / "jobs.sqlite3", max_queue_size=2)
    first, _ = store.submit(_request(tmp_path, "one"), idempotency_key="one")
    second, _ = store.submit(_request(tmp_path, "two"), idempotency_key="two")
    claimed = store.claim_next(worker_id="worker")
    assert claimed["public"]["job_id"] == first["job_id"]
    assert store.claim_next(worker_id="other") is None
    assert store.status(second["job_id"])["queue_position"] == 1
    third, _ = store.submit(_request(tmp_path, "three"), idempotency_key="three")
    assert third["overall_state"] == "queued"
    with pytest.raises(RuntimeError, match="queue_full"):
        store.submit(_request(tmp_path, "four"), idempotency_key="four")


def test_expired_lease_recovers_without_global_step_regression(tmp_path) -> None:
    store = TrainingBetaJobStore(tmp_path / "jobs.sqlite3")
    submitted, _ = store.submit(_request(tmp_path), idempotency_key="recover")
    claimed = store.claim_next(worker_id="first", lease_seconds=1)
    running = {
        **claimed["public"],
        "overall_state": "running",
        "current_phase": "checkpoint",
        "global_step": 4,
    }
    store.update_status(submitted["job_id"], running, event_id="step-4")
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires_at=? WHERE job_id=?",
            (time.time() - 1, submitted["job_id"]),
        )
    recovered = store.claim_next(worker_id="second")
    assert recovered["public"]["current_phase"] == "recovery"
    assert recovered["public"]["global_step"] == 4
    assert recovered["public"]["retry_count"] == 1
    regressed = {**recovered["public"], "global_step": 3}
    with pytest.raises(ValueError, match="global_step_regression"):
        store.update_status(submitted["job_id"], regressed, event_id="bad-step")


def test_duplicate_status_event_does_not_repeat_revision_or_step(tmp_path) -> None:
    store = TrainingBetaJobStore(tmp_path / "jobs.sqlite3")
    submitted, _ = store.submit(_request(tmp_path), idempotency_key="duplicate")
    claimed = store.claim_next(worker_id="worker")["public"]
    value = {
        **claimed,
        "overall_state": "running",
        "current_phase": "forward",
        "global_step": 2,
    }
    first = store.update_status(submitted["job_id"], value, event_id="forward-step-2")
    second = store.update_status(submitted["job_id"], value, event_id="forward-step-2")
    assert second["revision"] == first["revision"]
    assert second["global_step"] == 2


def test_authenticated_api_uses_shared_controller_and_keeps_private_inputs_private(
    tmp_path,
) -> None:
    store = TrainingBetaJobStore(tmp_path / "jobs.sqlite3")

    def runner(request: dict) -> dict:
        assert request["job_dir"].endswith("api-job")
        return {
            "overall_state": "completed",
            "current_phase": "cleanup",
            "global_step": 8,
            "retry_count": 0,
            "ok": True,
            "phases": {"cleanup": {"state": "completed"}},
            "private_paths_public": False,
            "public_artifact_safe": True,
        }

    controller = TrainingBetaController(store, runner=runner)
    client = TestClient(create_training_beta_app(controller, token="service-secret"))
    assert client.get("/health").status_code == 200
    request = {
        **_request(tmp_path, "api-job"),
        "idempotency_key": "api-submit",
        "execute": False,
    }
    assert client.post("/v1/training/jobs", json=request).status_code == 401
    headers = {"x-crowdtensor-training-token": "service-secret"}
    submitted = client.post("/v1/training/jobs", json=request, headers=headers).json()
    repeated = client.post("/v1/training/jobs", json=request, headers=headers).json()
    assert submitted["job_id"] == repeated["job_id"]
    status = client.get(
        f"/v1/training/jobs/{submitted['job_id']}", headers=headers
    ).json()
    assert status["overall_state"] == "queued"
    encoded = json.dumps(status, sort_keys=True)
    assert "service-secret" not in encoded
    assert str(tmp_path) not in encoded
    cancelled = client.post(
        f"/v1/training/jobs/{submitted['job_id']}/cancel", headers=headers
    ).json()
    assert cancelled["overall_state"] == "cancelled"
    cancelled_again = client.post(
        f"/v1/training/jobs/{submitted['job_id']}/cancel", headers=headers
    ).json()
    assert cancelled_again["overall_state"] == "cancelled"
    assert cancelled_again["revision"] == cancelled["revision"]
    cancel_events = [
        item for item in store.events(submitted["job_id"])
        if item["event_id"] == "cancel-requested"
    ]
    assert len(cancel_events) == 1


def test_controller_process_restart_reads_persisted_job(tmp_path) -> None:
    path = tmp_path / "jobs.sqlite3"
    first = TrainingBetaController(TrainingBetaJobStore(path), runner=lambda _request: {})
    submitted = first.submit(_request(tmp_path), idempotency_key="restart")
    second = TrainingBetaController(TrainingBetaJobStore(path), runner=lambda _request: {})
    recovered = second.status(submitted["job_id"])
    assert recovered["job_id"] == submitted["job_id"]
    assert recovered["global_step"] == 0
    assert recovered["revision"] >= 1


def test_running_cancel_writes_private_marker_and_is_fully_idempotent(tmp_path) -> None:
    store = TrainingBetaJobStore(tmp_path / "jobs.sqlite3")
    submitted, _ = store.submit(_request(tmp_path), idempotency_key="running-cancel")
    store.claim_next(worker_id="worker", preferred_job_id=submitted["job_id"])
    controller = TrainingBetaController(store, runner=lambda _request: {})
    first = controller.cancel(submitted["job_id"])
    marker = tmp_path / "job" / ".private-service" / "cancel.requested"
    assert first["overall_state"] == "running"
    assert first["cancel_requested"] is True
    assert marker.read_text(encoding="utf-8") == "cancel requested\n"
    assert marker.stat().st_mode & 0o777 == 0o600
    second = controller.cancel(submitted["job_id"])
    assert second["revision"] == first["revision"]
    assert len(
        [event for event in store.events(submitted["job_id"]) if event["event_id"] == "cancel-requested"]
    ) == 1
    assert str(tmp_path) not in json.dumps(second, sort_keys=True)
