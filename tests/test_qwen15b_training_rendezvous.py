from __future__ import annotations

import base64
import hashlib
import json
import stat

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from crowdtensor.qwen15b_training_rendezvous import (
    Qwen15BTrainingRendezvous,
    install_qwen15b_training_routes,
)


def _hash(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _registration(role: str) -> dict:
    return {
        "run_id": "private-run-id",
        "role": role,
        "worker_id_hash": f"sha256:{role}",
        "stage_ids": [0, 1] if role == "kernel_a" else [2, 3],
        "stage_pids": [101, 102] if role == "kernel_a" else [201, 202],
        "cuda_devices": ["cuda:0", "cuda:1"],
        "cuda_device_name_hashes": ["sha256:gpu0", "sha256:gpu1"],
        "cuda_live": True,
    }


def test_rendezvous_keeps_tensor_and_token_payload_private() -> None:
    rendezvous = Qwen15BTrainingRendezvous(run_id="private-run-id")
    rendezvous.register(_registration("kernel_a"))
    rendezvous.register(_registration("kernel_b"))
    raw = b"private activation and token ids 151643,42,7"
    payload = {
        "run_id": "private-run-id",
        "run_kind": "baseline",
        "kind": "activation",
        "step": 1,
        "microbatch": 2,
        "producer_role": "kernel_a",
        "payload_b64": base64.b64encode(raw).decode("ascii"),
        "payload_hash": _hash(raw),
        "tensor_count": 2,
    }
    assert rendezvous.put_payload(payload)["ok"] is True
    private = rendezvous.get_payload(
        run_id="private-run-id",
        run_kind="baseline",
        kind="activation",
        step=1,
        microbatch=2,
    )
    assert base64.b64decode(private["payload_b64"]) == raw
    status = rendezvous.public_status()
    encoded = json.dumps(status, sort_keys=True)
    assert "payload_b64" not in encoded
    assert "private activation" not in encoded
    assert "151643" not in encoded
    assert "private-run-id" not in encoded
    assert status["activation_values_public"] is False
    assert status["gradient_values_public"] is False
    assert status["token_ids_public"] is False
    cleanup = rendezvous.cleanup()
    assert cleanup["private_payload_count_removed"] == 1
    assert cleanup["private_payloads_removed"] is True


def test_rendezvous_rejects_wrong_producer_conflict_and_private_event_fields() -> None:
    rendezvous = Qwen15BTrainingRendezvous(run_id="run")
    raw = b"value"
    payload = {
        "run_id": "run",
        "run_kind": "resumed",
        "kind": "gradient",
        "step": 4,
        "microbatch": 0,
        "producer_role": "kernel_a",
        "payload_b64": base64.b64encode(raw).decode("ascii"),
        "payload_hash": _hash(raw),
        "tensor_count": 1,
    }
    with pytest.raises(ValueError, match="producer_invalid"):
        rendezvous.put_payload(payload)
    payload["producer_role"] = "kernel_b"
    rendezvous.put_payload(payload)
    changed = b"changed"
    payload["payload_b64"] = base64.b64encode(changed).decode("ascii")
    payload["payload_hash"] = _hash(changed)
    with pytest.raises(ValueError, match="payload_conflict"):
        rendezvous.put_payload(payload)
    with pytest.raises(ValueError, match="event_invalid"):
        rendezvous.add_event(
            {
                "run_id": "run",
                "role": "kernel_b",
                "run_kind": "resumed",
                "operation": "raw_activation",
                "stage_id": 2,
            }
        )


def test_authenticated_routes_round_trip_private_payload_without_status_leak() -> None:
    app = FastAPI()
    rendezvous = Qwen15BTrainingRendezvous(run_id="run-route")

    def authorize(value: str | None) -> None:
        if value != "secret":
            raise HTTPException(status_code=401, detail="unauthorized")

    install_qwen15b_training_routes(app, rendezvous=rendezvous, authorize=authorize)
    client = TestClient(app)
    headers = {"x-crowdtensor-miner-token": "secret"}
    assert client.get("/qwen15b-training/status").status_code == 401
    assert client.post(
        "/qwen15b-training/register",
        headers=headers,
        json={**_registration("kernel_a"), "run_id": "run-route"},
    ).status_code == 200
    raw = b"private-gradient"
    request = {
        "run_id": "run-route",
        "run_kind": "baseline",
        "kind": "gradient",
        "step": 0,
        "microbatch": 0,
        "producer_role": "kernel_b",
        "payload_b64": base64.b64encode(raw).decode("ascii"),
        "payload_hash": _hash(raw),
        "tensor_count": 1,
    }
    assert client.post("/qwen15b-training/payload", headers=headers, json=request).status_code == 200
    private = client.get(
        "/qwen15b-training/payload/baseline/gradient/0/0",
        headers=headers,
        params={"run_id": "run-route"},
    ).json()
    assert base64.b64decode(private["payload_b64"]) == raw
    status = client.get("/qwen15b-training/status", headers=headers).json()
    assert "payload_b64" not in json.dumps(status)


def test_persistent_rendezvous_recovers_payloads_and_tracks_restart_registration(
    tmp_path,
) -> None:
    state = tmp_path / "private" / "rendezvous.json"
    first = Qwen15BTrainingRendezvous(run_id="persistent-run", state_path=state)
    first.register({**_registration("kernel_a"), "run_id": "persistent-run"})
    first.register({**_registration("kernel_b"), "run_id": "persistent-run"})
    raw = b"persistent-private-activation"
    first.put_payload(
        {
            "run_id": "persistent-run",
            "run_kind": "resumed",
            "kind": "activation",
            "step": 3,
            "microbatch": 3,
            "producer_role": "kernel_a",
            "payload_b64": base64.b64encode(raw).decode("ascii"),
            "payload_hash": _hash(raw),
            "tensor_count": 1,
        }
    )
    assert stat.S_IMODE(state.stat().st_mode) == 0o600
    payload_files = list((state.parent / "rendezvous-payloads").glob("*.bin"))
    assert len(payload_files) == 1
    assert stat.S_IMODE(payload_files[0].stat().st_mode) == 0o600
    assert stat.S_IMODE(payload_files[0].parent.stat().st_mode) == 0o700
    first.begin_coordinator_restart(after_step=4)
    recovered = Qwen15BTrainingRendezvous(run_id="persistent-run", state_path=state)
    restart = recovered.complete_coordinator_restart()
    assert restart["after_step"] == 4
    private = recovered.get_payload(
        run_id="persistent-run",
        run_kind="resumed",
        kind="activation",
        step=3,
        microbatch=3,
    )
    assert base64.b64decode(private["payload_b64"]) == raw
    recovered.register({**_registration("kernel_a"), "run_id": "persistent-run"})
    recovered.register({**_registration("kernel_b"), "run_id": "persistent-run"})
    status = recovered.public_status()
    assert status["persistent_state_enabled"] is True
    assert status["recovered_from_persistent_state"] is True
    assert status["coordinator_generation"] == 1
    assert status["post_restart_registered_roles"] == ["kernel_a", "kernel_b"]
    assert status["coordinator_restart_verified"] is True
    encoded = json.dumps(status, sort_keys=True)
    assert "persistent-private-activation" not in encoded
    assert "payload_b64" not in encoded


def test_persistent_rendezvous_deduplicates_event_and_completion(tmp_path) -> None:
    rendezvous = Qwen15BTrainingRendezvous(
        run_id="idempotent-run",
        state_path=tmp_path / "private" / "rendezvous.json",
    )
    event = {
        "run_id": "idempotent-run",
        "role": "kernel_a",
        "run_kind": "resumed",
        "operation": "checkpoint",
        "stage_id": 0,
        "step": 4,
        "pid": 100,
        "checkpoint_hash": _hash(b"checkpoint"),
    }
    rendezvous.add_event(event)
    rendezvous.add_event(event)
    summary = {"ok": True, "baseline_steps_completed": 8, "resumed_steps_completed": 8}
    rendezvous.complete(
        {"run_id": "idempotent-run", "role": "kernel_a", "summary": summary}
    )
    rendezvous.complete(
        {"run_id": "idempotent-run", "role": "kernel_a", "summary": summary}
    )
    status = rendezvous.public_status()
    checkpoint_events = [
        item for item in status["events"] if item.get("operation") == "checkpoint"
    ]
    assert len(checkpoint_events) == 1
    assert len(status["completions"]) == 1
    with pytest.raises(ValueError, match="completion_conflict"):
        rendezvous.complete(
            {
                "run_id": "idempotent-run",
                "role": "kernel_a",
                "summary": {**summary, "ok": False},
            }
        )
