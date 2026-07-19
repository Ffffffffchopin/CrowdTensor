from __future__ import annotations

import base64
import hashlib
import json

from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from crowdtensor.cuda_training_rendezvous import (
    CUDATrainingRendezvous,
    install_cuda_training_routes,
)


def _encoded(value: bytes) -> tuple[str, str]:
    return base64.b64encode(value).decode("ascii"), "sha256:" + hashlib.sha256(value).hexdigest()


def test_rendezvous_keeps_activation_and_gradient_values_out_of_public_status() -> None:
    rendezvous = CUDATrainingRendezvous(run_id="run-1")
    for role, device in (("stage0", 0), ("stage1", 0)):
        rendezvous.register(
            {
                "run_id": "run-1",
                "role": role,
                "worker_id_hash": f"sha256:{role}",
                "pid": device + 100,
                "cuda_device_index": device,
                "cuda_device_name_hash": "sha256:t4",
                "cuda_live": True,
            }
        )
    for kind, role, raw in (("activation", "stage0", b"private-a"), ("gradient", "stage1", b"private-g")):
        encoded, digest = _encoded(raw)
        rendezvous.put_payload(
            {
                "run_id": "run-1",
                "kind": kind,
                "step": 0,
                "producer_role": role,
                "payload_b64": encoded,
                "payload_hash": digest,
                "shape": [1, 2, 3],
                "dtype": "float16",
            }
        )
        assert base64.b64decode(rendezvous.get_payload(run_id="run-1", kind=kind, step=0)["payload_b64"]) == raw
    public = rendezvous.public_status()
    encoded_public = json.dumps(public)
    assert "private-a" not in encoded_public
    assert "private-g" not in encoded_public
    assert "payload_b64" not in encoded_public
    assert public["activation_values_public"] is False
    assert public["gradient_values_public"] is False


def test_installed_routes_require_auth_and_return_private_global_adapter(tmp_path) -> None:
    adapter = tmp_path / "adapter.safetensors"
    config = tmp_path / "adapter_config.json"
    adapter.write_bytes(b"adapter-bytes")
    config.write_text("{}\n", encoding="utf-8")

    class Store:
        training_state = {
            "round_status": "aggregated",
            "global_adapter_path": str(adapter),
            "adapter_version": 1,
            "outer_step": 1,
        }

    def authorize(token: str | None) -> None:
        if token != "secret":
            raise HTTPException(status_code=401, detail="invalid miner token")

    app = FastAPI()
    rendezvous = CUDATrainingRendezvous(run_id="run-1")
    install_cuda_training_routes(
        app,
        rendezvous=rendezvous,
        authorize=authorize,
        store=Store(),
        adapter_config_path=config,
    )
    client = TestClient(app)
    assert client.get("/cuda-training/status").status_code == 401
    headers = {"x-crowdtensor-miner-token": "secret"}
    response = client.get("/cuda-training/global-adapter?run_id=run-1", headers=headers)
    assert response.status_code == 200
    payload = response.json()
    assert base64.b64decode(payload["adapter_b64"]) == b"adapter-bytes"
    status = client.get("/cuda-training/status", headers=headers).json()
    assert "adapter_b64" not in json.dumps(status)
