from __future__ import annotations

import base64
import json

import torch
from fastapi.testclient import TestClient

from coordinator import create_app
from crowdtensor.hf_lora_training import create_local_training_fixture
from crowdtensor.named_tensor_optimizer import load_tensors, save_tensors
from crowdtensor.training_contract import (
    RESULT_SCHEMA,
    delta_manifest,
    public_training_spec,
    sha256_json,
)


def test_remote_cuda_delta_upload_uses_existing_state_store_aggregation(tmp_path) -> None:
    fixture = create_local_training_fixture(
        tmp_path / "fixture",
        job_id="cuda-remote-test",
        row_count=8,
        sequence_length=8,
        local_steps=2,
    )
    job_path = tmp_path / "fixture" / "training_job_private.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["backend"] = "pytorch_transformers_peft_cuda"
    job["job_hash"] = sha256_json(public_training_spec(job))
    job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    app = create_app(
        state_dir=tmp_path / "state",
        backlog=0,
        task_lanes=[],
        hf_lora_job_manifest=job_path,
        miner_token="secret",
    )
    client = TestClient(app)
    headers = {"x-crowdtensor-miner-token": "secret"}
    initial = load_tensors(job["lora"]["adapter_tensor_path"])
    responses = []
    for index in range(2):
        claim_response = client.post(
            "/tasks/claim",
            headers=headers,
            json={
                "miner_id": f"cuda-miner-{index}",
                "capabilities": {
                    "runtime": "python-cli",
                    "backend": "cuda",
                    "protocol_version": "runtime_contract_v1",
                    "supported_workloads": ["hf_lora_train"],
                },
            },
        )
        assert claim_response.status_code == 200
        claim = claim_response.json()
        spec = claim["workload_spec"]
        delta_path = save_tensors(
            {name: torch.full_like(value, 0.001 * (index + 1)) for name, value in initial.items()},
            tmp_path / f"remote-{index}.safetensors",
        )
        result_id = f"result-{index}"
        manifest = delta_manifest(
            delta_path=delta_path,
            job_id=spec["job_id"],
            round_id=spec["round_id"],
            result_id=result_id,
            miner_id=spec["miner_id"],
            model_manifest_hash=spec["model_manifest_hash"],
            base_model_hash=spec["base_model_hash"],
            base_adapter_hash=spec["base_adapter_hash"],
            base_model_version=spec["base_model_version"],
            adapter_version=spec["adapter_version"],
            dataset_shard_index=spec["dataset_shard_index"],
            dataset_shard_hash=spec["dataset_shard_hash"],
            loss_start=2.0,
            loss_end=1.0,
            samples_seen=4,
            tokens_seen=32,
        )
        encoded = base64.b64encode(delta_path.read_bytes()).decode("ascii")
        delta_path.unlink()
        result = {
            "schema": RESULT_SCHEMA,
            "result_id": result_id,
            "task_id": claim["task_id"],
            "claim_hash": spec["claim_hash"],
            "dataset_shard_index": spec["dataset_shard_index"],
            "dataset_shard_hash": spec["dataset_shard_hash"],
            "adapter_delta": manifest,
            "real_backward": True,
            "base_weights_frozen": True,
            "only_lora_trainable": True,
            "runtime": {
                "real_pytorch_autograd": True,
                "real_transformers": True,
                "real_peft_lora": True,
                "cuda_used": True,
                "gpu_live_verified": True,
            },
        }
        response = client.post(
            f"/tasks/{claim['task_id']}/result",
            headers=headers,
            json={
                "lease_token": claim["lease_token"],
                "attempt": claim["attempt"],
                "training_result": result,
                "training_adapter_delta_b64": encoded,
            },
        )
        assert response.status_code == 200, response.text
        responses.append(response.json())
    assert responses[0]["accepted"] is True
    assert responses[1]["training_updated"] is True
    assert responses[1]["adapter_version"] == 1
    assert responses[1]["outer_step"] == 1
    assert app.state.store.training_state["round_status"] == "aggregated"
    status = client.get("/cuda-training/status", headers=headers)
    assert status.status_code == 200
    assert status.json()["public_artifact_safe"] is True
