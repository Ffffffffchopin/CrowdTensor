from __future__ import annotations

import json

from crowdtensor.state_store import StateStore
from crowdtensor.training_contract import WORKLOAD_TYPE


def test_hf_lora_event_tail_redacts_private_training_paths_and_result(tmp_path) -> None:
    store = StateStore(tmp_path / "state", task_lanes=[])
    private_path = str(tmp_path / "private_dataset.jsonl")
    event = {
        "type": "task_completed",
        "workload_type": WORKLOAD_TYPE,
        "workload_metadata": {
            "job_id": "job",
            "job_manifest_path": str(tmp_path / "private-job.json"),
        },
        "claim_workload_spec": {
            "workload_type": WORKLOAD_TYPE,
            "job_id": "job",
            "dataset_path": private_path,
            "adapter_path": str(tmp_path / "adapter"),
            "dataset_shard_hash": "sha256:shard",
        },
        "training_result": {
            "adapter_delta": {"delta_path": str(tmp_path / "delta.safetensors")},
            "raw_text": "private training text",
        },
    }
    public = store._redact_event(event)
    serialized = json.dumps(public, sort_keys=True)
    assert public["training_result"] == "<redacted>"
    assert public["claim_workload_spec"]["dataset_shard_hash"] == "sha256:shard"
    assert public["claim_workload_spec"]["private_paths_public"] is False
    assert private_path not in serialized
    assert str(tmp_path) not in serialized
    assert "private training text" not in serialized
