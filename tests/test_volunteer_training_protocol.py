from __future__ import annotations

import copy

import pytest

from crowdtensor.training_contract import sha256_json
from crowdtensor.volunteer_training_protocol import (
    CAMPAIGN_SCHEMA,
    PROTOCOL_VERSION,
    SUBMISSION_SCHEMA,
    VolunteerProtocolError,
    campaign_content_hash,
    decode_submission_envelope,
    encode_submission_envelope,
    public_safe,
    validate_campaign_manifest,
    with_public_safety,
)


def _campaign() -> dict:
    value = with_public_safety(
        {
            "schema": CAMPAIGN_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "campaign_id": "test-campaign",
            "campaign_revision": 1,
            "model_manifest_hash": "sha256:" + "1" * 64,
            "dataset_snapshot_hash": "sha256:" + "2" * 64,
            "initial_adapter_hash": "sha256:" + "3" * 64,
            "adapter_tensor_contract_hash": "sha256:" + "4" * 64,
            "dataset_shards": [
                {"shard_index": 0, "shard_hash": "sha256:" + "5" * 64, "sample_count": 2},
                {"shard_index": 1, "shard_hash": "sha256:" + "6" * 64, "sample_count": 2},
            ],
            "local_training": {
                "local_steps": 4,
                "max_local_steps": 16,
                "learning_rate": 0.01,
            },
            "round_policy": {
                "minimum_quorum": 2,
                "target_rounds": 2,
                "lease_seconds": 30,
            },
            "outer_optimizer": {
                "optimizer_type": "diloco_momentum",
                "outer_lr": 0.5,
                "momentum": 0.9,
            },
            "update_admission": {
                "clip_delta_norm": 1.0,
                "hard_max_delta_norm": 10.0,
            },
        }
    )
    value["manifest_hash"] = campaign_content_hash(value)
    return value


def test_campaign_contract_is_versioned_and_hash_bound() -> None:
    campaign = _campaign()
    assert validate_campaign_manifest(campaign)["campaign_id"] == "test-campaign"
    changed = copy.deepcopy(campaign)
    changed["local_training"]["local_steps"] = 5
    with pytest.raises(VolunteerProtocolError, match="manifest_hash_mismatch"):
        validate_campaign_manifest(changed)


def test_binary_submission_envelope_is_bounded_and_round_trips() -> None:
    metadata = {"schema": SUBMISSION_SCHEMA, "result_id": sha256_json({"result": 1})}
    encoded = encode_submission_envelope(metadata, b"safetensors")
    decoded, delta = decode_submission_envelope(encoded, max_delta_bytes=1024)
    assert decoded == metadata
    assert delta == b"safetensors"
    with pytest.raises(VolunteerProtocolError, match="too_large"):
        decode_submission_envelope(encoded, max_delta_bytes=2)


def test_public_projection_removes_paths_urls_and_tokens_recursively() -> None:
    value = public_safe(
        {
            "invite_token": "secret",
            "coordinator_url": "https://private",
            "nested": {"delta_path": "/private/delta", "hash": "safe"},
            "items": [{"lease_token": "secret", "metric": 1}],
        }
    )
    assert value == {"nested": {"hash": "safe"}, "items": [{"metric": 1}]}
