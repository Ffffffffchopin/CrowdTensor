from __future__ import annotations

import json
from urllib.error import URLError

import numpy as np
import pytest
import crowdtensor.cuda_training_worker as worker

from crowdtensor.cuda_training_worker import (
    _localized_training_spec,
    _payload_tensor,
    _request_json,
    _tensor_payload,
)


def test_cross_node_tensor_transport_round_trips_exact_values() -> None:
    value = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
    encoded, digest, shape, dtype = _tensor_payload(value)
    restored = _payload_tensor(encoded)
    assert digest.startswith("sha256:")
    assert shape == [2, 3, 4]
    assert dtype == "float16"
    assert np.array_equal(restored, value)


def test_remote_fixture_localization_preserves_claim_identity(tmp_path) -> None:
    fixture = tmp_path / "fixture"
    (fixture / "base_model").mkdir(parents=True)
    (fixture / "initial_adapter").mkdir()
    (fixture / "initial_adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
    (fixture / "initial_adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (fixture / "private_dataset.jsonl").write_text("{}\n", encoding="utf-8")
    original = {"claim_hash": "sha256:claim", "device": "cuda:0"}
    localized = _localized_training_spec(original, fixture)
    assert localized["claim_hash"] == original["claim_hash"]
    assert localized["device"] == "cuda:0"
    assert localized["base_model_path"].endswith("base_model")


class _JSONResponse:
    def __init__(self, value: dict) -> None:
        self.payload = json.dumps(value).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self.payload


def test_idempotent_coordinator_request_retries_transient_dns(monkeypatch) -> None:
    calls = []

    def fake_urlopen(_request, *, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise URLError("temporary DNS failure")
        return _JSONResponse({"ok": True})

    monkeypatch.setattr(worker, "urlopen", fake_urlopen)
    monkeypatch.setattr(worker.time, "sleep", lambda _seconds: None)
    value = _request_json(
        "POST",
        "https://private-route.invalid",
        "/cuda-training/register",
        token="private-token",
        payload={"run_id": "run-1"},
        transient_attempts=3,
    )
    assert value == {"ok": True}
    assert len(calls) == 2


def test_coordinator_request_does_not_retry_by_default(monkeypatch) -> None:
    calls = []

    def fake_urlopen(_request, *, timeout):
        calls.append(timeout)
        raise URLError("temporary DNS failure")

    monkeypatch.setattr(worker, "urlopen", fake_urlopen)
    with pytest.raises(URLError):
        _request_json(
            "POST",
            "https://private-route.invalid",
            "/tasks/claim",
            token="private-token",
            payload={"miner_id": "stage0"},
        )
    assert len(calls) == 1
