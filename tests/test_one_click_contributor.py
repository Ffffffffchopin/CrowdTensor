from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from crowdtensor.hf_lora_training import create_local_training_fixture
from crowdtensor.volunteer_agent_status import VolunteerAgentStatusServer
from crowdtensor.volunteer_browser_probe import browser_probe_digest
from crowdtensor.volunteer_training_api import create_volunteer_training_app
from crowdtensor.volunteer_training_cell import VolunteerTrainingCell
from crowdtensor.volunteer_training_coordinator import VolunteerTrainingCoordinator
from crowdtensor.volunteer_training_protocol import VolunteerProtocolError


def _coordinator(tmp_path) -> VolunteerTrainingCoordinator:
    root = tmp_path / "campaign"
    fixture = create_local_training_fixture(
        root / ".private" / "fixture",
        job_id="one-click-test",
        local_steps=1,
    )
    return VolunteerTrainingCoordinator.create_from_fixture(
        root,
        fixture,
        campaign_id="one-click-test",
        target_rounds=2,
        lease_seconds=120,
    )


def _headers(token: str, nonce: str) -> dict[str, str]:
    return {
        "Authorization": "Bearer " + token,
        "X-CrowdTensor-Nonce": nonce,
    }


def test_browser_pairing_and_server_recomputed_task_do_not_update_model(tmp_path) -> None:
    coordinator = _coordinator(tmp_path)
    invite = coordinator.private_invite()["invite_token"]
    before = coordinator.status()
    issued = coordinator.create_pairing_code(
        invite_token=invite, mode="browser", ttl_seconds=300
    )
    code = issued["pairing_code"]
    private_state = coordinator.state_path.read_text(encoding="utf-8")
    assert code not in private_state
    assert code.replace("-", "") not in private_state

    paired = coordinator.redeem_pairing_code(
        pairing_code=code, cell_id="browser-cell"
    )
    assert paired["pairing_mode"] == "browser"
    assert paired["scopes"] == [
        "browser:claim",
        "browser:heartbeat",
        "browser:submit",
    ]
    with pytest.raises(VolunteerProtocolError, match="pairing_code_consumed"):
        coordinator.redeem_pairing_code(
            pairing_code=code, cell_id="second-browser-cell"
        )

    claim = coordinator.claim_browser_probe(
        cell_id="browser-cell",
        credential_token=paired["credential_token"],
        capability={"webgpu_available": True, "model_training": False},
        request_nonce="browser-claim-nonce-0001",
    )
    task = claim["task"]
    heartbeat = coordinator.heartbeat_browser_probe(
        cell_id="browser-cell",
        credential_token=paired["credential_token"],
        task_id=task["task_id"],
        lease_generation=task["lease_generation"],
        lease_token=task["lease_token"],
        request_nonce="browser-heartbeat-nonce-0001",
    )
    assert heartbeat["heartbeat_count"] == 1
    digest = browser_probe_digest(
        seed=task["seed"],
        vector_length=task["vector_length"],
        rounds=task["rounds"],
    )
    accepted = coordinator.submit_browser_probe(
        cell_id="browser-cell",
        credential_token=paired["credential_token"],
        task_id=task["task_id"],
        lease_generation=task["lease_generation"],
        lease_token=task["lease_token"],
        output_sha256=digest,
        runtime="webgpu",
        duration_ms=41,
        request_nonce="browser-submit-nonce-0001",
    )
    after = coordinator.status()
    assert accepted["accepted"] is True
    assert accepted["server_recomputed"] is True
    assert accepted["model_update"] is False
    assert after["browser_calibration"]["accepted_task_count"] == 1
    assert after["browser_calibration"]["heartbeat_count"] == 1
    assert after["adapter_version"] == before["adapter_version"]
    assert after["canonical_adapter_hash"] == before["canonical_adapter_hash"]
    snapshot = coordinator.public_campaign_snapshot()
    assert snapshot["progress"]["accepted_browser_task_count"] == 1
    assert snapshot["browser_calibration"]["model_update_count"] == 0
    serialized = json.dumps(snapshot, sort_keys=True)
    assert code not in serialized
    assert paired["credential_token"] not in serialized
    assert task["task_id"] not in serialized


def test_browser_output_rejection_and_scope_separation(tmp_path) -> None:
    coordinator = _coordinator(tmp_path)
    invite = coordinator.private_invite()["invite_token"]
    browser_code = coordinator.create_pairing_code(
        invite_token=invite, mode="browser", ttl_seconds=300
    )["pairing_code"]
    browser = coordinator.redeem_pairing_code(
        pairing_code=browser_code, cell_id="browser-cell"
    )
    with pytest.raises(VolunteerProtocolError, match="scope_missing"):
        coordinator.claim(
            cell_id="browser-cell",
            invite_token=browser["credential_token"],
            request_nonce="native-claim-nonce-0001",
        )
    claim = coordinator.claim_browser_probe(
        cell_id="browser-cell",
        credential_token=browser["credential_token"],
        request_nonce="browser-claim-nonce-0002",
    )
    task = claim["task"]
    with pytest.raises(VolunteerProtocolError, match="output_invalid"):
        coordinator.submit_browser_probe(
            cell_id="browser-cell",
            credential_token=browser["credential_token"],
            task_id=task["task_id"],
            lease_generation=task["lease_generation"],
            lease_token=task["lease_token"],
            output_sha256="sha256:" + "0" * 64,
            runtime="wasm-cpu",
            duration_ms=10,
            request_nonce="browser-submit-nonce-0002",
        )
    assert coordinator.status()["browser_calibration"]["rejected_task_count"] == 1


def test_pairing_and_browser_http_contract(tmp_path) -> None:
    coordinator = _coordinator(tmp_path)
    invite = coordinator.private_invite()["invite_token"]
    code = coordinator.create_pairing_code(
        invite_token=invite, mode="browser", ttl_seconds=300
    )["pairing_code"]
    client = TestClient(create_volunteer_training_app(coordinator))
    paired_response = client.post(
        "/v1/volunteer/pairing/redeem",
        json={"pairing_code": code, "cell_id": "http-browser"},
    )
    assert paired_response.status_code == 200
    paired = paired_response.json()
    claim_response = client.post(
        "/v1/volunteer/browser/claim",
        headers=_headers(paired["credential_token"], "http-browser-claim-0001"),
        json={"cell_id": "http-browser", "capability": {"webgpu_available": False}},
    )
    assert claim_response.status_code == 200
    task = claim_response.json()["task"]
    heartbeat_response = client.post(
        "/v1/volunteer/browser/heartbeat",
        headers=_headers(paired["credential_token"], "http-browser-heartbeat-0001"),
        json={
            "cell_id": "http-browser",
            "task_id": task["task_id"],
            "lease_generation": task["lease_generation"],
            "lease_token": task["lease_token"],
        },
    )
    assert heartbeat_response.status_code == 200
    digest = browser_probe_digest(
        seed=task["seed"],
        vector_length=task["vector_length"],
        rounds=task["rounds"],
    )
    submit_response = client.post(
        "/v1/volunteer/browser/submit",
        headers=_headers(paired["credential_token"], "http-browser-submit-0001"),
        json={
            "cell_id": "http-browser",
            "task_id": task["task_id"],
            "lease_generation": task["lease_generation"],
            "lease_token": task["lease_token"],
            "output_sha256": digest,
            "runtime": "wasm-cpu",
            "duration_ms": 25,
        },
    )
    assert submit_response.status_code == 200
    assert submit_response.json()["accepted"] is True


class _StatusCell:
    def __init__(self) -> None:
        self.state = "ready"

    def local_status(self):
        return {"ok": True, "state": self.state, "credential_values_public": False}

    def pause(self):
        self.state = "paused"
        return self.local_status()

    def resume(self):
        self.state = "ready"
        return self.local_status()


def test_native_agent_status_page_is_loopback_and_controls_graceful_stop() -> None:
    cell = _StatusCell()
    with VolunteerAgentStatusServer(cell, port=0) as server:
        assert server.endpoint.startswith("http://127.0.0.1:")
        page = httpx.get(server.endpoint, timeout=5)
        assert page.status_code == 200
        assert "CrowdTensor native Agent" in page.text
        assert "credential" not in page.text.lower() or "no credential" in page.text.lower()
        paused = httpx.post(server.endpoint + "/pause", follow_redirects=True, timeout=5)
        assert paused.status_code == 200
        assert cell.state == "paused"
        stopped = httpx.post(server.endpoint + "/stop", follow_redirects=True, timeout=5)
        assert stopped.status_code == 200
        deadline = time.time() + 2
        while not server.stop_event.is_set() and time.time() < deadline:
            time.sleep(0.01)
        assert server.stop_event.is_set()


def test_one_click_agent_rejects_unwired_jax_tpu_policy(tmp_path, monkeypatch) -> None:
    cell = VolunteerTrainingCell(object(), tmp_path / "cell", device="jax_tpu")
    monkeypatch.setattr(
        cell,
        "hardware",
        lambda: {"cuda_available": False, "cuda_device_count": 0},
    )
    with pytest.raises(VolunteerProtocolError, match="volunteer_device_policy_invalid"):
        cell.selected_device()


def test_installer_selects_bounded_cpu_runtime_without_storage_extra() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "install_contributor.sh"
    ).read_text(encoding="utf-8")
    assert 'VERSION="0.3.0a1"' in script
    assert "https://download.pytorch.org/whl/cpu" in script
    assert 'TORCH_VERSION="${CROWDTENSOR_TORCH_VERSION:-2.11.0}"' in script
    assert 'TORCH_WHEEL_PATH="${CROWDTENSOR_TORCH_WHEEL_PATH:-}"' in script
    assert 'PIP_TIMEOUT="${CROWDTENSOR_PIP_TIMEOUT_SECONDS:-600}"' in script
    assert 'PIP_RETRIES="${CROWDTENSOR_PIP_RETRIES:-5}"' in script
    assert "command -v nvidia-smi" in script
    assert "CROWDTENSOR_TORCH_INDEX_URL" in script
    assert 'DEVICE="${CROWDTENSOR_DEVICE:-auto}"' in script
    assert '[ "${DEVICE}" != "cpu" ]' in script
    assert "CHECKSUMS_URL" in script
    assert "EXPECTED_SHA256" in script
    assert "ACTUAL_SHA256" in script
    assert "--continue-at -" in script
    assert "--retry-all-errors" in script
    assert "--timeout \"${PIP_TIMEOUT}\"" in script
    assert "sha256sum" in script
    assert "Path(sys.argv[1]).resolve().as_uri()" in script
    assert 'crowdtensor" train join "${WORKSPACE}"' in script
    assert "--dry-run --json" in script
    assert "volunteer join" not in script
    assert "crowdtensord[hf,storage]" not in script
    assert "--no-cache-dir" in script
