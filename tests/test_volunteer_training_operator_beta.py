from __future__ import annotations

import json
import tarfile

import pytest
from fastapi.testclient import TestClient

from crowdtensor.hf_lora_training import create_local_training_fixture
from crowdtensor.training_contract import sha256_json
from crowdtensor.volunteer_training_api import create_volunteer_training_app
from crowdtensor.volunteer_training_coordinator import VolunteerTrainingCoordinator
from crowdtensor.volunteer_training_operator import STATE_SCHEMA_V2
from crowdtensor.volunteer_training_protocol import VolunteerProtocolError


def _coordinator(tmp_path) -> VolunteerTrainingCoordinator:
    fixture = create_local_training_fixture(
        tmp_path / "fixture", row_count=8, local_steps=1
    )
    return VolunteerTrainingCoordinator.create_from_fixture(
        tmp_path / "campaign",
        fixture,
        target_rounds=1,
        lease_seconds=30,
    )


def test_per_cell_scope_replay_revocation_and_rate_policy(tmp_path) -> None:
    coordinator = _coordinator(tmp_path)
    invite = coordinator.private_invite()["invite_token"]
    issued = coordinator.issue_cell_credential(
        invite_token=invite,
        cell_id="cell-a",
        scopes=["work:claim"],
        ttl_seconds=120,
    )
    token = issued["credential_token"]
    claim = coordinator.claim(
        cell_id="cell-a",
        invite_token=token,
        capability={"device": "cpu"},
        request_nonce="nonce-claim-0001",
    )
    assert claim["state"] == "leased"
    snapshot = coordinator.public_campaign_snapshot()
    assert snapshot["content_hash"] == sha256_json(
        {key: value for key, value in snapshot.items() if key != "content_hash"}
    )
    assert snapshot["progress"]["active_contributor_count"] == 1
    serialized_snapshot = json.dumps(snapshot, sort_keys=True)
    for private_key in (
        '"cell_id_hash"',
        '"work_id"',
        '"active_leases"',
        '"credential_id"',
        '"lease_expires_at"',
    ):
        assert private_key not in serialized_snapshot
    assert snapshot["privacy"]["cell_identifiers_public"] is False
    with pytest.raises(VolunteerProtocolError, match="replay_detected"):
        coordinator.claim(
            cell_id="cell-a",
            invite_token=token,
            request_nonce="nonce-claim-0001",
        )
    with pytest.raises(VolunteerProtocolError, match="scope_missing"):
        coordinator.heartbeat(
            cell_id="cell-a",
            invite_token=token,
            work_id=claim["work_unit"]["work_id"],
            lease_generation=claim["work_unit"]["lease_generation"],
            lease_token=claim["work_unit"]["lease_token"],
            request_nonce="nonce-heartbeat-01",
        )

    coordinator.revoke_cell_credential(
        invite_token=invite, credential_id=issued["credential_id"]
    )
    with pytest.raises(VolunteerProtocolError, match="credential_revoked"):
        coordinator.claim(
            cell_id="cell-a",
            invite_token=token,
            request_nonce="nonce-claim-after-revoke",
        )

    coordinator.configure_operator_policy(
        invite_token=invite,
        updates={"maximum_requests_per_window": 2},
    )
    limited = coordinator.issue_cell_credential(
        invite_token=invite,
        cell_id="cell-rate",
        scopes=["upload:read"],
        ttl_seconds=120,
    )
    for index in range(2):
        coordinator.authorize_cell_request(
            token=limited["credential_token"],
            cell_id="cell-rate",
            scope="upload:read",
            request_nonce=f"nonce-rate-{index:04d}",
        )
    with pytest.raises(VolunteerProtocolError, match="request_rate_limited"):
        coordinator.authorize_cell_request(
            token=limited["credential_token"],
            cell_id="cell-rate",
            scope="upload:read",
            request_nonce="nonce-rate-over-limit",
        )
    policy = coordinator.status()["operator_policy"]
    assert policy["short_lived_per_cell_credentials"] is True
    assert policy["revoked_credential_count"] == 1
    assert policy["counters"]["rate_limit_rejections"] == 1
    assert policy["counters"]["replay_rejections"] == 1
    assert policy["counters"]["scope_rejections"] == 1
    private_text = coordinator.state_path.read_text(encoding="utf-8")
    assert token not in private_text
    assert limited["credential_token"] not in private_text


def test_real_http_credential_enrollment_nonce_and_revocation(tmp_path) -> None:
    coordinator = _coordinator(tmp_path)
    invite = coordinator.private_invite()["invite_token"]
    client = TestClient(create_volunteer_training_app(coordinator))
    issued_response = client.post(
        "/v1/volunteer/credentials/issue",
        headers={"Authorization": "Bearer " + invite},
        json={"cell_id": "http-cell", "ttl_seconds": 120},
    )
    assert issued_response.status_code == 200
    issued = issued_response.json()
    headers = {
        "Authorization": "Bearer " + issued["credential_token"],
        "X-CrowdTensor-Cell-Id": "http-cell",
        "X-CrowdTensor-Nonce": "http-request-nonce-0001",
    }
    claim = client.post(
        "/v1/volunteer/work/claim",
        headers=headers,
        json={"cell_id": "http-cell", "capability": {"device": "cpu"}},
    )
    assert claim.status_code == 200
    replay = client.post(
        "/v1/volunteer/work/claim",
        headers=headers,
        json={"cell_id": "http-cell"},
    )
    assert replay.status_code == 409
    assert replay.json()["error"] == "volunteer_request_replay_detected"
    revoke = client.post(
        "/v1/volunteer/credentials/revoke",
        headers={"Authorization": "Bearer " + invite},
        json={"credential_id": issued["credential_id"]},
    )
    assert revoke.status_code == 200
    rejected = client.post(
        "/v1/volunteer/work/claim",
        headers={**headers, "X-CrowdTensor-Nonce": "http-request-nonce-0002"},
        json={"cell_id": "http-cell"},
    )
    assert rejected.status_code == 403
    assert "crowdtensor_volunteer_active_credentials" in client.get(
        "/v1/volunteer/metrics"
    ).text


def test_lifecycle_backup_restore_export_and_v1_migration(tmp_path) -> None:
    coordinator = _coordinator(tmp_path)
    invite = coordinator.private_invite()["invite_token"]
    assert coordinator.validate_campaign()["ok"] is True
    assert coordinator.pause_campaign(invite_token=invite)["lifecycle"] == "paused"
    paused_claim = coordinator.claim(cell_id="paused", invite_token=invite)
    assert paused_claim["state"] == "campaign_paused"
    assert coordinator.resume_campaign(invite_token=invite)["lifecycle"] == "running"

    state = json.loads(coordinator.state_path.read_text(encoding="utf-8"))
    state["schema"] = "crowdtensor_volunteer_training_coordinator_state_v1"
    state.pop("state_revision", None)
    coordinator.state_path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    migrated = VolunteerTrainingCoordinator(coordinator.root)
    migrated_state = json.loads(migrated.state_path.read_text(encoding="utf-8"))
    assert migrated_state["schema"] == STATE_SCHEMA_V2
    assert migrated_state["state_revision"] == 2
    assert migrated.validate_campaign()["ok"] is True

    backup_path = tmp_path / "campaign-private-backup.tar.gz"
    backup = migrated.backup_campaign(backup_path)
    assert backup["backup_permissions_restricted"] is True
    restored, restore = VolunteerTrainingCoordinator.restore_campaign(
        backup_path, tmp_path / "restored"
    )
    assert restore["coordinator_recovery_verified"] is True
    assert restored.verify_ledger()["ok"] is True
    assert restored.validate_campaign()["ok"] is True
    export_path = tmp_path / "campaign-export.zip"
    exported = restored.export_campaign(export_path)
    assert exported["credential_values_included"] is False
    assert export_path.is_file()


def test_restore_rejects_archive_links(tmp_path) -> None:
    backup = tmp_path / "unsafe.tar.gz"
    with tarfile.open(backup, "w:gz") as archive:
        member = tarfile.TarInfo("linked-state.json")
        member.type = tarfile.LNKTYPE
        member.linkname = "outside-state.json"
        archive.addfile(member)
    with pytest.raises(VolunteerProtocolError, match="backup_member_unsafe"):
        VolunteerTrainingCoordinator.restore_campaign(
            backup, tmp_path / "unsafe-restore"
        )
