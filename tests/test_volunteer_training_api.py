from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

from crowdtensor.community_security import TLSProxyPolicy
from crowdtensor.volunteer_training_api import create_volunteer_training_app
from crowdtensor.volunteer_training_protocol import (
    SUBMISSION_SCHEMA,
    VolunteerProtocolError,
    encode_submission_envelope,
)


class StubCoordinator:
    def __init__(self, root: Path) -> None:
        self.private = root / ".private"
        self.private.mkdir(parents=True)
        self.artifact = root / "artifact.bin"
        self.artifact.write_bytes(b"artifact")
        self.submitted_path_existed = False

    def campaign_manifest(self):
        return {
            "campaign_id": "api-test",
            "update_admission": {"max_delta_bytes": 2048},
        }

    def status(self):
        return {"ok": True, "adapter_version": 0}

    def authenticate_invite(self, invite_token):
        if invite_token != "invite":
            raise VolunteerProtocolError(
                "volunteer_invite_authentication_failed", status_code=401
            )
        return {"ok": True, "invite_authenticated": True}

    def claim(self, *, cell_id, invite_token, capability):
        assert invite_token == "invite"
        return {"ok": True, "state": "leased", "cell_id": cell_id}

    def heartbeat(self, **kwargs):
        assert kwargs["invite_token"] == "invite"
        return {"ok": True}

    def artifact_path(self, _artifact_id, *, invite_token):
        assert invite_token == "invite"
        return self.artifact

    def submit(self, *, delta_manifest, invite_token, **_kwargs):
        assert invite_token == "invite"
        self.submitted_path_existed = Path(delta_manifest["delta_path"]).is_file()
        return {"ok": True, "accepted": True}


def test_http_routes_authenticate_binary_submission_and_remove_upload(tmp_path) -> None:
    coordinator = StubCoordinator(tmp_path)
    client = TestClient(create_volunteer_training_app(coordinator))
    assert client.get("/v1/volunteer/health").status_code == 200
    assert client.get("/v1/volunteer/status").json()["adapter_version"] == 0
    unauthorized = client.post(
        "/v1/volunteer/work/claim", json={"cell_id": "cell"}
    )
    assert unauthorized.status_code == 401
    headers = {"Authorization": "Bearer invite"}
    claim = client.post(
        "/v1/volunteer/work/claim",
        headers=headers,
        json={"cell_id": "cell", "capability": {"device": "cpu"}},
    )
    assert claim.json()["state"] == "leased"
    assert client.get("/v1/volunteer/artifacts/id", headers=headers).content == b"artifact"

    body = encode_submission_envelope(
        {
            "schema": SUBMISSION_SCHEMA,
            "cell_id": "cell",
            "work_id": "work",
            "lease_generation": 1,
            "lease_token": "lease",
            "delta_manifest": {"result_id": "result"},
        },
        b"safetensors",
    )
    response = client.post(
        "/v1/volunteer/work/submit",
        headers={**headers, "Content-Type": "application/octet-stream"},
        content=body,
    )
    assert response.status_code == 200
    assert response.json()["accepted"] is True
    assert coordinator.submitted_path_existed is True
    assert list((coordinator.private / "uploads").glob("*")) == []


def test_public_dashboard_assets_and_snapshot_route(tmp_path) -> None:
    client = TestClient(create_volunteer_training_app(StubCoordinator(tmp_path)))
    home = client.get("/")
    assert home.status_code == 200
    assert client.head("/").status_code == 200
    assert "Train a model together" in home.text
    assert "/v1/volunteer/public-snapshot" not in home.text
    assert "default-src 'self'" in home.headers["content-security-policy"]
    site_css = client.get("/assets/site.css")
    site_script = client.get("/assets/site.js")
    hero = client.get("/assets/hero-dashboard.png")
    favicon = client.get("/favicon.ico")
    assert site_css.status_code == 200
    assert "campaign-layout" in site_css.text
    assert "gradient" not in site_css.text
    assert site_script.status_code == 200
    assert "/v1/volunteer/public-snapshot" in site_script.text
    assert hero.status_code == 200
    assert hero.headers["content-type"] == "image/png"
    assert hero.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert favicon.status_code == 200
    assert favicon.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert client.get("/assets/private.json").status_code == 404
    dashboard = client.get("/v1/volunteer/dashboard")
    assert dashboard.status_code == 200
    assert "CrowdTensor" in dashboard.text
    assert "default-src 'self'" in dashboard.headers["content-security-policy"]
    stylesheet = client.get(
        "/v1/volunteer/dashboard/assets/dashboard.css"
    )
    script = client.get("/v1/volunteer/dashboard/assets/dashboard.js")
    assert stylesheet.status_code == 200
    assert "metric-grid" in stylesheet.text
    assert script.status_code == 200
    assert "/v1/volunteer/public-snapshot" in script.text
    assert client.get(
        "/v1/volunteer/dashboard/assets/private.json"
    ).status_code == 404
    snapshot = client.get("/v1/volunteer/public-snapshot").json()
    assert snapshot["ok"] is True
    assert snapshot["schema"] == "crowdtensor_volunteer_public_campaign_snapshot_v1"


def test_resumable_routes_survive_app_recreation(tmp_path) -> None:
    coordinator = StubCoordinator(tmp_path)
    headers = {"Authorization": "Bearer invite", "X-CrowdTensor-Cell-Id": "cell"}
    data = b"z" * 1500
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    submission = {
        "schema": SUBMISSION_SCHEMA,
        "cell_id": "cell",
        "work_id": "work",
        "lease_generation": 1,
        "lease_token": "lease",
        "delta_manifest": {"result_id": "result", "delta_file_hash": digest},
    }
    client = TestClient(
        create_volunteer_training_app(coordinator, upload_chunk_bytes=1024)
    )
    started_response = client.post(
        "/v1/volunteer/uploads/start",
        headers=headers,
        json={
            "cell_id": "cell",
            "idempotency_key": "result",
            "expected_blob_hash": digest,
            "total_bytes": len(data),
            "submission": submission,
        },
    )
    assert started_response.status_code == 200, started_response.json()
    started = started_response.json()
    upload_id = started["upload_id"]
    first = data[:1024]
    chunk_headers = {
        **headers,
        "X-CrowdTensor-Chunk-SHA256": "sha256:"
        + hashlib.sha256(first).hexdigest(),
    }
    assert client.put(
        f"/v1/volunteer/uploads/{upload_id}/chunks/0",
        headers=chunk_headers,
        content=first,
    ).status_code == 200

    recovered = TestClient(
        create_volunteer_training_app(coordinator, upload_chunk_bytes=1024)
    )
    status = recovered.get(
        f"/v1/volunteer/uploads/{upload_id}", headers=headers
    ).json()
    assert status["received_chunk_indexes"] == [0]
    second = data[1024:]
    assert recovered.put(
        f"/v1/volunteer/uploads/{upload_id}/chunks/1",
        headers={
            **headers,
            "X-CrowdTensor-Chunk-SHA256": "sha256:"
            + hashlib.sha256(second).hexdigest(),
        },
        content=second,
    ).status_code == 200
    complete = recovered.post(
        f"/v1/volunteer/uploads/{upload_id}/complete", headers=headers
    )
    assert complete.status_code == 200
    assert complete.json()["accepted"] is True
    assert complete.json()["resumable_upload"]["complete"] is True
    assert coordinator.submitted_path_existed is True


def test_resumable_start_accepts_bounded_real_delta_metadata(tmp_path) -> None:
    coordinator = StubCoordinator(tmp_path)
    data = b"z"
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    response = TestClient(create_volunteer_training_app(coordinator)).post(
        "/v1/volunteer/uploads/start",
        headers={"Authorization": "Bearer invite"},
        json={
            "cell_id": "cell",
            "idempotency_key": "large-contract",
            "expected_blob_hash": digest,
            "total_bytes": 1,
            "submission": {
                "schema": SUBMISSION_SCHEMA,
                "cell_id": "cell",
                "work_id": "work",
                "lease_generation": 1,
                "lease_token": "lease",
                "delta_manifest": {
                    "result_id": "large-contract",
                    "delta_file_hash": digest,
                    "tensor_contract_padding": "x" * (96 * 1024),
                },
            },
        },
    )
    assert response.status_code == 200, response.json()
    assert response.json()["chunk_count"] == 1


def test_resumable_routes_validate_invite_before_allocating(tmp_path) -> None:
    coordinator = StubCoordinator(tmp_path)
    data = b"z"
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    response = TestClient(create_volunteer_training_app(coordinator)).post(
        "/v1/volunteer/uploads/start",
        headers={"Authorization": "Bearer wrong"},
        json={
            "cell_id": "cell",
            "idempotency_key": "unauthorized",
            "expected_blob_hash": digest,
            "total_bytes": 1,
            "submission": {
                "schema": SUBMISSION_SCHEMA,
                "cell_id": "cell",
                "work_id": "work",
                "lease_generation": 1,
                "lease_token": "lease",
                "delta_manifest": {"result_id": "unauthorized"},
            },
        },
    )
    assert response.status_code == 401
    assert list(
        (coordinator.private / "resumable-uploads" / "sessions").glob(
            "*/session.json"
        )
    ) == []


def test_https_proxy_contract_rejects_direct_and_untrusted_http(tmp_path) -> None:
    proxy_id = "trusted-proxy"
    proxy_hash = "sha256:" + hashlib.sha256(proxy_id.encode()).hexdigest()
    app = create_volunteer_training_app(
        StubCoordinator(tmp_path),
        tls_policy=TLSProxyPolicy(
            require_https=True,
            trust_forwarded_headers=True,
            trusted_proxy_hashes=(proxy_hash,),
        ),
    )
    client = TestClient(app)
    assert client.get("/v1/volunteer/health").status_code == 400
    assert client.get(
        "/v1/volunteer/health",
        headers={"X-Forwarded-Proto": "https", "X-CrowdTensor-Proxy-Id": "wrong"},
    ).status_code == 400
    accepted = client.get(
        "/v1/volunteer/health",
        headers={
            "X-Forwarded-Proto": "https",
            "X-CrowdTensor-Proxy-Id": proxy_id,
        },
    )
    assert accepted.status_code == 200
    assert accepted.json()["tls_required"] is True
