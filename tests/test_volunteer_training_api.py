from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from crowdtensor.community_security import TLSProxyPolicy
from crowdtensor.version import __version__
from crowdtensor.volunteer_training_api import (
    PUBLIC_RELEASE_ARTIFACT_NAMES,
    create_volunteer_training_app,
    service_contract,
)
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

    def checkpoint_lineage(self):
        return {"ok": True, "schema": "fixture-lineage"}

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


def _write_release(root: Path) -> Path:
    root.mkdir()
    manifest_names = PUBLIC_RELEASE_ARTIFACT_NAMES - {"SHA256SUMS", "release.json"}
    artifacts = []
    for name in sorted(manifest_names):
        path = root / name
        path.write_bytes((name + "\n").encode("utf-8"))
        artifacts.append(
            {
                "name": name,
                "byte_count": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (root / "release.json").write_text(
        json.dumps(
            {
                "schema": "crowdtensor_release_v1",
                "version": __version__,
                "commit": "test",
                "artifacts": artifacts,
                "public_artifact_safe": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checksum_names = PUBLIC_RELEASE_ARTIFACT_NAMES - {"SHA256SUMS"}
    (root / "SHA256SUMS").write_text(
        "".join(
            hashlib.sha256((root / name).read_bytes()).hexdigest()
            + "  "
            + name
            + "\n"
            for name in sorted(checksum_names)
        ),
        encoding="utf-8",
    )
    return root


def test_service_contract_lists_v2_public_projection_routes() -> None:
    contract = service_contract()
    assert "GET /v1/volunteer/checkpoint-lineage" in contract["routes"]
    assert "GET /v1/volunteer/session" in contract["routes"]
    assert contract["optional_v2_session_controller_bridge"] is True


def test_invalid_v2_controller_bridge_is_rejected_at_startup(tmp_path) -> None:
    with pytest.raises(ValueError, match="volunteer_v2_controller_bridge_invalid"):
        create_volunteer_training_app(
            StubCoordinator(tmp_path), controller_transport=object()
        )


def test_complete_versioned_release_directory_is_served(tmp_path) -> None:
    release = _write_release(tmp_path / "release")
    client = TestClient(
        create_volunteer_training_app(
            StubCoordinator(tmp_path / "coordinator"), public_release_dir=release
        )
    )
    health = client.get("/v1/volunteer/health").json()
    assert health["public_release_download"] is True
    assert health["package_version"] == __version__
    wheel_name = f"crowdtensord-{__version__}-py3-none-any.whl"
    assert client.get("/downloads/" + wheel_name).content == (
        wheel_name + "\n"
    ).encode("utf-8")
    assert client.get("/downloads/private-invite.json").status_code == 404


def test_incomplete_release_directory_fails_closed(tmp_path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    (release / "install-contributor.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    with pytest.raises(ValueError, match="volunteer_release_directory_incomplete"):
        create_volunteer_training_app(
            StubCoordinator(tmp_path / "coordinator"), public_release_dir=release
        )


def test_http_routes_authenticate_binary_submission_and_remove_upload(tmp_path) -> None:
    coordinator = StubCoordinator(tmp_path)
    client = TestClient(create_volunteer_training_app(coordinator))
    assert client.get("/v1/volunteer/health").status_code == 200
    assert client.get("/v1/volunteer/status").json()["adapter_version"] == 0
    assert client.get("/v1/volunteer/checkpoint-lineage").json()["ok"] is True
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
    join = client.get("/join")
    assert home.status_code == 200
    assert client.head("/").status_code == 200
    assert "Train a model together" in home.text
    assert join.status_code == 200
    assert "Contribute from this device" in join.text
    assert 'id="agent-tab" class="mode-tab active"' in join.text
    assert 'id="agent-panel" role="tabpanel" aria-labelledby="agent-tab">' in join.text
    assert 'id="browser-panel" role="tabpanel" aria-labelledby="browser-tab" hidden' in join.text
    assert client.head("/join").status_code == 200
    assert "/v1/volunteer/public-snapshot" not in home.text
    assert "issues/new?template=beta_enrollment.yml" not in home.text
    assert "docs/campaigns/qwen25-7b-gsm8k-rfc.md" not in home.text
    assert "crowdtensor.24.199.118.54.nip.io" not in home.text + join.text
    assert "/CrowdTensor/discussions" not in home.text
    assert "default-src 'self'" in home.headers["content-security-policy"]
    site_css = client.get("/assets/site.css")
    site_script = client.get("/assets/site.js")
    join_script = client.get("/assets/join.js")
    join_worker = client.get("/assets/join_worker.js")
    hero = client.get("/assets/hero-dashboard.png")
    favicon = client.get("/favicon.ico")
    assert site_css.status_code == 200
    assert "campaign-layout" in site_css.text
    assert "gradient" not in site_css.text
    assert site_script.status_code == 200
    assert "/v1/volunteer/public-snapshot" in site_script.text
    assert join_script.status_code == 200
    assert "/v1/volunteer/pairing/redeem" in join_script.text
    assert "/v1/volunteer/browser/submit" in join_script.text
    assert "window.location.origin" in join_script.text
    assert "/downloads/install-contributor.sh" in join_script.text
    assert join_worker.status_code == 200
    assert "requestAdapter" in join_worker.text
    assert "WebAssembly.instantiate" in join_worker.text
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
