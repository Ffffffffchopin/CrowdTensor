from fastapi.testclient import TestClient

from crowdtensor.community_api import CommunitySecurityContext, create_community_app
from crowdtensor.community_security import RestrictedExecutionPolicy, TLSProxyPolicy
from crowdtensor.community_workflow import CommunityWorkflow
from crowdtensor.version import COMMUNITY_PROTOCOL_VERSION


def setup_client(tmp_path):
    workspace = tmp_path / "community"
    CommunityWorkflow.initialize(workspace)
    workflow = CommunityWorkflow(workspace)
    policy = RestrictedExecutionPolicy(
        allowed_file_roots=(str(workspace),),
        allowed_network_hosts=("huggingface.co",),
    )
    context = CommunitySecurityContext(
        issuer="test",
        signing_key=b"z" * 32,
        tls_policy=TLSProxyPolicy(require_https=False),
        execution_policy=policy,
    )
    return TestClient(create_community_app(workflow, context=context)), context, workspace


def bearer(value: str) -> dict[str, str]:
    return {"Authorization": "Bearer " + value}


def test_role_separation_and_redacted_audit(tmp_path) -> None:
    client, context, _workspace = setup_client(tmp_path)
    assert client.get("/v1/community/health").status_code == 200
    assert client.get("/v1/community/status", headers=bearer(context.credentials.observer)).status_code == 200
    assert client.get("/v1/community/status", headers=bearer(context.credentials.miner)).status_code == 403
    assert client.post("/v1/community/control/pause", headers=bearer(context.credentials.owner)).status_code == 200
    assert client.post("/v1/community/control/pause", headers=bearer(context.credentials.observer)).status_code == 403
    audit = client.get("/v1/community/audit", headers=bearer(context.credentials.observer)).json()
    assert len(audit["events"]) >= 4
    assert "Bearer" not in str(audit)
    assert audit["credential_values_public"] is False


def test_task_signature_replay_execution_and_update_quarantine_routes(tmp_path) -> None:
    client, context, workspace = setup_client(tmp_path)
    envelope = context.task_signer.sign(
        {"step": 1, "assignment": "stage0"},
        protocol_version=COMMUNITY_PROTOCOL_VERSION,
    )
    first = client.post(
        "/v1/community/tasks/verify",
        headers=bearer(context.credentials.miner),
        json=envelope,
    )
    replay = client.post(
        "/v1/community/tasks/verify",
        headers=bearer(context.credentials.miner),
        json=envelope,
    )
    assert first.status_code == 200
    assert replay.status_code == 409

    allowed = client.post(
        "/v1/community/execution/validate",
        headers=bearer(context.credentials.miner),
        json={
            "command": ["python", "-m", "crowdtensor.community_worker"],
            "file_paths": [str(workspace / "checkpoint")],
            "network_urls": ["https://huggingface.co/model"],
            "memory_bytes": 1024,
        },
    ).json()
    denied = client.post(
        "/v1/community/execution/validate",
        headers=bearer(context.credentials.miner),
        json={"command": ["bash", "-c", "curl evil"], "network_urls": ["http://127.0.0.1"]},
    ).json()
    assert allowed["allowed"] is True
    assert denied["allowed"] is False

    for _ in range(2):
        anomaly = client.post(
            "/v1/community/updates/inspect",
            headers=bearer(context.credentials.miner),
            json={"miner_id": "bad", "values": [1e9], "expected_count": 1},
        ).json()
    assert anomaly["quarantined"] is True
    assert "values" not in anomaly


def test_rotation_issues_new_private_credentials_and_preserves_public_contract(tmp_path) -> None:
    client, context, _workspace = setup_client(tmp_path)
    old = context.credentials.owner
    credentials, report = context.rotate()
    assert credentials.owner != old
    assert report["credential_values_public"] is False
    assert client.get("/v1/community/status", headers=bearer(credentials.observer)).status_code == 200
