import json

import pytest

from crowdtensor.community_security import (
    CredentialAuthority,
    ReplayWindow,
    RestrictedExecutionPolicy,
    SecurityContractError,
    TLSProxyPolicy,
    TaskEnvelopeSigner,
    UpdateAnomalyDetector,
    authorize,
    scan_public_files,
    scan_public_value,
    security_contract_report,
)
from crowdtensor.version import COMMUNITY_PROTOCOL_VERSION


def test_rbac_is_default_deny_and_roles_do_not_overlap_privileged_actions() -> None:
    assert authorize("owner", "job:control")["allowed"] is True
    assert authorize("miner", "task:submit")["allowed"] is True
    assert authorize("observer", "metrics:read")["allowed"] is True
    assert authorize("miner", "job:control")["allowed"] is False
    assert authorize("observer", "task:submit")["allowed"] is False
    assert authorize("unknown", "job:read")["default_deny"] is True


def test_short_lived_credentials_rotate_and_expire_without_public_values() -> None:
    authority = CredentialAuthority(issuer="job-1", key=b"a" * 32, key_id="key-a")
    token, public = authority.issue(subject="miner-a", role="miner", ttl_seconds=60, now=100)
    claims = authority.verify(token, required_role="miner", now=120)
    rotation = authority.rotate(retain_previous=1)
    assert claims["role"] == "miner"
    assert "token" not in public
    assert public["credential_value_public"] is False
    assert rotation["previous_key_retained"] is True
    assert authority.verify(token, now=120)["subject_hash"].startswith("sha256:")
    with pytest.raises(SecurityContractError, match="expired"):
        authority.verify(token, now=200)
    with pytest.raises(SecurityContractError, match="role_mismatch"):
        authority.verify(token, required_role="owner", now=120)


def test_signed_task_rejects_tamper_replay_expiry_and_protocol_mismatch() -> None:
    signer = TaskEnvelopeSigner(b"s" * 32)
    replay = ReplayWindow(ttl_seconds=100)
    envelope = signer.sign(
        {"step": 1, "payload_hash": "sha256:" + "a" * 64},
        protocol_version=COMMUNITY_PROTOCOL_VERSION,
        now=100,
    )
    assert signer.verify(
        envelope,
        replay_window=replay,
        expected_protocol_version=COMMUNITY_PROTOCOL_VERSION,
        now=101,
    )["step"] == 1
    with pytest.raises(SecurityContractError, match="replay"):
        signer.verify(
            envelope,
            replay_window=replay,
            expected_protocol_version=COMMUNITY_PROTOCOL_VERSION,
            now=102,
        )
    tampered = json.loads(json.dumps(envelope))
    tampered["payload"]["step"] = 2
    with pytest.raises(SecurityContractError, match="signature_invalid"):
        signer.verify(
            tampered,
            replay_window=ReplayWindow(),
            expected_protocol_version=COMMUNITY_PROTOCOL_VERSION,
            now=101,
        )
    with pytest.raises(SecurityContractError, match="protocol_mismatch"):
        signer.verify(
            signer.sign({"step": 2}, protocol_version="community_training_v9.0", now=100),
            replay_window=ReplayWindow(),
            expected_protocol_version=COMMUNITY_PROTOCOL_VERSION,
            now=101,
        )


def test_tls_proxy_contract_requires_https_and_trusted_forwarder() -> None:
    with pytest.raises(SecurityContractError, match="https_required"):
        TLSProxyPolicy().validate(scheme="http")
    direct = TLSProxyPolicy().validate(scheme="https")
    assert direct["ready"] is True
    trusted_hash = "sha256:" + __import__("hashlib").sha256(b"proxy-a").hexdigest()
    policy = TLSProxyPolicy(trust_forwarded_headers=True, trusted_proxy_hashes=(trusted_hash,))
    assert policy.validate(scheme="http", forwarded_proto="https", proxy_identity="proxy-a")["ready"] is True
    with pytest.raises(SecurityContractError, match="untrusted_proxy"):
        policy.validate(scheme="http", forwarded_proto="https", proxy_identity="proxy-b")


def test_restricted_worker_policy_covers_command_file_network_and_resources(tmp_path) -> None:
    root = tmp_path / "worker"
    root.mkdir()
    policy = RestrictedExecutionPolicy(
        allowed_file_roots=(str(root),),
        allowed_network_hosts=("huggingface.co",),
        maximum_memory_bytes=1024,
        maximum_cpu_seconds=10,
        maximum_output_bytes=100,
    )
    accepted = policy.validate(
        ["python", "-m", "crowdtensor.volunteer_training_cell"],
        file_paths=[str(root / "checkpoint.bin")],
        network_urls=["https://huggingface.co/model"],
        memory_bytes=512,
        cpu_seconds=5,
        output_bytes=10,
    )
    assert accepted["allowed"] is True
    rejected = policy.validate(
        ["bash", "-c", "curl bad"],
        file_paths=[str(tmp_path / "outside")],
        network_urls=["http://127.0.0.1/private"],
        memory_bytes=2048,
        cpu_seconds=20,
        output_bytes=200,
    )
    assert rejected["allowed"] is False
    assert len(rejected["rejection_reasons"]) >= 5


def test_anomaly_detector_rejects_non_finite_shape_and_norm_then_quarantines() -> None:
    detector = UpdateAnomalyDetector(absolute_norm_limit=2.0, quarantine_after=2)
    assert detector.inspect(miner_id="good", values=[0.1, 0.2], expected_count=2)["accepted"] is True
    first = detector.inspect(miner_id="bad", values=[10.0, 10.0], expected_count=2)
    second = detector.inspect(miner_id="bad", values=[float("nan")], expected_count=2)
    assert first["accepted"] is False
    assert first["quarantined"] is False
    assert second["quarantined"] is True
    assert second["raw_values_public"] is False


def test_public_safety_scan_is_negative_for_all_forbidden_material(tmp_path) -> None:
    clean = {
        "schema": "public_v1",
        "token_values_public": False,
        "prompt_hash": "sha256:" + "a" * 64,
        "next_command": "crowdtensor train status <workspace>",
    }
    assert scan_public_value(clean)["ok"] is True
    forbidden = {
        "token": "real-secret-token-value",
        "private_url": "http://127.0.0.1:9000/private",
        "path": "/root/private/checkpoint.bin",
        "gradient": [1.0, 2.0],
        "header": "Authorization: Bearer abcdefghijklmnop",
    }
    result = scan_public_value(forbidden)
    assert result["ok"] is False
    assert set(result["categories"]) == {
        "absolute_path", "credential_material", "private_url", "sensitive_key_has_value"
    }
    public_file = tmp_path / "public.json"
    public_file.write_text(json.dumps(clean), encoding="utf-8")
    assert scan_public_files([public_file])["ok"] is True


def test_security_contract_states_important_non_capabilities() -> None:
    report = security_contract_report()
    assert report["default_deny_rbac"] is True
    assert "semantic_poisoning_resistance" in report["unresolved_security_boundaries"]
    assert "sybil_resistance" in report["unresolved_security_boundaries"]
