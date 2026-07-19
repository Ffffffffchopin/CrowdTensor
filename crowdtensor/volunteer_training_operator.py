"""Persistent Operator Beta policy for volunteer campaign Cells.

The campaign invite remains an enrollment/operator secret. Runtime Cells use
independent, short-lived credentials whose values are never written to public
artifacts or Coordinator state.
"""

from __future__ import annotations

import base64
import secrets
from typing import Any

from .community_security import CredentialAuthority, SecurityContractError
from .training_contract import sha256_json
from .volunteer_training_protocol import (
    VolunteerProtocolError,
    hash_cell_id,
    token_hash,
    with_public_safety,
)


OPERATOR_POLICY_SCHEMA = "crowdtensor_volunteer_operator_policy_v1"
CELL_CREDENTIAL_SCHEMA = "crowdtensor_volunteer_cell_credential_v1"
POLICY_STATUS_SCHEMA = "crowdtensor_volunteer_operator_policy_status_v1"
STATE_SCHEMA_V2 = "crowdtensor_volunteer_training_coordinator_state_v2"

CELL_SCOPES = frozenset(
    {
        "artifact:read",
        "upload:read",
        "upload:write",
        "work:claim",
        "work:heartbeat",
        "work:submit",
    }
)


def default_operator_policy() -> dict[str, Any]:
    return {
        "schema": OPERATOR_POLICY_SCHEMA,
        "credential_default_ttl_seconds": 900,
        "credential_maximum_ttl_seconds": 3600,
        "maximum_active_credentials": 10_000,
        "maximum_active_credentials_per_cell": 4,
        "maximum_active_leases_per_cell": 1,
        "request_window_seconds": 60,
        "maximum_requests_per_window": 240,
        "maximum_upload_bytes_per_window": 512 * 1024 * 1024,
        "maximum_upload_bytes_per_credential": 2 * 1024 * 1024 * 1024,
        "maximum_submissions_per_credential": 128,
        "replay_ttl_seconds": 900,
        "maximum_replay_entries": 100_000,
    }


def _new_authority_state(campaign_id: str) -> dict[str, Any]:
    key = secrets.token_bytes(32)
    return {
        "issuer": "volunteer-campaign:" + str(campaign_id),
        "key_b64": base64.urlsafe_b64encode(key).decode("ascii"),
        "key_id": "cell-" + secrets.token_hex(8),
    }


def migrate_operator_state(state: dict[str, Any], *, now: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Idempotently migrate a v1 campaign state into the Operator Beta v2 state."""

    previous_schema = str(state.get("schema") or "")
    changed = previous_schema != STATE_SCHEMA_V2
    campaign_id = str((state.get("campaign") or {}).get("campaign_id") or "unknown")
    state["schema"] = STATE_SCHEMA_V2
    state.setdefault("state_revision", 2)
    state.setdefault("operator_policy", default_operator_policy())
    state.setdefault("credential_authority", _new_authority_state(campaign_id))
    state.setdefault("cell_credentials", {})
    state.setdefault("credential_request_windows", {})
    state.setdefault("replay_nonces", {})
    state.setdefault(
        "policy_counters",
        {
            "credentials_issued": 0,
            "credentials_revoked": 0,
            "credential_authentications": 0,
            "legacy_invite_authentications": 0,
            "rate_limit_rejections": 0,
            "quota_rejections": 0,
            "replay_rejections": 0,
            "scope_rejections": 0,
        },
    )
    state.setdefault("campaign_lifecycle", "running")
    state.setdefault("validated_at", float(now))
    state.setdefault("finalized_at", 0.0)
    state.setdefault("migration_history", [])
    if changed:
        state["migration_history"].append(
            {
                "from_schema": previous_schema or "unknown",
                "to_schema": STATE_SCHEMA_V2,
                "migrated_at": float(now),
                "private_state_migration": True,
            }
        )
    return state, {
        "schema": "crowdtensor_volunteer_operator_state_migration_v1",
        "ok": True,
        "migrated": changed,
        "from_schema": previous_schema,
        "to_schema": STATE_SCHEMA_V2,
        "state_revision": int(state["state_revision"]),
        "credential_values_public": False,
        "public_artifact_safe": True,
    }


def _authority(state: dict[str, Any]) -> CredentialAuthority:
    private = state["credential_authority"]
    try:
        key = base64.urlsafe_b64decode(str(private["key_b64"]).encode("ascii"))
    except (KeyError, ValueError) as exc:
        raise RuntimeError("volunteer_credential_authority_state_invalid") from exc
    policy = state["operator_policy"]
    return CredentialAuthority(
        issuer=str(private["issuer"]),
        key=key,
        key_id=str(private["key_id"]),
        maximum_ttl_seconds=int(policy["credential_maximum_ttl_seconds"]),
        clock_skew_seconds=0,
    )


def _raise(code: str, *, status_code: int = 403) -> None:
    raise VolunteerProtocolError(code, status_code=status_code)


def _active_credentials(state: dict[str, Any], *, now: float) -> list[dict[str, Any]]:
    return [
        record
        for record in state.get("cell_credentials", {}).values()
        if isinstance(record, dict)
        and not record.get("revoked")
        and float(record.get("expires_at") or 0.0) > float(now)
    ]


def issue_cell_credential(
    state: dict[str, Any],
    *,
    cell_id: str,
    scopes: list[str] | tuple[str, ...] | None,
    ttl_seconds: int | None,
    now: float,
) -> tuple[str, dict[str, Any]]:
    cell_hash = hash_cell_id(cell_id)
    policy = state["operator_policy"]
    requested = sorted(set(scopes or CELL_SCOPES))
    if not requested or not set(requested).issubset(CELL_SCOPES):
        _raise("volunteer_credential_scope_invalid", status_code=400)
    ttl = int(
        policy["credential_default_ttl_seconds"]
        if ttl_seconds is None
        else ttl_seconds
    )
    active = _active_credentials(state, now=now)
    if len(active) >= int(policy["maximum_active_credentials"]):
        state["policy_counters"]["quota_rejections"] += 1
        _raise("volunteer_credential_capacity_exceeded", status_code=429)
    if sum(record.get("cell_id_hash") == cell_hash for record in active) >= int(
        policy["maximum_active_credentials_per_cell"]
    ):
        state["policy_counters"]["quota_rejections"] += 1
        _raise("volunteer_cell_credential_capacity_exceeded", status_code=429)
    try:
        token, issued = _authority(state).issue(
            subject=cell_id,
            role="miner",
            scopes=requested,
            ttl_seconds=ttl,
            now=now,
        )
    except SecurityContractError as exc:
        _raise(str(exc), status_code=400)
    digest = token_hash(token)
    credential_id = sha256_json({"credential_token_hash": digest})
    claims = _authority(state).verify(token, now=now)
    state["cell_credentials"][digest] = {
        "schema": CELL_CREDENTIAL_SCHEMA,
        "credential_id": credential_id,
        "cell_id_hash": cell_hash,
        "token_hash": digest,
        "jti_hash": sha256_json({"jti": claims["jti"]}),
        "scopes": requested,
        "issued_at": float(issued["issued_at"]),
        "expires_at": float(issued["expires_at"]),
        "revoked": False,
        "revoked_at": 0.0,
        "request_count": 0,
        "upload_bytes": 0,
        "submission_count": 0,
    }
    state["policy_counters"]["credentials_issued"] += 1
    public = with_public_safety(
        {
            "schema": CELL_CREDENTIAL_SCHEMA,
            "credential_id": credential_id,
            "cell_id_hash": cell_hash,
            "scopes": requested,
            "issued_at": float(issued["issued_at"]),
            "expires_at": float(issued["expires_at"]),
            "ttl_seconds": ttl,
            "short_lived": True,
            "revocable": True,
            "credential_value_returned_privately": True,
        }
    )
    return token, public


def revoke_cell_credential(
    state: dict[str, Any], *, credential_id: str, now: float
) -> dict[str, Any]:
    matches = [
        record
        for record in state.get("cell_credentials", {}).values()
        if isinstance(record, dict) and record.get("credential_id") == credential_id
    ]
    if len(matches) != 1:
        _raise("volunteer_credential_not_found", status_code=404)
    record = matches[0]
    changed = not bool(record.get("revoked"))
    record["revoked"] = True
    record["revoked_at"] = float(now)
    if changed:
        state["policy_counters"]["credentials_revoked"] += 1
    return with_public_safety(
        {
            "schema": "crowdtensor_volunteer_cell_credential_revocation_v1",
            "ok": True,
            "credential_id": credential_id,
            "cell_id_hash": record["cell_id_hash"],
            "revoked": True,
            "idempotent": not changed,
        }
    )


def authorize_cell_credential(
    state: dict[str, Any],
    *,
    token: str,
    cell_id: str,
    required_scope: str,
    nonce: str,
    now: float,
    upload_bytes: int = 0,
    submission: bool = False,
) -> dict[str, Any]:
    digest = token_hash(token)
    record = state.get("cell_credentials", {}).get(digest)
    if not isinstance(record, dict):
        _raise("volunteer_cell_credential_invalid", status_code=401)
    if record.get("revoked"):
        _raise("volunteer_cell_credential_revoked", status_code=403)
    if float(record.get("expires_at") or 0.0) <= float(now):
        _raise("volunteer_cell_credential_expired", status_code=401)
    cell_hash = hash_cell_id(cell_id)
    if record.get("cell_id_hash") != cell_hash:
        _raise("volunteer_cell_credential_identity_mismatch", status_code=403)
    if required_scope not in set(record.get("scopes") or []):
        state["policy_counters"]["scope_rejections"] += 1
        _raise("volunteer_cell_credential_scope_missing", status_code=403)
    try:
        claims = _authority(state).verify(
            token,
            required_role="miner",
            required_scope=required_scope,
            now=now,
        )
    except SecurityContractError as exc:
        _raise(str(exc), status_code=401)
    if sha256_json({"jti": claims["jti"]}) != record.get("jti_hash"):
        _raise("volunteer_cell_credential_claims_mismatch", status_code=401)

    nonce_text = str(nonce or "")
    if len(nonce_text) < 12 or len(nonce_text) > 256:
        _raise("volunteer_request_nonce_required", status_code=400)
    replay = state["replay_nonces"]
    expired = [key for key, expiry in replay.items() if float(expiry) <= float(now)]
    for key in expired:
        replay.pop(key, None)
    replay_key = sha256_json({"credential_id": record["credential_id"], "nonce": nonce_text})
    if replay_key in replay:
        state["policy_counters"]["replay_rejections"] += 1
        _raise("volunteer_request_replay_detected", status_code=409)
    policy = state["operator_policy"]
    if len(replay) >= int(policy["maximum_replay_entries"]):
        state["policy_counters"]["quota_rejections"] += 1
        _raise("volunteer_replay_window_capacity_exceeded", status_code=429)

    window = state["credential_request_windows"].setdefault(
        record["credential_id"],
        {"started_at": float(now), "request_count": 0, "upload_bytes": 0},
    )
    if float(now) - float(window["started_at"]) >= float(
        policy["request_window_seconds"]
    ):
        window.update({"started_at": float(now), "request_count": 0, "upload_bytes": 0})
    if int(window["request_count"]) >= int(policy["maximum_requests_per_window"]):
        state["policy_counters"]["rate_limit_rejections"] += 1
        _raise("volunteer_request_rate_limited", status_code=429)
    byte_count = int(upload_bytes)
    if byte_count < 0:
        _raise("volunteer_upload_byte_count_invalid", status_code=400)
    if int(window["upload_bytes"]) + byte_count > int(
        policy["maximum_upload_bytes_per_window"]
    ):
        state["policy_counters"]["rate_limit_rejections"] += 1
        _raise("volunteer_upload_rate_limited", status_code=429)
    if int(record["upload_bytes"]) + byte_count > int(
        policy["maximum_upload_bytes_per_credential"]
    ):
        state["policy_counters"]["quota_rejections"] += 1
        _raise("volunteer_upload_quota_exceeded", status_code=429)
    if submission and int(record["submission_count"]) >= int(
        policy["maximum_submissions_per_credential"]
    ):
        state["policy_counters"]["quota_rejections"] += 1
        _raise("volunteer_submission_quota_exceeded", status_code=429)

    replay[replay_key] = float(now) + float(policy["replay_ttl_seconds"])
    window["request_count"] += 1
    window["upload_bytes"] += byte_count
    record["request_count"] += 1
    record["upload_bytes"] += byte_count
    if submission:
        record["submission_count"] += 1
    state["policy_counters"]["credential_authentications"] += 1
    return {
        "authentication_mode": "per_cell_short_lived_credential",
        "credential_id": record["credential_id"],
        "cell_id_hash": cell_hash,
        "required_scope": required_scope,
    }


def public_policy_status(state: dict[str, Any], *, now: float) -> dict[str, Any]:
    credentials = list(state.get("cell_credentials", {}).values())
    active = _active_credentials(state, now=now)
    policy = state["operator_policy"]
    return with_public_safety(
        {
            "schema": POLICY_STATUS_SCHEMA,
            "short_lived_per_cell_credentials": True,
            "scope_enforcement": True,
            "revocation_supported": True,
            "persistent_replay_protection": True,
            "request_rate_limit": True,
            "upload_rate_limit": True,
            "upload_quota": True,
            "submission_quota": True,
            "lease_concurrency_limit": True,
            "issued_credential_count": len(credentials),
            "active_credential_count": len(active),
            "revoked_credential_count": sum(bool(item.get("revoked")) for item in credentials),
            "expired_credential_count": sum(
                not item.get("revoked") and float(item.get("expires_at") or 0.0) <= float(now)
                for item in credentials
            ),
            "tracked_replay_nonce_count": len(state.get("replay_nonces", {})),
            "limits": {
                key: int(value)
                for key, value in policy.items()
                if key != "schema"
            },
            "counters": dict(state.get("policy_counters") or {}),
        }
    )
