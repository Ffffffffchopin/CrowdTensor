"""Helpers for creating role-scoped operator registry entries."""

from __future__ import annotations

import base64
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

from crowdtensor.auth import hash_token, validate_token_verifier


ALLOWED_OPERATOR_ROLES = {"owner", "admin", "accounting", "auditor"}
WORKLOAD_ALIASES = {
    "model-bundle": "model_bundle_infer",
    "external-llm": "external_llm_infer",
    "sharded-model-bundle": "sharded_model_bundle_infer",
    "sharded": "sharded_model_bundle_infer",
    "micro-llm-sharded": "micro_llm_sharded_infer",
    "micro-llm-shard": "micro_llm_sharded_infer",
    "real-llm-sharded": "real_llm_sharded_infer",
    "real-llm-shard": "real_llm_sharded_infer",
}


def load_operator_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"operators": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid operator registry JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("operator registry must be a JSON object")
    operators = payload.get("operators")
    if operators is None:
        payload["operators"] = []
    elif not isinstance(operators, list):
        raise ValueError("operator registry operators must be a list")
    return payload


def write_operator_registry(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def normalize_operator_roles(roles: list[str] | str) -> list[str]:
    raw_roles = [roles] if isinstance(roles, str) else list(roles or [])
    normalized = sorted({str(role or "").strip() for role in raw_roles if str(role or "").strip()})
    if not normalized:
        raise ValueError("at least one role is required")
    unknown = sorted(role for role in normalized if role not in ALLOWED_OPERATOR_ROLES)
    if unknown:
        raise ValueError(f"unknown operator roles: {', '.join(unknown)}")
    return normalized


def normalize_allowed_workloads(values: list[str] | str) -> list[str]:
    raw_values = [values] if isinstance(values, str) else list(values or [])
    normalized = []
    for value in raw_values:
        workload = str(value or "").strip()
        if workload:
            normalized.append(WORKLOAD_ALIASES.get(workload, workload))
    return sorted(set(normalized))


def _non_negative_int(name: str, value: int) -> int:
    result = int(value or 0)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _non_negative_float(name: str, value: float) -> float:
    result = float(value or 0.0)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def build_session_policy(
    *,
    allowed_workloads: list[str] | str | None = None,
    max_request_count: int = 0,
    max_decode_steps: int = 0,
    max_new_tokens: int = 0,
    max_active_sessions: int = 0,
    max_total_sessions: int = 0,
    rate_limit: int = 0,
    rate_window_seconds: float = 0.0,
) -> dict[str, Any]:
    policy = {
        "schema": "crowdtensor_operator_session_policy_v1",
        "allowed_workloads": normalize_allowed_workloads(allowed_workloads or []),
        "max_request_count": _non_negative_int("max_request_count", max_request_count),
        "max_decode_steps": _non_negative_int("max_decode_steps", max_decode_steps),
        "max_new_tokens": _non_negative_int("max_new_tokens", max_new_tokens),
        "max_active_sessions": _non_negative_int("max_active_sessions", max_active_sessions),
        "max_total_sessions": _non_negative_int("max_total_sessions", max_total_sessions),
        "rate_limit": _non_negative_int("rate_limit", rate_limit),
        "rate_window_seconds": _non_negative_float("rate_window_seconds", rate_window_seconds),
    }
    if (policy["rate_limit"] > 0) != (policy["rate_window_seconds"] > 0):
        raise ValueError("rate_limit and rate_window_seconds must be set together")
    has_limit = bool(
        policy["allowed_workloads"]
        or policy["max_request_count"] > 0
        or policy["max_decode_steps"] > 0
        or policy["max_new_tokens"] > 0
        or policy["max_active_sessions"] > 0
        or policy["max_total_sessions"] > 0
        or policy["rate_limit"] > 0
    )
    return policy if has_limit else {}


def create_operator_invite(
    *,
    registry_path: Path,
    operator_id: str,
    roles: list[str] | str,
    label: str = "",
    token: str = "",
    replace: bool = False,
    allowed_workloads: list[str] | str | None = None,
    max_request_count: int = 0,
    max_decode_steps: int = 0,
    max_new_tokens: int = 0,
    max_active_sessions: int = 0,
    max_total_sessions: int = 0,
    rate_limit: int = 0,
    rate_window_seconds: float = 0.0,
    invite_file: Path | None = None,
) -> dict[str, Any]:
    operator_name = str(operator_id or "").strip()
    if not operator_name:
        raise ValueError("operator_id is required")
    normalized_roles = normalize_operator_roles(roles)
    session_policy = build_session_policy(
        allowed_workloads=allowed_workloads or [],
        max_request_count=max_request_count,
        max_decode_steps=max_decode_steps,
        max_new_tokens=max_new_tokens,
        max_active_sessions=max_active_sessions,
        max_total_sessions=max_total_sessions,
        rate_limit=rate_limit,
        rate_window_seconds=rate_window_seconds,
    )
    plaintext_token = token or secrets.token_urlsafe(32)
    token_hash = hash_token(plaintext_token)
    registry = load_operator_registry(registry_path)
    operators = registry.setdefault("operators", [])
    now = int(time.time())
    entry = {
        "enabled": True,
        "label": str(label or ""),
        "operator_id": operator_name,
        "roles": normalized_roles,
        "token": validate_token_verifier(token_hash, field_name="operator token"),
        "updated_at": now,
    }
    if session_policy:
        entry["session_policy"] = session_policy
    existing_index = next(
        (index for index, item in enumerate(operators) if isinstance(item, dict) and item.get("operator_id") == operator_name),
        None,
    )
    if existing_index is not None and not replace:
        raise ValueError(f"operator_id {operator_name!r} already exists; pass --replace to update it")
    if existing_index is None:
        entry["created_at"] = now
        operators.append(entry)
    else:
        previous = operators[existing_index]
        entry["created_at"] = int(previous.get("created_at", now)) if isinstance(previous, dict) else now
        operators[existing_index] = entry

    write_operator_registry(registry_path, registry)
    invite = {
        "schema": "crowdtensor_operator_invite_v1",
        "operator_id": operator_name,
        "operator_token": plaintext_token,
        "token_hash": token_hash,
        "roles": normalized_roles,
        "label": str(label or ""),
        "session_policy": session_policy,
        "registry_path_hint": str(registry_path),
        "public_artifact_safe": False,
    }
    invite_code = base64.urlsafe_b64encode(
        json.dumps(invite, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    invite_file_path = ""
    if invite_file is not None:
        invite_file.parent.mkdir(parents=True, exist_ok=True)
        invite_file.write_text(json.dumps(invite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        invite_file_path = str(invite_file)
    return {
        "schema": "crowdtensor_operator_invite_result_v1",
        "operator_id": operator_name,
        "roles": normalized_roles,
        "registry": str(registry_path),
        "invite_file": invite_file_path,
        "operator_invite": invite,
        "operator_invite_code": invite_code,
        "env": {"CROWDTENSOR_ADMIN_TOKEN": plaintext_token},
        "token_hash": token_hash,
        "session_policy": session_policy,
        "plaintext_token_public": False,
        "registry_plaintext_token_public": False,
    }
