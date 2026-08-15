"""Security contracts for controlled Community training deployments.

These controls reduce accidental exposure and reject malformed participants.
They do not provide Byzantine consensus, Sybil resistance, confidential
computing, secure aggregation, or semantic poisoning resistance.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shlex
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse


SECURITY_CONTRACT_SCHEMA = "crowdtensor_community_security_contract_v1"
CREDENTIAL_SCHEMA = "crowdtensor_short_lived_credential_v1"
SIGNED_TASK_SCHEMA = "crowdtensor_signed_training_task_v1"
EXECUTION_POLICY_SCHEMA = "crowdtensor_restricted_worker_execution_v1"
ANOMALY_SCHEMA = "crowdtensor_training_update_anomaly_v1"
PUBLIC_SAFETY_SCHEMA = "crowdtensor_public_safety_scan_v1"

ROLES = ("owner", "miner", "observer")
ROLE_PERMISSIONS = {
    "owner": frozenset(
        {
            "job:create",
            "job:control",
            "job:read",
            "job:export",
            "miner:admit",
            "credential:rotate",
        }
    ),
    "miner": frozenset(
        {
            "job:read_assignment",
            "task:claim",
            "task:heartbeat",
            "task:submit",
            "checkpoint:stage_write",
        }
    ),
    "observer": frozenset({"job:read", "metrics:read", "events:read_redacted"}),
}


class SecurityContractError(ValueError):
    """Fail-closed error with a public-safe reason code."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _digest(value: bytes | str) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def authorize(role: str, permission: str) -> dict[str, Any]:
    normalized = str(role or "").strip().lower()
    allowed = normalized in ROLE_PERMISSIONS and permission in ROLE_PERMISSIONS[normalized]
    return {
        "schema": "crowdtensor_community_rbac_decision_v1",
        "role": normalized if normalized in ROLES else "unknown",
        "permission": str(permission),
        "allowed": allowed,
        "default_deny": True,
        "public_artifact_safe": True,
    }


def require_permission(role: str, permission: str) -> None:
    if not authorize(role, permission)["allowed"]:
        raise SecurityContractError("community_rbac_permission_denied")


@dataclass
class ReplayWindow:
    ttl_seconds: float = 900.0
    maximum_entries: int = 100_000
    _entries: dict[str, float] = field(default_factory=dict)

    def consume(self, nonce: str, *, now: float | None = None) -> None:
        current = float(time.time() if now is None else now)
        expired = [key for key, expiry in self._entries.items() if expiry <= current]
        for key in expired:
            self._entries.pop(key, None)
        key = _digest(str(nonce))
        if key in self._entries:
            raise SecurityContractError("community_task_replay_detected")
        if len(self._entries) >= int(self.maximum_entries):
            raise SecurityContractError("community_replay_window_capacity_exceeded")
        self._entries[key] = current + float(self.ttl_seconds)


class CredentialAuthority:
    """HMAC credentials with bounded TTL and overlapping key rotation."""

    def __init__(
        self,
        *,
        issuer: str,
        key: bytes | None = None,
        key_id: str | None = None,
        maximum_ttl_seconds: int = 3600,
        clock_skew_seconds: int = 30,
    ) -> None:
        self.issuer_hash = _digest(str(issuer))
        self.maximum_ttl_seconds = int(maximum_ttl_seconds)
        self.clock_skew_seconds = int(clock_skew_seconds)
        self._keys: dict[str, bytes] = {}
        initial = key or secrets.token_bytes(32)
        initial_id = key_id or _digest(initial)[:23]
        self._keys[initial_id] = initial
        self.active_key_id = initial_id

    def issue(
        self,
        *,
        subject: str,
        role: str,
        scopes: Iterable[str] | None = None,
        ttl_seconds: int = 900,
        now: float | None = None,
    ) -> tuple[str, dict[str, Any]]:
        normalized = str(role).lower()
        if normalized not in ROLES:
            raise SecurityContractError("community_credential_role_invalid")
        ttl = int(ttl_seconds)
        if ttl < 1 or ttl > self.maximum_ttl_seconds:
            raise SecurityContractError("community_credential_ttl_invalid")
        issued = int(time.time() if now is None else now)
        normalized_scopes = sorted(
            {
                str(scope).strip()
                for scope in (scopes or ())
                if str(scope).strip()
            }
        )
        claims = {
            "schema": CREDENTIAL_SCHEMA,
            "issuer_hash": self.issuer_hash,
            "subject_hash": _digest(str(subject)),
            "role": normalized,
            "scopes": normalized_scopes,
            "issued_at": issued,
            "expires_at": issued + ttl,
            "jti": _b64(secrets.token_bytes(18)),
            "key_id": self.active_key_id,
        }
        payload = _b64(_canonical(claims))
        signature = _b64(hmac.new(self._keys[self.active_key_id], payload.encode(), hashlib.sha256).digest())
        token = payload + "." + signature
        public = {
            "schema": CREDENTIAL_SCHEMA,
            "issuer_hash": claims["issuer_hash"],
            "subject_hash": claims["subject_hash"],
            "role": normalized,
            "scopes": normalized_scopes,
            "issued_at": issued,
            "expires_at": issued + ttl,
            "ttl_seconds": ttl,
            "key_id_hash": _digest(self.active_key_id),
            "credential_value_public": False,
            "short_lived": True,
            "rotatable": True,
            "public_artifact_safe": True,
        }
        return token, public

    def verify(
        self,
        token: str,
        *,
        required_role: str | None = None,
        required_scope: str | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        try:
            payload, supplied = str(token).split(".", 1)
            claims = json.loads(_unb64(payload))
            key_id = str(claims["key_id"])
            key = self._keys[key_id]
            expected = hmac.new(key, payload.encode(), hashlib.sha256).digest()
            signature = _unb64(supplied)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise SecurityContractError("community_credential_invalid") from exc
        if not hmac.compare_digest(signature, expected):
            raise SecurityContractError("community_credential_signature_invalid")
        current = int(time.time() if now is None else now)
        if claims.get("schema") != CREDENTIAL_SCHEMA or claims.get("issuer_hash") != self.issuer_hash:
            raise SecurityContractError("community_credential_claims_invalid")
        if current + self.clock_skew_seconds < int(claims["issued_at"]):
            raise SecurityContractError("community_credential_not_yet_valid")
        if current - self.clock_skew_seconds >= int(claims["expires_at"]):
            raise SecurityContractError("community_credential_expired")
        if required_role and claims.get("role") != required_role:
            raise SecurityContractError("community_credential_role_mismatch")
        if required_scope and required_scope not in set(claims.get("scopes") or []):
            raise SecurityContractError("community_credential_scope_missing")
        return dict(claims)

    def rotate(self, *, retain_previous: int = 1) -> dict[str, Any]:
        previous = self.active_key_id
        key = secrets.token_bytes(32)
        key_id = _digest(key)[:23]
        self._keys[key_id] = key
        self.active_key_id = key_id
        keep = [key_id, previous][: max(1, int(retain_previous) + 1)]
        self._keys = {item: self._keys[item] for item in keep if item in self._keys}
        return {
            "schema": "crowdtensor_community_credential_rotation_v1",
            "rotated": True,
            "active_key_id_hash": _digest(key_id),
            "previous_key_retained": previous in self._keys,
            "retained_key_count": len(self._keys),
            "credential_values_public": False,
            "public_artifact_safe": True,
        }


class TaskEnvelopeSigner:
    def __init__(self, key: bytes, *, key_id: str = "training-task-key") -> None:
        if len(key) < 32:
            raise SecurityContractError("community_task_signing_key_too_short")
        self.key = key
        self.key_id_hash = _digest(key_id)

    def sign(
        self,
        payload: Mapping[str, Any],
        *,
        protocol_version: str,
        ttl_seconds: int = 300,
        now: float | None = None,
    ) -> dict[str, Any]:
        issued = int(time.time() if now is None else now)
        envelope = {
            "schema": SIGNED_TASK_SCHEMA,
            "protocol_version": str(protocol_version),
            "payload": dict(payload),
            "payload_hash": _digest(_canonical(payload)),
            "issued_at": issued,
            "expires_at": issued + int(ttl_seconds),
            "nonce": _b64(secrets.token_bytes(18)),
            "key_id_hash": self.key_id_hash,
        }
        envelope["signature"] = _b64(hmac.new(self.key, _canonical(envelope), hashlib.sha256).digest())
        return envelope

    def verify(
        self,
        envelope: Mapping[str, Any],
        *,
        replay_window: ReplayWindow,
        expected_protocol_version: str,
        now: float | None = None,
    ) -> dict[str, Any]:
        value = dict(envelope)
        supplied = str(value.pop("signature", ""))
        expected = hmac.new(self.key, _canonical(value), hashlib.sha256).digest()
        try:
            actual = _unb64(supplied)
        except ValueError as exc:
            raise SecurityContractError("community_task_signature_invalid") from exc
        if not hmac.compare_digest(actual, expected):
            raise SecurityContractError("community_task_signature_invalid")
        current = int(time.time() if now is None else now)
        if value.get("schema") != SIGNED_TASK_SCHEMA:
            raise SecurityContractError("community_task_envelope_schema_invalid")
        if value.get("protocol_version") != expected_protocol_version:
            raise SecurityContractError("community_task_protocol_mismatch")
        if current < int(value["issued_at"]) or current >= int(value["expires_at"]):
            raise SecurityContractError("community_task_envelope_expired")
        if value.get("payload_hash") != _digest(_canonical(value.get("payload"))):
            raise SecurityContractError("community_task_payload_hash_mismatch")
        replay_window.consume(str(value.get("nonce") or ""), now=current)
        return dict(value["payload"])


@dataclass(frozen=True)
class TLSProxyPolicy:
    require_https: bool = True
    trust_forwarded_headers: bool = False
    trusted_proxy_hashes: tuple[str, ...] = ()

    def validate(self, *, scheme: str, forwarded_proto: str = "", proxy_identity: str = "") -> dict[str, Any]:
        effective = str(scheme).lower()
        forwarded_accepted = False
        if self.trust_forwarded_headers and forwarded_proto:
            identity_hash = _digest(proxy_identity)
            if identity_hash not in self.trusted_proxy_hashes:
                raise SecurityContractError("community_tls_untrusted_proxy")
            effective = str(forwarded_proto).split(",", 1)[0].strip().lower()
            forwarded_accepted = True
        ready = not self.require_https or effective == "https"
        if not ready:
            raise SecurityContractError("community_tls_https_required")
        return {
            "schema": "crowdtensor_tls_proxy_contract_v1",
            "tls_required": self.require_https,
            "effective_scheme": effective,
            "forwarded_header_accepted": forwarded_accepted,
            "trusted_proxy_identity_public": False,
            "ready": ready,
            "public_artifact_safe": True,
        }


@dataclass(frozen=True)
class RestrictedExecutionPolicy:
    allowed_executables: tuple[str, ...] = ("python", "python3")
    allowed_modules: tuple[str, ...] = ("crowdtensor.volunteer_training_cell",)
    allowed_file_roots: tuple[str, ...] = ()
    allowed_network_hosts: tuple[str, ...] = ()
    maximum_memory_bytes: int = 16 * 1024**3
    maximum_cpu_seconds: int = 3600
    maximum_output_bytes: int = 64 * 1024**2
    allow_shell: bool = False

    def validate(
        self,
        command: Iterable[str],
        *,
        file_paths: Iterable[str] = (),
        network_urls: Iterable[str] = (),
        memory_bytes: int = 0,
        cpu_seconds: int = 0,
        output_bytes: int = 0,
    ) -> dict[str, Any]:
        argv = [str(item) for item in command]
        reasons: list[str] = []
        if not argv or Path(argv[0]).name not in self.allowed_executables:
            reasons.append("restricted_worker_executable_denied")
        if any(any(token in arg for token in (";", "&&", "||", "`", "$(")) for arg in argv):
            reasons.append("restricted_worker_shell_syntax_denied")
        module = ""
        if "-m" in argv:
            index = argv.index("-m")
            module = argv[index + 1] if index + 1 < len(argv) else ""
        if module not in self.allowed_modules:
            reasons.append("restricted_worker_module_denied")
        roots = [Path(root).expanduser().resolve() for root in self.allowed_file_roots]
        for value in file_paths:
            resolved = Path(value).expanduser().resolve()
            if not any(resolved == root or root in resolved.parents for root in roots):
                reasons.append("restricted_worker_file_path_denied")
                break
        for value in network_urls:
            host = (urlparse(str(value)).hostname or "").lower()
            if host not in self.allowed_network_hosts:
                reasons.append("restricted_worker_network_host_denied")
                break
        if int(memory_bytes) < 0 or int(memory_bytes) > self.maximum_memory_bytes:
            reasons.append("restricted_worker_memory_limit_exceeded")
        if int(cpu_seconds) < 0 or int(cpu_seconds) > self.maximum_cpu_seconds:
            reasons.append("restricted_worker_cpu_limit_exceeded")
        if int(output_bytes) < 0 or int(output_bytes) > self.maximum_output_bytes:
            reasons.append("restricted_worker_output_limit_exceeded")
        return {
            "schema": EXECUTION_POLICY_SCHEMA,
            "allowed": not reasons,
            "rejection_reasons": sorted(set(reasons)),
            "executable": Path(argv[0]).name if argv else "",
            "module": module,
            "shell_allowed": self.allow_shell,
            "file_path_values_public": False,
            "network_url_values_public": False,
            "resource_limits": {
                "maximum_memory_bytes": self.maximum_memory_bytes,
                "maximum_cpu_seconds": self.maximum_cpu_seconds,
                "maximum_output_bytes": self.maximum_output_bytes,
            },
            "public_artifact_safe": True,
        }

    def command_text(self, command: Iterable[str]) -> str:
        report = self.validate(command)
        if not report["allowed"]:
            raise SecurityContractError(str(report["rejection_reasons"][0]))
        return shlex.join(str(item) for item in command)


@dataclass
class UpdateAnomalyDetector:
    absolute_norm_limit: float = 1_000.0
    history_factor_limit: float = 10.0
    quarantine_after: int = 2
    history_size: int = 100
    _accepted_norms: deque[float] = field(default_factory=deque)
    _violations: dict[str, int] = field(default_factory=dict)

    def inspect(self, *, miner_id: str, values: Iterable[float], expected_count: int) -> dict[str, Any]:
        materialized = [float(item) for item in values]
        reasons: list[str] = []
        if len(materialized) != int(expected_count):
            reasons.append("training_update_shape_anomaly")
        if not all(math.isfinite(item) for item in materialized):
            reasons.append("training_update_non_finite")
        norm = math.sqrt(sum(item * item for item in materialized)) if not reasons else math.inf
        baseline = sorted(self._accepted_norms)
        median = baseline[len(baseline) // 2] if baseline else 0.0
        if norm > self.absolute_norm_limit:
            reasons.append("training_update_absolute_norm_anomaly")
        if median > 0 and norm > median * self.history_factor_limit:
            reasons.append("training_update_history_norm_anomaly")
        miner_hash = _digest(str(miner_id))
        if reasons:
            self._violations[miner_hash] = self._violations.get(miner_hash, 0) + 1
        else:
            self._accepted_norms.append(norm)
            while len(self._accepted_norms) > self.history_size:
                self._accepted_norms.popleft()
        quarantined = self._violations.get(miner_hash, 0) >= self.quarantine_after
        return {
            "schema": ANOMALY_SCHEMA,
            "accepted": not reasons,
            "quarantined": quarantined,
            "miner_id_hash": miner_hash,
            "reason_codes": sorted(set(reasons)),
            "element_count": len(materialized),
            "norm_finite": math.isfinite(norm),
            "norm_limit": self.absolute_norm_limit,
            "raw_values_public": False,
            "semantic_poisoning_detected": False,
            "public_artifact_safe": True,
        }


_SENSITIVE_KEY = re.compile(
    r"(^|_)(token|cookie|authorization|secret|password|api_key|private_key|hidden_b64|gradient|activation|tensor|token_ids?)(_|$)",
    re.IGNORECASE,
)
_SAFE_BOOLEAN_SUFFIXES = ("_public", "_present", "_redacted", "_verified", "_count", "_hash")
_SECRET_TEXT = [
    re.compile(r"(?i)authorization\s*:\s*bearer\s+\S+"),
    re.compile(r"(?i)(?:api[_-]?key|token|secret|password)\s*[=:]\s*['\"]?[A-Za-z0-9_./+\-=]{12,}"),
    re.compile(r"(?i)(?:set-cookie|cookie)\s*:\s*\S+"),
    re.compile(r"(?i)X-Amz-(?:Credential|Signature)="),
]
_ABSOLUTE_PATH = re.compile(r"(?:^|[\s'\"])(/(?:root|home|Users|tmp|var/tmp)/[^\s'\"]+|[A-Za-z]:\\[^\s'\"]+)")
_PRIVATE_URL = re.compile(
    r"(?i)https?://(?:localhost|127(?:\.[0-9]+){3}|10(?:\.[0-9]+){3}|192\.168(?:\.[0-9]+){2}|172\.(?:1[6-9]|2[0-9]|3[01])(?:\.[0-9]+){2})\b"
)


def scan_public_value(value: Any) -> dict[str, Any]:
    findings: list[dict[str, str]] = []

    def add(category: str, location: str) -> None:
        findings.append({"category": category, "location_hash": _digest(location)})

    def walk(current: Any, location: str) -> None:
        if isinstance(current, Mapping):
            for raw_key, child in current.items():
                key = str(raw_key)
                lowered = key.lower()
                if (
                    _SENSITIVE_KEY.search(key)
                    and not lowered.endswith(_SAFE_BOOLEAN_SUFFIXES)
                    and lowered not in {"token_count", "generated_token_count", "tensor_count"}
                    and child not in (None, "", False, 0, [], {})
                ):
                    add("sensitive_key_has_value", location + "." + key)
                walk(child, location + "." + key)
        elif isinstance(current, (list, tuple)):
            for index, child in enumerate(current):
                walk(child, f"{location}[{index}]")
        elif isinstance(current, str):
            for pattern in _SECRET_TEXT:
                if pattern.search(current):
                    add("credential_material", location)
                    break
            if _ABSOLUTE_PATH.search(current):
                add("absolute_path", location)
            if _PRIVATE_URL.search(current):
                add("private_url", location)

    walk(value, "$")
    categories = sorted({item["category"] for item in findings})
    return {
        "schema": PUBLIC_SAFETY_SCHEMA,
        "ok": not findings,
        "finding_count": len(findings),
        "categories": categories,
        "findings": findings,
        "sensitive_values_public": False,
        "public_artifact_safe": not findings,
    }


def scan_public_files(paths: Iterable[str | Path]) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for index, raw in enumerate(paths):
        path = Path(raw)
        try:
            text = path.read_text(encoding="utf-8")
            value: Any = json.loads(text) if path.suffix.lower() == ".json" else text
            report = scan_public_value(value)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            report = {
                "schema": PUBLIC_SAFETY_SCHEMA,
                "ok": False,
                "finding_count": 1,
                "categories": ["unscannable_public_file"],
                "findings": [],
                "public_artifact_safe": False,
            }
        reports.append(
            {
                "file_index": index,
                "file_name_hash": _digest(path.name),
                "ok": report["ok"],
                "finding_count": report["finding_count"],
                "categories": report["categories"],
            }
        )
    return {
        "schema": PUBLIC_SAFETY_SCHEMA,
        "ok": all(item["ok"] for item in reports),
        "file_count": len(reports),
        "finding_count": sum(int(item["finding_count"]) for item in reports),
        "files": reports,
        "absolute_file_paths_public": False,
        "public_artifact_safe": all(item["ok"] for item in reports),
    }


def security_contract_report() -> dict[str, Any]:
    return {
        "schema": SECURITY_CONTRACT_SCHEMA,
        "roles": list(ROLES),
        "role_permissions": {role: sorted(values) for role, values in ROLE_PERMISSIONS.items()},
        "default_deny_rbac": True,
        "short_lived_rotatable_credentials": True,
        "task_signatures": "hmac_sha256",
        "replay_protection": True,
        "tls_proxy_contract": True,
        "restricted_worker_execution": True,
        "resource_limits": True,
        "anomaly_quarantine_interface": True,
        "public_safety_scanner": True,
        "unresolved_security_boundaries": sorted(
            [
                "byzantine_fault_tolerance",
                "confidential_computing_or_tee",
                "privacy_preserving_computation",
                "secure_aggregation",
                "semantic_poisoning_resistance",
                "sybil_resistance",
            ]
        ),
        "public_artifact_safe": True,
    }
