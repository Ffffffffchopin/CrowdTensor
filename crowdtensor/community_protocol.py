"""Fail-closed protocol negotiation for Community training participants."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .version import COMMUNITY_PROTOCOL_VERSION


PROTOCOL_COMPATIBILITY_SCHEMA = "crowdtensor_community_protocol_compatibility_v1"
_VERSION = re.compile(r"^(?P<family>[a-z][a-z0-9_]*)_v(?P<major>[0-9]+)\.(?P<minor>[0-9]+)$")


class ProtocolCompatibilityError(ValueError):
    """Raised when a participant cannot safely join the current protocol."""


@dataclass(frozen=True)
class ParsedProtocol:
    family: str
    major: int
    minor: int


def parse_protocol_version(value: str) -> ParsedProtocol:
    match = _VERSION.fullmatch(str(value or "").strip())
    if match is None:
        raise ProtocolCompatibilityError("community_protocol_version_invalid")
    return ParsedProtocol(
        family=match.group("family"),
        major=int(match.group("major")),
        minor=int(match.group("minor")),
    )


def negotiate_protocol(
    peer_version: str,
    *,
    peer_capabilities: Iterable[str] = (),
    required_capabilities: Iterable[str] = (),
    local_version: str = COMMUNITY_PROTOCOL_VERSION,
) -> dict[str, Any]:
    """Accept same-major peers and reject downgrade/unknown capability gaps."""

    local = parse_protocol_version(local_version)
    peer = parse_protocol_version(peer_version)
    supplied = sorted({str(item).strip() for item in peer_capabilities if str(item).strip()})
    required = sorted({str(item).strip() for item in required_capabilities if str(item).strip()})
    missing = sorted(set(required) - set(supplied))
    reasons: list[str] = []
    if peer.family != local.family:
        reasons.append("community_protocol_family_incompatible")
    if peer.major != local.major:
        reasons.append("community_protocol_major_incompatible")
    if peer.minor > local.minor:
        reasons.append("community_protocol_peer_minor_newer")
    if missing:
        reasons.append("community_protocol_required_capability_missing")
    accepted = not reasons
    return {
        "schema": PROTOCOL_COMPATIBILITY_SCHEMA,
        "accepted": accepted,
        "local_version": local_version,
        "peer_version": peer_version,
        "negotiated_version": local_version if accepted else "",
        "required_capabilities": required,
        "missing_capabilities": missing,
        "rejection_reasons": reasons,
        "silent_downgrade_allowed": False,
        "unknown_major_rejected": True,
        "public_artifact_safe": True,
    }


def require_compatible_protocol(*args: Any, **kwargs: Any) -> dict[str, Any]:
    report = negotiate_protocol(*args, **kwargs)
    if not report["accepted"]:
        raise ProtocolCompatibilityError(
            str(report["rejection_reasons"][0] or "community_protocol_incompatible")
        )
    return report
