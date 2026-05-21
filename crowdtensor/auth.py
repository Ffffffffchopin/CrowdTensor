"""Token matching helpers for CrowdTensorD control-plane auth."""

from __future__ import annotations

import hashlib
import hmac
import re


HASH_PREFIX = "sha256:"
_SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")


def hash_token(token: str) -> str:
    """Return a sha256 token verifier string for a non-empty token."""
    text = str(token)
    if not text:
        raise ValueError("token must be non-empty")
    return f"{HASH_PREFIX}{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def validate_token_verifier(verifier: str, *, field_name: str = "token") -> str:
    """Validate a plaintext or sha256 token verifier and return the stripped value."""
    text = str(verifier or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    if text.startswith(HASH_PREFIX):
        digest = text[len(HASH_PREFIX):]
        if not _SHA256_HEX.fullmatch(digest):
            raise ValueError(f"{field_name} sha256 digest must be 64 hex characters")
    return text


def token_matches(candidate: str | None, verifier: str | None) -> bool:
    """Match a request token against a plaintext or sha256 verifier."""
    if candidate is None or verifier is None:
        return False
    expected = str(verifier)
    token = str(candidate)
    if expected.startswith(HASH_PREFIX):
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return hmac.compare_digest(digest, expected[len(HASH_PREFIX):])
    return hmac.compare_digest(token, expected)
