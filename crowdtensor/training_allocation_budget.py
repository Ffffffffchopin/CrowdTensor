"""Validated one-time allocation-budget amendment for the CUDA training RC."""

from __future__ import annotations

from typing import Any


AMENDMENT_SCHEMA = "crowdtensor_cuda_training_allocation_budget_amendment_v1"
ORIGINAL_ATTEMPT_LIMIT = 2
MAX_AUTHORIZED_ATTEMPT_LIMIT = 3


def allocation_budget_summary(ledger: dict[str, Any]) -> dict[str, Any]:
    amendment = dict(ledger.get("allocation_budget_amendment") or {})
    authorization_hash = str(amendment.get("authorization_hash") or "")
    valid = bool(
        amendment.get("schema") == AMENDMENT_SCHEMA
        and amendment.get("authorized") is True
        and amendment.get("authorization_text_public") is False
        and amendment.get("same_authorized_account_only") is True
        and int(amendment.get("original_single_kernel_attempt_limit") or 0)
        == ORIGINAL_ATTEMPT_LIMIT
        and int(amendment.get("original_two_node_attempt_limit") or 0)
        == ORIGINAL_ATTEMPT_LIMIT
        and int(amendment.get("additional_single_kernel_attempts") or 0) == 1
        and int(amendment.get("additional_two_node_attempts") or 0) == 1
        and int(amendment.get("revised_single_kernel_attempt_limit") or 0)
        == MAX_AUTHORIZED_ATTEMPT_LIMIT
        and int(amendment.get("revised_two_node_attempt_limit") or 0)
        == MAX_AUTHORIZED_ATTEMPT_LIMIT
        and 0 < int(amendment.get("allocation_timeout_seconds") or 0) <= 1800
        and authorization_hash.startswith("sha256:")
        and len(authorization_hash) == len("sha256:") + 64
        and bool(str(amendment.get("authorized_at") or ""))
    )
    single_limit = MAX_AUTHORIZED_ATTEMPT_LIMIT if valid else ORIGINAL_ATTEMPT_LIMIT
    two_node_limit = MAX_AUTHORIZED_ATTEMPT_LIMIT if valid else ORIGINAL_ATTEMPT_LIMIT
    return {
        "schema": "crowdtensor_cuda_training_allocation_budget_summary_v1",
        "amendment_present": bool(amendment),
        "amendment_valid": valid,
        "single_kernel_attempt_limit": single_limit,
        "two_node_attempt_limit": two_node_limit,
        "additional_single_kernel_attempts": 1 if valid else 0,
        "additional_two_node_attempts": 1 if valid else 0,
        "allocation_timeout_seconds": (
            int(amendment.get("allocation_timeout_seconds") or 0) if valid else 1800
        ),
        "same_authorized_account_only": valid,
        "authorization_hash": authorization_hash if valid else "",
        "authorization_text_public": False,
        "credential_values_public": False,
        "public_artifact_safe": True,
    }


def require_attempt_limit(ledger: dict[str, Any], *, kind: str, requested_limit: int) -> int:
    summary = allocation_budget_summary(ledger)
    key = (
        "single_kernel_attempt_limit"
        if kind == "single_kernel"
        else "two_node_attempt_limit"
    )
    allowed = int(summary[key])
    if int(requested_limit) > allowed:
        raise RuntimeError(f"{kind}_allocation_attempt_limit_not_authorized")
    return min(int(requested_limit), allowed)
