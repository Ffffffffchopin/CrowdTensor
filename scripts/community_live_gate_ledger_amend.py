#!/usr/bin/env python3
"""Apply the one-time Community full-live-gate 2 -> 3 authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LEDGER_SCHEMA = "crowdtensor_community_live_gate_ledger_v1"
REPORT_SCHEMA = "crowdtensor_community_live_gate_ledger_amendment_v1"
ORIGINAL_MAXIMUM = 2
AMENDED_MAXIMUM = 3
AMENDMENT_SCOPE = (
    "one_additional_kaggle_cpu_gpu_full_live_gate_maximum_2700_seconds_"
    "other_boundaries_unchanged"
)
AMENDMENT_FIELDS = {
    "approval_statement_hash",
    "amended_at",
    "old_maximum_full_live_gates",
    "new_maximum_full_live_gates",
    "scope",
}


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + secrets.token_hex(4) + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _timestamp_valid(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo is not None


def validate_amended_ledger(
    ledger: object, *, expected_attempt_count: int | None = None
) -> list[str]:
    errors: list[str] = []
    value = ledger if isinstance(ledger, dict) else {}
    if value.get("schema") != LEDGER_SCHEMA:
        errors.append("community_live_gate_ledger_schema_invalid")
    if int(value.get("maximum_full_live_gates") or 0) != AMENDED_MAXIMUM:
        errors.append("community_live_gate_ledger_maximum_not_amended")
    if value.get("public_artifact_safe") is not True:
        errors.append("community_live_gate_ledger_public_safety_invalid")

    attempts = value.get("attempts") if isinstance(value.get("attempts"), list) else []
    if expected_attempt_count is not None and len(attempts) != expected_attempt_count:
        errors.append("community_live_gate_ledger_attempt_count_invalid")
    if len(attempts) > AMENDED_MAXIMUM:
        errors.append("community_live_gate_ledger_attempt_limit_exceeded")
    if [int(item.get("attempt") or 0) for item in attempts if isinstance(item, dict)] != list(
        range(1, len(attempts) + 1)
    ):
        errors.append("community_live_gate_ledger_attempt_sequence_invalid")

    amendments = value.get("amendments") if isinstance(value.get("amendments"), list) else []
    if len(amendments) != 1 or not isinstance(amendments[0], dict):
        errors.append("community_live_gate_ledger_amendment_count_invalid")
        return sorted(set(errors))
    amendment = amendments[0]
    if set(amendment) != AMENDMENT_FIELDS:
        errors.append("community_live_gate_ledger_amendment_fields_invalid")
    if not re.fullmatch(
        r"sha256:[0-9a-f]{64}", str(amendment.get("approval_statement_hash") or "")
    ):
        errors.append("community_live_gate_ledger_approval_hash_invalid")
    if not _timestamp_valid(amendment.get("amended_at")):
        errors.append("community_live_gate_ledger_amendment_timestamp_invalid")
    if int(amendment.get("old_maximum_full_live_gates") or 0) != ORIGINAL_MAXIMUM:
        errors.append("community_live_gate_ledger_old_maximum_invalid")
    if int(amendment.get("new_maximum_full_live_gates") or 0) != AMENDED_MAXIMUM:
        errors.append("community_live_gate_ledger_new_maximum_invalid")
    if amendment.get("scope") != AMENDMENT_SCOPE:
        errors.append("community_live_gate_ledger_amendment_scope_invalid")
    return sorted(set(errors))


def amend_ledger(
    path: str | Path,
    *,
    approval_statement: str,
    amended_at: str | None = None,
) -> dict[str, Any]:
    ledger_path = Path(path).expanduser().resolve()
    if not approval_statement.strip():
        raise ValueError("community_live_gate_approval_statement_required")
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("community_live_gate_ledger_unreadable") from exc
    if not isinstance(ledger, dict) or ledger.get("schema") != LEDGER_SCHEMA:
        raise ValueError("community_live_gate_ledger_schema_invalid")

    approval_hash = _hash(approval_statement)
    if int(ledger.get("maximum_full_live_gates") or 0) == AMENDED_MAXIMUM:
        errors = validate_amended_ledger(ledger, expected_attempt_count=ORIGINAL_MAXIMUM)
        amendment = (ledger.get("amendments") or [{}])[0]
        if errors or amendment.get("approval_statement_hash") != approval_hash:
            raise ValueError("community_live_gate_existing_amendment_mismatch")
        changed = False
    else:
        attempts = ledger.get("attempts") if isinstance(ledger.get("attempts"), list) else []
        if int(ledger.get("maximum_full_live_gates") or 0) != ORIGINAL_MAXIMUM:
            raise ValueError("community_live_gate_original_maximum_invalid")
        if len(attempts) != ORIGINAL_MAXIMUM:
            raise ValueError("community_live_gate_original_attempt_count_invalid")
        if [int(item.get("attempt") or 0) for item in attempts if isinstance(item, dict)] != [1, 2]:
            raise ValueError("community_live_gate_original_attempt_sequence_invalid")
        if any(
            not isinstance(item, dict)
            or float(item.get("completed_at") or 0.0) <= 0
            or str(item.get("outcome") or "") == "running"
            for item in attempts
        ):
            raise ValueError("community_live_gate_original_attempt_incomplete")
        if ledger.get("amendments") not in (None, []):
            raise ValueError("community_live_gate_unexpected_existing_amendment")
        timestamp = amended_at or datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )
        amendment = {
            "approval_statement_hash": approval_hash,
            "amended_at": timestamp,
            "old_maximum_full_live_gates": ORIGINAL_MAXIMUM,
            "new_maximum_full_live_gates": AMENDED_MAXIMUM,
            "scope": AMENDMENT_SCOPE,
        }
        ledger["maximum_full_live_gates"] = AMENDED_MAXIMUM
        ledger["amendments"] = [amendment]
        errors = validate_amended_ledger(ledger, expected_attempt_count=ORIGINAL_MAXIMUM)
        if errors:
            raise ValueError("community_live_gate_amendment_invalid:" + ",".join(errors))
        _write_atomic(ledger_path, ledger)
        changed = True

    return {
        "schema": REPORT_SCHEMA,
        "ok": True,
        "changed": changed,
        "approval_statement_hash": approval_hash,
        "old_maximum_full_live_gates": ORIGINAL_MAXIMUM,
        "new_maximum_full_live_gates": AMENDED_MAXIMUM,
        "scope": AMENDMENT_SCOPE,
        "attempt_count": len(ledger.get("attempts") or []),
        "attempt_history_preserved": True,
        "approval_statement_public": False,
        "public_artifact_safe": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--approval-statement", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = amend_ledger(args.ledger, approval_statement=args.approval_statement)
    print(json.dumps(report, sort_keys=True) if args.json else "ledger_amended=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
