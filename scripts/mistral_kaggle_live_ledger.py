#!/usr/bin/env python3
"""Independent bounded-attempt ledger for the Mistral Adapter live gate."""

from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

from crowdtensor.model_adapter import stable_hash


SCHEMA = "crowdtensor_mistral_kaggle_live_gate_ledger_v1"
MAXIMUM_ATTEMPTS = 2


def _payload_hash(value: dict[str, Any]) -> str:
    return stable_hash({key: item for key, item in value.items() if key != "content_hash"})


def validate_ledger(value: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        return ["mistral_live_gate_ledger_schema_invalid"]
    attempts = value.get("attempts")
    if value.get("maximum_attempts") != MAXIMUM_ATTEMPTS:
        errors.append("mistral_live_gate_ledger_maximum_invalid")
    if not isinstance(attempts, list) or len(attempts) > MAXIMUM_ATTEMPTS:
        errors.append("mistral_live_gate_ledger_attempts_invalid")
        attempts = []
    if [item.get("attempt") for item in attempts if isinstance(item, dict)] != list(
        range(1, len(attempts) + 1)
    ):
        errors.append("mistral_live_gate_ledger_sequence_invalid")
    if any(
        not isinstance(item, dict)
        or item.get("outcome") not in {"running", "achieved", "failed"}
        or float(item.get("started_at") or 0.0) <= 0.0
        or (
            item.get("outcome") != "running"
            and float(item.get("completed_at") or 0.0)
            < float(item.get("started_at") or 0.0)
        )
        for item in attempts
    ):
        errors.append("mistral_live_gate_ledger_attempt_record_invalid")
    if sum(item.get("outcome") == "running" for item in attempts) > 1:
        errors.append("mistral_live_gate_ledger_multiple_running_attempts")
    if value.get("community_maturity_ledger_modified") is not False:
        errors.append("mistral_live_gate_ledger_scope_invalid")
    if value.get("public_artifact_safe") is not True:
        errors.append("mistral_live_gate_ledger_public_safety_invalid")
    if value.get("content_hash") != _payload_hash(value):
        errors.append("mistral_live_gate_ledger_hash_invalid")
    return sorted(set(errors))


def _write(path: Path, value: dict[str, Any]) -> None:
    value["content_hash"] = _payload_hash(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + secrets.token_hex(4) + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_or_create(path: str | Path) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if target.is_file():
        value = json.loads(target.read_text(encoding="utf-8"))
        errors = validate_ledger(value)
        if errors:
            raise RuntimeError(errors[0])
        return value
    value = {
        "schema": SCHEMA,
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "attempts": [],
        "scope": "Mistral Adapter Kaggle CPU+CUDA heterogeneous live gate only",
        "community_maturity_ledger_modified": False,
        "credential_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    _write(target, value)
    return value


def reserve_attempt(path: str | Path) -> int:
    target = Path(path).expanduser().resolve()
    value = load_or_create(target)
    attempts = list(value["attempts"])
    if any(item.get("outcome") == "running" for item in attempts):
        raise RuntimeError("mistral_live_gate_attempt_already_running")
    if len(attempts) >= MAXIMUM_ATTEMPTS:
        raise RuntimeError("mistral_live_gate_attempt_limit_reached")
    number = len(attempts) + 1
    attempts.append(
        {
            "attempt": number,
            "started_at": time.time(),
            "completed_at": 0.0,
            "outcome": "running",
        }
    )
    value["attempts"] = attempts
    _write(target, value)
    return number


def finish_attempt(path: str | Path, *, achieved: bool) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    value = load_or_create(target)
    attempts = list(value["attempts"])
    if not attempts or attempts[-1].get("outcome") != "running":
        raise RuntimeError("mistral_live_gate_running_attempt_missing")
    attempts[-1]["completed_at"] = time.time()
    attempts[-1]["outcome"] = "achieved" if achieved else "failed"
    value["attempts"] = attempts
    _write(target, value)
    errors = validate_ledger(value)
    if errors:
        raise RuntimeError(errors[0])
    return value


def supersede_attempt(
    path: str | Path,
    *,
    attempt: int,
    superseded_by: int,
    reason: str,
) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    value = load_or_create(target)
    attempts = list(value["attempts"])
    if not str(reason).startswith("mistral_"):
        raise ValueError("mistral_live_gate_supersede_reason_invalid")
    selected = next(
        (item for item in attempts if int(item.get("attempt") or 0) == int(attempt)),
        None,
    )
    replacement = next(
        (
            item
            for item in attempts
            if int(item.get("attempt") or 0) == int(superseded_by)
        ),
        None,
    )
    if (
        selected is None
        or replacement is None
        or int(superseded_by) <= int(attempt)
        or selected.get("outcome") == "running"
        or replacement.get("outcome") != "achieved"
    ):
        raise RuntimeError("mistral_live_gate_supersede_attempt_invalid")
    selected["strict_status"] = "superseded"
    selected["superseded_by_attempt"] = int(superseded_by)
    selected["superseded_reason"] = str(reason)
    replacement["strict_status"] = "current"
    value["attempts"] = attempts
    _write(target, value)
    errors = validate_ledger(value)
    if errors:
        raise RuntimeError(errors[0])
    return value
