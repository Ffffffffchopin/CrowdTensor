import json
from copy import deepcopy

import pytest

from scripts.community_kaggle_reliability_live_probe import _finish_gate, _reserve_gate
from scripts.community_live_gate_ledger_amend import (
    AMENDMENT_FIELDS,
    AMENDMENT_SCOPE,
    amend_ledger,
    validate_amended_ledger,
)


def original_ledger() -> dict:
    return {
        "schema": "crowdtensor_community_live_gate_ledger_v1",
        "maximum_full_live_gates": 2,
        "attempts": [
            {
                "attempt": 1,
                "started_at": 1.0,
                "completed_at": 2.0,
                "outcome": "failed:one",
            },
            {
                "attempt": 2,
                "started_at": 3.0,
                "completed_at": 4.0,
                "outcome": "failed:two",
            },
        ],
        "public_artifact_safe": True,
    }


def test_amendment_preserves_history_and_stores_only_public_audit_fields(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    original = original_ledger()
    path.write_text(json.dumps(original), encoding="utf-8")
    statement = "explicit one-time approval"
    result = amend_ledger(
        path,
        approval_statement=statement,
        amended_at="2026-07-17T00:00:00Z",
    )
    amended = json.loads(path.read_text(encoding="utf-8"))
    assert result["changed"] is True
    assert amended["attempts"] == original["attempts"]
    assert amended["maximum_full_live_gates"] == 3
    assert len(amended["amendments"]) == 1
    assert set(amended["amendments"][0]) == AMENDMENT_FIELDS
    assert amended["amendments"][0]["scope"] == AMENDMENT_SCOPE
    assert statement not in path.read_text(encoding="utf-8")
    assert validate_amended_ledger(amended, expected_attempt_count=2) == []


def test_amendment_is_idempotent_only_for_same_approval(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(original_ledger()), encoding="utf-8")
    amend_ledger(path, approval_statement="approved", amended_at="2026-07-17T00:00:00Z")
    before = path.read_bytes()
    assert amend_ledger(path, approval_statement="approved")["changed"] is False
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="existing_amendment_mismatch"):
        amend_ledger(path, approval_statement="different")


def test_third_reservation_is_only_additional_attempt_and_cannot_repeat(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(original_ledger()), encoding="utf-8")
    amend_ledger(path, approval_statement="approved", amended_at="2026-07-17T00:00:00Z")
    original_attempts = deepcopy(json.loads(path.read_text(encoding="utf-8"))["attempts"])
    reservation = _reserve_gate(path)
    assert reservation["attempt"]["attempt"] == 3
    running = json.loads(path.read_text(encoding="utf-8"))
    assert running["attempts"][:2] == original_attempts
    assert running["attempts"][2]["outcome"] == "running"
    with pytest.raises(RuntimeError, match="gate_limit_reached"):
        _reserve_gate(path)
    finished = _finish_gate(path, outcome="achieved")
    assert finished["attempts"][:2] == original_attempts
    assert finished["attempts"][2]["outcome"] == "achieved"


def test_amendment_rejects_incomplete_original_attempt(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    value = original_ledger()
    value["attempts"][1]["outcome"] = "running"
    value["attempts"][1]["completed_at"] = 0.0
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="original_attempt_incomplete"):
        amend_ledger(path, approval_statement="approved")
