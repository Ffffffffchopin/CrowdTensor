import pytest

from scripts.mistral_kaggle_live_ledger import (
    MAXIMUM_ATTEMPTS,
    SCHEMA,
    finish_attempt,
    load_or_create,
    reserve_attempt,
    supersede_attempt,
    validate_ledger,
)


def test_mistral_live_ledger_is_independent_and_bounded(tmp_path) -> None:
    path = tmp_path / "mistral-ledger.json"
    created = load_or_create(path)
    assert created["schema"] == SCHEMA
    assert created["maximum_attempts"] == MAXIMUM_ATTEMPTS == 2
    assert created["community_maturity_ledger_modified"] is False
    assert validate_ledger(created) == []
    assert reserve_attempt(path) == 1
    first = finish_attempt(path, achieved=False)
    assert first["attempts"][0]["outcome"] == "failed"
    assert reserve_attempt(path) == 2
    second = finish_attempt(path, achieved=True)
    assert second["attempts"][1]["outcome"] == "achieved"
    superseded = supersede_attempt(
        path,
        attempt=1,
        superseded_by=2,
        reason="mistral_live_replacement_step_boundary_invalid",
    )
    assert superseded["attempts"][0]["strict_status"] == "superseded"
    assert superseded["attempts"][0]["superseded_by_attempt"] == 2
    assert superseded["attempts"][1]["strict_status"] == "current"
    with pytest.raises(RuntimeError, match="attempt_limit_reached"):
        reserve_attempt(path)


def test_mistral_live_ledger_rejects_parallel_running_attempt(tmp_path) -> None:
    path = tmp_path / "mistral-ledger.json"
    reserve_attempt(path)
    with pytest.raises(RuntimeError, match="already_running"):
        reserve_attempt(path)
