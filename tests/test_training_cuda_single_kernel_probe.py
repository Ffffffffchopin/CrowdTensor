from __future__ import annotations

import json
import re

import pytest

from scripts.training_cuda_kaggle_common import (
    delete_succeeded_or_absent,
    public_command,
    public_safety_errors,
)
from scripts.training_cuda_single_kernel_probe import OUTPUT_PATTERN, finish_attempt, reserve_attempt


def _amended_ledger() -> dict:
    return {
        "allocation_budget_amendment": {
            "schema": "crowdtensor_cuda_training_allocation_budget_amendment_v1",
            "authorized": True,
            "authorized_at": "2026-07-11T18:04:09Z",
            "authorization_hash": "sha256:" + "a" * 64,
            "authorization_text_public": False,
            "same_authorized_account_only": True,
            "original_single_kernel_attempt_limit": 2,
            "original_two_node_attempt_limit": 2,
            "additional_single_kernel_attempts": 1,
            "additional_two_node_attempts": 1,
            "revised_single_kernel_attempt_limit": 3,
            "revised_two_node_attempt_limit": 3,
            "allocation_timeout_seconds": 1800,
        }
    }


def test_single_kernel_attempt_ledger_enforces_two_attempt_limit(tmp_path) -> None:
    ledger = tmp_path / "attempts.json"
    assert reserve_attempt(ledger, limit=2) == 1
    finish_attempt(ledger, attempt=1, outcome="quota")
    assert reserve_attempt(ledger, limit=2) == 2
    with pytest.raises(RuntimeError, match="attempt_limit_reached"):
        reserve_attempt(ledger, limit=2)
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert len(payload["single_kernel_attempts"]) == 2


def test_single_kernel_third_attempt_requires_valid_amendment(tmp_path) -> None:
    ledger = tmp_path / "attempts.json"
    ledger.write_text(json.dumps(_amended_ledger()), encoding="utf-8")
    assert reserve_attempt(ledger, limit=3) == 1
    assert reserve_attempt(ledger, limit=3) == 2
    assert reserve_attempt(ledger, limit=3) == 3
    with pytest.raises(RuntimeError, match="attempt_limit_reached"):
        reserve_attempt(ledger, limit=3)


def test_single_kernel_rejects_unamended_third_attempt_limit(tmp_path) -> None:
    ledger = tmp_path / "attempts.json"
    ledger.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="limit_not_authorized"):
        reserve_attempt(ledger, limit=3)


def test_public_command_redacts_local_paths_and_private_inputs() -> None:
    command = public_command(
        ["kaggle", "kernels", "push", "-p", "/private/package", "--token-file", "/secret"]
    )
    assert "/private/package" not in command
    assert "/secret" not in command
    assert "<local-path>" in command
    assert "<private>" in command


def test_public_safety_allows_explicit_false_flags_but_rejects_payloads() -> None:
    assert not public_safety_errors(
        {
            "raw_training_text_public": False,
            "activation_values_public": False,
            "gradient_values_public": False,
        }
    )
    assert '"payload_b64":' in public_safety_errors({"payload_b64": "secret"})


def test_kernel_delete_treats_verified_absence_as_clean() -> None:
    assert delete_succeeded_or_absent({"ok": True, "output_tail": "deleted"}) is True
    assert delete_succeeded_or_absent({"ok": False, "output_tail": "Notebook not found"}) is True
    assert delete_succeeded_or_absent({"ok": False, "output_tail": "HTTP 429"}) is False


def test_single_kernel_output_pattern_collects_report_and_checkpoint_only() -> None:
    assert re.fullmatch(OUTPUT_PATTERN, "training_cuda_single_kernel_gate.json")
    assert re.fullmatch(OUTPUT_PATTERN, "training_cuda_single_kernel_checkpoint_bundle.zip")
    assert not re.fullmatch(OUTPUT_PATTERN, "private_dataset.jsonl")
