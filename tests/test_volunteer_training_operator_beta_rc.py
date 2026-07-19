from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from crowdtensor.training_contract import sha256_json
from scripts.volunteer_training_operator_beta_check import check


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    ROOT
    / "dist"
    / "volunteer-training-operator-beta-rc-20260718-r2"
    / "volunteer_training_operator_beta_rc.json"
)


@pytest.mark.skipif(not CANONICAL.is_file(), reason="canonical Operator Beta RC absent")
def test_canonical_operator_beta_rc_passes_strict_checker() -> None:
    result = check(CANONICAL, require_ready=True)
    assert result["ok"] is True
    assert result["error_count"] == 0
    assert result["goal_achieved"] is True


@pytest.mark.skipif(not CANONICAL.is_file(), reason="canonical Operator Beta RC absent")
def test_checker_rejects_rehashed_tls_overclaim(tmp_path) -> None:
    copied = tmp_path / "rc"
    shutil.copytree(CANONICAL.parent, copied)
    report_path = copied / CANONICAL.name
    report = json.loads(report_path.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(report)
    mutated["deployment"]["tls_handshake_and_certificate_verification"] = False
    mutated["content_hash"] = sha256_json(
        {key: value for key, value in mutated.items() if key != "content_hash"}
    )
    report_path.write_text(
        json.dumps(mutated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = check(report_path, require_ready=True)
    assert result["ok"] is False
    assert "operator_beta_deployment_tls_handshake_and_certificate_verification_missing" in result[
        "errors"
    ]
    assert "operator_beta_false_ready_claim" in result["errors"]
