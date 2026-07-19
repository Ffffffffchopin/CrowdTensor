from __future__ import annotations

import json
import re

import pytest
import scripts.training_cuda_two_node_probe as probe

from scripts.training_cuda_two_node_probe import (
    CoordinatorRouteError,
    OUTPUT_PATTERN,
    TUNNEL_URL_PATTERN,
    _build_embedded_single_gate,
    _wait_authenticated_route,
    finish_attempt,
    reserve_attempt,
)


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


def test_two_node_attempt_budget_is_independent_and_capped_at_two(tmp_path) -> None:
    ledger = tmp_path / "attempts.json"
    ledger.write_text(
        json.dumps({"single_kernel_attempts": [{"attempt": 1}, {"attempt": 2}]}) + "\n",
        encoding="utf-8",
    )
    assert reserve_attempt(ledger, limit=2) == 1
    first = json.loads(ledger.read_text(encoding="utf-8"))["two_node_attempts"][0]
    assert first["allocation_started"] is True
    finish_attempt(ledger, attempt=1, outcome="runtime")
    assert reserve_attempt(ledger, limit=2) == 2
    with pytest.raises(RuntimeError, match="attempt_limit_reached"):
        reserve_attempt(ledger, limit=2)
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert len(payload["single_kernel_attempts"]) == 2
    assert len(payload["two_node_attempts"]) == 2


def test_two_node_third_attempt_requires_valid_amendment(tmp_path) -> None:
    ledger = tmp_path / "attempts.json"
    ledger.write_text(json.dumps(_amended_ledger()), encoding="utf-8")
    assert reserve_attempt(ledger, limit=3) == 1
    assert reserve_attempt(ledger, limit=3) == 2
    assert reserve_attempt(ledger, limit=3) == 3
    with pytest.raises(RuntimeError, match="attempt_limit_reached"):
        reserve_attempt(ledger, limit=3)


def test_cloudflare_tunnel_parser_extracts_only_https_quick_tunnel_url() -> None:
    text = "route ready https://small-field-123.trycloudflare.com and http://ignored.example"
    assert TUNNEL_URL_PATTERN.findall(text) == ["https://small-field-123.trycloudflare.com"]


class _RunningProcess:
    def poll(self):
        return None


def test_authenticated_route_requires_consecutive_stable_successes(monkeypatch) -> None:
    observations = iter(
        [
            {"ok": False, "error_class": "URLError:gaierror"},
            {"ok": True},
            {"ok": True},
        ]
    )
    monkeypatch.setattr(probe, "_probe_authenticated_route", lambda *args, **kwargs: next(observations))
    monkeypatch.setattr(probe.time, "sleep", lambda _seconds: None)
    result = _wait_authenticated_route(
        "https://private-route.invalid",
        coordinator_token="private-token",
        run_id="run-1",
        timeout=1.0,
        stable_successes=2,
        process=_RunningProcess(),
    )
    assert result["verified"] is True
    assert result["observation_count"] == 3
    assert result["stable_successes_observed"] == 2
    assert result["authenticated_status_verified"] is True
    encoded = json.dumps(result, sort_keys=True)
    assert "private-token" not in encoded
    assert "private-route" not in encoded


def test_authenticated_route_failure_is_public_safe(monkeypatch) -> None:
    monkeypatch.setattr(
        probe,
        "_probe_authenticated_route",
        lambda *args, **kwargs: {"ok": False, "error_class": "URLError:gaierror"},
    )
    monkeypatch.setattr(probe.time, "sleep", lambda _seconds: None)
    with pytest.raises(CoordinatorRouteError) as captured:
        _wait_authenticated_route(
            "https://private-route.invalid",
            coordinator_token="private-token",
            run_id="run-1",
            timeout=0.01,
            stable_successes=2,
            process=_RunningProcess(),
        )
    assert captured.value.code == "cloudflare_quick_tunnel_authenticated_readiness_timeout"
    encoded = json.dumps(captured.value.diagnostics, sort_keys=True)
    assert "private-token" not in encoded
    assert "private-route" not in encoded
    assert captured.value.diagnostics["public_artifact_safe"] is True


def test_two_node_output_pattern_collects_worker_and_role_checkpoint_bundles() -> None:
    assert re.fullmatch(OUTPUT_PATTERN, "training_cuda_two_node_worker.json")
    assert re.fullmatch(OUTPUT_PATTERN, "training_cuda_two_node_stage0_checkpoint_bundle.zip")
    assert re.fullmatch(OUTPUT_PATTERN, "training_cuda_two_node_stage1_checkpoint_bundle.zip")
    assert not re.fullmatch(OUTPUT_PATTERN, "private_dataset.jsonl")


def test_embedded_single_gate_binds_stage0_worker_bundle_and_kernel_hash() -> None:
    worker_report = {
        "single_kernel_t4x2_verified": True,
        "source_role": "stage0",
        "execution_order": "before_cross_node_stage0",
        "coallocated_with_two_node_attempt": True,
        "checkpoint_bundle": {"file_hash": "sha256:bundle"},
    }
    gate = _build_embedded_single_gate(
        stage0_worker={
            "role": "stage0",
            "embedded_single_kernel_gate_verified": True,
            "embedded_single_kernel_gate": worker_report,
        },
        stage0_bundle={
            "preserved": True,
            "worker_hash_match": True,
            "file_hash": "sha256:bundle",
            "byte_count": 1024,
            "file_count": 8,
            "contains_baseline_and_resumed_checkpoints": True,
        },
        stage0_kernel_ref_hash="sha256:kernel-stage0",
        attempt=3,
    )
    assert gate["single_kernel_t4x2_verified"] is True
    assert gate["source_binding_verified"] is True
    assert gate["source_two_node_attempt"] == 3
    assert gate["source_kernel_ref_hash"] == "sha256:kernel-stage0"
    assert gate["checkpoint_bundle"]["contains_baseline_and_resumed_checkpoints"] is True
    assert gate["ok"] is False
