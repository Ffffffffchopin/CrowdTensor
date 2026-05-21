"""Deterministic replay audit helpers for CrowdTensorD worker results."""

from __future__ import annotations

from typing import Iterable

from .diloco import run_inner_loop
from .lora_mock import run_lora_inner_loop
from .micro_transformer import run_micro_transformer_inner_loop


AUDIT_MODE_NONE = "none"
AUDIT_MODE_REPLAY = "replay"
REPLAY_TOLERANCE = 1e-6


def max_abs_error(actual: Iterable[float], expected: Iterable[float]) -> float:
    pairs = list(zip([float(value) for value in actual], [float(value) for value in expected]))
    if not pairs:
        return 0.0
    return max(abs(left - right) for left, right in pairs)


def _audit_result(
    *,
    accepted: bool,
    code: str,
    reason: str,
    max_error: float | None = None,
    tolerance: float = REPLAY_TOLERANCE,
) -> dict:
    return {
        "audit_mode": AUDIT_MODE_REPLAY,
        "audit_accepted": accepted,
        "audit_code": code,
        "audit_reason": reason,
        "audit_max_abs_error": max_error,
        "audit_tolerance": tolerance,
    }


def verify_diloco_replay(task: dict, local_delta: Iterable[float], *, tolerance: float = REPLAY_TOLERANCE) -> dict:
    """Recompute a dense DiLoCo mock result from the persisted claim contract."""

    claim_weights = task.get("claim_weights")
    training_spec = task.get("claim_training_spec")
    if not isinstance(claim_weights, list) or not isinstance(training_spec, dict):
        return _audit_result(
            accepted=True,
            code="replay_contract_missing",
            reason="replay audit skipped because claim contract is missing",
            tolerance=tolerance,
        )

    expected = run_inner_loop(
        claim_weights,
        task_id=task["task_id"],
        miner_id=task.get("miner_id") or "anonymous",
        model_version=int(task.get("model_version", 0) or 0),
        inner_steps=int(task.get("inner_steps", 1) or 1),
        training_spec=training_spec,
    )["local_delta"]
    actual = [float(value) for value in local_delta]
    if len(actual) != len(expected):
        return _audit_result(
            accepted=False,
            code="local_delta_replay_mismatch",
            reason=f"local_delta length {len(actual)} does not match replay length {len(expected)}",
            max_error=None,
            tolerance=tolerance,
        )

    error = max_abs_error(actual, expected)
    accepted = error <= tolerance
    return _audit_result(
        accepted=accepted,
        code="ok" if accepted else "local_delta_replay_mismatch",
        reason="accepted" if accepted else (
            f"local_delta replay max_abs_error {error:.9f} exceeds tolerance {tolerance:.9f}"
        ),
        max_error=error,
        tolerance=tolerance,
    )


def verify_lora_replay(task: dict, adapter_delta: dict, *, tolerance: float = REPLAY_TOLERANCE) -> dict:
    """Recompute a CPU LoRA mock result from the persisted claim contract."""

    workload_spec = task.get("claim_workload_spec")
    if not isinstance(workload_spec, dict):
        return _audit_result(
            accepted=True,
            code="replay_contract_missing",
            reason="replay audit skipped because claim workload contract is missing",
            tolerance=tolerance,
        )

    expected_delta = run_lora_inner_loop(
        workload_spec,
        inner_steps=int(task.get("inner_steps", 1) or 1),
    )["adapter_delta"]
    actual_values = [float(value) for value in adapter_delta.get("values", [])]
    expected_values = [float(value) for value in expected_delta.get("values", [])]
    if len(actual_values) != len(expected_values):
        return _audit_result(
            accepted=False,
            code="adapter_delta_replay_mismatch",
            reason=f"adapter_delta length {len(actual_values)} does not match replay length {len(expected_values)}",
            max_error=None,
            tolerance=tolerance,
        )

    rank_matches = int(adapter_delta.get("rank", expected_delta.get("rank", 1))) == int(expected_delta.get("rank", 1))
    error = max_abs_error(actual_values, expected_values)
    accepted = rank_matches and error <= tolerance
    reason = "accepted"
    if not rank_matches:
        reason = "adapter_delta rank does not match replay rank"
    elif not accepted:
        reason = f"adapter_delta replay max_abs_error {error:.9f} exceeds tolerance {tolerance:.9f}"
    return _audit_result(
        accepted=accepted,
        code="ok" if accepted else "adapter_delta_replay_mismatch",
        reason=reason,
        max_error=error,
        tolerance=tolerance,
    )


def verify_micro_transformer_replay(
    task: dict,
    local_delta: Iterable[float],
    *,
    tolerance: float = REPLAY_TOLERANCE,
) -> dict:
    """Recompute a micro Transformer LM result from the persisted claim contract."""

    workload_spec = task.get("claim_workload_spec")
    if not isinstance(workload_spec, dict):
        return _audit_result(
            accepted=True,
            code="replay_contract_missing",
            reason="replay audit skipped because claim workload contract is missing",
            tolerance=tolerance,
        )

    expected = run_micro_transformer_inner_loop(
        workload_spec,
        inner_steps=int(task.get("inner_steps", 1) or 1),
    )["local_delta"]
    actual = [float(value) for value in local_delta]
    if len(actual) != len(expected):
        return _audit_result(
            accepted=False,
            code="micro_transformer_delta_replay_mismatch",
            reason=f"micro_transformer local_delta length {len(actual)} does not match replay length {len(expected)}",
            max_error=None,
            tolerance=tolerance,
        )

    error = max_abs_error(actual, expected)
    accepted = error <= tolerance
    return _audit_result(
        accepted=accepted,
        code="ok" if accepted else "micro_transformer_delta_replay_mismatch",
        reason="accepted" if accepted else (
            f"micro_transformer local_delta replay max_abs_error {error:.9f} exceeds tolerance {tolerance:.9f}"
        ),
        max_error=error,
        tolerance=tolerance,
    )
