"""Result validation for CrowdTensorD worker submissions."""

from __future__ import annotations

import math
from typing import Iterable

from .diloco import apply_outer_update, loss, normalize_model


DEFAULT_MAX_DELTA_NORM = 5.0
DEFAULT_MAX_LOSS_DELTA = 1.0


def _base_result(*, accepted: bool, reason: str, code: str, values: list[float] | None = None) -> dict:
    return {
        "accepted": accepted,
        "code": code,
        "reason": reason,
        "local_delta": values,
        "delta_norm": None,
        "loss_before": None,
        "loss_after": None,
        "loss_delta": None,
    }


def validate_local_delta(
    model: dict,
    local_delta: Iterable[float],
    *,
    max_delta_norm: float = DEFAULT_MAX_DELTA_NORM,
    max_loss_delta: float = DEFAULT_MAX_LOSS_DELTA,
) -> dict:
    """Validate a miner result before it can update the global model."""

    current = normalize_model(model)
    try:
        values = [float(value) for value in local_delta]
    except (TypeError, ValueError):
        return _base_result(
            accepted=False,
            code="delta_not_numeric",
            reason="local_delta must be a numeric iterable",
        )

    if len(values) != len(current["weights"]):
        return _base_result(
            accepted=False,
            code="delta_length_mismatch",
            reason=f"local_delta length {len(values)} does not match weights length {len(current['weights'])}",
            values=values,
        )

    if not all(math.isfinite(value) for value in values):
        return _base_result(
            accepted=False,
            code="delta_non_finite",
            reason="local_delta contains NaN or infinite values",
            values=values,
        )

    delta_norm = math.sqrt(sum(value * value for value in values))
    if delta_norm > max_delta_norm:
        result = _base_result(
            accepted=False,
            code="delta_norm_too_large",
            reason=f"local_delta norm {delta_norm:.6f} exceeds max_delta_norm {max_delta_norm:.6f}",
            values=values,
        )
        result["delta_norm"] = delta_norm
        return result

    loss_before = loss(current)
    candidate = apply_outer_update(current, values)
    loss_after = loss(candidate)
    loss_delta = loss_after - loss_before
    result = {
        "accepted": loss_delta <= max_loss_delta,
        "code": "ok" if loss_delta <= max_loss_delta else "loss_spike",
        "reason": "accepted" if loss_delta <= max_loss_delta else (
            f"candidate loss increase {loss_delta:.6f} exceeds max_loss_delta {max_loss_delta:.6f}"
        ),
        "local_delta": values,
        "delta_norm": delta_norm,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "loss_delta": loss_delta,
        "max_delta_norm": max_delta_norm,
        "max_loss_delta": max_loss_delta,
    }
    return result
