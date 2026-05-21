"""Compatibility wrappers for the old V1 toy-compute API."""

from __future__ import annotations

from typing import Iterable

from .diloco import (
    DEFAULT_WEIGHTS,
    TARGET_WEIGHTS,
    apply_outer_update,
    default_model,
    loss,
    run_inner_loop,
)


def compute_pseudo_gradient(
    weights: Iterable[float],
    *,
    task_id: str,
    miner_id: str,
    inner_steps: int,
    compute_seconds: float = 0.0,
) -> list[float]:
    return run_inner_loop(
        weights,
        task_id=task_id,
        miner_id=miner_id,
        inner_steps=inner_steps,
        compute_seconds=compute_seconds,
    )["local_delta"]


def apply_pseudo_gradient(model: dict, pseudo_gradient: Iterable[float]) -> dict:
    return apply_outer_update(model, pseudo_gradient)
