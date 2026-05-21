"""Dependency-free CPU LoRA-style adapter mock for CrowdTensorD."""

from __future__ import annotations

import math
import time
from typing import Iterable

from .diloco import (
    DEFAULT_INNER_LR,
    FEATURES,
    TARGETS,
    LORA_MOCK_SCHEMA_VERSION,
    normalize_lora_adapter,
    normalize_model,
    predict,
    stable_offset,
)


DEFAULT_ADAPTER_DELTA_SCALE = 0.1
DEFAULT_MAX_ADAPTER_DELTA_NORM = 5.0
DEFAULT_MAX_ADAPTER_LOSS_DELTA = 1.0


def adapter_weights(base_weights: Iterable[float], adapter: dict) -> list[float]:
    base = [float(value) for value in base_weights]
    normalized = normalize_lora_adapter(adapter, width=len(base))
    return [
        float(base) + float(delta)
        for base, delta in zip(base, normalized["values"])
    ]


def _adapter_loss_for(
    base_weights: Iterable[float],
    adapter_values: Iterable[float],
    features_set: list[list[float]],
    targets: list[float],
) -> float:
    weights = [
        float(base) + float(adapter_value)
        for base, adapter_value in zip(base_weights, adapter_values)
    ]
    total = 0.0
    for features, target in zip(features_set, targets):
        error = predict(weights, features) - target
        total += error * error
    return math.sqrt(total / len(features_set))


def adapter_loss(model: dict) -> float:
    current = normalize_model(model)
    return _adapter_loss_for(
        current["weights"],
        current["lora_adapter"]["values"],
        FEATURES,
        TARGETS,
    )


def lora_training_spec_for(task_id: str, miner_id: str, model: dict) -> dict:
    current = normalize_model(model)
    features = [list(row) for row in FEATURES]
    targets = list(TARGETS)
    return {
        "type": "cpu_lora_mock",
        "schema_version": LORA_MOCK_SCHEMA_VERSION,
        "rank": int(current["lora_adapter"]["rank"]),
        "base_weights": list(current["weights"]),
        "adapter": dict(current["lora_adapter"]),
        "adapter_step": int(current["adapter_step"]),
        "features": features,
        "targets": targets,
        "inner_lr": DEFAULT_INNER_LR,
        "adapter_delta_scale": DEFAULT_ADAPTER_DELTA_SCALE,
        "sample_offset": stable_offset(
            task_id,
            miner_id,
            current["version"],
            current["adapter_step"],
        ) % len(features),
    }


def _local_gradient_for(
    base_weights: list[float],
    adapter_values: list[float],
    sample_index: int,
    features_set: list[list[float]],
    targets: list[float],
) -> list[float]:
    features = features_set[sample_index % len(features_set)]
    target = targets[sample_index % len(targets)]
    weights = [
        base + adapter
        for base, adapter in zip(base_weights, adapter_values)
    ]
    error = predict(weights, features) - target
    return [2.0 * error * feature for feature in features]


def run_lora_inner_loop(
    workload_spec: dict,
    *,
    inner_steps: int,
    compute_seconds: float = 0.0,
) -> dict:
    base_weights = [float(value) for value in workload_spec["base_weights"]]
    adapter = normalize_lora_adapter(workload_spec.get("adapter"), width=len(base_weights))
    initial = list(adapter["values"])
    local = list(initial)
    features_set = [list(row) for row in workload_spec.get("features", FEATURES)]
    targets = [float(value) for value in workload_spec.get("targets", TARGETS)]
    if not features_set or not targets:
        raise ValueError("cpu_lora_mock workload_spec must provide features and targets")

    steps = max(1, int(inner_steps))
    inner_lr = float(workload_spec.get("inner_lr", DEFAULT_INNER_LR))
    delta_scale = float(workload_spec.get("adapter_delta_scale", DEFAULT_ADAPTER_DELTA_SCALE))
    sample_offset = int(workload_spec.get("sample_offset", 0))

    started_at = time.monotonic()
    deadline = started_at + max(0.0, compute_seconds)
    step = 0
    loss_start = _adapter_loss_for(base_weights, local, features_set, targets)

    while step < steps or (compute_seconds > 0 and time.monotonic() < deadline):
        gradient = _local_gradient_for(base_weights, local, sample_offset + step, features_set, targets)
        for index, value in enumerate(gradient):
            local[index] -= inner_lr * value
        step += 1

        if compute_seconds > 0 and step % 200 == 0:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(0.05, remaining))

    values = [
        (local_value - initial_value) * delta_scale
        for local_value, initial_value in zip(local, initial)
    ]
    return {
        "schema_version": LORA_MOCK_SCHEMA_VERSION,
        "adapter_delta": {
            "schema_version": LORA_MOCK_SCHEMA_VERSION,
            "rank": int(adapter["rank"]),
            "values": values,
        },
        "adapter_loss_start": loss_start,
        "adapter_loss_end": _adapter_loss_for(base_weights, local, features_set, targets),
        "samples_seen": step,
        "inner_steps": steps,
        "inner_lr": inner_lr,
        "adapter_delta_scale": delta_scale,
        "sample_offset": sample_offset,
    }


def _base_validation(
    *,
    accepted: bool,
    code: str,
    reason: str,
    adapter_delta: dict | None = None,
) -> dict:
    return {
        "accepted": accepted,
        "code": code,
        "reason": reason,
        "adapter_delta": adapter_delta,
        "delta_norm": None,
        "loss_before": None,
        "loss_after": None,
        "loss_delta": None,
    }


def validate_adapter_delta(
    model: dict,
    adapter_delta: dict | None,
    *,
    max_delta_norm: float = DEFAULT_MAX_ADAPTER_DELTA_NORM,
    max_loss_delta: float = DEFAULT_MAX_ADAPTER_LOSS_DELTA,
) -> dict:
    current = normalize_model(model)
    if not isinstance(adapter_delta, dict):
        return _base_validation(
            accepted=False,
            code="adapter_delta_missing",
            reason="cpu_lora_mock requires an adapter_delta object",
        )

    try:
        values = [float(value) for value in adapter_delta.get("values", [])]
    except (TypeError, ValueError):
        return _base_validation(
            accepted=False,
            code="adapter_delta_not_numeric",
            reason="adapter_delta values must be numeric",
        )

    if len(values) != len(current["weights"]):
        return _base_validation(
            accepted=False,
            code="adapter_delta_length_mismatch",
            reason=f"adapter_delta length {len(values)} does not match adapter length {len(current['weights'])}",
            adapter_delta={**adapter_delta, "values": values},
        )

    if not all(math.isfinite(value) for value in values):
        return _base_validation(
            accepted=False,
            code="adapter_delta_non_finite",
            reason="adapter_delta contains NaN or infinite values",
            adapter_delta={**adapter_delta, "values": values},
        )

    normalized_delta = {
        "schema_version": adapter_delta.get("schema_version", LORA_MOCK_SCHEMA_VERSION),
        "rank": int(adapter_delta.get("rank", current["lora_adapter"]["rank"])),
        "values": values,
    }
    delta_norm = math.sqrt(sum(value * value for value in values))
    if delta_norm > max_delta_norm:
        result = _base_validation(
            accepted=False,
            code="adapter_delta_norm_too_large",
            reason=f"adapter_delta norm {delta_norm:.6f} exceeds max_delta_norm {max_delta_norm:.6f}",
            adapter_delta=normalized_delta,
        )
        result["delta_norm"] = delta_norm
        return result

    loss_before = adapter_loss(current)
    candidate = apply_adapter_update(current, normalized_delta)
    loss_after = adapter_loss(candidate)
    loss_delta = loss_after - loss_before
    return {
        "accepted": loss_delta <= max_loss_delta,
        "code": "ok" if loss_delta <= max_loss_delta else "adapter_loss_spike",
        "reason": "accepted" if loss_delta <= max_loss_delta else (
            f"candidate adapter loss increase {loss_delta:.6f} exceeds max_loss_delta {max_loss_delta:.6f}"
        ),
        "adapter_delta": normalized_delta,
        "delta_norm": delta_norm,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "loss_delta": loss_delta,
        "max_delta_norm": max_delta_norm,
        "max_loss_delta": max_loss_delta,
    }


def apply_adapter_update(model: dict, adapter_delta: dict) -> dict:
    current = normalize_model(model)
    adapter = normalize_lora_adapter(current["lora_adapter"], width=len(current["weights"]))
    values = [float(value) for value in adapter_delta["values"]]
    lr = float(current.get("adapter_lr", 1.0))
    next_values = [
        current_value + lr * update
        for current_value, update in zip(adapter["values"], values)
    ]
    return {
        **current,
        "adapter_step": int(current["adapter_step"]) + 1,
        "lora_adapter": {
            **adapter,
            "rank": int(adapter_delta.get("rank", adapter["rank"])),
            "values": next_values,
        },
    }
