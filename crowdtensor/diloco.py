"""CPU-only DiLoCo-style mock math for CrowdTensorD.

This module is intentionally dependency-free. It models the training control
flow we need for V2: local inner optimization on a miner, followed by an outer
Coordinator update with optimizer state.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Iterable

from .micro_transformer import default_micro_transformer_model, normalize_micro_transformer_model


SCHEMA_VERSION = "diloco_mock_v1"
DEFAULT_WEIGHTS = [0.0, 0.0, 0.0]
TARGET_WEIGHTS = [1.0, -2.0, 0.5]
DEFAULT_INNER_LR = 0.03
DEFAULT_LOCAL_DELTA_SCALE = 0.1
LORA_MOCK_SCHEMA_VERSION = "lora_mock_v1"
DEFAULT_LORA_RANK = 1

FEATURES = [
    [1.0, 0.0, 0.5],
    [0.5, -1.0, 1.0],
    [-1.0, 0.25, 0.75],
    [0.25, 1.0, -0.5],
    [1.5, -0.5, 0.25],
    [-0.75, -1.25, 1.0],
]
TARGETS = [
    sum(weight * feature for weight, feature in zip(TARGET_WEIGHTS, features))
    for features in FEATURES
]


def default_model() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "version": 0,
        "global_step": 0,
        "weights": list(DEFAULT_WEIGHTS),
        "outer_lr": 0.5,
        "outer_momentum": 0.9,
        "outer_velocity": [0.0 for _ in DEFAULT_WEIGHTS],
        "optimizer_step": 0,
        "adapter_step": 0,
        "adapter_lr": 1.0,
        "lora_adapter": default_lora_adapter(len(DEFAULT_WEIGHTS)),
        "micro_transformer": default_micro_transformer_model(),
    }


def default_lora_adapter(width: int | None = None) -> dict:
    size = len(DEFAULT_WEIGHTS) if width is None else int(width)
    return {
        "schema_version": LORA_MOCK_SCHEMA_VERSION,
        "rank": DEFAULT_LORA_RANK,
        "values": [0.0 for _ in range(size)],
    }


def normalize_lora_adapter(adapter: dict | None, *, width: int | None = None) -> dict:
    size = len(DEFAULT_WEIGHTS) if width is None else int(width)
    source = dict(adapter or {})
    values = [float(value) for value in source.get("values", [])]
    if len(values) != size:
        values = [0.0 for _ in range(size)]
    return {
        "schema_version": source.get("schema_version", LORA_MOCK_SCHEMA_VERSION),
        "rank": int(source.get("rank", DEFAULT_LORA_RANK)),
        "values": values,
    }


def normalize_model(model: dict | None) -> dict:
    source = dict(model or {})
    weights = [float(value) for value in source.get("weights", DEFAULT_WEIGHTS)]
    velocity = [float(value) for value in source.get("outer_velocity", [])]
    if len(velocity) != len(weights):
        velocity = [0.0 for _ in weights]
    adapter = normalize_lora_adapter(source.get("lora_adapter"), width=len(weights))

    return {
        "schema_version": source.get("schema_version", SCHEMA_VERSION),
        "version": int(source.get("version", 0)),
        "global_step": int(source.get("global_step", 0)),
        "weights": weights,
        "outer_lr": float(source.get("outer_lr", 0.5)),
        "outer_momentum": float(source.get("outer_momentum", 0.9)),
        "outer_velocity": velocity,
        "optimizer_step": int(source.get("optimizer_step", source.get("global_step", 0))),
        "adapter_step": int(source.get("adapter_step", 0)),
        "adapter_lr": float(source.get("adapter_lr", 1.0)),
        "lora_adapter": adapter,
        "micro_transformer": normalize_micro_transformer_model(source.get("micro_transformer")),
    }


def stable_offset(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def training_spec_for(task_id: str, miner_id: str, model_version: int) -> dict:
    """Return the runtime math contract for a claimed task."""

    return {
        "schema_version": SCHEMA_VERSION,
        "features": [list(row) for row in FEATURES],
        "targets": list(TARGETS),
        "inner_lr": DEFAULT_INNER_LR,
        "local_delta_scale": DEFAULT_LOCAL_DELTA_SCALE,
        "sample_offset": stable_offset(task_id, miner_id, model_version) % len(FEATURES),
    }


def predict(weights: Iterable[float], features: Iterable[float]) -> float:
    return sum(float(weight) * float(feature) for weight, feature in zip(weights, features))


def _synthetic_loss_for(weights: Iterable[float], features_set: list[list[float]], targets: list[float]) -> float:
    values = [float(value) for value in weights]
    total = 0.0
    for features, target in zip(features_set, targets):
        error = predict(values, features) - target
        total += error * error
    return math.sqrt(total / len(features_set))


def synthetic_loss(weights_or_model: Iterable[float] | dict) -> float:
    if isinstance(weights_or_model, dict):
        weights = normalize_model(weights_or_model)["weights"]
    else:
        weights = [float(value) for value in weights_or_model]

    return _synthetic_loss_for(weights, FEATURES, TARGETS)


def _local_gradient_for(
    weights: list[float],
    sample_index: int,
    features_set: list[list[float]],
    targets: list[float],
) -> list[float]:
    features = features_set[sample_index % len(features_set)]
    target = targets[sample_index % len(targets)]
    error = predict(weights, features) - target
    return [2.0 * error * feature for feature in features]


def local_gradient(weights: list[float], sample_index: int) -> list[float]:
    return _local_gradient_for(weights, sample_index, FEATURES, TARGETS)


def run_inner_loop(
    weights: Iterable[float],
    *,
    task_id: str,
    miner_id: str,
    model_version: int = 0,
    inner_steps: int,
    inner_lr: float = DEFAULT_INNER_LR,
    compute_seconds: float = 0.0,
    training_spec: dict | None = None,
) -> dict:
    """Run deterministic local SGD and return a DiLoCo-style local delta."""

    initial = [float(value) for value in weights]
    local = list(initial)
    steps = max(1, int(inner_steps))
    features_set = [list(row) for row in (training_spec or {}).get("features", FEATURES)]
    targets = [float(value) for value in (training_spec or {}).get("targets", TARGETS)]
    if not features_set or not targets:
        raise ValueError("training_spec must provide non-empty features and targets")
    if training_spec:
        inner_lr = float(training_spec.get("inner_lr", inner_lr))
        local_delta_scale = float(training_spec.get("local_delta_scale", 1.0))
        sample_offset = int(training_spec.get(
            "sample_offset",
            stable_offset(task_id, miner_id, model_version) % len(features_set),
        ))
    else:
        local_delta_scale = 1.0
        sample_offset = stable_offset(task_id, miner_id, model_version) % len(features_set)

    start = time.monotonic()
    deadline = start + max(0.0, compute_seconds)
    step = 0
    loss_start = _synthetic_loss_for(local, features_set, targets)

    while step < steps or (compute_seconds > 0 and time.monotonic() < deadline):
        gradient = _local_gradient_for(local, sample_offset + step, features_set, targets)
        for index, value in enumerate(gradient):
            local[index] -= inner_lr * value
        step += 1

        if compute_seconds > 0 and step % 200 == 0:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(0.05, remaining))

    local_delta = [
        (local_value - initial_value) * local_delta_scale
        for local_value, initial_value in zip(local, initial)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "local_delta": local_delta,
        "inner_loss_start": loss_start,
        "inner_loss_end": _synthetic_loss_for(local, features_set, targets),
        "samples_seen": step,
        "inner_steps": steps,
        "inner_lr": inner_lr,
        "local_delta_scale": local_delta_scale,
        "sample_offset": sample_offset,
    }


def apply_outer_update(model: dict, local_delta: Iterable[float]) -> dict:
    current = normalize_model(model)
    delta = [float(value) for value in local_delta]
    weights = current["weights"]
    if len(delta) != len(weights):
        raise ValueError(f"local delta length {len(delta)} does not match weights length {len(weights)}")

    momentum = current["outer_momentum"]
    next_velocity = [
        momentum * velocity + update
        for velocity, update in zip(current["outer_velocity"], delta)
    ]
    next_weights = [
        weight + current["outer_lr"] * velocity
        for weight, velocity in zip(weights, next_velocity)
    ]
    return {
        **current,
        "schema_version": SCHEMA_VERSION,
        "version": current["version"] + 1,
        "global_step": current["global_step"] + 1,
        "weights": next_weights,
        "outer_velocity": next_velocity,
        "optimizer_step": current["optimizer_step"] + 1,
    }


def loss(model: dict) -> float:
    return synthetic_loss(model)
