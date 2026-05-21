"""Explicit outer optimizer contract for CrowdTensorD dense DiLoCo updates."""

from __future__ import annotations

import math
from typing import Iterable


CONTRACT_VERSION = "outer_optimizer_contract_v1"
OPTIMIZER_DILOCO_MOMENTUM = "diloco_momentum"
DELTA_FORMAT_DENSE_FLOAT = "dense_float"
DELTA_FORMAT_SIGN_COMPRESSED = "sign_compressed"
SIGN_ENCODING_TERNARY_V1 = "ternary_signs_v1"


def default_outer_optimizer_contract(model: dict | None = None) -> dict:
    source = dict(model or {})
    weights = [float(value) for value in source.get("weights", [])]
    return {
        "contract_version": CONTRACT_VERSION,
        "optimizer_type": OPTIMIZER_DILOCO_MOMENTUM,
        "delta_format": DELTA_FORMAT_DENSE_FLOAT,
        "outer_lr": float(source.get("outer_lr", 0.5)),
        "outer_momentum": float(source.get("outer_momentum", 0.9)),
        "optimizer_step": int(source.get("optimizer_step", source.get("global_step", 0))),
        "weight_count": len(weights),
    }


def normalize_outer_optimizer_contract(model: dict | None) -> dict:
    source = dict(model or {})
    raw_contract = dict(source.get("outer_optimizer_contract") or {})
    weights = [float(value) for value in source.get("weights", [])]
    return {
        "contract_version": raw_contract.get("contract_version", CONTRACT_VERSION),
        "optimizer_type": raw_contract.get(
            "optimizer_type",
            source.get("outer_optimizer_type", OPTIMIZER_DILOCO_MOMENTUM),
        ),
        "delta_format": raw_contract.get("delta_format", DELTA_FORMAT_DENSE_FLOAT),
        "outer_lr": float(source.get("outer_lr", raw_contract.get("outer_lr", 0.5))),
        "outer_momentum": float(source.get("outer_momentum", raw_contract.get("outer_momentum", 0.9))),
        "optimizer_step": int(source.get(
            "optimizer_step",
            raw_contract.get("optimizer_step", source.get("global_step", 0)),
        )),
        "weight_count": len(weights) if weights else int(raw_contract.get("weight_count", 0)),
    }


def optimizer_claim_spec(model: dict) -> dict:
    current = normalize_outer_optimizer_contract(model)
    return {
        **current,
        "optimizer_step": int(model.get("optimizer_step", current["optimizer_step"])),
        "outer_lr": float(model.get("outer_lr", current["outer_lr"])),
        "outer_momentum": float(model.get("outer_momentum", current["outer_momentum"])),
        "weight_count": len(model.get("weights", [])),
    }


def l2_norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def compress_sign_delta(local_delta: Iterable[float]) -> dict:
    values = [float(value) for value in local_delta]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("local_delta contains NaN or infinite values")
    if not values:
        scale = 0.0
    else:
        scale = sum(abs(value) for value in values) / len(values)
    signs = [
        1 if value > 0 else -1 if value < 0 else 0
        for value in values
    ]
    return {
        "format": DELTA_FORMAT_SIGN_COMPRESSED,
        "encoding": SIGN_ENCODING_TERNARY_V1,
        "scale": scale,
        "signs": signs,
    }


def decode_delta_payload(
    *,
    local_delta: Iterable[float] | None = None,
    pseudo_gradient: Iterable[float] | None = None,
    compressed_delta: dict | None = None,
) -> tuple[list[float] | None, dict]:
    if local_delta is not None:
        values = [float(value) for value in local_delta]
        return values, {
            "delta_format": DELTA_FORMAT_DENSE_FLOAT,
            "decoded_delta_norm": l2_norm(values),
            "compression_ratio_estimate": 1.0,
        }
    if pseudo_gradient is not None:
        values = [float(value) for value in pseudo_gradient]
        return values, {
            "delta_format": DELTA_FORMAT_DENSE_FLOAT,
            "decoded_delta_norm": l2_norm(values),
            "compression_ratio_estimate": 1.0,
        }
    if compressed_delta is None:
        return None, {"delta_format": DELTA_FORMAT_DENSE_FLOAT}
    if not isinstance(compressed_delta, dict):
        raise ValueError("compressed_delta must be an object")
    if compressed_delta.get("format") != DELTA_FORMAT_SIGN_COMPRESSED:
        raise ValueError("compressed_delta format must be sign_compressed")
    if compressed_delta.get("encoding") != SIGN_ENCODING_TERNARY_V1:
        raise ValueError("compressed_delta encoding must be ternary_signs_v1")
    try:
        scale = float(compressed_delta.get("scale"))
    except (TypeError, ValueError) as exc:
        raise ValueError("compressed_delta scale must be numeric") from exc
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("compressed_delta scale must be finite and non-negative")
    signs = compressed_delta.get("signs")
    if not isinstance(signs, list):
        raise ValueError("compressed_delta signs must be a list")
    values: list[float] = []
    for sign in signs:
        if not isinstance(sign, int) or sign not in {-1, 0, 1}:
            raise ValueError("compressed_delta signs must contain only -1, 0, or 1")
        values.append(float(sign) * scale)
    dense_bytes = max(1, len(values) * 8)
    compressed_bytes = max(1, len(signs) + 8)
    return values, {
        "delta_format": DELTA_FORMAT_SIGN_COMPRESSED,
        "decoded_delta_norm": l2_norm(values),
        "compression_ratio_estimate": round(dense_bytes / compressed_bytes, 6),
        "encoding": SIGN_ENCODING_TERNARY_V1,
    }


def apply_outer_optimizer_update(
    model: dict,
    local_delta: Iterable[float],
    *,
    delta_metadata: dict | None = None,
) -> tuple[dict, dict]:
    current = dict(model)
    contract = normalize_outer_optimizer_contract(current)
    if contract["contract_version"] != CONTRACT_VERSION:
        raise ValueError(f"unsupported outer optimizer contract {contract['contract_version']}")
    if contract["optimizer_type"] != OPTIMIZER_DILOCO_MOMENTUM:
        raise ValueError(f"unsupported outer optimizer type {contract['optimizer_type']}")
    if contract["delta_format"] != DELTA_FORMAT_DENSE_FLOAT:
        raise ValueError(f"unsupported delta format {contract['delta_format']}")

    delta = [float(value) for value in local_delta]
    weights = [float(value) for value in current.get("weights", [])]
    velocity = [float(value) for value in current.get("outer_velocity", [])]
    if len(velocity) != len(weights):
        velocity = [0.0 for _ in weights]
    if len(delta) != len(weights):
        raise ValueError(f"local delta length {len(delta)} does not match weights length {len(weights)}")

    momentum = float(current.get("outer_momentum", contract["outer_momentum"]))
    outer_lr = float(current.get("outer_lr", contract["outer_lr"]))
    next_velocity = [
        momentum * old_velocity + update
        for old_velocity, update in zip(velocity, delta)
    ]
    next_weights = [
        weight + outer_lr * update_velocity
        for weight, update_velocity in zip(weights, next_velocity)
    ]
    step_before = int(current.get("optimizer_step", contract["optimizer_step"]))
    step_after = step_before + 1
    next_contract = {
        **contract,
        "outer_lr": outer_lr,
        "outer_momentum": momentum,
        "optimizer_step": step_after,
        "weight_count": len(next_weights),
    }
    summary = optimizer_result_summary(
        claim_spec=optimizer_claim_spec(current),
        result_contract=next_contract,
        local_delta=delta,
        next_velocity=next_velocity,
        delta_format=(delta_metadata or {}).get("delta_format"),
        decoded_delta_norm=(delta_metadata or {}).get("decoded_delta_norm"),
        compression_ratio_estimate=(delta_metadata or {}).get("compression_ratio_estimate"),
    )
    next_model = {
        **current,
        "outer_optimizer_contract": next_contract,
        "outer_optimizer_type": next_contract["optimizer_type"],
        "outer_lr": outer_lr,
        "outer_momentum": momentum,
        "outer_velocity": next_velocity,
        "optimizer_step": step_after,
        "weights": next_weights,
    }
    return next_model, summary


def optimizer_result_summary(
    *,
    claim_spec: dict,
    result_contract: dict,
    local_delta: Iterable[float],
    next_velocity: Iterable[float],
    delta_format: str | None = None,
    decoded_delta_norm: float | None = None,
    compression_ratio_estimate: float | None = None,
) -> dict:
    summary = {
        "contract_version": result_contract.get("contract_version", CONTRACT_VERSION),
        "optimizer_type": result_contract.get("optimizer_type", OPTIMIZER_DILOCO_MOMENTUM),
        "delta_format": delta_format or result_contract.get("delta_format", DELTA_FORMAT_DENSE_FLOAT),
        "optimizer_step_before": int(claim_spec.get("optimizer_step", 0)),
        "optimizer_step_after": int(result_contract.get("optimizer_step", 0)),
        "outer_lr": float(result_contract.get("outer_lr", 0.0)),
        "outer_momentum": float(result_contract.get("outer_momentum", 0.0)),
        "weight_count": int(result_contract.get("weight_count", 0)),
        "delta_norm": l2_norm(local_delta),
        "velocity_norm": l2_norm(next_velocity),
    }
    if decoded_delta_norm is not None:
        summary["decoded_delta_norm"] = float(decoded_delta_norm)
    if compression_ratio_estimate is not None:
        summary["compression_ratio_estimate"] = float(compression_ratio_estimate)
    return summary
