"""Named-tensor DiLoCo aggregation and error-feedback transport."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

from .training_contract import (
    OUTER_OPTIMIZER_SCHEMA,
    sha256_file,
    sha256_json,
    tensor_specs,
)


DENSE_TRANSPORT = "named_tensor_dense_v1"
SIGN_EF_TRANSPORT = "named_tensor_sign_error_feedback_v1"
AGGREGATION_SCHEMA = "crowdtensor_named_tensor_diloco_round_v1"


def _torch() -> Any:
    import torch

    return torch


def load_tensors(path: str | Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    return dict(load_file(str(path), device="cpu"))


def save_tensors(tensors: dict[str, Any], path: str | Path) -> Path:
    from safetensors.torch import save_file

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in sorted(tensors.items())},
        str(output),
    )
    return output


def named_tensor_hash(tensors: dict[str, Any]) -> str:
    return sha256_json(tensor_specs(tensors))


def adapter_delta(base: dict[str, Any], local: dict[str, Any]) -> dict[str, Any]:
    if set(base) != set(local):
        raise ValueError("adapter tensor names do not match")
    delta: dict[str, Any] = {}
    for name in sorted(base):
        if tuple(base[name].shape) != tuple(local[name].shape):
            raise ValueError(f"adapter tensor shape mismatch for {name}")
        delta[name] = local[name].detach().cpu() - base[name].detach().cpu()
    return delta


def tensor_l2_norm(tensors: dict[str, Any]) -> float:
    return math.sqrt(sum(float(value.float().norm().item()) ** 2 for value in tensors.values()))


def compress_sign_with_error_feedback(
    delta: dict[str, Any],
    *,
    transport_path: str | Path,
    residual_path: str | Path,
    previous_residual_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compress named tensors to ternary signs and retain local residuals."""

    torch = _torch()
    residuals: dict[str, Any] = {}
    if previous_residual_path and Path(previous_residual_path).is_file():
        residuals = load_tensors(previous_residual_path)
    encoded: dict[str, Any] = {}
    next_residuals: dict[str, Any] = {}
    entries: list[dict[str, Any]] = []
    dense_bytes = 0
    encoded_bytes = 0
    for index, (name, value) in enumerate(sorted(delta.items())):
        dense = value.detach().cpu().float().contiguous()
        residual = residuals.get(name)
        if residual is None or tuple(residual.shape) != tuple(dense.shape):
            residual = torch.zeros_like(dense)
        corrected = dense + residual.float()
        scale = float(corrected.abs().mean().item()) if corrected.numel() else 0.0
        signs = torch.sign(corrected).to(torch.int8)
        decoded = signs.float() * scale
        next_residuals[name] = corrected - decoded
        sign_key = f"sign_{index:04d}"
        scale_key = f"scale_{index:04d}"
        encoded[sign_key] = signs
        encoded[scale_key] = torch.tensor([scale], dtype=torch.float32)
        dense_bytes += int(dense.numel() * dense.element_size())
        encoded_bytes += int(signs.numel() * signs.element_size() + 4)
        entries.append(
            {
                "name": name,
                "shape": list(dense.shape),
                "sign_key": sign_key,
                "scale_key": scale_key,
                "dtype": str(value.dtype).replace("torch.", ""),
            }
        )
    transport = save_tensors(encoded, transport_path)
    residual = save_tensors(next_residuals, residual_path)
    return {
        "schema": SIGN_EF_TRANSPORT,
        "transport_path": str(transport.resolve()),
        "transport_hash": sha256_file(transport),
        "residual_path": str(residual.resolve()),
        "residual_hash": sha256_file(residual),
        "entries": entries,
        "tensor_count": len(entries),
        "dense_byte_count": dense_bytes,
        "transport_byte_count": encoded_bytes,
        "compression_ratio": float(dense_bytes / max(1, encoded_bytes)),
        "input_norm": tensor_l2_norm(delta),
        "residual_norm": tensor_l2_norm(next_residuals),
        "error_feedback": True,
        "public_artifact_safe": False,
    }


def decode_sign_transport(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema") != SIGN_EF_TRANSPORT:
        raise ValueError("unsupported named tensor transport schema")
    path = Path(str(manifest.get("transport_path") or ""))
    if not path.is_file() or sha256_file(path) != manifest.get("transport_hash"):
        raise ValueError("named tensor transport hash mismatch")
    encoded = load_tensors(path)
    decoded: dict[str, Any] = {}
    for entry in manifest.get("entries") or []:
        name = str(entry["name"])
        signs = encoded[str(entry["sign_key"])]
        scale = float(encoded[str(entry["scale_key"])].item())
        value = signs.float() * scale
        if list(value.shape) != list(entry.get("shape") or []):
            raise ValueError(f"decoded transport shape mismatch for {name}")
        decoded[name] = value
    return decoded


def average_named_deltas(deltas: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(deltas)
    if not values:
        raise ValueError("at least one adapter delta is required")
    names = set(values[0])
    if not names or any(set(item) != names for item in values[1:]):
        raise ValueError("adapter delta tensor names do not match")
    return {
        name: sum((item[name].float() for item in values), start=_torch().zeros_like(values[0][name].float()))
        / float(len(values))
        for name in sorted(names)
    }


def apply_diloco_outer_step(
    *,
    base_adapter_path: str | Path,
    delta_paths: Iterable[str | Path],
    output_adapter_path: str | Path,
    velocity_path: str | Path,
    outer_step: int,
    adapter_version: int,
    outer_lr: float = 1.0,
    momentum: float = 0.0,
) -> dict[str, Any]:
    base = load_tensors(base_adapter_path)
    deltas = [load_tensors(path) for path in delta_paths]
    average = average_named_deltas(deltas)
    previous_velocity = load_tensors(velocity_path) if Path(velocity_path).is_file() else {}
    next_velocity: dict[str, Any] = {}
    updated: dict[str, Any] = {}
    for name in sorted(base):
        if name not in average:
            raise ValueError(f"aggregated adapter delta is missing {name}")
        velocity = previous_velocity.get(name)
        if velocity is None or tuple(velocity.shape) != tuple(average[name].shape):
            velocity = _torch().zeros_like(average[name])
        next_velocity[name] = float(momentum) * velocity.float() + average[name].float()
        updated[name] = (base[name].float() + float(outer_lr) * next_velocity[name]).to(base[name].dtype)
    output = save_tensors(updated, output_adapter_path)
    velocity_output = save_tensors(next_velocity, velocity_path)
    specs = tensor_specs(updated)
    return {
        "schema": AGGREGATION_SCHEMA,
        "optimizer_contract": {
            "schema": OUTER_OPTIMIZER_SCHEMA,
            "optimizer_type": "diloco_momentum",
            "outer_lr": float(outer_lr),
            "momentum": float(momentum),
        },
        "input_delta_count": len(deltas),
        "outer_step_before": int(outer_step),
        "outer_step_after": int(outer_step) + 1,
        "adapter_version_before": int(adapter_version),
        "adapter_version_after": int(adapter_version) + 1,
        "average_delta_norm": tensor_l2_norm(average),
        "velocity_norm": tensor_l2_norm(next_velocity),
        "global_adapter_norm": tensor_l2_norm(updated),
        "global_adapter_path": str(output.resolve()),
        "global_adapter_file_hash": sha256_file(output),
        "global_adapter_tensor_hash": sha256_json(specs),
        "global_adapter_tensor_specs": specs,
        "velocity_path": str(velocity_output.resolve()),
        "velocity_file_hash": sha256_file(velocity_output),
        "base_adapter_updated": True,
        "public_artifact_safe": False,
    }


def export_standard_peft_adapter(
    *,
    adapter_tensor_path: str | Path,
    adapter_config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "adapter_model.safetensors"
    config_path = output / "adapter_config.json"
    shutil.copyfile(adapter_tensor_path, model_path)
    shutil.copyfile(adapter_config_path, config_path)
    return {
        "adapter_dir": str(output.resolve()),
        "adapter_model_hash": sha256_file(model_path),
        "adapter_config_hash": sha256_file(config_path),
        "standard_peft_layout": True,
    }
