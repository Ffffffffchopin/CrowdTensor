"""Versioned contracts and validation for real LoRA training results."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


WORKLOAD_TYPE = "hf_lora_train"
JOB_SCHEMA = "crowdtensor_training_job_v1"
DATASET_SCHEMA = "crowdtensor_training_dataset_v1"
MODEL_SCHEMA = "crowdtensor_training_model_v1"
LORA_SCHEMA = "crowdtensor_lora_config_v1"
TRAINING_SPEC_SCHEMA = "crowdtensor_hf_lora_training_spec_v1"
DELTA_SCHEMA = "crowdtensor_named_adapter_delta_v1"
RESULT_SCHEMA = "crowdtensor_hf_lora_training_result_v1"
VALIDATION_SCHEMA = "crowdtensor_adapter_delta_validation_v1"
OUTER_OPTIMIZER_SCHEMA = "crowdtensor_named_tensor_outer_optimizer_v1"
GPU_CONTINUATION_SCHEMA = "crowdtensor_gpu_training_continuation_v1"

PRIVATE_FIELDS = {
    "adapter_path",
    "base_model_path",
    "checkpoint_path",
    "dataset_path",
    "delta_path",
    "export_path",
    "job_manifest_path",
    "optimizer_path",
    "private_result_path",
    "raw_text",
    "resume_path",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def tensor_bytes(tensor: Any) -> bytes:
    """Return the exact contiguous storage bytes without dtype conversion."""

    torch = __import__("torch")
    detached = tensor.detach().cpu().contiguous()
    return detached.view(torch.uint8).numpy().tobytes()


def tensor_specs(tensors: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for name, tensor in sorted(tensors.items()):
        detached = tensor.detach().cpu().contiguous()
        raw = tensor_bytes(detached)
        specs.append(
            {
                "name": str(name),
                "shape": [int(item) for item in detached.shape],
                "dtype": str(detached.dtype).replace("torch.", ""),
                "numel": int(detached.numel()),
                "byte_count": len(raw),
                "tensor_hash": sha256_bytes(raw),
                "l2_norm": float(detached.float().norm().item()),
            }
        )
    return specs


def tensor_specs_hash(specs: Iterable[dict[str, Any]]) -> str:
    return sha256_json(list(specs))


def load_safetensors(path: str | Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    return dict(load_file(str(path), device="cpu"))


def delta_manifest(
    *,
    delta_path: str | Path,
    job_id: str,
    round_id: str,
    result_id: str,
    miner_id: str,
    model_manifest_hash: str,
    base_model_hash: str,
    base_adapter_hash: str,
    base_model_version: int,
    adapter_version: int,
    dataset_shard_index: int,
    dataset_shard_hash: str,
    loss_start: float,
    loss_end: float,
    samples_seen: int,
    tokens_seen: int,
) -> dict[str, Any]:
    path = Path(delta_path)
    tensors = load_safetensors(path)
    specs = tensor_specs(tensors)
    return {
        "schema": DELTA_SCHEMA,
        "job_id": str(job_id),
        "round_id": str(round_id),
        "result_id": str(result_id),
        "miner_id": str(miner_id),
        "model_manifest_hash": str(model_manifest_hash),
        "base_model_hash": str(base_model_hash),
        "base_adapter_hash": str(base_adapter_hash),
        "base_model_version": int(base_model_version),
        "adapter_version": int(adapter_version),
        "dataset_shard_index": int(dataset_shard_index),
        "dataset_shard_hash": str(dataset_shard_hash),
        "delta_path": str(path.resolve()),
        "delta_file_hash": sha256_file(path),
        "tensor_count": len(specs),
        "tensor_specs": specs,
        "tensor_specs_hash": tensor_specs_hash(specs),
        "delta_norm": math.sqrt(sum(float(item["l2_norm"]) ** 2 for item in specs)),
        "loss_start": float(loss_start),
        "loss_end": float(loss_end),
        "samples_seen": int(samples_seen),
        "tokens_seen": int(tokens_seen),
        "raw_dataset_public": False,
        "raw_text_public": False,
        "tensor_values_public": False,
        "public_artifact_safe": True,
    }


def public_delta_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key not in PRIVATE_FIELDS and not key.endswith("_path")
    }


def public_training_spec(spec: dict[str, Any]) -> dict[str, Any]:
    private = {
        "dataset_rows",
        "token_rows",
        *PRIVATE_FIELDS,
    }
    public: dict[str, Any] = {}
    for key, value in spec.items():
        if key in private or key.endswith("_path"):
            continue
        if isinstance(value, dict):
            public[key] = public_training_spec(value)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            public[key] = [public_training_spec(item) for item in value]
        else:
            public[key] = value
    public["private_paths_public"] = False
    public["raw_dataset_public"] = False
    public["public_artifact_safe"] = True
    return public


def _validation_error(code: str, reason: str) -> dict[str, Any]:
    return {
        "schema": VALIDATION_SCHEMA,
        "accepted": False,
        "code": code,
        "reason": reason,
        "public_artifact_safe": True,
    }


def validate_adapter_delta(
    manifest: dict[str, Any] | None,
    *,
    expected: dict[str, Any],
    seen_result_ids: Iterable[str] = (),
    max_delta_norm: float = 100.0,
    max_loss_increase: float = 0.25,
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        return _validation_error("adapter_delta_missing", "training result requires an adapter delta manifest")
    if manifest.get("schema") != DELTA_SCHEMA:
        return _validation_error("adapter_delta_schema_mismatch", "unsupported adapter delta schema")
    result_id = str(manifest.get("result_id") or "")
    if not result_id:
        return _validation_error("result_id_missing", "adapter delta result_id is required")
    if result_id in {str(value) for value in seen_result_ids}:
        return _validation_error("duplicate_result", "adapter delta result_id was already accepted")

    fields = {
        "job_id": str,
        "round_id": str,
        "model_manifest_hash": str,
        "base_model_hash": str,
        "base_adapter_hash": str,
        "base_model_version": int,
        "adapter_version": int,
        "dataset_shard_index": int,
        "dataset_shard_hash": str,
    }
    for field, caster in fields.items():
        if field not in expected:
            continue
        try:
            actual = caster(manifest.get(field))
            wanted = caster(expected[field])
        except (TypeError, ValueError):
            return _validation_error(f"{field}_invalid", f"adapter delta {field} is invalid")
        if actual != wanted:
            return _validation_error(f"{field}_mismatch", f"adapter delta {field} does not match claim")

    path = Path(str(manifest.get("delta_path") or ""))
    if not path.is_file():
        return _validation_error("adapter_delta_file_missing", "adapter delta safetensors file is missing")
    if sha256_file(path) != manifest.get("delta_file_hash"):
        return _validation_error("adapter_delta_file_hash_mismatch", "adapter delta file hash does not match manifest")

    try:
        tensors = load_safetensors(path)
    except Exception as exc:
        return _validation_error("adapter_delta_load_failed", f"adapter delta could not be loaded: {type(exc).__name__}")
    if not tensors:
        return _validation_error("adapter_delta_empty", "adapter delta contains no tensors")

    actual_specs = tensor_specs(tensors)
    if tensor_specs_hash(actual_specs) != manifest.get("tensor_specs_hash"):
        return _validation_error("adapter_delta_tensor_specs_mismatch", "adapter tensor metadata does not match file")
    expected_specs = expected.get("tensor_specs")
    if isinstance(expected_specs, list):
        expected_by_name = {str(item.get("name")): item for item in expected_specs if isinstance(item, dict)}
        actual_by_name = {str(item.get("name")): item for item in actual_specs}
        if set(expected_by_name) != set(actual_by_name):
            return _validation_error("adapter_delta_tensor_names_mismatch", "adapter tensor names do not match base adapter")
        for name, actual in actual_by_name.items():
            wanted = expected_by_name[name]
            if list(actual.get("shape") or []) != list(wanted.get("shape") or []):
                return _validation_error("adapter_delta_shape_mismatch", f"adapter tensor shape mismatch for {name}")
            if str(actual.get("dtype")) != str(wanted.get("dtype")):
                return _validation_error("adapter_delta_dtype_mismatch", f"adapter tensor dtype mismatch for {name}")

    torch = __import__("torch")
    for name, tensor in tensors.items():
        if not bool(torch.isfinite(tensor).all().item()):
            return _validation_error("adapter_delta_non_finite", f"adapter tensor {name} contains NaN or infinity")
    delta_norm = math.sqrt(sum(float(tensor.float().norm().item()) ** 2 for tensor in tensors.values()))
    if not math.isfinite(delta_norm) or delta_norm > float(max_delta_norm):
        return _validation_error("adapter_delta_norm_too_large", "adapter delta norm exceeds the configured limit")

    try:
        loss_start = float(manifest.get("loss_start"))
        loss_end = float(manifest.get("loss_end"))
    except (TypeError, ValueError):
        return _validation_error("training_loss_invalid", "training result loss values are invalid")
    if not math.isfinite(loss_start) or not math.isfinite(loss_end):
        return _validation_error("training_loss_non_finite", "training result loss values must be finite")
    if loss_end - loss_start > float(max_loss_increase):
        return _validation_error("training_loss_spike", "local training loss increased beyond the configured limit")

    return {
        "schema": VALIDATION_SCHEMA,
        "accepted": True,
        "code": "ok",
        "reason": "accepted",
        "result_id": result_id,
        "tensor_count": len(tensors),
        "delta_norm": delta_norm,
        "loss_start": loss_start,
        "loss_end": loss_end,
        "loss_delta": loss_end - loss_start,
        "public_delta_summary": public_delta_summary(manifest),
        "public_artifact_safe": True,
    }
