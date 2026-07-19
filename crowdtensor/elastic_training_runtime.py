"""Persistent control plane for elastic, stage-partitioned volunteer training.

The numerical training loop is intentionally kept outside this module.  A
stage may apply its optimizer update speculatively, but that update only
becomes a globally committed step after every stage submits a validated
checkpoint for the same barrier epoch.  Losing any assigned Miner aborts the
whole uncommitted epoch and fences its submissions.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import io
import json
import math
import os
import re
import secrets
import sqlite3
import stat
import struct
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator

from fastapi import Request as FastAPIRequest
from fastapi import Response as FastAPIResponse
from pydantic import BaseModel, Field

from .elastic_checkpoint_storage import (
    CheckpointBlobStore,
    LocalCheckpointBlobStore,
    checkpoint_blob_store_from_configuration,
)
from .heterogeneous_training_manifest import validate_training_manifest
from .heterogeneous_training_scheduler import (
    PlacementError,
    build_placement_plan,
    validate_miner_capability,
)
from .heterogeneous_tensor_transport import (
    ChunkedTensorStore,
    TensorTransportError,
    validate_tensor_envelope,
)
from .heterogeneous_training_checkpoint import validate_stage_checkpoint_archive
from .qwen15b_training import (
    MODEL_ID,
    MODEL_REVISION,
    QWEN_LORA_TARGET_MODULES,
    QWEN_STAGE_CHECKPOINT_SCHEMA,
    canonical_stage_specs,
    stable_hash,
)


RUNTIME_SCHEMA = "crowdtensor_elastic_training_runtime_v1"
STATUS_SCHEMA = "crowdtensor_elastic_training_status_v1"
ARCHIVE_SCHEMA = "crowdtensor_elastic_qwen_stage_checkpoint_archive_v1"
DEFAULT_MAX_CHECKPOINT_BYTES = 768 * 1024 * 1024
SESSION_STATES = {"online", "offline", "expired", "quarantined"}
EPOCH_STATES = {"active", "committed", "aborted"}
ADAPTER_NAME_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.(?:self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(gate_proj|up_proj|down_proj))\.lora_(A|B)(?:\.default)?\.weight$"
)
SAFETENSORS_DTYPES = {
    "BOOL": (1, None),
    "U8": (1, None),
    "I8": (1, None),
    "I16": (2, None),
    "I32": (4, None),
    "I64": (8, None),
    "F16": (2, "float16"),
    "BF16": (2, "bfloat16"),
    "F32": (4, "float32"),
    "F64": (8, "float64"),
}


class ElasticMinerRegistrationRequest(BaseModel):
    run_id: str = Field(min_length=1)
    miner_id_hash: str = Field(min_length=1)
    registration_nonce: str = Field(min_length=1)
    supported_stage_ids: list[int]
    slot_count: int = Field(ge=1)
    accelerator: str = "cuda"
    capability: dict[str, Any] | None = None


class ElasticStageRuntimeReportRequest(BaseModel):
    placement_generation: int = Field(ge=1)
    stage_id: int = Field(ge=0)
    device_id: str
    event_type: str = "profile"
    forward_latency_ms: float = Field(default=0.0, ge=0.0)
    backward_latency_ms: float = Field(default=0.0, ge=0.0)
    peak_memory_bytes: int = Field(default=0, ge=0)
    sample_count: int = Field(default=1, ge=1)
    compile_latency_ms: float = Field(default=0.0, ge=0.0)
    steady_forward_latency_ms: float = Field(default=0.0, ge=0.0)
    steady_backward_latency_ms: float = Field(default=0.0, ge=0.0)


class ElasticDeviceTelemetryRequest(BaseModel):
    device_id: str = Field(min_length=1)
    free_memory_bytes: int = Field(ge=0)
    utilization_fraction: float = Field(ge=0.0, le=1.0)
    throughput_units_per_second: float = Field(ge=0.0)
    network_bandwidth_bytes_per_second: float = Field(default=0.0, ge=0.0)
    network_latency_ms: float = Field(default=0.0, ge=0.0)
    checkpoint_step: int = Field(default=0, ge=0)
    health_score: float = Field(default=1.0, ge=0.0, le=1.0)


class ElasticDeviceFailureRequest(BaseModel):
    device_id: str = Field(min_length=1)
    failure_class: str = Field(min_length=1)
    quarantine_threshold: int = Field(default=3, ge=1, le=20)
    quarantine_seconds: float = Field(default=300.0, ge=1.0, le=86400.0)


class ElasticInlineTensorUploadRequest(BaseModel):
    envelope: dict[str, Any]
    chunk_b64: str = Field(min_length=1, max_length=6 * 1024 * 1024)


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _token_hash(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(float(item) for item in values if math.isfinite(float(item)))
    if not ordered:
        return 0.0
    position = max(0.0, min(1.0, float(fraction))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _atomic_write(path: Path, value: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(4)}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_file_names(stage_id: int) -> dict[str, str]:
    prefix = f"stage{int(stage_id)}"
    return {
        "manifest": f"{prefix}_checkpoint.json",
        "adapter": f"{prefix}_adapter.safetensors",
        "optimizer": f"{prefix}_optimizer.pt",
        "scaler": f"{prefix}_grad_scaler.pt",
        "rng": f"{prefix}_rng.pt",
    }


def checkpoint_signature_message(
    *,
    run_id: str,
    session_id: str,
    epoch_id: int,
    stage_id: int,
    assignment_token: str,
    archive_hash: str,
) -> bytes:
    """Return the canonical message signed by an assigned Miner."""

    value = {
        "archive_hash": str(archive_hash),
        "assignment_token_hash": _token_hash(str(assignment_token)),
        "epoch_id": int(epoch_id),
        "run_id": str(run_id),
        "session_id": str(session_id),
        "stage_id": int(stage_id),
    }
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def sign_checkpoint_submission(
    *,
    session_token: str,
    run_id: str,
    session_id: str,
    epoch_id: int,
    stage_id: int,
    assignment_token: str,
    archive_hash: str,
) -> str:
    message = checkpoint_signature_message(
        run_id=run_id,
        session_id=session_id,
        epoch_id=epoch_id,
        stage_id=stage_id,
        assignment_token=assignment_token,
        archive_hash=archive_hash,
    )
    return "hmac-sha256:" + hmac.new(
        str(session_token).encode("utf-8"), message, hashlib.sha256
    ).hexdigest()


def _safe_tensor_header(value: bytes) -> tuple[dict[str, Any], int]:
    if len(value) < 10:
        raise ValueError("elastic_checkpoint_adapter_safetensors_invalid")
    header_length = int.from_bytes(value[:8], "little", signed=False)
    if header_length < 2 or header_length > 16 * 1024 * 1024:
        raise ValueError("elastic_checkpoint_adapter_safetensors_header_invalid")
    data_start = 8 + header_length
    if data_start > len(value):
        raise ValueError("elastic_checkpoint_adapter_safetensors_header_invalid")
    try:
        header = json.loads(value[8:data_start].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("elastic_checkpoint_adapter_safetensors_header_invalid") from exc
    if not isinstance(header, dict):
        raise ValueError("elastic_checkpoint_adapter_safetensors_header_invalid")
    return header, data_start


def _tensor_is_finite(value: bytes, dtype_name: str) -> bool:
    torch_name = SAFETENSORS_DTYPES[dtype_name][1]
    if torch_name is None:
        return True
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "elastic_checkpoint_tensor_validation_requires_torch"
        ) from exc
    dtype = getattr(torch, str(torch_name))
    tensor = torch.frombuffer(bytearray(value), dtype=dtype)
    return bool(torch.isfinite(tensor).all())


def _validate_adapter_safetensors(
    value: bytes,
    *,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    header, data_start = _safe_tensor_header(value)
    entries = {
        str(name): dict(item)
        for name, item in header.items()
        if name != "__metadata__" and isinstance(item, dict)
    }
    stage_id = int(manifest["stage_id"])
    spec = canonical_stage_specs()[stage_id]
    expected_names = {
        (
            f"model.layers.{layer}."
            f"{'self_attn' if target.endswith('_proj') and target in {'q_proj', 'k_proj', 'v_proj', 'o_proj'} else 'mlp'}."
            f"{target}.lora_{side}.weight"
        )
        for layer in range(int(spec.layer_start), int(spec.layer_end))
        for target in QWEN_LORA_TARGET_MODULES
        for side in ("A", "B")
    }
    actual_contract_names: set[str] = set()
    spans: list[tuple[int, int, str]] = []
    tensor_digest = hashlib.sha256()
    total_elements = 0
    for name in sorted(entries):
        item = entries[name]
        match = ADAPTER_NAME_PATTERN.fullmatch(name)
        if match is None:
            raise ValueError("elastic_checkpoint_adapter_tensor_name_invalid")
        layer = int(match.group(1))
        target = str(match.group(2) or match.group(3) or "")
        side = str(match.group(4) or "")
        if (
            layer < int(spec.layer_start)
            or layer >= int(spec.layer_end)
            or target not in QWEN_LORA_TARGET_MODULES
            or side not in {"A", "B"}
        ):
            raise ValueError("elastic_checkpoint_adapter_tensor_ownership_invalid")
        canonical_name = (
            f"model.layers.{layer}."
            f"{'self_attn' if target in {'q_proj', 'k_proj', 'v_proj', 'o_proj'} else 'mlp'}."
            f"{target}.lora_{side}.weight"
        )
        actual_contract_names.add(canonical_name)
        dtype_name = str(item.get("dtype") or "").upper()
        if dtype_name not in SAFETENSORS_DTYPES:
            raise ValueError("elastic_checkpoint_adapter_tensor_dtype_invalid")
        shape = item.get("shape")
        offsets = item.get("data_offsets")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(not isinstance(dimension, int) or dimension < 1 for dimension in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
        ):
            raise ValueError("elastic_checkpoint_adapter_tensor_shape_invalid")
        start, end = (int(offsets[0]), int(offsets[1]))
        element_count = math.prod(shape)
        if (
            start < 0
            or end <= start
            or end - start != element_count * SAFETENSORS_DTYPES[dtype_name][0]
            or data_start + end > len(value)
        ):
            raise ValueError("elastic_checkpoint_adapter_tensor_offsets_invalid")
        spans.append((start, end, name))
        raw = value[data_start + start : data_start + end]
        if not _tensor_is_finite(raw, dtype_name):
            raise ValueError("elastic_checkpoint_adapter_tensor_non_finite")
        tensor_digest.update(name.encode("utf-8") + b"\0")
        tensor_digest.update(len(raw).to_bytes(8, "little") + raw)
        total_elements += element_count
    spans.sort()
    previous_end = 0
    for start, end, _name in spans:
        if start != previous_end:
            raise ValueError("elastic_checkpoint_adapter_tensor_offsets_invalid")
        previous_end = end
    if data_start + previous_end != len(value):
        raise ValueError("elastic_checkpoint_adapter_tensor_offsets_invalid")
    if actual_contract_names != expected_names:
        raise ValueError("elastic_checkpoint_adapter_tensor_coverage_invalid")
    if int(manifest.get("adapter_tensor_count") or -1) != len(entries):
        raise ValueError("elastic_checkpoint_adapter_tensor_count_invalid")
    tensor_hash = "sha256:" + tensor_digest.hexdigest()
    if tensor_hash != str(manifest.get("adapter_tensor_hash") or ""):
        raise ValueError("elastic_checkpoint_adapter_tensor_hash_invalid")
    return {
        "adapter_safetensors_validated": True,
        "adapter_tensor_names_validated": True,
        "adapter_tensor_coverage_validated": True,
        "adapter_tensor_finite": True,
        "adapter_tensor_count": len(entries),
        "adapter_element_count": total_elements,
    }


def _validate_resume_state_component(value: bytes, *, kind: str) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "elastic_checkpoint_tensor_validation_requires_torch"
        ) from exc
    try:
        parsed = torch.load(io.BytesIO(value), map_location="cpu", weights_only=True)
    except BaseException as exc:
        raise ValueError(f"elastic_checkpoint_{kind}_state_invalid") from exc
    tensor_count = 0
    tensor_bytes = 0
    seen: set[int] = set()

    def inspect(item: Any, *, depth: int = 0) -> None:
        nonlocal tensor_count, tensor_bytes
        if depth > 32:
            raise ValueError(f"elastic_checkpoint_{kind}_state_too_deep")
        if isinstance(item, torch.Tensor):
            tensor_count += 1
            tensor_bytes += int(item.numel()) * int(item.element_size())
            if (item.is_floating_point() or item.is_complex()) and not bool(
                torch.isfinite(item).all()
            ):
                raise ValueError(f"elastic_checkpoint_{kind}_state_non_finite")
            return
        if item is None or isinstance(item, (str, bytes, bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"elastic_checkpoint_{kind}_state_non_finite")
            return
        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(item, dict):
            if len(item) > 1_000_000:
                raise ValueError(f"elastic_checkpoint_{kind}_state_too_large")
            for key, child in item.items():
                inspect(key, depth=depth + 1)
                inspect(child, depth=depth + 1)
            return
        if isinstance(item, (list, tuple)):
            if len(item) > 1_000_000:
                raise ValueError(f"elastic_checkpoint_{kind}_state_too_large")
            for child in item:
                inspect(child, depth=depth + 1)
            return
        raise ValueError(f"elastic_checkpoint_{kind}_state_type_invalid")

    inspect(parsed)
    if kind == "optimizer" and (
        not isinstance(parsed, dict)
        or not isinstance(parsed.get("state"), dict)
        or not isinstance(parsed.get("param_groups"), list)
    ):
        raise ValueError("elastic_checkpoint_optimizer_state_invalid")
    if kind == "rng" and (
        not isinstance(parsed, dict) or not isinstance(parsed.get("cpu"), torch.Tensor)
    ):
        raise ValueError("elastic_checkpoint_rng_state_invalid")
    if kind == "scaler" and not isinstance(parsed, dict):
        raise ValueError("elastic_checkpoint_scaler_state_invalid")
    return {
        f"{kind}_state_safe_loaded": True,
        f"{kind}_tensor_count": tensor_count,
        f"{kind}_tensor_bytes": tensor_bytes,
        f"{kind}_tensor_values_finite": True,
    }


def _validated_checkpoint_manifest(
    manifest: dict[str, Any],
    *,
    expected_stage_id: int | None = None,
    expected_step: int | None = None,
    expected_dataset_cursor: int | None = None,
    expected_model_id: str = MODEL_ID,
    expected_model_revision: str = MODEL_REVISION,
    stage_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if manifest.get("schema") != QWEN_STAGE_CHECKPOINT_SCHEMA:
        raise ValueError("elastic_checkpoint_manifest_schema_invalid")
    stage_id = int(manifest.get("stage_id", -1))
    specs = {
        int(spec["stage_id"]): dict(spec)
        for spec in (
            stage_specs
            if stage_specs is not None
            else [item.public_dict() for item in canonical_stage_specs()]
        )
    }
    spec = specs.get(stage_id)
    if spec is None:
        raise ValueError("elastic_checkpoint_stage_invalid")
    if expected_stage_id is not None and stage_id != int(expected_stage_id):
        raise ValueError("elastic_checkpoint_stage_mismatch")
    if (
        manifest.get("model_id") != str(expected_model_id)
        or manifest.get("model_revision") != str(expected_model_revision)
        or int(manifest.get("layer_start", -1)) != int(spec["layer_start"])
        or int(manifest.get("layer_end", -1)) != int(spec["layer_end"])
    ):
        raise ValueError("elastic_checkpoint_ownership_invalid")
    global_step = int(manifest.get("global_step", -1))
    optimizer_step = int(manifest.get("optimizer_step", -1))
    dataset_cursor = int(manifest.get("dataset_cursor", -1))
    if global_step < 1 or optimizer_step != global_step or dataset_cursor < 1:
        raise ValueError("elastic_checkpoint_progress_invalid")
    if expected_step is not None and global_step != int(expected_step):
        raise ValueError("elastic_checkpoint_step_mismatch")
    if expected_dataset_cursor is not None and dataset_cursor != int(
        expected_dataset_cursor
    ):
        raise ValueError("elastic_checkpoint_cursor_mismatch")
    expected_names = _checkpoint_file_names(stage_id)
    actual_names = {
        "adapter": str(manifest.get("adapter_file") or ""),
        "optimizer": str(manifest.get("optimizer_file") or ""),
        "scaler": str(manifest.get("grad_scaler_file") or ""),
        "rng": str(manifest.get("rng_file") or ""),
    }
    if any(actual_names[key] != expected_names[key] for key in actual_names):
        raise ValueError("elastic_checkpoint_component_name_invalid")
    expected_hash_keys = (
        "adapter_file_hash",
        "optimizer_file_hash",
        "grad_scaler_file_hash",
        "rng_file_hash",
        "adapter_tensor_hash",
        "content_hash",
    )
    if any(
        not str(manifest.get(key) or "").startswith("sha256:")
        for key in expected_hash_keys
    ):
        raise ValueError("elastic_checkpoint_hash_contract_invalid")
    content_hash = str(manifest["content_hash"])
    unhashed = {key: value for key, value in manifest.items() if key != "content_hash"}
    if stable_hash(unhashed) != content_hash:
        raise ValueError("elastic_checkpoint_manifest_content_hash_invalid")
    if (
        manifest.get("grad_scaler_state_present") is not True
        or manifest.get("rng_state_present") is not True
    ):
        raise ValueError("elastic_checkpoint_resume_state_incomplete")
    return dict(manifest)


def _inspect_checkpoint_archive(
    archive: bytes,
    *,
    expected_stage_id: int | None = None,
    expected_step: int | None = None,
    expected_dataset_cursor: int | None = None,
    max_checkpoint_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
    validate_tensor_payloads: bool = False,
    include_files: bool = False,
    expected_model_id: str = MODEL_ID,
    expected_model_revision: str = MODEL_REVISION,
    stage_specs: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bytes]]:
    if not archive or len(archive) > int(max_checkpoint_bytes):
        raise ValueError("elastic_checkpoint_archive_size_invalid")
    try:
        bundle = zipfile.ZipFile(io.BytesIO(archive), mode="r")
    except zipfile.BadZipFile as exc:
        raise ValueError("elastic_checkpoint_archive_invalid") from exc
    with bundle:
        infos = bundle.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)) or len(names) != 5:
            raise ValueError("elastic_checkpoint_archive_entries_invalid")
        total_size = 0
        for info in infos:
            path = PurePosixPath(info.filename)
            mode = int(info.external_attr >> 16)
            if (
                path.is_absolute()
                or len(path.parts) != 1
                or path.name in {"", ".", ".."}
                or "\\" in info.filename
                or stat.S_ISLNK(mode)
                or info.flag_bits & 0x1
                or int(info.file_size) < 0
            ):
                raise ValueError("elastic_checkpoint_archive_path_invalid")
            total_size += int(info.file_size)
        if total_size > int(max_checkpoint_bytes):
            raise ValueError("elastic_checkpoint_archive_unpacked_size_invalid")
        manifest_names = [name for name in names if name.endswith("_checkpoint.json")]
        if len(manifest_names) != 1:
            raise ValueError("elastic_checkpoint_archive_manifest_missing")
        raw_manifest = bundle.read(manifest_names[0])
        if len(raw_manifest) > 1024 * 1024:
            raise ValueError("elastic_checkpoint_manifest_size_invalid")
        try:
            parsed = json.loads(raw_manifest.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("elastic_checkpoint_manifest_json_invalid") from exc
        if not isinstance(parsed, dict):
            raise ValueError("elastic_checkpoint_manifest_json_invalid")
        manifest = _validated_checkpoint_manifest(
            parsed,
            expected_stage_id=expected_stage_id,
            expected_step=expected_step,
            expected_dataset_cursor=expected_dataset_cursor,
            expected_model_id=expected_model_id,
            expected_model_revision=expected_model_revision,
            stage_specs=stage_specs,
        )
        expected_names = _checkpoint_file_names(int(manifest["stage_id"]))
        if set(names) != set(expected_names.values()):
            raise ValueError("elastic_checkpoint_archive_entries_invalid")
        files: dict[str, bytes] = {}
        component_contract = {
            expected_names["adapter"]: str(manifest["adapter_file_hash"]),
            expected_names["optimizer"]: str(manifest["optimizer_file_hash"]),
            expected_names["scaler"]: str(manifest["grad_scaler_file_hash"]),
            expected_names["rng"]: str(manifest["rng_file_hash"]),
        }
        component_values: dict[str, bytes] = {}
        for name, expected_hash in component_contract.items():
            raw = bundle.read(name)
            if _sha256_bytes(raw) != expected_hash:
                raise ValueError("elastic_checkpoint_component_hash_invalid")
            component_values[name] = raw
            if include_files:
                files[name] = raw
        if include_files:
            files[expected_names["manifest"]] = raw_manifest
        tensor_validation: dict[str, Any] = {
            "tensor_payload_validation_enabled": bool(validate_tensor_payloads),
            "adapter_safetensors_validated": False,
            "resume_state_safe_loaded": False,
        }
        if validate_tensor_payloads:
            tensor_validation.update(
                _validate_adapter_safetensors(
                    component_values[expected_names["adapter"]], manifest=manifest
                )
            )
            for key, kind in (
                ("optimizer", "optimizer"),
                ("scaler", "scaler"),
                ("rng", "rng"),
            ):
                tensor_validation.update(
                    _validate_resume_state_component(
                        component_values[expected_names[key]], kind=kind
                    )
                )
            tensor_validation["resume_state_safe_loaded"] = True
    report = {
        "schema": ARCHIVE_SCHEMA,
        "archive_hash": _sha256_bytes(archive),
        "archive_bytes": len(archive),
        "stage_id": int(manifest["stage_id"]),
        "layer_start": int(manifest["layer_start"]),
        "layer_end": int(manifest["layer_end"]),
        "model_id": str(manifest["model_id"]),
        "model_revision": str(manifest["model_revision"]),
        "global_step": int(manifest["global_step"]),
        "optimizer_step": int(manifest["optimizer_step"]),
        "dataset_cursor": int(manifest["dataset_cursor"]),
        "checkpoint_content_hash": str(manifest["content_hash"]),
        "adapter_tensor_hash": str(manifest["adapter_tensor_hash"]),
        "component_hashes_hash": stable_hash(
            {
                "adapter": manifest["adapter_file_hash"],
                "optimizer": manifest["optimizer_file_hash"],
                "scaler": manifest["grad_scaler_file_hash"],
                "rng": manifest["rng_file_hash"],
            }
        ),
        "optimizer_state_present": True,
        "grad_scaler_state_present": True,
        "rng_state_present": True,
        **tensor_validation,
        "archive_paths_validated": True,
        "private_paths_public": False,
        "tensor_values_public": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = stable_hash(report)
    return report, manifest, files


def validate_qwen_stage_checkpoint_archive(
    archive: bytes,
    *,
    expected_stage_id: int | None = None,
    expected_step: int | None = None,
    expected_dataset_cursor: int | None = None,
    max_checkpoint_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
    validate_tensor_payloads: bool = False,
    expected_model_id: str = MODEL_ID,
    expected_model_revision: str = MODEL_REVISION,
    stage_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate an archive without extracting private optimizer state."""

    report, _manifest, _files = _inspect_checkpoint_archive(
        archive,
        expected_stage_id=expected_stage_id,
        expected_step=expected_step,
        expected_dataset_cursor=expected_dataset_cursor,
        max_checkpoint_bytes=max_checkpoint_bytes,
        validate_tensor_payloads=validate_tensor_payloads,
        expected_model_id=expected_model_id,
        expected_model_revision=expected_model_revision,
        stage_specs=stage_specs,
    )
    return report


def build_qwen_stage_checkpoint_archive(
    checkpoint_dir: str | Path,
    *,
    stage_id: int,
    output_path: str | Path | None = None,
    max_checkpoint_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
) -> tuple[bytes, dict[str, Any]]:
    """Build a deterministic five-file archive from one Qwen stage checkpoint."""

    root = Path(checkpoint_dir)
    names = _checkpoint_file_names(stage_id)
    manifest_path = root / names["manifest"]
    try:
        parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("elastic_checkpoint_manifest_unreadable") from exc
    manifest = _validated_checkpoint_manifest(
        parsed,
        expected_stage_id=stage_id,
        expected_model_id=str(parsed.get("model_id") or ""),
        expected_model_revision=str(parsed.get("model_revision") or ""),
    )
    component_contract = {
        names["adapter"]: str(manifest["adapter_file_hash"]),
        names["optimizer"]: str(manifest["optimizer_file_hash"]),
        names["scaler"]: str(manifest["grad_scaler_file_hash"]),
        names["rng"]: str(manifest["rng_file_hash"]),
    }
    values: dict[str, bytes] = {names["manifest"]: manifest_path.read_bytes()}
    for name, expected_hash in component_contract.items():
        path = root / name
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ValueError("elastic_checkpoint_component_hash_invalid")
        values[name] = path.read_bytes()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as bundle:
        for name in sorted(values):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100600 << 16
            bundle.writestr(info, values[name])
    archive = stream.getvalue()
    report = validate_qwen_stage_checkpoint_archive(
        archive,
        expected_stage_id=stage_id,
        expected_step=int(manifest["global_step"]),
        expected_dataset_cursor=int(manifest["dataset_cursor"]),
        max_checkpoint_bytes=max_checkpoint_bytes,
        expected_model_id=str(manifest["model_id"]),
        expected_model_revision=str(manifest["model_revision"]),
    )
    if output_path is not None:
        _atomic_write(Path(output_path), archive)
    return archive, report


def restore_qwen_stage_checkpoint_archive(
    archive: bytes,
    checkpoint_dir: str | Path,
    *,
    expected_stage_id: int,
    expected_step: int,
    expected_dataset_cursor: int,
    max_checkpoint_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
    validate_tensor_payloads: bool = False,
    expected_model_id: str | None = None,
    expected_model_revision: str | None = None,
) -> dict[str, Any]:
    """Hash-check and restore one stage, publishing its manifest last."""

    if expected_model_id is None or expected_model_revision is None:
        with zipfile.ZipFile(io.BytesIO(archive), mode="r") as bundle:
            names = [name for name in bundle.namelist() if name.endswith("_checkpoint.json")]
            parsed = json.loads(bundle.read(names[0])) if len(names) == 1 else {}
        expected_model_id = str(parsed.get("model_id") or "")
        expected_model_revision = str(parsed.get("model_revision") or "")
    report, manifest, files = _inspect_checkpoint_archive(
        archive,
        expected_stage_id=expected_stage_id,
        expected_step=expected_step,
        expected_dataset_cursor=expected_dataset_cursor,
        max_checkpoint_bytes=max_checkpoint_bytes,
        validate_tensor_payloads=validate_tensor_payloads,
        include_files=True,
        expected_model_id=str(expected_model_id),
        expected_model_revision=str(expected_model_revision),
    )
    output = Path(checkpoint_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = _checkpoint_file_names(int(manifest["stage_id"]))
    for key in ("adapter", "optimizer", "scaler", "rng", "manifest"):
        name = names[key]
        _atomic_write(output / name, files[name])
    return report


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    for private_path in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if private_path.exists():
            private_path.chmod(0o600)
    return connection


class ElasticTrainingRuntime:
    """SQLite-backed elastic scheduler and atomic stage checkpoint barrier."""

    def __init__(
        self,
        state_path: str | Path,
        *,
        run_id: str,
        target_steps: int = 8,
        microbatches_per_step: int = 4,
        lease_seconds: float = 30.0,
        max_checkpoint_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
        blob_store: CheckpointBlobStore | None = None,
        checkpoint_retention_steps: int = 2,
        require_checkpoint_signatures: bool = False,
        validate_checkpoint_tensors: bool = False,
        max_online_miners: int = 32,
        max_rejected_submissions_per_session: int = 5,
        max_checkpoint_bytes_per_session: int = 0,
        tensor_lookup_optimization_after_step: int = 0,
        training_manifest: dict[str, Any] | None = None,
        legacy_model_id: str = MODEL_ID,
        legacy_model_revision: str = MODEL_REVISION,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.state_path = Path(state_path).expanduser().resolve()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.blob_dir = self.state_path.parent / f"{self.state_path.stem}-checkpoint-blobs"
        self.private_config_path = self.state_path.parent / (
            f"{self.state_path.stem}-runtime-private.json"
        )
        self.run_id = str(run_id)
        self.target_steps = int(target_steps)
        self.microbatches_per_step = int(microbatches_per_step)
        self.lease_seconds = float(lease_seconds)
        self.max_checkpoint_bytes = int(max_checkpoint_bytes)
        self.checkpoint_retention_steps = int(checkpoint_retention_steps)
        self.require_checkpoint_signatures = bool(require_checkpoint_signatures)
        self.validate_checkpoint_tensors = bool(validate_checkpoint_tensors)
        self.max_online_miners = int(max_online_miners)
        self.max_rejected_submissions_per_session = int(
            max_rejected_submissions_per_session
        )
        self.max_checkpoint_bytes_per_session = int(
            max_checkpoint_bytes_per_session
            or self.target_steps * 2 * self.max_checkpoint_bytes
        )
        self.tensor_lookup_optimization_after_step = int(
            tensor_lookup_optimization_after_step
        )
        self.training_manifest = (
            validate_training_manifest(training_manifest)
            if training_manifest is not None
            else None
        )
        self.legacy_model_id = str(legacy_model_id)
        self.legacy_model_revision = str(legacy_model_revision)
        if not self.legacy_model_id or not self.legacy_model_revision:
            raise ValueError("elastic_legacy_model_identity_required")
        self.heterogeneous_scheduler_enabled = self.training_manifest is not None
        if self.training_manifest is not None and (
            int(self.training_manifest["training"]["target_steps"])
            != self.target_steps
            or int(self.training_manifest["training"]["microbatches_per_step"])
            != self.microbatches_per_step
        ):
            raise ValueError("elastic_training_manifest_progress_conflict")
        self._clock = clock
        if not self.run_id:
            raise ValueError("elastic_run_id_required")
        if self.target_steps < 1 or self.microbatches_per_step < 1:
            raise ValueError("elastic_progress_contract_invalid")
        if self.lease_seconds <= 0:
            raise ValueError("elastic_lease_seconds_invalid")
        if (
            self.checkpoint_retention_steps < 1
            or self.max_online_miners < 1
            or self.max_rejected_submissions_per_session < 1
            or self.max_checkpoint_bytes_per_session < self.max_checkpoint_bytes
            or self.tensor_lookup_optimization_after_step < 0
            or self.tensor_lookup_optimization_after_step > self.target_steps
        ):
            raise ValueError("elastic_runtime_quota_contract_invalid")
        private_configuration: dict[str, Any] = {}
        if self.private_config_path.is_file():
            try:
                private_configuration = json.loads(
                    self.private_config_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("elastic_runtime_private_configuration_invalid") from exc
        if blob_store is None:
            storage_configuration = dict(
                private_configuration.get("checkpoint_storage") or {}
            )
            blob_store = checkpoint_blob_store_from_configuration(
                storage_configuration,
                default_root=self.blob_dir,
            )
        if private_configuration and "tensor_lookup_optimization_after_step" not in private_configuration:
            private_configuration["tensor_lookup_optimization_after_step"] = 0
        self.blob_store = blob_store
        if isinstance(self.blob_store, LocalCheckpointBlobStore):
            self.blob_dir = self.blob_store.root
        runtime_configuration = {
            "schema": "crowdtensor_elastic_training_runtime_private_config_v1",
            "run_id": self.run_id,
            "checkpoint_storage": self.blob_store.private_configuration(),
            "checkpoint_retention_steps": self.checkpoint_retention_steps,
            "require_checkpoint_signatures": self.require_checkpoint_signatures,
            "validate_checkpoint_tensors": self.validate_checkpoint_tensors,
            "max_online_miners": self.max_online_miners,
            "max_rejected_submissions_per_session": self.max_rejected_submissions_per_session,
            "max_checkpoint_bytes_per_session": self.max_checkpoint_bytes_per_session,
            "tensor_lookup_optimization_after_step": self.tensor_lookup_optimization_after_step,
            "public_artifact": False,
        }
        if self.training_manifest is not None:
            runtime_configuration["training_manifest"] = self.training_manifest
        if (
            self.legacy_model_id != MODEL_ID
            or self.legacy_model_revision != MODEL_REVISION
        ):
            runtime_configuration["legacy_model"] = {
                "model_id": self.legacy_model_id,
                "model_revision": self.legacy_model_revision,
            }
        if private_configuration and private_configuration != runtime_configuration:
            raise ValueError("elastic_runtime_private_configuration_conflict")
        if not private_configuration:
            _atomic_write(
                self.private_config_path,
                (json.dumps(runtime_configuration, sort_keys=True) + "\n").encode(
                    "utf-8"
                ),
            )
        self._initialize()
        self.tensor_store = (
            ChunkedTensorStore(
                self.state_path.parent / f"{self.state_path.stem}-tensor-payloads"
            )
            if self.training_manifest is not None
            else None
        )

    @classmethod
    def open_existing(
        cls,
        state_path: str | Path,
        *,
        run_id: str,
        lease_seconds: float = 30.0,
        clock: Callable[[], float] = time.time,
    ) -> "ElasticTrainingRuntime":
        path = Path(state_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("elastic_training_state_not_found")
        with _connect(path) as connection:
            try:
                row = connection.execute(
                    """
                    SELECT target_steps,microbatches_per_step FROM jobs
                    WHERE run_id=?
                    """,
                    (str(run_id),),
                ).fetchone()
            except sqlite3.Error as exc:
                raise ValueError("elastic_training_state_schema_invalid") from exc
        if row is None:
            raise KeyError("elastic_job_not_found")
        private_path = path.parent / f"{path.stem}-runtime-private.json"
        private: dict[str, Any] = {}
        if private_path.is_file():
            try:
                private = json.loads(private_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("elastic_runtime_private_configuration_invalid") from exc
        storage = checkpoint_blob_store_from_configuration(
            dict(private.get("checkpoint_storage") or {}),
            default_root=path.parent / f"{path.stem}-checkpoint-blobs",
        )
        return cls(
            path,
            run_id=str(run_id),
            target_steps=int(row["target_steps"]),
            microbatches_per_step=int(row["microbatches_per_step"]),
            lease_seconds=float(lease_seconds),
            blob_store=storage,
            checkpoint_retention_steps=int(
                private.get("checkpoint_retention_steps") or 2
            ),
            require_checkpoint_signatures=bool(
                private.get("require_checkpoint_signatures", False)
            ),
            validate_checkpoint_tensors=bool(
                private.get("validate_checkpoint_tensors", False)
            ),
            max_online_miners=int(private.get("max_online_miners") or 32),
            max_rejected_submissions_per_session=int(
                private.get("max_rejected_submissions_per_session") or 5
            ),
            max_checkpoint_bytes_per_session=int(
                private.get("max_checkpoint_bytes_per_session") or 0
            ),
            tensor_lookup_optimization_after_step=int(
                private.get("tensor_lookup_optimization_after_step") or 0
            ),
            training_manifest=(
                dict(private["training_manifest"])
                if isinstance(private.get("training_manifest"), dict)
                else None
            ),
            legacy_model_id=str(
                (private.get("legacy_model") or {}).get("model_id") or MODEL_ID
            ),
            legacy_model_revision=str(
                (private.get("legacy_model") or {}).get("model_revision")
                or MODEL_REVISION
            ),
            clock=clock,
        )
    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = _connect(self.state_path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        now = float(self._clock())
        specs = (
            [dict(spec) for spec in self.training_manifest["stages"]]
            if self.training_manifest is not None
            else [spec.public_dict() for spec in canonical_stage_specs()]
        )
        model_id = (
            str(self.training_manifest["model"]["model_id"])
            if self.training_manifest is not None
            else self.legacy_model_id
        )
        model_revision = (
            str(self.training_manifest["model"]["model_revision"])
            if self.training_manifest is not None
            else self.legacy_model_revision
        )
        manifest_hash = str(
            (self.training_manifest or {}).get("content_hash") or ""
        )
        with _connect(self.state_path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    run_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    model_revision TEXT NOT NULL,
                    stage_specs_json TEXT NOT NULL,
                    stage_count INTEGER NOT NULL,
                    target_steps INTEGER NOT NULL,
                    microbatches_per_step INTEGER NOT NULL,
                    committed_step INTEGER NOT NULL DEFAULT 0,
                    dataset_cursor INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL,
                    active_epoch_id INTEGER,
                    next_epoch_id INTEGER NOT NULL DEFAULT 1,
                    revision INTEGER NOT NULL DEFAULT 1,
                    manifest_hash TEXT NOT NULL DEFAULT '',
                    placement_generation INTEGER NOT NULL DEFAULT 0,
                    placement_plan_json TEXT NOT NULL DEFAULT '{}',
                    placement_error_json TEXT NOT NULL DEFAULT '{}',
                    pending_rebalance_reason TEXT NOT NULL DEFAULT '',
                    owner_paused INTEGER NOT NULL DEFAULT 0,
                    coordinator_generation INTEGER NOT NULL DEFAULT 0,
                    coordinator_started_at REAL NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS miners (
                    session_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    miner_id_hash TEXT NOT NULL,
                    registration_hash TEXT NOT NULL,
                    session_token TEXT NOT NULL,
                    session_token_hash TEXT NOT NULL,
                    supported_stage_ids_json TEXT NOT NULL,
                    slot_count INTEGER NOT NULL,
                    accelerator TEXT NOT NULL,
                    state TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    registered_at REAL NOT NULL,
                    heartbeat_at REAL NOT NULL,
                    lease_expires_at REAL NOT NULL,
                    offline_at REAL NOT NULL DEFAULT 0,
                    accepted_upload_bytes INTEGER NOT NULL DEFAULT 0,
                    rejected_submission_count INTEGER NOT NULL DEFAULT 0,
                    capability_json TEXT NOT NULL DEFAULT '{}',
                    stage_metrics_json TEXT NOT NULL DEFAULT '[]',
                    last_failure_reason TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(run_id) REFERENCES jobs(run_id)
                );
                CREATE INDEX IF NOT EXISTS miners_run_state
                    ON miners(run_id,state,lease_expires_at);
                CREATE INDEX IF NOT EXISTS miners_registration
                    ON miners(run_id,registration_hash,state);
                CREATE TABLE IF NOT EXISTS epochs (
                    run_id TEXT NOT NULL,
                    epoch_id INTEGER NOT NULL,
                    base_step INTEGER NOT NULL,
                    target_step INTEGER NOT NULL,
                    dataset_cursor INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    aborted_at REAL NOT NULL DEFAULT 0,
                    abort_reason TEXT NOT NULL DEFAULT '',
                    committed_at REAL NOT NULL DEFAULT 0,
                    checkpoint_set_hash TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(run_id,epoch_id),
                    FOREIGN KEY(run_id) REFERENCES jobs(run_id)
                );
                CREATE TABLE IF NOT EXISTS assignments (
                    run_id TEXT NOT NULL,
                    epoch_id INTEGER NOT NULL,
                    stage_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    assignment_token TEXT NOT NULL,
                    assignment_token_hash TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    revoked_at REAL NOT NULL DEFAULT 0,
                    placement_generation INTEGER NOT NULL DEFAULT 0,
                    device_id TEXT NOT NULL DEFAULT '',
                    device_type TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(run_id,epoch_id,stage_id),
                    FOREIGN KEY(run_id,epoch_id) REFERENCES epochs(run_id,epoch_id),
                    FOREIGN KEY(session_id) REFERENCES miners(session_id)
                );
                CREATE INDEX IF NOT EXISTS assignments_session
                    ON assignments(session_id,state);
                CREATE TABLE IF NOT EXISTS submissions (
                    run_id TEXT NOT NULL,
                    epoch_id INTEGER NOT NULL,
                    stage_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    target_step INTEGER NOT NULL,
                    dataset_cursor INTEGER NOT NULL,
                    archive_hash TEXT NOT NULL,
                    archive_bytes INTEGER NOT NULL,
                    checkpoint_content_hash TEXT NOT NULL,
                    adapter_tensor_hash TEXT NOT NULL,
                    component_hashes_hash TEXT NOT NULL,
                    state TEXT NOT NULL,
                    submitted_at REAL NOT NULL,
                    PRIMARY KEY(run_id,epoch_id,stage_id),
                    FOREIGN KEY(run_id,epoch_id,stage_id)
                        REFERENCES assignments(run_id,epoch_id,stage_id)
                );
                CREATE TABLE IF NOT EXISTS commits (
                    run_id TEXT NOT NULL,
                    target_step INTEGER NOT NULL,
                    epoch_id INTEGER NOT NULL,
                    dataset_cursor INTEGER NOT NULL,
                    checkpoint_set_hash TEXT NOT NULL,
                    committed_at REAL NOT NULL,
                    PRIMARY KEY(run_id,target_step),
                    UNIQUE(run_id,epoch_id),
                    FOREIGN KEY(run_id,epoch_id) REFERENCES epochs(run_id,epoch_id)
                );
                CREATE TABLE IF NOT EXISTS elastic_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(run_id,operation,event_json)
                );
                CREATE TABLE IF NOT EXISTS device_health (
                    run_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'healthy',
                    consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    total_failures INTEGER NOT NULL DEFAULT 0,
                    quarantine_until REAL NOT NULL DEFAULT 0,
                    last_success_at REAL NOT NULL DEFAULT 0,
                    last_failure_at REAL NOT NULL DEFAULT 0,
                    last_failure_class TEXT NOT NULL DEFAULT '',
                    checkpoint_step INTEGER NOT NULL DEFAULT 0,
                    telemetry_json TEXT NOT NULL DEFAULT '{}',
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(run_id,session_id,device_id),
                    FOREIGN KEY(session_id) REFERENCES miners(session_id)
                );
                CREATE INDEX IF NOT EXISTS device_health_run_state
                    ON device_health(run_id,state,quarantine_until);
                """
            )
            miner_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(miners)").fetchall()
            }
            if "accepted_upload_bytes" not in miner_columns:
                connection.execute(
                    "ALTER TABLE miners ADD COLUMN accepted_upload_bytes INTEGER NOT NULL DEFAULT 0"
                )
            if "rejected_submission_count" not in miner_columns:
                connection.execute(
                    "ALTER TABLE miners ADD COLUMN rejected_submission_count INTEGER NOT NULL DEFAULT 0"
                )
            if "capability_json" not in miner_columns:
                connection.execute(
                    "ALTER TABLE miners ADD COLUMN capability_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "stage_metrics_json" not in miner_columns:
                connection.execute(
                    "ALTER TABLE miners ADD COLUMN stage_metrics_json TEXT NOT NULL DEFAULT '[]'"
                )
            if "last_failure_reason" not in miner_columns:
                connection.execute(
                    "ALTER TABLE miners ADD COLUMN last_failure_reason TEXT NOT NULL DEFAULT ''"
                )
            job_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
            }
            for column, declaration in (
                ("manifest_hash", "TEXT NOT NULL DEFAULT ''"),
                ("placement_generation", "INTEGER NOT NULL DEFAULT 0"),
                ("placement_plan_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("placement_error_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("pending_rebalance_reason", "TEXT NOT NULL DEFAULT ''"),
                ("owner_paused", "INTEGER NOT NULL DEFAULT 0"),
                ("coordinator_generation", "INTEGER NOT NULL DEFAULT 0"),
                ("coordinator_started_at", "REAL NOT NULL DEFAULT 0"),
            ):
                if column not in job_columns:
                    connection.execute(
                        f"ALTER TABLE jobs ADD COLUMN {column} {declaration}"
                    )
            assignment_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(assignments)"
                ).fetchall()
            }
            for column, declaration in (
                ("placement_generation", "INTEGER NOT NULL DEFAULT 0"),
                ("device_id", "TEXT NOT NULL DEFAULT ''"),
                ("device_type", "TEXT NOT NULL DEFAULT ''"),
            ):
                if column not in assignment_columns:
                    connection.execute(
                        f"ALTER TABLE assignments ADD COLUMN {column} {declaration}"
                    )
            row = connection.execute(
                "SELECT * FROM jobs WHERE run_id=?", (self.run_id,)
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO jobs(
                        run_id,model_id,model_revision,stage_specs_json,stage_count,
                        target_steps,microbatches_per_step,committed_step,dataset_cursor,
                        state,active_epoch_id,next_epoch_id,revision,created_at,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        self.run_id,
                        model_id,
                        model_revision,
                        json.dumps(specs, sort_keys=True),
                        len(specs),
                        self.target_steps,
                        self.microbatches_per_step,
                        0,
                        0,
                        "paused_waiting_for_miners",
                        None,
                        1,
                        1,
                        now,
                        now,
                    ),
                )
                if manifest_hash:
                    connection.execute(
                        "UPDATE jobs SET manifest_hash=? WHERE run_id=?",
                        (manifest_hash, self.run_id),
                    )
            else:
                expected = (
                    str(row["model_id"]) == model_id
                    and str(row["model_revision"]) == model_revision
                    and int(row["target_steps"]) == self.target_steps
                    and int(row["microbatches_per_step"])
                    == self.microbatches_per_step
                    and int(row["stage_count"]) == len(specs)
                    and str(row["manifest_hash"] or "") == manifest_hash
                )
                if not expected:
                    raise ValueError("elastic_persistent_job_contract_conflict")
        self.state_path.chmod(0o600)
        self.private_config_path.chmod(0o600)

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        *,
        run_id: str,
        operation: str,
        value: dict[str, Any],
        now: float,
    ) -> None:
        public = dict(value)
        public["operation"] = operation
        connection.execute(
            """
            INSERT OR IGNORE INTO elastic_events(
                run_id,operation,event_json,created_at
            ) VALUES(?,?,?,?)
            """,
            (run_id, operation, json.dumps(public, sort_keys=True), now),
        )

    def _job(self, connection: sqlite3.Connection) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM jobs WHERE run_id=?", (self.run_id,)
        ).fetchone()
        if row is None:
            raise KeyError("elastic_job_not_found")
        return row

    @staticmethod
    def _stage_mapping(
        miner_rows: list[sqlite3.Row], stage_ids: list[int]
    ) -> dict[int, str] | None:
        supported = {
            str(row["session_id"]): {
                int(value) for value in json.loads(str(row["supported_stage_ids_json"]))
            }
            for row in miner_rows
        }
        capacities = {
            str(row["session_id"]): int(row["slot_count"]) for row in miner_rows
        }
        if stage_ids == [0, 1, 2, 3]:
            grouped_result: dict[int, str] = {}
            groups = ([0, 1], [2, 3])

            def assign_group(index: int) -> bool:
                if index == len(groups):
                    return True
                group = groups[index]
                choices = sorted(
                    (
                        str(row["session_id"])
                        for row in miner_rows
                        if set(group).issubset(supported[str(row["session_id"])])
                        and capacities[str(row["session_id"])] >= len(group)
                    ),
                    key=lambda session: (
                        len(supported[session]),
                        -capacities[session],
                        session,
                    ),
                )
                for session_id in choices:
                    capacities[session_id] -= len(group)
                    grouped_result.update(
                        {stage_id: session_id for stage_id in group}
                    )
                    if assign_group(index + 1):
                        return True
                    for stage_id in group:
                        grouped_result.pop(stage_id, None)
                    capacities[session_id] += len(group)
                return False

            if assign_group(0):
                return grouped_result
            capacities = {
                str(row["session_id"]): int(row["slot_count"])
                for row in miner_rows
            }
        candidates = {
            stage_id: [
                str(row["session_id"])
                for row in miner_rows
                if stage_id in supported[str(row["session_id"])]
            ]
            for stage_id in stage_ids
        }
        ordered = sorted(stage_ids, key=lambda value: (len(candidates[value]), value))
        result: dict[int, str] = {}

        def assign(index: int) -> bool:
            if index == len(ordered):
                return True
            stage_id = ordered[index]
            choices = sorted(
                candidates[stage_id],
                key=lambda session: (
                    len(supported[session]),
                    -capacities[session],
                    session,
                ),
            )
            for session_id in choices:
                if capacities[session_id] <= 0:
                    continue
                capacities[session_id] -= 1
                result[stage_id] = session_id
                if assign(index + 1):
                    return True
                result.pop(stage_id, None)
                capacities[session_id] += 1
            return False

        return result if assign(0) else None

    def _heterogeneous_stage_mapping(
        self,
        connection: sqlite3.Connection,
        miner_rows: list[sqlite3.Row],
        stage_ids: list[int],
        *,
        now: float,
    ) -> tuple[dict[int, str] | None, dict[str, Any] | None]:
        if self.training_manifest is None:
            return self._stage_mapping(miner_rows, stage_ids), None
        capabilities = []
        session_by_effective_hash: dict[str, str] = {}
        for row in miner_rows:
            try:
                capability = json.loads(str(row["capability_json"] or "{}"))
            except json.JSONDecodeError:
                continue
            if not isinstance(capability, dict) or not capability:
                continue
            capability.pop("content_hash", None)
            effective_hash = _token_hash(str(row["session_id"]))
            capability["miner_id_hash"] = effective_hash
            capability["max_stage_count"] = min(
                int(capability.get("max_stage_count") or row["slot_count"]),
                int(row["slot_count"]),
            )
            accelerator = str(row["accelerator"])
            if accelerator == "cpu":
                capability["gpus"] = []
                if "tpu_groups" in capability:
                    capability["tpu_groups"] = []
                    capability["jax_tpu_stage_supported"] = False
                capability["cpu_stage_supported"] = True
            elif accelerator == "cuda":
                capability["cpu_stage_supported"] = False
                if "tpu_groups" in capability:
                    capability["tpu_groups"] = []
                    capability["jax_tpu_stage_supported"] = False
            elif accelerator == "tpu":
                capability["gpus"] = []
                capability["cpu_stage_supported"] = False
                if not capability.get("tpu_groups"):
                    continue
            elif accelerator != "mixed":
                continue
            try:
                canonical = validate_miner_capability(capability)
            except ValueError:
                continue
            capabilities.append(canonical)
            session_by_effective_hash[effective_hash] = str(row["session_id"])
        job = self._job(connection)
        previous_plan: dict[str, Any] | None = None
        try:
            parsed_plan = json.loads(str(job["placement_plan_json"] or "{}"))
            if isinstance(parsed_plan, dict) and parsed_plan.get("assignments"):
                previous_plan = parsed_plan
        except json.JSONDecodeError:
            previous_plan = None
        reason = str(job["pending_rebalance_reason"] or "") or "initial_placement"
        telemetry_rows = []
        for health in connection.execute(
            """
            SELECT * FROM device_health WHERE run_id=?
            ORDER BY session_id,device_id
            """,
            (self.run_id,),
        ).fetchall():
            effective_hash = _token_hash(str(health["session_id"]))
            if effective_hash not in session_by_effective_hash:
                continue
            try:
                telemetry = json.loads(str(health["telemetry_json"] or "{}"))
            except json.JSONDecodeError:
                telemetry = {}
            telemetry_rows.append(
                {
                    **(telemetry if isinstance(telemetry, dict) else {}),
                    "miner_id_hash": effective_hash,
                    "device_id": str(health["device_id"]),
                    "health_score": (
                        0.0
                        if str(health["state"]) == "quarantined"
                        else float(
                            (telemetry if isinstance(telemetry, dict) else {}).get(
                                "health_score", 1.0
                            )
                        )
                    ),
                    "checkpoint_step": int(health["checkpoint_step"]),
                    "consecutive_failures": int(
                        health["consecutive_failures"]
                    ),
                    "reported_at": float(health["updated_at"]),
                }
            )
        try:
            plan = build_placement_plan(
                self.training_manifest,
                capabilities,
                previous_plan=previous_plan,
                reason=reason,
                runtime_telemetry=telemetry_rows,
                current_checkpoint_step=int(job["committed_step"]),
            )
        except PlacementError as exc:
            error = {
                "code": exc.code,
                "diagnostics": exc.diagnostics,
                "at": now,
                "public_artifact_safe": True,
            }
            connection.execute(
                "UPDATE jobs SET placement_error_json=? WHERE run_id=?",
                (json.dumps(error, sort_keys=True), self.run_id),
            )
            return None, None
        mapping = {
            int(item["stage_id"]): session_by_effective_hash[
                str(item["miner_id_hash"])
            ]
            for item in plan["assignments"]
        }
        if set(mapping) != set(stage_ids):
            return None, None
        supported_by_session = {
            str(row["session_id"]): {
                int(value)
                for value in json.loads(str(row["supported_stage_ids_json"]))
            }
            for row in miner_rows
        }
        if any(
            stage_id not in supported_by_session.get(session_id, set())
            for stage_id, session_id in mapping.items()
        ):
            connection.execute(
                "UPDATE jobs SET placement_error_json=? WHERE run_id=?",
                (
                    json.dumps(
                        {
                            "code": "heterogeneous_placement_stage_support_mismatch",
                            "at": now,
                            "public_artifact_safe": True,
                        },
                        sort_keys=True,
                    ),
                    self.run_id,
                ),
            )
            return None, None
        return mapping, plan

    def _abort_epoch(
        self,
        connection: sqlite3.Connection,
        *,
        epoch_id: int,
        reason: str,
        now: float,
    ) -> None:
        epoch = connection.execute(
            "SELECT state,target_step FROM epochs WHERE run_id=? AND epoch_id=?",
            (self.run_id, int(epoch_id)),
        ).fetchone()
        if epoch is None or str(epoch["state"]) != "active":
            return
        connection.execute(
            """
            UPDATE epochs SET state='aborted',aborted_at=?,abort_reason=?
            WHERE run_id=? AND epoch_id=? AND state='active'
            """,
            (now, reason, self.run_id, int(epoch_id)),
        )
        connection.execute(
            """
            UPDATE assignments SET state='revoked',revoked_at=?
            WHERE run_id=? AND epoch_id=? AND state='active'
            """,
            (now, self.run_id, int(epoch_id)),
        )
        connection.execute(
            """
            UPDATE submissions SET state='discarded'
            WHERE run_id=? AND epoch_id=? AND state='candidate'
            """,
            (self.run_id, int(epoch_id)),
        )
        self._event(
            connection,
            run_id=self.run_id,
            operation="barrier_epoch_aborted",
            value={
                "epoch_id": int(epoch_id),
                "target_step": int(epoch["target_step"]),
                "reason": reason,
                "uncommitted_candidates_discarded": True,
            },
            now=now,
        )

    def _reconcile(self, connection: sqlite3.Connection, *, now: float) -> None:
        expired = connection.execute(
            """
            SELECT session_id FROM miners
            WHERE run_id=? AND state='online' AND lease_expires_at<=?
            """,
            (self.run_id, now),
        ).fetchall()
        for row in expired:
            session_id = str(row["session_id"])
            connection.execute(
                "UPDATE miners SET state='expired',offline_at=? WHERE session_id=?",
                (now, session_id),
            )
            self._event(
                connection,
                run_id=self.run_id,
                operation="miner_lease_expired",
                value={"miner_session_hash": _token_hash(session_id)},
                now=now,
            )
        if expired and self.training_manifest is not None:
            connection.execute(
                "UPDATE jobs SET pending_rebalance_reason='lease_expired' WHERE run_id=?",
                (self.run_id,),
            )
        job = self._job(connection)
        if str(job["state"]) in {"cancelled", "failed", "cleaned"}:
            return
        if bool(job["owner_paused"]):
            if job["active_epoch_id"] is not None:
                self._abort_epoch(
                    connection,
                    epoch_id=int(job["active_epoch_id"]),
                    reason="owner_paused",
                    now=now,
                )
            connection.execute(
                """
                UPDATE jobs SET state='paused_by_owner',active_epoch_id=NULL,
                    updated_at=? WHERE run_id=?
                """,
                (now, self.run_id),
            )
            return
        active_epoch = job["active_epoch_id"]
        if active_epoch is not None:
            assignments = connection.execute(
                """
                SELECT a.stage_id,m.state,m.lease_expires_at
                FROM assignments a JOIN miners m ON m.session_id=a.session_id
                WHERE a.run_id=? AND a.epoch_id=? AND a.state='active'
                """,
                (self.run_id, int(active_epoch)),
            ).fetchall()
            if len(assignments) != int(job["stage_count"]) or any(
                str(row["state"]) != "online"
                or float(row["lease_expires_at"]) <= now
                for row in assignments
            ):
                self._abort_epoch(
                    connection,
                    epoch_id=int(active_epoch),
                    reason="assigned_miner_unavailable",
                    now=now,
                )
                connection.execute(
                    "UPDATE jobs SET active_epoch_id=NULL WHERE run_id=?",
                    (self.run_id,),
                )
                job = self._job(connection)
        if int(job["committed_step"]) >= int(job["target_steps"]):
            connection.execute(
                """
                UPDATE jobs SET state='completed',active_epoch_id=NULL,
                    revision=revision+1,updated_at=? WHERE run_id=? AND state!='completed'
                """,
                (now, self.run_id),
            )
            return
        if job["active_epoch_id"] is not None:
            connection.execute(
                "UPDATE jobs SET state='running',updated_at=? WHERE run_id=?",
                (now, self.run_id),
            )
            return
        miner_rows = connection.execute(
            """
            SELECT * FROM miners WHERE run_id=? AND state='online' AND lease_expires_at>?
            ORDER BY registered_at,session_id
            """,
            (self.run_id, now),
        ).fetchall()
        stage_ids = [
            int(item["stage_id"])
            for item in json.loads(str(job["stage_specs_json"]))
        ]
        mapping, placement_plan = self._heterogeneous_stage_mapping(
            connection, list(miner_rows), stage_ids, now=now
        )
        if mapping is None:
            previous = str(job["state"])
            connection.execute(
                """
                UPDATE jobs SET state='paused_waiting_for_miners',active_epoch_id=NULL,
                    revision=revision+1,updated_at=?
                WHERE run_id=? AND state!='paused_waiting_for_miners'
                """,
                (now, self.run_id),
            )
            if previous != "paused_waiting_for_miners":
                self._event(
                    connection,
                    run_id=self.run_id,
                    operation="training_paused",
                    value={
                        "committed_step": int(job["committed_step"]),
                        "reason": "incomplete_stage_coverage",
                    },
                    now=now,
                )
            return
        epoch_id = int(job["next_epoch_id"])
        base_step = int(job["committed_step"])
        target_step = base_step + 1
        dataset_cursor = target_step * int(job["microbatches_per_step"])
        connection.execute(
            """
            INSERT INTO epochs(
                run_id,epoch_id,base_step,target_step,dataset_cursor,state,created_at
            ) VALUES(?,?,?,?,?,'active',?)
            """,
            (self.run_id, epoch_id, base_step, target_step, dataset_cursor, now),
        )
        for stage_id in sorted(mapping):
            token = secrets.token_urlsafe(32)
            placement = next(
                (
                    item
                    for item in (placement_plan or {}).get("assignments") or []
                    if int(item["stage_id"]) == int(stage_id)
                ),
                {},
            )
            connection.execute(
                """
                INSERT INTO assignments(
                    run_id,epoch_id,stage_id,session_id,assignment_token,
                    assignment_token_hash,state,created_at,placement_generation,
                    device_id,device_type
                ) VALUES(?,?,?,?,?,?,'active',?,?,?,?)
                """,
                (
                    self.run_id,
                    epoch_id,
                    int(stage_id),
                    mapping[stage_id],
                    token,
                    _token_hash(token),
                    now,
                    int((placement_plan or {}).get("placement_generation") or 0),
                    str(placement.get("device_id") or ""),
                    str(placement.get("device_type") or ""),
                ),
            )
        previous = str(job["state"])
        connection.execute(
            """
            UPDATE jobs SET state='running',active_epoch_id=?,next_epoch_id=?,
                revision=revision+1,updated_at=? WHERE run_id=?
            """,
            (epoch_id, epoch_id + 1, now, self.run_id),
        )
        if placement_plan is not None:
            connection.execute(
                """
                UPDATE jobs SET placement_generation=?,placement_plan_json=?,
                    placement_error_json='{}',pending_rebalance_reason=''
                WHERE run_id=?
                """,
                (
                    int(placement_plan["placement_generation"]),
                    json.dumps(placement_plan, sort_keys=True),
                    self.run_id,
                ),
            )
        self._event(
            connection,
            run_id=self.run_id,
            operation=("training_auto_woke" if previous == "paused_waiting_for_miners" else "barrier_epoch_started"),
            value={
                "epoch_id": epoch_id,
                "base_step": base_step,
                "target_step": target_step,
                "stage_count": len(stage_ids),
                "complete_stage_coverage": True,
            },
            now=now,
        )

    def _require_session(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        session_token: str,
        allow_inactive: bool = False,
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM miners WHERE run_id=? AND session_id=?",
            (self.run_id, str(session_id)),
        ).fetchone()
        if row is None or not hmac.compare_digest(
            str(row["session_token"]), str(session_token)
        ):
            raise ValueError("elastic_miner_session_invalid")
        if not allow_inactive and str(row["state"]) != "online":
            raise ValueError("elastic_miner_session_stale")
        return row

    def _assignment_response(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
    ) -> dict[str, Any]:
        job = self._job(connection)
        rows = connection.execute(
            """
            SELECT a.*,e.base_step,e.target_step,e.dataset_cursor,e.state AS epoch_state
            FROM assignments a JOIN epochs e
              ON e.run_id=a.run_id AND e.epoch_id=a.epoch_id
            WHERE a.run_id=? AND a.session_id=? AND a.state='active'
              AND e.state='active'
            ORDER BY a.stage_id
            """,
            (self.run_id, session_id),
        ).fetchall()
        assignments = []
        stage_specs = {
            int(item["stage_id"]): dict(item)
            for item in json.loads(str(job["stage_specs_json"]))
        }
        for row in rows:
            checkpoint_hash = ""
            if int(row["base_step"]) > 0:
                checkpoint = connection.execute(
                    """
                    SELECT s.archive_hash FROM commits c JOIN submissions s
                      ON s.run_id=c.run_id AND s.epoch_id=c.epoch_id
                    WHERE c.run_id=? AND c.target_step=? AND s.stage_id=?
                    """,
                    (self.run_id, int(row["base_step"]), int(row["stage_id"])),
                ).fetchone()
                if checkpoint is None:
                    raise RuntimeError("elastic_committed_checkpoint_missing")
                checkpoint_hash = str(checkpoint["archive_hash"])
            assignments.append(
                {
                    "epoch_id": int(row["epoch_id"]),
                    "stage_id": int(row["stage_id"]),
                    "assignment_token": str(row["assignment_token"]),
                    "assignment_token_hash": str(row["assignment_token_hash"]),
                    "base_step": int(row["base_step"]),
                    "base_dataset_cursor": int(row["base_step"])
                    * self.microbatches_per_step,
                    "target_step": int(row["target_step"]),
                    "dataset_cursor": int(row["dataset_cursor"]),
                    "restore_required": int(row["base_step"]) > 0,
                    "committed_checkpoint_archive_hash": checkpoint_hash,
                    "placement_generation": int(row["placement_generation"]),
                    "device_id": str(row["device_id"]),
                    "device_type": str(row["device_type"]),
                    "stage_spec": stage_specs[int(row["stage_id"])],
                }
            )
        return {
            "schema": RUNTIME_SCHEMA,
            "run_id": self.run_id,
            "session_id": session_id,
            "runtime_state": str(job["state"]),
            "committed_step": int(job["committed_step"]),
            "target_steps": int(job["target_steps"]),
            "placement_generation": int(job["placement_generation"]),
            "assignments": assignments,
            "assignment_tokens_public": False,
            "session_token_public": False,
            "public_artifact": False,
        }

    def register_miner(
        self,
        *,
        miner_id_hash: str,
        registration_nonce: str,
        supported_stage_ids: list[int],
        slot_count: int,
        accelerator: str = "cuda",
        capability: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = float(self._clock())
        miner_hash = str(miner_id_hash)
        nonce = str(registration_nonce)
        stage_ids = sorted({int(value) for value in supported_stage_ids})
        valid_stage_ids = {
            int(spec["stage_id"])
            for spec in (
                self.training_manifest["stages"]
                if self.training_manifest is not None
                else [item.public_dict() for item in canonical_stage_specs()]
            )
        }
        canonical_capability: dict[str, Any] = {}
        if capability is not None:
            canonical_capability = validate_miner_capability(capability)
            if canonical_capability["miner_id_hash"] != miner_hash:
                raise ValueError("elastic_miner_capability_identity_mismatch")
        if (
            not miner_hash.startswith("sha256:")
            or not nonce
            or not stage_ids
            or not set(stage_ids).issubset(valid_stage_ids)
            or int(slot_count) < 1
            or int(slot_count) > len(stage_ids)
            or str(accelerator) not in {"cpu", "cuda", "tpu", "mixed"}
            or (self.training_manifest is not None and not canonical_capability)
        ):
            raise ValueError("elastic_miner_registration_invalid")
        registration_hash = _token_hash(nonce)
        with self._transaction() as connection:
            self._reconcile(connection, now=now)
            previous = connection.execute(
                """
                SELECT * FROM miners WHERE run_id=? AND registration_hash=?
                  AND state='online' AND lease_expires_at>?
                ORDER BY registered_at DESC LIMIT 1
                """,
                (self.run_id, registration_hash, now),
            ).fetchone()
            if previous is not None:
                if (
                    str(previous["miner_id_hash"]) != miner_hash
                    or json.loads(str(previous["supported_stage_ids_json"])) != stage_ids
                    or int(previous["slot_count"]) != int(slot_count)
                    or str(previous["accelerator"]) != str(accelerator)
                ):
                    raise ValueError("elastic_miner_registration_conflict")
                session_id = str(previous["session_id"])
                previous_capability = json.loads(
                    str(previous["capability_json"] or "{}")
                )
                capability_refreshed = previous_capability != canonical_capability
                connection.execute(
                    """
                    UPDATE miners SET heartbeat_at=?,lease_expires_at=?,capability_json=?
                    WHERE session_id=?
                    """,
                    (
                        now,
                        now + self.lease_seconds,
                        json.dumps(canonical_capability, sort_keys=True),
                        session_id,
                    ),
                )
                if capability_refreshed:
                    self._event(
                        connection,
                        run_id=self.run_id,
                        operation="miner_capability_refreshed",
                        value={
                            "miner_id_hash": miner_hash,
                            "miner_session_hash": _token_hash(session_id),
                            "generation": int(previous["generation"]),
                            "previous_capability_hash": str(
                                previous_capability.get("content_hash") or ""
                            ),
                            "capability_hash": str(
                                canonical_capability.get("content_hash") or ""
                            ),
                        },
                        now=now,
                    )
                self._reconcile(connection, now=now)
                response = self._assignment_response(connection, session_id=session_id)
                response.update(
                    {
                        "session_token": str(previous["session_token"]),
                        "lease_expires_at": now + self.lease_seconds,
                        "registration_idempotent": True,
                        "capability_refreshed": capability_refreshed,
                        "checkpoint_signatures_required": self.require_checkpoint_signatures,
                        "checkpoint_tensor_validation_required": self.validate_checkpoint_tensors,
                    }
                )
                return response
            online_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM miners
                    WHERE run_id=? AND state='online' AND lease_expires_at>?
                    """,
                    (self.run_id, now),
                ).fetchone()[0]
            )
            if online_count >= self.max_online_miners:
                raise RuntimeError("elastic_online_miner_quota_exceeded")
            generation = int(
                connection.execute(
                    "SELECT COUNT(*) FROM miners WHERE run_id=? AND miner_id_hash=?",
                    (self.run_id, miner_hash),
                ).fetchone()[0]
            ) + 1
            session_id = "elastic-miner-" + secrets.token_hex(12)
            session_token = secrets.token_urlsafe(32)
            connection.execute(
                """
                INSERT INTO miners(
                    session_id,run_id,miner_id_hash,registration_hash,session_token,
                    session_token_hash,supported_stage_ids_json,slot_count,accelerator,
                    state,generation,registered_at,heartbeat_at,lease_expires_at,
                    capability_json
                ) VALUES(?,?,?,?,?,?,?,?,?,'online',?,?,?,?,?)
                """,
                (
                    session_id,
                    self.run_id,
                    miner_hash,
                    registration_hash,
                    session_token,
                    _token_hash(session_token),
                    json.dumps(stage_ids),
                    int(slot_count),
                    str(accelerator),
                    generation,
                    now,
                    now,
                    now + self.lease_seconds,
                    json.dumps(canonical_capability, sort_keys=True),
                ),
            )
            self._event(
                connection,
                run_id=self.run_id,
                operation="miner_joined",
                value={
                    "miner_id_hash": miner_hash,
                    "miner_session_hash": _token_hash(session_id),
                    "generation": generation,
                    "supported_stage_ids": stage_ids,
                    "slot_count": int(slot_count),
                    "accelerator": str(accelerator),
                    "capability_hash": str(
                        canonical_capability.get("content_hash") or ""
                    ),
                },
                now=now,
            )
            if self.training_manifest is not None:
                connection.execute(
                    "UPDATE jobs SET pending_rebalance_reason='miner_joined' WHERE run_id=?",
                    (self.run_id,),
                )
            self._reconcile(connection, now=now)
            response = self._assignment_response(connection, session_id=session_id)
            response.update(
                {
                    "session_token": session_token,
                    "lease_expires_at": now + self.lease_seconds,
                    "registration_idempotent": False,
                    "checkpoint_signatures_required": self.require_checkpoint_signatures,
                    "checkpoint_tensor_validation_required": self.validate_checkpoint_tensors,
                }
            )
            return response

    def heartbeat(self, *, session_id: str, session_token: str) -> dict[str, Any]:
        now = float(self._clock())
        with self._transaction() as connection:
            self._reconcile(connection, now=now)
            self._require_session(
                connection, session_id=session_id, session_token=session_token
            )
            connection.execute(
                "UPDATE miners SET heartbeat_at=?,lease_expires_at=? WHERE session_id=?",
                (now, now + self.lease_seconds, session_id),
            )
            self._reconcile(connection, now=now)
            response = self._assignment_response(connection, session_id=session_id)
            response["lease_expires_at"] = now + self.lease_seconds
            return response

    def assignments(self, *, session_id: str, session_token: str) -> dict[str, Any]:
        now = float(self._clock())
        with self._transaction() as connection:
            self._reconcile(connection, now=now)
            row = self._require_session(
                connection, session_id=session_id, session_token=session_token
            )
            if float(row["lease_expires_at"]) <= now:
                raise ValueError("elastic_miner_session_stale")
            return self._assignment_response(connection, session_id=session_id)

    def mark_offline(self, *, session_id: str, session_token: str) -> dict[str, Any]:
        now = float(self._clock())
        with self._transaction() as connection:
            row = self._require_session(
                connection,
                session_id=session_id,
                session_token=session_token,
                allow_inactive=True,
            )
            changed = str(row["state"]) == "online"
            if changed:
                connection.execute(
                    """
                    UPDATE miners SET state='offline',offline_at=?,lease_expires_at=?
                    WHERE session_id=?
                    """,
                    (now, now, session_id),
                )
                self._event(
                    connection,
                    run_id=self.run_id,
                    operation="miner_left",
                    value={
                        "miner_id_hash": str(row["miner_id_hash"]),
                        "miner_session_hash": _token_hash(session_id),
                    },
                    now=now,
                )
                if self.training_manifest is not None:
                    connection.execute(
                        "UPDATE jobs SET pending_rebalance_reason='miner_left' WHERE run_id=?",
                        (self.run_id,),
                    )
            self._reconcile(connection, now=now)
            job = self._job(connection)
            return {
                "schema": RUNTIME_SCHEMA,
                "ok": True,
                "offline_transition_applied": changed,
                "runtime_state": str(job["state"]),
                "committed_step": int(job["committed_step"]),
                "session_token_public": False,
                "public_artifact_safe": True,
            }

    def tick(self) -> dict[str, Any]:
        now = float(self._clock())
        with self._transaction() as connection:
            self._reconcile(connection, now=now)
        return self.public_status(reconcile=False)

    def record_coordinator_start(self, *, instance_id_hash: str) -> dict[str, Any]:
        """Persist a service-instance generation without exposing its identity."""

        identity = str(instance_id_hash)
        if not identity.startswith("sha256:"):
            raise ValueError("elastic_coordinator_instance_identity_invalid")
        now = float(self._clock())
        with self._transaction() as connection:
            job = self._job(connection)
            generation = int(job["coordinator_generation"]) + 1
            connection.execute(
                """
                UPDATE jobs SET coordinator_generation=?,coordinator_started_at=?,
                    revision=revision+1,updated_at=? WHERE run_id=?
                """,
                (generation, now, now, self.run_id),
            )
            self._event(
                connection,
                run_id=self.run_id,
                operation="coordinator_started",
                value={
                    "coordinator_generation": generation,
                    "instance_id_hash": identity,
                    "persistent_journal_reopened": generation > 1,
                },
                now=now,
            )
            self._reconcile(connection, now=now)
        return {
            "schema": RUNTIME_SCHEMA,
            "ok": True,
            "coordinator_generation": generation,
            "persistent_journal_reopened": generation > 1,
            "instance_id_public": False,
            "public_artifact_safe": True,
        }

    def pause(self, *, reason: str = "owner_paused") -> dict[str, Any]:
        """Idempotently stop issuing epochs while preserving Miners and state."""

        now = float(self._clock())
        with self._transaction() as connection:
            job = self._job(connection)
            if str(job["state"]) in {"cancelled", "cleaned", "failed"}:
                raise RuntimeError("elastic_training_pause_terminal_state")
            changed = not bool(job["owner_paused"]) and str(job["state"]) != "completed"
            if changed and job["active_epoch_id"] is not None:
                self._abort_epoch(
                    connection,
                    epoch_id=int(job["active_epoch_id"]),
                    reason=str(reason),
                    now=now,
                )
            if changed:
                connection.execute(
                    """
                    UPDATE jobs SET owner_paused=1,state='paused_by_owner',
                        active_epoch_id=NULL,revision=revision+1,updated_at=?
                    WHERE run_id=?
                    """,
                    (now, self.run_id),
                )
                self._event(
                    connection,
                    run_id=self.run_id,
                    operation="training_owner_paused",
                    value={
                        "reason": str(reason),
                        "committed_step": int(job["committed_step"]),
                    },
                    now=now,
                )
        return {
            **self.public_status(reconcile=False),
            "pause_transition_applied": changed,
            "command_ok": True,
        }

    def resume(self, *, reason: str = "owner_resumed") -> dict[str, Any]:
        """Idempotently clear an owner pause and reconcile from durable state."""

        now = float(self._clock())
        with self._transaction() as connection:
            job = self._job(connection)
            if str(job["state"]) in {"cancelled", "cleaned", "failed"}:
                raise RuntimeError("elastic_training_resume_terminal_state")
            changed = bool(job["owner_paused"])
            if changed:
                connection.execute(
                    """
                    UPDATE jobs SET owner_paused=0,state='paused_waiting_for_miners',
                        pending_rebalance_reason='checkpoint_recovery',
                        revision=revision+1,updated_at=? WHERE run_id=?
                    """,
                    (now, self.run_id),
                )
                self._event(
                    connection,
                    run_id=self.run_id,
                    operation="training_owner_resumed",
                    value={
                        "reason": str(reason),
                        "committed_step": int(job["committed_step"]),
                    },
                    now=now,
                )
                self._reconcile(connection, now=now)
        return {
            **self.public_status(reconcile=False),
            "resume_transition_applied": changed,
            "command_ok": True,
        }

    def request_rebalance(self, *, reason: str = "owner_requested") -> dict[str, Any]:
        allowed = {
            "owner_requested",
            "performance_rebalance",
            "health_degraded",
            "coordinator_recovery",
        }
        if str(reason) not in allowed:
            raise ValueError("elastic_rebalance_reason_invalid")
        now = float(self._clock())
        with self._transaction() as connection:
            job = self._job(connection)
            if str(job["state"]) in {"cancelled", "completed", "cleaned", "failed"}:
                return {
                    **self.public_status(reconcile=False),
                    "rebalance_transition_applied": False,
                    "command_ok": True,
                }
            if job["active_epoch_id"] is not None:
                self._abort_epoch(
                    connection,
                    epoch_id=int(job["active_epoch_id"]),
                    reason=str(reason),
                    now=now,
                )
            connection.execute(
                """
                UPDATE jobs SET active_epoch_id=NULL,pending_rebalance_reason=?,
                    revision=revision+1,updated_at=? WHERE run_id=?
                """,
                (str(reason), now, self.run_id),
            )
            self._event(
                connection,
                run_id=self.run_id,
                operation="placement_rebalance_requested",
                value={
                    "reason": str(reason),
                    "previous_placement_generation": int(
                        job["placement_generation"]
                    ),
                },
                now=now,
            )
            self._reconcile(connection, now=now)
        return {
            **self.public_status(reconcile=False),
            "rebalance_transition_applied": True,
            "command_ok": True,
        }

    def report_device_telemetry(
        self,
        *,
        session_id: str,
        session_token: str,
        device_id: str,
        free_memory_bytes: int,
        utilization_fraction: float,
        throughput_units_per_second: float,
        network_bandwidth_bytes_per_second: float = 0.0,
        network_latency_ms: float = 0.0,
        checkpoint_step: int = 0,
        health_score: float = 1.0,
    ) -> dict[str, Any]:
        """Store bounded, public-safe resource telemetry used by placement."""

        identifier = str(device_id)
        values = (
            float(utilization_fraction),
            float(throughput_units_per_second),
            float(network_bandwidth_bytes_per_second),
            float(network_latency_ms),
            float(health_score),
        )
        if (
            not identifier
            or int(free_memory_bytes) < 0
            or any(not math.isfinite(item) or item < 0 for item in values)
            or float(utilization_fraction) > 1.0
            or float(health_score) > 1.0
            or int(checkpoint_step) < 0
        ):
            raise ValueError("elastic_device_telemetry_invalid")
        now = float(self._clock())
        with self._transaction() as connection:
            session = self._require_session(
                connection,
                session_id=session_id,
                session_token=session_token,
            )
            capability = json.loads(str(session["capability_json"] or "{}"))
            device_ids = {"cpu"} if capability.get("cpu_stage_supported") else set()
            device_ids.update(
                str(item.get("device_id") or "")
                for item in capability.get("gpus") or []
            )
            device_ids.update(
                str(item.get("device_id") or "")
                for item in capability.get("tpu_groups") or []
            )
            if identifier not in device_ids:
                raise ValueError("elastic_device_telemetry_device_mismatch")
            job = self._job(connection)
            if int(checkpoint_step) > int(job["committed_step"]):
                raise ValueError("elastic_device_telemetry_checkpoint_ahead")
            telemetry = {
                "free_memory_bytes": int(free_memory_bytes),
                "utilization_fraction": float(utilization_fraction),
                "throughput_units_per_second": float(
                    throughput_units_per_second
                ),
                "network_bandwidth_bytes_per_second": float(
                    network_bandwidth_bytes_per_second
                ),
                "network_latency_ms": float(network_latency_ms),
                "health_score": float(health_score),
                "reported_at": now,
                "public_artifact_safe": True,
            }
            previous = connection.execute(
                """
                SELECT * FROM device_health
                WHERE run_id=? AND session_id=? AND device_id=?
                """,
                (self.run_id, str(session_id), identifier),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO device_health(
                    run_id,session_id,device_id,state,consecutive_failures,
                    total_failures,quarantine_until,last_success_at,last_failure_at,
                    last_failure_class,checkpoint_step,telemetry_json,updated_at
                ) VALUES(?,?,?,'healthy',0,0,0,?,0,'',?,?,?)
                ON CONFLICT(run_id,session_id,device_id) DO UPDATE SET
                    state='healthy',consecutive_failures=0,quarantine_until=0,
                    last_success_at=excluded.last_success_at,
                    checkpoint_step=excluded.checkpoint_step,
                    telemetry_json=excluded.telemetry_json,updated_at=excluded.updated_at
                """,
                (
                    self.run_id,
                    str(session_id),
                    identifier,
                    now,
                    int(checkpoint_step),
                    json.dumps(telemetry, sort_keys=True),
                    now,
                ),
            )
            materially_changed = previous is None
            if previous is not None:
                try:
                    old = json.loads(str(previous["telemetry_json"] or "{}"))
                except json.JSONDecodeError:
                    old = {}
                materially_changed = (
                    abs(float(old.get("utilization_fraction") or 0.0) - float(utilization_fraction)) >= 0.1
                    or abs(float(old.get("health_score", 1.0)) - float(health_score)) >= 0.1
                    or int(previous["checkpoint_step"]) != int(checkpoint_step)
                )
            if materially_changed:
                self._event(
                    connection,
                    run_id=self.run_id,
                    operation="device_telemetry_updated",
                    value={
                        "miner_session_hash": _token_hash(str(session_id)),
                        "device_id": identifier,
                        "health_score": float(health_score),
                        "checkpoint_step": int(checkpoint_step),
                        "free_memory_bytes": int(free_memory_bytes),
                        "utilization_fraction": float(utilization_fraction),
                    },
                    now=now,
                )
            if float(health_score) < 0.25 and job["active_epoch_id"] is not None:
                self._abort_epoch(
                    connection,
                    epoch_id=int(job["active_epoch_id"]),
                    reason="health_degraded",
                    now=now,
                )
                connection.execute(
                    """
                    UPDATE jobs SET active_epoch_id=NULL,
                        pending_rebalance_reason='health_degraded' WHERE run_id=?
                    """,
                    (self.run_id,),
                )
                self._reconcile(connection, now=now)
        return {
            "schema": "crowdtensor_elastic_device_telemetry_v1",
            "ok": True,
            "telemetry_accepted": True,
            "device_id": identifier,
            "checkpoint_step": int(checkpoint_step),
            "health_score": float(health_score),
            "miner_identity_public": False,
            "public_artifact_safe": True,
        }

    def record_device_failure(
        self,
        *,
        session_id: str,
        session_token: str,
        device_id: str,
        failure_class: str,
        quarantine_threshold: int = 3,
        quarantine_seconds: float = 300.0,
    ) -> dict[str, Any]:
        """Apply a persistent circuit breaker to a repeatedly failing device."""

        supported = {
            "network_timeout",
            "worker_crash",
            "device_oom",
            "straggler",
            "checkpoint_corrupt",
            "runtime_error",
        }
        failure = str(failure_class)
        if (
            failure not in supported
            or int(quarantine_threshold) < 1
            or not 1.0 <= float(quarantine_seconds) <= 86400.0
        ):
            raise ValueError("elastic_device_failure_policy_invalid")
        now = float(self._clock())
        with self._transaction() as connection:
            session = self._require_session(
                connection,
                session_id=session_id,
                session_token=session_token,
                allow_inactive=True,
            )
            previous = connection.execute(
                """
                SELECT * FROM device_health
                WHERE run_id=? AND session_id=? AND device_id=?
                """,
                (self.run_id, str(session_id), str(device_id)),
            ).fetchone()
            consecutive = int(previous["consecutive_failures"] if previous else 0) + 1
            total = int(previous["total_failures"] if previous else 0) + 1
            quarantined = consecutive >= int(quarantine_threshold)
            until = now + float(quarantine_seconds) if quarantined else 0.0
            telemetry_json = str(previous["telemetry_json"] if previous else "{}")
            checkpoint_step = int(previous["checkpoint_step"] if previous else 0)
            connection.execute(
                """
                INSERT INTO device_health(
                    run_id,session_id,device_id,state,consecutive_failures,
                    total_failures,quarantine_until,last_success_at,last_failure_at,
                    last_failure_class,checkpoint_step,telemetry_json,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id,session_id,device_id) DO UPDATE SET
                    state=excluded.state,
                    consecutive_failures=excluded.consecutive_failures,
                    total_failures=excluded.total_failures,
                    quarantine_until=excluded.quarantine_until,
                    last_failure_at=excluded.last_failure_at,
                    last_failure_class=excluded.last_failure_class,
                    updated_at=excluded.updated_at
                """,
                (
                    self.run_id,
                    str(session_id),
                    str(device_id),
                    "quarantined" if quarantined else "degraded",
                    consecutive,
                    total,
                    until,
                    float(previous["last_success_at"] if previous else 0.0),
                    now,
                    failure,
                    checkpoint_step,
                    telemetry_json,
                    now,
                ),
            )
            self._event(
                connection,
                run_id=self.run_id,
                operation=("device_quarantined" if quarantined else "device_failure_recorded"),
                value={
                    "miner_session_hash": _token_hash(str(session_id)),
                    "device_id": str(device_id),
                    "failure_class": failure,
                    "consecutive_failures": consecutive,
                    "quarantine_seconds": float(quarantine_seconds) if quarantined else 0.0,
                },
                now=now,
            )
            if quarantined and str(session["state"]) == "online":
                connection.execute(
                    """
                    UPDATE miners SET state='quarantined',offline_at=?,
                        lease_expires_at=?,last_failure_reason=? WHERE session_id=?
                    """,
                    (now, now, failure, str(session_id)),
                )
                job = self._job(connection)
                if job["active_epoch_id"] is not None:
                    self._abort_epoch(
                        connection,
                        epoch_id=int(job["active_epoch_id"]),
                        reason="health_degraded",
                        now=now,
                    )
                connection.execute(
                    """
                    UPDATE jobs SET active_epoch_id=NULL,
                        pending_rebalance_reason='health_degraded' WHERE run_id=?
                    """,
                    (self.run_id,),
                )
                self._reconcile(connection, now=now)
        return {
            "schema": "crowdtensor_elastic_device_failure_v1",
            "ok": True,
            "failure_class": failure,
            "consecutive_failures": consecutive,
            "device_quarantined": quarantined,
            "quarantine_until": until,
            "session_identity_public": False,
            "public_artifact_safe": True,
        }

    def capabilities(self) -> dict[str, Any]:
        specs = (
            [dict(spec) for spec in self.training_manifest["stages"]]
            if self.training_manifest is not None
            else [spec.public_dict() for spec in canonical_stage_specs()]
        )
        model = dict((self.training_manifest or {}).get("model") or {})
        return {
            "schema": "crowdtensor_elastic_training_capabilities_v1",
            "run_id": self.run_id,
            "model_id": str(model.get("model_id") or self.legacy_model_id),
            "model_revision": str(
                model.get("model_revision") or self.legacy_model_revision
            ),
            "target_steps": self.target_steps,
            "microbatches_per_step": self.microbatches_per_step,
            "stage_specs": specs,
            "stage_groups": (
                [] if self.training_manifest is not None else [[0, 1], [2, 3]]
            ),
            "minimum_cuda_devices_per_miner": (
                1 if self.training_manifest is not None else 2
            ),
            "single_gpu_miner_supported": self.training_manifest is not None,
            "cpu_trainable_stages_supported": self.training_manifest is not None,
            "heterogeneous_scheduler_enabled": self.training_manifest is not None,
            "training_manifest": self.training_manifest,
            "automatic_role_assignment": True,
            "tensor_lookup_optimization_after_step": self.tensor_lookup_optimization_after_step,
            "indexed_tensor_lookup_supported": True,
            "checkpoint_signatures_required": self.require_checkpoint_signatures,
            "checkpoint_tensor_validation_required": self.validate_checkpoint_tensors,
            "checkpoint_storage": self.blob_store.public_report(),
            "private_paths_public": False,
            "credential_values_public": False,
            "public_artifact_safe": True,
        }

    def _require_stage_assignment(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        session_token: str,
        assignment_token: str,
        stage_id: int,
        placement_generation: int,
    ) -> sqlite3.Row:
        self._require_session(
            connection,
            session_id=session_id,
            session_token=session_token,
        )
        row = connection.execute(
            """
            SELECT a.*,e.target_step,e.state AS epoch_state
            FROM assignments a JOIN epochs e
              ON e.run_id=a.run_id AND e.epoch_id=a.epoch_id
            WHERE a.run_id=? AND a.session_id=? AND a.stage_id=?
              AND a.state='active' AND e.state='active'
            ORDER BY a.epoch_id DESC LIMIT 1
            """,
            (self.run_id, str(session_id), int(stage_id)),
        ).fetchone()
        if row is None:
            raise ValueError("elastic_stage_assignment_stale")
        if not hmac.compare_digest(
            str(row["assignment_token"]), str(assignment_token)
        ):
            historical = connection.execute(
                """
                SELECT a.state,e.state AS epoch_state
                FROM assignments a JOIN epochs e
                  ON e.run_id=a.run_id AND e.epoch_id=a.epoch_id
                WHERE a.run_id=? AND a.session_id=? AND a.stage_id=?
                  AND a.assignment_token=?
                ORDER BY a.epoch_id DESC LIMIT 1
                """,
                (
                    self.run_id,
                    str(session_id),
                    int(stage_id),
                    str(assignment_token),
                ),
            ).fetchone()
            if historical is not None:
                raise ValueError("elastic_stage_assignment_stale")
            raise ValueError("elastic_stage_assignment_token_invalid")
        if int(row["placement_generation"]) != int(placement_generation):
            raise ValueError("elastic_stage_placement_generation_stale")
        return row

    def report_stage_runtime(
        self,
        *,
        session_id: str,
        session_token: str,
        assignment_token: str,
        placement_generation: int,
        stage_id: int,
        device_id: str,
        event_type: str,
        forward_latency_ms: float = 0.0,
        backward_latency_ms: float = 0.0,
        peak_memory_bytes: int = 0,
        sample_count: int = 1,
        compile_latency_ms: float = 0.0,
        steady_forward_latency_ms: float = 0.0,
        steady_backward_latency_ms: float = 0.0,
    ) -> dict[str, Any]:
        """Accept measured stage performance or trigger a fenced rebalance."""

        if self.training_manifest is None:
            raise ValueError("elastic_heterogeneous_scheduler_not_enabled")
        kind = str(event_type)
        if kind not in {"profile", "oom", "straggler"}:
            raise ValueError("elastic_stage_runtime_event_invalid")
        now = float(self._clock())
        with self._transaction() as connection:
            assignment = self._require_stage_assignment(
                connection,
                session_id=session_id,
                session_token=session_token,
                assignment_token=assignment_token,
                stage_id=stage_id,
                placement_generation=placement_generation,
            )
            if str(assignment["device_id"]) != str(device_id):
                raise ValueError("elastic_stage_runtime_device_mismatch")
            miner = connection.execute(
                "SELECT * FROM miners WHERE session_id=?", (str(session_id),)
            ).fetchone()
            if miner is None:
                raise ValueError("elastic_miner_session_invalid")
            capability = json.loads(str(miner["capability_json"] or "{}"))
            capability.pop("content_hash", None)
            profiles = [
                dict(item)
                for item in capability.get("stage_profiles") or []
                if not (
                    int(item.get("stage_id", -1)) == int(stage_id)
                    and str(item.get("device_id") or "") == str(device_id)
                )
            ]
            if kind == "profile":
                if (
                    int(sample_count) < 1
                    or float(forward_latency_ms) + float(backward_latency_ms) <= 0
                    or not math.isfinite(float(forward_latency_ms))
                    or not math.isfinite(float(backward_latency_ms))
                ):
                    raise ValueError("elastic_stage_runtime_profile_invalid")
                profile = {
                    "stage_id": int(stage_id),
                    "device_id": str(device_id),
                    "forward_latency_ms": float(forward_latency_ms),
                    "backward_latency_ms": float(backward_latency_ms),
                    "peak_memory_bytes": int(peak_memory_bytes),
                    "sample_count": int(sample_count),
                    "measured_at": now,
                }
                if capability.get("schema") == "crowdtensor_heterogeneous_miner_capability_v2":
                    profile.update(
                        {
                            "compile_latency_ms": float(compile_latency_ms),
                            "steady_forward_latency_ms": float(
                                steady_forward_latency_ms or forward_latency_ms
                            ),
                            "steady_backward_latency_ms": float(
                                steady_backward_latency_ms or backward_latency_ms
                            ),
                        }
                    )
                profiles.append(profile)
                capability["stage_profiles"] = profiles
                canonical_capability = validate_miner_capability(capability)
                connection.execute(
                    """
                    UPDATE miners SET capability_json=?,stage_metrics_json=?,
                        heartbeat_at=?,lease_expires_at=? WHERE session_id=?
                    """,
                    (
                        json.dumps(canonical_capability, sort_keys=True),
                        json.dumps(profiles, sort_keys=True),
                        now,
                        now + self.lease_seconds,
                        str(session_id),
                    ),
                )
                self._event(
                    connection,
                    run_id=self.run_id,
                    operation="stage_profile_updated",
                    value={
                        "stage_id": int(stage_id),
                        "device_id": str(device_id),
                        "placement_generation": int(placement_generation),
                        "forward_latency_ms": float(forward_latency_ms),
                        "backward_latency_ms": float(backward_latency_ms),
                        "peak_memory_bytes": int(peak_memory_bytes),
                        "sample_count": int(sample_count),
                        "compile_latency_ms": float(compile_latency_ms),
                        "steady_forward_latency_ms": float(
                            steady_forward_latency_ms or forward_latency_ms
                        ),
                        "steady_backward_latency_ms": float(
                            steady_backward_latency_ms or backward_latency_ms
                        ),
                    },
                    now=now,
                )
                return {
                    "schema": RUNTIME_SCHEMA,
                    "ok": True,
                    "profile_accepted": True,
                    "rebalance_triggered": False,
                    "placement_generation": int(placement_generation),
                    "public_artifact_safe": True,
                }

            if str(device_id) == "cpu":
                capability["cpu"]["free_memory_bytes"] = 0
            elif str(device_id).startswith("cuda:"):
                found = False
                for gpu in capability.get("gpus") or []:
                    if str(gpu.get("device_id") or "") == str(device_id):
                        gpu["free_memory_bytes"] = 0
                        found = True
                if not found:
                    raise ValueError("elastic_stage_runtime_device_mismatch")
            elif str(device_id).startswith("jax_tpu:"):
                found = False
                for group in capability.get("tpu_groups") or []:
                    if str(group.get("device_id") or "") == str(device_id):
                        group["free_hbm_bytes"] = 0
                        found = True
                if not found:
                    raise ValueError("elastic_stage_runtime_device_mismatch")
            else:
                raise ValueError("elastic_stage_runtime_device_mismatch")
            capability["stage_profiles"] = profiles
            canonical_capability = validate_miner_capability(capability)
            reason = "device_oom" if kind == "oom" else "straggler_detected"
            connection.execute(
                """
                UPDATE miners SET capability_json=?,last_failure_reason=?
                WHERE session_id=?
                """,
                (
                    json.dumps(canonical_capability, sort_keys=True),
                    reason,
                    str(session_id),
                ),
            )
            self._abort_epoch(
                connection,
                epoch_id=int(assignment["epoch_id"]),
                reason=reason,
                now=now,
            )
            connection.execute(
                """
                UPDATE jobs SET active_epoch_id=NULL,pending_rebalance_reason=?
                WHERE run_id=?
                """,
                (reason, self.run_id),
            )
            self._event(
                connection,
                run_id=self.run_id,
                operation="stage_rebalance_requested",
                value={
                    "stage_id": int(stage_id),
                    "device_id": str(device_id),
                    "placement_generation": int(placement_generation),
                    "reason": reason,
                },
                now=now,
            )
            self._reconcile(connection, now=now)
            job = self._job(connection)
            return {
                "schema": RUNTIME_SCHEMA,
                "ok": True,
                "profile_accepted": False,
                "rebalance_triggered": True,
                "reason": reason,
                "previous_placement_generation": int(placement_generation),
                "placement_generation": int(job["placement_generation"]),
                "runtime_state": str(job["state"]),
                "public_artifact_safe": True,
            }

    def _authorize_tensor_message(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        session_token: str,
        assignment_token: str,
        envelope: dict[str, Any],
        role: str,
    ) -> sqlite3.Row:
        canonical = validate_tensor_envelope(envelope)
        if self.training_manifest is None or self.tensor_store is None:
            raise ValueError("elastic_heterogeneous_tensor_transport_not_enabled")
        if (
            canonical["job_id"] != self.run_id
            or canonical["manifest_hash"]
            != self.training_manifest["content_hash"]
        ):
            raise ValueError("elastic_tensor_manifest_identity_mismatch")
        stage_id = int(
            canonical[
                "source_stage_id" if role == "source" else "target_stage_id"
            ]
        )
        assignment = self._require_stage_assignment(
            connection,
            session_id=session_id,
            session_token=session_token,
            assignment_token=assignment_token,
            stage_id=stage_id,
            placement_generation=int(canonical["placement_generation"]),
        )
        if int(assignment["target_step"]) != int(canonical["global_step"]):
            raise ValueError("elastic_tensor_global_step_stale")
        if role == "source" and not hmac.compare_digest(
            str(canonical["assignment_token_hash"]),
            _token_hash(str(assignment_token)),
        ):
            raise ValueError("elastic_tensor_assignment_identity_invalid")
        return assignment

    def begin_tensor_message(
        self,
        *,
        session_id: str,
        session_token: str,
        assignment_token: str,
        envelope: dict[str, Any],
    ) -> dict[str, Any]:
        canonical = validate_tensor_envelope(envelope)
        with self._transaction() as connection:
            self._reconcile(connection, now=float(self._clock()))
            self._authorize_tensor_message(
                connection,
                session_id=session_id,
                session_token=session_token,
                assignment_token=assignment_token,
                envelope=canonical,
                role="source",
            )
        assert self.tensor_store is not None
        return self.tensor_store.begin(
            canonical,
            expected_generation=int(canonical["placement_generation"]),
        )

    def put_tensor_chunk(
        self,
        *,
        session_id: str,
        session_token: str,
        assignment_token: str,
        message_id: str,
        chunk_index: int,
        value: bytes,
    ) -> dict[str, Any]:
        if self.tensor_store is None:
            raise ValueError("elastic_heterogeneous_tensor_transport_not_enabled")
        envelope = self.tensor_store.envelope(message_id)
        with self._transaction() as connection:
            self._reconcile(connection, now=float(self._clock()))
            self._authorize_tensor_message(
                connection,
                session_id=session_id,
                session_token=session_token,
                assignment_token=assignment_token,
                envelope=envelope,
                role="source",
            )
        return self.tensor_store.put_chunk(
            message_id,
            int(chunk_index),
            value,
            expected_generation=int(envelope["placement_generation"]),
        )

    def find_tensor_message(
        self,
        *,
        session_id: str,
        session_token: str,
        assignment_token: str,
        global_step: int,
        microbatch_id: int,
        source_stage_id: int,
        target_stage_id: int,
        direction: str,
        placement_generation: int,
    ) -> dict[str, Any]:
        if self.tensor_store is None:
            raise ValueError("elastic_heterogeneous_tensor_transport_not_enabled")
        envelope = self.tensor_store.find_message(
            job_id=self.run_id,
            global_step=global_step,
            microbatch_id=microbatch_id,
            source_stage_id=source_stage_id,
            target_stage_id=target_stage_id,
            direction=direction,
            placement_generation=placement_generation,
            use_index=(
                self.tensor_lookup_optimization_after_step <= 0
                or int(global_step)
                > self.tensor_lookup_optimization_after_step
            ),
        )
        if envelope is None:
            return {
                "schema": RUNTIME_SCHEMA,
                "found": False,
                "indexed_lookup_enabled": bool(
                    self.tensor_lookup_optimization_after_step <= 0
                    or int(global_step)
                    > self.tensor_lookup_optimization_after_step
                ),
                "public_artifact_safe": True,
            }
        with self._transaction() as connection:
            self._reconcile(connection, now=float(self._clock()))
            self._authorize_tensor_message(
                connection,
                session_id=session_id,
                session_token=session_token,
                assignment_token=assignment_token,
                envelope=envelope,
                role="target",
            )
        return {
            "schema": RUNTIME_SCHEMA,
            "found": True,
            "envelope": envelope,
            "status": self.tensor_store.status(envelope["message_id"]),
            "indexed_lookup_enabled": bool(
                self.tensor_lookup_optimization_after_step <= 0
                or int(global_step)
                > self.tensor_lookup_optimization_after_step
            ),
            "public_artifact_safe": True,
        }

    def read_tensor_chunk(
        self,
        *,
        session_id: str,
        session_token: str,
        assignment_token: str,
        message_id: str,
        chunk_index: int,
    ) -> tuple[bytes, dict[str, Any]]:
        if self.tensor_store is None:
            raise ValueError("elastic_heterogeneous_tensor_transport_not_enabled")
        envelope = self.tensor_store.envelope(message_id)
        with self._transaction() as connection:
            self._reconcile(connection, now=float(self._clock()))
            self._authorize_tensor_message(
                connection,
                session_id=session_id,
                session_token=session_token,
                assignment_token=assignment_token,
                envelope=envelope,
                role="target",
            )
        return self.tensor_store.read_chunk(message_id, int(chunk_index)), envelope

    def cancel(self, *, reason: str = "owner_cancelled") -> dict[str, Any]:
        now = float(self._clock())
        with self._transaction() as connection:
            job = self._job(connection)
            changed = str(job["state"]) not in {"cancelled", "completed", "cleaned"}
            if changed and job["active_epoch_id"] is not None:
                self._abort_epoch(
                    connection,
                    epoch_id=int(job["active_epoch_id"]),
                    reason=str(reason),
                    now=now,
                )
            if changed:
                connection.execute(
                    """
                    UPDATE jobs SET state='cancelled',active_epoch_id=NULL,
                        revision=revision+1,updated_at=? WHERE run_id=?
                    """,
                    (now, self.run_id),
                )
                connection.execute(
                    """
                    UPDATE miners SET state='offline',offline_at=?,lease_expires_at=?
                    WHERE run_id=? AND state='online'
                    """,
                    (now, now, self.run_id),
                )
                self._event(
                    connection,
                    run_id=self.run_id,
                    operation="training_cancelled",
                    value={"reason": str(reason)},
                    now=now,
                )
        return {
            **self.public_status(reconcile=False),
            "cancel_transition_applied": changed,
            "command_ok": True,
        }

    def cleanup(self) -> dict[str, Any]:
        """Fence active Miners and retain only the configured checkpoint window."""

        now = float(self._clock())
        with self._transaction() as connection:
            job = self._job(connection)
            changed = str(job["state"]) != "cleaned"
            if changed and job["active_epoch_id"] is not None:
                self._abort_epoch(
                    connection,
                    epoch_id=int(job["active_epoch_id"]),
                    reason="owner_cleanup",
                    now=now,
                )
            if changed:
                connection.execute(
                    """
                    UPDATE jobs SET state='cleaned',active_epoch_id=NULL,
                        revision=revision+1,updated_at=? WHERE run_id=?
                    """,
                    (now, self.run_id),
                )
                connection.execute(
                    """
                    UPDATE miners SET state='offline',offline_at=?,lease_expires_at=?
                    WHERE run_id=? AND state='online'
                    """,
                    (now, now, self.run_id),
                )
                self._event(
                    connection,
                    run_id=self.run_id,
                    operation="training_cleaned",
                    value={
                        "committed_step": int(job["committed_step"]),
                        "active_miner_leases_revoked": True,
                    },
                    now=now,
                )
        retention = self.enforce_checkpoint_retention()
        uncommitted = self.cleanup_uncommitted_blobs()
        return {
            **self.public_status(reconcile=False),
            "cleanup_transition_applied": changed,
            "checkpoint_retention": retention,
            "uncommitted_blob_cleanup": uncommitted,
            "command_ok": bool(retention.get("ok") and uncommitted.get("ok")),
        }

    def _record_checkpoint_rejection(
        self,
        *,
        session_id: str,
        reason: str,
        now: float,
    ) -> None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM miners WHERE run_id=? AND session_id=?",
                (self.run_id, str(session_id)),
            ).fetchone()
            if row is None:
                return
            rejected = int(row["rejected_submission_count"]) + 1
            quarantined = rejected >= self.max_rejected_submissions_per_session
            connection.execute(
                """
                UPDATE miners SET rejected_submission_count=?,
                    state=CASE WHEN ? THEN 'quarantined' ELSE state END,
                    offline_at=CASE WHEN ? THEN ? ELSE offline_at END,
                    lease_expires_at=CASE WHEN ? THEN ? ELSE lease_expires_at END
                WHERE session_id=?
                """,
                (
                    rejected,
                    int(quarantined),
                    int(quarantined),
                    now,
                    int(quarantined),
                    now,
                    str(session_id),
                ),
            )
            self._event(
                connection,
                run_id=self.run_id,
                operation="checkpoint_submission_rejected",
                value={
                    "miner_session_hash": _token_hash(str(session_id)),
                    "reason": str(reason)[:160],
                    "rejected_submission_count": rejected,
                    "miner_quarantined": quarantined,
                },
                now=now,
            )
            self._reconcile(connection, now=now)

    def _store_blob(self, archive: bytes, archive_hash: str) -> dict[str, Any]:
        return self.blob_store.put(archive_hash, archive)

    def submit_checkpoint(
        self,
        *,
        session_id: str,
        session_token: str,
        epoch_id: int,
        stage_id: int,
        assignment_token: str,
        archive: bytes,
        checkpoint_signature: str = "",
    ) -> dict[str, Any]:
        now = float(self._clock())
        archive_hash = _sha256_bytes(archive)
        with _connect(self.state_path) as connection:
            row = connection.execute(
                """
                SELECT a.*,e.target_step,e.dataset_cursor,e.state AS epoch_state
                FROM assignments a JOIN epochs e
                  ON e.run_id=a.run_id AND e.epoch_id=a.epoch_id
                WHERE a.run_id=? AND a.epoch_id=? AND a.stage_id=? AND a.session_id=?
                """,
                (self.run_id, int(epoch_id), int(stage_id), str(session_id)),
            ).fetchone()
            if row is None or not hmac.compare_digest(
                str(row["assignment_token"]), str(assignment_token)
            ):
                raise ValueError("elastic_stage_assignment_invalid")
            session = self._require_session(
                connection,
                session_id=session_id,
                session_token=session_token,
                allow_inactive=True,
            )
            expected_step = int(row["target_step"])
            expected_cursor = int(row["dataset_cursor"])
            expected_placement_generation = int(row["placement_generation"])
            session_was_online = str(session["state"]) == "online"
            if str(row["epoch_state"]) == "aborted" or str(row["state"]) == "revoked":
                raise ValueError("elastic_stage_assignment_stale")
            if str(row["epoch_state"]) == "active" and not session_was_online:
                raise ValueError("elastic_stage_assignment_stale")
            previous_preflight = connection.execute(
                """
                SELECT archive_hash FROM submissions
                WHERE run_id=? AND epoch_id=? AND stage_id=?
                """,
                (self.run_id, int(epoch_id), int(stage_id)),
            ).fetchone()
            duplicate_preflight = bool(
                previous_preflight is not None
                and str(previous_preflight["archive_hash"]) == archive_hash
            )
            if (
                not duplicate_preflight
                and int(session["accepted_upload_bytes"]) + len(archive)
                > self.max_checkpoint_bytes_per_session
            ):
                rejection = "elastic_checkpoint_session_byte_quota_exceeded"
            elif self.require_checkpoint_signatures:
                expected_signature = sign_checkpoint_submission(
                    session_token=session_token,
                    run_id=self.run_id,
                    session_id=session_id,
                    epoch_id=epoch_id,
                    stage_id=stage_id,
                    assignment_token=assignment_token,
                    archive_hash=archive_hash,
                )
                rejection = (
                    ""
                    if hmac.compare_digest(
                        str(checkpoint_signature), expected_signature
                    )
                    else "elastic_checkpoint_signature_invalid"
                )
            else:
                rejection = ""
        if rejection:
            self._record_checkpoint_rejection(
                session_id=session_id, reason=rejection, now=now
            )
            raise ValueError(rejection)
        try:
            if self.training_manifest is not None:
                report = validate_stage_checkpoint_archive(
                    archive,
                    training_manifest=self.training_manifest,
                    expected_stage_id=int(stage_id),
                    expected_step=expected_step,
                    expected_dataset_cursor=expected_cursor,
                    expected_placement_generation=expected_placement_generation,
                    max_checkpoint_bytes=self.max_checkpoint_bytes,
                    validate_tensor_payloads=self.validate_checkpoint_tensors,
                )
            else:
                report = validate_qwen_stage_checkpoint_archive(
                    archive,
                    expected_stage_id=int(stage_id),
                    expected_step=expected_step,
                    expected_dataset_cursor=expected_cursor,
                    max_checkpoint_bytes=self.max_checkpoint_bytes,
                    validate_tensor_payloads=self.validate_checkpoint_tensors,
                    expected_model_id=self.legacy_model_id,
                    expected_model_revision=self.legacy_model_revision,
                )
        except (ValueError, RuntimeError) as exc:
            self._record_checkpoint_rejection(
                session_id=session_id, reason=str(exc), now=now
            )
            raise
        self._store_blob(archive, str(report["archive_hash"]))
        with self._transaction() as connection:
            self._reconcile(connection, now=now)
            assignment = connection.execute(
                """
                SELECT a.*,e.target_step,e.dataset_cursor,e.state AS epoch_state
                FROM assignments a JOIN epochs e
                  ON e.run_id=a.run_id AND e.epoch_id=a.epoch_id
                WHERE a.run_id=? AND a.epoch_id=? AND a.stage_id=? AND a.session_id=?
                """,
                (self.run_id, int(epoch_id), int(stage_id), str(session_id)),
            ).fetchone()
            if assignment is None or not hmac.compare_digest(
                str(assignment["assignment_token"]), str(assignment_token)
            ):
                raise ValueError("elastic_stage_assignment_invalid")
            session = self._require_session(
                connection,
                session_id=session_id,
                session_token=session_token,
                allow_inactive=True,
            )
            previous = connection.execute(
                """
                SELECT * FROM submissions
                WHERE run_id=? AND epoch_id=? AND stage_id=?
                """,
                (self.run_id, int(epoch_id), int(stage_id)),
            ).fetchone()
            if previous is not None:
                if (
                    str(previous["archive_hash"]) != str(report["archive_hash"])
                    or str(previous["checkpoint_content_hash"])
                    != str(report["checkpoint_content_hash"])
                    or str(previous["session_id"]) != str(session_id)
                ):
                    raise ValueError("elastic_checkpoint_submission_conflict")
                epoch = connection.execute(
                    "SELECT state FROM epochs WHERE run_id=? AND epoch_id=?",
                    (self.run_id, int(epoch_id)),
                ).fetchone()
                job = self._job(connection)
                return {
                    "schema": RUNTIME_SCHEMA,
                    "ok": True,
                    "idempotent": True,
                    "global_commit_created": False,
                    "barrier_state": str(epoch["state"]),
                    "epoch_id": int(epoch_id),
                    "target_step": expected_step,
                    "committed_step": int(job["committed_step"]),
                    "archive_hash": str(report["archive_hash"]),
                    "exactly_once_commit": True,
                    "public_artifact_safe": True,
                }
            if (
                str(assignment["state"]) != "active"
                or str(assignment["epoch_state"]) != "active"
                or str(session["state"]) != "online"
                or float(session["lease_expires_at"]) <= now
                or not session_was_online
            ):
                raise ValueError("elastic_stage_assignment_stale")
            if (
                int(assignment["target_step"]) != int(report["global_step"])
                or int(assignment["dataset_cursor"]) != int(report["dataset_cursor"])
            ):
                raise ValueError("elastic_checkpoint_barrier_position_invalid")
            connection.execute(
                """
                INSERT INTO submissions(
                    run_id,epoch_id,stage_id,session_id,target_step,dataset_cursor,
                    archive_hash,archive_bytes,checkpoint_content_hash,
                    adapter_tensor_hash,component_hashes_hash,state,submitted_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,'candidate',?)
                """,
                (
                    self.run_id,
                    int(epoch_id),
                    int(stage_id),
                    str(session_id),
                    int(report["global_step"]),
                    int(report["dataset_cursor"]),
                    str(report["archive_hash"]),
                    int(report["archive_bytes"]),
                    str(report["checkpoint_content_hash"]),
                    str(report["adapter_tensor_hash"]),
                    str(report["component_hashes_hash"]),
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE miners SET accepted_upload_bytes=accepted_upload_bytes+?
                WHERE session_id=?
                """,
                (int(report["archive_bytes"]), str(session_id)),
            )
            self._event(
                connection,
                run_id=self.run_id,
                operation="stage_checkpoint_submitted",
                value={
                    "epoch_id": int(epoch_id),
                    "stage_id": int(stage_id),
                    "target_step": int(report["global_step"]),
                    "archive_hash": str(report["archive_hash"]),
                    "checkpoint_content_hash": str(report["checkpoint_content_hash"]),
                    "miner_session_hash": _token_hash(str(session_id)),
                },
                now=now,
            )
            job = self._job(connection)
            rows = connection.execute(
                """
                SELECT * FROM submissions
                WHERE run_id=? AND epoch_id=? AND state='candidate'
                ORDER BY stage_id
                """,
                (self.run_id, int(epoch_id)),
            ).fetchall()
            commit_created = False
            if len(rows) == int(job["stage_count"]):
                expected_ids = list(range(int(job["stage_count"])))
                if [int(item["stage_id"]) for item in rows] != expected_ids:
                    raise RuntimeError("elastic_barrier_stage_coverage_invalid")
                target_steps = {int(item["target_step"]) for item in rows}
                cursors = {int(item["dataset_cursor"]) for item in rows}
                if target_steps != {expected_step} or cursors != {expected_cursor}:
                    raise RuntimeError("elastic_barrier_progress_divergence")
                if int(job["committed_step"]) != expected_step - 1:
                    raise RuntimeError("elastic_exactly_once_step_sequence_invalid")
                checkpoint_set_hash = stable_hash(
                    [
                        {
                            "stage_id": int(item["stage_id"]),
                            "archive_hash": str(item["archive_hash"]),
                            "checkpoint_content_hash": str(
                                item["checkpoint_content_hash"]
                            ),
                        }
                        for item in rows
                    ]
                )
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO commits(
                        run_id,target_step,epoch_id,dataset_cursor,
                        checkpoint_set_hash,committed_at
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    (
                        self.run_id,
                        expected_step,
                        int(epoch_id),
                        expected_cursor,
                        checkpoint_set_hash,
                        now,
                    ),
                )
                if cursor.rowcount != 1:
                    existing = connection.execute(
                        "SELECT * FROM commits WHERE run_id=? AND target_step=?",
                        (self.run_id, expected_step),
                    ).fetchone()
                    if (
                        existing is None
                        or str(existing["checkpoint_set_hash"]) != checkpoint_set_hash
                    ):
                        raise RuntimeError("elastic_exactly_once_commit_conflict")
                else:
                    commit_created = True
                    connection.execute(
                        """
                        UPDATE epochs SET state='committed',committed_at=?,
                            checkpoint_set_hash=?
                        WHERE run_id=? AND epoch_id=? AND state='active'
                        """,
                        (now, checkpoint_set_hash, self.run_id, int(epoch_id)),
                    )
                    connection.execute(
                        "UPDATE assignments SET state='completed' WHERE run_id=? AND epoch_id=?",
                        (self.run_id, int(epoch_id)),
                    )
                    connection.execute(
                        "UPDATE submissions SET state='committed' WHERE run_id=? AND epoch_id=?",
                        (self.run_id, int(epoch_id)),
                    )
                    connection.execute(
                        """
                        UPDATE jobs SET committed_step=?,dataset_cursor=?,
                            active_epoch_id=NULL,revision=revision+1,updated_at=?
                        WHERE run_id=?
                        """,
                        (expected_step, expected_cursor, now, self.run_id),
                    )
                    self._event(
                        connection,
                        run_id=self.run_id,
                        operation="optimizer_step_committed",
                        value={
                            "epoch_id": int(epoch_id),
                            "committed_step": expected_step,
                            "dataset_cursor": expected_cursor,
                            "checkpoint_set_hash": checkpoint_set_hash,
                            "stage_count": len(rows),
                            "exactly_once": True,
                        },
                        now=now,
                    )
                    self._reconcile(connection, now=now)
            current_epoch = connection.execute(
                "SELECT state FROM epochs WHERE run_id=? AND epoch_id=?",
                (self.run_id, int(epoch_id)),
            ).fetchone()
            current_job = self._job(connection)
            return {
                "schema": RUNTIME_SCHEMA,
                "ok": True,
                "idempotent": False,
                "global_commit_created": commit_created,
                "barrier_state": str(current_epoch["state"]),
                "epoch_id": int(epoch_id),
                "target_step": expected_step,
                "committed_step": int(current_job["committed_step"]),
                "archive_hash": str(report["archive_hash"]),
                "exactly_once_commit": True,
                "public_artifact_safe": True,
            }

    def barrier_status(
        self,
        *,
        session_id: str,
        session_token: str,
        epoch_id: int,
    ) -> dict[str, Any]:
        now = float(self._clock())
        with self._transaction() as connection:
            self._reconcile(connection, now=now)
            self._require_session(
                connection,
                session_id=session_id,
                session_token=session_token,
                allow_inactive=True,
            )
            assigned = connection.execute(
                """
                SELECT COUNT(*) FROM assignments
                WHERE run_id=? AND epoch_id=? AND session_id=?
                """,
                (self.run_id, int(epoch_id), session_id),
            ).fetchone()[0]
            if not int(assigned):
                raise ValueError("elastic_barrier_session_not_assigned")
            epoch = connection.execute(
                "SELECT * FROM epochs WHERE run_id=? AND epoch_id=?",
                (self.run_id, int(epoch_id)),
            ).fetchone()
            if epoch is None:
                raise KeyError("elastic_barrier_not_found")
            count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM submissions
                    WHERE run_id=? AND epoch_id=? AND state IN ('candidate','committed')
                    """,
                    (self.run_id, int(epoch_id)),
                ).fetchone()[0]
            )
            job = self._job(connection)
            return {
                "schema": RUNTIME_SCHEMA,
                "epoch_id": int(epoch_id),
                "base_step": int(epoch["base_step"]),
                "target_step": int(epoch["target_step"]),
                "state": str(epoch["state"]),
                "submitted_stage_count": count,
                "required_stage_count": int(job["stage_count"]),
                "committed_step": int(job["committed_step"]),
                "abort_reason": str(epoch["abort_reason"]),
                "assignment_tokens_public": False,
                "public_artifact_safe": True,
            }

    def download_committed_checkpoint(
        self,
        *,
        session_id: str,
        session_token: str,
        epoch_id: int,
        stage_id: int,
        assignment_token: str,
    ) -> tuple[bytes, dict[str, Any]]:
        now = float(self._clock())
        with self._transaction() as connection:
            self._reconcile(connection, now=now)
            session = self._require_session(
                connection, session_id=session_id, session_token=session_token
            )
            if float(session["lease_expires_at"]) <= now:
                raise ValueError("elastic_miner_session_stale")
            assignment = connection.execute(
                """
                SELECT a.*,e.base_step,e.state AS epoch_state
                FROM assignments a JOIN epochs e
                  ON e.run_id=a.run_id AND e.epoch_id=a.epoch_id
                WHERE a.run_id=? AND a.epoch_id=? AND a.stage_id=? AND a.session_id=?
                """,
                (self.run_id, int(epoch_id), int(stage_id), session_id),
            ).fetchone()
            if (
                assignment is None
                or str(assignment["state"]) != "active"
                or str(assignment["epoch_state"]) != "active"
                or not hmac.compare_digest(
                    str(assignment["assignment_token"]), str(assignment_token)
                )
            ):
                raise ValueError("elastic_stage_assignment_invalid")
            base_step = int(assignment["base_step"])
            if base_step < 1:
                raise ValueError("elastic_committed_checkpoint_not_required")
            row = connection.execute(
                """
                SELECT s.* FROM commits c JOIN submissions s
                  ON s.run_id=c.run_id AND s.epoch_id=c.epoch_id
                WHERE c.run_id=? AND c.target_step=? AND s.stage_id=?
                  AND s.state='committed'
                """,
                (self.run_id, base_step, int(stage_id)),
            ).fetchone()
            if row is None:
                raise RuntimeError("elastic_committed_checkpoint_missing")
            archive_hash = str(row["archive_hash"])
            expected_cursor = base_step * self.microbatches_per_step
        archive = self.blob_store.get(archive_hash)
        if self.training_manifest is not None:
            report = validate_stage_checkpoint_archive(
                archive,
                training_manifest=self.training_manifest,
                expected_stage_id=int(stage_id),
                expected_step=base_step,
                expected_dataset_cursor=expected_cursor,
                max_checkpoint_bytes=self.max_checkpoint_bytes,
                validate_tensor_payloads=self.validate_checkpoint_tensors,
            )
        else:
            report = validate_qwen_stage_checkpoint_archive(
                archive,
                expected_stage_id=int(stage_id),
                expected_step=base_step,
                expected_dataset_cursor=expected_cursor,
                max_checkpoint_bytes=self.max_checkpoint_bytes,
                validate_tensor_payloads=self.validate_checkpoint_tensors,
                expected_model_id=self.legacy_model_id,
                expected_model_revision=self.legacy_model_revision,
            )
        return archive, report

    def read_committed_checkpoint(
        self,
        *,
        stage_id: int,
        target_step: int | None = None,
    ) -> tuple[bytes, dict[str, Any]]:
        """Read a committed checkpoint for an authenticated owner-side export."""

        with _connect(self.state_path) as connection:
            job = self._job(connection)
            step = int(target_step or job["committed_step"])
            if step < 1 or step > int(job["committed_step"]):
                raise ValueError("elastic_committed_checkpoint_step_invalid")
            row = connection.execute(
                """
                SELECT s.archive_hash,c.dataset_cursor FROM commits c
                JOIN submissions s ON s.run_id=c.run_id AND s.epoch_id=c.epoch_id
                WHERE c.run_id=? AND c.target_step=? AND s.stage_id=?
                  AND s.state='committed'
                """,
                (self.run_id, step, int(stage_id)),
            ).fetchone()
        if row is None:
            raise RuntimeError("elastic_committed_checkpoint_missing")
        archive = self.blob_store.get(str(row["archive_hash"]))
        if self.training_manifest is not None:
            report = validate_stage_checkpoint_archive(
                archive,
                training_manifest=self.training_manifest,
                expected_stage_id=int(stage_id),
                expected_step=step,
                expected_dataset_cursor=int(row["dataset_cursor"]),
                max_checkpoint_bytes=self.max_checkpoint_bytes,
                validate_tensor_payloads=self.validate_checkpoint_tensors,
            )
        else:
            report = validate_qwen_stage_checkpoint_archive(
                archive,
                expected_stage_id=int(stage_id),
                expected_step=step,
                expected_dataset_cursor=int(row["dataset_cursor"]),
                max_checkpoint_bytes=self.max_checkpoint_bytes,
                validate_tensor_payloads=self.validate_checkpoint_tensors,
                expected_model_id=self.legacy_model_id,
                expected_model_revision=self.legacy_model_revision,
            )
        return archive, report

    def public_status(self, *, reconcile: bool = True) -> dict[str, Any]:
        observed_at = float(self._clock())
        if reconcile:
            with self._transaction() as connection:
                self._reconcile(connection, now=observed_at)
        with _connect(self.state_path) as connection:
            job = self._job(connection)
            miners = connection.execute(
                "SELECT * FROM miners WHERE run_id=? ORDER BY registered_at,session_id",
                (self.run_id,),
            ).fetchall()
            epochs = connection.execute(
                "SELECT * FROM epochs WHERE run_id=? ORDER BY epoch_id",
                (self.run_id,),
            ).fetchall()
            assignments = connection.execute(
                """
                SELECT a.*,m.miner_id_hash FROM assignments a JOIN miners m
                  ON m.session_id=a.session_id
                WHERE a.run_id=? ORDER BY a.epoch_id,a.stage_id
                """,
                (self.run_id,),
            ).fetchall()
            commits = connection.execute(
                "SELECT * FROM commits WHERE run_id=? ORDER BY target_step",
                (self.run_id,),
            ).fetchall()
            events = connection.execute(
                """
                SELECT sequence,event_json,created_at FROM elastic_events
                WHERE run_id=? ORDER BY sequence
                """,
                (self.run_id,),
            ).fetchall()
            health_rows = connection.execute(
                """
                SELECT * FROM device_health WHERE run_id=?
                ORDER BY session_id,device_id
                """,
                (self.run_id,),
            ).fetchall()
            live_miners = [row for row in miners if str(row["state"]) == "online"]
            active_assignments = [
                row for row in assignments if str(row["state"]) == "active"
            ]
            active_stage_ids = sorted(
                {int(row["stage_id"]) for row in active_assignments}
            )
            missing_stage_ids = sorted(
                set(range(int(job["stage_count"]))) - set(active_stage_ids)
            )
            commit_steps = [int(row["target_step"]) for row in commits]
            try:
                placement_plan = json.loads(str(job["placement_plan_json"] or "{}"))
            except json.JSONDecodeError:
                placement_plan = {}
            try:
                placement_error = json.loads(str(job["placement_error_json"] or "{}"))
            except json.JSONDecodeError:
                placement_error = {
                    "code": "heterogeneous_placement_error_state_invalid"
                }
            public = {
                "schema": STATUS_SCHEMA,
                "runtime_schema": RUNTIME_SCHEMA,
                "run_id_hash": _token_hash(self.run_id),
                "model_id": str(job["model_id"]),
                "model_revision": str(job["model_revision"]),
                "runtime_state": str(job["state"]),
                "committed_step": int(job["committed_step"]),
                "dataset_cursor": int(job["dataset_cursor"]),
                "target_steps": int(job["target_steps"]),
                "stage_count": int(job["stage_count"]),
                "training_manifest_hash": str(job["manifest_hash"] or ""),
                "heterogeneous_scheduler_enabled": self.training_manifest is not None,
                "placement_generation": int(job["placement_generation"]),
                "placement_plan": placement_plan,
                "placement_error": placement_error,
                "pending_rebalance_reason": str(
                    job["pending_rebalance_reason"] or ""
                ),
                "owner_paused": bool(job["owner_paused"]),
                "coordinator_generation": int(job["coordinator_generation"]),
                "coordinator_started_at": float(job["coordinator_started_at"]),
                "persistent_coordinator_journal": True,
                "run_created_at": float(job["created_at"]),
                "status_observed_at": observed_at,
                "active_epoch_id": (
                    int(job["active_epoch_id"])
                    if job["active_epoch_id"] is not None
                    else None
                ),
                "live_miner_count": len(live_miners),
                "zero_live_miners": not live_miners,
                "active_stage_ids": active_stage_ids,
                "missing_stage_ids": missing_stage_ids,
                "stage_coverage_complete": len(active_assignments)
                == int(job["stage_count"]),
                "automatic_pause_wake_enabled": True,
                "paused_waiting_for_miners": str(job["state"])
                == "paused_waiting_for_miners",
                "pause_reason": (
                    "incomplete_stage_coverage"
                    if str(job["state"]) == "paused_waiting_for_miners"
                    else ""
                ),
                "exactly_once_optimizer_commit_enabled": True,
                "tensor_lookup_optimization_after_step": self.tensor_lookup_optimization_after_step,
                "indexed_tensor_lookup_active": bool(
                    self.tensor_lookup_optimization_after_step <= 0
                    or int(job["committed_step"])
                    >= self.tensor_lookup_optimization_after_step
                ),
                "optimizer_commit_count": len(commits),
                "committed_steps": commit_steps,
                "committed_steps_contiguous": commit_steps
                == list(range(1, int(job["committed_step"]) + 1)),
                "miners": [
                    {
                        "miner_id_hash": str(row["miner_id_hash"]),
                        "miner_session_hash": _token_hash(str(row["session_id"])),
                        "supported_stage_ids": json.loads(
                            str(row["supported_stage_ids_json"])
                        ),
                        "slot_count": int(row["slot_count"]),
                        "accelerator": str(row["accelerator"]),
                        "capability": json.loads(
                            str(row["capability_json"] or "{}")
                        ),
                        "stage_metrics": json.loads(
                            str(row["stage_metrics_json"] or "[]")
                        ),
                        "last_failure_reason": str(
                            row["last_failure_reason"] or ""
                        ),
                        "state": str(row["state"]),
                        "generation": int(row["generation"]),
                        "accepted_upload_bytes": int(row["accepted_upload_bytes"]),
                        "rejected_submission_count": int(
                            row["rejected_submission_count"]
                        ),
                        "quarantined": str(row["state"]) == "quarantined",
                        "registered_at": float(row["registered_at"]),
                        "heartbeat_age_seconds": max(
                            0.0, observed_at - float(row["heartbeat_at"])
                        ),
                        "lease_remaining_seconds": max(
                            0.0, float(row["lease_expires_at"]) - observed_at
                        ),
                        "offline_at": float(row["offline_at"]),
                    }
                    for row in miners
                ],
                "epochs": [
                    {
                        "epoch_id": int(row["epoch_id"]),
                        "base_step": int(row["base_step"]),
                        "target_step": int(row["target_step"]),
                        "dataset_cursor": int(row["dataset_cursor"]),
                        "state": str(row["state"]),
                        "abort_reason": str(row["abort_reason"]),
                        "checkpoint_set_hash": str(row["checkpoint_set_hash"]),
                    }
                    for row in epochs
                ],
                "assignments": [
                    {
                        "epoch_id": int(row["epoch_id"]),
                        "stage_id": int(row["stage_id"]),
                        "miner_id_hash": str(row["miner_id_hash"]),
                        "miner_session_hash": _token_hash(str(row["session_id"])),
                        "assignment_token_hash": str(row["assignment_token_hash"]),
                        "placement_generation": int(row["placement_generation"]),
                        "device_id": str(row["device_id"]),
                        "device_type": str(row["device_type"]),
                        "state": str(row["state"]),
                    }
                    for row in assignments
                ],
                "commits": [
                    {
                        "target_step": int(row["target_step"]),
                        "epoch_id": int(row["epoch_id"]),
                        "dataset_cursor": int(row["dataset_cursor"]),
                        "checkpoint_set_hash": str(row["checkpoint_set_hash"]),
                        "committed_at": float(row["committed_at"]),
                    }
                    for row in commits
                ],
                "events": [
                    {
                        "sequence": int(row["sequence"]),
                        **json.loads(str(row["event_json"])),
                        "created_at": float(row["created_at"]),
                    }
                    for row in events
                ],
                "event_count": len(events),
                "device_health": [
                    {
                        "miner_session_hash": _token_hash(
                            str(row["session_id"])
                        ),
                        "device_id": str(row["device_id"]),
                        "state": str(row["state"]),
                        "consecutive_failures": int(
                            row["consecutive_failures"]
                        ),
                        "total_failures": int(row["total_failures"]),
                        "quarantine_remaining_seconds": max(
                            0.0,
                            float(row["quarantine_until"]) - observed_at,
                        ),
                        "last_failure_class": str(
                            row["last_failure_class"] or ""
                        ),
                        "checkpoint_step": int(row["checkpoint_step"]),
                        "telemetry": json.loads(
                            str(row["telemetry_json"] or "{}")
                        ),
                    }
                    for row in health_rows
                ],
                "checkpoint_storage": {
                    **self.blob_store.public_report(),
                    "central_to_miner_sessions": True,
                    "retention_committed_steps": self.checkpoint_retention_steps,
                    "private_paths_public": False,
                },
                "topology_aware_stage_groups": (
                    [] if self.training_manifest is not None else [[0, 1], [2, 3]]
                ),
                "checkpoint_signatures_required": self.require_checkpoint_signatures,
                "checkpoint_tensor_validation_required": self.validate_checkpoint_tensors,
                "miner_quota_policy": {
                    "max_online_miners": self.max_online_miners,
                    "max_rejected_submissions_per_session": self.max_rejected_submissions_per_session,
                    "max_checkpoint_bytes_per_session": self.max_checkpoint_bytes_per_session,
                    "malicious_miner_quarantine_enabled": True,
                },
                "session_tokens_public": False,
                "assignment_tokens_public": False,
                "checkpoint_tensor_values_public": False,
                "private_paths_public": False,
                "public_artifact_safe": True,
            }
            public["content_hash"] = stable_hash(public)
            return public

    def event_tail(self, *, after_sequence: int = 0, limit: int = 200) -> dict[str, Any]:
        if int(after_sequence) < 0 or not 1 <= int(limit) <= 1000:
            raise ValueError("elastic_event_cursor_invalid")
        with _connect(self.state_path) as connection:
            rows = connection.execute(
                """
                SELECT sequence,event_json,created_at FROM elastic_events
                WHERE run_id=? AND sequence>? ORDER BY sequence LIMIT ?
                """,
                (self.run_id, int(after_sequence), int(limit)),
            ).fetchall()
        events = [
            {
                "sequence": int(row["sequence"]),
                **json.loads(str(row["event_json"])),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]
        return {
            "schema": "crowdtensor_elastic_training_event_page_v1",
            "run_id_hash": _token_hash(self.run_id),
            "events": events,
            "event_count": len(events),
            "next_after_sequence": (
                int(events[-1]["sequence"]) if events else int(after_sequence)
            ),
            "bounded_page": True,
            "credential_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }

    def _tensor_metric_summary(self) -> dict[str, Any]:
        directions: dict[str, dict[str, int]] = {}
        message_count = 0
        payload_bytes = 0
        if self.tensor_store is not None and self.tensor_store.root.is_dir():
            for directory in self.tensor_store.root.iterdir():
                if not directory.is_dir() or not re.fullmatch(r"[0-9a-f]{64}", directory.name):
                    continue
                try:
                    envelope = self.tensor_store.envelope("sha256:" + directory.name)
                except TensorTransportError:
                    continue
                direction = str(envelope.get("direction") or "unknown")
                size = int(envelope.get("payload_bytes") or 0)
                row = directions.setdefault(direction, {"messages": 0, "bytes": 0})
                row["messages"] += 1
                row["bytes"] += size
                message_count += 1
                payload_bytes += size
        return {
            "message_count": message_count,
            "payload_bytes": payload_bytes,
            "directions": {
                key: directions[key] for key in sorted(directions)
            },
            "tensor_values_public": False,
        }

    def metrics_snapshot(self) -> dict[str, Any]:
        """Return low-cardinality operational metrics with no private values."""

        status = self.public_status()
        commits = list(status["commits"])
        commit_times = [float(item["committed_at"]) for item in commits]
        step_durations = []
        previous = float(status["run_created_at"])
        for committed_at in commit_times:
            step_durations.append(max(0.0, committed_at - previous))
            previous = committed_at
        profiles = []
        worker_states: dict[str, int] = {}
        accelerator_states: dict[str, int] = {}
        for miner in status["miners"]:
            state = str(miner["state"])
            accelerator = str(miner["accelerator"])
            worker_states[state] = worker_states.get(state, 0) + 1
            accelerator_states[accelerator] = (
                accelerator_states.get(accelerator, 0) + 1
            )
            for profile in miner.get("stage_metrics") or []:
                profiles.append(
                    {
                        "stage_id": int(profile.get("stage_id") or 0),
                        "device_id": str(profile.get("device_id") or ""),
                        "accelerator": accelerator,
                        "forward_latency_ms": float(
                            profile.get("steady_forward_latency_ms")
                            or profile.get("forward_latency_ms")
                            or 0.0
                        ),
                        "backward_latency_ms": float(
                            profile.get("steady_backward_latency_ms")
                            or profile.get("backward_latency_ms")
                            or 0.0
                        ),
                        "compile_latency_ms": float(
                            profile.get("compile_latency_ms") or 0.0
                        ),
                        "peak_memory_bytes": int(
                            profile.get("peak_memory_bytes") or 0
                        ),
                        "sample_count": int(profile.get("sample_count") or 0),
                    }
                )
        operation_counts: dict[str, int] = {}
        for event in status["events"]:
            operation = str(event.get("operation") or "unknown")
            operation_counts[operation] = operation_counts.get(operation, 0) + 1
        now = float(status["status_observed_at"])
        elapsed = max(0.001, now - float(status["run_created_at"]))
        last_commit_at = commit_times[-1] if commit_times else 0.0
        transfer = self._tensor_metric_summary()
        snapshot = {
            "schema": "crowdtensor_heterogeneous_training_metrics_v1",
            "run_id_hash": status["run_id_hash"],
            "runtime_state": status["runtime_state"],
            "committed_step": int(status["committed_step"]),
            "target_steps": int(status["target_steps"]),
            "step_throughput_per_second": int(status["committed_step"]) / elapsed,
            "step_latency_seconds": {
                "sample_count": len(step_durations),
                "p50": _percentile(step_durations, 0.5),
                "p95": _percentile(step_durations, 0.95),
                "maximum": max(step_durations or [0.0]),
            },
            "checkpoint_age_seconds": (
                max(0.0, now - last_commit_at) if last_commit_at else elapsed
            ),
            "placement_generation": int(status["placement_generation"]),
            "reassignment_count": max(0, int(status["placement_generation"]) - 1),
            "worker_states": worker_states,
            "accelerator_worker_counts": accelerator_states,
            "stage_profiles": sorted(
                profiles,
                key=lambda item: (
                    item["stage_id"], item["accelerator"], item["device_id"]
                ),
            ),
            "queue": {
                "active_epoch": status["active_epoch_id"] is not None,
                "missing_stage_count": len(status["missing_stage_ids"]),
                "paused": bool(
                    status["paused_waiting_for_miners"]
                    or status["owner_paused"]
                ),
            },
            "operation_counts": {
                key: operation_counts[key] for key in sorted(operation_counts)
            },
            "retry_count": sum(
                count
                for operation, count in operation_counts.items()
                if "retry" in operation or "failure" in operation
            ),
            "device_health": status["device_health"],
            "transfer": transfer,
            "tensor_lookup": (
                self.tensor_store.lookup_performance_report()
                if self.tensor_store is not None
                else {}
            ),
            "coordinator_generation": int(status["coordinator_generation"]),
            "event_count": int(status["event_count"]),
            "low_cardinality_labels": True,
            "credential_values_public": False,
            "private_paths_public": False,
            "tensor_values_public": False,
            "public_artifact_safe": True,
        }
        snapshot["content_hash"] = stable_hash(snapshot)
        return snapshot

    def prometheus_metrics(self) -> str:
        snapshot = self.metrics_snapshot()
        lines = [
            "# HELP crowdtensor_training_committed_step Last atomically committed optimizer step.",
            "# TYPE crowdtensor_training_committed_step gauge",
            f"crowdtensor_training_committed_step {snapshot['committed_step']}",
            "# HELP crowdtensor_training_target_steps Configured optimizer step target.",
            "# TYPE crowdtensor_training_target_steps gauge",
            f"crowdtensor_training_target_steps {snapshot['target_steps']}",
            "# HELP crowdtensor_training_step_throughput_per_second Committed steps per wall-clock second.",
            "# TYPE crowdtensor_training_step_throughput_per_second gauge",
            f"crowdtensor_training_step_throughput_per_second {snapshot['step_throughput_per_second']:.12g}",
            "# HELP crowdtensor_training_step_latency_seconds Commit interval quantiles.",
            "# TYPE crowdtensor_training_step_latency_seconds gauge",
            f"crowdtensor_training_step_latency_seconds{{quantile=\"0.5\"}} {snapshot['step_latency_seconds']['p50']:.12g}",
            f"crowdtensor_training_step_latency_seconds{{quantile=\"0.95\"}} {snapshot['step_latency_seconds']['p95']:.12g}",
            "# HELP crowdtensor_training_checkpoint_age_seconds Age of the latest committed checkpoint set.",
            "# TYPE crowdtensor_training_checkpoint_age_seconds gauge",
            f"crowdtensor_training_checkpoint_age_seconds {snapshot['checkpoint_age_seconds']:.12g}",
            "# HELP crowdtensor_training_placement_generation Current fenced placement generation.",
            "# TYPE crowdtensor_training_placement_generation gauge",
            f"crowdtensor_training_placement_generation {snapshot['placement_generation']}",
            "# HELP crowdtensor_training_worker_count Workers by accelerator and state.",
            "# TYPE crowdtensor_training_worker_count gauge",
        ]
        status = self.public_status(reconcile=False)
        counts: dict[tuple[str, str], int] = {}
        for miner in status["miners"]:
            key = (str(miner["accelerator"]), str(miner["state"]))
            counts[key] = counts.get(key, 0) + 1
        for (accelerator, state), count in sorted(counts.items()):
            lines.append(
                "crowdtensor_training_worker_count"
                f'{{accelerator="{accelerator}",state="{state}"}} {count}'
            )
        lines.extend(
            [
                "# HELP crowdtensor_training_transfer_bytes Tensor bytes by direction; values never exposed.",
                "# TYPE crowdtensor_training_transfer_bytes counter",
            ]
        )
        for direction, row in snapshot["transfer"]["directions"].items():
            safe_direction = re.sub(r"[^a-zA-Z0-9_]", "_", direction)
            lines.append(
                "crowdtensor_training_transfer_bytes"
                f'{{direction="{safe_direction}"}} {int(row["bytes"])}'
            )
        return "\n".join(lines) + "\n"

    def audit_checkpoint_integrity(self) -> dict[str, Any]:
        """Find the newest complete retained checkpoint set without mutating progress."""

        with _connect(self.state_path) as connection:
            job = self._job(connection)
            commits = connection.execute(
                """
                SELECT * FROM commits WHERE run_id=? ORDER BY target_step DESC
                """,
                (self.run_id,),
            ).fetchall()
            rows_by_step: dict[int, list[sqlite3.Row]] = {}
            for row in connection.execute(
                """
                SELECT s.*,a.placement_generation FROM submissions s
                JOIN assignments a ON a.run_id=s.run_id AND a.epoch_id=s.epoch_id
                    AND a.stage_id=s.stage_id
                WHERE s.run_id=? AND s.state='committed'
                ORDER BY s.target_step DESC,s.stage_id
                """,
                (self.run_id,),
            ).fetchall():
                rows_by_step.setdefault(int(row["target_step"]), []).append(row)
        audits = []
        latest_valid_step = 0
        for commit in commits:
            step = int(commit["target_step"])
            rows = rows_by_step.get(step, [])
            errors = []
            if len(rows) != int(job["stage_count"]):
                errors.append("checkpoint_stage_coverage_missing")
            for row in rows:
                try:
                    archive = self.blob_store.get(str(row["archive_hash"]))
                    if self.training_manifest is not None:
                        validate_stage_checkpoint_archive(
                            archive,
                            training_manifest=self.training_manifest,
                            expected_stage_id=int(row["stage_id"]),
                            expected_step=step,
                            expected_dataset_cursor=int(row["dataset_cursor"]),
                            expected_placement_generation=int(
                                row["placement_generation"]
                            ),
                            max_checkpoint_bytes=self.max_checkpoint_bytes,
                            validate_tensor_payloads=self.validate_checkpoint_tensors,
                        )
                    elif _sha256_bytes(archive) != str(row["archive_hash"]):
                        raise ValueError("elastic_checkpoint_archive_hash_invalid")
                except (KeyError, OSError, RuntimeError, ValueError):
                    errors.append(
                        f"checkpoint_stage_{int(row['stage_id'])}_invalid"
                    )
            valid = not errors
            if valid and not latest_valid_step:
                latest_valid_step = step
            audits.append(
                {
                    "target_step": step,
                    "valid": valid,
                    "stage_count": len(rows),
                    "errors": sorted(set(errors)),
                }
            )
        committed_step = int(job["committed_step"])
        report = {
            "schema": "crowdtensor_elastic_checkpoint_integrity_audit_v1",
            "ok": committed_step == 0 or latest_valid_step > 0,
            "committed_step": committed_step,
            "latest_valid_checkpoint_step": latest_valid_step,
            "latest_checkpoint_valid": committed_step == latest_valid_step,
            "fallback_required": bool(
                committed_step > 0 and 0 < latest_valid_step < committed_step
            ),
            "fallback_step": (
                latest_valid_step
                if committed_step > 0 and latest_valid_step < committed_step
                else 0
            ),
            "checkpoint_sets": audits,
            "checkpoint_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        report["content_hash"] = stable_hash(report)
        return report

    def cleanup_uncommitted_blobs(self) -> dict[str, Any]:
        with _connect(self.state_path) as connection:
            retained = {
                str(row["archive_hash"]).split(":", 1)[-1]
                for row in connection.execute(
                    """
                    SELECT archive_hash FROM submissions
                    WHERE run_id=? AND state='committed'
                    """,
                    (self.run_id,),
                ).fetchall()
            }
        removed = 0
        removed_bytes = 0
        for archive_hash in list(self.blob_store.list_hashes()):
            if archive_hash.split(":", 1)[-1] in retained:
                continue
            try:
                value = self.blob_store.get(archive_hash)
                if self.blob_store.delete(archive_hash):
                    removed_bytes += len(value)
                    removed += 1
            except RuntimeError:
                pass
        return {
            "schema": "crowdtensor_elastic_training_blob_cleanup_v1",
            "ok": True,
            "uncommitted_blob_count_removed": removed,
            "uncommitted_blob_bytes_removed": removed_bytes,
            "committed_blob_count_retained": len(retained),
            "private_paths_public": False,
            "public_artifact_safe": True,
        }

    def enforce_checkpoint_retention(self) -> dict[str, Any]:
        with _connect(self.state_path) as connection:
            retained_steps = [
                int(row["target_step"])
                for row in connection.execute(
                    """
                    SELECT target_step FROM commits WHERE run_id=?
                    ORDER BY target_step DESC LIMIT ?
                    """,
                    (self.run_id, self.checkpoint_retention_steps),
                ).fetchall()
            ]
            retained = {
                str(row["archive_hash"])
                for row in connection.execute(
                    """
                    SELECT archive_hash FROM submissions
                    WHERE run_id=? AND (
                        state='candidate' OR
                        (state='committed' AND target_step IN (
                            SELECT target_step FROM commits WHERE run_id=?
                            ORDER BY target_step DESC LIMIT ?
                        ))
                    )
                    """,
                    (self.run_id, self.run_id, self.checkpoint_retention_steps),
                ).fetchall()
            }
        removed = 0
        removed_bytes = 0
        for archive_hash in list(self.blob_store.list_hashes()):
            if archive_hash in retained:
                continue
            value = self.blob_store.get(archive_hash)
            if self.blob_store.delete(archive_hash):
                removed += 1
                removed_bytes += len(value)
        return {
            "schema": "crowdtensor_elastic_training_checkpoint_retention_v1",
            "ok": True,
            "retention_committed_steps": self.checkpoint_retention_steps,
            "retained_committed_steps": sorted(retained_steps),
            "retained_blob_count": len(retained),
            "removed_blob_count": removed,
            "removed_blob_bytes": removed_bytes,
            "storage_backend": self.blob_store.backend,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }


def elastic_training_status(
    state_path: str | Path,
    *,
    run_id: str,
) -> dict[str, Any]:
    return ElasticTrainingRuntime.open_existing(
        state_path, run_id=run_id
    ).public_status()


def install_elastic_training_routes(
    app: Any,
    *,
    runtime: ElasticTrainingRuntime,
    authorize: Callable[[str | None], None],
) -> None:
    """Mount authenticated Miner routes on a Coordinator FastAPI app."""

    from fastapi import Header, HTTPException

    def mapped_error(exc: BaseException) -> HTTPException:
        detail = str(exc)
        if isinstance(exc, KeyError):
            return HTTPException(status_code=404, detail=detail.strip("'"))
        if "size" in detail:
            return HTTPException(status_code=413, detail=detail)
        if any(word in detail for word in ("stale", "conflict", "aborted")):
            return HTTPException(status_code=409, detail=detail)
        return HTTPException(status_code=422, detail=detail)

    @app.post("/elastic-training/miners/register")
    def register(
        request: ElasticMinerRegistrationRequest,
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        if request.run_id != runtime.run_id:
            raise HTTPException(status_code=422, detail="elastic_run_id_mismatch")
        try:
            return runtime.register_miner(
                miner_id_hash=request.miner_id_hash,
                registration_nonce=request.registration_nonce,
                supported_stage_ids=request.supported_stage_ids,
                slot_count=request.slot_count,
                accelerator=request.accelerator,
                capability=request.capability,
            )
        except (ValueError, RuntimeError, KeyError) as exc:
            raise mapped_error(exc) from exc

    @app.get("/elastic-training/capabilities")
    def capabilities(
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        return runtime.capabilities()

    @app.post("/elastic-training/miners/{session_id}/heartbeat")
    def heartbeat(
        session_id: str,
        x_crowdtensor_miner_token: str | None = Header(default=None),
        x_crowdtensor_elastic_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        try:
            return runtime.heartbeat(
                session_id=session_id,
                session_token=str(x_crowdtensor_elastic_session_token or ""),
            )
        except (ValueError, RuntimeError, KeyError) as exc:
            raise mapped_error(exc) from exc

    @app.get("/elastic-training/miners/{session_id}/assignments")
    def assignments(
        session_id: str,
        x_crowdtensor_miner_token: str | None = Header(default=None),
        x_crowdtensor_elastic_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        try:
            return runtime.assignments(
                session_id=session_id,
                session_token=str(x_crowdtensor_elastic_session_token or ""),
            )
        except (ValueError, RuntimeError, KeyError) as exc:
            raise mapped_error(exc) from exc

    @app.post("/elastic-training/miners/{session_id}/offline")
    def offline(
        session_id: str,
        x_crowdtensor_miner_token: str | None = Header(default=None),
        x_crowdtensor_elastic_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        try:
            return runtime.mark_offline(
                session_id=session_id,
                session_token=str(x_crowdtensor_elastic_session_token or ""),
            )
        except (ValueError, RuntimeError, KeyError) as exc:
            raise mapped_error(exc) from exc

    @app.post("/elastic-training/miners/{session_id}/stage-runtime")
    def stage_runtime(
        session_id: str,
        request: ElasticStageRuntimeReportRequest,
        x_crowdtensor_miner_token: str | None = Header(default=None),
        x_crowdtensor_elastic_session_token: str | None = Header(default=None),
        x_crowdtensor_elastic_assignment_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        try:
            return runtime.report_stage_runtime(
                session_id=session_id,
                session_token=str(x_crowdtensor_elastic_session_token or ""),
                assignment_token=str(x_crowdtensor_elastic_assignment_token or ""),
                **request.model_dump(),
            )
        except (ValueError, RuntimeError, KeyError) as exc:
            raise mapped_error(exc) from exc

    @app.post("/elastic-training/miners/{session_id}/telemetry")
    def device_telemetry(
        session_id: str,
        request: ElasticDeviceTelemetryRequest,
        x_crowdtensor_miner_token: str | None = Header(default=None),
        x_crowdtensor_elastic_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        try:
            return runtime.report_device_telemetry(
                session_id=session_id,
                session_token=str(x_crowdtensor_elastic_session_token or ""),
                **request.model_dump(),
            )
        except (ValueError, RuntimeError, KeyError) as exc:
            raise mapped_error(exc) from exc

    @app.post("/elastic-training/miners/{session_id}/device-failure")
    def device_failure(
        session_id: str,
        request: ElasticDeviceFailureRequest,
        x_crowdtensor_miner_token: str | None = Header(default=None),
        x_crowdtensor_elastic_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        try:
            return runtime.record_device_failure(
                session_id=session_id,
                session_token=str(x_crowdtensor_elastic_session_token or ""),
                **request.model_dump(),
            )
        except (ValueError, RuntimeError, KeyError) as exc:
            raise mapped_error(exc) from exc

    @app.post("/elastic-training/tensors/begin")
    def tensor_begin(
        envelope: dict[str, Any],
        x_crowdtensor_miner_token: str | None = Header(default=None),
        x_crowdtensor_elastic_session_id: str | None = Header(default=None),
        x_crowdtensor_elastic_session_token: str | None = Header(default=None),
        x_crowdtensor_elastic_assignment_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        try:
            return runtime.begin_tensor_message(
                session_id=str(x_crowdtensor_elastic_session_id or ""),
                session_token=str(x_crowdtensor_elastic_session_token or ""),
                assignment_token=str(x_crowdtensor_elastic_assignment_token or ""),
                envelope=envelope,
            )
        except (ValueError, RuntimeError, KeyError, TensorTransportError) as exc:
            raise mapped_error(exc) from exc

    @app.post("/elastic-training/tensors/inline")
    def tensor_inline_upload(
        request: ElasticInlineTensorUploadRequest,
        x_crowdtensor_miner_token: str | None = Header(default=None),
        x_crowdtensor_elastic_session_id: str | None = Header(default=None),
        x_crowdtensor_elastic_session_token: str | None = Header(default=None),
        x_crowdtensor_elastic_assignment_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        try:
            chunk = base64.b64decode(request.chunk_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail="elastic_inline_tensor_payload_invalid"
            ) from exc
        if len(chunk) > 4 * 1024 * 1024:
            raise HTTPException(
                status_code=413, detail="heterogeneous_tensor_chunk_size_invalid"
            )
        envelope = dict(request.envelope)
        if int(envelope.get("chunk_count") or 0) != 1:
            raise HTTPException(
                status_code=422, detail="elastic_inline_tensor_single_chunk_required"
            )
        try:
            common = {
                "session_id": str(x_crowdtensor_elastic_session_id or ""),
                "session_token": str(x_crowdtensor_elastic_session_token or ""),
                "assignment_token": str(
                    x_crowdtensor_elastic_assignment_token or ""
                ),
            }
            begin = runtime.begin_tensor_message(envelope=envelope, **common)
            uploaded = runtime.put_tensor_chunk(
                message_id=str(envelope.get("message_id") or ""),
                chunk_index=0,
                value=chunk,
                **common,
            )
            return {
                "schema": "crowdtensor_elastic_inline_tensor_upload_v1",
                "complete": uploaded.get("complete") is True,
                "idempotent": bool(
                    begin.get("complete") is True
                    or uploaded.get("idempotent") is True
                ),
                "inline_payload": True,
                "tensor_values_public": False,
                "public_artifact_safe": True,
            }
        except (ValueError, RuntimeError, KeyError, TensorTransportError) as exc:
            raise mapped_error(exc) from exc

    @app.put("/elastic-training/tensors/{message_id}/{chunk_index}")
    async def tensor_chunk_upload(
        message_id: str,
        chunk_index: int,
        request: FastAPIRequest,
        x_crowdtensor_miner_token: str | None = Header(default=None),
        x_crowdtensor_elastic_session_id: str | None = Header(default=None),
        x_crowdtensor_elastic_session_token: str | None = Header(default=None),
        x_crowdtensor_elastic_assignment_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 4 * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail="heterogeneous_tensor_chunk_size_invalid",
                )
        try:
            return runtime.put_tensor_chunk(
                session_id=str(x_crowdtensor_elastic_session_id or ""),
                session_token=str(x_crowdtensor_elastic_session_token or ""),
                assignment_token=str(x_crowdtensor_elastic_assignment_token or ""),
                message_id=message_id,
                chunk_index=chunk_index,
                value=bytes(body),
            )
        except (ValueError, RuntimeError, KeyError, TensorTransportError) as exc:
            raise mapped_error(exc) from exc

    @app.get("/elastic-training/tensors/lookup")
    def tensor_lookup(
        global_step: int,
        microbatch_id: int,
        source_stage_id: int,
        target_stage_id: int,
        direction: str,
        placement_generation: int,
        x_crowdtensor_miner_token: str | None = Header(default=None),
        x_crowdtensor_elastic_session_id: str | None = Header(default=None),
        x_crowdtensor_elastic_session_token: str | None = Header(default=None),
        x_crowdtensor_elastic_assignment_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        try:
            return runtime.find_tensor_message(
                session_id=str(x_crowdtensor_elastic_session_id or ""),
                session_token=str(x_crowdtensor_elastic_session_token or ""),
                assignment_token=str(x_crowdtensor_elastic_assignment_token or ""),
                global_step=global_step,
                microbatch_id=microbatch_id,
                source_stage_id=source_stage_id,
                target_stage_id=target_stage_id,
                direction=direction,
                placement_generation=placement_generation,
            )
        except (ValueError, RuntimeError, KeyError, TensorTransportError) as exc:
            raise mapped_error(exc) from exc

    @app.get("/elastic-training/tensors/inline")
    def tensor_inline_download(
        global_step: int,
        microbatch_id: int,
        source_stage_id: int,
        target_stage_id: int,
        direction: str,
        placement_generation: int,
        x_crowdtensor_miner_token: str | None = Header(default=None),
        x_crowdtensor_elastic_session_id: str | None = Header(default=None),
        x_crowdtensor_elastic_session_token: str | None = Header(default=None),
        x_crowdtensor_elastic_assignment_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        common = {
            "session_id": str(x_crowdtensor_elastic_session_id or ""),
            "session_token": str(x_crowdtensor_elastic_session_token or ""),
            "assignment_token": str(x_crowdtensor_elastic_assignment_token or ""),
        }
        try:
            found = runtime.find_tensor_message(
                global_step=global_step,
                microbatch_id=microbatch_id,
                source_stage_id=source_stage_id,
                target_stage_id=target_stage_id,
                direction=direction,
                placement_generation=placement_generation,
                **common,
            )
            envelope = dict(found.get("envelope") or {})
            if (
                found.get("found") is not True
                or (found.get("status") or {}).get("complete") is not True
                or int(envelope.get("chunk_count") or 0) != 1
            ):
                return {**found, "inline_payload": False}
            chunk, _verified_envelope = runtime.read_tensor_chunk(
                message_id=str(envelope["message_id"]),
                chunk_index=0,
                **common,
            )
            return {
                **found,
                "chunk_b64": base64.b64encode(chunk).decode("ascii"),
                "inline_payload": True,
                "tensor_values_public": False,
                "public_artifact_safe": True,
            }
        except (ValueError, RuntimeError, KeyError, TensorTransportError) as exc:
            raise mapped_error(exc) from exc

    @app.get("/elastic-training/tensors/{message_id}/{chunk_index}")
    def tensor_chunk_download(
        message_id: str,
        chunk_index: int,
        x_crowdtensor_miner_token: str | None = Header(default=None),
        x_crowdtensor_elastic_session_id: str | None = Header(default=None),
        x_crowdtensor_elastic_session_token: str | None = Header(default=None),
        x_crowdtensor_elastic_assignment_token: str | None = Header(default=None),
    ) -> FastAPIResponse:
        authorize(x_crowdtensor_miner_token)
        try:
            value, envelope = runtime.read_tensor_chunk(
                session_id=str(x_crowdtensor_elastic_session_id or ""),
                session_token=str(x_crowdtensor_elastic_session_token or ""),
                assignment_token=str(x_crowdtensor_elastic_assignment_token or ""),
                message_id=message_id,
                chunk_index=chunk_index,
            )
        except (ValueError, RuntimeError, KeyError, TensorTransportError) as exc:
            raise mapped_error(exc) from exc
        return FastAPIResponse(
            content=value,
            media_type="application/octet-stream",
            headers={
                "x-crowdtensor-tensor-message-id": str(envelope["message_id"]),
                "x-crowdtensor-tensor-chunk-index": str(chunk_index),
                "x-crowdtensor-tensor-chunk-hash": str(
                    envelope["chunk_hashes"][int(chunk_index)]
                ),
            },
        )

    @app.post("/elastic-training/checkpoints/{epoch_id}/{stage_id}")
    async def submit_checkpoint(
        epoch_id: int,
        stage_id: int,
        request: FastAPIRequest,
        x_crowdtensor_miner_token: str | None = Header(default=None),
        x_crowdtensor_elastic_session_id: str | None = Header(default=None),
        x_crowdtensor_elastic_session_token: str | None = Header(default=None),
        x_crowdtensor_elastic_assignment_token: str | None = Header(default=None),
        x_crowdtensor_checkpoint_signature: str | None = Header(default=None),
    ) -> dict[str, Any]:
        from starlette.concurrency import run_in_threadpool

        authorize(x_crowdtensor_miner_token)
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                declared_length = int(content_length)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="elastic_checkpoint_content_length_invalid",
                ) from exc
            if declared_length < 1 or declared_length > runtime.max_checkpoint_bytes:
                raise HTTPException(
                    status_code=413,
                    detail="elastic_checkpoint_archive_size_invalid",
                )
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > runtime.max_checkpoint_bytes:
                raise HTTPException(
                    status_code=413,
                    detail="elastic_checkpoint_archive_size_invalid",
                )
        archive = bytes(body)
        try:
            result = await run_in_threadpool(
                runtime.submit_checkpoint,
                session_id=str(x_crowdtensor_elastic_session_id or ""),
                session_token=str(x_crowdtensor_elastic_session_token or ""),
                epoch_id=epoch_id,
                stage_id=stage_id,
                assignment_token=str(x_crowdtensor_elastic_assignment_token or ""),
                archive=archive,
                checkpoint_signature=str(x_crowdtensor_checkpoint_signature or ""),
            )
        except (ValueError, RuntimeError, KeyError) as exc:
            raise mapped_error(exc) from exc
        if result.get("global_commit_created") is True:
            result["checkpoint_retention"] = await run_in_threadpool(
                runtime.enforce_checkpoint_retention
            )
        return result

    @app.get("/elastic-training/checkpoints/{epoch_id}/{stage_id}")
    def download_checkpoint(
        epoch_id: int,
        stage_id: int,
        x_crowdtensor_miner_token: str | None = Header(default=None),
        x_crowdtensor_elastic_session_id: str | None = Header(default=None),
        x_crowdtensor_elastic_session_token: str | None = Header(default=None),
        x_crowdtensor_elastic_assignment_token: str | None = Header(default=None),
    ) -> FastAPIResponse:
        authorize(x_crowdtensor_miner_token)
        try:
            archive, report = runtime.download_committed_checkpoint(
                session_id=str(x_crowdtensor_elastic_session_id or ""),
                session_token=str(x_crowdtensor_elastic_session_token or ""),
                epoch_id=epoch_id,
                stage_id=stage_id,
                assignment_token=str(x_crowdtensor_elastic_assignment_token or ""),
            )
        except (ValueError, RuntimeError, KeyError) as exc:
            raise mapped_error(exc) from exc
        return FastAPIResponse(
            content=archive,
            media_type="application/vnd.crowdtensor.stage-checkpoint+zip",
            headers={
                "x-crowdtensor-checkpoint-hash": str(report["archive_hash"]),
                "x-crowdtensor-global-step": str(report["global_step"]),
                "x-crowdtensor-dataset-cursor": str(report["dataset_cursor"]),
            },
        )

    @app.get("/elastic-training/barriers/{epoch_id}")
    def barrier(
        epoch_id: int,
        x_crowdtensor_miner_token: str | None = Header(default=None),
        x_crowdtensor_elastic_session_id: str | None = Header(default=None),
        x_crowdtensor_elastic_session_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        try:
            return runtime.barrier_status(
                session_id=str(x_crowdtensor_elastic_session_id or ""),
                session_token=str(x_crowdtensor_elastic_session_token or ""),
                epoch_id=epoch_id,
            )
        except (ValueError, RuntimeError, KeyError) as exc:
            raise mapped_error(exc) from exc

    @app.get("/elastic-training/status")
    def status(
        x_crowdtensor_miner_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(x_crowdtensor_miner_token)
        return runtime.public_status()
