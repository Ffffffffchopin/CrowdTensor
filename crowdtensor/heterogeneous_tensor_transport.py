"""Safe chunked tensor transport for heterogeneous pipeline training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from .heterogeneous_training_manifest import stable_hash


ENVELOPE_SCHEMA = "crowdtensor_heterogeneous_tensor_envelope_v1"
STORE_SCHEMA = "crowdtensor_heterogeneous_tensor_store_v1"
DIRECTIONS = {"forward_activation", "backward_gradient"}
ALLOWED_DTYPES = {"float16", "bfloat16", "float32"}
MESSAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class TensorTransportError(RuntimeError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_tensors() -> tuple[Any, Any]:
    try:
        from safetensors.torch import load, save
    except ImportError as exc:
        raise TensorTransportError(
            "heterogeneous_tensor_transport_requires_safetensors"
        ) from exc
    return load, save


def _tensor_spec(name: str, tensor: Any) -> dict[str, Any]:
    dtype = str(tensor.dtype).replace("torch.", "")
    if dtype not in ALLOWED_DTYPES:
        raise TensorTransportError("heterogeneous_tensor_dtype_not_allowed")
    shape = [int(value) for value in tensor.shape]
    if len(shape) > 8 or any(value < 1 or value > 1_000_000 for value in shape):
        raise TensorTransportError("heterogeneous_tensor_shape_invalid")
    element_count = int(math.prod(shape)) if shape else 1
    if element_count > 256_000_000:
        raise TensorTransportError("heterogeneous_tensor_element_limit_exceeded")
    return {
        "name": str(name),
        "dtype": dtype,
        "shape": shape,
        "element_count": element_count,
        "byte_count": element_count * int(tensor.element_size()),
    }


def _canonical_tensors(tensors: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        import torch
    except ImportError as exc:
        raise TensorTransportError("heterogeneous_tensor_transport_requires_torch") from exc
    if not isinstance(tensors, dict) or not 1 <= len(tensors) <= 8:
        raise TensorTransportError("heterogeneous_tensor_count_invalid")
    canonical: dict[str, Any] = {}
    specs = []
    for name in sorted(tensors):
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_.-]{0,63}", str(name)):
            raise TensorTransportError("heterogeneous_tensor_name_invalid")
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            try:
                import jax
                import numpy as np

                if not isinstance(tensor, jax.Array):
                    raise TypeError
                dtype_name = str(tensor.dtype).lower()
                if dtype_name not in ALLOWED_DTYPES:
                    raise TensorTransportError(
                        "heterogeneous_tensor_dtype_not_allowed"
                    )
                host = np.asarray(jax.device_get(tensor), dtype=np.float32)
                tensor = torch.from_numpy(host.copy()).to(getattr(torch, dtype_name))
            except (ImportError, TypeError):
                raise TensorTransportError(
                    "heterogeneous_tensor_value_invalid"
                ) from None
        value = tensor.detach().to("cpu").contiguous()
        if not bool(torch.isfinite(value.float()).all().item()):
            raise TensorTransportError("heterogeneous_tensor_non_finite")
        canonical[str(name)] = value
        specs.append(_tensor_spec(str(name), value))
    return canonical, specs


def _envelope_identity(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "schema",
            "job_id",
            "manifest_hash",
            "global_step",
            "microbatch_id",
            "source_stage_id",
            "target_stage_id",
            "direction",
            "placement_generation",
            "assignment_token_hash",
            "tensor_specs",
            "payload_hash",
            "payload_bytes",
            "chunk_bytes",
            "chunk_count",
            "chunk_hashes",
            "created_at",
            "expires_at",
            "max_delivery_attempts",
        )
    }


def validate_tensor_envelope(
    value: Any,
    *,
    max_payload_bytes: int = 128 * 1024 * 1024,
    max_chunk_bytes: int = 4 * 1024 * 1024,
) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != ENVELOPE_SCHEMA:
        raise TensorTransportError("heterogeneous_tensor_envelope_schema_invalid")
    envelope = dict(value)
    job_id = str(envelope.get("job_id") or "")
    manifest_hash = str(envelope.get("manifest_hash") or "")
    assignment_hash = str(envelope.get("assignment_token_hash") or "")
    if not job_id or not manifest_hash.startswith("sha256:") or not assignment_hash.startswith(
        "sha256:"
    ):
        raise TensorTransportError("heterogeneous_tensor_envelope_identity_invalid")
    try:
        global_step = int(envelope["global_step"])
        microbatch_id = int(envelope["microbatch_id"])
        source_stage = int(envelope["source_stage_id"])
        target_stage = int(envelope["target_stage_id"])
        generation = int(envelope["placement_generation"])
        payload_bytes = int(envelope["payload_bytes"])
        chunk_bytes = int(envelope["chunk_bytes"])
        chunk_count = int(envelope["chunk_count"])
        created_at = float(envelope["created_at"])
        expires_at = float(envelope["expires_at"])
        max_attempts = int(envelope["max_delivery_attempts"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TensorTransportError(
            "heterogeneous_tensor_envelope_position_invalid"
        ) from exc
    direction = str(envelope.get("direction") or "")
    if (
        global_step < 1
        or microbatch_id < 0
        or source_stage < 0
        or target_stage < 0
        or generation < 1
        or direction not in DIRECTIONS
        or expires_at <= created_at
        or max_attempts < 1
        or max_attempts > 10
    ):
        raise TensorTransportError("heterogeneous_tensor_envelope_position_invalid")
    if direction == "forward_activation" and target_stage != source_stage + 1:
        raise TensorTransportError("heterogeneous_tensor_forward_route_invalid")
    if direction == "backward_gradient" and target_stage != source_stage - 1:
        raise TensorTransportError("heterogeneous_tensor_backward_route_invalid")
    if (
        payload_bytes < 1
        or payload_bytes > int(max_payload_bytes)
        or chunk_bytes < 1
        or chunk_bytes > int(max_chunk_bytes)
        or chunk_count != math.ceil(payload_bytes / chunk_bytes)
    ):
        raise TensorTransportError("heterogeneous_tensor_envelope_size_invalid")
    specs = envelope.get("tensor_specs")
    if not isinstance(specs, list) or not 1 <= len(specs) <= 8:
        raise TensorTransportError("heterogeneous_tensor_envelope_specs_invalid")
    canonical_specs = []
    names = set()
    for spec in specs:
        if not isinstance(spec, dict):
            raise TensorTransportError("heterogeneous_tensor_envelope_specs_invalid")
        name = str(spec.get("name") or "")
        dtype = str(spec.get("dtype") or "")
        shape = spec.get("shape")
        if (
            not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_.-]{0,63}", name)
            or name in names
            or dtype not in ALLOWED_DTYPES
            or not isinstance(shape, list)
            or len(shape) > 8
            or any(not isinstance(item, int) or item < 1 for item in shape)
        ):
            raise TensorTransportError("heterogeneous_tensor_envelope_specs_invalid")
        names.add(name)
        canonical_specs.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": [int(item) for item in shape],
                "element_count": int(spec.get("element_count") or 0),
                "byte_count": int(spec.get("byte_count") or 0),
            }
        )
    chunk_hashes = [str(item) for item in envelope.get("chunk_hashes") or []]
    if len(chunk_hashes) != chunk_count or any(
        not MESSAGE_ID_PATTERN.fullmatch(item) for item in chunk_hashes
    ):
        raise TensorTransportError("heterogeneous_tensor_chunk_hashes_invalid")
    payload_hash = str(envelope.get("payload_hash") or "")
    if not MESSAGE_ID_PATTERN.fullmatch(payload_hash):
        raise TensorTransportError("heterogeneous_tensor_payload_hash_invalid")
    canonical = {
        "schema": ENVELOPE_SCHEMA,
        "job_id": job_id,
        "manifest_hash": manifest_hash,
        "global_step": global_step,
        "microbatch_id": microbatch_id,
        "source_stage_id": source_stage,
        "target_stage_id": target_stage,
        "direction": direction,
        "placement_generation": generation,
        "assignment_token_hash": assignment_hash,
        "tensor_specs": canonical_specs,
        "payload_hash": payload_hash,
        "payload_bytes": payload_bytes,
        "chunk_bytes": chunk_bytes,
        "chunk_count": chunk_count,
        "chunk_hashes": chunk_hashes,
        "created_at": created_at,
        "expires_at": expires_at,
        "max_delivery_attempts": max_attempts,
        "tensor_values_public": False,
        "public_artifact_safe": True,
    }
    message_id = stable_hash(_envelope_identity(canonical))
    supplied_message_id = str(envelope.get("message_id") or "")
    if supplied_message_id and supplied_message_id != message_id:
        raise TensorTransportError("heterogeneous_tensor_message_id_mismatch")
    canonical["message_id"] = message_id
    content_hash = stable_hash(canonical)
    supplied_content_hash = str(envelope.get("content_hash") or "")
    if supplied_content_hash and supplied_content_hash != content_hash:
        raise TensorTransportError("heterogeneous_tensor_envelope_hash_mismatch")
    canonical["content_hash"] = content_hash
    return canonical


def encode_tensor_message(
    tensors: dict[str, Any],
    *,
    job_id: str,
    manifest_hash: str,
    global_step: int,
    microbatch_id: int,
    source_stage_id: int,
    target_stage_id: int,
    direction: str,
    placement_generation: int,
    assignment_token_hash: str,
    chunk_bytes: int = 4 * 1024 * 1024,
    max_payload_bytes: int = 128 * 1024 * 1024,
    ttl_seconds: float = 300.0,
    max_delivery_attempts: int = 3,
    clock: Callable[[], float] = time.time,
) -> tuple[dict[str, Any], list[bytes]]:
    """Serialize only finite tensors into safetensors and bounded chunks."""

    _load, save = _safe_tensors()
    canonical_tensors, specs = _canonical_tensors(tensors)
    payload = bytes(save(canonical_tensors))
    if not payload or len(payload) > int(max_payload_bytes):
        raise TensorTransportError("heterogeneous_tensor_payload_size_invalid")
    if int(chunk_bytes) < 1 or int(chunk_bytes) > 4 * 1024 * 1024:
        raise TensorTransportError("heterogeneous_tensor_chunk_size_invalid")
    chunks = [
        payload[offset : offset + int(chunk_bytes)]
        for offset in range(0, len(payload), int(chunk_bytes))
    ]
    now = float(clock())
    value = {
        "schema": ENVELOPE_SCHEMA,
        "job_id": str(job_id),
        "manifest_hash": str(manifest_hash),
        "global_step": int(global_step),
        "microbatch_id": int(microbatch_id),
        "source_stage_id": int(source_stage_id),
        "target_stage_id": int(target_stage_id),
        "direction": str(direction),
        "placement_generation": int(placement_generation),
        "assignment_token_hash": str(assignment_token_hash),
        "tensor_specs": specs,
        "payload_hash": _sha256_bytes(payload),
        "payload_bytes": len(payload),
        "chunk_bytes": int(chunk_bytes),
        "chunk_count": len(chunks),
        "chunk_hashes": [_sha256_bytes(chunk) for chunk in chunks],
        "created_at": now,
        "expires_at": now + float(ttl_seconds),
        "max_delivery_attempts": int(max_delivery_attempts),
        "tensor_values_public": False,
        "public_artifact_safe": True,
    }
    return (
        validate_tensor_envelope(
            value,
            max_payload_bytes=max_payload_bytes,
            max_chunk_bytes=chunk_bytes,
        ),
        chunks,
    )


def decode_tensor_payload(
    payload: bytes,
    envelope: dict[str, Any],
    *,
    target_device: str = "cpu",
    target_dtype: str | None = None,
) -> dict[str, Any]:
    """Validate safetensors metadata and optionally move/cast the values."""

    load, _save = _safe_tensors()
    canonical = validate_tensor_envelope(envelope)
    if len(payload) != int(canonical["payload_bytes"]):
        raise TensorTransportError("heterogeneous_tensor_payload_length_mismatch")
    if _sha256_bytes(payload) != canonical["payload_hash"]:
        raise TensorTransportError("heterogeneous_tensor_payload_hash_mismatch")
    try:
        tensors = dict(load(payload))
    except Exception as exc:
        raise TensorTransportError("heterogeneous_tensor_safetensors_invalid") from exc
    actual_tensors, actual_specs = _canonical_tensors(tensors)
    if actual_specs != canonical["tensor_specs"]:
        raise TensorTransportError("heterogeneous_tensor_specs_mismatch")
    try:
        import torch
    except ImportError as exc:
        raise TensorTransportError("heterogeneous_tensor_transport_requires_torch") from exc
    device = torch.device(str(target_device))
    if device.type not in {"cpu", "cuda"}:
        raise TensorTransportError("heterogeneous_tensor_target_device_invalid")
    if device.type == "cuda" and (
        not torch.cuda.is_available() or int(device.index or 0) >= torch.cuda.device_count()
    ):
        raise TensorTransportError("heterogeneous_tensor_target_cuda_unavailable")
    dtype = None
    if target_dtype is not None:
        name = str(target_dtype).lower()
        if name not in ALLOWED_DTYPES:
            raise TensorTransportError("heterogeneous_tensor_target_dtype_invalid")
        dtype = getattr(torch, name)
    return {
        name: tensor.to(device=device, dtype=dtype or tensor.dtype).contiguous()
        for name, tensor in actual_tensors.items()
    }


def decode_tensor_payload_to_jax(
    payload: bytes,
    envelope: dict[str, Any],
    *,
    sharding: Any | None = None,
    target_dtype: str | None = None,
) -> dict[str, Any]:
    """Decode validated safetensors and place arrays on a JAX device/mesh."""

    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
    except ImportError as exc:
        raise TensorTransportError(
            "heterogeneous_tensor_transport_requires_jax"
        ) from exc
    tensors = decode_tensor_payload(payload, envelope, target_device="cpu")
    dtype_name = str(target_dtype or "").lower()
    if dtype_name and dtype_name not in ALLOWED_DTYPES:
        raise TensorTransportError("heterogeneous_tensor_target_dtype_invalid")
    dtype = getattr(jnp, dtype_name) if dtype_name else None
    result = {}
    for name, tensor in tensors.items():
        host = np.asarray(tensor.float().numpy(), dtype=np.float32)
        value = jnp.asarray(host, dtype=dtype or getattr(jnp, str(tensor.dtype).replace("torch.", "")))
        if sharding is not None:
            value = jax.device_put(value, sharding)
        if not bool(np.isfinite(np.asarray(jax.device_get(value), dtype=np.float32)).all()):
            raise TensorTransportError("heterogeneous_tensor_non_finite")
        result[name] = value
    return result


class ChunkedTensorStore:
    """Filesystem-backed bounded store with generation fencing and idempotency."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_payload_bytes: int = 128 * 1024 * 1024,
        max_chunk_bytes: int = 4 * 1024 * 1024,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.max_payload_bytes = int(max_payload_bytes)
        self.max_chunk_bytes = int(max_chunk_bytes)
        self._clock = clock
        self._lock = threading.RLock()
        self._delivery_attempts: dict[tuple[str, int], int] = {}
        self._consumers: dict[str, str] = {}
        self._lookup_index: dict[tuple[Any, ...], list[str]] = {}
        self._indexed_lookup_count = 0
        self._legacy_lookup_count = 0
        self._lookup_scanned_directory_count = 0
        self._rebuild_lookup_index()

    @staticmethod
    def _lookup_key(
        *,
        job_id: str,
        global_step: int,
        microbatch_id: int,
        source_stage_id: int,
        target_stage_id: int,
        direction: str,
        placement_generation: int,
    ) -> tuple[Any, ...]:
        return (
            str(job_id),
            int(global_step),
            int(microbatch_id),
            int(source_stage_id),
            int(target_stage_id),
            str(direction),
            int(placement_generation),
        )

    @classmethod
    def _envelope_lookup_key(cls, envelope: dict[str, Any]) -> tuple[Any, ...]:
        return cls._lookup_key(
            job_id=str(envelope["job_id"]),
            global_step=int(envelope["global_step"]),
            microbatch_id=int(envelope["microbatch_id"]),
            source_stage_id=int(envelope["source_stage_id"]),
            target_stage_id=int(envelope["target_stage_id"]),
            direction=str(envelope["direction"]),
            placement_generation=int(envelope["placement_generation"]),
        )

    def _rebuild_lookup_index(self) -> None:
        index: dict[tuple[Any, ...], list[str]] = {}
        for directory in self.root.iterdir():
            if not directory.is_dir() or not re.fullmatch(r"[0-9a-f]{64}", directory.name):
                continue
            message_id = "sha256:" + directory.name
            try:
                envelope = self._load_envelope(message_id)
            except TensorTransportError:
                continue
            index.setdefault(self._envelope_lookup_key(envelope), []).append(message_id)
        for values in index.values():
            values.sort()
        self._lookup_index = index

    @staticmethod
    def _digest(message_id: str) -> str:
        if not MESSAGE_ID_PATTERN.fullmatch(str(message_id)):
            raise TensorTransportError("heterogeneous_tensor_message_id_invalid")
        return str(message_id).split(":", 1)[1]

    def _message_dir(self, message_id: str) -> Path:
        return self.root / self._digest(message_id)

    def _load_envelope(self, message_id: str) -> dict[str, Any]:
        path = self._message_dir(message_id) / "envelope.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TensorTransportError("heterogeneous_tensor_message_not_found") from exc
        return validate_tensor_envelope(
            value,
            max_payload_bytes=self.max_payload_bytes,
            max_chunk_bytes=self.max_chunk_bytes,
        )

    def envelope(self, message_id: str) -> dict[str, Any]:
        """Return public-safe metadata; tensor bytes remain private."""

        return self._load_envelope(message_id)

    def read_chunk(self, message_id: str, chunk_index: int) -> bytes:
        envelope = self._load_envelope(message_id)
        index = int(chunk_index)
        if index < 0 or index >= int(envelope["chunk_count"]):
            raise TensorTransportError("heterogeneous_tensor_chunk_index_invalid")
        path = self._message_dir(message_id) / f"chunk-{index:08d}.bin"
        try:
            value = path.read_bytes()
        except OSError as exc:
            raise TensorTransportError("heterogeneous_tensor_chunk_not_found") from exc
        if _sha256_bytes(value) != envelope["chunk_hashes"][index]:
            raise TensorTransportError("heterogeneous_tensor_chunk_hash_mismatch")
        return value

    def find_message(
        self,
        *,
        job_id: str,
        global_step: int,
        microbatch_id: int,
        source_stage_id: int,
        target_stage_id: int,
        direction: str,
        placement_generation: int,
        use_index: bool = True,
    ) -> dict[str, Any] | None:
        matches = []
        with self._lock:
            if use_index:
                self._indexed_lookup_count += 1
                key = self._lookup_key(
                    job_id=job_id,
                    global_step=global_step,
                    microbatch_id=microbatch_id,
                    source_stage_id=source_stage_id,
                    target_stage_id=target_stage_id,
                    direction=direction,
                    placement_generation=placement_generation,
                )
                message_ids = list(self._lookup_index.get(key) or [])
            else:
                self._legacy_lookup_count += 1
                directories = [
                    item
                    for item in self.root.iterdir()
                    if item.is_dir()
                    and re.fullmatch(r"[0-9a-f]{64}", item.name)
                ]
                self._lookup_scanned_directory_count += len(directories)
                message_ids = ["sha256:" + item.name for item in directories]
            for message_id in message_ids:
                try:
                    envelope = self._load_envelope(message_id)
                except TensorTransportError:
                    continue
                if (
                    envelope["job_id"] == str(job_id)
                    and int(envelope["global_step"]) == int(global_step)
                    and int(envelope["microbatch_id"]) == int(microbatch_id)
                    and int(envelope["source_stage_id"]) == int(source_stage_id)
                    and int(envelope["target_stage_id"]) == int(target_stage_id)
                    and envelope["direction"] == str(direction)
                    and int(envelope["placement_generation"])
                    == int(placement_generation)
                ):
                    matches.append(envelope)
        if not matches:
            return None
        matches.sort(key=lambda item: (float(item["created_at"]), item["message_id"]))
        if len({item["payload_hash"] for item in matches}) > 1:
            raise TensorTransportError("heterogeneous_tensor_lookup_conflict")
        return matches[0]

    def begin(
        self,
        envelope: dict[str, Any],
        *,
        expected_generation: int,
    ) -> dict[str, Any]:
        canonical = validate_tensor_envelope(
            envelope,
            max_payload_bytes=self.max_payload_bytes,
            max_chunk_bytes=self.max_chunk_bytes,
        )
        if int(canonical["placement_generation"]) != int(expected_generation):
            raise TensorTransportError("heterogeneous_tensor_stale_generation")
        if float(canonical["expires_at"]) <= float(self._clock()):
            raise TensorTransportError("heterogeneous_tensor_message_expired")
        directory = self._message_dir(canonical["message_id"])
        with self._lock:
            if directory.is_dir():
                previous = self._load_envelope(canonical["message_id"])
                if previous["content_hash"] != canonical["content_hash"]:
                    raise TensorTransportError("heterogeneous_tensor_message_conflict")
                return self.status(canonical["message_id"])
            directory.mkdir(mode=0o700)
            _atomic_write(
                directory / "envelope.json",
                (json.dumps(canonical, sort_keys=True) + "\n").encode("utf-8"),
            )
            key = self._envelope_lookup_key(canonical)
            indexed = self._lookup_index.setdefault(key, [])
            if canonical["message_id"] not in indexed:
                indexed.append(canonical["message_id"])
                indexed.sort()
        return self.status(canonical["message_id"])

    def put_chunk(
        self,
        message_id: str,
        chunk_index: int,
        value: bytes,
        *,
        expected_generation: int,
    ) -> dict[str, Any]:
        envelope = self._load_envelope(message_id)
        if int(envelope["placement_generation"]) != int(expected_generation):
            raise TensorTransportError("heterogeneous_tensor_stale_generation")
        if float(envelope["expires_at"]) <= float(self._clock()):
            raise TensorTransportError("heterogeneous_tensor_message_expired")
        index = int(chunk_index)
        if index < 0 or index >= int(envelope["chunk_count"]):
            raise TensorTransportError("heterogeneous_tensor_chunk_index_invalid")
        chunk = bytes(value)
        expected_length = min(
            int(envelope["chunk_bytes"]),
            int(envelope["payload_bytes"]) - index * int(envelope["chunk_bytes"]),
        )
        if len(chunk) != expected_length:
            raise TensorTransportError("heterogeneous_tensor_chunk_length_invalid")
        if _sha256_bytes(chunk) != envelope["chunk_hashes"][index]:
            raise TensorTransportError("heterogeneous_tensor_chunk_hash_mismatch")
        key = (str(message_id), index)
        with self._lock:
            attempts = int(self._delivery_attempts.get(key, 0)) + 1
            self._delivery_attempts[key] = attempts
            if attempts > int(envelope["max_delivery_attempts"]):
                raise TensorTransportError("heterogeneous_tensor_retry_limit_exceeded")
            path = self._message_dir(message_id) / f"chunk-{index:08d}.bin"
            if path.is_file():
                if _sha256_bytes(path.read_bytes()) != envelope["chunk_hashes"][index]:
                    raise TensorTransportError("heterogeneous_tensor_chunk_conflict")
                return {
                    **self.status(message_id),
                    "chunk_index": index,
                    "idempotent_replay": True,
                }
            _atomic_write(path, chunk)
            return {
                **self.status(message_id),
                "chunk_index": index,
                "idempotent_replay": False,
            }

    def status(self, message_id: str) -> dict[str, Any]:
        envelope = self._load_envelope(message_id)
        directory = self._message_dir(message_id)
        present = sorted(
            int(path.stem.split("-", 1)[1])
            for path in directory.glob("chunk-*.bin")
            if path.is_file() and path.stem.split("-", 1)[1].isdigit()
        )
        return {
            "schema": STORE_SCHEMA,
            "message_id": envelope["message_id"],
            "placement_generation": int(envelope["placement_generation"]),
            "chunk_count": int(envelope["chunk_count"]),
            "received_chunk_count": len(present),
            "missing_chunk_indices": [
                index
                for index in range(int(envelope["chunk_count"]))
                if index not in set(present)
            ],
            "complete": len(present) == int(envelope["chunk_count"]),
            "expired": float(envelope["expires_at"]) <= float(self._clock()),
            "tensor_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }

    def assemble(
        self,
        message_id: str,
        *,
        expected_generation: int,
        consumer_id_hash: str = "",
        target_device: str = "cpu",
        target_dtype: str | None = None,
    ) -> dict[str, Any]:
        envelope = self._load_envelope(message_id)
        if int(envelope["placement_generation"]) != int(expected_generation):
            raise TensorTransportError("heterogeneous_tensor_stale_generation")
        if float(envelope["expires_at"]) <= float(self._clock()):
            raise TensorTransportError("heterogeneous_tensor_message_expired")
        status = self.status(message_id)
        if not status["complete"]:
            raise TensorTransportError("heterogeneous_tensor_message_incomplete")
        consumer = str(consumer_id_hash or "")
        if consumer and not consumer.startswith("sha256:"):
            raise TensorTransportError("heterogeneous_tensor_consumer_invalid")
        with self._lock:
            previous_consumer = self._consumers.get(message_id)
            if previous_consumer and consumer and previous_consumer != consumer:
                raise TensorTransportError("heterogeneous_tensor_already_consumed")
            payload = b"".join(
                (self._message_dir(message_id) / f"chunk-{index:08d}.bin").read_bytes()
                for index in range(int(envelope["chunk_count"]))
            )
            tensors = decode_tensor_payload(
                payload,
                envelope,
                target_device=target_device,
                target_dtype=target_dtype,
            )
            if consumer:
                self._consumers[message_id] = consumer
            return tensors

    def wait_for_complete(
        self,
        message_id: str,
        *,
        expected_generation: int,
        timeout: float,
        poll_interval: float = 0.05,
    ) -> dict[str, Any]:
        if float(timeout) <= 0 or float(poll_interval) <= 0:
            raise ValueError("heterogeneous_tensor_wait_policy_invalid")
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            envelope = self._load_envelope(message_id)
            if int(envelope["placement_generation"]) != int(expected_generation):
                raise TensorTransportError("heterogeneous_tensor_stale_generation")
            status = self.status(message_id)
            if status["complete"]:
                return status
            if status["expired"]:
                raise TensorTransportError("heterogeneous_tensor_message_expired")
            time.sleep(min(float(poll_interval), max(0.0, deadline - time.monotonic())))
        raise TimeoutError("heterogeneous_tensor_wait_timeout")

    def cleanup_expired(self) -> dict[str, Any]:
        removed = 0
        with self._lock:
            for directory in list(self.root.iterdir()):
                if not directory.is_dir() or not re.fullmatch(r"[0-9a-f]{64}", directory.name):
                    continue
                try:
                    envelope = self._load_envelope("sha256:" + directory.name)
                except TensorTransportError:
                    continue
                if float(envelope["expires_at"]) > float(self._clock()):
                    continue
                for path in directory.iterdir():
                    if path.is_file():
                        path.unlink()
                directory.rmdir()
                removed += 1
            self._rebuild_lookup_index()
        return {
            "schema": "crowdtensor_heterogeneous_tensor_cleanup_v1",
            "expired_messages_removed": removed,
            "tensor_values_public": False,
            "public_artifact_safe": True,
        }

    def cleanup_all(self) -> dict[str, Any]:
        removed = 0
        with self._lock:
            for directory in list(self.root.iterdir()):
                if not directory.is_dir() or not re.fullmatch(r"[0-9a-f]{64}", directory.name):
                    continue
                for path in directory.iterdir():
                    if path.is_file():
                        path.unlink()
                directory.rmdir()
                removed += 1
            self._delivery_attempts.clear()
            self._consumers.clear()
            self._lookup_index.clear()
        return {
            "schema": "crowdtensor_heterogeneous_tensor_cleanup_v1",
            "all_messages_removed": True,
            "message_count_removed": removed,
            "tensor_values_public": False,
            "public_artifact_safe": True,
        }

    def lookup_performance_report(self) -> dict[str, Any]:
        return {
            "schema": "crowdtensor_heterogeneous_tensor_lookup_performance_v1",
            "indexed_lookup_count": self._indexed_lookup_count,
            "legacy_lookup_count": self._legacy_lookup_count,
            "legacy_scanned_directory_count": self._lookup_scanned_directory_count,
            "indexed_key_count": len(self._lookup_index),
            "persistent_restart_rebuild_supported": True,
            "tensor_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }


def deliver_chunks_with_retry(
    envelope: dict[str, Any],
    chunks: Iterable[bytes],
    upload: Callable[[dict[str, Any], int, bytes], Any],
    *,
    sleep: Callable[[float], None] = time.sleep,
    base_delay_seconds: float = 0.05,
) -> dict[str, Any]:
    """Upload each chunk with an envelope-bounded finite retry policy."""

    canonical = validate_tensor_envelope(envelope)
    values = list(chunks)
    if len(values) != int(canonical["chunk_count"]):
        raise TensorTransportError("heterogeneous_tensor_chunk_count_mismatch")
    attempts = []
    for index, chunk in enumerate(values):
        last_error: BaseException | None = None
        for attempt in range(1, int(canonical["max_delivery_attempts"]) + 1):
            try:
                upload(canonical, index, chunk)
                attempts.append({"chunk_index": index, "attempt_count": attempt})
                last_error = None
                break
            except (OSError, TimeoutError, ConnectionError) as exc:
                last_error = exc
                if attempt < int(canonical["max_delivery_attempts"]):
                    sleep(float(base_delay_seconds) * (2 ** (attempt - 1)))
        if last_error is not None:
            raise TensorTransportError(
                "heterogeneous_tensor_delivery_retry_limit_exceeded"
            ) from last_error
    return {
        "schema": "crowdtensor_heterogeneous_tensor_delivery_v1",
        "message_id": canonical["message_id"],
        "chunk_count": len(values),
        "attempts": attempts,
        "delivery_complete": True,
        "finite_retry_policy": True,
        "tensor_values_public": False,
        "public_artifact_safe": True,
    }
