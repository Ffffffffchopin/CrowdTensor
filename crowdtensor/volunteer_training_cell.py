"""Ordinary-contributor Training Cell for Volunteer Training Protocol Alpha."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import secrets
import shutil
import stat
import threading
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import httpx

from .hf_lora_training import CPULoRATrainingRuntime, CUDALoRATrainingRuntime
from .training_contract import (
    TRAINING_SPEC_SCHEMA,
    public_training_spec,
    sha256_file,
    sha256_json,
)
from .volunteer_training_coordinator import VolunteerTrainingCoordinator
from .volunteer_training_protocol import (
    CELL_STATE_SCHEMA,
    INVITE_SCHEMA,
    PROTOCOL_VERSION,
    SUBMISSION_SCHEMA,
    VolunteerProtocolError,
    encode_submission_envelope,
    hash_cell_id,
    public_error,
    validate_campaign_manifest,
    validate_work_unit,
    with_public_safety,
)


CELL_REPORT_SCHEMA = "crowdtensor_volunteer_training_cell_report_v1"


class VolunteerUploadInterrupted(RuntimeError):
    """A deliberate or transport-caused pause after durable chunks exist."""

    def __init__(self, upload_id: str) -> None:
        super().__init__("volunteer_resumable_upload_interrupted")
        self.upload_id = str(upload_id)


def _atomic_json(path: Path, value: dict[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("volunteer JSON object required")
    return value


def detect_hardware() -> dict[str, Any]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_count = int(torch.cuda.device_count()) if cuda_available else 0
        device_names = [torch.cuda.get_device_name(index) for index in range(cuda_count)]
        torch_version = str(torch.__version__)
    except ImportError:
        cuda_available = False
        cuda_count = 0
        device_names = []
        torch_version = ""
    memory_bytes = 0
    try:
        memory_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError):
        pass
    return with_public_safety(
        {
            "schema": "crowdtensor_volunteer_training_hardware_v1",
            "cpu_count": int(os.cpu_count() or 1),
            "memory_bytes": memory_bytes,
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_count,
            "cuda_device_name_hashes": [
                sha256_json({"device_name": value}) for value in device_names
            ],
            "torch_version": torch_version,
            "supported_devices": ["cpu"] + (["cuda:0"] if cuda_count else []),
        }
    )


class VolunteerTransport(Protocol):
    def campaign(self) -> dict[str, Any]: ...

    def status(self) -> dict[str, Any]: ...

    def claim(self, *, cell_id: str, capability: dict[str, Any]) -> dict[str, Any]: ...

    def heartbeat(self, *, cell_id: str, work: dict[str, Any]) -> dict[str, Any]: ...

    def download_artifact(
        self, ref: dict[str, Any], destination: Path, *, max_bytes: int
    ) -> int: ...

    def submit(
        self,
        *,
        cell_id: str,
        work: dict[str, Any],
        delta_manifest: dict[str, Any],
    ) -> dict[str, Any]: ...


class HTTPVolunteerTransport:
    def __init__(
        self,
        coordinator_url: str,
        invite_token: str,
        *,
        timeout_seconds: float = 120.0,
        extra_headers: dict[str, str] | None = None,
        resumable_uploads: bool = True,
        interrupt_after_chunks: int = 0,
    ) -> None:
        self.coordinator_url = str(coordinator_url).rstrip("/")
        self.invite_token = str(invite_token)
        self.timeout_seconds = float(timeout_seconds)
        self.extra_headers = dict(extra_headers or {})
        self.resumable_uploads = bool(resumable_uploads)
        self.interrupt_after_chunks = int(interrupt_after_chunks)
        self.last_upload_id = ""
        self._cell_credential_token = ""
        self._cell_credential_id = ""
        self._credential_cell_id = ""
        self._credential_expires_at = 0.0
        self._legacy_invite_fallback = False
        if not self.coordinator_url:
            raise VolunteerProtocolError("volunteer_coordinator_url_missing", status_code=400)
        if not self.invite_token:
            raise VolunteerProtocolError("volunteer_invite_token_missing", status_code=400)

    @classmethod
    def from_invite(
        cls,
        invite_path: str | Path,
        *,
        timeout_seconds: float = 120.0,
        extra_headers: dict[str, str] | None = None,
        resumable_uploads: bool = True,
        interrupt_after_chunks: int = 0,
    ) -> "HTTPVolunteerTransport":
        path = Path(invite_path).expanduser().resolve()
        value = _json_file(path)
        if value.get("schema") != INVITE_SCHEMA:
            raise VolunteerProtocolError("volunteer_invite_schema_mismatch", status_code=400)
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise VolunteerProtocolError("volunteer_invite_file_permissions_too_open", status_code=400)
        return cls(
            str(value.get("coordinator_url") or ""),
            str(value.get("invite_token") or ""),
            timeout_seconds=timeout_seconds,
            extra_headers=extra_headers,
            resumable_uploads=resumable_uploads,
            interrupt_after_chunks=interrupt_after_chunks,
        )

    def _ensure_cell_credential(self, cell_id: str) -> None:
        normalized = str(cell_id).strip()
        if not normalized:
            raise VolunteerProtocolError("volunteer_cell_id_missing", status_code=400)
        if self._legacy_invite_fallback:
            self._credential_cell_id = normalized
            return
        if (
            self._cell_credential_token
            and self._credential_cell_id == normalized
            and self._credential_expires_at > time.time() + 5.0
        ):
            return
        response = httpx.post(
            self.coordinator_url + "/v1/volunteer/credentials/issue",
            headers={
                **self.extra_headers,
                "Authorization": "Bearer " + self.invite_token,
            },
            json={"cell_id": normalized, "ttl_seconds": 900},
            timeout=self.timeout_seconds,
        )
        try:
            value = self._response(response)
        except VolunteerProtocolError as exc:
            if exc.status_code == 404:
                self._legacy_invite_fallback = True
                self._credential_cell_id = normalized
                return
            raise
        credential = str(value.get("credential_token") or "")
        if not credential:
            raise VolunteerProtocolError(
                "volunteer_cell_credential_response_invalid", status_code=502
            )
        self._cell_credential_token = credential
        self._cell_credential_id = str(value.get("credential_id") or "")
        self._credential_cell_id = normalized
        self._credential_expires_at = float(value.get("expires_at") or 0.0)

    def _headers(self, cell_id: str = "") -> dict[str, str]:
        normalized = str(cell_id or self._credential_cell_id).strip()
        if normalized:
            self._ensure_cell_credential(normalized)
        credential = self._cell_credential_token
        headers = {
            **self.extra_headers,
            "Authorization": "Bearer " + (credential or self.invite_token),
        }
        if credential:
            headers.update(
                {
                    "X-CrowdTensor-Cell-Id": normalized,
                    "X-CrowdTensor-Nonce": secrets.token_urlsafe(24),
                }
            )
        return headers

    @staticmethod
    def _response(response: httpx.Response) -> dict[str, Any]:
        try:
            value = response.json()
        except json.JSONDecodeError as exc:
            raise VolunteerProtocolError(
                "volunteer_coordinator_response_invalid", status_code=502
            ) from exc
        if response.status_code >= 400:
            code = str(value.get("error") or "volunteer_coordinator_request_failed")
            raise VolunteerProtocolError(code, status_code=response.status_code)
        if not isinstance(value, dict):
            raise VolunteerProtocolError(
                "volunteer_coordinator_response_invalid", status_code=502
            )
        return value

    def campaign(self) -> dict[str, Any]:
        response = httpx.get(
            self.coordinator_url + "/v1/volunteer/campaign",
            headers=self.extra_headers,
            timeout=self.timeout_seconds,
        )
        return self._response(response)

    def status(self) -> dict[str, Any]:
        response = httpx.get(
            self.coordinator_url + "/v1/volunteer/status",
            headers=self.extra_headers,
            timeout=self.timeout_seconds,
        )
        return self._response(response)

    def claim(self, *, cell_id: str, capability: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(
            self.coordinator_url + "/v1/volunteer/work/claim",
            headers=self._headers(cell_id),
            json={"cell_id": cell_id, "capability": capability},
            timeout=self.timeout_seconds,
        )
        return self._response(response)

    def heartbeat(self, *, cell_id: str, work: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(
            self.coordinator_url + "/v1/volunteer/work/heartbeat",
            headers=self._headers(cell_id),
            json={
                "cell_id": cell_id,
                "work_id": work["work_id"],
                "lease_generation": int(work["lease_generation"]),
                "lease_token": work["lease_token"],
            },
            timeout=self.timeout_seconds,
        )
        return self._response(response)

    def download_artifact(
        self, ref: dict[str, Any], destination: Path, *, max_bytes: int
    ) -> int:
        expected_size = int(ref.get("byte_count") or 0)
        if expected_size < 1 or expected_size > int(max_bytes):
            raise VolunteerProtocolError("volunteer_artifact_size_limit_exceeded", status_code=413)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{secrets.token_hex(4)}.tmp")
        digest = hashlib.sha256()
        written = 0
        try:
            with httpx.stream(
                "GET",
                self.coordinator_url
                + "/v1/volunteer/artifacts/"
                + str(ref.get("artifact_id") or ""),
                headers=self._headers(),
                timeout=self.timeout_seconds,
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    self._response(response)
                with temporary.open("xb") as handle:
                    for chunk in response.iter_bytes():
                        written += len(chunk)
                        if written > int(max_bytes):
                            raise VolunteerProtocolError(
                                "volunteer_artifact_size_limit_exceeded", status_code=413
                            )
                        digest.update(chunk)
                        handle.write(chunk)
            actual_hash = "sha256:" + digest.hexdigest()
            if written != expected_size or actual_hash != ref.get("sha256"):
                raise VolunteerProtocolError("volunteer_artifact_integrity_failed")
            temporary.chmod(0o600)
            os.replace(temporary, destination)
            return written
        finally:
            temporary.unlink(missing_ok=True)

    def submit(
        self,
        *,
        cell_id: str,
        work: dict[str, Any],
        delta_manifest: dict[str, Any],
    ) -> dict[str, Any]:
        if self.resumable_uploads:
            return self._submit_resumable(
                cell_id=cell_id, work=work, delta_manifest=delta_manifest
            )
        return self._submit_legacy(
            cell_id=cell_id, work=work, delta_manifest=delta_manifest
        )

    def _submission_metadata(
        self,
        *,
        cell_id: str,
        work: dict[str, Any],
        delta_manifest: dict[str, Any],
    ) -> dict[str, Any]:
        public_manifest = {
            key: value for key, value in delta_manifest.items() if not key.endswith("_path")
        }
        return {
            "schema": SUBMISSION_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "cell_id": cell_id,
            "work_id": work["work_id"],
            "lease_generation": int(work.get("lease_generation") or 0),
            "lease_token": str(work.get("lease_token") or ""),
            "delta_manifest": public_manifest,
        }

    def _submit_resumable(
        self,
        *,
        cell_id: str,
        work: dict[str, Any],
        delta_manifest: dict[str, Any],
    ) -> dict[str, Any]:
        path = Path(str(delta_manifest.get("delta_path") or ""))
        metadata = self._submission_metadata(
            cell_id=cell_id, work=work, delta_manifest=delta_manifest
        )
        started = self._response(
            httpx.post(
                self.coordinator_url + "/v1/volunteer/uploads/start",
                headers=self._headers(cell_id),
                json={
                    "cell_id": cell_id,
                    "idempotency_key": str(
                        work.get("idempotency_key")
                        or delta_manifest.get("result_id")
                        or ""
                    ),
                    "expected_blob_hash": delta_manifest["delta_file_hash"],
                    "total_bytes": int(path.stat().st_size),
                    "submission": metadata,
                },
                timeout=self.timeout_seconds,
            )
        )
        upload_id = str(started["upload_id"])
        self.last_upload_id = upload_id
        received_before = set(int(item) for item in started["received_chunk_indexes"])
        chunk_bytes = int(started["chunk_bytes"])
        uploaded_now = 0
        with path.open("rb") as handle:
            for index in range(int(started["chunk_count"])):
                chunk = handle.read(chunk_bytes)
                if index in received_before:
                    continue
                headers = {
                    **self._headers(cell_id),
                    "Content-Type": "application/octet-stream",
                    "X-CrowdTensor-Cell-Id": cell_id,
                    "X-CrowdTensor-Chunk-SHA256": "sha256:"
                    + hashlib.sha256(chunk).hexdigest(),
                }
                self._response(
                    httpx.put(
                        self.coordinator_url
                        + f"/v1/volunteer/uploads/{upload_id}/chunks/{index}",
                        headers=headers,
                        content=chunk,
                        timeout=max(self.timeout_seconds, 300.0),
                    )
                )
                uploaded_now += 1
                if self.interrupt_after_chunks and uploaded_now >= self.interrupt_after_chunks:
                    raise VolunteerUploadInterrupted(upload_id)
        completed = self._response(
            httpx.post(
                self.coordinator_url
                + f"/v1/volunteer/uploads/{upload_id}/complete",
                headers={
                    **self._headers(cell_id),
                    "X-CrowdTensor-Cell-Id": cell_id,
                },
                timeout=max(self.timeout_seconds, 300.0),
            )
        )
        completed["upload_resume_summary"] = {
            "upload_id_hash": sha256_json({"upload_id": upload_id}),
            "received_chunks_before_resume": len(received_before),
            "uploaded_chunks_this_attempt": uploaded_now,
            "resumed_existing_upload": bool(received_before),
            "resumable_protocol_used": True,
        }
        return completed

    def _submit_legacy(
        self,
        *,
        cell_id: str,
        work: dict[str, Any],
        delta_manifest: dict[str, Any],
    ) -> dict[str, Any]:
        path = Path(str(delta_manifest.get("delta_path") or ""))
        metadata = self._submission_metadata(
            cell_id=cell_id, work=work, delta_manifest=delta_manifest
        )
        body = encode_submission_envelope(metadata, path.read_bytes())
        response = httpx.post(
            self.coordinator_url + "/v1/volunteer/work/submit",
            headers={
                **self._headers(cell_id),
                "Content-Type": "application/octet-stream",
            },
            content=body,
            timeout=max(self.timeout_seconds, 300.0),
        )
        return self._response(response)


class LocalVolunteerTransport:
    """The same protocol boundary without HTTP, useful for deterministic tests."""

    def __init__(self, coordinator: VolunteerTrainingCoordinator, invite_token: str) -> None:
        self.coordinator = coordinator
        self.invite_token = str(invite_token)

    def campaign(self) -> dict[str, Any]:
        return self.coordinator.campaign_manifest()

    def status(self) -> dict[str, Any]:
        return self.coordinator.status(invite_token=self.invite_token)

    def claim(self, *, cell_id: str, capability: dict[str, Any]) -> dict[str, Any]:
        return self.coordinator.claim(
            cell_id=cell_id, invite_token=self.invite_token, capability=capability
        )

    def heartbeat(self, *, cell_id: str, work: dict[str, Any]) -> dict[str, Any]:
        return self.coordinator.heartbeat(
            cell_id=cell_id,
            invite_token=self.invite_token,
            work_id=work["work_id"],
            lease_generation=int(work["lease_generation"]),
            lease_token=work["lease_token"],
        )

    def download_artifact(
        self, ref: dict[str, Any], destination: Path, *, max_bytes: int
    ) -> int:
        source = self.coordinator.artifact_path(
            str(ref["artifact_id"]), invite_token=self.invite_token
        )
        size = source.stat().st_size
        if size > int(max_bytes):
            raise VolunteerProtocolError("volunteer_artifact_size_limit_exceeded", status_code=413)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256_file(destination) != ref["sha256"]:
            destination.unlink(missing_ok=True)
            raise VolunteerProtocolError("volunteer_artifact_integrity_failed")
        return size

    def submit(
        self,
        *,
        cell_id: str,
        work: dict[str, Any],
        delta_manifest: dict[str, Any],
    ) -> dict[str, Any]:
        return self.coordinator.submit(
            cell_id=cell_id,
            invite_token=self.invite_token,
            work_id=work["work_id"],
            lease_generation=int(work["lease_generation"]),
            lease_token=work["lease_token"],
            delta_manifest=delta_manifest,
        )


class _Heartbeat:
    def __init__(
        self,
        transport: VolunteerTransport,
        *,
        cell_id: str,
        work: dict[str, Any],
        interval_seconds: float,
    ) -> None:
        self.transport = transport
        self.cell_id = cell_id
        self.work = work
        self.interval = max(0.2, float(interval_seconds))
        self.stop_event = threading.Event()
        self.errors: list[str] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                self.transport.heartbeat(cell_id=self.cell_id, work=self.work)
            except Exception as exc:  # the foreground reports only a public error class
                self.errors.append(type(exc).__name__)
                self.stop_event.set()

    def __enter__(self) -> "_Heartbeat":
        self.thread.start()
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(1.0, self.interval + 1.0))


class VolunteerTrainingCell:
    def __init__(
        self,
        transport: VolunteerTransport,
        workspace: str | Path,
        *,
        cell_id: str = "",
        device: str = "auto",
        max_local_steps: int = 64,
        max_download_bytes: int = 8 * 1024 * 1024 * 1024,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.transport = transport
        self.workspace = Path(workspace).expanduser().resolve()
        self.private = self.workspace / ".private"
        self.cache = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir
            else self.private / "cache"
        )
        self.shared_cache = bool(cache_dir)
        self.state_path = self.private / "cell_state.json"
        self.status_path = self.workspace / "status.json"
        self.pause_path = self.private / "paused"
        self.device_policy = str(device)
        self.max_local_steps = int(max_local_steps)
        self.max_download_bytes = int(max_download_bytes)
        if self.max_local_steps < 1 or self.max_download_bytes < 1:
            raise ValueError("volunteer Cell resource limits must be positive")
        self.private.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.cache.chmod(0o700)
        if self.state_path.is_file():
            private_state = _json_file(self.state_path)
            self.cell_id = str(private_state["cell_id"])
        else:
            self.cell_id = str(cell_id or "cell-" + secrets.token_hex(12))
            _atomic_json(
                self.state_path,
                {
                    "schema": CELL_STATE_SCHEMA,
                    "cell_id": self.cell_id,
                    "cell_id_hash": hash_cell_id(self.cell_id),
                    "completed_work_units": 0,
                    "created_at": time.time(),
                },
                mode=0o600,
            )

    def _private_state(self) -> dict[str, Any]:
        return _json_file(self.state_path)

    def _save_private_state(self, value: dict[str, Any]) -> None:
        _atomic_json(self.state_path, value, mode=0o600)

    def _write_public_status(self, value: dict[str, Any]) -> dict[str, Any]:
        public = with_public_safety(
            {
                "schema": CELL_REPORT_SCHEMA,
                "cell_id_hash": hash_cell_id(self.cell_id),
                **value,
            }
        )
        _atomic_json(self.status_path, public, mode=0o644)
        return public

    def hardware(self) -> dict[str, Any]:
        return detect_hardware()

    def selected_device(self) -> str:
        hardware = self.hardware()
        if self.device_policy == "auto":
            return "cuda:0" if hardware["cuda_available"] else "cpu"
        if self.device_policy == "cpu":
            return "cpu"
        if self.device_policy.startswith("cuda"):
            if not hardware["cuda_available"]:
                raise VolunteerProtocolError("volunteer_cuda_device_unavailable", status_code=400)
            return self.device_policy
        raise VolunteerProtocolError("volunteer_device_policy_invalid", status_code=400)

    def pause(self) -> dict[str, Any]:
        self.pause_path.parent.mkdir(parents=True, exist_ok=True)
        self.pause_path.write_text("paused\n", encoding="ascii")
        self.pause_path.chmod(0o600)
        return self._write_public_status({"ok": True, "state": "paused"})

    def resume(self) -> dict[str, Any]:
        self.pause_path.unlink(missing_ok=True)
        return self._write_public_status({"ok": True, "state": "ready"})

    def local_status(self) -> dict[str, Any]:
        if self.status_path.is_file():
            return _json_file(self.status_path)
        return self._write_public_status(
            {"ok": True, "state": "paused" if self.pause_path.exists() else "ready"}
        )

    @staticmethod
    def _safe_extract(archive_path: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                path = PurePosixPath(info.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise VolunteerProtocolError("volunteer_model_bundle_path_unsafe", status_code=400)
            archive.extractall(destination)

    def _cached_artifact(self, ref: dict[str, Any]) -> tuple[Path, int]:
        digest = str(ref.get("sha256") or "")
        if not digest.startswith("sha256:"):
            raise VolunteerProtocolError("volunteer_artifact_hash_invalid", status_code=400)
        destination = self.cache / "artifacts" / digest.split(":", 1)[1]
        if destination.is_file():
            if sha256_file(destination) == digest and destination.stat().st_size == int(
                ref.get("byte_count") or 0
            ):
                return destination, 0
            destination.unlink()
        downloaded = self.transport.download_artifact(
            ref, destination, max_bytes=self.max_download_bytes
        )
        return destination, downloaded

    def _materialize(self, work: dict[str, Any]) -> tuple[dict[str, Path], int]:
        refs = work.get("artifact_refs")
        if not isinstance(refs, dict):
            raise VolunteerProtocolError("volunteer_work_artifact_refs_missing", status_code=400)
        cached: dict[str, Path] = {}
        downloaded = 0
        for name in ("base_model", "base_adapter", "adapter_config", "dataset_shard"):
            if not isinstance(refs.get(name), dict):
                raise VolunteerProtocolError("volunteer_work_artifact_ref_missing", status_code=400)
            path, byte_count = self._cached_artifact(refs[name])
            cached[name] = path
            downloaded += byte_count
        work_root = self.private / "work" / str(work["work_unit_hash"]).split(":", 1)[-1]
        model_hash = str(refs["base_model"]["sha256"]).split(":", 1)[1]
        model_dir = self.cache / "extracted-models" / model_hash
        model_lock = self.cache / "extracted-models" / f".{model_hash}.lock"
        model_lock.parent.mkdir(parents=True, exist_ok=True)
        with model_lock.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                if not (model_dir / "config.json").is_file():
                    temporary = model_dir.with_name(
                        f".{model_dir.name}.{secrets.token_hex(4)}.tmp"
                    )
                    shutil.rmtree(temporary, ignore_errors=True)
                    self._safe_extract(cached["base_model"], temporary)
                    if model_dir.exists():
                        shutil.rmtree(model_dir)
                    os.replace(temporary, model_dir)
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        adapter_dir = work_root / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_path = adapter_dir / "adapter_model.safetensors"
        config_path = adapter_dir / "adapter_config.json"
        shutil.copyfile(cached["base_adapter"], adapter_path)
        shutil.copyfile(cached["adapter_config"], config_path)
        dataset_path = work_root / "dataset_shard.jsonl"
        shutil.copyfile(cached["dataset_shard"], dataset_path)
        return {
            "work_root": work_root,
            "base_model": model_dir,
            "adapter_dir": adapter_dir,
            "adapter_tensor": adapter_path,
            "adapter_config": config_path,
            "dataset": dataset_path,
        }, downloaded

    def _training_spec(
        self, campaign: dict[str, Any], work: dict[str, Any], paths: dict[str, Path]
    ) -> dict[str, Any]:
        device = self.selected_device()
        sample_count = int(work["dataset_sample_count"])
        spec: dict[str, Any] = {
            "schema": TRAINING_SPEC_SCHEMA,
            "workload_type": "hf_lora_train",
            "job_id": campaign["campaign_id"],
            "job_hash": campaign["manifest_hash"],
            "round_id": work["round_id"],
            "task_id": work["work_id"],
            "miner_id": self.cell_id,
            "model_manifest_hash": campaign["model_manifest_hash"],
            "base_model_hash": campaign["base_model_hash"],
            "base_model_version": int(campaign["model_revision"]),
            "base_model_path": str(paths["base_model"]),
            "base_adapter_hash": work["base_adapter_hash"],
            "adapter_version": int(work["adapter_version"]),
            "adapter_path": str(paths["adapter_dir"]),
            "adapter_tensor_path": str(paths["adapter_tensor"]),
            "adapter_config_path": str(paths["adapter_config"]),
            "dataset_path": str(paths["dataset"]),
            "dataset_manifest_hash": campaign["dataset_snapshot_hash"],
            "dataset_shard_index": int(work["dataset_shard_index"]),
            "dataset_shard_hash": work["dataset_shard_hash"],
            "sample_indexes": list(range(sample_count)),
            "sample_count": sample_count,
            "token_count": sample_count * int(work["sequence_length"]),
            "data_cursor": 0,
            "seed": int(str(work["work_unit_hash"]).split(":", 1)[-1][:8], 16),
            "step_start": int(work["step_start"]),
            "step_end": int(work["step_end"]),
            "local_steps": int(work["local_steps"]),
            "learning_rate": float(work["learning_rate"]),
            "batch_size": int(work["batch_size"]),
            "sequence_length": int(work["sequence_length"]),
            "gradient_accumulation": int(work["gradient_accumulation"]),
            "optimizer_contract": work["optimizer_contract"],
            "outer_step": int(work["round_index"]),
            "device": device,
            "trusted_worker_required": True,
            "raw_dataset_public": False,
        }
        spec["claim_hash"] = sha256_json(public_training_spec(spec))
        return spec

    def join_once(self) -> dict[str, Any]:
        if self.pause_path.exists():
            return self._write_public_status({"ok": True, "state": "paused", "work_completed": False})
        pending = self._private_state().get("pending_submission")
        if isinstance(pending, dict):
            return self._resume_pending_submission(pending, training_reexecuted=False)
        campaign = self.transport.campaign()
        validate_campaign_manifest(campaign)
        capability = {
            **self.hardware(),
            "selected_device": self.selected_device(),
            "max_local_steps": self.max_local_steps,
            "max_download_bytes": self.max_download_bytes,
            "real_peft_lora": True,
        }
        claim = self.transport.claim(cell_id=self.cell_id, capability=capability)
        work = claim.get("work_unit")
        if not isinstance(work, dict):
            state = str(claim.get("state") or "waiting_for_work")
            return self._write_public_status(
                {
                    "ok": True,
                    "state": state,
                    "campaign_id": campaign["campaign_id"],
                    "work_completed": False,
                }
            )
        validate_work_unit(work, campaign=campaign, now=time.time())
        if int(work["local_steps"]) > self.max_local_steps:
            raise VolunteerProtocolError("volunteer_work_exceeds_local_step_limit", status_code=400)

        lease_window = max(0.6, float(work["lease_expires_at"]) - time.time())
        heartbeat_interval = max(0.2, lease_window / 3.0)
        with _Heartbeat(
            self.transport,
            cell_id=self.cell_id,
            work=work,
            interval_seconds=heartbeat_interval,
        ) as heartbeat:
            paths, downloaded_bytes = self._materialize(work)
            spec = self._training_spec(campaign, work, paths)
            runtime = (
                CPULoRATrainingRuntime()
                if spec["device"] == "cpu"
                else CUDALoRATrainingRuntime(spec["device"])
            )
            result = runtime.run(spec, output_dir=paths["work_root"] / "training")
        if heartbeat.errors:
            raise VolunteerProtocolError("volunteer_lease_heartbeat_failed")
        private_state = self._private_state()
        private_state["pending_submission"] = {
            "work": work,
            "delta_manifest": result["adapter_delta"],
            "training_summary": {
                "campaign_id": campaign["campaign_id"],
                "campaign_manifest_hash": campaign["manifest_hash"],
                "round_id": work["round_id"],
                "round_index": int(work["round_index"]),
                "adapter_version": int(work["adapter_version"]),
                "work_unit_hash": work["work_unit_hash"],
                "real_pytorch_autograd": bool(result["real_backward"]),
                "real_transformers_peft_lora": bool(result["runtime"]["real_peft_lora"]),
                "base_weights_frozen": bool(result["base_weights_frozen"]),
                "optimizer_steps": int(result["optimizer_steps"]),
                "samples_seen": int(result["samples_seen"]),
                "tokens_seen": int(result["tokens_seen"]),
                "loss_start": float(result["loss_start"]),
                "loss_end": float(result["loss_end"]),
                "delta_file_hash": result["adapter_delta"]["delta_file_hash"],
                "delta_byte_count": Path(result["adapter_delta"]["delta_path"]).stat().st_size,
                "artifact_download_bytes": downloaded_bytes,
            },
        }
        self._save_private_state(private_state)
        return self._resume_pending_submission(
            private_state["pending_submission"], training_reexecuted=True
        )

    def _resume_pending_submission(
        self, pending: dict[str, Any], *, training_reexecuted: bool
    ) -> dict[str, Any]:
        work = pending.get("work")
        manifest = pending.get("delta_manifest")
        summary = pending.get("training_summary")
        if not all(isinstance(item, dict) for item in (work, manifest, summary)):
            raise VolunteerProtocolError("volunteer_pending_submission_invalid", status_code=409)
        try:
            response = self.transport.submit(
                cell_id=self.cell_id,
                work=work,
                delta_manifest=manifest,
            )
        except VolunteerProtocolError as exc:
            if exc.code in {
                "volunteer_stale_adapter_version_rejected",
                "volunteer_round_closed_stale_update",
                "volunteer_lease_not_active",
                "volunteer_lease_generation_mismatch",
                "volunteer_lease_expired",
            }:
                private_state = self._private_state()
                private_state.pop("pending_submission", None)
                self._save_private_state(private_state)
            raise
        private_state = self._private_state()
        private_state.pop("pending_submission", None)
        private_state["completed_work_units"] = int(private_state.get("completed_work_units") or 0) + 1
        private_state["last_round_id"] = work["round_id"]
        private_state["last_work_id"] = work["work_id"]
        private_state["last_result_id"] = manifest["result_id"]
        self._save_private_state(private_state)
        return self._write_public_status(
            {
                "ok": bool(response.get("accepted")),
                "state": "submitted",
                **summary,
                "work_completed": True,
                "lease_heartbeat_enabled": True,
                "pending_submission_recovery_used": bool(
                    (response.get("upload_resume_summary") or {}).get(
                        "resumed_existing_upload"
                    )
                ),
                "training_reexecuted_for_submission_resume": training_reexecuted,
                "shared_content_cache": self.shared_cache,
                "submission": response,
            }
        )

    def run(
        self,
        *,
        max_work_units: int = 0,
        poll_interval_seconds: float = 2.0,
    ) -> dict[str, Any]:
        completed = 0
        last: dict[str, Any] = {}
        while max_work_units <= 0 or completed < max_work_units:
            if self.pause_path.exists():
                last = self.local_status()
                break
            last = self.join_once()
            if last.get("work_completed"):
                completed += 1
                continue
            if last.get("state") == "campaign_complete":
                break
            time.sleep(max(0.05, float(poll_interval_seconds)))
        return with_public_safety(
            {
                "schema": CELL_REPORT_SCHEMA,
                "ok": bool(last.get("ok", True)),
                "cell_id_hash": hash_cell_id(self.cell_id),
                "completed_in_run": completed,
                "last_state": last.get("state", "ready"),
                "last_report": last,
            }
        )

    def cleanup(self) -> dict[str, Any]:
        work = self.private / "work"
        if work.exists():
            shutil.rmtree(work)
        return self._write_public_status(
            {
                "ok": True,
                "state": "clean",
                "temporary_work_removed": not work.exists(),
                "artifact_cache_preserved": self.cache.exists(),
                "live_resources_left_running": False,
            }
        )


def safe_cell_error(exc: VolunteerProtocolError) -> dict[str, Any]:
    return public_error(exc.code)
