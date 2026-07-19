"""Content-addressed artifacts and resumable uploads for volunteer training."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import secrets
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Protocol

from .training_contract import sha256_file, sha256_json
from .volunteer_training_protocol import VolunteerProtocolError, with_public_safety


BLOB_STORE_SCHEMA = "crowdtensor_volunteer_blob_store_v1"
BLOB_REF_SCHEMA = "crowdtensor_volunteer_blob_ref_v1"
UPLOAD_SESSION_SCHEMA = "crowdtensor_volunteer_resumable_upload_v1"
DEFAULT_CHUNK_BYTES = 1024 * 1024


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _digest(value: str) -> str:
    text = str(value)
    if (
        not text.startswith("sha256:")
        or len(text) != 71
        or any(character not in "0123456789abcdef" for character in text[7:])
    ):
        raise VolunteerProtocolError("volunteer_blob_hash_invalid", status_code=400)
    return text[7:]


def _atomic_json(path: Path, value: dict[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VolunteerProtocolError("volunteer_upload_session_invalid", status_code=409)
    return value


class VolunteerBlobStore(Protocol):
    backend: str

    def put_file(self, path: str | Path, *, expected_hash: str = "") -> dict[str, Any]: ...

    def local_path(self, blob_hash: str) -> Path: ...

    def public_report(self) -> dict[str, Any]: ...


class LocalVolunteerBlobStore:
    backend = "local_content_addressed"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)

    def local_path(self, blob_hash: str) -> Path:
        digest = _digest(blob_hash)
        return self.root / digest[:2] / f"{digest}.blob"

    def put_file(self, path: str | Path, *, expected_hash: str = "") -> dict[str, Any]:
        source = Path(path)
        actual_hash = sha256_file(source)
        if expected_hash and actual_hash != expected_hash:
            raise VolunteerProtocolError("volunteer_blob_hash_mismatch", status_code=409)
        destination = self.local_path(actual_hash)
        created = not destination.is_file()
        if not created:
            if sha256_file(destination) != actual_hash:
                raise VolunteerProtocolError("volunteer_blob_collision", status_code=409)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.parent.chmod(0o700)
            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
            )
            try:
                with source.open("rb") as input_handle, temporary.open("xb") as output_handle:
                    shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
                if sha256_file(temporary) != actual_hash:
                    raise VolunteerProtocolError("volunteer_blob_copy_hash_mismatch", status_code=409)
                temporary.chmod(0o600)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        return with_public_safety(
            {
                "schema": BLOB_REF_SCHEMA,
                "backend": self.backend,
                "blob_hash": actual_hash,
                "byte_count": int(source.stat().st_size),
                "created": created,
                "content_addressed": True,
            }
        )

    def put_bytes(self, value: bytes) -> dict[str, Any]:
        temporary = self.root / f".put-{secrets.token_hex(8)}.tmp"
        try:
            temporary.write_bytes(value)
            temporary.chmod(0o600)
            return self.put_file(temporary, expected_hash=_sha256_bytes(value))
        finally:
            temporary.unlink(missing_ok=True)

    def get_bytes(self, blob_hash: str) -> bytes:
        path = self.local_path(blob_hash)
        try:
            value = path.read_bytes()
        except FileNotFoundError as exc:
            raise VolunteerProtocolError("volunteer_blob_not_found", status_code=404) from exc
        if _sha256_bytes(value) != blob_hash:
            raise VolunteerProtocolError("volunteer_blob_integrity_failed", status_code=409)
        return value

    def list_hashes(self) -> list[str]:
        return sorted(
            "sha256:" + path.stem
            for path in self.root.glob("*/*.blob")
            if len(path.stem) == 64
        )

    def delete(self, blob_hash: str) -> bool:
        path = self.local_path(blob_hash)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        try:
            path.parent.rmdir()
        except OSError:
            pass
        return True

    def public_report(self) -> dict[str, Any]:
        return with_public_safety(
            {
                "schema": BLOB_STORE_SCHEMA,
                "backend": self.backend,
                "content_addressed": True,
                "atomic_blob_writes": True,
                "post_write_hash_verification": True,
                "s3_compatible": False,
                "presigned_download_supported": False,
                "blob_count": len(self.list_hashes()),
            }
        )


class S3VolunteerBlobStore:
    """S3/MinIO content-addressed store with private presigned downloads."""

    backend = "s3_content_addressed"

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "crowdtensor/volunteer",
        endpoint_url: str = "",
        region_name: str = "",
        access_key_env: str = "AWS_ACCESS_KEY_ID",
        secret_key_env: str = "AWS_SECRET_ACCESS_KEY",
        session_token_env: str = "AWS_SESSION_TOKEN",
        client: Any | None = None,
    ) -> None:
        self.bucket = str(bucket).strip()
        self.prefix = str(prefix).strip().strip("/")
        self.endpoint_url = str(endpoint_url).strip()
        self.region_name = str(region_name).strip()
        self.access_key_env = str(access_key_env).strip()
        self.secret_key_env = str(secret_key_env).strip()
        self.session_token_env = str(session_token_env).strip()
        self._client = client
        if not self.bucket or not self.access_key_env or not self.secret_key_env:
            raise ValueError("volunteer_s3_configuration_invalid")

    def _key(self, blob_hash: str) -> str:
        digest = _digest(blob_hash)
        suffix = f"{digest[:2]}/{digest}.blob"
        return f"{self.prefix}/{suffix}" if self.prefix else suffix

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError("volunteer_s3_requires_storage_extra") from exc
        access = os.environ.get(self.access_key_env, "")
        secret = os.environ.get(self.secret_key_env, "")
        session = os.environ.get(self.session_token_env, "") if self.session_token_env else ""
        if not access or not secret:
            raise RuntimeError("volunteer_s3_private_credentials_missing")
        kwargs: dict[str, Any] = {
            "aws_access_key_id": access,
            "aws_secret_access_key": secret,
        }
        if session:
            kwargs["aws_session_token"] = session
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        if self.region_name:
            kwargs["region_name"] = self.region_name
        kwargs["config"] = Config(
            connect_timeout=3,
            read_timeout=15,
            retries={"max_attempts": 2, "mode": "standard"},
        )
        self._client = boto3.client("s3", **kwargs)
        return self._client

    @staticmethod
    def _missing(exc: BaseException) -> bool:
        code = str((getattr(exc, "response", {}).get("Error") or {}).get("Code") or "")
        return code in {"404", "NoSuchKey", "NotFound"}

    def put_file(self, path: str | Path, *, expected_hash: str = "") -> dict[str, Any]:
        source = Path(path)
        blob_hash = sha256_file(source)
        if expected_hash and blob_hash != expected_hash:
            raise VolunteerProtocolError("volunteer_blob_hash_mismatch", status_code=409)
        client = self._get_client()
        key = self._key(blob_hash)
        created = True
        try:
            existing = client.head_object(Bucket=self.bucket, Key=key)
        except BaseException as exc:
            if not self._missing(exc):
                raise RuntimeError("volunteer_s3_head_failed") from exc
        else:
            created = False
            if (existing.get("Metadata") or {}).get("sha256") != _digest(blob_hash):
                raise VolunteerProtocolError("volunteer_blob_collision", status_code=409)
        if created:
            try:
                with source.open("rb") as handle:
                    client.upload_fileobj(
                        handle,
                        self.bucket,
                        key,
                        ExtraArgs={
                            "ContentType": "application/octet-stream",
                            "Metadata": {"sha256": _digest(blob_hash)},
                        },
                    )
            except BaseException as exc:
                raise RuntimeError("volunteer_s3_upload_failed") from exc
        return with_public_safety(
            {
                "schema": BLOB_REF_SCHEMA,
                "backend": self.backend,
                "blob_hash": blob_hash,
                "byte_count": int(source.stat().st_size),
                "created": created,
                "content_addressed": True,
            }
        )

    def local_path(self, _blob_hash: str) -> Path:
        raise VolunteerProtocolError("volunteer_s3_blob_not_local", status_code=409)

    def presign_download(self, blob_hash: str, *, expires_seconds: int = 900) -> str:
        if expires_seconds < 60 or expires_seconds > 3600:
            raise ValueError("volunteer_presign_expiry_out_of_bounds")
        try:
            return str(
                self._get_client().generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.bucket, "Key": self._key(blob_hash)},
                    ExpiresIn=int(expires_seconds),
                )
            )
        except BaseException as exc:
            raise RuntimeError("volunteer_s3_presign_failed") from exc

    def public_report(self) -> dict[str, Any]:
        return with_public_safety(
            {
                "schema": BLOB_STORE_SCHEMA,
                "backend": self.backend,
                "content_addressed": True,
                "atomic_blob_writes": True,
                "s3_compatible": True,
                "minio_compatible": True,
                "presigned_download_supported": True,
                "bucket_hash": _sha256_bytes(self.bucket.encode("utf-8")),
                "prefix_hash": _sha256_bytes(self.prefix.encode("utf-8")),
                "custom_endpoint_configured": bool(self.endpoint_url),
                "credential_env_names_public": False,
            }
        )


class ResumableUploadManager:
    """Persist chunk receipts so an upload survives transport/process failure."""

    def __init__(
        self,
        root: str | Path,
        *,
        blob_store: VolunteerBlobStore,
        max_upload_bytes: int,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        clock: Any = time.time,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.sessions = self.root / "sessions"
        self.lock_path = self.root / "uploads.lock"
        self.blob_store = blob_store
        self.max_upload_bytes = int(max_upload_bytes)
        self.chunk_bytes = int(chunk_bytes)
        self.clock = clock
        if self.max_upload_bytes < 1 or self.chunk_bytes < 1024:
            raise ValueError("volunteer_upload_limits_invalid")
        self.sessions.mkdir(parents=True, exist_ok=True)
        self.sessions.chmod(0o700)

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _session_dir(self, upload_id: str) -> Path:
        value = str(upload_id)
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise VolunteerProtocolError("volunteer_upload_id_invalid", status_code=400)
        return self.sessions / value

    def _manifest_path(self, upload_id: str) -> Path:
        return self._session_dir(upload_id) / "session.json"

    def _load(self, upload_id: str) -> dict[str, Any]:
        try:
            value = _read_json(self._manifest_path(upload_id))
        except FileNotFoundError as exc:
            raise VolunteerProtocolError("volunteer_upload_not_found", status_code=404) from exc
        if value.get("schema") != UPLOAD_SESSION_SCHEMA:
            raise VolunteerProtocolError("volunteer_upload_session_invalid", status_code=409)
        return value

    @staticmethod
    def _public(value: dict[str, Any]) -> dict[str, Any]:
        return with_public_safety(
            {
                "schema": value["schema"],
                "upload_id": value["upload_id"],
                "state": value["state"],
                "expected_blob_hash": value["expected_blob_hash"],
                "total_bytes": int(value["total_bytes"]),
                "chunk_bytes": int(value["chunk_bytes"]),
                "chunk_count": int(value["chunk_count"]),
                "received_chunk_indexes": sorted(int(item) for item in value["received_chunks"]),
                "received_chunk_count": len(value["received_chunks"]),
                "received_bytes": int(value["received_bytes"]),
                "start_count": int(value.get("start_count") or 1),
                "resume_count": int(value.get("resume_count") or 0),
                "chunk_replay_count": int(value.get("chunk_replay_count") or 0),
                "complete": value["state"] == "complete",
                "blob_ref": value.get("blob_ref") or {},
                "resumable_after_restart": True,
            }
        )

    def start(
        self,
        *,
        owner_cell_hash: str,
        idempotency_key: str,
        expected_blob_hash: str,
        total_bytes: int,
        private_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        _digest(expected_blob_hash)
        size = int(total_bytes)
        if size < 1 or size > self.max_upload_bytes:
            raise VolunteerProtocolError("volunteer_upload_size_out_of_bounds", status_code=413)
        if not str(owner_cell_hash).startswith("sha256:") or not str(idempotency_key):
            raise VolunteerProtocolError("volunteer_upload_identity_invalid", status_code=400)
        metadata_hash = sha256_json(private_metadata)
        upload_id = sha256_json(
            {
                "owner_cell_hash": owner_cell_hash,
                "idempotency_key": idempotency_key,
                "expected_blob_hash": expected_blob_hash,
            }
        ).split(":", 1)[1]
        with self._lock():
            path = self._manifest_path(upload_id)
            if path.is_file():
                existing = self._load(upload_id)
                if any(
                    existing[field] != wanted
                    for field, wanted in {
                        "owner_cell_hash": owner_cell_hash,
                        "idempotency_key": idempotency_key,
                        "expected_blob_hash": expected_blob_hash,
                        "total_bytes": size,
                        "metadata_hash": metadata_hash,
                    }.items()
                ):
                    raise VolunteerProtocolError("volunteer_upload_idempotency_collision")
                existing["start_count"] = int(existing.get("start_count") or 1) + 1
                existing["resume_count"] = int(existing.get("resume_count") or 0) + 1
                existing["updated_at"] = float(self.clock())
                _atomic_json(path, existing)
                return self._public(existing)
            directory = self._session_dir(upload_id)
            (directory / "chunks").mkdir(parents=True, exist_ok=False)
            directory.chmod(0o700)
            value = {
                "schema": UPLOAD_SESSION_SCHEMA,
                "upload_id": upload_id,
                "state": "active",
                "owner_cell_hash": owner_cell_hash,
                "idempotency_key": idempotency_key,
                "expected_blob_hash": expected_blob_hash,
                "total_bytes": size,
                "chunk_bytes": self.chunk_bytes,
                "chunk_count": int(math.ceil(size / self.chunk_bytes)),
                "received_chunks": {},
                "received_bytes": 0,
                "start_count": 1,
                "resume_count": 0,
                "chunk_replay_count": 0,
                "metadata_hash": metadata_hash,
                "private_metadata": private_metadata,
                "blob_ref": {},
                "created_at": float(self.clock()),
                "updated_at": float(self.clock()),
            }
            _atomic_json(path, value)
            return self._public(value)

    def status(self, upload_id: str, *, owner_cell_hash: str) -> dict[str, Any]:
        with self._lock():
            value = self._load(upload_id)
            if value["owner_cell_hash"] != owner_cell_hash:
                raise VolunteerProtocolError("volunteer_upload_owner_mismatch", status_code=403)
            return self._public(value)

    def put_chunk(
        self,
        upload_id: str,
        *,
        owner_cell_hash: str,
        chunk_index: int,
        chunk_hash: str,
        value: bytes,
    ) -> dict[str, Any]:
        _digest(chunk_hash)
        with self._lock():
            manifest = self._load(upload_id)
            if manifest["owner_cell_hash"] != owner_cell_hash:
                raise VolunteerProtocolError("volunteer_upload_owner_mismatch", status_code=403)
            if manifest["state"] == "complete":
                return self._public(manifest)
            index = int(chunk_index)
            if index < 0 or index >= int(manifest["chunk_count"]):
                raise VolunteerProtocolError("volunteer_upload_chunk_index_invalid", status_code=400)
            expected_size = min(
                int(manifest["chunk_bytes"]),
                int(manifest["total_bytes"]) - index * int(manifest["chunk_bytes"]),
            )
            if len(value) != expected_size or _sha256_bytes(value) != chunk_hash:
                raise VolunteerProtocolError("volunteer_upload_chunk_integrity_failed", status_code=409)
            chunks = manifest["received_chunks"]
            key = str(index)
            chunk_path = self._session_dir(upload_id) / "chunks" / f"{index:08d}.chunk"
            if key in chunks:
                if chunks[key]["chunk_hash"] != chunk_hash or not chunk_path.is_file():
                    raise VolunteerProtocolError("volunteer_upload_chunk_collision", status_code=409)
                manifest["chunk_replay_count"] = int(
                    manifest.get("chunk_replay_count") or 0
                ) + 1
                manifest["updated_at"] = float(self.clock())
                _atomic_json(self._manifest_path(upload_id), manifest)
                return self._public(manifest)
            temporary = chunk_path.with_name(f".{chunk_path.name}.{secrets.token_hex(4)}.tmp")
            try:
                with temporary.open("xb") as handle:
                    handle.write(value)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.chmod(0o600)
                os.replace(temporary, chunk_path)
            finally:
                temporary.unlink(missing_ok=True)
            chunks[key] = {"chunk_hash": chunk_hash, "byte_count": len(value)}
            manifest["received_bytes"] = sum(int(item["byte_count"]) for item in chunks.values())
            manifest["updated_at"] = float(self.clock())
            _atomic_json(self._manifest_path(upload_id), manifest)
            return self._public(manifest)

    def complete(self, upload_id: str, *, owner_cell_hash: str) -> dict[str, Any]:
        with self._lock():
            manifest = self._load(upload_id)
            if manifest["owner_cell_hash"] != owner_cell_hash:
                raise VolunteerProtocolError("volunteer_upload_owner_mismatch", status_code=403)
            if manifest["state"] == "complete":
                return {**self._public(manifest), "private_metadata": manifest["private_metadata"]}
            if len(manifest["received_chunks"]) != int(manifest["chunk_count"]):
                raise VolunteerProtocolError("volunteer_upload_chunks_incomplete", status_code=409)
            directory = self._session_dir(upload_id)
            assembled = directory / f".assembled-{secrets.token_hex(4)}.tmp"
            try:
                with assembled.open("xb") as output:
                    for index in range(int(manifest["chunk_count"])):
                        chunk_path = directory / "chunks" / f"{index:08d}.chunk"
                        with chunk_path.open("rb") as handle:
                            shutil.copyfileobj(handle, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                if (
                    assembled.stat().st_size != int(manifest["total_bytes"])
                    or sha256_file(assembled) != manifest["expected_blob_hash"]
                ):
                    raise VolunteerProtocolError("volunteer_upload_assembled_hash_mismatch")
                blob_ref = self.blob_store.put_file(
                    assembled, expected_hash=manifest["expected_blob_hash"]
                )
                completed_path = directory / "completed.blob"
                temporary = completed_path.with_name(
                    f".{completed_path.name}.{secrets.token_hex(4)}.tmp"
                )
                try:
                    shutil.copyfile(assembled, temporary)
                    temporary.chmod(0o600)
                    os.replace(temporary, completed_path)
                finally:
                    temporary.unlink(missing_ok=True)
            finally:
                assembled.unlink(missing_ok=True)
            manifest["state"] = "complete"
            manifest["blob_ref"] = blob_ref
            manifest["completed_local_path"] = str(completed_path)
            manifest["updated_at"] = float(self.clock())
            _atomic_json(self._manifest_path(upload_id), manifest)
            shutil.rmtree(directory / "chunks")
            result = self._public(manifest)
            result["private_metadata"] = manifest["private_metadata"]
            return result

    def completed_blob_path(self, upload_id: str, *, owner_cell_hash: str) -> Path:
        with self._lock():
            manifest = self._load(upload_id)
            if manifest["owner_cell_hash"] != owner_cell_hash:
                raise VolunteerProtocolError("volunteer_upload_owner_mismatch", status_code=403)
            if manifest["state"] != "complete":
                raise VolunteerProtocolError("volunteer_upload_not_complete", status_code=409)
            completed = Path(str(manifest.get("completed_local_path") or ""))
            if (
                completed.is_file()
                and sha256_file(completed) == manifest["expected_blob_hash"]
            ):
                return completed
            return self.blob_store.local_path(manifest["expected_blob_hash"])

    def public_report(self) -> dict[str, Any]:
        sessions = [
            _read_json(path)
            for path in sorted(self.sessions.glob("*/session.json"))
        ]
        return with_public_safety(
            {
                "schema": "crowdtensor_volunteer_upload_manager_report_v1",
                "session_count": len(sessions),
                "active_session_count": sum(item["state"] == "active" for item in sessions),
                "completed_session_count": sum(item["state"] == "complete" for item in sessions),
                "resumed_session_count": sum(
                    int(item.get("resume_count") or 0) > 0 for item in sessions
                ),
                "total_resume_count": sum(
                    int(item.get("resume_count") or 0) for item in sessions
                ),
                "chunk_replay_count": sum(
                    int(item.get("chunk_replay_count") or 0) for item in sessions
                ),
                "declared_upload_bytes": sum(
                    int(item.get("total_bytes") or 0) for item in sessions
                ),
                "persisted_received_bytes": sum(
                    int(item.get("received_bytes") or 0) for item in sessions
                ),
                "completed_upload_bytes": sum(
                    int(item.get("total_bytes") or 0)
                    for item in sessions
                    if item.get("state") == "complete"
                ),
                "resumable_after_process_restart": True,
                "chunk_hash_validation": True,
                "assembled_blob_hash_validation": True,
                "idempotent_chunk_replay": True,
                "blob_store": self.blob_store.public_report(),
            }
        )

    def cleanup(self) -> dict[str, Any]:
        if self.root.exists():
            shutil.rmtree(self.root)
        return with_public_safety(
            {
                "schema": "crowdtensor_volunteer_upload_cleanup_v1",
                "ok": True,
                "upload_sessions_removed": not self.root.exists(),
                "live_uploads_left_running": False,
            }
        )
