"""Content-addressed checkpoint storage for elastic training jobs."""

from __future__ import annotations

import hashlib
import os
import secrets
import threading
from pathlib import Path
from typing import Any, Iterable, Protocol


STORAGE_SCHEMA = "crowdtensor_elastic_checkpoint_storage_v1"


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _digest(archive_hash: str) -> str:
    value = str(archive_hash)
    if not value.startswith("sha256:") or len(value) != 71:
        raise ValueError("elastic_checkpoint_archive_hash_invalid")
    digest = value.split(":", 1)[1]
    if any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("elastic_checkpoint_archive_hash_invalid")
    return digest


class CheckpointBlobStore(Protocol):
    backend: str

    def put(self, archive_hash: str, value: bytes) -> dict[str, Any]: ...

    def get(self, archive_hash: str) -> bytes: ...

    def delete(self, archive_hash: str) -> bool: ...

    def list_hashes(self) -> Iterable[str]: ...

    def public_report(self) -> dict[str, Any]: ...

    def private_configuration(self) -> dict[str, Any]: ...


class LocalCheckpointBlobStore:
    backend = "local"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)

    def path_for_hash(self, archive_hash: str) -> Path:
        digest = _digest(archive_hash)
        return self.root / digest[:2] / f"{digest}.zip"

    def put(self, archive_hash: str, value: bytes) -> dict[str, Any]:
        if _sha256_bytes(value) != archive_hash:
            raise ValueError("elastic_checkpoint_archive_hash_mismatch")
        path = self.path_for_hash(archive_hash)
        if path.is_file():
            if _sha256_bytes(path.read_bytes()) != archive_hash:
                raise RuntimeError("elastic_checkpoint_blob_collision")
            return {
                "created": False,
                "archive_hash": archive_hash,
                "archive_bytes": len(value),
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(0o700)
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
        return {
            "created": True,
            "archive_hash": archive_hash,
            "archive_bytes": len(value),
        }

    def get(self, archive_hash: str) -> bytes:
        try:
            value = self.path_for_hash(archive_hash).read_bytes()
        except OSError as exc:
            raise RuntimeError("elastic_committed_checkpoint_blob_missing") from exc
        if _sha256_bytes(value) != archive_hash:
            raise RuntimeError("elastic_committed_checkpoint_blob_hash_invalid")
        return value

    def delete(self, archive_hash: str) -> bool:
        path = self.path_for_hash(archive_hash)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        try:
            path.parent.rmdir()
        except OSError:
            pass
        return True

    def list_hashes(self) -> Iterable[str]:
        for path in self.root.glob("*/*.zip"):
            if len(path.stem) == 64:
                yield "sha256:" + path.stem

    def public_report(self) -> dict[str, Any]:
        return {
            "schema": STORAGE_SCHEMA,
            "backend": self.backend,
            "content_addressed": True,
            "atomic_blob_writes": True,
            "s3_compatible": False,
            "private_paths_public": False,
            "credential_values_public": False,
            "public_artifact_safe": True,
        }

    def private_configuration(self) -> dict[str, Any]:
        return {
            "schema": STORAGE_SCHEMA,
            "backend": self.backend,
            "root": str(self.root),
        }


class S3CheckpointBlobStore:
    """S3/MinIO blob store using a lazily imported boto3 client."""

    backend = "s3"

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "crowdtensor/checkpoints",
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
        if not self.bucket or not self.access_key_env or not self.secret_key_env:
            raise ValueError("elastic_s3_storage_configuration_invalid")
        self._client = client

    def _key(self, archive_hash: str) -> str:
        digest = _digest(archive_hash)
        suffix = f"{digest[:2]}/{digest}.zip"
        return f"{self.prefix}/{suffix}" if self.prefix else suffix

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "elastic_s3_storage_requires_boto3_install_crowdtensord_storage_extra"
            ) from exc
        access_key = os.environ.get(self.access_key_env, "")
        secret_key = os.environ.get(self.secret_key_env, "")
        session_token = os.environ.get(self.session_token_env, "") if self.session_token_env else ""
        if not access_key or not secret_key:
            raise RuntimeError("elastic_s3_storage_private_credentials_missing")
        kwargs: dict[str, Any] = {
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
        }
        if session_token:
            kwargs["aws_session_token"] = session_token
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        if self.region_name:
            kwargs["region_name"] = self.region_name
        self._client = boto3.client("s3", **kwargs)
        return self._client

    @staticmethod
    def _missing(exc: BaseException) -> bool:
        response = getattr(exc, "response", {})
        code = str((response.get("Error") or {}).get("Code") or "")
        return code in {"404", "NoSuchKey", "NotFound"}

    def put(self, archive_hash: str, value: bytes) -> dict[str, Any]:
        if _sha256_bytes(value) != archive_hash:
            raise ValueError("elastic_checkpoint_archive_hash_mismatch")
        client = self._get_client()
        key = self._key(archive_hash)
        created = True
        try:
            existing = client.head_object(Bucket=self.bucket, Key=key)
        except BaseException as exc:
            if not self._missing(exc):
                raise RuntimeError("elastic_s3_checkpoint_head_failed") from exc
        else:
            created = False
            metadata = dict(existing.get("Metadata") or {})
            if metadata.get("sha256") != _digest(archive_hash):
                raise RuntimeError("elastic_checkpoint_blob_collision")
        if created:
            try:
                client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=value,
                    ContentType="application/vnd.crowdtensor.stage-checkpoint+zip",
                    Metadata={"sha256": _digest(archive_hash)},
                )
            except BaseException as exc:
                raise RuntimeError("elastic_s3_checkpoint_put_failed") from exc
        return {
            "created": created,
            "archive_hash": archive_hash,
            "archive_bytes": len(value),
        }

    def get(self, archive_hash: str) -> bytes:
        try:
            response = self._get_client().get_object(
                Bucket=self.bucket, Key=self._key(archive_hash)
            )
            value = response["Body"].read()
        except BaseException as exc:
            raise RuntimeError("elastic_committed_checkpoint_blob_missing") from exc
        if _sha256_bytes(value) != archive_hash:
            raise RuntimeError("elastic_committed_checkpoint_blob_hash_invalid")
        return value

    def delete(self, archive_hash: str) -> bool:
        client = self._get_client()
        key = self._key(archive_hash)
        try:
            client.head_object(Bucket=self.bucket, Key=key)
        except BaseException as exc:
            if self._missing(exc):
                return False
            raise RuntimeError("elastic_s3_checkpoint_head_failed") from exc
        try:
            client.delete_object(Bucket=self.bucket, Key=key)
        except BaseException as exc:
            raise RuntimeError("elastic_s3_checkpoint_delete_failed") from exc
        return True

    def list_hashes(self) -> Iterable[str]:
        prefix = f"{self.prefix}/" if self.prefix else ""
        client = self._get_client()
        token = ""
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            try:
                response = client.list_objects_v2(**kwargs)
            except BaseException as exc:
                raise RuntimeError("elastic_s3_checkpoint_list_failed") from exc
            for item in response.get("Contents") or []:
                name = str(item.get("Key") or "").rsplit("/", 1)[-1]
                digest = name[:-4] if name.endswith(".zip") else ""
                if len(digest) == 64 and all(
                    character in "0123456789abcdef" for character in digest
                ):
                    yield "sha256:" + digest
            if not response.get("IsTruncated"):
                break
            token = str(response.get("NextContinuationToken") or "")
            if not token:
                raise RuntimeError("elastic_s3_checkpoint_list_token_missing")

    def public_report(self) -> dict[str, Any]:
        return {
            "schema": STORAGE_SCHEMA,
            "backend": self.backend,
            "content_addressed": True,
            "atomic_blob_writes": True,
            "s3_compatible": True,
            "minio_compatible": True,
            "bucket_hash": _sha256_bytes(self.bucket.encode("utf-8")),
            "prefix_hash": _sha256_bytes(self.prefix.encode("utf-8")),
            "custom_endpoint_configured": bool(self.endpoint_url),
            "credential_env_names_public": False,
            "credential_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }

    def private_configuration(self) -> dict[str, Any]:
        return {
            "schema": STORAGE_SCHEMA,
            "backend": self.backend,
            "bucket": self.bucket,
            "prefix": self.prefix,
            "endpoint_url": self.endpoint_url,
            "region_name": self.region_name,
            "access_key_env": self.access_key_env,
            "secret_key_env": self.secret_key_env,
            "session_token_env": self.session_token_env,
        }


class MirroredCheckpointBlobStore:
    """Integrity-checking primary store with an independent local recovery copy."""

    backend = "mirrored"

    def __init__(
        self,
        *,
        primary: CheckpointBlobStore,
        mirror_root: str | Path,
    ) -> None:
        if getattr(primary, "backend", "") == self.backend:
            raise ValueError("elastic_checkpoint_nested_mirror_invalid")
        self.primary = primary
        self.mirror = LocalCheckpointBlobStore(mirror_root)
        self._lock = threading.RLock()
        self._fallback_read_count = 0
        self._primary_repair_count = 0

    def put(self, archive_hash: str, value: bytes) -> dict[str, Any]:
        if _sha256_bytes(value) != archive_hash:
            raise ValueError("elastic_checkpoint_archive_hash_mismatch")
        with self._lock:
            mirror = self.mirror.put(archive_hash, value)
            primary = self.primary.put(archive_hash, value)
            if self.mirror.get(archive_hash) != value or self.primary.get(archive_hash) != value:
                raise RuntimeError("elastic_checkpoint_mirror_post_write_verify_failed")
        return {
            "created": bool(mirror.get("created") or primary.get("created")),
            "archive_hash": archive_hash,
            "archive_bytes": len(value),
            "primary_created": bool(primary.get("created")),
            "mirror_created": bool(mirror.get("created")),
            "mirrored_write_verified": True,
        }

    def get(self, archive_hash: str) -> bytes:
        try:
            return self.primary.get(archive_hash)
        except (OSError, RuntimeError, ValueError) as primary_error:
            with self._lock:
                try:
                    value = self.mirror.get(archive_hash)
                except (OSError, RuntimeError, ValueError) as mirror_error:
                    raise RuntimeError(
                        "elastic_committed_checkpoint_primary_and_mirror_invalid"
                    ) from mirror_error
                self._fallback_read_count += 1
                try:
                    self.primary.delete(archive_hash)
                except (OSError, RuntimeError, ValueError):
                    pass
                try:
                    self.primary.put(archive_hash, value)
                    repaired = self.primary.get(archive_hash)
                except (OSError, RuntimeError, ValueError) as repair_error:
                    raise RuntimeError(
                        "elastic_checkpoint_primary_repair_failed"
                    ) from repair_error
                if repaired != value:
                    raise RuntimeError("elastic_checkpoint_primary_repair_hash_invalid")
                self._primary_repair_count += 1
                return value

    def delete(self, archive_hash: str) -> bool:
        with self._lock:
            primary = self.primary.delete(archive_hash)
            mirror = self.mirror.delete(archive_hash)
        return bool(primary or mirror)

    def list_hashes(self) -> Iterable[str]:
        values = set(self.primary.list_hashes()) | set(self.mirror.list_hashes())
        return iter(sorted(values))

    def public_report(self) -> dict[str, Any]:
        return {
            "schema": STORAGE_SCHEMA,
            "backend": self.backend,
            "primary": self.primary.public_report(),
            "recovery_mirror": self.mirror.public_report(),
            "content_addressed": True,
            "atomic_blob_writes": True,
            "post_write_integrity_verified": True,
            "automatic_fallback_read": True,
            "automatic_primary_repair": True,
            "fallback_read_count": self._fallback_read_count,
            "primary_repair_count": self._primary_repair_count,
            "private_paths_public": False,
            "credential_values_public": False,
            "public_artifact_safe": True,
        }

    def private_configuration(self) -> dict[str, Any]:
        return {
            "schema": STORAGE_SCHEMA,
            "backend": self.backend,
            "primary": self.primary.private_configuration(),
            "mirror_root": str(self.mirror.root),
        }


def checkpoint_blob_store_from_configuration(
    value: dict[str, Any],
    *,
    default_root: str | Path,
) -> CheckpointBlobStore:
    backend = str(value.get("backend") or "local").strip().lower()
    if backend == "local":
        return LocalCheckpointBlobStore(value.get("root") or default_root)
    if backend == "s3":
        return S3CheckpointBlobStore(
            bucket=str(value.get("bucket") or ""),
            prefix=str(value.get("prefix") or "crowdtensor/checkpoints"),
            endpoint_url=str(value.get("endpoint_url") or ""),
            region_name=str(value.get("region_name") or ""),
            access_key_env=str(value.get("access_key_env") or "AWS_ACCESS_KEY_ID"),
            secret_key_env=str(value.get("secret_key_env") or "AWS_SECRET_ACCESS_KEY"),
            session_token_env=str(value.get("session_token_env") or "AWS_SESSION_TOKEN"),
        )
    if backend == "mirrored":
        primary_configuration = dict(value.get("primary") or {"backend": "local"})
        if str(primary_configuration.get("backend") or "local") == "mirrored":
            raise ValueError("elastic_checkpoint_nested_mirror_invalid")
        root = Path(default_root).expanduser().resolve()
        primary = checkpoint_blob_store_from_configuration(
            primary_configuration,
            default_root=root / "primary",
        )
        return MirroredCheckpointBlobStore(
            primary=primary,
            mirror_root=value.get("mirror_root") or root / "recovery-mirror",
        )
    raise ValueError("elastic_checkpoint_storage_backend_invalid")
