from __future__ import annotations

import hashlib

import pytest

from crowdtensor.training_contract import sha256_file
from crowdtensor.volunteer_training_protocol import VolunteerProtocolError
from crowdtensor.volunteer_training_storage import (
    LocalVolunteerBlobStore,
    ResumableUploadManager,
    S3VolunteerBlobStore,
)


def _hash(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def test_resumable_upload_survives_manager_restart_and_chunk_replay(tmp_path) -> None:
    data = bytes(range(251)) * 13
    source = tmp_path / "source.bin"
    source.write_bytes(data)
    store = LocalVolunteerBlobStore(tmp_path / "blobs")
    manager = ResumableUploadManager(
        tmp_path / "uploads",
        blob_store=store,
        max_upload_bytes=10_000,
        chunk_bytes=1024,
        clock=lambda: 100.0,
    )
    started = manager.start(
        owner_cell_hash="sha256:" + "1" * 64,
        idempotency_key="result-1",
        expected_blob_hash=_hash(data),
        total_bytes=len(data),
        private_metadata={"lease_token": "private", "result_id": "result-1"},
    )
    upload_id = started["upload_id"]
    first = data[:1024]
    manager.put_chunk(
        upload_id,
        owner_cell_hash="sha256:" + "1" * 64,
        chunk_index=0,
        chunk_hash=_hash(first),
        value=first,
    )
    replay = manager.put_chunk(
        upload_id,
        owner_cell_hash="sha256:" + "1" * 64,
        chunk_index=0,
        chunk_hash=_hash(first),
        value=first,
    )
    assert replay["received_chunk_count"] == 1

    recovered = ResumableUploadManager(
        tmp_path / "uploads",
        blob_store=store,
        max_upload_bytes=10_000,
        chunk_bytes=1024,
        clock=lambda: 200.0,
    )
    restarted = recovered.start(
        owner_cell_hash="sha256:" + "1" * 64,
        idempotency_key="result-1",
        expected_blob_hash=_hash(data),
        total_bytes=len(data),
        private_metadata={"lease_token": "private", "result_id": "result-1"},
    )
    assert restarted["resume_count"] == 1
    assert recovered.status(upload_id, owner_cell_hash="sha256:" + "1" * 64)[
        "received_chunk_indexes"
    ] == [0]
    for index, offset in enumerate(range(1024, len(data), 1024), start=1):
        chunk = data[offset : offset + 1024]
        recovered.put_chunk(
            upload_id,
            owner_cell_hash="sha256:" + "1" * 64,
            chunk_index=index,
            chunk_hash=_hash(chunk),
            value=chunk,
        )
    complete = recovered.complete(
        upload_id, owner_cell_hash="sha256:" + "1" * 64
    )
    assert complete["complete"] is True
    assert complete["private_metadata"]["lease_token"] == "private"
    assert store.get_bytes(_hash(data)) == data
    assert recovered.complete(
        upload_id, owner_cell_hash="sha256:" + "1" * 64
    )["complete"] is True
    public = recovered.public_report()
    assert public["completed_session_count"] == 1
    assert public["resumed_session_count"] == 1
    assert public["completed_upload_bytes"] == len(data)
    assert "lease_token" not in str(public)
    assert "result-1" not in str(public)


def test_upload_rejects_wrong_owner_chunk_and_assembled_hash(tmp_path) -> None:
    data = b"a" * 1500
    manager = ResumableUploadManager(
        tmp_path / "uploads",
        blob_store=LocalVolunteerBlobStore(tmp_path / "blobs"),
        max_upload_bytes=2000,
        chunk_bytes=1024,
    )
    session = manager.start(
        owner_cell_hash="sha256:" + "2" * 64,
        idempotency_key="bad",
        expected_blob_hash=_hash(data),
        total_bytes=len(data),
        private_metadata={},
    )
    with pytest.raises(VolunteerProtocolError, match="owner_mismatch"):
        manager.put_chunk(
            session["upload_id"],
            owner_cell_hash="sha256:" + "3" * 64,
            chunk_index=0,
            chunk_hash=_hash(data[:1024]),
            value=data[:1024],
        )
    with pytest.raises(VolunteerProtocolError, match="chunk_integrity"):
        manager.put_chunk(
            session["upload_id"],
            owner_cell_hash="sha256:" + "2" * 64,
            chunk_index=0,
            chunk_hash=_hash(b"wrong"),
            value=data[:1024],
        )


class MissingObject(Exception):
    response = {"Error": {"Code": "404"}}


class FakeS3:
    def __init__(self) -> None:
        self.objects = {}

    def head_object(self, *, Bucket, Key):
        try:
            value = self.objects[(Bucket, Key)]
        except KeyError as exc:
            raise MissingObject() from exc
        return {"Metadata": value["Metadata"]}

    def upload_fileobj(self, handle, bucket, key, ExtraArgs):
        self.objects[(bucket, key)] = {
            "Body": handle.read(),
            "Metadata": ExtraArgs["Metadata"],
        }

    def generate_presigned_url(self, operation, *, Params, ExpiresIn):
        return f"https://objects.invalid/{Params['Bucket']}/{Params['Key']}?ttl={ExpiresIn}"


def test_s3_minio_contract_uses_content_hash_and_private_presign(tmp_path) -> None:
    source = tmp_path / "blob"
    source.write_bytes(b"content")
    client = FakeS3()
    store = S3VolunteerBlobStore(bucket="private", client=client)
    first = store.put_file(source)
    second = store.put_file(source)
    assert first["created"] is True
    assert second["created"] is False
    assert first["blob_hash"] == sha256_file(source)
    assert store.presign_download(first["blob_hash"], expires_seconds=120).startswith(
        "https://objects.invalid/"
    )
    public = store.public_report()
    assert public["s3_compatible"] is True
    assert public["minio_compatible"] is True
    assert public["presigned_download_supported"] is True
    assert "objects.invalid" not in str(public)
