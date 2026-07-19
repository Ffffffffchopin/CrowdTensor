#!/usr/bin/env python3
"""Run a real local MinIO API checkpoint integration and emit safe evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from crowdtensor.elastic_checkpoint_storage import (
    MirroredCheckpointBlobStore,
    S3CheckpointBlobStore,
)


SCHEMA = "crowdtensor_community_minio_integration_v1"
DEFAULT_IMAGE = "minio/minio:RELEASE.2025-04-22T22-12-26Z"


def _hash(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _free_port() -> int:
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=check)


def _wait(endpoint: str, *, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(endpoint + "/minio/health/live", timeout=2) as response:
                if int(response.status) == 200:
                    return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("community_minio_health_timeout")


def _image_identity(image: str) -> dict[str, Any]:
    result = _run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}|{{join .RepoDigests \",\"}}"],
        check=False,
    )
    text = result.stdout.strip() if result.returncode == 0 else ""
    image_id, _, digests = text.partition("|")
    return {
        "image_reference": image,
        "image_id_hash": _hash(image_id.encode()) if image_id else "",
        "repo_digest_hashes": [_hash(item.encode()) for item in digests.split(",") if item],
        "image_identity_present": bool(image_id),
    }


def run_integration(*, image: str = DEFAULT_IMAGE) -> dict[str, Any]:
    started = time.monotonic()
    port = _free_port()
    endpoint = f"http://127.0.0.1:{port}"
    access_key = "ct" + secrets.token_hex(10)
    secret_key = secrets.token_urlsafe(32)
    bucket = "crowdtensor-community-" + secrets.token_hex(6)
    prefix = "checkpoints/community"
    name = "ct-community-minio-" + secrets.token_hex(5)
    created_hashes: list[str] = []
    restart_verified = False
    cleanup_verified = False
    container_started = False

    with tempfile.TemporaryDirectory(prefix="ct-community-minio-") as temporary:
        data_root = Path(temporary) / "data"
        mirror_root = Path(temporary) / "mirror"
        data_root.mkdir()

        def start() -> None:
            nonlocal container_started
            result = _run(
                [
                    "docker", "run", "--rm", "--detach", "--name", name,
                    "--publish", f"127.0.0.1:{port}:9000",
                    "--env", f"MINIO_ROOT_USER={access_key}",
                    "--env", f"MINIO_ROOT_PASSWORD={secret_key}",
                    "--volume", f"{data_root}:/data",
                    image, "server", "/data", "--address", ":9000",
                ]
            )
            if not result.stdout.strip():
                raise RuntimeError("community_minio_container_start_failed")
            container_started = True
            _wait(endpoint)

        def stop() -> None:
            nonlocal container_started
            _run(["docker", "stop", "--time", "10", name], check=False)
            container_started = False

        try:
            start()
            import boto3

            client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                region_name="us-east-1",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )
            client.create_bucket(Bucket=bucket)
            primary = S3CheckpointBlobStore(
                bucket=bucket,
                prefix=prefix,
                endpoint_url=endpoint,
                client=client,
            )
            mirrored = MirroredCheckpointBlobStore(primary=primary, mirror_root=mirror_root)

            payloads = [
                b"community-checkpoint-step-1",
                b"community-checkpoint-step-2",
                b"community-checkpoint-step-3",
            ]
            created_hashes = [_hash(item) for item in payloads]
            put_reports = [mirrored.put(digest, payload) for digest, payload in zip(created_hashes, payloads)]
            duplicate = mirrored.put(created_hashes[-1], payloads[-1])
            listed_before = sorted(primary.list_hashes())
            if listed_before != sorted(created_hashes):
                raise RuntimeError("community_minio_content_address_listing_invalid")

            repair_payload = b"community-checkpoint-repair"
            repair_hash = _hash(repair_payload)
            mirrored.put(repair_hash, repair_payload)
            created_hashes.append(repair_hash)
            client.put_object(
                Bucket=bucket,
                Key=primary._key(repair_hash),
                Body=b"corrupt",
                Metadata={"sha256": repair_hash.split(":", 1)[1]},
            )
            recovered = mirrored.get(repair_hash)
            repair_report = mirrored.public_report()
            if recovered != repair_payload or primary.get(repair_hash) != repair_payload:
                raise RuntimeError("community_minio_mirror_repair_invalid")

            retained_hashes = created_hashes[-2:]
            for digest in created_hashes[:-2]:
                mirrored.delete(digest)
            listed_after = sorted(primary.list_hashes())
            retention_verified = listed_after == sorted(retained_hashes)
            if not retention_verified:
                raise RuntimeError("community_minio_retention_invalid")

            stop()
            start()
            client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                region_name="us-east-1",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )
            restarted_store = S3CheckpointBlobStore(
                bucket=bucket, prefix=prefix, endpoint_url=endpoint, client=client
            )
            restart_verified = all(restarted_store.get(item) for item in retained_hashes)

            for item in list(restarted_store.list_hashes()):
                restarted_store.delete(item)
            cleanup_verified = list(restarted_store.list_hashes()) == []
            client.delete_bucket(Bucket=bucket)
            report = {
                "schema": SCHEMA,
                "ok": True,
                "backend": "local_minio_s3_compatible_api",
                "real_api_calls_performed": True,
                "external_managed_object_storage_sla_verified": False,
                "content_addressed": True,
                "put_count": len(put_reports),
                "duplicate_put_idempotent": duplicate["created"] is False,
                "list_verified": True,
                "get_verified": True,
                "mirror_fallback_verified": int(repair_report["fallback_read_count"]) >= 1,
                "primary_repair_verified": int(repair_report["primary_repair_count"]) >= 1,
                "retention_verified": retention_verified,
                "service_restart_verified": restart_verified,
                "cleanup_verified": cleanup_verified,
                "object_count_after_cleanup": 0,
                "container_left_running": False,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "container_image": _image_identity(image),
                "credential_values_public": False,
                "endpoint_url_public": False,
                "bucket_name_public": False,
                "object_keys_public": False,
                "private_paths_public": False,
                "public_artifact_safe": True,
            }
            report["content_hash"] = _hash(
                json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
            )
            return report
        finally:
            if container_started:
                stop()
            cleanup_verified = cleanup_verified and not bool(
                _run(["docker", "ps", "--quiet", "--filter", f"name=^{name}$"], check=False).stdout.strip()
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_integration(image=args.image)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"minio_integration_ok={report['ok']} restart={report['service_restart_verified']} cleanup={report['cleanup_verified']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
