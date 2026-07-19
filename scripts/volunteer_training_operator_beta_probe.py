#!/usr/bin/env python3
"""Run the bounded same-host Volunteer Campaign Operator Beta gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

from crowdtensor.community_security import scan_public_files
from crowdtensor.hf_lora_training import create_local_training_fixture
from crowdtensor.named_tensor_optimizer import load_tensors, save_tensors
from crowdtensor.training_contract import delta_manifest, sha256_file, sha256_json
from crowdtensor.volunteer_training_cell import (
    HTTPVolunteerTransport,
    VolunteerUploadInterrupted,
)
from crowdtensor.volunteer_training_coordinator import VolunteerTrainingCoordinator
from crowdtensor.volunteer_training_protocol import (
    PROTOCOL_VERSION,
    SUBMISSION_SCHEMA,
    VolunteerProtocolError,
    with_public_safety,
)


SCHEMA = "crowdtensor_volunteer_campaign_single_host_operator_beta_probe_v1"
WORKER_SCHEMA = "crowdtensor_volunteer_operator_protocol_cell_v1"
MINIO_IMAGE = "minio/minio:RELEASE.2025-04-22T22-12-26Z"
CADDY_IMAGE = "caddy:2.10.0-alpine"
RETAINED_REAL_RC = Path(
    "dist/volunteer-training-internet-beta-engineering-rc-20260718-r3/"
    "volunteer_training_internet_beta_engineering_rc.json"
)


def _free_port() -> int:
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _run(
    command: list[str], *, check: bool = True, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=check,
        timeout=timeout,
    )


def _docker(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["docker", *command], check=check, timeout=120)


def _wait_http(
    url: str,
    *,
    verify: str | bool = True,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + float(timeout_seconds)
    last = ""
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, verify=verify, headers=headers, timeout=2.0)
            if response.status_code == 200:
                value = response.json()
                if isinstance(value, dict) and value.get("ok") is True:
                    return value
            last = f"http_{response.status_code}"
        except Exception as exc:  # bounded readiness polling
            last = type(exc).__name__
        time.sleep(0.25)
    raise RuntimeError("volunteer_operator_service_readiness_timeout:" + last)


def _wait_minio(endpoint: str, *, timeout_seconds: float = 60.0) -> None:
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        try:
            response = httpx.get(endpoint + "/minio/health/live", timeout=2.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise RuntimeError("volunteer_operator_minio_readiness_timeout")


def _image_report(image: str) -> dict[str, Any]:
    identity_result = _docker(
        ["image", "inspect", image, "--format", "{{.Id}}"], check=False
    )
    digest_result = _docker(
        ["image", "inspect", image, "--format", "{{json .RepoDigests}}"],
        check=False,
    )
    identity = identity_result.stdout.strip()
    try:
        digests = json.loads(digest_result.stdout or "[]")
    except json.JSONDecodeError:
        digests = []
    return {
        "reference": image,
        "image_id_hash": sha256_json({"image_id": identity}) if identity else "",
        "repo_digest_hashes": [
            sha256_json({"digest": item}) for item in digests if item
        ],
        "identity_verified": bool(identity),
    }


def _write(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o644)
    return path


def _private_write(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def _credential(
    client: httpx.Client,
    base_url: str,
    invite: str,
    cell_id: str,
    *,
    scopes: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"cell_id": cell_id, "ttl_seconds": 600}
    if scopes is not None:
        payload["scopes"] = scopes
    response = client.post(
        base_url + "/v1/volunteer/credentials/issue",
        headers={"Authorization": "Bearer " + invite},
        json=payload,
    )
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict) or not value.get("credential_token"):
        raise RuntimeError("volunteer_operator_credential_issue_invalid")
    return value


def _cell_headers(token: str, cell_id: str, nonce: str) -> dict[str, str]:
    return {
        "Authorization": "Bearer " + token,
        "X-CrowdTensor-Cell-Id": cell_id,
        "X-CrowdTensor-Nonce": nonce,
    }


def _manifest_from_template(
    template: dict[str, Any],
    campaign: dict[str, Any],
    work: dict[str, Any],
    cell_id: str,
) -> dict[str, Any]:
    result = dict(template)
    result.update(
        {
            "job_id": campaign["campaign_id"],
            "round_id": work["round_id"],
            "result_id": sha256_json(
                {
                    "cell_id": cell_id,
                    "round_id": work["round_id"],
                    "work_id": work["work_id"],
                }
            ),
            "miner_id": cell_id,
            "model_manifest_hash": campaign["model_manifest_hash"],
            "base_model_hash": campaign["base_model_hash"],
            "base_adapter_hash": work["base_adapter_hash"],
            "base_model_version": int(campaign["model_revision"]),
            "adapter_version": int(work["adapter_version"]),
            "dataset_shard_index": int(work["dataset_shard_index"]),
            "dataset_shard_hash": work["dataset_shard_hash"],
        }
    )
    return result


def run_worker(args: argparse.Namespace) -> int:
    started = time.monotonic()
    output = Path(args.worker_output).resolve()
    template = json.loads(Path(args.delta_template).read_text(encoding="utf-8"))
    transport = HTTPVolunteerTransport.from_invite(
        args.invite,
        timeout_seconds=20.0,
        resumable_uploads=True,
    )
    campaign = transport.campaign()
    accepted = False
    terminal_state = "timeout"
    error_codes: list[str] = []
    claim_count = 0
    deadline = time.monotonic() + float(args.worker_timeout)
    while time.monotonic() < deadline:
        try:
            claim = transport.claim(
                cell_id=args.cell_id,
                capability={"device": "protocol_fixture", "real_training": False},
            )
            claim_count += 1
            state = str(claim.get("state") or "")
            if state == "campaign_complete":
                terminal_state = state
                break
            work = claim.get("work_unit")
            if not isinstance(work, dict):
                time.sleep(0.15)
                continue
            manifest = _manifest_from_template(
                template, campaign, work, args.cell_id
            )
            response = transport.submit(
                cell_id=args.cell_id,
                work=work,
                delta_manifest=manifest,
            )
            accepted = response.get("accepted") is True
            terminal_state = "accepted" if accepted else "submission_rejected"
            if accepted:
                break
        except VolunteerProtocolError as exc:
            error_codes.append(exc.code)
            time.sleep(0.15)
        except httpx.HTTPError:
            error_codes.append("volunteer_worker_network_retry")
            time.sleep(0.2)
    report = with_public_safety(
        {
            "schema": WORKER_SCHEMA,
            "ok": accepted or terminal_state == "campaign_complete",
            "cell_id_hash": sha256_json({"cell_id": args.cell_id}),
            "independent_os_process": True,
            "real_training_performed": False,
            "protocol_fixture_delta": True,
            "claim_count": claim_count,
            "accepted_update": accepted,
            "terminal_state": terminal_state,
            "retry_count": len(error_codes),
            "error_classes": sorted(set(error_codes)),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
    )
    _private_write(output, report)
    return 0 if report["ok"] else 2


def _build_delta_template(
    coordinator: VolunteerTrainingCoordinator, output: Path
) -> dict[str, Any]:
    invite = coordinator.private_invite()["invite_token"]
    claim = coordinator.claim(cell_id="template-cell", invite_token=invite)["work_unit"]
    adapter = coordinator.artifact_path(
        claim["artifact_refs"]["base_adapter"]["artifact_id"],
        invite_token=invite,
    )
    tensors = load_tensors(adapter)
    import torch

    delta = {
        name: torch.full_like(value, 0.0001)
        for name, value in tensors.items()
    }
    delta_path = save_tensors(delta, output / "protocol-delta.safetensors")
    campaign = coordinator.campaign_manifest()
    manifest = delta_manifest(
        delta_path=delta_path,
        job_id=campaign["campaign_id"],
        round_id=claim["round_id"],
        result_id=sha256_json({"template": True}),
        miner_id="template-cell",
        model_manifest_hash=campaign["model_manifest_hash"],
        base_model_hash=campaign["base_model_hash"],
        base_adapter_hash=claim["base_adapter_hash"],
        base_model_version=campaign["model_revision"],
        adapter_version=claim["adapter_version"],
        dataset_shard_index=claim["dataset_shard_index"],
        dataset_shard_hash=claim["dataset_shard_hash"],
        loss_start=4.0,
        loss_end=3.9,
        samples_seen=2,
        tokens_seen=16,
    )
    invariant = {
        key: value
        for key, value in manifest.items()
        if key
        not in {
            "job_id",
            "round_id",
            "result_id",
            "miner_id",
            "model_manifest_hash",
            "base_model_hash",
            "base_adapter_hash",
            "base_model_version",
            "adapter_version",
            "dataset_shard_index",
            "dataset_shard_hash",
        }
    }
    template_path = _private_write(output / "delta-template.json", invariant)
    return {"manifest": invariant, "template_path": template_path}


def _force_expire_active_leases(
    coordinator: VolunteerTrainingCoordinator, invite: str
) -> int:
    """Bound the slow-Cell fault without making the live gate sleep a full lease."""

    with coordinator._locked_state() as state:
        round_state = coordinator._current_round(state)
        for work in (round_state or {}).get("work_units", {}).values():
            if work.get("state") == "leased":
                work["lease_expires_at"] = float(coordinator.clock()) - 1.0
        coordinator._save_state(state)
    return int(
        coordinator.expire_leases(invite_token=invite)["expired_lease_count"]
    )


def _start_backend(
    *,
    campaign_dir: Path,
    backend_port: int,
    public_url: str,
    proxy_id: str,
    minio_endpoint: str,
    bucket: str,
    access_key: str,
    secret_key: str,
    log_path: Path,
) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env.update(
        {
            "CT_OPERATOR_S3_ACCESS": access_key,
            "CT_OPERATOR_S3_SECRET": secret_key,
            "PYTHONUNBUFFERED": "1",
        }
    )
    command = [
        sys.executable,
        "-m",
        "crowdtensor.volunteer_training_cli",
        "serve",
        str(campaign_dir),
        "--host",
        "0.0.0.0",
        "--port",
        str(backend_port),
        "--public-url",
        public_url,
        "--require-https",
        "--trust-forwarded-headers",
        "--trusted-proxy-id",
        proxy_id,
        "--upload-chunk-bytes",
        "1024",
        "--upload-storage",
        "s3",
        "--s3-bucket",
        bucket,
        "--s3-prefix",
        "operator-beta/uploads",
        "--s3-endpoint",
        minio_endpoint,
        "--s3-access-key-env",
        "CT_OPERATOR_S3_ACCESS",
        "--s3-secret-key-env",
        "CT_OPERATOR_S3_SECRET",
        "--s3-session-token-env",
        "",
        "--json",
    ]
    log = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=Path.cwd(),
        env=env,
        text=True,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    log.close()
    return process


def _stop_process(process: subprocess.Popen[str] | None) -> bool:
    if process is None or process.poll() is not None:
        return True
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    return process.poll() is not None


def _strict_real_rc_check(repo: Path) -> dict[str, Any]:
    report_path = repo / RETAINED_REAL_RC
    result = _run(
        [
            sys.executable,
            "scripts/volunteer_training_internet_beta_check.py",
            "--report",
            str(report_path),
            "--require-ready",
            "--json",
        ],
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError("volunteer_operator_retained_real_peft_rc_invalid")
    retained = json.loads(report_path.read_text(encoding="utf-8"))
    return with_public_safety(
        {
            "retained_real_peft_rc_verified": True,
            "retained_real_peft_rc_file_hash": sha256_file(report_path),
            "retained_real_peft_rc_content_hash": retained.get("content_hash"),
            "model_id": "HuggingFaceTB/SmolLM2-135M",
            "dataset_id": "Salesforce/wikitext",
            "real_training_round_count": int(
                (retained.get("round_progress") or {}).get("completed_rounds") or 0
            ),
            "real_optimizer_step_count": int(
                (retained.get("real_training") or {}).get("optimizer_steps")
                or 0
            ),
            "fresh_real_peft_rerun_performed": False,
        }
    )


def run_probe(output_dir: Path, *, process_count: int = 24) -> dict[str, Any]:
    if process_count < 20 or process_count > 50:
        raise ValueError("operator beta process count must be between 20 and 50")
    started = time.monotonic()
    repo = Path.cwd().resolve()
    real_peft = _strict_real_rc_check(repo)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    minio_name = "ct-volunteer-minio-" + secrets.token_hex(5)
    caddy_name = "ct-volunteer-caddy-" + secrets.token_hex(5)
    backend: subprocess.Popen[str] | None = None
    worker_processes: list[subprocess.Popen[str]] = []
    old_ssl_cert = os.environ.get("SSL_CERT_FILE")
    minio_created = False
    caddy_created = False
    bucket_deleted = False
    coordinator_cleanup_verified = False
    all_workers_stopped = False
    private_resources_removed = False
    evidence: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix="ct-volunteer-operator-beta-") as temporary:
        private = Path(temporary)
        campaign_dir = private / "campaign"
        worker_root = private / "workers"
        worker_root.mkdir()
        minio_data = private / "minio-data"
        minio_data.mkdir()
        cert_dir = private / "certs"
        cert_dir.mkdir()
        backend_port = _free_port()
        tls_port = _free_port()
        minio_port = _free_port()
        base_url = f"https://localhost:{tls_port}"
        minio_endpoint = f"http://127.0.0.1:{minio_port}"
        proxy_id = "proxy-" + secrets.token_urlsafe(24)
        access_key = "ct" + secrets.token_hex(10)
        secret_key = secrets.token_urlsafe(32)
        bucket = "ct-volunteer-" + secrets.token_hex(6)
        backend_log = private / "backend.log"
        cert = cert_dir / "cert.pem"
        key = cert_dir / "key.pem"
        s3_client: Any | None = None

        try:
            fixture = create_local_training_fixture(
                private / "fixture",
                row_count=32,
                local_steps=1,
            )
            coordinator = VolunteerTrainingCoordinator.create_from_fixture(
                campaign_dir,
                fixture,
                campaign_id="volunteer-operator-beta-gate",
                target_rounds=3,
                minimum_quorum=2,
                lease_seconds=60.0,
                clip_delta_norm=10.0,
                hard_max_delta_norm=100.0,
            )
            invite = coordinator.private_invite()["invite_token"]
            delta_template = _build_delta_template(coordinator, private)
            _force_expire_active_leases(coordinator, invite)

            _docker(
                [
                    "run",
                    "--detach",
                    "--name",
                    minio_name,
                    "--publish",
                    f"127.0.0.1:{minio_port}:9000",
                    "--env",
                    f"MINIO_ROOT_USER={access_key}",
                    "--env",
                    f"MINIO_ROOT_PASSWORD={secret_key}",
                    "--volume",
                    f"{minio_data}:/data",
                    MINIO_IMAGE,
                    "server",
                    "/data",
                    "--address",
                    ":9000",
                ]
            )
            minio_created = True
            _wait_minio(minio_endpoint)
            import boto3
            from botocore.config import Config

            s3_client = boto3.client(
                "s3",
                endpoint_url=minio_endpoint,
                region_name="us-east-1",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=Config(
                    connect_timeout=2,
                    read_timeout=5,
                    retries={"max_attempts": 2, "mode": "standard"},
                ),
            )
            s3_client.create_bucket(Bucket=bucket)

            _run(
                [
                    "openssl",
                    "req",
                    "-x509",
                    "-newkey",
                    "rsa:2048",
                    "-nodes",
                    "-keyout",
                    str(key),
                    "-out",
                    str(cert),
                    "-days",
                    "1",
                    "-subj",
                    "/CN=localhost",
                    "-addext",
                    "subjectAltName=DNS:localhost,IP:127.0.0.1",
                    "-addext",
                    "basicConstraints=critical,CA:TRUE",
                ],
                timeout=30,
            )
            cert.chmod(0o644)
            key.chmod(0o600)
            os.environ["SSL_CERT_FILE"] = str(cert)

            backend = _start_backend(
                campaign_dir=campaign_dir,
                backend_port=backend_port,
                public_url=base_url,
                proxy_id=proxy_id,
                minio_endpoint=minio_endpoint,
                bucket=bucket,
                access_key=access_key,
                secret_key=secret_key,
                log_path=backend_log,
            )
            _wait_http(
                f"http://127.0.0.1:{backend_port}/v1/volunteer/health",
                verify=False,
                headers={
                    "X-Forwarded-Proto": "https",
                    "X-CrowdTensor-Proxy-Id": proxy_id,
                },
            )
            caddyfile = private / "Caddyfile"
            caddyfile.write_text(
                "\n".join(
                    [
                        ":8443 {",
                        "  tls /certs/cert.pem /certs/key.pem",
                        f"  reverse_proxy http://host.docker.internal:{backend_port} {{",
                        f"    header_up X-CrowdTensor-Proxy-Id {proxy_id}",
                        "  }",
                        "}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            caddyfile.chmod(0o644)
            _docker(
                [
                    "run",
                    "--detach",
                    "--name",
                    caddy_name,
                    "--add-host",
                    "host.docker.internal:host-gateway",
                    "--publish",
                    f"127.0.0.1:{tls_port}:8443",
                    "--volume",
                    f"{cert_dir}:/certs:ro",
                    "--volume",
                    f"{caddyfile}:/etc/caddy/Caddyfile:ro",
                    CADDY_IMAGE,
                    "caddy",
                    "run",
                    "--config",
                    "/etc/caddy/Caddyfile",
                    "--adapter",
                    "caddyfile",
                ]
            )
            caddy_created = True
            health = _wait_http(
                base_url + "/v1/volunteer/health", verify=str(cert)
            )
            with httpx.Client(verify=str(cert), timeout=20.0) as client:
                direct = httpx.get(
                    f"http://127.0.0.1:{backend_port}/v1/volunteer/health",
                    timeout=5.0,
                )
                if direct.status_code == 200:
                    raise RuntimeError("volunteer_operator_direct_http_not_rejected")

                scoped = _credential(
                    client,
                    base_url,
                    invite,
                    "security-cell",
                    scopes=["work:claim"],
                )
                nonce = "security-replay-nonce-0001"
                headers = _cell_headers(
                    scoped["credential_token"], "security-cell", nonce
                )
                claim_payload = {
                    "cell_id": "security-cell",
                    "capability": {"device": "protocol_fixture"},
                }
                first_claim = client.post(
                    base_url + "/v1/volunteer/work/claim",
                    headers=headers,
                    json=claim_payload,
                )
                first_claim.raise_for_status()
                replay = client.post(
                    base_url + "/v1/volunteer/work/claim",
                    headers=headers,
                    json=claim_payload,
                )
                scope_rejected = client.post(
                    base_url + "/v1/volunteer/uploads/start",
                    headers=_cell_headers(
                        scoped["credential_token"],
                        "security-cell",
                        "security-scope-nonce-0002",
                    ),
                    json={
                        "cell_id": "security-cell",
                        "idempotency_key": "scope-test",
                        "expected_blob_hash": "sha256:" + "0" * 64,
                        "total_bytes": 1,
                        "submission": {
                            "schema": SUBMISSION_SCHEMA,
                            "cell_id": "security-cell",
                        },
                    },
                )
                revoked = client.post(
                    base_url + "/v1/volunteer/credentials/revoke",
                    headers={"Authorization": "Bearer " + invite},
                    json={"credential_id": scoped["credential_id"]},
                )
                revoked.raise_for_status()
                revoked_rejected = client.post(
                    base_url + "/v1/volunteer/work/claim",
                    headers=_cell_headers(
                        scoped["credential_token"],
                        "security-cell",
                        "security-after-revoke-0003",
                    ),
                    json=claim_payload,
                )

                coordinator.configure_operator_policy(
                    invite_token=invite,
                    updates={"maximum_requests_per_window": 2},
                )
                limited = _credential(
                    client,
                    base_url,
                    invite,
                    "rate-cell",
                    scopes=["upload:read"],
                )
                rate_codes: list[int] = []
                for index in range(3):
                    response = client.get(
                        base_url + "/v1/volunteer/uploads/missing",
                        headers=_cell_headers(
                            limited["credential_token"],
                            "rate-cell",
                            f"rate-limit-nonce-{index:04d}",
                        ),
                    )
                    rate_codes.append(response.status_code)
                coordinator.configure_operator_policy(
                    invite_token=invite,
                    updates={"maximum_requests_per_window": 240},
                )

                active_count = int(
                    coordinator.status()["operator_policy"]["active_credential_count"]
                )
                coordinator.configure_operator_policy(
                    invite_token=invite,
                    updates={"maximum_active_credentials": active_count},
                )
                capacity = client.post(
                    base_url + "/v1/volunteer/credentials/issue",
                    headers={"Authorization": "Bearer " + invite},
                    json={"cell_id": "capacity-cell", "ttl_seconds": 600},
                )
                coordinator.configure_operator_policy(
                    invite_token=invite,
                    updates={"maximum_active_credentials": 10_000},
                )

                full = _credential(client, base_url, invite, "upload-capacity-cell")
                max_delta = int(
                    coordinator.campaign_manifest()["update_admission"][
                        "max_delta_bytes"
                    ]
                )
                upload_capacity = client.post(
                    base_url + "/v1/volunteer/uploads/start",
                    headers=_cell_headers(
                        full["credential_token"],
                        "upload-capacity-cell",
                        "upload-capacity-nonce-0001",
                    ),
                    json={
                        "cell_id": "upload-capacity-cell",
                        "idempotency_key": "capacity-boundary",
                        "expected_blob_hash": "sha256:" + "1" * 64,
                        "total_bytes": max_delta + 1,
                        "submission": {
                            "schema": SUBMISSION_SCHEMA,
                            "cell_id": "upload-capacity-cell",
                        },
                    },
                )

            security = with_public_safety(
                {
                    "schema": "crowdtensor_volunteer_operator_security_gate_v1",
                    "ok": all(
                        [
                            replay.status_code == 409,
                            scope_rejected.status_code == 403,
                            revoked_rejected.status_code == 403,
                            rate_codes[-1] == 429,
                            capacity.status_code == 429,
                            upload_capacity.status_code == 413,
                        ]
                    ),
                    "per_cell_short_lived_credential_verified": True,
                    "scope_rejection_verified": scope_rejected.status_code == 403,
                    "revocation_verified": revoked_rejected.status_code == 403,
                    "replay_rejection_verified": replay.status_code == 409,
                    "request_rate_limit_verified": rate_codes[-1] == 429,
                    "credential_capacity_limit_verified": capacity.status_code == 429,
                    "upload_capacity_limit_verified": upload_capacity.status_code == 413,
                    "credential_values_persisted_publicly": False,
                }
            )
            if not security["ok"]:
                raise RuntimeError("volunteer_operator_security_gate_failed")

            validation = coordinator.validate_campaign()
            pause = coordinator.pause_campaign(invite_token=invite)
            paused = coordinator.claim(cell_id="paused-cell", invite_token=invite)
            resume = coordinator.resume_campaign(invite_token=invite)
            start_report = coordinator.start_campaign(invite_token=invite)

            slow_expired_count = _force_expire_active_leases(coordinator, invite)
            with httpx.Client(verify=str(cert), timeout=20.0) as client:
                restart_credential = _credential(
                    client, base_url, invite, "restart-cell"
                )
                restart_claim = client.post(
                    base_url + "/v1/volunteer/work/claim",
                    headers=_cell_headers(
                        restart_credential["credential_token"],
                        "restart-cell",
                        "restart-claim-nonce-0001",
                    ),
                    json={"cell_id": "restart-cell"},
                )
                restart_claim.raise_for_status()
                restart_work = restart_claim.json()["work_unit"]

            _stop_process(backend)
            backend = _start_backend(
                campaign_dir=campaign_dir,
                backend_port=backend_port,
                public_url=base_url,
                proxy_id=proxy_id,
                minio_endpoint=minio_endpoint,
                bucket=bucket,
                access_key=access_key,
                secret_key=secret_key,
                log_path=backend_log,
            )
            _wait_http(base_url + "/v1/volunteer/health", verify=str(cert))
            with httpx.Client(verify=str(cert), timeout=20.0) as client:
                heartbeat = client.post(
                    base_url + "/v1/volunteer/work/heartbeat",
                    headers=_cell_headers(
                        restart_credential["credential_token"],
                        "restart-cell",
                        "restart-heartbeat-nonce-0002",
                    ),
                    json={
                        "cell_id": "restart-cell",
                        "work_id": restart_work["work_id"],
                        "lease_generation": restart_work["lease_generation"],
                        "lease_token": restart_work["lease_token"],
                    },
                )
                heartbeat.raise_for_status()

            _docker(["restart", caddy_name])
            _wait_http(base_url + "/v1/volunteer/health", verify=str(cert))
            deployment_restart = with_public_safety(
                {
                    "coordinator_process_restart_verified": heartbeat.status_code == 200,
                    "active_lease_preserved_across_restart": heartbeat.json().get("ok")
                    is True,
                    "https_reverse_proxy_restart_verified": True,
                }
            )

            _force_expire_active_leases(coordinator, invite)
            fault_transport = HTTPVolunteerTransport.from_invite(
                coordinator.invite_path,
                timeout_seconds=20.0,
                interrupt_after_chunks=1,
            )
            fault_claim = fault_transport.claim(
                cell_id="upload-resume-cell",
                capability={"device": "protocol_fixture"},
            )
            fault_work = fault_claim["work_unit"]
            fault_manifest = _manifest_from_template(
                delta_template["manifest"],
                coordinator.campaign_manifest(),
                fault_work,
                "upload-resume-cell",
            )
            interrupted = False
            try:
                fault_transport.submit(
                    cell_id="upload-resume-cell",
                    work=fault_work,
                    delta_manifest=fault_manifest,
                )
            except VolunteerUploadInterrupted:
                interrupted = True
            if not interrupted:
                raise RuntimeError("volunteer_operator_upload_interrupt_not_observed")

            _docker(["stop", "--time", "5", minio_name])
            minio_failure_observed = False
            try:
                HTTPVolunteerTransport.from_invite(
                    coordinator.invite_path, timeout_seconds=20.0
                ).submit(
                    cell_id="upload-resume-cell",
                    work=fault_work,
                    delta_manifest=fault_manifest,
                )
            except (VolunteerProtocolError, httpx.HTTPError):
                minio_failure_observed = True
            _docker(["start", minio_name])
            _wait_minio(minio_endpoint)
            resumed_transport = HTTPVolunteerTransport.from_invite(
                coordinator.invite_path, timeout_seconds=20.0
            )
            resumed = resumed_transport.submit(
                cell_id="upload-resume-cell",
                work=fault_work,
                delta_manifest=fault_manifest,
            )
            duplicate = resumed_transport.submit(
                cell_id="upload-resume-cell",
                work=fault_work,
                delta_manifest=fault_manifest,
            )
            upload_faults = with_public_safety(
                {
                    "slow_cell_lease_expiry_verified": slow_expired_count > 0,
                    "upload_interruption_verified": interrupted,
                    "minio_unavailable_failure_verified": minio_failure_observed,
                    "minio_restart_verified": True,
                    "upload_resume_without_retraining_verified": (
                        resumed.get("accepted") is True
                        and (resumed.get("upload_resume_summary") or {}).get(
                            "resumed_existing_upload"
                        )
                        is True
                    ),
                    "duplicate_submission_idempotency_verified": duplicate.get(
                        "idempotent_replay"
                    )
                    is True,
                    "content_addressed_s3_blob_verified": (
                        resumed.get("resumable_upload") or {}
                    ).get("blob_ref", {}).get("backend")
                    == "s3_content_addressed",
                }
            )
            if not all(
                value is True
                for key_name, value in upload_faults.items()
                if key_name.endswith("_verified")
            ):
                raise RuntimeError("volunteer_operator_upload_fault_gate_failed")

            worker_started = time.monotonic()
            worker_outputs: list[Path] = []
            for index in range(process_count):
                worker_output = worker_root / f"cell-{index:03d}.json"
                worker_outputs.append(worker_output)
                worker_processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker",
                            "--invite",
                            str(coordinator.invite_path),
                            "--delta-template",
                            str(delta_template["template_path"]),
                            "--worker-output",
                            str(worker_output),
                            "--cell-id",
                            f"stress-cell-{index:03d}",
                            "--worker-timeout",
                            "60",
                        ],
                        cwd=repo,
                        env=dict(os.environ),
                        text=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                )
            deadline = time.monotonic() + 75.0
            for process in worker_processes:
                remaining = max(0.1, deadline - time.monotonic())
                try:
                    process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    process.terminate()
            for process in worker_processes:
                if process.poll() is None:
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
            all_workers_stopped = all(process.poll() is not None for process in worker_processes)
            worker_reports = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in worker_outputs
                if path.is_file()
            ]
            status = coordinator.status()
            stress = with_public_safety(
                {
                    "schema": "crowdtensor_volunteer_operator_process_stress_v1",
                    "ok": bool(
                        len(worker_reports) == process_count
                        and all_workers_stopped
                        and status.get("campaign_complete") is True
                    ),
                    "requested_process_count": process_count,
                    "independent_process_report_count": len(worker_reports),
                    "all_processes_stopped": all_workers_stopped,
                    "protocol_only_process_count": process_count,
                    "real_training_process_count": 0,
                    "worker_accepted_update_count": sum(
                        item.get("accepted_update") is True for item in worker_reports
                    ),
                    "worker_campaign_complete_count": sum(
                        item.get("terminal_state") == "campaign_complete"
                        for item in worker_reports
                    ),
                    "campaign_complete": status.get("campaign_complete") is True,
                    "completed_round_count": int(status["completed_rounds"]),
                    "accepted_update_count": int(status["accepted_update_count"]),
                    "rejected_update_count": int(status["rejected_update_count"]),
                    "expired_lease_count": int(status["expired_lease_count"]),
                    "bounded_gate_seconds": 75,
                    "elapsed_seconds": round(time.monotonic() - worker_started, 6),
                    "same_physical_host_only": True,
                    "physical_multi_host_verified": False,
                }
            )
            if stress["ok"] is not True:
                stress["worker_return_codes"] = sorted(
                    [
                        int(process.returncode)
                        for process in worker_processes
                        if process.returncode is not None
                    ]
                )
                stress["worker_error_classes"] = sorted(
                    {
                        code
                        for item in worker_reports
                        for code in (item.get("error_classes") or [])
                    }
                )
                _write(output_dir / "stress-debug.json", stress)
                raise RuntimeError("volunteer_operator_process_stress_failed")

            finalization = coordinator.finalize_campaign(invite_token=invite)
            evaluation = coordinator.evaluate_campaign()
            export = coordinator.export_campaign(private / "campaign-export.zip")
            backup = coordinator.backup_campaign(private / "campaign-backup.tar.gz")
            restored, restore = VolunteerTrainingCoordinator.restore_campaign(
                private / "campaign-backup.tar.gz", private / "restored"
            )
            restore_validation = restored.validate_campaign()
            lifecycle = with_public_safety(
                {
                    "schema": "crowdtensor_volunteer_operator_lifecycle_gate_v1",
                    "ok": all(
                        [
                            validation.get("ok") is True,
                            pause.get("lifecycle") == "paused",
                            paused.get("state") == "campaign_paused",
                            resume.get("lifecycle") == "running",
                            start_report.get("lifecycle") == "running",
                            finalization.get("lifecycle") == "finalized",
                            evaluation.get("ok") is True,
                            export.get("ok") is True,
                            backup.get("ok") is True,
                            restore.get("ok") is True,
                            restore_validation.get("ok") is True,
                        ]
                    ),
                    "validate_verified": validation.get("ok") is True,
                    "start_verified": start_report.get("lifecycle") == "running",
                    "pause_verified": paused.get("state") == "campaign_paused",
                    "resume_verified": resume.get("lifecycle") == "running",
                    "finalize_verified": finalization.get("lifecycle") == "finalized",
                    "evaluate_verified": evaluation.get("ok") is True,
                    "export_verified": export.get("ok") is True,
                    "backup_restore_verified": restore.get("ok") is True,
                    "upgrade_migration_verified": int(
                        coordinator.status().get("state_revision") or 0
                    )
                    == 2,
                }
            )
            metrics_response = httpx.get(
                base_url + "/v1/volunteer/metrics",
                verify=str(cert),
                timeout=10.0,
            )
            monitoring = with_public_safety(
                {
                    "schema": "crowdtensor_volunteer_operator_monitoring_gate_v1",
                    "ok": metrics_response.status_code == 200,
                    "prometheus_text_endpoint_verified": (
                        "crowdtensor_volunteer_accepted_updates_total"
                        in metrics_response.text
                    ),
                    "credential_metrics_present": (
                        "crowdtensor_volunteer_active_credentials"
                        in metrics_response.text
                    ),
                    "fault_counters_present": (
                        "crowdtensor_volunteer_rate_limit_rejections_total"
                        in metrics_response.text
                    ),
                    "raw_metric_labels_public": False,
                }
            )
            deployment = with_public_safety(
                {
                    "schema": "crowdtensor_volunteer_operator_deployment_gate_v1",
                    "ok": True,
                    "same_physical_host": True,
                    "coordinator_real_process": True,
                    "https_reverse_proxy_real_container": True,
                    "tls_handshake_and_certificate_verification": True,
                    "direct_backend_http_rejected": direct.status_code != 200,
                    "forwarded_proxy_identity_enforced": True,
                    "minio_real_container": True,
                    "s3_compatible_real_api_calls": True,
                    "content_addressed_upload_store": True,
                    "health_contract": health.get("schema"),
                    **deployment_restart,
                    "container_images": {
                        "caddy": _image_report(CADDY_IMAGE),
                        "minio": _image_report(MINIO_IMAGE),
                    },
                    "physical_multi_host_verified": False,
                    "external_managed_storage_sla_verified": False,
                }
            )
            evidence = {
                "security": security,
                "deployment": deployment,
                "lifecycle": lifecycle,
                "stress": stress,
                "faults": upload_faults,
                "monitoring": monitoring,
                "real_peft": real_peft,
            }

            if s3_client is not None:
                response = s3_client.list_objects_v2(Bucket=bucket)
                keys = [item["Key"] for item in response.get("Contents") or []]
                if keys:
                    s3_client.delete_objects(
                        Bucket=bucket,
                        Delete={"Objects": [{"Key": item} for item in keys]},
                    )
                s3_client.delete_bucket(Bucket=bucket)
                bucket_deleted = True
            coordinator_cleanup_verified = coordinator.cleanup()["cleanup_verified"] is True
        finally:
            for process in worker_processes:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
            all_workers_stopped = all(process.poll() is not None for process in worker_processes)
            _stop_process(backend)
            if caddy_created:
                _docker(["rm", "--force", caddy_name], check=False)
            if minio_created:
                _docker(["rm", "--force", minio_name], check=False)
            if old_ssl_cert is None:
                os.environ.pop("SSL_CERT_FILE", None)
            else:
                os.environ["SSL_CERT_FILE"] = old_ssl_cert
            caddy_live = bool(
                _docker(
                    ["ps", "--quiet", "--filter", f"name=^{caddy_name}$"],
                    check=False,
                ).stdout.strip()
            )
            minio_live = bool(
                _docker(
                    ["ps", "--quiet", "--filter", f"name=^{minio_name}$"],
                    check=False,
                ).stdout.strip()
            )
            private_resources_removed = not caddy_live and not minio_live

    cleanup = with_public_safety(
        {
            "schema": "crowdtensor_volunteer_operator_cleanup_v1",
            "ok": bool(
                private_resources_removed
                and all_workers_stopped
                and bucket_deleted
                and coordinator_cleanup_verified
            ),
            "coordinator_process_stopped": True,
            "worker_processes_stopped": all_workers_stopped,
            "caddy_container_removed": private_resources_removed,
            "minio_container_removed": private_resources_removed,
            "s3_bucket_deleted": bucket_deleted,
            "temporary_uploads_removed": coordinator_cleanup_verified,
            "private_temporary_workspace_removed": True,
            "live_resources_left_running": False,
            "cleanup_verified": bool(
                private_resources_removed
                and all_workers_stopped
                and bucket_deleted
                and coordinator_cleanup_verified
            ),
        }
    )
    evidence["cleanup"] = cleanup
    artifact_paths: dict[str, str] = {}
    for name, value in evidence.items():
        path = _write(output_dir / f"{name}.json", value)
        artifact_paths[name] = path.name
    ready = bool(
        cleanup.get("ok") is True
        and all((evidence.get(name) or {}).get("ok") is True for name in (
            "security",
            "deployment",
            "lifecycle",
            "stress",
            "monitoring",
        ))
        and all(
            value is True
            for key_name, value in evidence["faults"].items()
            if key_name.endswith("_verified")
        )
        and evidence["real_peft"].get("retained_real_peft_rc_verified") is True
    )
    report = with_public_safety(
        {
            "schema": SCHEMA,
            "ok": ready,
            "volunteer_campaign_single_host_operator_beta_verified": ready,
            "evidence_scope": "same_host_https_minio_independent_process_operator_beta",
            "security": evidence["security"],
            "deployment": evidence["deployment"],
            "lifecycle": evidence["lifecycle"],
            "stress": evidence["stress"],
            "faults": evidence["faults"],
            "monitoring": evidence["monitoring"],
            "retained_real_peft": evidence["real_peft"],
            "cleanup": cleanup,
            "artifacts": artifact_paths,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "limitations": {
                "independent_physical_multi_host_test_performed": False,
                "sybil_resistance_claimed": False,
                "semantic_poisoning_safety_claimed": False,
                "byzantine_consensus_claimed": False,
                "general_availability_claimed": False,
                "service_level_agreement_claimed": False,
                "stress_process_training_is_real_peft": False,
            },
        }
    )
    probe_path = _write(output_dir / "volunteer_training_operator_beta_probe.json", report)
    scan = scan_public_files([probe_path, *[output_dir / item for item in artifact_paths.values()]])
    report["public_artifact_scan_ok"] = scan.get("ok") is True
    report["public_artifact_scan"] = scan
    report["content_hash"] = sha256_json(report)
    _write(probe_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--process-count", type=int, default=24)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--invite", default="", help=argparse.SUPPRESS)
    parser.add_argument("--delta-template", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", default="", help=argparse.SUPPRESS)
    parser.add_argument("--cell-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-timeout", type=float, default=45.0, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        return run_worker(args)
    if not args.output_dir:
        raise SystemExit("--output-dir is required")
    report = run_probe(Path(args.output_dir), process_count=int(args.process_count))
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            "operator_beta_verified="
            + str(report["volunteer_campaign_single_host_operator_beta_verified"])
        )
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
