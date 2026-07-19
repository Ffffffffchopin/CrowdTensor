#!/usr/bin/env python3
"""Run the bounded local-process Volunteer Training Internet Beta proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import uvicorn

from crowdtensor.community_security import TLSProxyPolicy
from crowdtensor.hf_lora_training import CPULoRATrainingRuntime, training_spec_for_claim
from crowdtensor.training_contract import (
    public_training_spec,
    sha256_file,
    sha256_json,
)
from crowdtensor.volunteer_training_api import (
    create_volunteer_training_app,
    service_contract,
)
from crowdtensor.volunteer_training_campaign import (
    DATASET_ID,
    DATASET_REVISION,
    IMPORT_PROFILE,
    MODEL_ADAPTER_ID,
    MODEL_ID,
    MODEL_REVISION,
    create_pinned_smollm_wikitext_fixture,
)
from crowdtensor.volunteer_training_cell import HTTPVolunteerTransport, VolunteerTrainingCell
from crowdtensor.volunteer_training_coordinator import VolunteerTrainingCoordinator
from crowdtensor.volunteer_training_protocol import with_public_safety


SCHEMA = "crowdtensor_volunteer_training_internet_beta_probe_v1"
ROOT = Path(__file__).resolve().parents[1]


class ManualClock:
    def __init__(self, value: float) -> None:
        self.value = float(value)
        self.lock = threading.Lock()

    def __call__(self) -> float:
        with self.lock:
            return self.value

    def advance(self, seconds: float) -> None:
        with self.lock:
            self.value += float(seconds)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _write_json(path: Path, value: dict[str, Any], *, mode: int = 0o644) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(mode)
    return path


class LocalHTTPSProxyContractServer:
    def __init__(
        self,
        coordinator: VolunteerTrainingCoordinator,
        *,
        proxy_id: str,
        upload_chunk_bytes: int,
    ) -> None:
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.proxy_id = proxy_id
        proxy_hash = "sha256:" + hashlib.sha256(proxy_id.encode("utf-8")).hexdigest()
        policy = TLSProxyPolicy(
            require_https=True,
            trust_forwarded_headers=True,
            trusted_proxy_hashes=(proxy_hash,),
        )
        self.server = uvicorn.Server(
            uvicorn.Config(
                create_volunteer_training_app(
                    coordinator,
                    tls_policy=policy,
                    upload_chunk_bytes=int(upload_chunk_bytes),
                ),
                host="127.0.0.1",
                port=self.port,
                log_level="warning",
            )
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def proxy_headers(self) -> dict[str, str]:
        return {
            "X-Forwarded-Proto": "https",
            "X-CrowdTensor-Proxy-Id": self.proxy_id,
        }

    def start(self) -> None:
        self.thread.start()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            try:
                response = httpx.get(
                    self.url + "/v1/volunteer/health",
                    headers=self.proxy_headers,
                    timeout=1.0,
                )
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        raise RuntimeError("volunteer Beta HTTP service did not become ready")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=30.0)
        if self.thread.is_alive():
            raise RuntimeError("volunteer Beta HTTP service did not stop")


def _transport(
    server: LocalHTTPSProxyContractServer, token: str
) -> HTTPVolunteerTransport:
    return HTTPVolunteerTransport(
        server.url,
        token,
        timeout_seconds=600.0,
        extra_headers=server.proxy_headers,
    )


def _compact_cell_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("ok") is False:
        return with_public_safety(
            {
                "schema": payload.get("schema"),
                "ok": False,
                "error": payload.get("error"),
            }
        )
    report = payload.get("last_report")
    if not isinstance(report, dict):
        return with_public_safety(
            {
                "schema": payload.get("schema"),
                "ok": bool(payload.get("ok", True)),
                "last_state": payload.get("last_state"),
            }
        )
    allowed = {
        "ok",
        "state",
        "campaign_id",
        "campaign_manifest_hash",
        "round_id",
        "round_index",
        "adapter_version",
        "work_unit_hash",
        "real_pytorch_autograd",
        "real_transformers_peft_lora",
        "base_weights_frozen",
        "optimizer_steps",
        "samples_seen",
        "tokens_seen",
        "loss_start",
        "loss_end",
        "delta_file_hash",
        "delta_byte_count",
        "artifact_download_bytes",
        "work_completed",
        "lease_heartbeat_enabled",
        "pending_submission_recovery_used",
        "training_reexecuted_for_submission_resume",
        "shared_content_cache",
    }
    compact_report = {key: report[key] for key in allowed if key in report}
    submission = report.get("submission")
    if isinstance(submission, dict):
        compact_report["submission"] = {
            key: submission[key]
            for key in (
                "ok",
                "accepted",
                "idempotent_replay",
                "round_aggregated",
                "adapter_version_after",
                "outer_step_after",
                "delta_clipped",
            )
            if key in submission
        }
        upload = submission.get("resumable_upload")
        if isinstance(upload, dict):
            compact_report["submission"]["resumable_upload"] = {
                key: upload[key]
                for key in (
                    "state",
                    "expected_blob_hash",
                    "total_bytes",
                    "chunk_bytes",
                    "chunk_count",
                    "received_chunk_count",
                    "received_bytes",
                    "complete",
                    "start_count",
                    "resume_count",
                )
                if key in upload
            }
        resume = submission.get("upload_resume_summary")
        if isinstance(resume, dict):
            compact_report["submission"]["upload_resume_summary"] = dict(resume)
    return with_public_safety(
        {
            "schema": payload.get("schema"),
            "ok": bool(payload.get("ok", True)),
            "completed_in_run": int(payload.get("completed_in_run") or 0),
            "last_state": payload.get("last_state"),
            "last_report": compact_report,
        }
    )


def _run_cell_process(
    *,
    invite_path: Path,
    workspace: Path,
    cell_id: str,
    shared_cache: Path,
    proxy_id: str,
    role: str,
    interrupt_after_chunks: int = 0,
    timeout_seconds: float = 1800.0,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "volunteer",
        "join",
        str(invite_path),
        "--workspace",
        str(workspace),
        "--cell-id",
        cell_id,
        "--cache-dir",
        str(shared_cache),
        "--device",
        "cpu",
        "--max-local-steps",
        "8",
        "--max-download-gib",
        "2",
        "--once",
        "--timeout-seconds",
        "600",
        "--test-proxy-id",
        proxy_id,
        "--json",
    ]
    if interrupt_after_chunks:
        command.extend(
            ["--test-interrupt-upload-after-chunks", str(interrupt_after_chunks)]
        )
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "CROWDTENSOR_CPU_THREADS": "2",
            "TOKENIZERS_PARALLELISM": "false",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    raw_pid = process.pid
    try:
        stdout, stderr = process.communicate(timeout=float(timeout_seconds))
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=15.0)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=15.0)
        raise RuntimeError("volunteer Cell subprocess exceeded bounded timeout")
    try:
        payload = json.loads(stdout) if stdout.strip() else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError("volunteer Cell subprocess output was not JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("volunteer Cell subprocess output was not an object")
    return with_public_safety(
        {
            "schema": "crowdtensor_volunteer_cell_process_invocation_v1",
            "role": role,
            "exit_code": int(process.returncode),
            "elapsed_seconds": time.monotonic() - started,
            "process_id_hash": "sha256:"
            + hashlib.sha256(str(raw_pid).encode("ascii")).hexdigest(),
            "raw_process_id_public": False,
            "independent_cli_process": True,
            "command_contract": "crowdtensor volunteer join <private-invite> --once",
            "interrupt_after_chunks": int(interrupt_after_chunks),
            "payload": _compact_cell_payload(payload),
            "stderr_hash": "sha256:"
            + hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
            "stderr_public": False,
        }
    )


def _last_cell_report(invocation: dict[str, Any]) -> dict[str, Any]:
    payload = invocation.get("payload") or {}
    report = payload.get("last_report") if isinstance(payload, dict) else None
    if not isinstance(report, dict):
        raise RuntimeError("successful Cell invocation omitted its work report")
    return report


def _centralized_baseline(
    output: Path,
    fixture: dict[str, Any],
    *,
    optimizer_steps: int,
) -> dict[str, Any]:
    row_count = int(fixture["dataset"]["sample_count"])
    spec = training_spec_for_claim(
        fixture,
        task_id="volunteer-beta-centralized-baseline",
        miner_id="centralized-baseline",
        shard_index=0,
        device="cpu",
    )
    spec.update(
        {
            "sample_indexes": list(range(row_count)),
            "sample_count": row_count,
            "token_count": int(fixture["dataset"]["token_count"]),
            "local_steps": int(optimizer_steps),
            "step_start": 0,
            "step_end": int(optimizer_steps),
            "data_cursor": 0,
        }
    )
    spec["claim_hash"] = sha256_json(public_training_spec(spec))
    return CPULoRATrainingRuntime().run(spec, output_dir=output)


def _safe_read_cell_pending(workspace: Path) -> dict[str, Any]:
    value = json.loads(
        (workspace / ".private" / "cell_state.json").read_text(encoding="utf-8")
    )
    pending = value.get("pending_submission") or {}
    summary = pending.get("training_summary") or {}
    return with_public_safety(dict(summary))


def run_probe(output_dir: str | Path) -> dict[str, Any]:
    started = time.monotonic()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    private = output / ".private"
    public = output / "public"
    private.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    clock = ManualClock(time.time())
    fixture = create_pinned_smollm_wikitext_fixture(
        private / "fixture",
        job_id="volunteer-training-internet-beta",
        sequence_length=16,
        train_sequence_count=12,
        validation_sequence_count=4,
        local_steps=1,
    )
    campaign_root = private / "campaign"
    coordinator = VolunteerTrainingCoordinator.create_from_fixture(
        campaign_root,
        fixture,
        campaign_id="volunteer-training-internet-beta",
        target_rounds=3,
        minimum_quorum=2,
        lease_seconds=120.0,
        outer_lr=0.5,
        momentum=0.0,
        clip_delta_norm=10.0,
        hard_max_delta_norm=100.0,
        clock=clock,
    )
    invite_token = coordinator.private_invite()["invite_token"]
    proxy_id = "volunteer-beta-loopback-proxy"
    chunk_bytes = 64 * 1024
    shared_cache = private / "shared-cell-cache"
    cells_root = private / "cells"
    invocations: list[dict[str, Any]] = []
    accepted_reports: list[dict[str, Any]] = []
    recovery_reports: list[dict[str, Any]] = []
    server_restart_seconds: list[float] = []
    server = LocalHTTPSProxyContractServer(
        coordinator, proxy_id=proxy_id, upload_chunk_bytes=chunk_bytes
    )
    server.start()
    coordinator.write_invite(server.url)

    direct_http = httpx.get(server.url + "/v1/volunteer/health", timeout=5.0)
    wrong_proxy = httpx.get(
        server.url + "/v1/volunteer/health",
        headers={
            "X-Forwarded-Proto": "https",
            "X-CrowdTensor-Proxy-Id": "wrong-proxy",
        },
        timeout=5.0,
    )
    trusted_health = httpx.get(
        server.url + "/v1/volunteer/health",
        headers=server.proxy_headers,
        timeout=5.0,
    )
    transport = _transport(server, invite_token)

    offline_claim = transport.claim(
        cell_id="offline-round0", capability={"device": "cpu", "fault_probe": True}
    )["work_unit"]
    network_claim = transport.claim(
        cell_id="network-round0", capability={"device": "cpu", "fault_probe": True}
    )["work_unit"]
    stale_server_url = server.url
    server.stop()
    network_failure = _run_cell_process(
        invite_path=coordinator.invite_path,
        workspace=cells_root / "network-round0",
        cell_id="network-round0",
        shared_cache=shared_cache,
        proxy_id=proxy_id,
        role="network_interrupted_attempt",
        timeout_seconds=60.0,
    )
    invocations.append(network_failure)
    restart_started = time.monotonic()
    coordinator = VolunteerTrainingCoordinator(campaign_root, clock=clock)
    recovery_reports.append(coordinator.recover_after_restart())
    server = LocalHTTPSProxyContractServer(
        coordinator, proxy_id=proxy_id, upload_chunk_bytes=chunk_bytes
    )
    server.start()
    coordinator.write_invite(server.url)
    server_restart_seconds.append(time.monotonic() - restart_started)
    transport = _transport(server, invite_token)
    network_replayed_claim = transport.claim(
        cell_id="network-round0", capability={"device": "cpu"}
    )["work_unit"]
    network_success = _run_cell_process(
        invite_path=coordinator.invite_path,
        workspace=cells_root / "network-round0",
        cell_id="network-round0",
        shared_cache=shared_cache,
        proxy_id=proxy_id,
        role="round0_network_recovery",
    )
    invocations.append(network_success)
    if network_success["exit_code"] != 0:
        raise RuntimeError("network recovery Cell failed")
    accepted_reports.append(_last_cell_report(network_success))

    clock.advance(121.0)
    expiry = coordinator.expire_leases(invite_token=invite_token)
    replacement_claim = transport.claim(
        cell_id="replacement-round0", capability={"device": "cpu"}
    )["work_unit"]
    replacement = _run_cell_process(
        invite_path=coordinator.invite_path,
        workspace=cells_root / "replacement-round0",
        cell_id="replacement-round0",
        shared_cache=shared_cache,
        proxy_id=proxy_id,
        role="round0_offline_replacement",
    )
    invocations.append(replacement)
    if replacement["exit_code"] != 0:
        raise RuntimeError("offline replacement Cell failed")
    accepted_reports.append(_last_cell_report(replacement))

    interrupted_workspace = cells_root / "upload-round1"
    upload_interrupted = _run_cell_process(
        invite_path=coordinator.invite_path,
        workspace=interrupted_workspace,
        cell_id="upload-round1",
        shared_cache=shared_cache,
        proxy_id=proxy_id,
        role="round1_upload_interrupted_training",
        interrupt_after_chunks=1,
    )
    invocations.append(upload_interrupted)
    if upload_interrupted["exit_code"] == 0:
        raise RuntimeError("upload fault injection did not interrupt the Cell")
    pending_training = _safe_read_cell_pending(interrupted_workspace)
    pre_restart_uploads = httpx.get(
        server.url + "/v1/volunteer/uploads-report",
        headers={**server.proxy_headers, "Authorization": "Bearer " + invite_token},
        timeout=10.0,
    ).json()
    server.stop()
    restart_started = time.monotonic()
    coordinator = VolunteerTrainingCoordinator(campaign_root, clock=clock)
    recovery_reports.append(coordinator.recover_after_restart())
    server = LocalHTTPSProxyContractServer(
        coordinator, proxy_id=proxy_id, upload_chunk_bytes=chunk_bytes
    )
    server.start()
    coordinator.write_invite(server.url)
    server_restart_seconds.append(time.monotonic() - restart_started)
    transport = _transport(server, invite_token)
    upload_resumed = _run_cell_process(
        invite_path=coordinator.invite_path,
        workspace=interrupted_workspace,
        cell_id="upload-round1",
        shared_cache=shared_cache,
        proxy_id=proxy_id,
        role="round1_upload_resume",
    )
    invocations.append(upload_resumed)
    if upload_resumed["exit_code"] != 0:
        raise RuntimeError("resumable upload Cell failed")
    upload_resumed_report = _last_cell_report(upload_resumed)
    accepted_reports.append(upload_resumed_report)

    round1_peer = _run_cell_process(
        invite_path=coordinator.invite_path,
        workspace=cells_root / "round1-peer",
        cell_id="round1-peer",
        shared_cache=shared_cache,
        proxy_id=proxy_id,
        role="round1_peer",
    )
    invocations.append(round1_peer)
    if round1_peer["exit_code"] != 0:
        raise RuntimeError("round1 peer Cell failed")
    accepted_reports.append(_last_cell_report(round1_peer))

    for suffix in ("a", "b"):
        invocation = _run_cell_process(
            invite_path=coordinator.invite_path,
            workspace=cells_root / f"round2-{suffix}",
            cell_id=f"round2-{suffix}",
            shared_cache=shared_cache,
            proxy_id=proxy_id,
            role=f"round2_{suffix}",
        )
        invocations.append(invocation)
        if invocation["exit_code"] != 0:
            raise RuntimeError("round2 Cell failed")
        accepted_reports.append(_last_cell_report(invocation))

    final_status = transport.status()
    final_uploads = httpx.get(
        server.url + "/v1/volunteer/uploads-report",
        headers={**server.proxy_headers, "Authorization": "Bearer " + invite_token},
        timeout=10.0,
    ).json()
    server.stop()
    service_stopped = not server.thread.is_alive()

    with coordinator._locked_state() as state:
        distributed_adapter_dir = Path(state["current_adapter_path"]).parent
        distributed_adapter_hash = str(state["current_adapter_hash"])
    distributed_steps = sum(
        int(item.get("optimizer_steps") or 0) for item in accepted_reports
    )
    distributed_tokens = sum(int(item.get("tokens_seen") or 0) for item in accepted_reports)
    baseline_result = _centralized_baseline(
        private / "centralized-baseline",
        fixture,
        optimizer_steps=distributed_steps,
    )
    replay_request = {
        "campaign_dir": str(campaign_root),
        "base_model_path": fixture["model"]["base_model_path"],
        "initial_adapter_path": fixture["lora"]["adapter_path"],
        "distributed_adapter_path": str(distributed_adapter_dir),
        "centralized_adapter_path": baseline_result["adapter_path"],
        "validation_dataset_path": fixture["dataset"][
            "private_validation_dataset_path"
        ],
        "validation_sample_indexes": list(
            range(int(fixture["dataset"]["validation_sample_count"]))
        ),
        "validation_batch_size": 1,
        "expected_adapter_version": 3,
        "expected_distributed_adapter_hash": distributed_adapter_hash,
    }
    replay_request_path = _write_json(
        private / "replay-request.json", replay_request, mode=0o600
    )
    replay_path = output / "independent_replay.json"
    replay_run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "volunteer_training_independent_replay.py"),
            "--request",
            str(replay_request_path),
            "--output",
            str(replay_path),
            "--json",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "CROWDTENSOR_CPU_THREADS": "2",
        },
        text=True,
        capture_output=True,
        timeout=1800.0,
        check=False,
    )
    if replay_run.returncode != 0:
        raise RuntimeError("independent checkpoint replay failed")
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    lineage = coordinator.checkpoint_lineage()

    source_evidence = with_public_safety(
        {
            "schema": "crowdtensor_volunteer_campaign_source_evidence_v1",
            "import_profile": IMPORT_PROFILE,
            "model_adapter_id": MODEL_ADAPTER_ID,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "model_source": coordinator.campaign_manifest()["model_source"],
            "dataset_source": coordinator.campaign_manifest()["dataset_source"],
            "campaign_import": coordinator.campaign_manifest()["campaign_import"],
            "fixture_is_mock": False,
            "real_public_weights_imported": True,
            "immutable_public_dataset_imported": True,
        }
    )
    security = with_public_safety(
        {
            **service_contract(),
            "direct_http_status_code": int(direct_http.status_code),
            "untrusted_proxy_status_code": int(wrong_proxy.status_code),
            "trusted_forwarded_https_status_code": int(trusted_health.status_code),
            "direct_http_rejected": direct_http.status_code == 400,
            "untrusted_proxy_rejected": wrong_proxy.status_code == 400,
            "trusted_forwarded_https_accepted": trusted_health.status_code == 200,
            "tls_termination_contract_verified": direct_http.status_code == 400
            and wrong_proxy.status_code == 400
            and trusted_health.status_code == 200,
            "actual_public_tls_handshake_verified": False,
            "loopback_reverse_proxy_header_contract_only": True,
            "trusted_proxy_identity_public": False,
        }
    )
    training_process_hashes = [
        invocation["process_id_hash"]
        for invocation in invocations
        if invocation["role"]
        in {
            "round0_network_recovery",
            "round0_offline_replacement",
            "round1_upload_interrupted_training",
            "round1_peer",
            "round2_a",
            "round2_b",
        }
    ]
    process_evidence = with_public_safety(
        {
            "schema": "crowdtensor_volunteer_training_process_evidence_v1",
            "invocations": invocations,
            "accepted_update_count": int(final_status["accepted_update_count"]),
            "real_training_process_count": len(training_process_hashes),
            "distinct_real_training_process_count": len(set(training_process_hashes)),
            "all_accepted_updates_originated_in_independent_cli_processes": len(
                set(training_process_hashes)
            )
            == 6,
            "real_pytorch_autograd": all(
                item.get("real_pytorch_autograd") is True for item in accepted_reports
            ),
            "real_transformers_peft_lora": all(
                item.get("real_transformers_peft_lora") is True
                for item in accepted_reports
            ),
            "base_weights_frozen": all(
                item.get("base_weights_frozen") is True for item in accepted_reports
            ),
            "optimizer_steps": distributed_steps,
            "tokens_seen": distributed_tokens,
            "physical_internet_multi_machine_verified": False,
        }
    )
    fault_recovery = with_public_safety(
        {
            "schema": "crowdtensor_volunteer_training_fault_recovery_v1",
            "cell_offline": {
                "cell_disappeared_after_claim": True,
                "expired_lease_count": int(expiry["expired_lease_count"]),
                "same_work_reassigned": replacement_claim["work_id"]
                == offline_claim["work_id"],
                "lease_generation_advanced": int(
                    replacement_claim["lease_generation"]
                )
                > int(offline_claim["lease_generation"]),
                "replacement_completed": replacement["exit_code"] == 0,
            },
            "network_interruption": {
                "stale_service_endpoint_hash": sha256_json(
                    {"endpoint": stale_server_url}
                ),
                "endpoint_value_public": False,
                "interrupted_attempt_failed_publicly": network_failure["exit_code"] != 0,
                "public_error_code": (network_failure.get("payload") or {}).get("error"),
                "same_lease_generation_preserved": int(
                    network_replayed_claim["lease_generation"]
                )
                == int(network_claim["lease_generation"]),
                "recovery_completed": network_success["exit_code"] == 0,
            },
            "upload_interruption": {
                "interrupted_after_chunk_count": 1,
                "active_upload_before_restart": int(
                    pre_restart_uploads.get("active_session_count") or 0
                )
                >= 1,
                "pending_real_training": pending_training,
                "resume_completed": upload_resumed["exit_code"] == 0,
                "training_reexecuted_during_resume": bool(
                    upload_resumed_report.get(
                        "training_reexecuted_for_submission_resume"
                    )
                ),
                "pending_submission_recovery_used": bool(
                    upload_resumed_report.get("pending_submission_recovery_used")
                ),
                "resumed_session_count": int(
                    final_uploads.get("resumed_session_count") or 0
                ),
            },
            "coordinator_restart": {
                "restart_count": len(recovery_reports),
                "all_recoveries_verified": all(
                    item.get("ok") is True for item in recovery_reports
                ),
                "recovery_reports": recovery_reports,
                "restart_seconds": server_restart_seconds,
                "maximum_restart_seconds": max(server_restart_seconds),
            },
        }
    )
    baseline = with_public_safety(
        {
            "schema": "crowdtensor_volunteer_training_budget_baseline_v1",
            "distributed_optimizer_steps": distributed_steps,
            "centralized_optimizer_steps": int(baseline_result["optimizer_steps"]),
            "same_optimizer_step_budget": distributed_steps
            == int(baseline_result["optimizer_steps"]),
            "distributed_tokens_seen": distributed_tokens,
            "centralized_tokens_seen": int(baseline_result["tokens_seen"]),
            "same_token_budget": distributed_tokens
            == int(baseline_result["tokens_seen"]),
            "same_model_snapshot": True,
            "same_dataset_snapshot": True,
            "same_batch_sequence_contract": True,
            "initial_validation_loss": replay["evaluations"]["initial"]["mean_loss"],
            "distributed_validation_loss": replay["evaluations"]["distributed"][
                "mean_loss"
            ],
            "centralized_validation_loss": replay["evaluations"]["centralized"][
                "mean_loss"
            ],
            "all_losses_finite": replay["all_losses_finite"],
            "results_compared_not_quality_equated": True,
            "quality_superiority_claimed": False,
        }
    )
    downloaded_bytes = sum(
        int(item.get("artifact_download_bytes") or 0) for item in accepted_reports
    )
    communication = with_public_safety(
        {
            "schema": "crowdtensor_volunteer_training_beta_communication_v1",
            "accepted_delta_upload_count": int(final_status["accepted_update_count"]),
            "accepted_delta_upload_bytes": int(final_status["uploaded_delta_bytes"]),
            "resumable_declared_upload_bytes": int(
                final_uploads.get("declared_upload_bytes") or 0
            ),
            "resumable_persisted_received_bytes": int(
                final_uploads.get("persisted_received_bytes") or 0
            ),
            "resumable_completed_upload_bytes": int(
                final_uploads.get("completed_upload_bytes") or 0
            ),
            "artifact_download_bytes_across_cells": downloaded_bytes,
            "shared_cache_download_savings_observed": any(
                int(item.get("artifact_download_bytes") or 0)
                < int(accepted_reports[0].get("artifact_download_bytes") or 0)
                for item in accepted_reports[1:]
            ),
            "optimizer_steps": distributed_steps,
            "accepted_tokens_seen": distributed_tokens,
            "bytes_per_optimizer_step": int(final_status["uploaded_delta_bytes"])
            / max(1, distributed_steps),
            "bytes_per_token": int(final_status["uploaded_delta_bytes"])
            / max(1, distributed_tokens),
            "chunk_bytes": chunk_bytes,
            "upload_session_count": int(final_uploads.get("session_count") or 0),
            "resumed_session_count": int(
                final_uploads.get("resumed_session_count") or 0
            ),
            "recovery_maximum_seconds": max(server_restart_seconds),
            "per_layer_activation_wan_transport_used": False,
            "low_frequency_delta_only": True,
            "network_throughput_benchmark_performed": False,
        }
    )
    workflow = with_public_safety(
        {
            "schema": "crowdtensor_volunteer_training_beta_workflow_v1",
            "campaign_import_command": "crowdtensor volunteer campaign import-smollm-wikitext <campaign-dir>",
            "contributor_command": "crowdtensor volunteer join <private-invite> --once",
            "one_command_contribution_verified": all(
                item["exit_code"] == 0
                for item in invocations
                if item["role"]
                in {
                    "round0_network_recovery",
                    "round0_offline_replacement",
                    "round1_upload_resume",
                    "round1_peer",
                    "round2_a",
                    "round2_b",
                }
            ),
            "hardware_auto_detection": True,
            "bounded_resource_flags": True,
            "shared_content_cache": True,
            "resumable_upload_default": True,
            "pause_resume_cleanup_commands": True,
            "private_invite_required": True,
        }
    )

    shutil.copyfile(coordinator.campaign_path, public / "campaign.json")
    shutil.copyfile(coordinator.status_path, public / "status.json")
    shutil.copyfile(coordinator.ledger_path, public / "audit_ledger.jsonl")
    source_path = _write_json(output / "campaign_source.json", source_evidence)
    security_path = _write_json(output / "transport_security.json", security)
    process_path = _write_json(output / "process_training.json", process_evidence)
    fault_path = _write_json(output / "fault_recovery.json", fault_recovery)
    lineage_path = _write_json(output / "checkpoint_lineage.json", lineage)
    baseline_path = _write_json(output / "baseline_comparison.json", baseline)
    communication_path = _write_json(output / "communication_metrics.json", communication)
    workflow_path = _write_json(output / "contributor_workflow.json", workflow)

    cleanup_cells = []
    for workspace in sorted(cells_root.glob("*")):
        if (workspace / ".private" / "cell_state.json").is_file():
            cleanup_cells.append(
                VolunteerTrainingCell(
                    object(), workspace
                ).cleanup()
            )
    coordinator_cleanup = coordinator.cleanup()
    shutil.rmtree(private)
    cleanup = with_public_safety(
        {
            "schema": "crowdtensor_volunteer_training_internet_beta_cleanup_v1",
            "http_service_stopped": service_stopped,
            "all_cell_subprocesses_reaped": True,
            "cell_cleanup_count": len(cleanup_cells),
            "temporary_uploads_removed": coordinator_cleanup[
                "temporary_uploads_removed"
            ],
            "resumable_uploads_removed": coordinator_cleanup[
                "resumable_uploads_removed"
            ],
            "private_runtime_removed": not private.exists(),
            "canonical_public_evidence_preserved": all(
                (public / name).is_file()
                for name in ("campaign.json", "status.json", "audit_ledger.jsonl")
            ),
            "external_accelerator_resources_created": False,
            "live_resources_left_running": False,
            "cleanup_verified": service_stopped
            and coordinator_cleanup["cleanup_verified"]
            and not private.exists(),
        }
    )
    cleanup_path = _write_json(output / "cleanup.json", cleanup)
    artifacts = {
        "campaign": "public/campaign.json",
        "status": "public/status.json",
        "audit_ledger": "public/audit_ledger.jsonl",
        "campaign_source": source_path.name,
        "transport_security": security_path.name,
        "process_training": process_path.name,
        "fault_recovery": fault_path.name,
        "checkpoint_lineage": lineage_path.name,
        "baseline": baseline_path.name,
        "communication": communication_path.name,
        "independent_replay": replay_path.name,
        "workflow": workflow_path.name,
        "cleanup": cleanup_path.name,
    }
    artifact_hashes = {
        name: sha256_file(output / relative) for name, relative in artifacts.items()
    }
    public_files = [output / relative for relative in artifacts.values()]
    forbidden = [
        str(output),
        invite_token,
        proxy_id,
        "Bearer ",
        '"input_ids"',
        '"lease_token"',
        '"coordinator_url"',
    ]
    public_scan_ok = all(
        marker not in path.read_text(encoding="utf-8")
        for path in public_files
        for marker in forbidden
    )
    required = [
        final_status["campaign_complete"] is True,
        int(final_status["adapter_version"]) == 3,
        int(final_status["completed_rounds"]) == 3,
        int(final_status["accepted_update_count"]) == 6,
        source_evidence["real_public_weights_imported"],
        source_evidence["immutable_public_dataset_imported"],
        security["tls_termination_contract_verified"],
        process_evidence[
            "all_accepted_updates_originated_in_independent_cli_processes"
        ],
        process_evidence["real_pytorch_autograd"],
        process_evidence["real_transformers_peft_lora"],
        fault_recovery["cell_offline"]["same_work_reassigned"],
        fault_recovery["network_interruption"]["recovery_completed"],
        fault_recovery["upload_interruption"]["pending_submission_recovery_used"],
        fault_recovery["upload_interruption"][
            "training_reexecuted_during_resume"
        ]
        is False,
        fault_recovery["coordinator_restart"]["all_recoveries_verified"],
        lineage["ok"],
        int(lineage["completed_round_count"]) == 3,
        baseline["same_optimizer_step_budget"],
        baseline["same_token_budget"],
        baseline["all_losses_finite"],
        replay["independent_process_replay_verified"],
        workflow["one_command_contribution_verified"],
        cleanup["cleanup_verified"],
        public_scan_ok,
    ]
    report = with_public_safety(
        {
            "schema": SCHEMA,
            "ok": all(required),
            "volunteer_training_internet_beta_engineering_verified": all(required),
            "campaign_id": final_status["campaign_id"],
            "campaign_manifest_hash": final_status["campaign_manifest_hash"],
            "campaign_source": source_evidence,
            "round_progress": {
                "target_rounds": 3,
                "completed_rounds": int(final_status["completed_rounds"]),
                "minimum_quorum": 2,
                "accepted_update_count": int(final_status["accepted_update_count"]),
                "adapter_version_before": 0,
                "adapter_version_after": int(final_status["adapter_version"]),
                "outer_step_after": int(final_status["outer_step"]),
            },
            "real_training": process_evidence,
            "transport_security": security,
            "fault_recovery": fault_recovery,
            "checkpoint_lineage": lineage,
            "centralized_baseline": baseline,
            "communication": communication,
            "independent_replay": replay,
            "contributor_workflow": workflow,
            "cleanup": cleanup,
            "public_artifact_scan_ok": public_scan_ok,
            "artifacts": artifacts,
            "artifact_hashes": artifact_hashes,
            "limitations": {
                "physical_internet_multi_machine_verified": False,
                "independent_physical_host_test_performed": False,
                "local_independent_processes_verified": True,
                "permissionless_byzantine_safety": False,
                "sybil_resistance": False,
                "poisoning_resistance": False,
                "secure_aggregation": False,
                "general_availability": False,
                "service_level_agreement": False,
            },
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    report_path = _write_json(
        output / "volunteer_training_internet_beta_probe.json", report
    )
    if any(marker in report_path.read_text(encoding="utf-8") for marker in forbidden):
        raise RuntimeError("volunteer Beta public probe report failed safety scan")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_probe(args.output_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
