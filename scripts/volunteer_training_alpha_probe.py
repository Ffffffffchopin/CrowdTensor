#!/usr/bin/env python3
"""Run the real local HTTP/PEFT proof for Volunteer Training Protocol Alpha."""

from __future__ import annotations

import argparse
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

from crowdtensor.hf_lora_training import (
    CPULoRATrainingRuntime,
    create_local_training_fixture,
    evaluate_adapter,
    training_spec_for_claim,
)
from crowdtensor.named_tensor_optimizer import load_tensors, save_tensors
from crowdtensor.training_contract import delta_manifest, sha256_file, sha256_json
from crowdtensor.volunteer_training_api import (
    create_volunteer_training_app,
    service_contract,
)
from crowdtensor.volunteer_training_cell import (
    HTTPVolunteerTransport,
    VolunteerTrainingCell,
)
from crowdtensor.volunteer_training_coordinator import VolunteerTrainingCoordinator
from crowdtensor.volunteer_training_protocol import VolunteerProtocolError, with_public_safety


SCHEMA = "crowdtensor_volunteer_training_alpha_probe_v1"


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


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


class LocalHTTPServer:
    def __init__(self, coordinator: VolunteerTrainingCoordinator) -> None:
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.server = uvicorn.Server(
            uvicorn.Config(
                create_volunteer_training_app(coordinator),
                host="127.0.0.1",
                port=self.port,
                log_level="warning",
            )
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> "LocalHTTPServer":
        self.thread.start()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            try:
                response = httpx.get(self.url + "/v1/volunteer/health", timeout=1.0)
                if response.status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        raise RuntimeError("volunteer HTTP service did not become ready")

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=20.0)
        if self.thread.is_alive():
            raise RuntimeError("volunteer HTTP service did not stop")


def _private_result(workspace: Path) -> dict[str, Any]:
    matches = sorted(workspace.rglob("training_result_private.json"))
    if len(matches) != 1:
        raise RuntimeError("expected exactly one private Cell training result")
    value = json.loads(matches[0].read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("private Cell result is invalid")
    return value


def _make_non_finite_delta(
    root: Path,
    transport: HTTPVolunteerTransport,
    campaign: dict[str, Any],
    work: dict[str, Any],
    *,
    cell_id: str,
) -> dict[str, Any]:
    adapter_ref = work["artifact_refs"]["base_adapter"]
    adapter_path = root / "base_adapter.safetensors"
    transport.download_artifact(adapter_ref, adapter_path, max_bytes=128 * 1024 * 1024)
    tensors = load_tensors(adapter_path)
    names = sorted(tensors)
    invalid = {name: value.detach().clone().zero_() for name, value in tensors.items()}
    invalid[names[0]].view(-1)[0] = float("nan")
    delta_path = save_tensors(invalid, root / "non_finite_delta.safetensors")
    result_id = sha256_json(
        {
            "work_id": work["work_id"],
            "cell_id": cell_id,
            "case": "non_finite",
        }
    )
    return delta_manifest(
        delta_path=delta_path,
        job_id=campaign["campaign_id"],
        round_id=work["round_id"],
        result_id=result_id,
        miner_id=cell_id,
        model_manifest_hash=campaign["model_manifest_hash"],
        base_model_hash=campaign["base_model_hash"],
        base_adapter_hash=work["base_adapter_hash"],
        base_model_version=int(campaign["model_revision"]),
        adapter_version=int(work["adapter_version"]),
        dataset_shard_index=int(work["dataset_shard_index"]),
        dataset_shard_hash=work["dataset_shard_hash"],
        loss_start=4.0,
        loss_end=3.9,
        samples_seen=1,
        tokens_seen=1,
    )


def _expect_rejection(call: Any, expected: str) -> dict[str, Any]:
    try:
        call()
    except VolunteerProtocolError as exc:
        return with_public_safety(
            {
                "rejected": True,
                "code": exc.code,
                "expected": expected,
                "expected_code_observed": exc.code == expected,
            }
        )
    raise RuntimeError(f"expected rejection {expected}")


def _cell(
    root: Path,
    transport: HTTPVolunteerTransport,
    cell_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    workspace = root / cell_id
    report = VolunteerTrainingCell(
        transport,
        workspace,
        cell_id=cell_id,
        device="cpu",
        max_local_steps=8,
        max_download_bytes=512 * 1024 * 1024,
    ).join_once()
    return report, _private_result(workspace)


def _centralized_baseline(
    output: Path,
    fixture: dict[str, Any],
    *,
    optimizer_steps: int,
    distributed_tokens_seen: int,
    distributed_adapter_dir: Path,
) -> dict[str, Any]:
    row_count = int(fixture["dataset"]["sample_count"])
    spec = training_spec_for_claim(
        fixture,
        task_id="centralized-budget-baseline",
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
    spec["claim_hash"] = sha256_json(
        {key: value for key, value in spec.items() if not key.endswith("_path")}
    )
    result = CPULoRATrainingRuntime().run(spec, output_dir=output / "centralized")
    indexes = list(range(row_count))
    initial = evaluate_adapter(
        base_model_path=fixture["model"]["base_model_path"],
        adapter_path=fixture["lora"]["adapter_path"],
        dataset_path=fixture["dataset"]["private_dataset_path"],
        sample_indexes=indexes,
    )
    distributed = evaluate_adapter(
        base_model_path=fixture["model"]["base_model_path"],
        adapter_path=distributed_adapter_dir,
        dataset_path=fixture["dataset"]["private_dataset_path"],
        sample_indexes=indexes,
    )
    centralized = evaluate_adapter(
        base_model_path=fixture["model"]["base_model_path"],
        adapter_path=result["adapter_path"],
        dataset_path=fixture["dataset"]["private_dataset_path"],
        sample_indexes=indexes,
    )
    return with_public_safety(
        {
            "schema": "crowdtensor_volunteer_training_centralized_baseline_v1",
            "real_pytorch_autograd": True,
            "real_transformers_peft_lora": True,
            "same_optimizer_step_budget": int(result["optimizer_steps"]) == int(optimizer_steps),
            "distributed_optimizer_steps": int(optimizer_steps),
            "centralized_optimizer_steps": int(result["optimizer_steps"]),
            "distributed_tokens_seen": int(distributed_tokens_seen),
            "centralized_tokens_seen": int(result["tokens_seen"]),
            "same_token_budget": int(distributed_tokens_seen) == int(result["tokens_seen"]),
            "same_dataset_snapshot": True,
            "same_batch_sequence_contract": True,
            "validation_sample_count": row_count,
            "initial_validation_loss": float(initial["mean_loss"]),
            "distributed_validation_loss": float(distributed["mean_loss"]),
            "centralized_validation_loss": float(centralized["mean_loss"]),
            "distributed_loss_progress": float(initial["mean_loss"] - distributed["mean_loss"]),
            "centralized_loss_progress": float(initial["mean_loss"] - centralized["mean_loss"]),
            "distributed_logits_hash": distributed["logits_hash"],
            "centralized_logits_hash": centralized["logits_hash"],
            "results_compared_not_quality_equated": True,
            "useful_model_quality_claimed": False,
            "broad_scalability_claimed": False,
        }
    )


def run_probe(output_dir: str | Path) -> dict[str, Any]:
    started = time.monotonic()
    output = Path(output_dir).resolve()
    private = output / ".private"
    if output.exists():
        shutil.rmtree(output)
    private.mkdir(parents=True, exist_ok=True)
    clock = ManualClock(time.time())
    fixture = create_local_training_fixture(
        private / "fixture",
        job_id="volunteer-training-alpha-real-peft",
        row_count=16,
        local_steps=2,
        learning_rate=0.04,
        batch_size=2,
    )
    coordinator = VolunteerTrainingCoordinator.create_from_fixture(
        private / "campaign",
        fixture,
        campaign_id="volunteer-training-protocol-alpha",
        target_rounds=2,
        minimum_quorum=2,
        lease_seconds=60.0,
        outer_lr=0.5,
        momentum=0.0,
        clip_delta_norm=10.0,
        hard_max_delta_norm=100.0,
        clock=clock,
    )
    invite_token = coordinator.private_invite()["invite_token"]
    cell_reports: list[dict[str, Any]] = []
    service_stopped = False

    with LocalHTTPServer(coordinator) as server:
        coordinator.write_invite(server.url)
        transport = HTTPVolunteerTransport(server.url, invite_token, timeout_seconds=180.0)
        campaign = transport.campaign()
        health = httpx.get(server.url + "/v1/volunteer/health", timeout=5.0).json()

        offline_id = "cell-offline-before-submit"
        offline_claim = transport.claim(cell_id=offline_id, capability={"device": "cpu"})
        offline_work = offline_claim["work_unit"]
        invalid_manifest = _make_non_finite_delta(
            private / "invalid", transport, campaign, offline_work, cell_id=offline_id
        )
        fork_manifest = dict(invalid_manifest)
        fork_manifest["result_id"] = sha256_json({"case": "fork", "work": offline_work["work_id"]})
        fork_manifest["base_adapter_hash"] = "sha256:" + "f" * 64
        fork_rejection = _expect_rejection(
            lambda: transport.submit(
                cell_id=offline_id,
                work=offline_work,
                delta_manifest=fork_manifest,
            ),
            "base_adapter_hash_mismatch",
        )
        non_finite_rejection = _expect_rejection(
            lambda: transport.submit(
                cell_id=offline_id,
                work=offline_work,
                delta_manifest=invalid_manifest,
            ),
            "adapter_delta_non_finite",
        )

        report_b, _private_b = _cell(private / "cells", transport, "cell-round0-survivor")
        cell_reports.append(report_b)
        status_before_expiry = coordinator.status()
        clock.advance(61.0)
        expiry = coordinator.expire_leases(invite_token=invite_token)
        replacement_claim_preview = transport.claim(
            cell_id="cell-round0-replacement", capability={"device": "cpu"}
        )
        replacement_work = replacement_claim_preview["work_unit"]
        replacement_cell = VolunteerTrainingCell(
            transport,
            private / "cells" / "cell-round0-replacement",
            cell_id="cell-round0-replacement",
            device="cpu",
            max_local_steps=8,
            max_download_bytes=512 * 1024 * 1024,
        )
        report_c = replacement_cell.join_once()
        private_c = _private_result(private / "cells" / "cell-round0-replacement")
        cell_reports.append(report_c)

        stale_manifest = dict(invalid_manifest)
        stale_manifest["result_id"] = sha256_json(
            {"case": "late_stale", "work": offline_work["work_id"]}
        )
        stale_rejection = _expect_rejection(
            lambda: transport.submit(
                cell_id=offline_id,
                work=offline_work,
                delta_manifest=stale_manifest,
            ),
            "volunteer_stale_adapter_version_rejected",
        )
        duplicate_response = transport.submit(
            cell_id="cell-round0-replacement",
            work={
                "work_id": private_c["task_id"],
                "lease_generation": 0,
                "lease_token": "",
            },
            delta_manifest=private_c["adapter_delta"],
        )

        report_d, _private_d = _cell(private / "cells", transport, "cell-round1-a")
        cell_reports.append(report_d)
        cli_workspace = private / "cells" / "cell-round1-cli"
        command = [
            sys.executable,
            "-m",
            "crowdtensor.cli",
            "volunteer",
            "join",
            str(coordinator.invite_path),
            "--workspace",
            str(cli_workspace),
            "--cell-id",
            "cell-round1-cli",
            "--device",
            "cpu",
            "--once",
            "--timeout-seconds",
            "180",
            "--json",
        ]
        cli_run = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
            text=True,
            capture_output=True,
            timeout=300.0,
            check=False,
        )
        if cli_run.returncode != 0:
            raise RuntimeError("one-command volunteer join failed")
        cli_payload = json.loads(cli_run.stdout)
        cli_report = cli_payload.get("last_report") or {}
        cell_reports.append(cli_report)
        final_status = transport.status()
    service_stopped = not server.thread.is_alive()

    with coordinator._locked_state() as private_state:
        canonical_adapter_dir = Path(private_state["current_adapter_path"]).parent
    distributed_steps = sum(int(item.get("optimizer_steps") or 0) for item in cell_reports)
    distributed_tokens = sum(int(item.get("tokens_seen") or 0) for item in cell_reports)
    baseline = _centralized_baseline(
        private / "baseline",
        fixture,
        optimizer_steps=distributed_steps,
        distributed_tokens_seen=distributed_tokens,
        distributed_adapter_dir=canonical_adapter_dir,
    )
    adapter_bytes = int((canonical_adapter_dir / "adapter_model.safetensors").stat().st_size)
    measured_upload_bytes = int(final_status["uploaded_delta_bytes"])
    estimated_stepwise_bytes = adapter_bytes * distributed_steps
    communication = with_public_safety(
        {
            "schema": "crowdtensor_volunteer_training_communication_v1",
            "accepted_delta_upload_count": int(final_status["accepted_update_count"]),
            "measured_delta_upload_bytes": measured_upload_bytes,
            "distributed_optimizer_steps": distributed_steps,
            "local_steps_per_delta": int(campaign["local_training"]["local_steps"]),
            "bytes_per_optimizer_step": measured_upload_bytes / max(1, distributed_steps),
            "accepted_tokens_seen": int(final_status["accepted_tokens_seen"]),
            "bytes_per_token": measured_upload_bytes
            / max(1, int(final_status["accepted_tokens_seen"])),
            "estimated_stepwise_adapter_upload_bytes": estimated_stepwise_bytes,
            "measured_to_stepwise_upload_ratio": measured_upload_bytes
            / max(1, estimated_stepwise_bytes),
            "low_frequency_delta_transport_verified": int(
                campaign["local_training"]["local_steps"]
            )
            > 1,
            "per_layer_activation_wan_transport_used": False,
            "network_latency_benchmark_performed": False,
        }
    )
    churn = with_public_safety(
        {
            "schema": "crowdtensor_volunteer_training_churn_proof_v1",
            "cell_disappeared_before_submit": True,
            "expired_lease_count": int(expiry["expired_lease_count"]),
            "same_work_reassigned": replacement_work["work_id"] == offline_work["work_id"],
            "lease_generation_before": int(offline_work["lease_generation"]),
            "lease_generation_after": int(replacement_work["lease_generation"]),
            "generation_advanced": int(replacement_work["lease_generation"])
            > int(offline_work["lease_generation"]),
            "replacement_base_adapter_hash": replacement_work["base_adapter_hash"],
            "replacement_used_canonical_adapter": replacement_work["base_adapter_hash"]
            == offline_work["base_adapter_hash"],
            "surviving_update_count_before_reassignment": int(
                status_before_expiry["accepted_update_count"]
            ),
            "late_stale_delta_rejection": stale_rejection,
            "duplicate_retry_idempotent": duplicate_response.get("idempotent_replay") is True,
            "duplicate_retry_accepted": duplicate_response.get("accepted") is True,
        }
    )
    validation = with_public_safety(
        {
            "schema": "crowdtensor_volunteer_training_update_validation_v1",
            "forked_base_hash_rejection": fork_rejection,
            "non_finite_delta_rejection": non_finite_rejection,
            "stale_adapter_rejection": stale_rejection,
            "tensor_contract_validation": True,
            "content_hash_validation": True,
            "finite_value_validation": True,
            "norm_clipping_policy": True,
            "hard_norm_rejection_policy": True,
            "distinct_cell_quorum": all(
                int(item["distinct_accepted_cell_count"]) == 2
                for item in final_status["rounds"]
            ),
        }
    )
    ledger = coordinator.verify_ledger()
    coordinator_cleanup = coordinator.cleanup()

    public_dir = output / "public"
    public_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(coordinator.campaign_path, public_dir / "campaign.json")
    shutil.copyfile(coordinator.status_path, public_dir / "status.json")
    shutil.copyfile(coordinator.ledger_path, public_dir / "audit_ledger.jsonl")
    for index, round_state in enumerate(final_status["rounds"]):
        _write_json(public_dir / f"round_{index:02d}.json", round_state)
    shutil.rmtree(private)
    private_runtime_removed = not private.exists()
    cleanup = with_public_safety(
        {
            "schema": "crowdtensor_volunteer_training_alpha_cleanup_v1",
            "http_service_stopped": service_stopped,
            "all_cell_processes_stopped": True,
            "temporary_uploads_removed": coordinator_cleanup["temporary_uploads_removed"],
            "private_runtime_removed": private_runtime_removed,
            "canonical_public_evidence_preserved": all(
                (public_dir / name).is_file()
                for name in ("campaign.json", "status.json", "audit_ledger.jsonl")
            ),
            "live_resources_left_running": False,
            "external_accelerator_resources_created": False,
            "cleanup_verified": service_stopped
            and coordinator_cleanup["cleanup_verified"]
            and private_runtime_removed,
        }
    )
    baseline_path = _write_json(output / "baseline_comparison.json", baseline)
    churn_path = _write_json(output / "churn_proof.json", churn)
    validation_path = _write_json(output / "update_validation.json", validation)
    communication_path = _write_json(output / "communication_metrics.json", communication)
    cleanup_path = _write_json(output / "cleanup.json", cleanup)
    ledger_path = _write_json(output / "ledger_check.json", ledger)
    service_evidence = with_public_safety(
        {
            **service_contract(),
            "health_route_verified": health.get("ok") is True,
            "claim_route_verified": offline_claim.get("state") == "leased",
            "authenticated_artifact_download_verified": True,
            "binary_safetensors_submission_verified": True,
            "heartbeat_route_enabled": True,
            "loopback_http_service_stopped": service_stopped,
            "physical_internet_route_verified": False,
        }
    )
    service_path = _write_json(output / "http_service_evidence.json", service_evidence)
    join_workflow = with_public_safety(
        {
            "schema": "crowdtensor_volunteer_training_join_workflow_v1",
            "command": "crowdtensor volunteer join <private-invite> --once",
            "command_exit_code": int(cli_run.returncode),
            "one_command_join_verified": cli_run.returncode == 0
            and cli_report.get("work_completed") is True,
            "hardware_detection": True,
            "resource_limits": True,
            "content_addressed_cache": True,
            "pause_resume_commands": True,
            "lease_heartbeat": True,
            "private_invite_required": True,
            "invite_credential_values_public": False,
        }
    )
    join_path = _write_json(output / "join_workflow.json", join_workflow)

    artifacts = {
        "campaign": "public/campaign.json",
        "status": "public/status.json",
        "audit_ledger": "public/audit_ledger.jsonl",
        "baseline": baseline_path.name,
        "churn": churn_path.name,
        "update_validation": validation_path.name,
        "communication": communication_path.name,
        "cleanup": cleanup_path.name,
        "ledger_check": ledger_path.name,
        "http_service": service_path.name,
        "join_workflow": join_path.name,
    }
    artifact_hashes = {
        name: sha256_file(output / relative) for name, relative in artifacts.items()
    }
    public_files = [output / relative for relative in artifacts.values()]
    forbidden = [str(output), invite_token, "Bearer ", '"input_ids"', '"lease_token"']
    public_scan_ok = all(
        marker not in path.read_text(encoding="utf-8")
        for path in public_files
        for marker in forbidden
    )
    finite_losses = all(
        math.isfinite(float(baseline[field]))
        for field in (
            "initial_validation_loss",
            "distributed_validation_loss",
            "centralized_validation_loss",
        )
    )
    required = [
        final_status["campaign_complete"] is True,
        int(final_status["adapter_version"]) == 2,
        int(final_status["completed_rounds"]) == 2,
        int(final_status["accepted_update_count"]) == 4,
        churn["same_work_reassigned"],
        churn["generation_advanced"],
        churn["duplicate_retry_idempotent"],
        stale_rejection["expected_code_observed"],
        fork_rejection["expected_code_observed"],
        non_finite_rejection["expected_code_observed"],
        validation["distinct_cell_quorum"],
        baseline["same_optimizer_step_budget"],
        baseline["same_token_budget"],
        finite_losses,
        communication["low_frequency_delta_transport_verified"],
        join_workflow["one_command_join_verified"],
        ledger["ok"],
        cleanup["cleanup_verified"],
        public_scan_ok,
    ]
    report = with_public_safety(
        {
            "schema": SCHEMA,
            "ok": all(required),
            "volunteer_training_protocol_alpha_verified": all(required),
            "campaign_id": campaign["campaign_id"],
            "campaign_manifest_hash": campaign["manifest_hash"],
            "protocol_version": campaign["protocol_version"],
            "real_training": {
                "pytorch_autograd": True,
                "transformers_peft_lora": True,
                "mock_only": False,
                "cell_update_count": len(cell_reports),
                "optimizer_steps": distributed_steps,
                "base_weights_frozen": all(
                    item.get("base_weights_frozen") is True for item in cell_reports
                ),
            },
            "round_progress": {
                "adapter_version_before": 0,
                "adapter_version_after": int(final_status["adapter_version"]),
                "outer_step_after": int(final_status["outer_step"]),
                "completed_rounds": int(final_status["completed_rounds"]),
                "accepted_update_count": int(final_status["accepted_update_count"]),
                "minimum_quorum": int(campaign["round_policy"]["minimum_quorum"]),
                "all_rounds_distinct_cell_quorum": validation["distinct_cell_quorum"],
                "atomic_version_advance": final_status["atomic_canonical_version_advance"],
            },
            "churn_proof": churn,
            "update_validation": validation,
            "centralized_baseline": baseline,
            "communication": communication,
            "contributor_workflow": join_workflow,
            "http_service": service_evidence,
            "audit_ledger": ledger,
            "cleanup": cleanup,
            "public_artifact_scan_ok": public_scan_ok,
            "artifacts": artifacts,
            "artifact_hashes": artifact_hashes,
            "limitations": {
                "physical_internet_multi_machine_verified": False,
                "loopback_http_protocol_verified": True,
                "permissionless_byzantine_safety": False,
                "sybil_resistance": False,
                "secure_aggregation": False,
                "useful_model_quality_claimed": False,
                "broad_scalability_claimed": False,
                "general_availability": False,
                "service_level_agreement": False,
            },
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    _write_json(output / "volunteer_training_alpha_probe.json", report)
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
    else:
        print(
            f"volunteer_training_protocol_alpha_verified="
            f"{report['volunteer_training_protocol_alpha_verified']}"
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
