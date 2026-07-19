#!/usr/bin/env python3
"""Run the bounded, public-safe two-Cell Campaign preview.

The preview exercises the ordinary HTTP/CLI path with two independent local
processes.  It is deliberately a same-host demonstration: it does not claim
independent administration, Internet-scale throughput, or model quality.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import uvicorn

from crowdtensor.community_security import scan_public_files
from crowdtensor.hf_lora_training import create_local_training_fixture
from crowdtensor.training_contract import sha256_file, sha256_json
from crowdtensor.volunteer_training_api import create_volunteer_training_app
from crowdtensor.volunteer_training_coordinator import VolunteerTrainingCoordinator
from crowdtensor.volunteer_training_protocol import with_public_safety


SCHEMA = "crowdtensor_volunteer_training_public_demo_v1"
CELL_SCHEMA = "crowdtensor_volunteer_training_public_demo_cell_v1"
MAX_RUNTIME_SECONDS = 300.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o644)


def _wait_for_health(url: str, deadline: float) -> dict[str, Any]:
    last_error = "unavailable"
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url + "/v1/volunteer/health", timeout=1.5)
            if response.status_code == 200:
                value = response.json()
                if isinstance(value, dict) and value.get("ok") is True:
                    return value
                last_error = "invalid_health_response"
            else:
                last_error = f"http_{response.status_code}"
        except Exception as exc:  # bounded readiness polling
            last_error = type(exc).__name__
        time.sleep(0.05)
    raise RuntimeError("volunteer_public_demo_service_timeout:" + last_error)


def _parse_cell_output(raw: str, *, returncode: int, cell_index: int) -> dict[str, Any]:
    """Keep only known public fields from a Cell's JSON response."""

    value: dict[str, Any] = {}
    try:
        candidate = json.loads(raw.strip())
        if isinstance(candidate, dict):
            value = candidate
    except json.JSONDecodeError:
        pass
    for line in reversed(raw.splitlines()):
        if value:
            break
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            value = candidate
            break
    nested = value.get("last_report") if isinstance(value.get("last_report"), dict) else {}
    if nested:
        value = {**value, **nested}
    allowed = (
        "ok",
        "state",
        "work_completed",
        "campaign_id",
        "cell_id_hash",
        "optimizer_steps",
        "samples_seen",
        "tokens_seen",
        "selected_device",
        "base_weights_frozen",
        "real_pytorch_autograd",
        "real_transformers_peft_lora",
    )
    summary = {key: value[key] for key in allowed if key in value}
    summary.update(
        {
            "schema": CELL_SCHEMA,
            "cell_index": int(cell_index),
            "process_returncode": int(returncode),
            "stdout_json_observed": bool(value),
        }
    )
    return with_public_safety(summary)


def _start_server(
    coordinator: VolunteerTrainingCoordinator, port: int
) -> tuple[uvicorn.Server, Any]:
    server = uvicorn.Server(
        uvicorn.Config(
            create_volunteer_training_app(coordinator),
            host="127.0.0.1",
            port=int(port),
            log_level="warning",
            access_log=False,
        )
    )
    import threading

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread


def _run_cells(
    *,
    invite_path: Path,
    root: Path,
    port: int,
    deadline: float,
) -> list[dict[str, Any]]:
    repository = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "PYTHONPATH": str(repository)
        + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
        "CROWDTENSOR_CPU_THREADS": "1",
    }
    processes: list[tuple[int, subprocess.Popen[str]]] = []
    for index in range(2):
        workspace = root / "private" / "cells" / f"cell-{index}"
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
            f"public-demo-cell-{index}",
            "--device",
            "cpu",
            "--max-local-steps",
            "1",
            "--once",
            "--timeout-seconds",
            "120",
            "--json",
        ]
        processes.append(
            (
                index,
                subprocess.Popen(
                    command,
                    cwd=repository,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                ),
            )
        )

    reports: list[dict[str, Any]] = []
    for index, process in processes:
        remaining = max(1.0, deadline - time.monotonic())
        try:
            stdout, _stderr = process.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, _stderr = process.communicate(timeout=10.0)
        reports.append(
            _parse_cell_output(
                stdout,
                returncode=(
                    process.returncode if process.returncode is not None else -1
                ),
                cell_index=index,
            )
        )
    return sorted(reports, key=lambda item: int(item["cell_index"]))


def run_demo(output_dir: str | Path, *, max_runtime_seconds: float = 180.0) -> dict[str, Any]:
    runtime = float(max_runtime_seconds)
    if runtime <= 0 or runtime > MAX_RUNTIME_SECONDS:
        raise ValueError(f"max_runtime_seconds must be in (0, {MAX_RUNTIME_SECONDS:g}]")
    started = time.monotonic()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    private = output / "private"
    public = output / "public"
    private.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    server: uvicorn.Server | None = None
    server_thread: Any = None
    cleanup_verified = False
    cell_reports: list[dict[str, Any]] = []
    try:
        fixture = create_local_training_fixture(
            private / "fixture",
            job_id="crowdtensor-public-preview-fixture",
            row_count=8,
            sequence_length=8,
            local_steps=1,
            learning_rate=0.04,
            batch_size=2,
        )
        coordinator = VolunteerTrainingCoordinator.create_from_fixture(
            private / "campaign",
            fixture,
            campaign_id="crowdtensor-public-preview-campaign",
            target_rounds=1,
            minimum_quorum=2,
            lease_seconds=120.0,
            outer_lr=0.5,
            momentum=0.0,
        )
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        coordinator.write_invite(base_url)
        server, server_thread = _start_server(coordinator, port)
        deadline = started + runtime
        health = _wait_for_health(base_url, min(deadline, time.monotonic() + 30.0))
        dashboard = httpx.get(base_url + "/v1/volunteer/dashboard", timeout=5.0)
        stylesheet = httpx.get(
            base_url + "/v1/volunteer/dashboard/assets/dashboard.css", timeout=5.0
        )
        script = httpx.get(
            base_url + "/v1/volunteer/dashboard/assets/dashboard.js", timeout=5.0
        )
        dashboard_routes = {
            "health": health.get("ok") is True,
            "dashboard": dashboard.status_code == 200 and "roundCanvas" in dashboard.text,
            "stylesheet": stylesheet.status_code == 200 and "metric-grid" in stylesheet.text,
            "script": script.status_code == 200 and "/v1/volunteer/public-snapshot" in script.text,
            "content_security_policy": "default-src 'self'"
            in dashboard.headers.get("content-security-policy", ""),
        }
        cell_reports = _run_cells(
            invite_path=coordinator.invite_path,
            root=output,
            port=port,
            deadline=deadline,
        )
        snapshot = coordinator.public_campaign_snapshot()
        status = coordinator.status()
        coordinator_cleanup = coordinator.cleanup()
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=15.0)
        service_stopped = not bool(server_thread and server_thread.is_alive())
        shutil.rmtree(private, ignore_errors=True)
        private_runtime_removed = not private.exists()
        cleanup_verified = bool(
            coordinator_cleanup.get("cleanup_verified") is True
            and service_stopped
            and private_runtime_removed
        )
        public_snapshot = with_public_safety(snapshot)
        _write_json(public / "public_snapshot.json", public_snapshot)
        public_status = with_public_safety(
            {
                "schema": "crowdtensor_volunteer_training_public_demo_status_v1",
                "campaign_complete": bool(status.get("campaign_complete")),
                "completed_rounds": int(status.get("completed_rounds") or 0),
                "accepted_update_count": int(status.get("accepted_update_count") or 0),
                "adapter_version": int(status.get("adapter_version") or 0),
                "physical_multi_host_verified": False,
            }
        )
        _write_json(public / "status_summary.json", public_status)
        elapsed = time.monotonic() - started
        verified = bool(
            snapshot.get("progress", {}).get("campaign_complete") is True
            and len(cell_reports) == 2
            and all(
                item.get("ok") is True
                and item.get("work_completed") is True
                and item.get("process_returncode") == 0
                for item in cell_reports
            )
            and dashboard_routes["dashboard"]
            and dashboard_routes["content_security_policy"]
            and cleanup_verified
        )
        report = with_public_safety(
            {
                "schema": SCHEMA,
                "ok": verified,
                "volunteer_training_public_demo_verified": verified,
                "demo_scope": "same_host_two_independent_cell_processes",
                "physical_multi_host_verified": False,
                "independent_cell_process_count": len(cell_reports),
                "cell_processes": cell_reports,
                "campaign": {
                    "campaign_id": public_snapshot.get("campaign", {}).get("campaign_id"),
                    "campaign_manifest_hash": public_snapshot.get("campaign", {}).get(
                        "campaign_manifest_hash"
                    ),
                    "model_id": public_snapshot.get("campaign", {}).get("model_id"),
                    "dataset_id": public_snapshot.get("campaign", {}).get("dataset_id"),
                    "fixture_model": True,
                    "real_transformers_peft_lora": all(
                        item.get("real_transformers_peft_lora") is True for item in cell_reports
                    ),
                },
                "dashboard_routes": dashboard_routes,
                "progress": {
                    "completed_rounds": int(
                        public_snapshot.get("progress", {}).get("completed_rounds") or 0
                    ),
                    "accepted_update_count": int(
                        public_snapshot.get("progress", {}).get("accepted_update_count") or 0
                    ),
                    "adapter_version": int(
                        public_snapshot.get("progress", {}).get("adapter_version") or 0
                    ),
                },
                "claims": {
                    "model_quality_improvement_claimed": False,
                    "permissionless_training_claimed": False,
                    "sybil_resistance_claimed": False,
                    "poisoning_resistance_claimed": False,
                    "internet_multi_host_claimed": False,
                },
                "runtime": {
                    "bounded_runtime_seconds": runtime,
                    "elapsed_seconds": round(elapsed, 3),
                    "within_bound": elapsed <= runtime + 5.0,
                },
                "cleanup": {
                    "cleanup_verified": cleanup_verified,
                    "http_service_stopped": service_stopped,
                    "private_runtime_removed": private_runtime_removed,
                    "live_resources_left_running": False,
                },
                "artifacts": {
                    "public_snapshot": "public/public_snapshot.json",
                    "status_summary": "public/status_summary.json",
                },
            }
        )
        report["public_artifact_scan"] = scan_public_files(
            [public / "public_snapshot.json", public / "status_summary.json"]
        )
        report["public_artifact_scan_ok"] = report["public_artifact_scan"]["ok"] is True
        report["artifacts_sha256"] = {
            name: sha256_file(output / relative)
            for name, relative in report["artifacts"].items()
        }
        report["content_hash"] = sha256_json(report)
        _write_json(output / "volunteer_training_public_demo.json", report)
        return report
    except Exception:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10.0)
        shutil.rmtree(private, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-runtime-seconds", type=float, default=180.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = run_demo(
        args.output_dir, max_runtime_seconds=float(args.max_runtime_seconds)
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "volunteer_training_public_demo_verified="
            + str(report["volunteer_training_public_demo_verified"])
            + " cleanup_verified="
            + str(report["cleanup"]["cleanup_verified"])
        )
    return 0 if report.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
