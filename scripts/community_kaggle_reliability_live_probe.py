#!/usr/bin/env python3
"""Run one bounded 100-step Kaggle CPU+GPU Community reliability gate."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import contextlib
import hashlib
import json
import math
import os
import secrets
import shutil
import statistics
import threading
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from crowdtensor.community_live_training import CommunityLiveCoordinator, create_live_app
from crowdtensor.model_adapter import stable_hash
from crowdtensor.community_reliability import validate_short_reliability_gate
from crowdtensor.community_security import scan_public_value
from crowdtensor.smollm_training import _independent_reload, _merge_stage_adapters
from scripts.community_kaggle_live_package import build_packages
from scripts.community_live_gate_ledger_amend import (
    AMENDED_MAXIMUM,
    validate_amended_ledger,
)
from scripts.kaggle_gpu_token_weekly_quota_probe import (
    fetch_accelerator_quota,
    parse_token_sections,
)
from scripts.glm52_kaggle_stage_worker_push_probe import isolated_kaggle_env
from scripts.training_cuda_kaggle_common import (
    authenticated_owner,
    delete_succeeded_or_absent,
    extract_kernel_ref,
    kaggle_env,
    push_accepted,
    run_command,
    status_class,
)
from scripts.training_heterogeneous_beta_live_probe import (
    _free_port,
    _wait_local_ready,
)
from scripts.training_cuda_two_node_probe import ensure_cloudflared, start_tunnel, stop_process


SCHEMA = "crowdtensor_community_kaggle_short_reliability_live_v1"
KERNEL_REPORT = "community_live_kernel.json"
PROGRESS_REPORT = "community_live_progress.json"


def _hash(value: str | bytes) -> str:
    raw = value.encode() if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _public_blocker(exc: BaseException) -> str:
    text = str(exc).splitlines()[0] if str(exc).splitlines() else ""
    if text.startswith("community_"):
        return text[:180]
    return "community_kaggle_live_failed:" + type(exc).__name__


def _write(path: Path, value: Any, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(mode)


@contextlib.contextmanager
def _authorized_kaggle_env(args: argparse.Namespace):
    if str(args.kaggle_account_label or ""):
        sections = parse_token_sections(Path(args.kaggle_token_file).expanduser())
        matches = [item for item in sections if item.get("label") == args.kaggle_account_label]
        if len(matches) != 1:
            raise RuntimeError("community_kaggle_authorized_account_section_invalid")
        with tempfile.TemporaryDirectory(prefix="ct-community-kaggle-config-") as config_dir:
            yield isolated_kaggle_env(matches[0]["env"], config_dir)
        return
    if not str(args.kaggle_username or ""):
        raise RuntimeError("community_kaggle_raw_token_username_required")
    with kaggle_env(args.kaggle_token_file, username_hint=args.kaggle_username) as env:
        yield env


def _start_server(
    coordinator: CommunityLiveCoordinator,
    *,
    port: int,
    token: str,
    wheel_path: Path,
) -> tuple[Any, threading.Thread]:
    import uvicorn

    app = create_live_app(coordinator, miner_token=token, wheel_path=wheel_path)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    )
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_local_ready(f"http://127.0.0.1:{port}", timeout=30)
    return server, thread


def _stop_server(server: Any, thread: threading.Thread | None) -> bool:
    if server is not None:
        server.should_exit = True
    if thread is not None:
        thread.join(timeout=30)
    return thread is None or not thread.is_alive()


def _start_live_tunnel(
    binary: Path,
    *,
    local_url: str,
    private_dir: Path,
    token: str,
    attempts: int = 3,
) -> tuple[Any, str]:
    failures: list[str] = []
    for attempt in range(1, attempts + 1):
        process = None
        try:
            process, url, _log = start_tunnel(
                binary, local_url, private_dir, log_name=f"community-cloudflared-{attempt}.log"
            )
            deadline = time.monotonic() + 180
            successes = 0
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(url + "/health", timeout=15) as response:
                        health = json.loads(response.read().decode())
                    request = urllib.request.Request(
                        url + "/v1/community-live/status",
                        headers={"x-crowdtensor-miner-token": token},
                    )
                    with urllib.request.urlopen(request, timeout=15) as response:
                        status = json.loads(response.read().decode())
                    if health.get("ok") is True and status.get("schema") == "crowdtensor_community_live_status_v1":
                        successes += 1
                        if successes >= 2:
                            return process, url
                    else:
                        successes = 0
                except Exception:
                    successes = 0
                time.sleep(2)
            failures.append("route_readiness_timeout")
        except Exception as exc:
            failures.append(type(exc).__name__)
        stop_process(process)
    raise RuntimeError("community_live_tunnel_unavailable:" + ",".join(failures))


def _reserve_gate(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("community_full_live_gate_amended_ledger_missing")
    ledger = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_amended_ledger(ledger)
    if errors:
        raise RuntimeError("community_full_live_gate_amended_ledger_invalid:" + errors[0])
    attempts = ledger.get("attempts") or []
    if len(attempts) >= AMENDED_MAXIMUM:
        raise RuntimeError("community_full_live_gate_limit_reached")
    if len(attempts) != AMENDED_MAXIMUM - 1:
        raise RuntimeError("community_full_live_gate_history_invalid")
    attempt = {
        "attempt": len(attempts) + 1,
        "started_at": time.time(),
        "completed_at": 0.0,
        "outcome": "running",
    }
    ledger["attempts"].append(attempt)
    _write(path, ledger)
    return {
        "ledger": ledger,
        "attempt": attempt,
        "amendment": dict(ledger["amendments"][0]),
    }


def _finish_gate(path: Path, *, outcome: str) -> dict[str, Any]:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_amended_ledger(ledger, expected_attempt_count=AMENDED_MAXIMUM)
    if errors or str(ledger["attempts"][-1].get("outcome") or "") != "running":
        raise RuntimeError("community_full_live_gate_finish_ledger_invalid")
    ledger["attempts"][-1]["completed_at"] = time.time()
    ledger["attempts"][-1]["outcome"] = str(outcome)
    _write(path, ledger)
    return ledger


def _collect_kernel_output(
    ref: str,
    *,
    env: dict[str, str],
    destination: Path,
    timeout_seconds: float = 300,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    attempts = 0
    while time.monotonic() < deadline:
        attempts += 1
        shutil.rmtree(destination, ignore_errors=True)
        destination.mkdir(parents=True, exist_ok=True)
        result = run_command(
            ["kaggle", "kernels", "output", ref, "-p", str(destination)],
            env=env,
            timeout=min(120.0, max(30.0, deadline - time.monotonic())),
        )
        report_path = destination / KERNEL_REPORT
        if result.get("ok") and report_path.is_file():
            value = json.loads(report_path.read_text(encoding="utf-8"))
            return {"report": value, "attempt_count": attempts, "found": True}
        time.sleep(min(10.0, attempts * 2.0))
    return {"report": {}, "attempt_count": attempts, "found": False}


def _collect_kernel_progress(
    ref: str,
    *,
    env: dict[str, str],
    destination: Path,
) -> dict[str, Any]:
    for attempt in range(1, 4):
        shutil.rmtree(destination, ignore_errors=True)
        destination.mkdir(parents=True, exist_ok=True)
        result = run_command(
            ["kaggle", "kernels", "output", ref, "-p", str(destination)],
            env=env,
            timeout=120,
        )
        path = destination / PROGRESS_REPORT
        if result.get("ok") and path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            return {
                "found": True,
                "phase": str(value.get("phase") or ""),
                "role": str(value.get("role") or ""),
                "attempt_count": attempt,
            }
        time.sleep(attempt * 2)
    return {"found": False, "phase": "", "role": "", "attempt_count": 3}


def _export_from_coordinator(coordinator: CommunityLiveCoordinator, private: Path, output: Path) -> dict[str, Any]:
    from safetensors.torch import load, save_file

    checkpoints = private / "final-checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    for stage_id, role in enumerate(("stage0", "stage1")):
        checkpoint = coordinator.private_checkpoint(role)
        payload = base64.b64decode(str(checkpoint.get("payload_b64") or ""), validate=True)
        if _hash(payload) != checkpoint.get("payload_hash") or int(checkpoint.get("step") or 0) != 100:
            raise RuntimeError("community_live_final_checkpoint_invalid")
        tensors = load(payload)
        adapter = {
            key[len("adapter."):]: value
            for key, value in tensors.items()
            if key.startswith("adapter.")
        }
        if not adapter:
            raise RuntimeError("community_live_final_adapter_empty")
        save_file(adapter, str(checkpoints / f"stage{stage_id}_adapter.safetensors"))
    exported = _merge_stage_adapters(checkpoints, output / "adapter", rank=8, alpha=16)
    reloaded = _independent_reload(output / "adapter", device="cpu", cache_dir=str(private / "hf-cache"))
    return {"export": exported, "reload": reloaded}


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = Path(args.attempt_ledger).expanduser().resolve()
    attempt_number = 0
    private = output / ".private-live"
    private.mkdir(parents=True, exist_ok=True)
    private.chmod(0o700)
    wheel = Path(args.wheel).expanduser().resolve()
    if not wheel.is_file():
        raise RuntimeError("community_live_wheel_missing")
    wheel_hash = _hash(wheel.read_bytes())
    run_id = "community-live-" + secrets.token_hex(12)
    miner_token = secrets.token_urlsafe(36)
    coordinator = CommunityLiveCoordinator(
        private / "coordinator-state.json",
        run_id=run_id,
        target_steps=100,
        sequence_length=8,
        lease_seconds=120,
        checkpoint_steps=(30, 50, 100),
    )
    server = None
    server_thread = None
    tunnel = None
    kernel_refs: list[str] = []
    cleanup = {
        "all_remote_kernels_deleted": False,
        "coordinator_stopped": False,
        "tunnel_stopped": False,
        "private_runtime_removed": False,
        "live_resources_left_running": True,
    }
    outcome = "failed"
    report: dict[str, Any] = {}
    live_started = 0.0
    runtime_diagnostic: dict[str, Any] = {}
    gate_authorization: dict[str, Any] = {}
    try:
        with _authorized_kaggle_env(args) as env:
            owner = authenticated_owner(env)
            if not owner:
                raise RuntimeError("community_kaggle_authentication_failed")
            quota = fetch_accelerator_quota(env)
            gpu = dict(quota.get("gpu_quota") or {})
            if quota.get("ok") is not True or float(gpu.get("effective_remaining_after_reserved_seconds") or 0) < 2700:
                raise RuntimeError("community_kaggle_gpu_quota_unavailable")
            port = _free_port()
            server, server_thread = _start_server(
                coordinator, port=port, token=miner_token, wheel_path=wheel
            )
            cloudflared = ensure_cloudflared(private)
            tunnel, tunnel_url = _start_live_tunnel(
                cloudflared,
                local_url=f"http://127.0.0.1:{port}",
                private_dir=private,
                token=miner_token,
                attempts=3,
            )
            suffix = time.strftime("%m%d-%H%M", time.gmtime()) + "-" + secrets.token_hex(3)
            packages = build_packages(
                private / "packages",
                owner=owner,
                coordinator_url=tunnel_url,
                miner_token=miner_token,
                unique_suffix=suffix,
                timeout_seconds=int(args.timeout_seconds),
            )
            package_report = packages["report"]
            reservation = _reserve_gate(ledger_path)
            attempt_number = int(reservation["attempt"]["attempt"])
            gate_authorization = {
                **reservation["amendment"],
                "verified": True,
            }

            def push(role: str) -> tuple[str, dict[str, Any]]:
                step = run_command(
                    ["kaggle", "kernels", "push", "-p", str(private / "packages" / role)],
                    env=env,
                    timeout=300,
                )
                fallback_meta = json.loads(
                    (private / "packages" / role / "kernel-metadata.json").read_text(encoding="utf-8")
                )
                ref = extract_kernel_ref(str(step.get("output_tail") or ""), str(fallback_meta["id"]))
                if not push_accepted(step):
                    raise RuntimeError("community_kaggle_kernel_push_rejected:" + role)
                return ref, step

            live_started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                pushed = dict(executor.map(lambda role: (role, push(role)), ("stage0", "stage1")))
            kernel_refs = [pushed[role][0] for role in ("stage0", "stage1")]
            deadline = live_started + float(args.timeout_seconds)
            restart: dict[str, Any] = {}
            observations: list[dict[str, Any]] = []
            while time.monotonic() < deadline:
                status = coordinator.public_status()
                states: dict[str, str] = {}
                failed_ref = ""
                failed_role = ""
                for role, ref in zip(("stage0", "stage1"), kernel_refs):
                    step = run_command(["kaggle", "kernels", "status", ref], env=env, timeout=30)
                    states[_hash(ref)] = status_class(str(step.get("output_tail") or ""))
                    if states[_hash(ref)] == "failed":
                        failed_ref = ref
                        failed_role = role
                observations.append(
                    {
                        "observed_at_seconds": round(time.monotonic() - live_started, 3),
                        "committed_step": int(status["committed_step"]),
                        "phase": status["phase"],
                        "coordinator_generation": int(status["coordinator_generation"]),
                        "kernel_states": states,
                    }
                )
                if not restart and int(status["committed_step"]) >= 50:
                    barrier = coordinator.request_restart_barrier(after_step=50)
                    if barrier["ready"]:
                        restart_started = time.monotonic()
                        stopped = _stop_server(server, server_thread)
                        before = coordinator.public_status()
                        coordinator = CommunityLiveCoordinator(private / "coordinator-state.json", run_id=run_id)
                        restart_record = coordinator.record_restart()
                        server, server_thread = _start_server(
                            coordinator, port=port, token=miner_token, wheel_path=wheel
                        )
                        after = coordinator.public_status()
                        restart = {
                            **restart_record,
                            "restart_barrier_verified": bool(
                                barrier["ready"]
                                and barrier["phase"] == "stage0_forward"
                            ),
                            "server_stopped": stopped,
                            "server_restarted": True,
                            "restart_at_committed_step": int(before["committed_step"]),
                            "same_committed_step_after_restart": int(after["committed_step"])
                            == int(before["committed_step"]),
                            "downtime_seconds": round(time.monotonic() - restart_started, 6),
                            "verified": bool(
                                stopped
                                and barrier["ready"]
                                and barrier["phase"] == "stage0_forward"
                                and restart_record["generation_after"]
                                == restart_record["generation_before"] + 1
                                and int(after["committed_step"])
                                == int(before["committed_step"])
                            ),
                        }
                if status["completed"]:
                    break
                if failed_ref:
                    runtime_diagnostic = _collect_kernel_progress(
                        failed_ref,
                        env=env,
                        destination=private / "failed-output" / failed_role,
                    )
                    phase = str(runtime_diagnostic.get("phase") or "unknown")
                    raise RuntimeError(
                        "community_kaggle_kernel_terminal_failure:"
                        + failed_role
                        + ":"
                        + phase
                    )
                if all(value == "complete" for value in states.values()) and not status["completed"]:
                    raise RuntimeError("community_kaggle_kernels_completed_before_job")
                time.sleep(5)
            final = coordinator.public_status()
            if not final["completed"]:
                raise RuntimeError("community_kaggle_live_gate_timeout")
            kernel_reports: list[dict[str, Any]] = []
            collection: list[dict[str, Any]] = []
            for index, (role, ref) in enumerate(zip(("stage0", "stage1"), kernel_refs)):
                collected = _collect_kernel_output(
                    ref, env=env, destination=private / "outputs" / role, timeout_seconds=300
                )
                collection.append({"role": role, "found": collected["found"], "attempt_count": collected["attempt_count"]})
                if not collected["found"]:
                    raise RuntimeError("community_kaggle_kernel_output_missing:" + role)
                kernel_reports.append(dict(collected["report"]))
            exported = _export_from_coordinator(coordinator, private, output)
            stage0_kernel = next(item for item in kernel_reports if item.get("kernel_role") == "stage0")
            stage1_kernel = next(item for item in kernel_reports if item.get("kernel_role") == "stage1")
            stage0_workers = list(stage0_kernel.get("worker_reports") or [])
            stage1_workers = list(stage1_kernel.get("worker_reports") or [])
            second_model_live = dict(stage0_kernel.get("second_model_live") or {})
            if (
                stage0_kernel.get("second_model_dual_gpu_verified") is not True
                or second_model_live.get("ok") is not True
            ):
                raise RuntimeError("community_smollm_dual_gpu_evidence_missing")
            _write(output / "community_smollm_two_stage_lora_live.json", second_model_live)
            replacement = bool(
                stage0_kernel.get("worker_replacement_verified") is True
                and len(stage0_workers) == 2
                and stage0_workers[1].get("checkpoint_restored") is True
                and stage0_workers[1].get("optimizer_state_restored") is True
            )
            worker_ids = {
                item.get("worker_id_hash")
                for kernel in kernel_reports for item in kernel.get("worker_reports") or []
            }
            clean_install = all(
                kernel.get("wheel_installed_in_fresh_environment") is True
                and kernel.get("installed_package_under_install_root") is True
                and kernel.get("model_stack_import_verified") is True
                and kernel.get("workspace_import_used") is False
                and all(worker.get("install_source") == "wheel" for worker in kernel.get("worker_reports") or [])
                for kernel in kernel_reports
            )
            internal_ledger = list(coordinator.state["ledger"])
            checkpoint_events = [
                item
                for item in coordinator.state["events"]
                if item.get("operation") == "checkpoint_committed"
            ]
            committed_times = [float(item["committed_at"]) for item in internal_ledger]
            intervals = [right - left for left, right in zip(committed_times, committed_times[1:])]
            duration = float(final["duration_seconds"])
            report = {
                "schema": SCHEMA,
                "ok": True,
                "live_run_performed": True,
                "live_gate_attempt": attempt_number,
                "maximum_full_live_gates": AMENDED_MAXIMUM,
                "live_gate_amendment": gate_authorization,
                "maximum_gate_seconds": 2700,
                "node_scope": "Kaggle logical multi-node",
                "physical_multi_machine_verified": False,
                "model_id": final["model_id"],
                "model_revision": final["model_revision"],
                "real_open_model_weights": True,
                "random_or_synthetic_weights_used": False,
                "providers": ["kaggle_cpu", "kaggle_cuda"],
                "logical_kernel_count": 2,
                "logical_miner_count": len(worker_ids),
                "clean_install": {
                    "verified": clean_install,
                    "wheel_hash": wheel_hash,
                    "fresh_install_root_per_kernel": True,
                    "fresh_install_kind": "pip_target",
                    "workspace_import_used": False,
                },
                "committed_step_ids": final["committed_step_ids"],
                "duration_seconds": duration,
                "worker_replacement_verified": replacement,
                "coordinator_restart_verified": restart.get("verified") is True,
                "checkpoint_recovery_verified": replacement and set(final["checkpoint_summary"]) == {"stage0", "stage1"},
                "ledger_exactly_once_verified": final["strictly_contiguous_steps"] is True and final["ledger_entry_count"] == 100,
                "finite_update_verified": final["finite_losses"] is True and all(
                    worker.get("adapter_updated") is True
                    for kernel in kernel_reports for worker in kernel.get("worker_reports") or []
                ),
                "adapter_reload_verified": exported["reload"]["adapter_reload_verified"] is True,
                "monitoring_verified": len(observations) >= 2 and final["coordinator_generation"] >= 2,
                "cleanup_verified": False,
                "coordinator_restart": restart,
                "worker_replacement": {
                    "verified": replacement,
                    "replacement_after_step": 30,
                    "old_worker_id_hash": stage0_workers[0].get("worker_id_hash") if stage0_workers else "",
                    "replacement_worker_id_hash": stage0_workers[1].get("worker_id_hash") if len(stage0_workers) > 1 else "",
                    "restored_checkpoint_step": int(stage0_workers[1].get("restored_checkpoint_step") or 0) if len(stage0_workers) > 1 else 0,
                    "optimizer_state_restored": stage0_workers[1].get("optimizer_state_restored") is True if len(stage0_workers) > 1 else False,
                },
                "kernel_evidence": [
                    {
                        "kernel_role": item.get("kernel_role"),
                        "backend": item.get("backend"),
                        "ok": item.get("ok") is True,
                        "worker_process_count": int(item.get("worker_process_count") or 0),
                        "wheel_clean_install": item.get("wheel_installed_in_fresh_environment") is True,
                        "model_stack_import_verified": item.get("model_stack_import_verified") is True,
                        "report_hash": stable_hash(item),
                    }
                    for item in kernel_reports
                ],
                "second_model_live": {
                    "verified": True,
                    "report_hash": stable_hash(second_model_live),
                    "logical_stage_count": int(second_model_live.get("logical_stage_count") or 0),
                    "devices": list(second_model_live.get("devices") or []),
                    "adapter_reload_verified": bool(
                        (second_model_live.get("reload") or {}).get("adapter_reload_verified")
                    ),
                },
                "package": package_report,
                "output_collection": collection,
                "export": exported["export"],
                "reload": exported["reload"],
                "benchmark": {
                    "steps_per_second": 100.0 / duration if duration > 0 else 0.0,
                    "p50_step_seconds": statistics.median(intervals) if intervals else 0.0,
                    "p95_step_seconds": sorted(intervals)[max(0, math.ceil(len(intervals) * 0.95) - 1)] if intervals else 0.0,
                    "coordinator_restart_downtime_seconds": float(restart.get("downtime_seconds") or 0.0),
                    "checkpoint_count": len(final["checkpoint_summary"]),
                    "checkpoint_write_count": len(checkpoint_events),
                    "checkpoint_bytes": sum(
                        int(item.get("payload_bytes") or 0)
                        for item in checkpoint_events
                    ),
                    "forward_payload_count": len(internal_ledger),
                    "forward_payload_bytes": sum(
                        int(item.get("activation_bytes") or 0)
                        for item in internal_ledger
                    ),
                    "backward_payload_count": len(internal_ledger),
                    "backward_payload_bytes": sum(
                        int(item.get("gradient_bytes") or 0)
                        for item in internal_ledger
                    ),
                    "transfer_payloads_private": True,
                    "resource_scope": "one Kaggle GPU Kernel plus one Kaggle CPU Kernel",
                },
                "tpu": {
                    "required": False,
                    "used": False,
                    "acquisition_windows_used": 0,
                    "maximum_acquisition_windows": 2,
                    "maximum_window_seconds": 3600,
                },
                "observations": observations,
                "cleanup": cleanup,
                "credential_values_public": False,
                "credential_paths_public": False,
                "coordinator_url_public": False,
                "raw_training_text_public": False,
                "token_ids_public": False,
                "activation_values_public": False,
                "gradient_values_public": False,
                "checkpoint_tensor_values_public": False,
                "private_paths_public": False,
                "public_artifact_safe": True,
            }
            outcome = "completed"
    except BaseException as exc:
        report = {
            "schema": SCHEMA,
            "ok": False,
            "live_run_performed": bool(kernel_refs),
            "live_gate_attempt": attempt_number,
            "maximum_full_live_gates": AMENDED_MAXIMUM,
            "live_gate_amendment": gate_authorization,
            "node_scope": "Kaggle logical multi-node",
            "blockers": [_public_blocker(exc)],
            "runtime_diagnostic": runtime_diagnostic,
            "cleanup": cleanup,
            "credential_values_public": False,
            "credential_paths_public": False,
            "coordinator_url_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        outcome = "failed:" + _public_blocker(exc)
    finally:
        remote_deleted = True
        try:
            with _authorized_kaggle_env(args) as cleanup_env:
                for ref in kernel_refs:
                    deleted = run_command(
                        ["kaggle", "kernels", "delete", ref, "-y"],
                        env=cleanup_env,
                        timeout=120,
                    )
                    remote_deleted = remote_deleted and delete_succeeded_or_absent(deleted)
        except BaseException:
            remote_deleted = False
        cleanup["all_remote_kernels_deleted"] = remote_deleted
        cleanup["coordinator_stopped"] = _stop_server(server, server_thread)
        cleanup["tunnel_stopped"] = stop_process(tunnel)
        shutil.rmtree(private, ignore_errors=True)
        cleanup["private_runtime_removed"] = not private.exists()
        cleanup["live_resources_left_running"] = not all(
            [cleanup["all_remote_kernels_deleted"], cleanup["coordinator_stopped"], cleanup["tunnel_stopped"], cleanup["private_runtime_removed"]]
        )
        cleanup_ok = not cleanup["live_resources_left_running"]
        report["cleanup"] = cleanup
        report["cleanup_verified"] = cleanup_ok
        if report.get("ok") is True:
            gate = validate_short_reliability_gate(report)
            report["acceptance"] = gate
            report["ok"] = bool(
                gate["ok"]
                and report.get("clean_install", {}).get("verified") is True
                and cleanup_ok
            )
        safety = scan_public_value(report)
        report["public_safety"] = safety
        report["public_artifact_safe"] = safety["ok"] is True
        report["ok"] = bool(report.get("ok") and safety["ok"])
        report["content_hash"] = stable_hash(report)
        _write(output / "community_kaggle_short_reliability_live.json", report)
        if attempt_number:
            _finish_gate(ledger_path, outcome="achieved" if report.get("ok") else outcome)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--kaggle-token-file", required=True)
    parser.add_argument("--kaggle-username", default="")
    parser.add_argument("--kaggle-account-label", default="")
    parser.add_argument("--attempt-ledger", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=2700)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.timeout_seconds > 2700:
        parser.error("--timeout-seconds must be in (0, 2700]")
    report = run_probe(args)
    print(json.dumps(report, sort_keys=True) if args.json else f"ok={report['ok']} steps={len(report.get('committed_step_ids') or [])}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
