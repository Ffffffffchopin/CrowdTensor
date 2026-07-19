#!/usr/bin/env python3
"""Run a bounded real Mistral LoRA gate on one Kaggle T4x2 and one CPU Kernel."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import importlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from crowdtensor.adapter_stage_training import merge_stage_adapters
from crowdtensor.community_live_training import CommunityLiveCoordinator, create_live_app
from crowdtensor.community_security import scan_public_value
from crowdtensor.model_adapter import get_model_adapter, get_model_adapter_registration, stable_hash
from scripts.community_kaggle_reliability_live_probe import (
    _authorized_kaggle_env,
    _start_live_tunnel,
    _stop_server,
)
from scripts.kaggle_gpu_token_weekly_quota_probe import fetch_accelerator_quota
from scripts.mistral_kaggle_live_check import SCHEMA, check_report
from scripts.mistral_kaggle_live_ledger import (
    MAXIMUM_ATTEMPTS,
    SCHEMA as LEDGER_SCHEMA,
    finish_attempt,
    reserve_attempt,
)
from scripts.mistral_kaggle_live_package import (
    KERNEL_REPORT,
    MODEL_ADAPTER_ID,
    MODEL_ID,
    MODEL_REVISION,
    PROGRESS_REPORT,
    build_packages,
)
from scripts.training_cuda_kaggle_common import (
    authenticated_owner,
    delete_succeeded_or_absent,
    extract_kernel_ref,
    push_accepted,
    run_command,
    status_class,
)
from scripts.training_cuda_two_node_probe import ensure_cloudflared, stop_process
from scripts.training_heterogeneous_beta_live_probe import _free_port, _wait_local_ready


PARAMETER_COUNT = 248_024_064
HASH_RE = re.compile(r"sha256:[0-9a-f]{64}")


def _hash(value: str | bytes | Path) -> str:
    if isinstance(value, Path):
        raw = value.read_bytes()
    else:
        raw = value.encode("utf-8") if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _public_blocker(exc: BaseException) -> str:
    message = str(exc).splitlines()[0] if str(exc).splitlines() else ""
    if message.startswith(("mistral_", "community_")):
        return message[:180]
    return "mistral_kaggle_live_failed:" + type(exc).__name__


def _is_hash(value: Any) -> bool:
    return bool(HASH_RE.fullmatch(str(value or "")))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _install_host_wheels(core: Path, plugin: Path, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--target",
            str(destination),
            "--no-deps",
            str(core),
            str(plugin),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
    )
    sys.path.insert(0, str(destination))
    importlib.invalidate_caches()
    adapter = get_model_adapter(MODEL_ADAPTER_ID)
    registration = get_model_adapter_registration(MODEL_ADAPTER_ID)
    if (
        adapter.default_model_id != MODEL_ID
        or adapter.default_revision != MODEL_REVISION
        or registration.get("kind") != "entry_point_plugin"
    ):
        raise RuntimeError("mistral_host_plugin_discovery_invalid")
    return {
        "adapter_id": adapter.adapter_id,
        "registration_kind": registration["kind"],
        "distribution_name": registration["distribution_name"],
        "distribution_version": registration["distribution_version"],
        "core_wheel_hash": _hash(core),
        "adapter_wheel_hash": _hash(plugin),
    }


def _start_server(
    coordinator: CommunityLiveCoordinator,
    *,
    port: int,
    token: str,
    core_wheel: Path,
    adapter_wheel: Path,
) -> tuple[Any, threading.Thread]:
    import uvicorn

    app = create_live_app(
        coordinator,
        miner_token=token,
        wheel_path=core_wheel,
        adapter_wheel_path=adapter_wheel,
    )
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
        )
    )
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_local_ready(f"http://127.0.0.1:{port}", timeout=30)
    return server, thread


def _collect_output(
    ref: str,
    *,
    env: dict[str, str],
    destination: Path,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    attempts = 0
    while time.monotonic() < deadline:
        attempts += 1
        shutil.rmtree(destination, ignore_errors=True)
        destination.mkdir(parents=True, exist_ok=True)
        step = run_command(
            ["kaggle", "kernels", "output", ref, "-p", str(destination)],
            env=env,
            timeout=min(120.0, max(30.0, deadline - time.monotonic())),
        )
        path = destination / KERNEL_REPORT
        if step.get("ok") is True and path.is_file():
            return {
                "found": True,
                "attempt_count": attempts,
                "report": json.loads(path.read_text(encoding="utf-8")),
            }
        time.sleep(min(10.0, attempts * 2.0))
    return {"found": False, "attempt_count": attempts, "report": {}}


def _collect_progress(
    ref: str, *, env: dict[str, str], destination: Path
) -> dict[str, Any]:
    for attempt in range(1, 4):
        shutil.rmtree(destination, ignore_errors=True)
        destination.mkdir(parents=True, exist_ok=True)
        step = run_command(
            ["kaggle", "kernels", "output", ref, "-p", str(destination)],
            env=env,
            timeout=120,
        )
        path = destination / PROGRESS_REPORT
        if step.get("ok") is True and path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            return {
                "found": True,
                "phase": str(value.get("phase") or ""),
                "role": str(value.get("role") or ""),
                "attempt_count": attempt,
            }
        time.sleep(attempt * 2)
    return {"found": False, "phase": "", "role": "", "attempt_count": 3}


def _independent_reload(
    *,
    install_root: Path,
    adapter_dir: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    code = f'''import json
from crowdtensor.adapter_stage_training import independent_reload
from crowdtensor.model_adapter import get_model_adapter
adapter = get_model_adapter({MODEL_ADAPTER_ID!r})
print(json.dumps(independent_reload({str(adapter_dir)!r}, adapter=adapter, device="cpu", cache_dir={str(cache_dir)!r}), sort_keys=True))
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(install_root)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=install_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=1200,
    )
    value = json.loads(completed.stdout.strip().splitlines()[-1])
    if value.get("adapter_reload_verified") is not True:
        raise RuntimeError("mistral_independent_reload_invalid")
    return value


def _export_from_coordinator(
    coordinator: CommunityLiveCoordinator,
    *,
    private: Path,
    output: Path,
    host_install_root: Path,
) -> dict[str, Any]:
    from safetensors.torch import load, save_file

    checkpoints = private / "final-checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    for stage_id, role in enumerate(("stage0", "stage1")):
        checkpoint = coordinator.private_checkpoint(role)
        payload = base64.b64decode(str(checkpoint.get("payload_b64") or ""), validate=True)
        if _hash(payload) != checkpoint.get("payload_hash") or int(
            checkpoint.get("step") or 0
        ) != 8:
            raise RuntimeError("mistral_final_checkpoint_invalid")
        tensors = load(payload)
        adapter_state = {
            key[len("adapter.") :]: value
            for key, value in tensors.items()
            if key.startswith("adapter.")
        }
        if not adapter_state or not any(key.startswith("optimizer.") for key in tensors):
            raise RuntimeError("mistral_final_checkpoint_components_missing")
        save_file(
            adapter_state,
            str(checkpoints / f"stage{stage_id}_adapter.safetensors"),
        )
    adapter = get_model_adapter(MODEL_ADAPTER_ID)
    exported = merge_stage_adapters(
        checkpoints, output / "adapter", adapter=adapter, rank=8, alpha=16
    )
    reloaded = _independent_reload(
        install_root=host_install_root,
        adapter_dir=output / "adapter",
        cache_dir=private / "host-hf-cache",
    )
    return {"export": exported, "reload": reloaded}


def _checkpoint_summary(coordinator: CommunityLiveCoordinator) -> dict[str, Any]:
    steps = {"stage0": [], "stage1": []}
    for event in coordinator.state.get("events") or []:
        if event.get("operation") == "checkpoint_committed" and event.get("role") in steps:
            steps[str(event["role"])].append(int(event["step"]))
    return {
        "steps_by_role": {role: sorted(set(values)) for role, values in steps.items()},
        "adapter_state_saved": True,
        "adam_state_saved": True,
        "hash_integrity_verified": True,
        "final_stage_checkpoints_present": all(
            int((coordinator.private_checkpoint(role) or {}).get("step") or 0) == 8
            for role in ("stage0", "stage1")
        ),
        "checkpoint_tensor_values_public": False,
    }


def _success_report(
    *,
    coordinator: CommunityLiveCoordinator,
    kernel_reports: list[dict[str, Any]],
    package_report: dict[str, Any],
    host_plugin: dict[str, Any],
    exported: dict[str, Any],
    attempt_number: int,
    duration_seconds: float,
) -> dict[str, Any]:
    final = coordinator.public_status()
    gpu = next(item for item in kernel_reports if item.get("kernel_role") == "stage0")
    stage0_workers = list(gpu.get("worker_reports") or [])
    replacement = stage0_workers[1] if len(stage0_workers) > 1 else {}
    internal_ledger = list(coordinator.state.get("ledger") or [])
    plugin_ok = all(
        item.get("core_wheel_hash_verified") is True
        and item.get("adapter_wheel_hash_verified") is True
        and item.get("both_wheels_installed_in_fresh_environment") is True
        and item.get("adapter_plugin_discovered") is True
        and item.get("workspace_import_used") is False
        for item in kernel_reports
    )
    return {
        "schema": SCHEMA,
        "ok": True,
        "live_run_performed": True,
        "node_scope": "Kaggle logical multi-node",
        "accepted_providers": ["kaggle_cpu", "kaggle_cuda"],
        "model": {
            "adapter_id": MODEL_ADAPTER_ID,
            "family": "mistral",
            "architecture": "MistralForCausalLM",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "license": "apache-2.0",
            "parameter_count": PARAMETER_COUNT,
            "real_trained_weights": True,
            "random_or_synthetic_weights_used": False,
            "training_input_kind": "deterministic_private_token_sequence",
            "real_dataset_training_verified": False,
            "meaningful_task_quality_verified": False,
        },
        "plugin_installation": {
            "core_wheel_hash_verified": plugin_ok,
            "adapter_wheel_hash_verified": plugin_ok,
            "both_wheels_installed_in_fresh_environment": plugin_ok,
            "entry_point_plugin_discovered": plugin_ok,
            "workspace_import_not_used": plugin_ok,
            "registration_kind": host_plugin["registration_kind"],
            "distribution_name": host_plugin["distribution_name"],
            "distribution_version": host_plugin["distribution_version"],
            "core_wheel_hash": host_plugin["core_wheel_hash"],
            "adapter_wheel_hash": host_plugin["adapter_wheel_hash"],
        },
        "target_steps": 8,
        "checkpoint_steps": [4, 8],
        "final_status": final,
        "kernel_evidence": kernel_reports,
        "gpu_worker_replacement": {
            "verified": bool(
                gpu.get("worker_replacement_verified") is True
                and len(stage0_workers) == 2
                and replacement.get("checkpoint_restored") is True
            ),
            "after_step": 4,
            "old_worker_id_hash": str(
                stage0_workers[0].get("worker_id_hash") if stage0_workers else ""
            ),
            "new_worker_id_hash": str(replacement.get("worker_id_hash") or ""),
            "checkpoint_restored": replacement.get("checkpoint_restored") is True,
            "restored_checkpoint_step": int(
                replacement.get("restored_checkpoint_step") or 0
            ),
            "optimizer_state_restored": replacement.get("optimizer_state_restored")
            is True,
        },
        "checkpoints": _checkpoint_summary(coordinator),
        "cross_device_transfer": {
            "activation_gradient_transfer_verified": bool(
                len(internal_ledger) == 8
                and all(
                    _is_hash(item.get("activation_hash"))
                    and _is_hash(item.get("gradient_hash"))
                    and int(item.get("activation_bytes") or 0) > 0
                    and int(item.get("gradient_bytes") or 0) > 0
                    for item in internal_ledger
                )
            ),
            "forward_activation_count": len(internal_ledger),
            "backward_gradient_count": len(internal_ledger),
            "forward_payload_bytes": sum(
                int(item.get("activation_bytes") or 0) for item in internal_ledger
            ),
            "backward_payload_bytes": sum(
                int(item.get("gradient_bytes") or 0) for item in internal_ledger
            ),
            "safetensors_serialization": True,
            "all_payload_hashes_verified": True,
            "payload_values_public": False,
        },
        "export": exported["export"],
        "reload": exported["reload"],
        "package": package_report,
        "benchmark": {
            "duration_seconds": round(duration_seconds, 6),
            "steps_per_second": 8.0 / duration_seconds if duration_seconds > 0 else 0.0,
            "gpu_kernel_count": 1,
            "cpu_kernel_count": 1,
            "logical_node_count": 2,
            "resource_scope": "one Kaggle T4x2 Kernel plus one Kaggle CPU Kernel",
        },
        "attempt_ledger": {
            "schema": LEDGER_SCHEMA,
            "attempt": attempt_number,
            "maximum_attempts": MAXIMUM_ATTEMPTS,
            "community_maturity_ledger_modified": False,
        },
        "unsupported_claims": {
            "arbitrary_mistral_models_supported": False,
            "full_parameter_training_verified": False,
            "mistral_7b_live_verified": False,
            "physical_multi_machine_verified": False,
            "production_sla_verified": False,
        },
        "cleanup": {},
        "cleanup_verified": False,
        "credential_values_public": False,
        "credential_paths_public": False,
        "coordinator_url_public": False,
        "raw_training_text_public": False,
        "token_ids_public": False,
        "activation_values_public": False,
        "gradient_values_public": False,
        "checkpoint_tensor_values_public": False,
        "adapter_tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
        "blockers": [],
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    core_wheel = Path(args.core_wheel).expanduser().resolve()
    adapter_wheel = Path(args.adapter_wheel).expanduser().resolve()
    if not core_wheel.is_file() or not adapter_wheel.is_file():
        raise RuntimeError("mistral_live_wheel_missing")
    ledger_path = Path(args.attempt_ledger).expanduser().resolve()
    private = output / ".private-live"
    private.mkdir(parents=True, exist_ok=True)
    private.chmod(0o700)
    host_install_root = private / "host-wheel-site"
    server = None
    server_thread = None
    tunnel = None
    kernel_refs: dict[str, str] = {}
    coordinator: CommunityLiveCoordinator | None = None
    attempt_number = 0
    live_started = 0.0
    cleanup = {
        "all_remote_kernels_deleted": False,
        "coordinator_stopped": False,
        "tunnel_stopped": False,
        "private_runtime_removed": False,
        "live_resources_left_running": True,
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": False,
        "live_run_performed": False,
        "blockers": [],
        "public_artifact_safe": True,
    }
    try:
        host_plugin = _install_host_wheels(
            core_wheel, adapter_wheel, host_install_root
        )
        run_id = "mistral-live-" + secrets.token_hex(12)
        miner_token = secrets.token_urlsafe(36)
        coordinator = CommunityLiveCoordinator(
            private / "coordinator-state.json",
            run_id=run_id,
            target_steps=8,
            sequence_length=8,
            lease_seconds=180,
            checkpoint_steps=(4, 8),
            model_adapter_id=MODEL_ADAPTER_ID,
        )
        with _authorized_kaggle_env(args) as env:
            owner = authenticated_owner(env)
            if not owner:
                raise RuntimeError("mistral_kaggle_authentication_failed")
            quota = fetch_accelerator_quota(env)
            gpu_quota = dict(quota.get("gpu_quota") or {})
            remaining = float(
                gpu_quota.get("effective_remaining_after_reserved_seconds") or 0.0
            )
            if quota.get("ok") is not True or remaining < float(args.timeout_seconds):
                raise RuntimeError("mistral_kaggle_gpu_quota_unavailable")
            port = _free_port()
            server, server_thread = _start_server(
                coordinator,
                port=port,
                token=miner_token,
                core_wheel=core_wheel,
                adapter_wheel=adapter_wheel,
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
            package_report = dict(packages["report"])
            attempt_number = reserve_attempt(ledger_path)

            def push(role: str) -> tuple[str, str]:
                step = run_command(
                    [
                        "kaggle",
                        "kernels",
                        "push",
                        "-p",
                        str(private / "packages" / role),
                    ],
                    env=env,
                    timeout=300,
                )
                metadata = json.loads(
                    (private / "packages" / role / "kernel-metadata.json").read_text(
                        encoding="utf-8"
                    )
                )
                ref = extract_kernel_ref(
                    str(step.get("output_tail") or ""), str(metadata["id"])
                )
                if not push_accepted(step):
                    raise RuntimeError("mistral_kaggle_kernel_push_rejected:" + role)
                return role, ref

            live_started = time.monotonic()
            push_errors: list[BaseException] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(push, role) for role in ("stage0", "stage1")]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        role, ref = future.result()
                        kernel_refs[role] = ref
                    except BaseException as exc:
                        push_errors.append(exc)
            if push_errors:
                raise push_errors[0]
            deadline = live_started + float(args.timeout_seconds)
            while time.monotonic() < deadline:
                status = coordinator.public_status()
                terminal: tuple[str, str] | None = None
                states: dict[str, str] = {}
                for role, ref in sorted(kernel_refs.items()):
                    step = run_command(
                        ["kaggle", "kernels", "status", ref],
                        env=env,
                        timeout=30,
                    )
                    state = status_class(str(step.get("output_tail") or ""))
                    states[role] = state
                    if state == "failed":
                        terminal = (role, ref)
                if status["completed"]:
                    break
                if terminal:
                    role, ref = terminal
                    progress = _collect_progress(
                        ref,
                        env=env,
                        destination=private / "failed-output" / role,
                    )
                    raise RuntimeError(
                        "mistral_kaggle_kernel_terminal_failure:"
                        + role
                        + ":"
                        + str(progress.get("phase") or "unknown")
                    )
                if states and all(value == "complete" for value in states.values()):
                    raise RuntimeError("mistral_kaggle_kernels_completed_before_job")
                time.sleep(5)
            final = coordinator.public_status()
            if not final["completed"]:
                raise RuntimeError("mistral_kaggle_live_gate_timeout")
            kernel_reports: list[dict[str, Any]] = []
            for role in ("stage0", "stage1"):
                collected = _collect_output(
                    kernel_refs[role],
                    env=env,
                    destination=private / "outputs" / role,
                    timeout_seconds=300,
                )
                if collected["found"] is not True:
                    raise RuntimeError("mistral_kaggle_kernel_output_missing:" + role)
                kernel_reports.append(dict(collected["report"]))
            exported = _export_from_coordinator(
                coordinator,
                private=private,
                output=output,
                host_install_root=host_install_root,
            )
            report = _success_report(
                coordinator=coordinator,
                kernel_reports=kernel_reports,
                package_report=package_report,
                host_plugin=host_plugin,
                exported=exported,
                attempt_number=attempt_number,
                duration_seconds=time.monotonic() - live_started,
            )
    except BaseException as exc:
        report = {
            "schema": SCHEMA,
            "ok": False,
            "live_run_performed": bool(kernel_refs),
            "node_scope": "Kaggle logical multi-node",
            "accepted_providers": [],
            "attempt_ledger": {
                "schema": LEDGER_SCHEMA,
                "attempt": attempt_number,
                "maximum_attempts": MAXIMUM_ATTEMPTS,
                "community_maturity_ledger_modified": False,
            },
            "blockers": [_public_blocker(exc)],
            "unsupported_claims": {
                "arbitrary_mistral_models_supported": False,
                "full_parameter_training_verified": False,
                "mistral_7b_live_verified": False,
                "physical_multi_machine_verified": False,
                "production_sla_verified": False,
            },
            "cleanup": cleanup,
            "cleanup_verified": False,
            "credential_values_public": False,
            "credential_paths_public": False,
            "coordinator_url_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
    finally:
        remote_deleted = True
        try:
            with _authorized_kaggle_env(args) as cleanup_env:
                for ref in kernel_refs.values():
                    deleted = run_command(
                        ["kaggle", "kernels", "delete", ref, "-y"],
                        env=cleanup_env,
                        timeout=120,
                    )
                    remote_deleted = remote_deleted and delete_succeeded_or_absent(
                        deleted
                    )
        except BaseException:
            remote_deleted = not kernel_refs
        cleanup["all_remote_kernels_deleted"] = remote_deleted
        cleanup["coordinator_stopped"] = _stop_server(server, server_thread)
        cleanup["tunnel_stopped"] = stop_process(tunnel)
        if str(host_install_root) in sys.path:
            sys.path.remove(str(host_install_root))
        shutil.rmtree(private, ignore_errors=True)
        cleanup["private_runtime_removed"] = not private.exists()
        cleanup["live_resources_left_running"] = not all(
            (
                cleanup["all_remote_kernels_deleted"],
                cleanup["coordinator_stopped"],
                cleanup["tunnel_stopped"],
                cleanup["private_runtime_removed"],
            )
        )
        report["cleanup"] = cleanup
        report["cleanup_verified"] = not cleanup["live_resources_left_running"]
        safety = scan_public_value(
            {key: item for key, item in report.items() if key != "public_safety"}
        )
        report["public_safety"] = safety
        report["public_artifact_safe"] = safety["ok"] is True
        report["ok"] = bool(report.get("ok") and report["cleanup_verified"] and safety["ok"])
        report["content_hash"] = stable_hash(report)
        check = check_report(report)
        _write(output / "mistral_kaggle_heterogeneous_live.json", report)
        _write(output / "mistral_kaggle_heterogeneous_live_check.json", check)
        if attempt_number:
            finish_attempt(ledger_path, achieved=check["ok"] is True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--core-wheel", required=True)
    parser.add_argument("--adapter-wheel", required=True)
    parser.add_argument("--kaggle-token-file", required=True)
    parser.add_argument("--kaggle-username", default="")
    parser.add_argument("--kaggle-account-label", default="")
    parser.add_argument("--attempt-ledger", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=2700.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.timeout_seconds > 2700:
        parser.error("--timeout-seconds must be in (0, 2700]")
    report = run_probe(args)
    print(
        json.dumps(report, sort_keys=True)
        if args.json
        else f"ok={report['ok']} blocker={(report.get('blockers') or [''])[0]}"
    )
    return 0 if check_report(report)["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
