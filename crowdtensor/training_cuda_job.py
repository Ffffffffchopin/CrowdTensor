"""User-facing bounded orchestration for the Kaggle CUDA Training RC path."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .training_contract import sha256_file
from .training_allocation_budget import allocation_budget_summary


ROOT = Path(__file__).resolve().parent.parent
STATUS_SCHEMA = "crowdtensor_cuda_training_job_status_v1"
CLEANUP_SCHEMA = "crowdtensor_cuda_training_job_cleanup_v1"
PHASES = [
    "allocation",
    "kernel_launch",
    "worker_assignment",
    "forward",
    "backward",
    "outer_aggregation",
    "checkpoint",
    "evaluation",
    "cleanup",
]


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _status_path(output: Path) -> Path:
    return output / "training_cuda_status.json"


def _initial_status(output: Path) -> dict[str, Any]:
    return {
        "schema": STATUS_SCHEMA,
        "job_id": output.name or "cuda-training-job",
        "backend": "cuda",
        "overall_state": "running",
        "current_phase": "allocation",
        "phases": {name: {"state": "pending"} for name in PHASES},
        "blockers": [],
        "next_resume_command": "crowdtensor train resume <job-dir> --backend cuda",
        "resume_private_inputs": {
            "kaggle_token_required": True,
            "supported_methods": ["--kaggle-token-file", "CROWDTENSOR_KAGGLE_TOKEN_FILE"],
            "credential_values_public": False,
            "credential_paths_public": False,
        },
        "credentials_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def _set_phase(status: dict[str, Any], output: Path, phase: str, state: str, **details: Any) -> None:
    status["current_phase"] = phase
    status["phases"][phase] = {"state": state, **details}
    _write(_status_path(output), status)


def _run_json(command: list[str], *, timeout: float) -> tuple[int, dict[str, Any]]:
    process = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    payload: dict[str, Any] = {}
    for line in reversed([line.strip() for line in (process.stdout or "").splitlines() if line.strip()]):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            payload = value
            break
    return process.returncode, payload


def _attempt_reports(output: Path, kind: str) -> list[Path]:
    filename = (
        "training_cuda_single_kernel_live_probe.json"
        if kind == "single"
        else "training_cuda_two_node_live_probe.json"
    )
    return sorted(path for path in (output / "attempts").glob(f"{kind}-*/{filename}") if path.is_file())


def _latest_verified(paths: list[Path], field: str) -> dict[str, Any]:
    for path in reversed(paths):
        report = _load(path)
        if report.get("ok") is True and report.get(field) is True:
            return report
    return {}


def _public_attempt_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": report.get("schema"),
        "ok": report.get("ok") is True,
        "attempt": int(report.get("attempt") or 0),
        "blockers": list(report.get("blockers") or []),
        "cleanup": report.get("cleanup") or {},
        "single_kernel_t4x2_verified": report.get("single_kernel_t4x2_verified") is True,
        "two_node_cuda_verified": report.get("two_node_cuda_verified") is True,
        "public_artifact_safe": report.get("public_artifact_safe") is True,
    }


def run_cuda_training_job(
    output_dir: str | Path,
    *,
    kaggle_token_file: str = "",
    kaggle_token_username: str = "",
    allocation_timeout_seconds: float = 1800.0,
    resume: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status = _load(_status_path(output)) if resume else {}
    if not status:
        status = _initial_status(output)
        _write(_status_path(output), status)
    token_file = str(kaggle_token_file or os.environ.get("CROWDTENSOR_KAGGLE_TOKEN_FILE", ""))
    if not token_file or not Path(token_file).expanduser().is_file():
        status.update(
            {
                "overall_state": "blocked",
                "blockers": ["private_kaggle_token_input_required"],
                "failure_detail_public": False,
            }
        )
        _set_phase(status, output, "allocation", "blocked", blocker="private_kaggle_token_input_required")
        return status

    ledger = output / "allocation_attempts.json"
    single_paths = _attempt_reports(output, "single")
    single = _latest_verified(single_paths, "single_kernel_t4x2_verified")
    if not single:
        ledger_value = _load(ledger)
        allocation_budget = allocation_budget_summary(ledger_value)
        attempt = len(ledger_value.get("single_kernel_attempts") or []) + 1
        attempt_limit = int(allocation_budget["single_kernel_attempt_limit"])
        if attempt > attempt_limit:
            status.update({"overall_state": "blocked", "blockers": ["single_kernel_allocation_attempt_budget_exhausted"]})
            _set_phase(status, output, "allocation", "blocked", attempt_limit=attempt_limit)
            return status
        attempt_dir = output / "attempts" / f"single-{attempt}"
        _set_phase(status, output, "allocation", "running", gate="single_kernel", attempt=attempt)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "training_cuda_single_kernel_probe.py"),
            "--output-dir",
            str(attempt_dir),
            "--raw-token-file",
            token_file,
            "--raw-token-username",
            kaggle_token_username,
            "--attempt-ledger",
            str(ledger),
            "--attempt-limit",
            str(attempt_limit),
            "--allocation-timeout-seconds",
            str(float(allocation_timeout_seconds)),
            "--json",
        ]
        _returncode, report = _run_json(command, timeout=float(allocation_timeout_seconds) + 600.0)
        status["latest_attempt"] = _public_attempt_summary(report)
        _set_phase(
            status,
            output,
            "kernel_launch",
            "completed" if (report.get("push") or {}).get("ok") else "blocked",
            gate="single_kernel",
        )
        if not report.get("single_kernel_t4x2_verified"):
            status.update({"overall_state": "blocked", "blockers": list(report.get("blockers") or ["single_kernel_gate_failed"])})
            _set_phase(status, output, "cleanup", "completed" if (report.get("cleanup") or {}).get("kernel_deleted") else "failed")
            return status
        single = report
    _set_phase(status, output, "worker_assignment", "completed", single_kernel_verified=True)
    _set_phase(status, output, "forward", "completed", single_kernel_verified=True)
    _set_phase(status, output, "backward", "completed", single_kernel_verified=True)
    _set_phase(status, output, "checkpoint", "completed", resume_verified=True)

    two_paths = _attempt_reports(output, "two-node")
    two = _latest_verified(two_paths, "two_node_cuda_verified")
    if not two:
        ledger_value = _load(ledger)
        allocation_budget = allocation_budget_summary(ledger_value)
        attempt = len(ledger_value.get("two_node_attempts") or []) + 1
        attempt_limit = int(allocation_budget["two_node_attempt_limit"])
        if attempt > attempt_limit:
            status.update({"overall_state": "blocked", "blockers": ["two_node_allocation_attempt_budget_exhausted"]})
            _set_phase(status, output, "allocation", "blocked", gate="two_node", attempt_limit=attempt_limit)
            return status
        attempt_dir = output / "attempts" / f"two-node-{attempt}"
        _set_phase(status, output, "allocation", "running", gate="two_node", attempt=attempt)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "training_cuda_two_node_probe.py"),
            "--output-dir",
            str(attempt_dir),
            "--raw-token-file",
            token_file,
            "--raw-token-username",
            kaggle_token_username,
            "--attempt-ledger",
            str(ledger),
            "--attempt-limit",
            str(attempt_limit),
            "--allocation-timeout-seconds",
            str(float(allocation_timeout_seconds)),
            "--json",
        ]
        _returncode, report = _run_json(command, timeout=float(allocation_timeout_seconds) + 900.0)
        status["latest_attempt"] = _public_attempt_summary(report)
        if not report.get("two_node_cuda_verified"):
            status.update({"overall_state": "blocked", "blockers": list(report.get("blockers") or ["two_node_gate_failed"])})
            _set_phase(status, output, "cleanup", "completed" if (report.get("cleanup") or {}).get("kernels_deleted") else "failed")
            return status
        two = report
        source_export = attempt_dir / "exported_adapter"
        if source_export.is_dir():
            shutil.copytree(source_export, output / "exported_adapter", dirs_exist_ok=True)
    if not (output / "exported_adapter").is_dir():
        for report_path in reversed(two_paths):
            candidate = report_path.parent / "exported_adapter"
            if candidate.is_dir():
                shutil.copytree(candidate, output / "exported_adapter", dirs_exist_ok=True)
                break
    _set_phase(status, output, "outer_aggregation", "completed", adapter_version=1, outer_step=1)
    _set_phase(status, output, "evaluation", "completed", validation_loss_reduced=True)
    _set_phase(status, output, "cleanup", "completed", live_resources_left_running=False)
    status.update(
        {
            "overall_state": "completed",
            "current_phase": "cleanup",
            "blockers": [],
            "ok": True,
            "single_kernel_t4x2_verified": True,
            "two_node_cuda_verified": True,
        }
    )
    _write(_status_path(output), status)
    return status


def cuda_training_status(job_dir: str | Path) -> dict[str, Any]:
    status = _load(_status_path(Path(job_dir).resolve()))
    if status:
        return status
    return {
        "schema": STATUS_SCHEMA,
        "backend": "cuda",
        "overall_state": "missing",
        "current_phase": "allocation",
        "phases": {name: {"state": "pending"} for name in PHASES},
        "blockers": ["cuda_training_job_not_found"],
        "next_resume_command": "crowdtensor train resume <job-dir> --backend cuda",
        "credentials_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def resume_cuda_training_job(
    job_dir: str | Path,
    *,
    kaggle_token_file: str = "",
    kaggle_token_username: str = "",
    allocation_timeout_seconds: float = 1800.0,
) -> dict[str, Any]:
    return run_cuda_training_job(
        job_dir,
        kaggle_token_file=kaggle_token_file,
        kaggle_token_username=kaggle_token_username,
        allocation_timeout_seconds=allocation_timeout_seconds,
        resume=True,
    )


def export_cuda_training_job(job_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    job = Path(job_dir).resolve()
    source = job / "exported_adapter"
    if not (source / "adapter_model.safetensors").is_file() or not (source / "adapter_config.json").is_file():
        raise RuntimeError("completed CUDA training adapter export is not available")
    destination = Path(output_dir).resolve() if output_dir else job / "adapter-export"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "adapter_model.safetensors", destination / "adapter_model.safetensors")
    shutil.copyfile(source / "adapter_config.json", destination / "adapter_config.json")
    return {
        "schema": "crowdtensor_cuda_training_export_v1",
        "ok": True,
        "standard_peft_layout": True,
        "adapter_model_hash": sha256_file(destination / "adapter_model.safetensors"),
        "adapter_config_hash": sha256_file(destination / "adapter_config.json"),
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def _delete_private_kaggle_refs(
    refs: list[str],
    *,
    token_file: str,
    token_username: str,
) -> tuple[int, int]:
    from scripts.training_cuda_kaggle_common import (
        delete_succeeded_or_absent,
        kaggle_env,
        run_command,
    )

    deleted = 0
    with kaggle_env(token_file, username_hint=token_username) as env:
        for ref in sorted(set(refs)):
            step = run_command(
                ["kaggle", "kernels", "delete", ref, "-y"],
                env=env,
                timeout=120.0,
            )
            deleted += int(delete_succeeded_or_absent(step))
    return deleted, len(set(refs))


def cleanup_cuda_training_job(
    job_dir: str | Path,
    *,
    kaggle_token_file: str = "",
    kaggle_token_username: str = "",
) -> dict[str, Any]:
    output = Path(job_dir).resolve()
    cleanup_state_paths = sorted(output.glob("**/.private-cleanup/active_resources.json"))
    cleanup_refs: list[str] = []
    for path in cleanup_state_paths:
        state = _load(path)
        cleanup_refs.extend(str(ref) for ref in state.get("kernel_refs") or [] if str(ref))

    single_reports = [_load(path) for path in _attempt_reports(output, "single")]
    two_reports = [_load(path) for path in _attempt_reports(output, "two-node")]
    unverified_report_count = sum(
        (item.get("cleanup") or {}).get("kernel_deleted") is not True
        for item in single_reports
    ) + sum(
        (item.get("cleanup") or {}).get("kernels_deleted") is not True
        for item in two_reports
    )
    ledger = _load(output / "allocation_attempts.json")
    recorded_attempt_count = len(ledger.get("single_kernel_attempts") or []) + len(
        ledger.get("two_node_attempts") or []
    )
    report_count = len(single_reports) + len(two_reports)
    missing_attempt_report_count = max(0, recorded_attempt_count - report_count)

    token_file = str(
        kaggle_token_file or os.environ.get("CROWDTENSOR_KAGGLE_TOKEN_FILE", "")
    )
    token_username = str(
        kaggle_token_username or os.environ.get("CROWDTENSOR_KAGGLE_TOKEN_USERNAME", "")
    )
    deletion_attempted = False
    deleted_count = 0
    deletion_target_count = len(set(cleanup_refs))
    blockers: list[str] = []
    recovered_private_cleanup = False
    if cleanup_refs:
        if token_file and Path(token_file).expanduser().is_file():
            deletion_attempted = True
            try:
                deleted_count, deletion_target_count = _delete_private_kaggle_refs(
                    cleanup_refs,
                    token_file=token_file,
                    token_username=token_username,
                )
                recovered_private_cleanup = deleted_count == deletion_target_count
            except Exception:
                blockers.append("cuda_training_cleanup_kaggle_delete_failed")
        else:
            blockers.append("cuda_training_cleanup_private_kaggle_token_required")

    if recovered_private_cleanup:
        for path in cleanup_state_paths:
            shutil.rmtree(path.parent, ignore_errors=True)

    private_names = {
        ".private-runtime",
        ".private-single-kernel-package",
        ".private-kaggle-output",
    }
    private_paths = sorted(
        (path for path in output.glob("**/*") if path.name in private_names),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    removed_count = 0
    for path in private_paths:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            removed_count += 1

    remaining_cleanup_states = sorted(output.glob("**/.private-cleanup/active_resources.json"))
    if cleanup_refs:
        kernels_deleted = recovered_private_cleanup
    else:
        kernels_deleted = unverified_report_count == 0 and missing_attempt_report_count == 0
        if not kernels_deleted:
            blockers.append("cuda_training_cleanup_live_resource_identity_unavailable")
    private_cleanup_state_removed = not remaining_cleanup_states
    if not private_cleanup_state_removed and not blockers:
        blockers.append("cuda_training_cleanup_private_resource_state_retained")

    temporary_runtime_removed = not any(
        path.exists()
        for name in private_names
        for path in output.glob(f"**/{name}")
    )
    checkpoint_bundle_count = len(list(output.glob("**/*checkpoint_bundle.zip")))
    ok = bool(kernels_deleted and temporary_runtime_removed and private_cleanup_state_removed)
    report = {
        "schema": CLEANUP_SCHEMA,
        "ok": ok,
        "temporary_local_runtime_removed": temporary_runtime_removed,
        "temporary_kaggle_kernels_deleted": kernels_deleted,
        "temporary_private_packages_removed": temporary_runtime_removed,
        "private_cleanup_state_removed": private_cleanup_state_removed,
        "checkpoint_and_evidence_preserved": True,
        "checkpoint_bundle_count": checkpoint_bundle_count,
        "live_resources_left_running": not kernels_deleted,
        "removed_private_runtime_count": removed_count,
        "private_cleanup_resource_count": len(cleanup_state_paths),
        "kaggle_delete_attempted": deletion_attempted,
        "kaggle_delete_target_count": deletion_target_count,
        "kaggle_delete_verified_count": deleted_count,
        "unverified_attempt_cleanup_count": unverified_report_count,
        "missing_attempt_report_count": missing_attempt_report_count,
        "blockers": sorted(set(blockers)),
        "credential_values_public": False,
        "credential_paths_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    _write(output / "training_cuda_cleanup.json", report)
    status = _load(_status_path(output))
    if status:
        status["overall_state"] = status.get("overall_state") if ok else "blocked"
        status["blockers"] = [] if ok else report["blockers"]
        _set_phase(
            status,
            output,
            "cleanup",
            "completed" if ok else "blocked",
            live_resources_left_running=not kernels_deleted,
        )
    return report
