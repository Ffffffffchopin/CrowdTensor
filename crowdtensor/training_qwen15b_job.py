"""User-facing bounded orchestration for Qwen 1.5B four-GPU training."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .qwen15b_training import MODEL_ID, MODEL_REVISION, sha256_file


ROOT = Path(__file__).resolve().parent.parent
STATUS_SCHEMA = "crowdtensor_qwen15b_training_job_status_v1"
CLEANUP_SCHEMA = "crowdtensor_qwen15b_training_job_cleanup_v1"
PHASES = [
    "model_resolution",
    "dataset",
    "account_preflight",
    "allocation",
    "kernel_launch",
    "stage_loading",
    "forward",
    "backward",
    "checkpoint",
    "recovery",
    "evaluation",
    "export",
    "cleanup",
]
SOURCE_REPORT = ROOT / "dist/training-qwen15b-source-20260712-r1/training_qwen15b_source_probe.json"
DATASET_REPORT = ROOT / "dist/training-qwen15b-dataset-20260712-r1/training_qwen15b_dataset_prepare.json"
PRIVATE_DATASET = ROOT / "dist/training-qwen15b-dataset-20260712-r1/qwen15b_tokenized_private.json"
BETA_ATTEMPT_LIMIT = 3
BETA_AUTHORIZATION_HASH = "sha256:" + hashlib.sha256(
    b"crowdtensor-qwen15b-training-beta-rc-three-live-attempts-20260712"
).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _status_path(output: Path) -> Path:
    return output / "training_qwen15b_status.json"


def _initial_status(output: Path) -> dict[str, Any]:
    return {
        "schema": STATUS_SCHEMA,
        "job_id": output.name or "qwen15b-four-gpu-training",
        "backend": "cuda",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "topology": "kaggle-2x-t4x2",
        "steps": 8,
        "overall_state": "running",
        "current_phase": "model_resolution",
        "global_step": 0,
        "retry_count": 0,
        "updated_at": time.time(),
        "phases": {name: {"state": "pending"} for name in PHASES},
        "events": [],
        "blockers": [],
        "next_resume_command": f"crowdtensor train resume {output} --backend cuda",
        "resume_private_inputs": {
            "kaggle_credentials_required": True,
            "supported_methods": [
                "--kaggle-token-file",
                "--kaggle-raw-token-file",
                "CROWDTENSOR_KAGGLE_TOKEN_FILE",
            ],
            "credential_values_public": False,
            "credential_paths_public": False,
        },
        "credentials_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def _set_phase(status: dict[str, Any], output: Path, phase: str, state: str, **details: Any) -> None:
    now = time.time()
    status["current_phase"] = phase
    status["updated_at"] = now
    status.setdefault("phases", {})[phase] = {"state": state, "updated_at": now, **details}
    status.setdefault("events", []).append(
        {
            "sequence": len(status.get("events") or []) + 1,
            "phase": phase,
            "state": state,
            "global_step": int(status.get("global_step") or 0),
            "retry_count": int(status.get("retry_count") or 0),
            "at": now,
        }
    )
    _write(_status_path(output), status)
    store_path = output / ".private-service" / "training_beta_jobs.sqlite3"
    if store_path.is_file():
        try:
            from .training_qwen15b_beta_service import TrainingBetaJobStore

            store = TrainingBetaJobStore(store_path)
            job_id = store.only_job_id()
            store.update_status(
                job_id,
                status,
                event_id=f"phase:{len(status['events'])}:{phase}:{state}",
            )
        except BaseException:
            # The file status remains the recovery source if the optional service mirror is busy.
            pass


def reconcile_qwen15b_training_job_status(job_dir: str | Path) -> dict[str, Any]:
    """Repair stale derived phase state from a completed, verified live attempt."""
    output = Path(job_dir).resolve()
    status = _load(_status_path(output))
    phases = status.get("phases") if isinstance(status.get("phases"), dict) else {}
    allocation = phases.get("allocation")
    latest = status.get("latest_attempt") if isinstance(status.get("latest_attempt"), dict) else {}
    if (
        status.get("overall_state") == "completed"
        and latest.get("ok") is True
        and isinstance(allocation, dict)
        and allocation.get("state") != "completed"
    ):
        _set_phase(
            status,
            output,
            "allocation",
            "completed",
            attempt=int(latest.get("attempt") or status.get("attempts_used") or 0),
            attempt_limit=int(status.get("attempt_limit") or BETA_ATTEMPT_LIMIT),
            reconciled_from_verified_live_attempt=True,
        )
        status["current_phase"] = "cleanup"
        status["overall_state"] = "completed"
        _write(_status_path(output), status)
        store_path = output / ".private-service" / "training_beta_jobs.sqlite3"
        if store_path.is_file():
            try:
                from .training_qwen15b_beta_service import TrainingBetaJobStore

                store = TrainingBetaJobStore(store_path)
                store.update_status(
                    store.only_job_id(),
                    status,
                    event_id="phase-reconciled:allocation-completed",
                )
            except BaseException:
                pass
    return status


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
    payload = {}
    for line in reversed([line.strip() for line in (process.stdout or "").splitlines() if line.strip()]):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            payload = value
            break
    return process.returncode, payload


def _test_summary(output: Path) -> Path:
    destination = output / "training_qwen15b_test_summary.json"
    if destination.is_file() and _load(destination).get("ok") is True:
        return destination
    command = [
        sys.executable,
        str(ROOT / "scripts/training_qwen15b_test_suite.py"),
        "--output-dir",
        str(output),
        "--json",
    ]
    _returncode, report = _run_json(command, timeout=1800.0)
    if report and not destination.is_file():
        _write(destination, report)
    return destination


def _attempt_count(ledger: Path) -> int:
    return len(_load(ledger).get("qwen15b_four_gpu_attempts") or [])


def _ensure_beta_ledger(ledger: Path) -> None:
    value = _load(ledger)
    if value.get("qwen15b_four_gpu_attempts"):
        return
    value.update(
        {
            "schema": "crowdtensor_qwen15b_four_gpu_allocation_ledger_v1",
            "qwen15b_four_gpu_attempts": [],
            "beta_goal_allocation_authorization": {
                "schema": "crowdtensor_qwen15b_beta_goal_allocation_authorization_v1",
                "authorized": True,
                "authorized_at": "2026-07-12T12:00:00Z",
                "authorization_hash": BETA_AUTHORIZATION_HASH,
                "authorization_text_public": False,
                "same_authorized_account_only": True,
                "topology": "kaggle-2x-t4x2",
                "goal_attempt_limit": BETA_ATTEMPT_LIMIT,
                "one_attempt_per_probe_invocation": True,
                "automatic_retry_loop": False,
                "allocation_timeout_seconds": 1800,
            },
        }
    )
    _write(ledger, value)


def _ensure_job_inputs(output: Path) -> dict[str, Any]:
    from scripts.training_qwen15b_dataset_prepare import build as build_dataset
    from scripts.training_qwen15b_source_probe import build as build_source

    source_dir = output / "inputs" / "source"
    private_dataset_dir = output / ".private-inputs" / "dataset"
    source_report_path = source_dir / "training_qwen15b_source_probe.json"
    dataset_report_path = private_dataset_dir / "training_qwen15b_dataset_prepare.json"
    private_dataset_path = private_dataset_dir / "qwen15b_tokenized_private.json"
    source = _load(source_report_path)
    if source.get("ok") is not True:
        source = build_source(source_dir)
    dataset = _load(dataset_report_path)
    if dataset.get("ok") is not True or not private_dataset_path.is_file():
        dataset = build_dataset(private_dataset_dir)
    if source.get("ok") is not True or dataset.get("ok") is not True:
        raise RuntimeError("qwen15b_source_or_private_dataset_prepare_failed")
    source_manifest = source_dir / "qwen15b_source_manifest.json"
    if not source_manifest.is_file() or not private_dataset_path.is_file():
        raise RuntimeError("qwen15b_prepared_input_artifact_missing")
    return {
        "source": source,
        "dataset": dataset,
        "source_report": source_report_path,
        "dataset_report": dataset_report_path,
        "source_manifest": source_manifest,
        "private_dataset": private_dataset_path,
        "generated_by_user_command": True,
        "prebuilt_dist_inputs_used": False,
    }


def run_qwen15b_training_job(
    output_dir: str | Path,
    *,
    model: str,
    topology: str,
    steps: int,
    kaggle_token_files: list[str] | None = None,
    kaggle_raw_token_file: str = "",
    kaggle_raw_token_username: str = "",
    allocation_timeout_seconds: float = 1800.0,
    resume: bool = False,
    beta_mode: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status = _load(_status_path(output)) if resume else {}
    if not status:
        status = _initial_status(output)
        if beta_mode:
            status["schema"] = "crowdtensor_training_qwen15b_beta_job_runtime_status_v1"
            status["next_resume_command"] = "crowdtensor train resume <job> --backend cuda"
            status["beta_mode"] = True
        _write(_status_path(output), status)
    if model != MODEL_ID or topology != "kaggle-2x-t4x2" or int(steps) != 8:
        status.update(
            {
                "overall_state": "blocked",
                "blockers": ["qwen15b_training_request_contract_invalid"],
            }
        )
        _set_phase(
            status,
            output,
            "model_resolution",
            "blocked",
            requested_model=model,
            requested_topology=topology,
            requested_steps=int(steps),
        )
        return status
    files = [str(value) for value in (kaggle_token_files or []) if str(value)]
    env_file = str(os.environ.get("CROWDTENSOR_KAGGLE_TOKEN_FILE") or "")
    if env_file and env_file not in files:
        files.append(env_file)
    has_private_input = any(Path(value).expanduser().is_file() for value in files) or (
        bool(kaggle_raw_token_file) and Path(kaggle_raw_token_file).expanduser().is_file()
    )
    if not has_private_input:
        status.update(
            {"overall_state": "blocked", "blockers": ["private_kaggle_token_input_required"]}
        )
        _set_phase(status, output, "account_preflight", "blocked")
        return status

    if beta_mode:
        try:
            inputs = _ensure_job_inputs(output)
        except BaseException:
            status.update(
                {
                    "overall_state": "blocked",
                    "blockers": ["qwen15b_source_or_private_dataset_prepare_failed"],
                }
            )
            _set_phase(status, output, "model_resolution", "blocked")
            return status
        source = dict(inputs["source"])
        dataset = dict(inputs["dataset"])
        source_report_path = Path(inputs["source_report"])
        dataset_report_path = Path(inputs["dataset_report"])
        source_manifest_path = Path(inputs["source_manifest"])
        private_dataset_path = Path(inputs["private_dataset"])
        status["input_preparation"] = {
            "generated_by_user_command": True,
            "prebuilt_dist_inputs_used": False,
            "source_report_hash": sha256_file(source_report_path),
            "dataset_report_hash": sha256_file(dataset_report_path),
            "private_payload_present": True,
            "private_paths_public": False,
        }
    else:
        source = _load(SOURCE_REPORT)
        dataset = _load(DATASET_REPORT)
        source_report_path = SOURCE_REPORT
        dataset_report_path = DATASET_REPORT
        source_manifest_path = SOURCE_REPORT.parent / "qwen15b_source_manifest.json"
        private_dataset_path = PRIVATE_DATASET
        if not source.get("ok") or not dataset.get("ok") or not private_dataset_path.is_file():
            status.update(
                {
                    "overall_state": "blocked",
                    "blockers": ["qwen15b_source_or_private_dataset_missing"],
                }
            )
            _set_phase(status, output, "model_resolution", "blocked")
            return status
    _set_phase(
        status,
        output,
        "model_resolution",
        "completed",
        model=MODEL_ID,
        revision=MODEL_REVISION,
        parameter_count=int((source.get("source") or {}).get("parameter_count") or 0),
        stage_count=4,
    )
    _set_phase(
        status,
        output,
        "dataset",
        "completed",
        sequence_length=int((dataset.get("manifest") or {}).get("sequence_length") or 0),
        token_ids_public=False,
    )

    test_summary = _test_summary(output) if not beta_mode else None
    if test_summary is not None and not _load(test_summary).get("ok"):
        status.update(
            {"overall_state": "blocked", "blockers": ["qwen15b_required_tests_failed"]}
        )
        _set_phase(status, output, "model_resolution", "blocked", tests_passed=False)
        return status

    ledger = output / "allocation_attempts.json"
    if beta_mode:
        _ensure_beta_ledger(ledger)
    attempt_limit = BETA_ATTEMPT_LIMIT if beta_mode else 2
    attempt = _attempt_count(ledger) + 1
    if attempt > attempt_limit:
        status.update(
            {
                "overall_state": "blocked",
                "blockers": ["qwen15b_four_gpu_allocation_attempt_budget_exhausted"],
            }
        )
        _set_phase(status, output, "allocation", "blocked", attempt_limit=attempt_limit)
        return status
    attempt_dir = output / "attempts" / f"qwen15b-{'beta' if beta_mode else 'four-gpu'}-{attempt}"
    cancel_file = output / ".private-service" / "cancel.requested"
    _set_phase(status, output, "account_preflight", "running", attempt=attempt)
    _set_phase(status, output, "allocation", "running", attempt=attempt, attempt_limit=attempt_limit)
    command = [
        sys.executable,
        str(ROOT / "scripts/training_qwen15b_four_gpu_probe.py"),
        "--output-dir",
        str(attempt_dir),
        "--tokenized-payload",
        str(private_dataset_path),
        "--source-manifest",
        str(source_manifest_path),
        "--attempt-ledger",
        str(ledger),
        "--attempt-limit",
        str(attempt_limit),
        "--allocation-timeout-seconds",
        str(min(1800.0, float(allocation_timeout_seconds))),
    ]
    if beta_mode:
        command.extend(
            [
                "--coordinator-restart-after-step",
                "4",
                "--coordinator-restart-downtime-seconds",
                "3",
                "--cancel-file",
                str(cancel_file),
            ]
        )
    for value in files:
        command.extend(["--token-file", str(Path(value).expanduser())])
    if kaggle_raw_token_file:
        command.extend(["--raw-token-file", str(Path(kaggle_raw_token_file).expanduser())])
    if kaggle_raw_token_username:
        command.extend(["--raw-token-username", kaggle_raw_token_username])
    command.append("--json")
    _returncode, live = _run_json(command, timeout=float(allocation_timeout_seconds) + 1200.0)
    if not live:
        live = _load(attempt_dir / "training_qwen15b_four_gpu_live_probe.json")
    status["latest_attempt"] = {
        "attempt": int(live.get("attempt") or attempt),
        "ok": live.get("ok") is True,
        "blockers": list(live.get("blockers") or []),
        "eligible_account_count": int(live.get("eligible_account_count") or 0),
        "route_preflight_verified": live.get("route_preflight_verified") is True,
        "max_observed_running_kernel_count": int(
            live.get("max_observed_running_kernel_count") or 0
        ),
        "cleanup": live.get("cleanup") or {},
        "public_artifact_safe": live.get("public_artifact_safe") is True,
    }
    preflight_ok = int(live.get("eligible_account_count") or 0) > 0
    _set_phase(
        status,
        output,
        "account_preflight",
        "completed" if preflight_ok else "blocked",
        eligible_account_count=int(live.get("eligible_account_count") or 0),
    )
    allocation_ok = bool(
        live.get("allocation_started") is True and int(live.get("attempt") or 0) == attempt
    )
    _set_phase(
        status,
        output,
        "allocation",
        "completed" if allocation_ok else "blocked",
        attempt=attempt,
        attempt_limit=attempt_limit,
    )
    _set_phase(
        status,
        output,
        "kernel_launch",
        "completed" if len(live.get("kernel_ref_hashes") or []) == 2 else "blocked",
        same_account=True,
    )
    workers = list(live.get("worker_reports") or [])
    evidence = dict(live.get("evidence") or {})
    stage_count = len(
        {
            int(ready.get("stage_id", -1))
            for worker in workers
            for values in ((worker.get("worker") or {}).get("stage_ready") or {}).values()
            for ready in values
        }
    )
    _set_phase(
        status,
        output,
        "stage_loading",
        "completed" if stage_count == 4 else "blocked",
        stage_count=stage_count,
    )
    _set_phase(
        status,
        output,
        "forward",
        "completed" if evidence.get("four_stage_compute_overlap_verified") else "blocked",
        four_gpu_overlap=evidence.get("four_stage_compute_overlap_verified") is True,
    )
    _set_phase(
        status,
        output,
        "backward",
        "completed" if int(evidence.get("gradient_payload_count") or 0) == 64 else "blocked",
        gradient_payload_count=int(evidence.get("gradient_payload_count") or 0),
    )
    _set_phase(
        status,
        output,
        "checkpoint",
        "completed" if evidence.get("checkpoint_archives_verified") else "blocked",
    )
    _set_phase(
        status,
        output,
        "recovery",
        "completed" if evidence.get("controlled_restart_verified") else "blocked",
        controlled_restart=evidence.get("controlled_restart_verified") is True,
    )
    _set_phase(
        status,
        output,
        "evaluation",
        "completed" if evidence.get("evaluation_verified") else "blocked",
    )
    _set_phase(
        status,
        output,
        "export",
        "completed" if evidence.get("standard_peft_export_verified") else "blocked",
    )
    cleanup_ok = all(
        (live.get("cleanup") or {}).get(key) is True
        for key in (
            "kernels_deleted",
            "private_packages_removed",
            "coordinator_stopped",
            "tunnel_stopped",
            "private_runtime_removed",
        )
    )
    _set_phase(
        status,
        output,
        "cleanup",
        "completed" if cleanup_ok else "blocked",
        live_resources_left_running=not (live.get("cleanup") or {}).get("kernels_deleted", False),
    )
    if (attempt_dir / "exported_adapter").is_dir():
        shutil.copytree(attempt_dir / "exported_adapter", output / "exported_adapter", dirs_exist_ok=True)

    if beta_mode:
        strict_ready = bool(
            live.get("ok") and live.get("training_qwen15b_beta_live_verified")
        )
    else:
        alpha_dir = output / "alpha"
        alpha_command = [
            sys.executable,
            str(ROOT / "scripts/training_qwen15b_four_gpu_alpha_pack.py"),
            "--output-dir",
            str(alpha_dir),
            "--source-report",
            str(source_report_path),
            "--dataset-report",
            str(dataset_report_path),
            "--test-summary",
            str(test_summary),
            "--live-report",
            str(attempt_dir / "training_qwen15b_four_gpu_live_probe.json"),
            "--allocation-ledger",
            str(ledger),
            "--json",
        ]
        _pack_code, alpha = _run_json(alpha_command, timeout=120.0)
        if not alpha:
            alpha = _load(alpha_dir / "training_qwen15b_four_gpu_alpha.json")
        strict_ready = bool((alpha.get("checker") or {}).get("qwen15b_four_gpu_alpha_ready"))
    cancelled = bool(
        beta_mode
        and (
            cancel_file.is_file()
            or "qwen15b_user_cancelled" in set(live.get("blockers") or [])
        )
    )
    status.update(
        {
            "overall_state": "cancelled" if cancelled else "completed" if strict_ready else "blocked",
            "current_phase": "cleanup",
            "blockers": []
            if strict_ready
            else list(live.get("blockers") or ["qwen15b_four_gpu_alpha_not_ready"]),
            "ok": strict_ready,
            "goal_achieved": strict_ready,
            "qwen15b_four_gpu_alpha_ready": strict_ready,
            "training_qwen15b_beta_live_candidate_ready": strict_ready if beta_mode else False,
            "global_step": 8 if strict_ready else int(status.get("global_step") or 0),
            "attempts_used": _attempt_count(ledger),
            "attempt_limit": attempt_limit,
            "user_command_path_executed": beta_mode,
            "prebuilt_dist_inputs_used": False if beta_mode else True,
        }
    )
    _write(_status_path(output), status)
    return status


def qwen15b_training_status(job_dir: str | Path) -> dict[str, Any]:
    output = Path(job_dir).resolve()
    status = reconcile_qwen15b_training_job_status(output)
    if status:
        return status
    return {
        "schema": STATUS_SCHEMA,
        "backend": "cuda",
        "model": MODEL_ID,
        "topology": "kaggle-2x-t4x2",
        "overall_state": "missing",
        "current_phase": "model_resolution",
        "phases": {name: {"state": "pending"} for name in PHASES},
        "blockers": ["qwen15b_training_job_not_found"],
        "next_resume_command": "crowdtensor train resume <job> --backend cuda",
        "credentials_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def resume_qwen15b_training_job(
    job_dir: str | Path,
    *,
    kaggle_token_files: list[str] | None = None,
    kaggle_raw_token_file: str = "",
    kaggle_raw_token_username: str = "",
    allocation_timeout_seconds: float = 1800.0,
) -> dict[str, Any]:
    return run_qwen15b_training_job(
        job_dir,
        model=MODEL_ID,
        topology="kaggle-2x-t4x2",
        steps=8,
        kaggle_token_files=kaggle_token_files,
        kaggle_raw_token_file=kaggle_raw_token_file,
        kaggle_raw_token_username=kaggle_raw_token_username,
        allocation_timeout_seconds=allocation_timeout_seconds,
        resume=True,
        beta_mode=(
            _load(_status_path(Path(job_dir).resolve())).get("beta_mode") is True
            or (
                Path(job_dir).resolve()
                / ".private-service"
                / "training_beta_jobs.sqlite3"
            ).is_file()
        ),
    )


def export_qwen15b_training_job(
    job_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    job = Path(job_dir).resolve()
    source = job / "exported_adapter"
    required = [source / "adapter_model.safetensors", source / "adapter_config.json"]
    if not all(path.is_file() for path in required):
        raise RuntimeError("Qwen 1.5B standard PEFT export is not available")
    destination = Path(output_dir).resolve() if output_dir else job / "adapter-export"
    destination.mkdir(parents=True, exist_ok=True)
    for path in required:
        shutil.copyfile(path, destination / path.name)
    report = {
        "schema": "crowdtensor_qwen15b_training_export_v1",
        "ok": True,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "standard_peft_layout": True,
        "adapter_model_hash": sha256_file(destination / "adapter_model.safetensors"),
        "adapter_config_hash": sha256_file(destination / "adapter_config.json"),
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    _write(job / "training_qwen15b_export.json", report)
    return report


def cleanup_qwen15b_training_job(
    job_dir: str | Path,
    *,
    kaggle_token_files: list[str] | None = None,
    kaggle_raw_token_file: str = "",
    kaggle_raw_token_username: str = "",
) -> dict[str, Any]:
    from scripts.kaggle_gpu_token_weekly_quota_probe import (
        clean_env,
        parse_raw_token_file,
        parse_token_sections,
    )
    from scripts.training_cuda_kaggle_common import (
        authenticated_owner,
        delete_succeeded_or_absent,
        run_command,
    )

    job = Path(job_dir).resolve()
    states = sorted(job.glob("attempts/**/.private-runtime/active_resources.json"))
    refs = sorted(
        {
            str(ref)
            for state in states
            for ref in (_load(state).get("kernel_refs") or [])
            if str(ref)
        }
    )
    sections = []
    for value in kaggle_token_files or []:
        path = Path(value).expanduser()
        if path.is_file():
            sections.extend(parse_token_sections(path))
    if kaggle_raw_token_file and Path(kaggle_raw_token_file).expanduser().is_file():
        sections.append(
            parse_raw_token_file(
                Path(kaggle_raw_token_file).expanduser(),
                username_hint=kaggle_raw_token_username,
            )
        )
    deleted = 0
    unresolved = set(refs)
    for section in sections:
        with tempfile.TemporaryDirectory(prefix="ct-qwen15b-user-cleanup-") as config_dir:
            env = clean_env(dict(section.get("env") or {}), config_dir=Path(config_dir))
            owner = authenticated_owner(env)
            owned = sorted(ref for ref in unresolved if ref.split("/", 1)[0] == owner)
            for ref in owned:
                step = run_command(
                    ["kaggle", "kernels", "delete", ref, "-y"],
                    env=env,
                    timeout=120.0,
                )
                if delete_succeeded_or_absent(step):
                    unresolved.discard(ref)
                    deleted += 1
    for path in sorted(job.glob("attempts/**/.private-runtime"), reverse=True):
        if not unresolved:
            shutil.rmtree(path, ignore_errors=True)
    latest_reports = sorted(job.glob("attempts/**/training_qwen15b_four_gpu_live_probe.json"))
    latest_cleanup = _load(latest_reports[-1]).get("cleanup") if latest_reports else {}
    kernels_deleted = not unresolved and (
        not refs or bool((latest_cleanup or {}).get("kernels_deleted")) or deleted == len(refs)
    )
    report = {
        "schema": CLEANUP_SCHEMA,
        "ok": kernels_deleted,
        "temporary_kaggle_kernels_deleted": kernels_deleted,
        "only_recorded_job_kernel_refs_targeted": True,
        "target_count": len(refs),
        "deleted_count": deleted,
        "live_resources_left_running": bool(unresolved),
        "temporary_private_runtime_removed": not any(
            job.glob("attempts/**/.private-runtime")
        ),
        "checkpoint_and_evidence_preserved": True,
        "credentials_public": False,
        "credential_paths_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    _write(job / "training_qwen15b_cleanup.json", report)
    status = _load(_status_path(job))
    if status:
        existing_cleanup = status.get("cleanup")
        already_cleaned = bool(
            status.get("overall_state") == "cleaned"
            and isinstance(existing_cleanup, dict)
            and existing_cleanup.get("ok") is True
        )
        if not already_cleaned:
            _set_phase(
                status,
                job,
                "cleanup",
                "completed" if report["ok"] else "blocked",
                live_resources_left_running=report["live_resources_left_running"],
            )
            if report["ok"]:
                status["overall_state"] = "cleaned"
                status["current_phase"] = "cleanup"
                status["cleanup"] = report
                _write(_status_path(job), status)
                store_path = job / ".private-service" / "training_beta_jobs.sqlite3"
                if store_path.is_file():
                    try:
                        from .training_qwen15b_beta_service import TrainingBetaJobStore

                        store = TrainingBetaJobStore(store_path)
                        store.update_status(
                            store.only_job_id(),
                            status,
                            event_id="job-cleanup-state-v1",
                        )
                    except BaseException:
                        pass
    return report
