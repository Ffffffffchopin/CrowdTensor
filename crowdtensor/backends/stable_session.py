"""Bounded stable-rank-group execution through Training Architecture v2."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from crowdtensor.core.contracts import (
    ArtifactRef,
    CheckpointLineage,
    CheckpointRef,
    ContributionReceipt,
    ReceiptOutcome,
    TrainingMode,
    WorkUnit,
    stable_hash,
)
from crowdtensor.core.controller import SessionController
from crowdtensor.core.execution import StableShardedLaunchSpec, TrainingExecutionPlan
from crowdtensor.core.workspace import (
    CONTROL_DIR,
    load_project,
    load_recorded_plan,
    run_workspace,
)


STABLE_TRAINER_RESULT_SCHEMA = "crowdtensor_stable_sharded_trainer_result_v1"
STABLE_SESSION_REPORT_SCHEMA = "crowdtensor_stable_sharded_session_v2"
_PLACEHOLDER = re.compile(r"\$\{([A-Z][A-Z0-9_]{0,127})\}")
_METRIC = re.compile(r"[A-Za-z0-9_.:/-]{1,128}\Z")


class StableShardedSessionError(ValueError):
    """A public-safe stable rank-group orchestration error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("content_hash", None)
    result["content_hash"] = stable_hash(result)
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _directory_digest(root: Path) -> tuple[str, int, int]:
    if not root.is_dir() or root.is_symlink():
        raise StableShardedSessionError("stable_checkpoint_payload_missing")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise StableShardedSessionError("stable_checkpoint_payload_empty")
    digest = hashlib.sha256()
    total = 0
    for path in files:
        if path.is_symlink():
            raise StableShardedSessionError("stable_checkpoint_symlink_forbidden")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                total += len(chunk)
    return "sha256:" + digest.hexdigest(), total, len(files)


def validate_stable_trainer_result(
    value: Mapping[str, Any],
    *,
    work: WorkUnit,
    base_checkpoint: CheckpointRef,
    expected_rank_count: int,
) -> dict[str, Any]:
    """Validate the small framework-facing result contract from rank zero."""

    if not isinstance(value, Mapping):
        raise StableShardedSessionError("stable_trainer_result_invalid")
    payload = dict(value)
    required = {
        "schema",
        "work_unit_hash",
        "base_checkpoint_hash",
        "step_start",
        "steps_completed",
        "rank_count",
        "distributed_type",
        "device_type",
        "restored_step",
        "samples",
        "tokens",
        "metrics",
        "credential_values_public",
        "private_paths_public",
        "public_artifact_safe",
        "content_hash",
    }
    if set(payload) != required or payload.get("schema") != STABLE_TRAINER_RESULT_SCHEMA:
        raise StableShardedSessionError("stable_trainer_result_invalid")
    supplied = str(payload.pop("content_hash") or "")
    if supplied != stable_hash(payload):
        raise StableShardedSessionError("stable_trainer_result_hash_invalid")
    payload["content_hash"] = supplied
    if (
        payload.get("work_unit_hash") != work.content_hash
        or payload.get("base_checkpoint_hash") != base_checkpoint.content_hash
        or payload.get("step_start") != work.step_start
        or payload.get("steps_completed") != work.step_count
        or payload.get("restored_step") != base_checkpoint.step
        or payload.get("rank_count") != int(expected_rank_count)
        or payload.get("distributed_type") != "fsdp2"
        or payload.get("device_type") not in {"cpu", "cuda"}
        or payload.get("credential_values_public") is not False
        or payload.get("private_paths_public") is not False
        or payload.get("public_artifact_safe") is not True
    ):
        raise StableShardedSessionError("stable_trainer_result_binding_invalid")
    for field in ("samples", "tokens"):
        current = payload.get(field)
        if isinstance(current, bool) or not isinstance(current, int) or current < 0:
            raise StableShardedSessionError("stable_trainer_result_counter_invalid")
    metrics = payload.get("metrics")
    if not isinstance(metrics, list) or len(metrics) > 64:
        raise StableShardedSessionError("stable_trainer_result_metrics_invalid")
    names: set[str] = set()
    for item in metrics:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not _METRIC.fullmatch(item[0])
            or item[0] in names
            or isinstance(item[1], bool)
            or not isinstance(item[1], (int, float))
            or not math.isfinite(float(item[1]))
        ):
            raise StableShardedSessionError("stable_trainer_result_metrics_invalid")
        names.add(item[0])
    return payload


@contextmanager
def _session_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StableShardedSessionError("stable_session_already_running") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _render_command(
    spec: StableShardedLaunchSpec,
    bindings: Mapping[str, str],
    *,
    command_factory: Callable[
        [StableShardedLaunchSpec, Mapping[str, str]], Sequence[str]
    ]
    | None,
) -> tuple[str, ...]:
    if command_factory is not None:
        command = tuple(str(item) for item in command_factory(spec, bindings))
    else:
        rendered = []
        for argument in spec.command_template:
            value = str(argument)
            for name, replacement in bindings.items():
                value = value.replace("${" + name + "}", replacement)
            rendered.append(value)
        command = tuple(rendered)
    if not command or any(not item or "\x00" in item or "\n" in item for item in command):
        raise StableShardedSessionError("stable_launch_command_invalid")
    if any(_PLACEHOLDER.search(item) for item in command):
        raise StableShardedSessionError("stable_launch_environment_unbound")
    if command_factory is None and command[0] == "accelerate":
        from .accelerate import accelerate_executable

        executable = accelerate_executable()
        if not executable:
            raise StableShardedSessionError("stable_accelerate_executable_missing")
        command = (executable, *command[1:])
    return command


def _run_rank_group(
    command: Sequence[str],
    *,
    workspace: Path,
    log_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment["PYTHONUNBUFFERED"] = "1"
    timed_out = False
    start_failed = False
    with log_path.open("wb") as output:
        try:
            process = subprocess.Popen(
                list(command),
                cwd=workspace,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError:
            start_failed = True
            return_code = 127
        else:
            try:
                return_code = process.wait(timeout=float(timeout_seconds))
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    return_code = process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    return_code = process.wait(timeout=5.0)
    output_digest, output_bytes = _file_digest(log_path)
    log_path.unlink(missing_ok=True)
    return {
        "return_code": int(return_code),
        "start_failed": start_failed,
        "timed_out": timed_out,
        "output_digest": output_digest,
        "output_bytes": output_bytes,
    }


def _genesis_checkpoint(project: Any) -> CheckpointRef:
    return CheckpointRef(
        checkpoint_id="full-v0",
        project_hash=project.content_hash,
        step=0,
        generation=0,
        artifact=ArtifactRef(
            f"crowdtensor://project/{project.project_id}/base",
            project.model.revision,
            stable_hash({"model": project.model.to_dict()}),
        ),
        adapter_only=False,
    )


def _work_for_head(
    project: Any,
    head: CheckpointRef,
    *,
    step_count: int,
    generation: int,
) -> WorkUnit:
    end = head.step + int(step_count)
    return WorkUnit(
        work_id=f"stable-step-{head.step}-{end}",
        project_hash=project.content_hash,
        mode=TrainingMode.STABLE_SHARDED,
        generation=int(generation),
        backend=project.training_backend,
        base_checkpoint_hash=head.content_hash,
        data_shard_hash=stable_hash(
            {"dataset": project.dataset.to_dict(), "step_start": head.step, "step_end": end}
        ),
        step_start=head.step,
        step_count=int(step_count),
        required_capabilities=("distributed_collective", "stable_sharded"),
    )


def _rank_group_owner(plan: TrainingExecutionPlan) -> str:
    return stable_hash(
        {
            "plan_hash": plan.content_hash,
            "resources": [item.content_hash for item in plan.selected_resources],
        }
    )


def _failure_receipt(
    work: WorkUnit,
    *,
    owner_hash: str,
    code: str,
    process_result: Mapping[str, Any],
) -> ContributionReceipt:
    return ContributionReceipt(
        receipt_id=f"failed-{work.work_id}-g{work.generation}",
        project_hash=work.project_hash,
        work_id=work.work_id,
        work_generation=work.generation,
        contributor_id_hash=owner_hash,
        base_checkpoint_hash=work.base_checkpoint_hash,
        submitted_artifact_hash=stable_hash(
            {
                "work_unit_hash": work.content_hash,
                "code": code,
                "return_code": int(process_result.get("return_code") or 0),
                "start_failed": bool(process_result.get("start_failed")),
                "timed_out": bool(process_result.get("timed_out")),
                "output_digest": str(process_result.get("output_digest") or stable_hash("")),
            }
        ),
        outcome=ReceiptOutcome.REJECTED,
        completed_at=_utc_now(),
        steps=0,
        samples=0,
        tokens=0,
        checkpoint_committed=False,
        rejection_code=code,
    )


def _read_result(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StableShardedSessionError("stable_trainer_result_invalid") from exc
    if not isinstance(value, dict):
        raise StableShardedSessionError("stable_trainer_result_invalid")
    path.chmod(0o600)
    return value


def _commit_success(
    controller: SessionController,
    project: Any,
    work: WorkUnit,
    base: CheckpointRef,
    *,
    owner_hash: str,
    result_path: Path,
    attempt_path: Path,
    checkpoint_root: Path,
    expected_rank_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = validate_stable_trainer_result(
        _read_result(result_path),
        work=work,
        base_checkpoint=base,
        expected_rank_count=expected_rank_count,
    )
    end = work.step_start + work.step_count
    checkpoint_id = f"full-step-{end}"
    canonical_path = checkpoint_root / "payloads" / checkpoint_id
    payload_path = attempt_path if attempt_path.is_dir() else canonical_path
    payload_digest, payload_bytes, payload_files = _directory_digest(payload_path)
    if canonical_path.is_dir():
        canonical_digest, _, _ = _directory_digest(canonical_path)
        if canonical_digest != payload_digest:
            raise StableShardedSessionError("stable_checkpoint_payload_conflict")
        if attempt_path.is_dir() and attempt_path != canonical_path:
            shutil.rmtree(attempt_path)
    else:
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(attempt_path, canonical_path)
    output = CheckpointRef(
        checkpoint_id=checkpoint_id,
        project_hash=project.content_hash,
        step=end,
        generation=base.generation + 1,
        artifact=ArtifactRef(
            f"crowdtensor://project/{project.project_id}/stable-sharded/{checkpoint_id}",
            checkpoint_id,
            payload_digest,
        ),
        parent_hash=base.content_hash,
        created_by_work_id=work.work_id,
        adapter_only=False,
    )
    receipt = ContributionReceipt(
        receipt_id=f"accepted-{work.work_id}-g{work.generation}",
        project_hash=project.content_hash,
        work_id=work.work_id,
        work_generation=work.generation,
        contributor_id_hash=owner_hash,
        base_checkpoint_hash=base.content_hash,
        submitted_artifact_hash=payload_digest,
        outcome=ReceiptOutcome.ACCEPTED,
        completed_at=_utc_now(),
        steps=work.step_count,
        samples=int(result["samples"]),
        tokens=int(result["tokens"]),
        checkpoint_committed=True,
        output_checkpoint_hash=output.content_hash,
        metrics=tuple((str(name), float(value)) for name, value in result["metrics"]),
    )
    report = controller.commit(
        work,
        receipt,
        base_checkpoint=base,
        output_checkpoint=output,
    )
    return report, {
        "device_type": result["device_type"],
        "rank_count": result["rank_count"],
        "restored_step": result["restored_step"],
        "payload_digest": payload_digest,
        "payload_bytes": payload_bytes,
        "payload_file_count": payload_files,
    }


def _load_plan_and_spec(
    workspace: Path,
) -> tuple[TrainingExecutionPlan, StableShardedLaunchSpec]:
    recorded = load_recorded_plan(workspace)
    if recorded is None:
        raise StableShardedSessionError("stable_execution_plan_required")
    try:
        plan = TrainingExecutionPlan.from_dict(recorded.get("plan") or {})
        spec = StableShardedLaunchSpec.from_dict(recorded.get("launch") or {})
    except (TypeError, ValueError) as exc:
        raise StableShardedSessionError("stable_execution_plan_invalid") from exc
    if (
        plan.mode is not TrainingMode.STABLE_SHARDED
        or plan.content_hash != spec.plan_hash
        or plan.project_hash != spec.project_hash
        or plan.backend_id != spec.backend_id
        or not plan.execution_ready
        or not spec.execution_ready
    ):
        raise StableShardedSessionError("stable_execution_plan_not_ready")
    if spec.num_machines != 1:
        raise StableShardedSessionError("stable_multimachine_launcher_required")
    return plan, spec


def run_stable_sharded_session(
    workspace: str | Path,
    *,
    backend: Any,
    steps_per_work_unit: int = 0,
    max_work_units: int = 1,
    timeout_seconds: float = 3600.0,
    dry_run: bool = False,
    command_factory: Callable[
        [StableShardedLaunchSpec, Mapping[str, str]], Sequence[str]
    ]
    | None = None,
    launcher_label: str = "accelerate",
) -> dict[str, Any]:
    """Run bounded stable work and recover whole rank groups from lineage head."""

    root = Path(workspace).expanduser().resolve()
    project = load_project(root)
    if (
        project.mode is not TrainingMode.STABLE_SHARDED
        or project.training_backend != getattr(backend, "backend_id", None)
    ):
        raise StableShardedSessionError("stable_session_backend_mismatch")
    if isinstance(steps_per_work_unit, bool) or int(steps_per_work_unit) < 0:
        raise StableShardedSessionError("stable_steps_per_work_unit_invalid")
    if isinstance(max_work_units, bool) or not 1 <= int(max_work_units) <= 1024:
        raise StableShardedSessionError("stable_max_work_units_invalid")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or not 1.0 <= float(timeout_seconds) <= 86400.0
    ):
        raise StableShardedSessionError("stable_timeout_invalid")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", str(launcher_label)):
        raise StableShardedSessionError("stable_launcher_label_invalid")

    plan, spec = _load_plan_and_spec(root)
    if plan.project_hash != project.content_hash or spec.project_hash != project.content_hash:
        raise StableShardedSessionError("stable_session_project_mismatch")
    trainer_path = root / spec.trainer_entrypoint
    if not trainer_path.is_file():
        raise StableShardedSessionError("stable_trainer_entrypoint_missing")
    plan_device_types = {item.device_type for item in plan.selected_resources}
    if plan_device_types == {"cpu"} and command_factory is None:
        raise StableShardedSessionError("stable_cpu_validation_launcher_required")
    if plan_device_types not in ({"cuda"}, {"cpu"}):
        raise StableShardedSessionError("stable_session_device_group_invalid")
    runtime_root = root / CONTROL_DIR / "runtime/stable-sharded"
    runtime_root.mkdir(parents=True, exist_ok=True)
    runtime_root.chmod(0o700)
    lock_path = root / CONTROL_DIR / "state/stable-sharded-session.lock"
    with _session_lock(lock_path):
        controller = SessionController(root)
        controller.initialize(
            CheckpointLineage(project.content_hash, (_genesis_checkpoint(project),))
        )
        materialization = backend.materialize_launch(spec, workspace=root)
        initial_status = controller.status()
        initial_step = controller.lineage().checkpoints[-1].step
        if dry_run:
            return _seal(
                {
                    "schema": STABLE_SESSION_REPORT_SCHEMA,
                    "command_ok": True,
                    "state": "preflight_ready",
                    "project_hash": project.content_hash,
                    "plan_hash": plan.content_hash,
                    "launch_spec_hash": spec.content_hash,
                    "launcher": str(launcher_label),
                    "target_steps": project.target_steps,
                    "initial_step": initial_step,
                    "final_step": initial_step,
                    "work_units_completed": 0,
                    "failed_attempts": 0,
                    "restart_count": 0,
                    "recovered_commit_count": 0,
                    "rank_count": spec.num_processes,
                    "device_types": [],
                    "command_attempted": False,
                    "command_executed": False,
                    "execution_complete": initial_step >= project.target_steps,
                    "fsdp2_trainer_result_verified": False,
                    "cpu_execution_reported": False,
                    "gpu_execution_reported": False,
                    "hardware_attestation_verified": False,
                    "physical_multi_host_verified": False,
                    "materialization_hash": materialization["content_hash"],
                    "checkpoint_count": initial_status["checkpoint_count"],
                    "terminal_count": initial_status["terminal_count"],
                    "credential_values_public": False,
                    "private_paths_public": False,
                    "public_artifact_safe": True,
                }
            )

        run_workspace(root, controller_ready=True, execution_started=True)
        completed = 0
        failed = 0
        restarts = 0
        recovered = 0
        process_reports: list[dict[str, Any]] = []
        verified_results: list[dict[str, Any]] = []
        final_failure = ""
        owner_hash = _rank_group_owner(plan)
        try:
            while completed < int(max_work_units):
                base = controller.lineage().checkpoints[-1]
                if base.step >= project.target_steps:
                    break
                count = min(
                    int(steps_per_work_unit) or project.target_steps - base.step,
                    project.target_steps - base.step,
                )
                work_id = f"stable-step-{base.step}-{base.step + count}"
                active = controller.active_work()
                generation = controller.next_generation(work_id)
                if active is not None:
                    expected = _work_for_head(
                        project,
                        base,
                        step_count=count,
                        generation=active.generation,
                    )
                    if active.content_hash != expected.content_hash:
                        raise StableShardedSessionError(
                            "stable_active_work_contract_conflict"
                        )
                    generation = active.generation
                    result_path = runtime_root / "results" / f"{work_id}-g{generation}.json"
                    attempt_path = (
                        root
                        / spec.checkpoint_path
                        / "attempts"
                        / f"{work_id}-g{generation}"
                    )
                    canonical_path = (
                        root
                        / spec.checkpoint_path
                        / "payloads"
                        / f"full-step-{base.step + count}"
                    )
                    if result_path.is_file() and (
                        attempt_path.is_dir() or canonical_path.is_dir()
                    ):
                        _, verified = _commit_success(
                            controller,
                            project,
                            active,
                            base,
                            owner_hash=owner_hash,
                            result_path=result_path,
                            attempt_path=attempt_path,
                            checkpoint_root=root / spec.checkpoint_path,
                            expected_rank_count=spec.num_processes,
                        )
                        verified_results.append(verified)
                        recovered += 1
                        completed += 1
                        continue
                    interrupted = _failure_receipt(
                        active,
                        owner_hash=owner_hash,
                        code="stable_rank_group_interrupted",
                        process_result={},
                    )
                    controller.commit(active, interrupted, base_checkpoint=base)
                    failed += 1
                    generation = controller.next_generation(work_id)

                succeeded = False
                for attempt_index in range(spec.max_restarts + 1):
                    work = _work_for_head(
                        project,
                        base,
                        step_count=count,
                        generation=generation,
                    )
                    controller.issue(work, contributor_id_hash=owner_hash)
                    attempt_name = f"{work.work_id}-g{work.generation}"
                    checkpoint_root = root / spec.checkpoint_path
                    attempt_path = checkpoint_root / "attempts" / attempt_name
                    result_path = runtime_root / "results" / f"{attempt_name}.json"
                    contract_root = runtime_root / "contracts"
                    work_path = contract_root / f"{attempt_name}.work.json"
                    base_path = contract_root / f"{attempt_name}.base.json"
                    base_payload = (
                        checkpoint_root / "genesis"
                        if base.step == 0
                        else checkpoint_root / "payloads" / base.checkpoint_id
                    )
                    if base.step > 0:
                        digest, _, _ = _directory_digest(base_payload)
                        if digest != base.artifact.digest:
                            raise StableShardedSessionError(
                                "stable_base_checkpoint_digest_mismatch"
                            )
                    else:
                        base_payload.mkdir(parents=True, exist_ok=True)
                    if attempt_path.exists():
                        shutil.rmtree(attempt_path)
                    attempt_path.mkdir(parents=True)
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    result_path.unlink(missing_ok=True)
                    _atomic_json(work_path, work.to_dict())
                    _atomic_json(base_path, base.to_dict())
                    bindings = {
                        "CROWDTENSOR_BASE_CHECKPOINT_PATH": str(base_path),
                        "CROWDTENSOR_BASE_PAYLOAD_PATH": str(base_payload),
                        "CROWDTENSOR_MACHINE_RANK": "0",
                        "CROWDTENSOR_MAIN_PROCESS_IP": "127.0.0.1",
                        "CROWDTENSOR_MAIN_PROCESS_PORT": str(_free_loopback_port()),
                        "CROWDTENSOR_OUTPUT_CHECKPOINT_PATH": str(attempt_path),
                        "CROWDTENSOR_TRAINER_RESULT_PATH": str(result_path),
                        "CROWDTENSOR_WORK_UNIT_PATH": str(work_path),
                    }
                    command = _render_command(
                        spec, bindings, command_factory=command_factory
                    )
                    process_result = _run_rank_group(
                        command,
                        workspace=root,
                        log_path=runtime_root / "logs" / f"{attempt_name}.log",
                        timeout_seconds=float(timeout_seconds),
                    )
                    process_reports.append(process_result)
                    code = ""
                    if process_result["start_failed"]:
                        code = "stable_rank_group_start_failed"
                    elif process_result["timed_out"]:
                        code = "stable_rank_group_timeout"
                    elif process_result["return_code"] != 0:
                        code = "stable_rank_group_failed"
                    elif not result_path.is_file():
                        code = "stable_trainer_result_missing"
                    if not code:
                        try:
                            _, verified = _commit_success(
                                controller,
                                project,
                                work,
                                base,
                                owner_hash=owner_hash,
                                result_path=result_path,
                                attempt_path=attempt_path,
                                checkpoint_root=checkpoint_root,
                                expected_rank_count=spec.num_processes,
                            )
                        except StableShardedSessionError as exc:
                            code = str(exc)
                        else:
                            verified_results.append(verified)
                            completed += 1
                            succeeded = True
                            break
                    failure = _failure_receipt(
                        work,
                        owner_hash=owner_hash,
                        code=code,
                        process_result=process_result,
                    )
                    controller.commit(work, failure, base_checkpoint=base)
                    failed += 1
                    if attempt_path.is_dir():
                        shutil.rmtree(attempt_path)
                    result_path.unlink(missing_ok=True)
                    if attempt_index >= spec.max_restarts:
                        final_failure = code
                        break
                    restarts += 1
                    generation = controller.next_generation(work_id)
                if not succeeded:
                    break
        finally:
            final_step = controller.lineage().checkpoints[-1].step
            run_workspace(
                root,
                controller_ready=True,
                execution_started=False,
                execution_complete=final_step >= project.target_steps,
            )

        status = controller.status()
        final_step = controller.lineage().checkpoints[-1].step
        device_types = sorted(
            {str(item["device_type"]) for item in verified_results}
        )
        return _seal(
            {
                "schema": STABLE_SESSION_REPORT_SCHEMA,
                "command_ok": not final_failure,
                "state": (
                    "completed"
                    if final_step >= project.target_steps
                    else "ready" if not final_failure else "failed"
                ),
                "blocker": final_failure or None,
                "project_hash": project.content_hash,
                "plan_hash": plan.content_hash,
                "launch_spec_hash": spec.content_hash,
                "launcher": str(launcher_label),
                "target_steps": project.target_steps,
                "initial_step": initial_step,
                "final_step": final_step,
                "work_units_completed": completed,
                "failed_attempts": failed,
                "restart_count": restarts,
                "recovered_commit_count": recovered,
                "rank_count": spec.num_processes,
                "device_types": device_types,
                "process_output_digests": [
                    item["output_digest"] for item in process_reports
                ],
                "process_output_bytes": sum(
                    int(item["output_bytes"]) for item in process_reports
                ),
                "command_attempted": bool(process_reports),
                "command_executed": any(
                    not item["start_failed"] for item in process_reports
                ),
                "execution_complete": final_step >= project.target_steps,
                "fsdp2_trainer_result_verified": bool(verified_results),
                "cpu_execution_reported": "cpu" in device_types,
                "gpu_execution_reported": "cuda" in device_types,
                "hardware_attestation_verified": False,
                "physical_multi_host_verified": False,
                "materialization_hash": materialization["content_hash"],
                "checkpoint_count": status["checkpoint_count"],
                "terminal_count": status["terminal_count"],
                "credential_values_public": False,
                "private_paths_public": False,
                "public_artifact_safe": True,
            }
        )


__all__ = [
    "STABLE_SESSION_REPORT_SCHEMA",
    "STABLE_TRAINER_RESULT_SCHEMA",
    "StableShardedSessionError",
    "run_stable_sharded_session",
    "validate_stable_trainer_result",
]
