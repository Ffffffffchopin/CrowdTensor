"""Minimal local workspace for the Training-First Architecture v2."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import (
    ArtifactRef,
    ContractError,
    TrainingMode,
    TrainingProject,
    stable_hash,
)


WORKSPACE_STATUS_SCHEMA = "crowdtensor_training_workspace_status_v2"
WORKSPACE_INIT_SCHEMA = "crowdtensor_training_workspace_init_v2"
WORKSPACE_ACTION_SCHEMA = "crowdtensor_training_workspace_action_v2"
WORKSPACE_EXPORT_SCHEMA = "crowdtensor_training_workspace_export_v2"
WORKSPACE_PLAN_SCHEMA = "crowdtensor_training_plan_command_v2"
CONTROL_DIR = ".crowdtensor"
PROJECT_FILE = "project.json"
CONTROL_STATE_FILE = "state/workspace-control.json"
PLAN_FILE = "state/execution-plan.json"

_WORKSPACE_STATES = frozenset(
    {"initialized", "planned", "blocked", "ready", "paused", "running", "completed"}
)
_WORKSPACE_ACTIONS = frozenset(
    {"init", "plan", "run", "join", "status", "pause", "resume", "export"}
)
_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_UNCHANGED = object()


class WorkspaceError(ValueError):
    """Raised when workspace state is missing, unsafe, or conflicting."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a complete public-safe JSON document without partial replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("content_hash", None)
    result["content_hash"] = stable_hash(result)
    return result


def _read_sealed_json(path: Path, *, schema: str, error: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkspaceError(error) from exc
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise WorkspaceError(error)
    supplied = str(value.get("content_hash") or "")
    unsigned = dict(value)
    unsigned.pop("content_hash", None)
    if supplied != stable_hash(unsigned):
        raise WorkspaceError(error)
    return value


def _control_path(root: Path) -> Path:
    return root / CONTROL_DIR / CONTROL_STATE_FILE


def _plan_path(root: Path) -> Path:
    return root / CONTROL_DIR / PLAN_FILE


def _initial_control_state(project: TrainingProject) -> dict[str, Any]:
    return _seal(
        {
            "schema": "crowdtensor_training_workspace_control_v2",
            "project_hash": project.content_hash,
            "state": "initialized",
            "generation": 0,
            "pause_reason": None,
            "last_action": "init",
            "last_plan_hash": None,
            "last_plan_ready": False,
            "blockers": [],
            "updated_at": _utc_now(),
            "command_executed": False,
            "execution_started": False,
            "credential_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
    )


def _validate_control_state(
    value: dict[str, Any], *, project: TrainingProject
) -> dict[str, Any]:
    if value.get("project_hash") != project.content_hash:
        raise WorkspaceError("workspace_control_project_mismatch")
    if value.get("state") not in _WORKSPACE_STATES:
        raise WorkspaceError("workspace_control_state_invalid")
    generation = value.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise WorkspaceError("workspace_control_generation_invalid")
    if value.get("last_action") not in _WORKSPACE_ACTIONS:
        raise WorkspaceError("workspace_control_action_invalid")
    if value.get("last_plan_hash") is not None:
        plan_hash = str(value["last_plan_hash"])
        if not _HASH_PATTERN.fullmatch(plan_hash):
            raise WorkspaceError("workspace_control_plan_hash_invalid")
    if not isinstance(value.get("last_plan_ready"), bool):
        raise WorkspaceError("workspace_control_plan_ready_invalid")
    blockers = value.get("blockers")
    if not isinstance(blockers, list) or any(not isinstance(item, str) for item in blockers):
        raise WorkspaceError("workspace_control_blockers_invalid")
    if value.get("credential_values_public") is not False:
        raise WorkspaceError("workspace_control_public_safety_invalid")
    if value.get("private_paths_public") is not False:
        raise WorkspaceError("workspace_control_public_safety_invalid")
    if value.get("public_artifact_safe") is not True:
        raise WorkspaceError("workspace_control_public_safety_invalid")
    return value


def _load_control_state(
    root: Path, *, project: TrainingProject, create: bool = False
) -> dict[str, Any]:
    path = _control_path(root)
    if not path.is_file():
        state = _initial_control_state(project)
        if create:
            _create_json_exclusive(path, state)
            if path.is_file():
                return _validate_control_state(
                    _read_sealed_json(
                        path,
                        schema="crowdtensor_training_workspace_control_v2",
                        error="workspace_control_state_invalid",
                    ),
                    project=project,
                )
        return state
    return _validate_control_state(
        _read_sealed_json(
            path,
            schema="crowdtensor_training_workspace_control_v2",
            error="workspace_control_state_invalid",
        ),
        project=project,
    )


def _save_control_state(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    sealed = _seal(state)
    _atomic_write_json(_control_path(root), sealed)
    return sealed


def _load_plan(root: Path, *, project: TrainingProject) -> dict[str, Any] | None:
    path = _plan_path(root)
    if not path.is_file():
        return None
    value = _read_sealed_json(
        path, schema=WORKSPACE_PLAN_SCHEMA, error="workspace_execution_plan_invalid"
    )
    if value.get("project_hash") != project.content_hash:
        raise WorkspaceError("workspace_execution_plan_project_mismatch")
    if value.get("public_artifact_safe") is not True:
        raise WorkspaceError("workspace_execution_plan_public_safety_invalid")
    if value.get("private_paths_public") is not False:
        raise WorkspaceError("workspace_execution_plan_public_safety_invalid")
    return value


def load_recorded_plan(workspace: str | Path) -> dict[str, Any] | None:
    """Return the validated public execution plan recorded for a workspace."""

    root = Path(workspace).expanduser().resolve()
    project = load_project(root)
    return _load_plan(root, project=project)


def _set_control_state(
    root: Path,
    *,
    project: TrainingProject,
    action: str,
    state: str | None = None,
    pause_reason: str | None | object = _UNCHANGED,
    blockers: list[str] | None = None,
    last_plan_hash: str | None = None,
    last_plan_ready: bool | None = None,
    increment_generation: bool = False,
) -> dict[str, Any]:
    current = _load_control_state(root, project=project, create=True)
    if action not in _WORKSPACE_ACTIONS:
        raise WorkspaceError("workspace_control_action_invalid")
    updated = dict(current)
    if state is not None:
        if state not in _WORKSPACE_STATES:
            raise WorkspaceError("workspace_control_state_invalid")
        updated["state"] = state
    if increment_generation:
        updated["generation"] = int(current["generation"]) + 1
    if pause_reason is not _UNCHANGED:
        updated["pause_reason"] = pause_reason
    updated["last_action"] = action
    if blockers is not None:
        updated["blockers"] = sorted({str(item) for item in blockers if str(item)})
    if last_plan_hash is not None or last_plan_ready is not None:
        updated["last_plan_hash"] = last_plan_hash
        updated["last_plan_ready"] = bool(last_plan_ready)
    updated["updated_at"] = _utc_now()
    return _save_control_state(root, updated)


def _action_report(
    *,
    project: TrainingProject,
    action: str,
    previous_state: str,
    control: dict[str, Any],
    blockers: list[str] | tuple[str, ...] = (),
    command_ok: bool = True,
    command_executed: bool = False,
    execution_started: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": WORKSPACE_ACTION_SCHEMA,
        "command_ok": bool(command_ok),
        "action": action,
        "project_hash": project.content_hash,
        "previous_state": previous_state,
        "state": control["state"],
        "generation": control["generation"],
        "blockers": sorted({str(item) for item in blockers if str(item)}),
        "command_executed": bool(command_executed),
        "execution_started": bool(execution_started),
        "credential_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    if extra:
        payload.update(extra)
    return _seal(payload)


def record_plan(workspace: str | Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Persist a validated planner result and update the local lifecycle state."""

    root = Path(workspace).expanduser().resolve()
    project = load_project(root)
    if not isinstance(plan, dict) or plan.get("schema") != WORKSPACE_PLAN_SCHEMA:
        raise WorkspaceError("workspace_execution_plan_invalid")
    supplied_hash = str(plan.get("content_hash") or "")
    unsigned = dict(plan)
    unsigned.pop("content_hash", None)
    if supplied_hash != stable_hash(unsigned):
        raise WorkspaceError("workspace_execution_plan_invalid")
    if plan.get("project_hash") != project.content_hash:
        raise WorkspaceError("workspace_execution_plan_project_mismatch")
    if plan.get("command_ok") is not True:
        raise WorkspaceError("workspace_execution_plan_command_failed")
    if plan.get("private_paths_public") is not False:
        raise WorkspaceError("workspace_execution_plan_public_safety_invalid")
    if plan.get("credential_values_public") is not False:
        raise WorkspaceError("workspace_execution_plan_public_safety_invalid")
    if plan.get("public_artifact_safe") is not True:
        raise WorkspaceError("workspace_execution_plan_public_safety_invalid")
    _atomic_write_json(_plan_path(root), plan)
    execution_ready = bool(plan.get("execution_ready"))
    blockers = list(plan.get("plan", {}).get("blockers") or [])
    current = _load_control_state(root, project=project, create=True)
    control = _set_control_state(
        root,
        project=project,
        action="plan",
        state=(
            "paused"
            if current["state"] == "paused"
            else "ready" if execution_ready else "blocked"
        ),
        blockers=blockers,
        last_plan_hash=str(plan["content_hash"]),
        last_plan_ready=execution_ready,
    )
    return _action_report(
        project=project,
        action="plan",
        previous_state=str(current["state"]),
        control=control,
        blockers=blockers,
        extra={
            "plan_hash": plan["content_hash"],
            "execution_ready": execution_ready,
        },
    )


def _plan_state(root: Path, project: TrainingProject) -> tuple[dict[str, Any] | None, list[str]]:
    plan = _load_plan(root, project=project)
    if plan is None:
        return None, ["execution_plan_required"]
    blockers = [str(item) for item in plan.get("plan", {}).get("blockers") or []]
    if not bool(plan.get("execution_ready")):
        return plan, sorted(set(blockers or ["execution_plan_not_ready"]))
    return plan, []


def run_workspace(
    workspace: str | Path,
    *,
    controller_ready: bool = False,
    execution_started: bool = False,
    execution_complete: bool = False,
) -> dict[str, Any]:
    """Record controller readiness, or fail closed before execution exists."""

    root = Path(workspace).expanduser().resolve()
    project = load_project(root)
    current = _load_control_state(root, project=project, create=True)
    if current["state"] == "paused":
        control = _set_control_state(
            root,
            project=project,
            action="run",
            state="paused",
            blockers=["workspace_paused"],
        )
        return _action_report(
            project=project,
            action="run",
            previous_state="paused",
            control=control,
            blockers=["workspace_paused"],
            command_ok=False,
        )
    if controller_ready:
        if (
            project.mode is TrainingMode.ELASTIC_DELTA
            and project.training_backend != "volunteer_peft"
        ):
            blockers = ["elastic_controller_backend_not_supported"]
        elif (
            project.mode is TrainingMode.STABLE_SHARDED
            and project.training_backend != "accelerate_fsdp2"
        ):
            blockers = ["stable_controller_backend_not_supported"]
        else:
            from .controller import inspect_session_controller

            session = inspect_session_controller(root)
            blockers = (
                []
                if session is not None and session.get("initialized") is True
                else ["v2_session_controller_required"]
            )
        if blockers:
            control = _set_control_state(
                root,
                project=project,
                action="run",
                state="blocked",
                blockers=blockers,
            )
            return _action_report(
                project=project,
                action="run",
                previous_state=str(current["state"]),
                control=control,
                blockers=blockers,
                command_ok=False,
            )
        control = _set_control_state(
            root,
            project=project,
            action="run",
            state=(
                "completed"
                if execution_complete
                else "running" if execution_started else "ready"
            ),
            blockers=[],
        )
        return _action_report(
            project=project,
            action="run",
            previous_state=str(current["state"]),
            control=control,
            command_executed=execution_started,
            execution_started=execution_started,
            extra={
                "controller_ready": True,
                "controller_owned_by_session_user": True,
                "execution_kind": (
                    "volunteer_campaign_controller"
                    if project.mode is TrainingMode.ELASTIC_DELTA
                    else "stable_rank_group_controller"
                ),
                "execution_complete": bool(execution_complete),
            },
        )
    plan, blockers = _plan_state(root, project)
    if blockers:
        control = _set_control_state(
            root,
            project=project,
            action="run",
            state="blocked",
            blockers=blockers,
        )
        return _action_report(
            project=project,
            action="run",
            previous_state=str(current["state"]),
            control=control,
            blockers=blockers,
            command_ok=False,
            extra={"plan_hash": plan.get("content_hash") if plan else None},
        )
    control = _set_control_state(
        root,
        project=project,
        action="run",
        state="blocked",
        blockers=["v2_controller_execution_pending"],
    )
    return _action_report(
        project=project,
        action="run",
        previous_state=str(current["state"]),
        control=control,
        blockers=["v2_controller_execution_pending"],
        command_ok=False,
        extra={
            "plan_hash": plan["content_hash"],
            "execution_ready": True,
        },
    )


def join_workspace(
    workspace: str | Path,
    *,
    admission_ready: bool = False,
    command_executed: bool = False,
    campaign_complete: bool = False,
    completed_work_units: int = 0,
    last_state: str = "",
) -> dict[str, Any]:
    """Record a bounded contributor admission or report its fail-closed boundary."""

    root = Path(workspace).expanduser().resolve()
    project = load_project(root)
    current = _load_control_state(root, project=project, create=True)
    if current["state"] == "paused":
        blocker = "workspace_paused"
    elif project.mode is not TrainingMode.ELASTIC_DELTA:
        blocker = "stable_sharded_join_not_supported"
    elif project.training_backend != "volunteer_peft":
        blocker = "elastic_join_backend_not_supported"
    elif not admission_ready:
        blocker = "v2_controller_join_pending"
    else:
        blocker = ""
    if not blocker:
        completed = int(completed_work_units)
        if completed < 0:
            raise WorkspaceError("workspace_completed_work_units_invalid")
        control = _set_control_state(
            root,
            project=project,
            action="join",
            state="completed" if campaign_complete else "ready",
            blockers=[],
        )
        return _action_report(
            project=project,
            action="join",
            previous_state=str(current["state"]),
            control=control,
            command_executed=command_executed,
            execution_started=command_executed,
            extra={
                "mode": project.mode.value,
                "admission_ready": True,
                "completed_work_units": completed,
                "last_contributor_state": str(last_state or "ready"),
                "controller_authority": "remote_session_owner",
            },
        )
    control = _set_control_state(
        root,
        project=project,
        action="join",
        state=str(current["state"]),
        blockers=[blocker],
    )
    return _action_report(
        project=project,
        action="join",
        previous_state=str(current["state"]),
        control=control,
        blockers=[blocker],
        command_ok=False,
        extra={"mode": project.mode.value},
    )


def pause_workspace(workspace: str | Path, *, reason: str = "owner_paused") -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    project = load_project(root)
    current = _load_control_state(root, project=project, create=True)
    if current["state"] == "completed":
        raise WorkspaceError("workspace_pause_completed_forbidden")
    control = _set_control_state(
        root,
        project=project,
        action="pause",
        state="paused",
        pause_reason=str(reason or "owner_paused"),
        increment_generation=current["state"] != "paused",
    )
    return _action_report(
        project=project,
        action="pause",
        previous_state=str(current["state"]),
        control=control,
        extra={"pause_reason": control["pause_reason"]},
    )


def resume_workspace(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    project = load_project(root)
    current = _load_control_state(root, project=project, create=True)
    plan, blockers = _plan_state(root, project)
    next_state = "initialized" if plan is None else "ready" if not blockers else "blocked"
    control = _set_control_state(
        root,
        project=project,
        action="resume",
        state=next_state,
        pause_reason=None,
        blockers=blockers,
        increment_generation=current["state"] == "paused",
    )
    return _action_report(
        project=project,
        action="resume",
        previous_state=str(current["state"]),
        control=control,
        blockers=blockers,
        extra={
            "plan_hash": plan.get("content_hash") if plan else None,
            "execution_ready": bool(plan and plan.get("execution_ready")),
        },
    )


def _copy_public_json(source: Path, destination: Path) -> None:
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkspaceError("workspace_export_json_invalid") from exc
    if not isinstance(payload, dict) or payload.get("public_artifact_safe") is not True:
        raise WorkspaceError("workspace_export_private_artifact_forbidden")
    if (
        payload.get("private_paths_public") is True
        or payload.get("credential_values_public") is True
    ):
        raise WorkspaceError("workspace_export_private_artifact_forbidden")
    supplied_hash = str(payload.get("content_hash") or "")
    unsigned = dict(payload)
    unsigned.pop("content_hash", None)
    if not _HASH_PATTERN.fullmatch(supplied_hash) or supplied_hash != stable_hash(unsigned):
        raise WorkspaceError("workspace_export_json_hash_invalid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(destination, payload)


def export_workspace(
    workspace: str | Path, output_dir: str | Path | None = None
) -> dict[str, Any]:
    """Export only public contract JSON, never model weights or private state."""

    root = Path(workspace).expanduser().resolve()
    project = load_project(root)
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else root.parent / f"{root.name}-export"
    )
    if destination == root or root in destination.parents:
        raise WorkspaceError("workspace_export_destination_inside_workspace")
    if destination.is_file() or (destination.is_dir() and any(destination.iterdir())):
        raise WorkspaceError("workspace_export_destination_not_empty")
    destination.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    sources = [(root / CONTROL_DIR / PROJECT_FILE, Path("project.json"))]
    control_path = _control_path(root)
    if control_path.is_file():
        sources.append((control_path, Path("workspace-control.json")))
    plan_path = _plan_path(root)
    if plan_path.is_file():
        sources.append((plan_path, Path("execution-plan.json")))
    for relative_root in ("receipts", "checkpoints"):
        source_root = root / CONTROL_DIR / relative_root
        if source_root.is_dir():
            for source in sorted(source_root.glob("*.json")):
                sources.append((source, Path(relative_root) / source.name))
    for source, relative in sources:
        _copy_public_json(source, destination / relative)
        files.append(relative.as_posix())
    session_controller_path = root / CONTROL_DIR / "state/session-controller.json"
    if session_controller_path.is_file():
        from .controller import inspect_session_controller

        session = inspect_session_controller(root)
        if session is None:
            raise WorkspaceError("workspace_session_controller_missing")
        _atomic_write_json(destination / "session-controller.json", session)
        files.append("session-controller.json")
    status_path = destination / "status.json"
    _atomic_write_json(status_path, inspect_workspace(root))
    files.append("status.json")
    report = _seal(
        {
            "schema": WORKSPACE_EXPORT_SCHEMA,
            "command_ok": True,
            "project_hash": project.content_hash,
            "artifact_count": len(files),
            "artifacts": files,
            "weights_exported": False,
            "credentials_exported": False,
            "private_runtime_exported": False,
            "public_artifact_safe": True,
            "private_paths_public": False,
        }
    )
    _atomic_write_json(destination / "training-workspace-export.json", report)
    return report


def _create_json_exclusive(path: Path, payload: dict[str, Any]) -> bool:
    """Publish complete JSON only when no initializer won the race."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _default_project_id(root: Path) -> str:
    value = re.sub(r"[^a-z0-9._-]+", "-", root.name.lower()).strip("-._")
    if not value:
        value = "training-project"
    if len(value) == 1:
        value = f"project-{value}"
    return value[:128].rstrip("-._")


def _project_path(root: Path) -> Path:
    return root / CONTROL_DIR / PROJECT_FILE


def init_project(
    workspace: str | Path,
    *,
    model: str,
    model_revision: str,
    dataset: str,
    dataset_revision: str,
    model_adapter: str,
    training_backend: str = "volunteer_peft",
    mode: TrainingMode | str = TrainingMode.ELASTIC_DELTA,
    target_steps: int = 100,
    project_id: str | None = None,
    optimization_plugins: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Create an idempotent v2 project manifest and stable directory layout."""

    root = Path(workspace).expanduser().resolve()
    project = TrainingProject(
        project_id=project_id or _default_project_id(root),
        mode=TrainingMode(mode),
        model=ArtifactRef(model, model_revision),
        dataset=ArtifactRef(dataset, dataset_revision),
        model_adapter=model_adapter,
        training_backend=training_backend,
        target_steps=target_steps,
        optimization_plugins=optimization_plugins,
    )
    return init_project_contract(root, project)


def init_project_contract(
    workspace: str | Path, project: TrainingProject
) -> dict[str, Any]:
    """Create a workspace from an already validated migration contract."""

    if not isinstance(project, TrainingProject):
        raise WorkspaceError("workspace_training_project_required")
    root = Path(workspace).expanduser().resolve()
    path = _project_path(root)
    created = False
    if path.exists():
        existing = load_project(root)
        if existing.content_hash != project.content_hash:
            raise WorkspaceError("workspace_project_manifest_conflict")
    else:
        root.mkdir(parents=True, exist_ok=True)
    for relative in ("checkpoints", "receipts", "state"):
        (root / CONTROL_DIR / relative).mkdir(parents=True, exist_ok=True)
    if not path.exists():
        created = _create_json_exclusive(path, project.to_dict())
    if not created:
        existing = load_project(root)
        if existing.content_hash != project.content_hash:
            raise WorkspaceError("workspace_project_manifest_conflict")
    _load_control_state(root, project=project, create=True)
    status = inspect_workspace(root)
    result = {
        "schema": WORKSPACE_INIT_SCHEMA,
        "created": created,
        "command_ok": True,
        "status": status,
        "public_artifact_safe": True,
    }
    result["content_hash"] = stable_hash(result)
    return result


def load_project(workspace: str | Path) -> TrainingProject:
    root = Path(workspace).expanduser().resolve()
    path = _project_path(root)
    if not path.is_file():
        raise WorkspaceError("workspace_project_manifest_missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return TrainingProject.from_dict(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ContractError) as exc:
        raise WorkspaceError("workspace_project_manifest_invalid") from exc


def inspect_workspace(workspace: str | Path) -> dict[str, Any]:
    """Return public-safe, path-independent status for a v2 workspace."""

    root = Path(workspace).expanduser().resolve()
    project = load_project(root)
    control_state = _load_control_state(root, project=project, create=True)
    plan = _load_plan(root, project=project)
    control = root / CONTROL_DIR
    required_directories = tuple(
        control / item for item in ("checkpoints", "receipts", "state")
    )
    layout_ready = all(path.is_dir() for path in required_directories)
    checkpoint_count = sum(1 for _ in (control / "checkpoints").glob("*.json"))
    receipt_count = sum(1 for _ in (control / "receipts").glob("*.json"))
    from .controller import inspect_session_controller

    session_controller = inspect_session_controller(root)
    lifecycle_state = str(control_state["state"])
    if not layout_ready:
        execution_state = "invalid_layout"
    else:
        execution_state = lifecycle_state
    payload: dict[str, Any] = {
        "schema": WORKSPACE_STATUS_SCHEMA,
        "project": project.to_dict(),
        "project_hash": project.content_hash,
        "mode": project.mode.value,
        "training_backend": project.training_backend,
        "checkpoint_count": checkpoint_count,
        "receipt_count": receipt_count,
        "session_controller": session_controller,
        "execution_state": execution_state,
        "lifecycle_state": lifecycle_state,
        "generation": int(control_state["generation"]),
        "pause_reason": control_state["pause_reason"],
        "last_action": control_state["last_action"],
        "last_plan_hash": control_state["last_plan_hash"],
        "last_plan_ready": bool(control_state["last_plan_ready"]),
        "plan_present": plan is not None,
        "plan_execution_ready": bool(plan and plan.get("execution_ready")),
        "workspace_layout_ready": layout_ready,
        "workspace_layout": {
            "manifest": f"{CONTROL_DIR}/{PROJECT_FILE}",
            "checkpoints": f"{CONTROL_DIR}/checkpoints",
            "receipts": f"{CONTROL_DIR}/receipts",
            "state": f"{CONTROL_DIR}/state",
        },
        "blockers": sorted(
            set(
                ([] if layout_ready else ["workspace_layout_incomplete"])
                + list(control_state.get("blockers") or [])
            )
        ),
        "framework_runtime_required_for_inspection": False,
        "framework_runtime_probed": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    payload["content_hash"] = stable_hash(payload)
    return payload
