"""Ordinary v2 operator and contributor workflow over Volunteer PEFT."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import secrets
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from crowdtensor.community_security import TLSProxyPolicy
from crowdtensor.core.contracts import stable_hash
from crowdtensor.core.workspace import (
    CONTROL_DIR,
    init_project_contract,
    join_workspace,
    run_workspace,
)
from crowdtensor.volunteer_training_api import create_volunteer_training_app
from crowdtensor.volunteer_agent_status import (
    VolunteerAgentStatusServer,
    graceful_agent_signals,
)
from crowdtensor.volunteer_training_cell import (
    HTTPVolunteerTransport,
    LocalVolunteerTransport,
    VolunteerTrainingCell,
    detect_hardware,
)
from crowdtensor.volunteer_training_coordinator import VolunteerTrainingCoordinator

from .elastic_peft import VolunteerControllerTransport, project_from_campaign


VOLUNTEER_SESSION_REPORT_SCHEMA = "crowdtensor_volunteer_session_v2"
VOLUNTEER_JOIN_REPORT_SCHEMA = "crowdtensor_volunteer_join_v2"


class VolunteerSessionError(ValueError):
    """A public-safe v2 session workflow error."""


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value.pop("content_hash", None)
    value["content_hash"] = stable_hash(value)
    return value


def _loopback_host(host: str) -> bool:
    value = str(host or "").strip().strip("[]").lower()
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _validate_coordinator_url(value: str) -> str:
    normalized = str(value or "").strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise VolunteerSessionError("volunteer_session_coordinator_url_invalid")
    try:
        parsed_port = parsed.port
    except ValueError as exc:
        raise VolunteerSessionError("volunteer_session_coordinator_url_invalid") from exc
    if parsed_port is not None and (parsed_port < 1 or parsed_port > 65535):
        raise VolunteerSessionError("volunteer_session_coordinator_url_invalid")
    if (
        parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise VolunteerSessionError("volunteer_session_coordinator_url_invalid")
    if parsed.scheme == "http" and not _loopback_host(parsed.hostname):
        raise VolunteerSessionError("volunteer_session_public_https_required")
    return normalized


def _default_public_url(host: str, port: int) -> str:
    normalized = str(host).strip()
    if not _loopback_host(normalized):
        raise VolunteerSessionError("volunteer_session_public_url_required")
    display_host = f"[{normalized}]" if ":" in normalized else normalized
    return f"http://{display_host}:{int(port)}"


def _worker_root(workspace: str | Path) -> Path:
    return Path(workspace).expanduser().resolve() / CONTROL_DIR / "contributor"


def _cell_id(worker: Path, requested: str) -> str:
    state_path = worker / ".private" / "cell_state.json"
    if state_path.is_file():
        if state_path.stat().st_mode & 0o077:
            raise VolunteerSessionError("volunteer_session_cell_state_permissions_invalid")
        try:
            value = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise VolunteerSessionError("volunteer_session_cell_state_invalid") from exc
        existing = str(value.get("cell_id") or "") if isinstance(value, dict) else ""
        if not existing:
            raise VolunteerSessionError("volunteer_session_cell_state_invalid")
        if requested and str(requested) != existing:
            raise VolunteerSessionError("volunteer_session_cell_id_conflict")
        return existing
    return str(requested or "cell-" + secrets.token_hex(12))


def _enrollment_path(worker: Path) -> Path:
    return worker / ".private" / "agent_enrollment.json"


def _load_enrollment(
    worker: Path, *, coordinator_url: str, cell_id: str
) -> dict[str, Any] | None:
    path = _enrollment_path(worker)
    if not path.is_file() or path.stat().st_mode & 0o077:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    try:
        expires_at = float(value.get("expires_at") or 0.0)
    except (TypeError, ValueError):
        return None
    credential = value.get("credential_token")
    if (
        value.get("schema") != "crowdtensor_volunteer_agent_enrollment_v1"
        or str(value.get("coordinator_url") or "").rstrip("/") != coordinator_url
        or value.get("cell_id") != cell_id
        or not math.isfinite(expires_at)
        or expires_at <= time.time() + 5.0
        or not isinstance(credential, str)
        or not credential
    ):
        return None
    return value


def _save_enrollment(worker: Path, value: dict[str, Any]) -> None:
    path = _enrollment_path(worker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class PreparedVolunteerSession:
    """In-process service objects; never serialize this container."""

    coordinator: VolunteerTrainingCoordinator
    controller_transport: VolunteerControllerTransport
    app: Any
    report: dict[str, Any]


def prepare_volunteer_session(
    workspace: str | Path,
    *,
    campaign_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 8789,
    public_url: str = "",
    require_https: bool | None = None,
    trust_forwarded_headers: bool = False,
    trusted_proxy_id: str = "",
    upload_chunk_bytes: int = 1024 * 1024,
    public_release_dir: str | Path | None = None,
    recover: bool = True,
) -> PreparedVolunteerSession:
    """Bind an existing Campaign to a user-owned v2 Session Controller."""

    if int(port) < 1 or int(port) > 65535:
        raise VolunteerSessionError("volunteer_session_port_invalid")
    if int(upload_chunk_bytes) < 64 * 1024 or int(upload_chunk_bytes) > 64 * 1024**2:
        raise VolunteerSessionError("volunteer_session_upload_chunk_bytes_invalid")
    root = Path(campaign_dir).expanduser().resolve()
    coordinator = VolunteerTrainingCoordinator(root)
    validation = coordinator.validate_campaign()
    if validation.get("ok") is not True:
        raise VolunteerSessionError("volunteer_session_campaign_invalid")
    url = _validate_coordinator_url(
        public_url or _default_public_url(host, int(port))
    )
    external_https = url.startswith("https://")
    tls_required = external_https if require_https is None else bool(require_https)
    if not tls_required and external_https:
        raise VolunteerSessionError("volunteer_session_https_policy_mismatch")
    if tls_required and url.startswith("http://"):
        raise VolunteerSessionError("volunteer_session_https_policy_mismatch")
    proxy_id = str(trusted_proxy_id or "")
    if bool(trust_forwarded_headers) != bool(proxy_id):
        raise VolunteerSessionError("volunteer_session_trusted_proxy_identity_required")
    if external_https and not trust_forwarded_headers:
        raise VolunteerSessionError("volunteer_session_tls_termination_required")
    proxy_hashes = (
        ("sha256:" + hashlib.sha256(proxy_id.encode("utf-8")).hexdigest(),)
        if proxy_id
        else ()
    )
    token = coordinator.private_invite()["invite_token"]
    controller_transport = VolunteerControllerTransport(
        LocalVolunteerTransport(coordinator, token), workspace
    )
    lifecycle = run_workspace(
        workspace, controller_ready=True, execution_started=False
    )
    if lifecycle.get("command_ok") is not True:
        raise VolunteerSessionError("volunteer_session_controller_not_ready")
    coordinator.write_invite(url)
    recovery = (
        coordinator.recover_after_restart()
        if recover
        else {
            "ok": True,
            "coordinator_state_reloaded": False,
            "prepare_only": True,
        }
    )
    policy = TLSProxyPolicy(
        require_https=tls_required,
        trust_forwarded_headers=bool(trust_forwarded_headers),
        trusted_proxy_hashes=proxy_hashes,
    )
    app = create_volunteer_training_app(
        coordinator,
        controller_transport=controller_transport,
        tls_policy=policy,
        upload_chunk_bytes=int(upload_chunk_bytes),
        public_release_dir=public_release_dir,
    )
    manifest = coordinator.campaign_manifest()
    controller = controller_transport.controller.status()
    report = _seal(
        {
            "schema": VOLUNTEER_SESSION_REPORT_SCHEMA,
            "command_ok": True,
            "state": "prepared",
            "project_hash": controller_transport.project.content_hash,
            "campaign_id": manifest["campaign_id"],
            "campaign_manifest_hash": manifest["manifest_hash"],
            "coordinator_url": url,
            "bind_host": str(host),
            "port": int(port),
            "tls_required": tls_required,
            "trusted_forwarded_headers": bool(trust_forwarded_headers),
            "private_invite_updated": True,
            "private_invite_path_public": False,
            "controller_owned_by_session_user": True,
            "controller_revision": 2,
            "concurrent_elastic_work_supported": True,
            "active_work_count": int(controller["active_work_count"]),
            "checkpoint_count": int(controller["checkpoint_count"]),
            "coordinator_restart_recovery_performed": bool(recover),
            "coordinator_restart_recovery_verified": bool(
                recover and recovery.get("ok") is True
            ),
            "execution_started": False,
            "public_release_download": bool(app.state.public_release_download),
            "credential_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
    )
    return PreparedVolunteerSession(
        coordinator=coordinator,
        controller_transport=controller_transport,
        app=app,
        report=report,
    )


def run_volunteer_session(
    workspace: str | Path,
    *,
    campaign_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 8789,
    public_url: str = "",
    prepare_only: bool = False,
    require_https: bool | None = None,
    trust_forwarded_headers: bool = False,
    trusted_proxy_id: str = "",
    upload_chunk_bytes: int = 1024 * 1024,
    public_release_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare or serve one user-owned v2 elastic training session."""

    prepared = prepare_volunteer_session(
        workspace,
        campaign_dir=campaign_dir,
        host=host,
        port=port,
        public_url=public_url,
        require_https=require_https,
        trust_forwarded_headers=trust_forwarded_headers,
        trusted_proxy_id=trusted_proxy_id,
        upload_chunk_bytes=upload_chunk_bytes,
        public_release_dir=public_release_dir,
        recover=not prepare_only,
    )
    if prepare_only:
        return prepared.report
    started = run_workspace(
        workspace, controller_ready=True, execution_started=True
    )
    if started.get("command_ok") is not True:
        raise VolunteerSessionError("volunteer_session_controller_not_ready")
    try:
        import uvicorn

        uvicorn.run(
            prepared.app,
            host=str(host),
            port=int(port),
            log_level="info",
        )
    finally:
        stopped = run_workspace(
            workspace, controller_ready=True, execution_started=False
        )
    return _seal(
        {
            **prepared.report,
            "state": "stopped",
            "execution_started": True,
            "server_stopped": True,
            "workspace_state": stopped["state"],
        }
    )


def _http_transport(
    worker: Path,
    *,
    invite: str,
    coordinator_url: str,
    pairing_code: str,
    cell_id: str,
    timeout_seconds: float,
) -> HTTPVolunteerTransport:
    if invite:
        transport = HTTPVolunteerTransport.from_invite(
            invite, timeout_seconds=float(timeout_seconds)
        )
        _validate_coordinator_url(transport.coordinator_url)
        return transport
    url = _validate_coordinator_url(coordinator_url)
    enrollment = _load_enrollment(
        worker, coordinator_url=url, cell_id=cell_id
    )
    if enrollment is not None:
        return HTTPVolunteerTransport.from_cell_credential(
            url,
            cell_id=cell_id,
            credential_token=str(enrollment["credential_token"]),
            credential_id=str(enrollment.get("credential_id") or ""),
            expires_at=float(enrollment["expires_at"]),
            timeout_seconds=float(timeout_seconds),
        )
    if not pairing_code:
        raise VolunteerSessionError("volunteer_session_pairing_code_required")
    transport = HTTPVolunteerTransport.from_pairing_code(
        url,
        pairing_code,
        cell_id=cell_id,
        timeout_seconds=float(timeout_seconds),
    )
    _save_enrollment(worker, transport.private_enrollment())
    return transport


def _public_document(
    coordinator_url: str, route: str, *, timeout_seconds: float
) -> dict[str, Any]:
    response = httpx.get(
        coordinator_url + route,
        timeout=float(timeout_seconds),
    )
    return HTTPVolunteerTransport._response(response)


def _select_device(
    requested: str, hardware: dict[str, Any], requirements: dict[str, Any] | None
) -> str:
    policy = str(requested or "auto").lower()
    required_memory = int((requirements or {}).get("minimum_memory_bytes") or 0)
    cuda_available = bool(hardware.get("cuda_available"))
    cuda_count = int(hardware.get("cuda_device_count") or 0)
    cuda_free = [int(value) for value in hardware.get("cuda_memory_available_bytes") or []]
    if policy == "auto":
        if cuda_available and cuda_count:
            if not required_memory or (cuda_free and cuda_free[0] >= required_memory):
                return "cuda:0"
        return "cpu"
    if policy == "cpu":
        return "cpu"
    if policy == "cuda":
        policy = "cuda:0"
    if policy.startswith("cuda:"):
        try:
            index = int(policy.split(":", 1)[1])
        except ValueError as exc:
            raise VolunteerSessionError("volunteer_session_device_invalid") from exc
        if index < 0 or not cuda_available or index >= cuda_count:
            raise VolunteerSessionError("volunteer_session_cuda_device_unavailable")
        return f"cuda:{index}"
    raise VolunteerSessionError("volunteer_session_device_invalid")


def _resource_preflight(
    *,
    campaign: dict[str, Any],
    hardware: dict[str, Any],
    selected_device: str,
    max_local_steps: int,
    max_download_bytes: int,
    worker: Path,
) -> dict[str, Any]:
    requirements = campaign.get("resource_requirements")
    if not isinstance(requirements, dict):
        return {
            "requirements_known": False,
            "resource_ready": True,
            "blockers": [],
            "requirements": None,
        }
    blockers: list[str] = []
    device_kind = "cuda" if selected_device.startswith("cuda:") else "cpu"
    if device_kind not in requirements.get("supported_devices", []):
        blockers.append("campaign_device_unsupported")
    if int(requirements["local_steps"]) > int(max_local_steps):
        blockers.append("campaign_local_steps_exceed_limit")
    if int(requirements["first_work_unit_download_bytes"]) > int(max_download_bytes):
        blockers.append("campaign_download_exceeds_limit")
    required_memory = int(requirements["minimum_memory_bytes"])
    if device_kind == "cuda":
        index = int(selected_device.split(":", 1)[1])
        available_values = hardware.get("cuda_memory_available_bytes") or []
        available_memory = (
            int(available_values[index]) if index < len(available_values) else 0
        )
        memory_capacity = available_memory
    else:
        available_memory = int(
            hardware.get("memory_available_bytes")
            or hardware.get("memory_bytes")
            or 0
        )
        memory_capacity = int(hardware.get("memory_bytes") or available_memory)
    if memory_capacity and memory_capacity < required_memory:
        blockers.append("campaign_memory_insufficient")
    try:
        available_disk = int(shutil.disk_usage(worker.parent).free)
    except OSError:
        available_disk = 0
    if available_disk and available_disk < int(requirements["minimum_free_disk_bytes"]):
        blockers.append("campaign_disk_insufficient")
    return {
        "requirements_known": True,
        "resource_ready": not blockers,
        "blockers": blockers,
        "requirements": requirements,
        "available_memory_bytes": available_memory,
        "memory_capacity_bytes": memory_capacity,
        "available_disk_bytes": available_disk,
    }


def join_volunteer_session(
    workspace: str | Path,
    *,
    invite: str = "",
    coordinator_url: str = "",
    pairing_code: str = "",
    cell_id: str = "",
    device: str = "auto",
    max_local_steps: int = 64,
    max_download_bytes: int = 8 * 1024**3,
    max_work_units: int = 1,
    poll_interval_seconds: float = 2.0,
    timeout_seconds: float = 120.0,
    cache_dir: str | Path | None = None,
    status_page: bool = True,
    status_port: int = 8765,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Join one remote Session and delegate bounded work to the PEFT Cell."""

    if bool(invite) == bool(coordinator_url):
        raise VolunteerSessionError("volunteer_session_enrollment_source_required")
    if pairing_code and not coordinator_url:
        raise VolunteerSessionError("volunteer_session_pairing_code_without_url")
    if int(max_local_steps) < 1 or int(max_local_steps) > 1024:
        raise VolunteerSessionError("volunteer_session_max_local_steps_invalid")
    if int(max_download_bytes) < 1:
        raise VolunteerSessionError("volunteer_session_max_download_invalid")
    if int(max_work_units) < 0:
        raise VolunteerSessionError("volunteer_session_max_work_units_invalid")
    if bool(status_page) and (int(status_port) < 0 or int(status_port) > 65535):
        raise VolunteerSessionError("volunteer_session_status_port_invalid")
    if (
        not math.isfinite(float(poll_interval_seconds))
        or not math.isfinite(float(timeout_seconds))
        or float(poll_interval_seconds) <= 0
        or float(timeout_seconds) <= 0
    ):
        raise VolunteerSessionError("volunteer_session_timeout_invalid")
    worker = _worker_root(workspace)
    selected_cell_id = _cell_id(worker, cell_id)
    transport: HTTPVolunteerTransport | None = None
    if coordinator_url:
        url = _validate_coordinator_url(coordinator_url)
        health = _public_document(
            url, "/v1/volunteer/health", timeout_seconds=timeout_seconds
        )
        if (
            health.get("ok") is not True
            or health.get("v2_session_controller") is not True
            or health.get("concurrent_elastic_work") is not True
        ):
            raise VolunteerSessionError("volunteer_session_v2_controller_required")
        campaign = _public_document(
            url, "/v1/volunteer/campaign", timeout_seconds=timeout_seconds
        )
        before = _public_document(
            url, "/v1/volunteer/status", timeout_seconds=timeout_seconds
        )
    else:
        transport = _http_transport(
            worker,
            invite=str(invite or ""),
            coordinator_url=str(coordinator_url or ""),
            pairing_code=str(pairing_code or ""),
            cell_id=selected_cell_id,
            timeout_seconds=float(timeout_seconds),
        )
        health = transport.health()
        campaign = transport.campaign()
        before = transport.status()
    if (
        health.get("ok") is not True
        or health.get("v2_session_controller") is not True
        or health.get("concurrent_elastic_work") is not True
    ):
        raise VolunteerSessionError("volunteer_session_v2_controller_required")
    project = project_from_campaign(campaign)
    init_project_contract(workspace, project)
    hardware = detect_hardware()
    requirements = campaign.get("resource_requirements")
    selected_device = _select_device(
        str(device), hardware, requirements if isinstance(requirements, dict) else None
    )
    resource_preflight = _resource_preflight(
        campaign=campaign,
        hardware=hardware,
        selected_device=selected_device,
        max_local_steps=int(max_local_steps),
        max_download_bytes=int(max_download_bytes),
        worker=worker,
    )
    if not dry_run and not resource_preflight["resource_ready"]:
        raise VolunteerSessionError("volunteer_session_resource_preflight_failed")
    if transport is None and not dry_run:
        transport = _http_transport(
            worker,
            invite="",
            coordinator_url=str(coordinator_url),
            pairing_code=str(pairing_code or ""),
            cell_id=selected_cell_id,
            timeout_seconds=float(timeout_seconds),
        )
        authenticated_campaign = transport.campaign()
        if authenticated_campaign.get("manifest_hash") != campaign.get("manifest_hash"):
            raise VolunteerSessionError("volunteer_session_campaign_changed")
    cell = VolunteerTrainingCell(
        transport if transport is not None else object(),
        worker,
        cell_id=selected_cell_id,
        device=selected_device,
        max_local_steps=int(max_local_steps),
        max_download_bytes=int(max_download_bytes),
        cache_dir=cache_dir,
    )
    if cell.selected_device() != selected_device:
        raise VolunteerSessionError("volunteer_session_device_selection_changed")
    local_status_endpoint = ""
    if dry_run:
        result = {
            "ok": bool(resource_preflight["resource_ready"]),
            "completed_in_run": 0,
            "last_state": (
                "preflight_ready"
                if resource_preflight["resource_ready"]
                else "preflight_blocked"
            ),
        }
    else:
        if transport is None:
            raise VolunteerSessionError("volunteer_session_transport_required")
        if status_page:
            with VolunteerAgentStatusServer(cell, port=int(status_port)) as control:
                local_status_endpoint = control.endpoint
                with graceful_agent_signals(control.stop_event):
                    result = cell.run(
                        max_work_units=int(max_work_units),
                        poll_interval_seconds=float(poll_interval_seconds),
                        stop_requested=control.stop_event.is_set,
                    )
        else:
            stop_event = threading.Event()
            with graceful_agent_signals(stop_event):
                result = cell.run(
                    max_work_units=int(max_work_units),
                    poll_interval_seconds=float(poll_interval_seconds),
                    stop_requested=stop_event.is_set,
                )
    after = before if transport is None else transport.status()
    completed = int(result.get("completed_in_run") or 0)
    lifecycle = join_workspace(
        workspace,
        admission_ready=True,
        command_executed=not dry_run,
        campaign_complete=bool(after.get("campaign_complete")),
        completed_work_units=completed,
        last_state=str(result.get("last_state") or "ready"),
    )
    return _seal(
        {
            "schema": VOLUNTEER_JOIN_REPORT_SCHEMA,
            "command_ok": result.get("ok", True) is True,
            "state": str(result.get("last_state") or "ready"),
            "project_hash": project.content_hash,
            "campaign_id": campaign["campaign_id"],
            "campaign_manifest_hash": campaign["manifest_hash"],
            "mode": project.mode.value,
            "training_backend": project.training_backend,
            "requested_device": str(device),
            "selected_device": selected_device,
            "hardware": hardware,
            "resource_preflight": resource_preflight,
            "resource_ready": bool(resource_preflight["resource_ready"]),
            "blockers": list(resource_preflight["blockers"]),
            "max_local_steps": int(max_local_steps),
            "max_download_bytes": int(max_download_bytes),
            "max_work_units": int(max_work_units),
            "dry_run": bool(dry_run),
            "local_status_page_enabled": bool(status_page and not dry_run),
            "local_status_endpoint": local_status_endpoint,
            "graceful_signal_stop": not dry_run,
            "work_claimed": not dry_run and completed > 0,
            "real_peft_work_completed": not dry_run and completed > 0,
            "completed_work_units": completed,
            "accepted_update_count_before": int(before.get("accepted_update_count") or 0),
            "accepted_update_count_after": int(after.get("accepted_update_count") or 0),
            "campaign_complete": bool(after.get("campaign_complete")),
            "workspace_state": lifecycle["state"],
            "controller_authority": "remote_session_owner",
            "v2_session_controller_verified": True,
            "concurrent_elastic_work_verified": True,
            "cell_identity_public": False,
            "pairing_code_public": False,
            "credential_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
    )


__all__ = [
    "PreparedVolunteerSession",
    "VolunteerSessionError",
    "join_volunteer_session",
    "prepare_volunteer_session",
    "run_volunteer_session",
]
