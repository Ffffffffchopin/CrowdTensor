"""Colab CUDA GPU session management helpers.

The saved Colab runtime proxy can outlive the actual Jupyter kernel. These
helpers keep retry/reacquire behavior in one place and return only public-safe
metadata to reports.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from scripts import colab_cli_runtime
from scripts import colab_cuda_session_probe


DEFAULT_SESSION_NAME = "ct-colab-cuda-gpu"
DEFAULT_STATE_PATH = Path.home() / ".config" / "colab-cli" / "sessions.json"
DEFAULT_TOKEN_CACHE = Path.home() / ".config" / "colab-exec" / "token.json"
STALE_ERROR_NEEDLES = (
    "404 client error",
    "not found",
    "/api/kernels",
    "kernel not found",
    "connection was lost",
    "must first start a kernel",
)


def sha256_short(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def is_stale_error(error: BaseException | str | None) -> bool:
    text = str(error or "").lower()
    return any(needle in text for needle in STALE_ERROR_NEEDLES)


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)


def load_session(config_path: str | Path, session_name: str = DEFAULT_SESSION_NAME) -> dict[str, Any]:
    state = load_state(Path(config_path).expanduser())
    session = state.get(session_name)
    if not isinstance(session, dict):
        raise RuntimeError("colab_cuda_session_not_found")
    missing = [key for key in ("url", "token", "endpoint") if not session.get(key)]
    if missing:
        raise RuntimeError("colab_cuda_session_missing_runtime_proxy_fields")
    return dict(session)


def save_runtime_ids(
    config_path: str | Path,
    session_name: str,
    session: dict[str, Any],
    *,
    kernel_id: str | None,
    session_id: str | None,
) -> None:
    state_path = Path(config_path).expanduser()
    state = load_state(state_path)
    current = dict(state.get(session_name) if isinstance(state.get(session_name), dict) else session)
    current.update(session)
    if kernel_id:
        current["kernel_id"] = kernel_id
    if session_id:
        current["session_id"] = session_id
    current["last_execution"] = time.time()
    state[session_name] = current
    write_state(state_path, state)


def clear_runtime_ids(config_path: str | Path, session_name: str) -> None:
    state_path = Path(config_path).expanduser()
    state = load_state(state_path)
    session = state.get(session_name)
    if isinstance(session, dict):
        session["kernel_id"] = None
        session["session_id"] = None
        session["last_execution"] = None
        state[session_name] = session
        write_state(state_path, state)


def public_session_metadata(session: dict[str, Any], *, session_name: str) -> dict[str, Any]:
    parsed = urlparse(str(session.get("url") or ""))
    return {
        "session_name": session_name,
        "accelerator": str(session.get("accelerator") or ""),
        "variant": str(session.get("variant") or ""),
        "endpoint_hash": sha256_short(str(session.get("endpoint") or "")),
        "runtime_proxy_host_hash": sha256_short(parsed.netloc),
        "runtime_proxy_token_public": False,
        "runtime_proxy_url_public": False,
        "endpoint_public": False,
        "credentials_public": False,
        "private_runtime_state_public": False,
    }


def reacquire_gpu_session(
    *,
    session_name: str = DEFAULT_SESSION_NAME,
    state_path: str | Path = DEFAULT_STATE_PATH,
    token_cache: str | Path = DEFAULT_TOKEN_CACHE,
    accelerator: str = "T4",
    authuser: str = "0",
    cleanup_before_gpu: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    args = type("Args", (), {})()
    args.session_name = session_name
    args.accelerator = accelerator
    args.authuser = authuser
    started = time.time()
    try:
        access_token = colab_cuda_session_probe.base.refresh_access_token(Path(token_cache).expanduser())
        precleaned: list[dict[str, Any]] = []
        if cleanup_before_gpu:
            for item in colab_cuda_session_probe.base.list_assignments(access_token, authuser=str(authuser)):
                endpoint = str(item.get("endpoint") or "")
                if endpoint and colab_cuda_session_probe.assignment_is_gpu(item):
                    precleaned.append({
                        "endpoint_hash": sha256_short(endpoint),
                        "status": colab_cuda_session_probe.base.unassign(access_token, endpoint, authuser=str(authuser)),
                    })
        assignment = colab_cuda_session_probe.allocate_gpu(access_token, accelerator, authuser=str(authuser))
        session = colab_cuda_session_probe.save_gpu_session(Path(state_path).expanduser(), session_name, assignment)
        report = {
            "ok": True,
            "reacquired": True,
            "precleaned_assignment_count": len(precleaned),
            "precleaned_assignments": precleaned,
            "duration_seconds": round(time.time() - started, 3),
            **public_session_metadata(session, session_name=session_name),
        }
        return session, report
    except Exception as exc:
        report = colab_cuda_session_probe.build_failure_report(args, exc, started=started)
        report["reacquired"] = False
        return {}, report


def build_runtime(session: dict[str, Any]) -> Any:
    ColabRuntime = colab_cli_runtime.load_colab_runtime_class()
    return ColabRuntime(
        session["url"],
        session["token"],
        kernel_id=session.get("kernel_id"),
        session_id=session.get("session_id"),
    )


def execute_with_retry(
    code: str,
    *,
    session_name: str = DEFAULT_SESSION_NAME,
    state_path: str | Path = DEFAULT_STATE_PATH,
    timeout: float = 1800.0,
    max_attempts: int = 3,
    token_cache: str | Path = DEFAULT_TOKEN_CACHE,
    accelerator: str = "T4",
    authuser: str = "0",
    cleanup_before_reacquire: bool = True,
    force_reacquire_before: bool = False,
    heartbeat_code: str | None = None,
    on_attempt: Callable[[int, dict[str, Any]], None] | None = None,
    stop_runtime_after_success: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    session: dict[str, Any] = {}
    outputs: list[dict[str, Any]] = []
    if force_reacquire_before:
        session, reacquire = reacquire_gpu_session(
            session_name=session_name,
            state_path=state_path,
            token_cache=token_cache,
            accelerator=accelerator,
            authuser=authuser,
            cleanup_before_gpu=cleanup_before_reacquire,
        )
        attempts.append({"attempt": 0, "event": "force_reacquire_before", **reacquire})
        if not reacquire.get("ok"):
            return [], {}, {"ok": False, "attempts": attempts, "blocker": "colab_cuda_reacquire_failed"}
    for attempt_index in range(1, max(1, int(max_attempts)) + 1):
        try:
            session = session or load_session(state_path, session_name)
            runtime = build_runtime(session)
            if heartbeat_code:
                runtime.execute_code(heartbeat_code, timeout=min(120.0, max(30.0, float(timeout) / 10.0)))
            outputs = runtime.execute_code(code, timeout=float(timeout))
            save_runtime_ids(
                state_path,
                session_name,
                session,
                kernel_id=getattr(runtime, "kernel_id", None),
                session_id=getattr(runtime, "session_id", None),
            )
            if stop_runtime_after_success:
                try:
                    runtime.stop()
                except Exception:
                    pass
            attempt = {"attempt": attempt_index, "ok": True, "stale_detected": False}
            attempts.append(attempt)
            if on_attempt:
                on_attempt(attempt_index, attempt)
            return outputs, session, {"ok": True, "attempts": attempts}
        except Exception as exc:  # noqa: BLE001 - public-safe retry boundary
            stale = is_stale_error(exc)
            attempt = {
                "attempt": attempt_index,
                "ok": False,
                "stale_detected": stale,
                "error_type": type(exc).__name__,
                "error_digest": "sha256:" + hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
            }
            attempts.append(attempt)
            if on_attempt:
                on_attempt(attempt_index, attempt)
            if not stale or attempt_index >= max(1, int(max_attempts)):
                return [], session, {"ok": False, "attempts": attempts, "blocker": "colab_cuda_execute_failed"}
            clear_runtime_ids(state_path, session_name)
            session, reacquire = reacquire_gpu_session(
                session_name=session_name,
                state_path=state_path,
                token_cache=token_cache,
                accelerator=accelerator,
                authuser=authuser,
                cleanup_before_gpu=cleanup_before_reacquire,
            )
            attempts.append({"attempt": attempt_index, "event": "reacquire_after_stale", **reacquire})
            if not reacquire.get("ok"):
                return [], session, {"ok": False, "attempts": attempts, "blocker": "colab_cuda_reacquire_failed"}
    return outputs, session, {"ok": False, "attempts": attempts, "blocker": "colab_cuda_execute_exhausted"}
