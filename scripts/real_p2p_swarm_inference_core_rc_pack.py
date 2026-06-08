#!/usr/bin/env python3
"""Build Real P2P Swarm Inference Core RC evidence.

This is the successor to the HTTP P2P-lite v1 RC.  It proves the product
serve/join/generate path through the new real-P2P provider-record daemon while
keeping Coordinator leases and result validation as the execution authority.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import product_swarm_mvp_check as product_mvp  # noqa: E402
import p2p_swarm_inference_v06_pack as p2p_v06  # noqa: E402
import support_bundle  # noqa: E402
from crowdtensor.real_p2p import DISCOVERY_BACKENDS, LIBP2P_KAD_BACKEND, LIBP2P_KAD_COMPAT_BACKEND  # noqa: E402
from crowdtensor.real_llm import missing_hf_dependencies  # noqa: E402
from crowdtensor.session_protocol import public_leak_paths  # noqa: E402


SCHEMA = "real_p2p_swarm_inference_core_rc_v1"
SUPPORT_SCHEMA = "real_p2p_swarm_inference_core_rc_support_bundle_v1"
MODE_LOCAL_SMOKE = "local-smoke"
MODE_PACKAGE = "package"
MODE_EXTERNAL_EXISTING = "external-existing"
MODE_EVIDENCE_IMPORT = "evidence-import"
MODE_KAGGLE_AUTO = "kaggle-auto"
MODE_KAGGLE_CONNECTIVITY = "kaggle-connectivity"
MODE_KAGGLE_RUNTIME_SMOKE = "kaggle-runtime-smoke"
MODES = [
    MODE_LOCAL_SMOKE,
    MODE_PACKAGE,
    MODE_EXTERNAL_EXISTING,
    MODE_EVIDENCE_IMPORT,
    MODE_KAGGLE_AUTO,
    MODE_KAGGLE_CONNECTIVITY,
    MODE_KAGGLE_RUNTIME_SMOKE,
]
DEFAULT_OUTPUT_DIR = "dist/real-p2p-swarm-inference-core-rc"
DEFAULT_P2P_PORT = 9760
DEFAULT_COORDINATOR_PORT = 9761
DEFAULT_HF_MODEL_ID = "sshleifer/tiny-gpt2"
WORKLOAD_TYPE = "real_llm_sharded_infer"
LIBP2P_BACKENDS = {LIBP2P_KAD_BACKEND, LIBP2P_KAD_COMPAT_BACKEND}
DEFAULT_LIBP2P_PORT_OFFSET = 1000
FAILURE_NONE = "none"
FAILURE_KILL_STAGE0_AFTER_CLAIM = "kill-stage0-after-claim"
FAILURE_KILL_STAGE1_AFTER_CLAIM = "kill-stage1-after-claim"
FAILURE_MODES = [FAILURE_NONE, FAILURE_KILL_STAGE0_AFTER_CLAIM, FAILURE_KILL_STAGE1_AFTER_CLAIM]
ADMIN_TOKEN = product_mvp.ADMIN_TOKEN
MINER_TOKEN = product_mvp.MINER_TOKEN
OBSERVER_TOKEN = product_mvp.OBSERVER_TOKEN
Runner = Callable[..., subprocess.CompletedProcess[str]]

SECRET_FRAGMENTS = (
    ADMIN_TOKEN,
    MINER_TOKEN,
    OBSERVER_TOKEN,
    "CROWDTENSOR_P2P_PEER_SECRET=",
    "lease_token",
    "idempotency_key",
    "hidden_state",
    "input_ids",
    "logits",
    "activation_results",
    "activation_result",
    "inference_results",
    "inference_result",
    "sharded_inference_result",
    '"generated_text":',
    '"generated_token_ids":',
    '"prompt_text":',
    "kernel.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in str(stdout or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def redact_text(value: str, secret_values: list[str] | None = None) -> str:
    result = str(value)
    for secret in secret_values or []:
        if secret:
            result = result.replace(secret, "<redacted>")
    for fragment in SECRET_FRAGMENTS:
        result = result.replace(fragment, "<redacted>")
    return result


def redact_values(value: Any, secret_values: list[str] | None = None) -> Any:
    if isinstance(value, str):
        return redact_text(value, secret_values)
    if isinstance(value, list):
        return [redact_values(item, secret_values) for item in value]
    if isinstance(value, dict):
        return {key: redact_values(item, secret_values) for key, item in value.items()}
    return value


def first_string_value(payload: dict[str, Any], key: str) -> str:
    pending: list[Any] = [payload]
    seen: set[int] = set()
    while pending:
        item = pending.pop(0)
        if not isinstance(item, dict):
            continue
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
        for nested in item.values():
            if isinstance(nested, dict):
                pending.append(nested)
            elif isinstance(nested, list):
                pending.extend(entry for entry in nested if isinstance(entry, dict))
    return ""


def model_compatibility(payload: dict[str, Any], expected_model_id: str) -> dict[str, Any]:
    observed = first_string_value(payload, "hf_model_id")
    return {
        "expected_hf_model_id": expected_model_id,
        "observed_hf_model_id": observed,
        "model_id_present": bool(observed),
        "model_id_match": bool(observed and observed == expected_model_id),
        "compatible": bool(observed and observed == expected_model_id),
        "default_model_retained_evidence": False,
    }


def output_request_summary() -> dict[str, Any]:
    return {
        "include_output": False,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Real P2P Core RC public artifacts summarize provider discovery, route "
            "selection, safe generation hashes/counts, stage assignment, peer scoring, "
            "and requeue evidence only. Run `crowdtensor generate --p2p --p2p-backend "
            "real` in local human mode to display answer text."
        ),
    }


def safe_prompt_count(value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, count)


def normalize_prompt_scope(scope: dict[str, Any]) -> dict[str, Any]:
    source = str(scope.get("source") or "imported-or-built-in-validation-prompts")
    inline_prompt_text = bool(scope.get("inline_prompt_text"))
    return {
        "source": source,
        "prompt_count": safe_prompt_count(scope.get("prompt_count")),
        "inline_prompt_text": inline_prompt_text,
        "terminal_next_commands_local_private": bool(scope.get("terminal_next_commands_local_private")),
        "terminal_logs_local_private": bool(scope.get("terminal_logs_local_private")),
        "saved_artifacts_prompt_placeholders": scope.get("saved_artifacts_prompt_placeholders") is not False,
        "saved_artifacts_public_safe": scope.get("saved_artifacts_public_safe") is not False,
        "prefer_prompt_file_or_stdin_for_shareable_logs": bool(scope.get("prefer_prompt_file_or_stdin_for_shareable_logs")),
        "prompt_file_path_public": bool(scope.get("prompt_file_path_public")),
        "raw_prompt_public": bool(scope.get("raw_prompt_public")),
        "public_artifact_safe": scope.get("public_artifact_safe") is not False,
        "summary": (
            "Real P2P Core RC reports record prompt source/count and placeholder "
            "safety only; raw prompt text is excluded from public JSON, Markdown, "
            "and support bundles."
        ),
    }


def prompt_scope_summary(args: argparse.Namespace) -> dict[str, Any]:
    prompt_texts = str(getattr(args, "prompt_texts", "") or "")
    prompt_count = len(product_mvp.prompt_list_from_args(args))
    source = "prompt-texts" if prompt_texts else "prompt-text"
    inline_prompt_text = source in {"prompt-text", "prompt-texts"}
    return normalize_prompt_scope({
        "source": source,
        "prompt_count": prompt_count,
        "inline_prompt_text": inline_prompt_text,
        "terminal_next_commands_local_private": inline_prompt_text,
        "terminal_logs_local_private": inline_prompt_text,
        "saved_artifacts_prompt_placeholders": True,
        "saved_artifacts_public_safe": True,
        "prefer_prompt_file_or_stdin_for_shareable_logs": inline_prompt_text,
        "prompt_file_path_public": False,
        "raw_prompt_public": False,
        "public_artifact_safe": True,
    })


def inherited_prompt_scope(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    prompt_scope = payload.get("prompt_scope") if isinstance(payload.get("prompt_scope"), dict) else {}
    if prompt_scope:
        return normalize_prompt_scope(prompt_scope)
    return normalize_prompt_scope({
        "source": "imported-or-built-in-validation-prompts",
        "prompt_count": len(product_mvp.prompt_list_from_args(args)),
        "inline_prompt_text": False,
        "terminal_next_commands_local_private": False,
        "terminal_logs_local_private": False,
        "saved_artifacts_prompt_placeholders": True,
        "saved_artifacts_public_safe": True,
        "prefer_prompt_file_or_stdin_for_shareable_logs": False,
        "prompt_file_path_public": False,
        "raw_prompt_public": False,
        "public_artifact_safe": True,
    })


def prompt_scope_text(prompt_scope: dict[str, Any]) -> str:
    return (
        f"source={prompt_scope.get('source') or 'unknown'} "
        f"count={prompt_scope.get('prompt_count')} "
        f"inline_prompt_text={bool(prompt_scope.get('inline_prompt_text'))} "
        f"terminal_next_commands_local_private={bool(prompt_scope.get('terminal_next_commands_local_private'))} "
        f"saved_artifacts_prompt_placeholders={bool(prompt_scope.get('saved_artifacts_prompt_placeholders'))} "
        f"prompt_file_path_public={bool(prompt_scope.get('prompt_file_path_public'))} "
        f"raw_prompt_public={bool(prompt_scope.get('raw_prompt_public'))} "
        f"public_artifact_safe={bool(prompt_scope.get('public_artifact_safe'))}"
    )


def answer_scope_summary() -> dict[str, Any]:
    return {
        "scope_state": "no-local-answer",
        "terminal_only": False,
        "visible_in_terminal": False,
        "saved_json_display": "hash-only",
        "saved_markdown_display": "hash-only",
        "json_stdout_display": "hash-only-json",
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "public_artifact_safe": True,
        "summary": (
            "This Real P2P Core RC report is shareable route and readiness evidence, "
            "not an answer transcript. Raw prompts, generated text, generated token ids, "
            "activations, lease tokens, peer secrets, private runtime payloads, and "
            "raw runtime state are excluded."
        ),
    }


def shareable_summary() -> dict[str, Any]:
    return {
        "saved_artifacts_public_safe": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "answer_scope_state": "no-local-answer",
        "local_answer_terminal_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Share `real_p2p_swarm_inference_core_rc.json`, "
            "`real_p2p_swarm_inference_core_rc.md`, and `support_bundle.json`; they "
            "contain Real P2P route evidence, hashes, counts, and readiness summaries, "
            "not raw prompts or answers."
        ),
    }


def request_json(base_url: str, path: str, *, timeout: float = 5.0, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    request = Request(f"{base_url.rstrip('/')}{path}", data=data, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def run_json_step(
    name: str,
    command: list[str],
    *,
    runner: Runner,
    timeout_seconds: float,
    secret_values: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    try:
        completed = runner(
            command,
            cwd=str(ROOT),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "ok": False,
            "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": "timeout",
            "stdout_tail": redact_text((exc.stdout or "")[-1000:] if isinstance(exc.stdout, str) else "", secret_values),
            "stderr_tail": redact_text((exc.stderr or "")[-1000:] if isinstance(exc.stderr, str) else "", secret_values),
        }, {}
    payload = json_from_stdout(completed.stdout)
    step = {
        "name": name,
        "ok": bool(completed.returncode == 0 and payload.get("ok") is not False),
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "payload_schema": payload.get("schema"),
        "payload_ok": payload.get("ok"),
    }
    if not payload:
        step["ok"] = False
        step["error"] = "json_payload_missing"
    if not step["ok"]:
        step["stdout_tail"] = redact_text((completed.stdout or "")[-1000:], secret_values)
        step["stderr_tail"] = redact_text((completed.stderr or "")[-1000:], secret_values)
    return step, payload


def _open_process_log() -> Any:
    return tempfile.TemporaryFile(mode="w+", encoding="utf-8")


def popen_process(command: list[str], *, capture_output: bool = False) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    stdout = subprocess.PIPE if capture_output else _open_process_log()
    stderr = subprocess.PIPE if capture_output else _open_process_log()
    return subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )


def _stream_tail(stream: Any, *, limit: int = 1000) -> str:
    if stream is None:
        return ""
    try:
        stream.flush()
        stream.seek(0)
        return stream.read()[-limit:]
    except (OSError, ValueError):
        return ""


def finish_json_process(
    name: str,
    proc: subprocess.Popen[str],
    *,
    timeout: float,
    secret_values: list[str] | None = None,
    terminate: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if terminate and proc.poll() is None:
        stop_process(proc, secret_values=secret_values)
    started = time.monotonic()
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        stop_process(proc, secret_values=secret_values)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", "process did not exit after timeout"
    payload = json_from_stdout(stdout or "")
    step = {
        "name": name,
        "ok": bool(proc.returncode == 0 and payload.get("ok") is not False),
        "returncode": proc.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "payload_schema": payload.get("schema"),
        "payload_ok": payload.get("ok"),
    }
    if not payload:
        step["ok"] = False
        step["error"] = "json_payload_missing"
    if not step["ok"]:
        step["stdout_tail"] = redact_text((stdout or "")[-1200:], secret_values)
        step["stderr_tail"] = redact_text((stderr or "")[-1200:], secret_values)
    return step, payload


def _completed_persistent_miner(payload: dict[str, Any], *, expected_tasks: int) -> bool:
    try:
        accepted = int(payload.get("accepted_tasks") or 0)
    except (TypeError, ValueError):
        accepted = 0
    return bool(accepted >= int(expected_tasks))


def stop_process(proc: subprocess.Popen[str] | None, *, secret_values: list[str] | None = None) -> dict[str, Any]:
    if proc is None:
        return {}
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate(timeout=5)
    else:
        stdout, stderr = proc.communicate(timeout=1)
    stdout = stdout if isinstance(stdout, str) else _stream_tail(proc.stdout)
    stderr = stderr if isinstance(stderr, str) else _stream_tail(proc.stderr)
    return {
        "returncode": proc.returncode,
        "stdout_tail": redact_text((stdout or "")[-1000:], secret_values),
        "stderr_tail": redact_text((stderr or "")[-1000:], secret_values),
    }


def wait_real_p2p(base_url: str, proc: subprocess.Popen[str], *, timeout: float) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() <= deadline:
        if proc.poll() is not None:
            details = stop_process(proc)
            return False, f"p2p-daemon exited early: {details}"
        try:
            payload = request_json(base_url, "/real-p2p/health", timeout=2.0)
            if payload.get("ok") is True:
                return True, ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.2)
    return False, f"p2p-daemon did not become healthy: {last_error}"


def wait_real_p2p_stage_miners(base_url: str, *, timeout: float, http_timeout: float) -> tuple[bool, dict[str, Any], str]:
    deadline = time.monotonic() + timeout
    last_error = ""
    last_catalog: dict[str, Any] = {}
    while time.monotonic() <= deadline:
        try:
            last_catalog = request_json(base_url, "/real-p2p/providers", timeout=http_timeout)
            _coordinators, stage0, stage1 = provider_counts(last_catalog)
            if stage0 >= 1 and stage1 >= 1:
                return True, last_catalog, ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2.0)
    return False, last_catalog, last_error or "stage miners did not announce before timeout"


def fetch_coordinator_state(base_url: str, *, observer_token: str, timeout: float) -> dict[str, Any]:
    request = Request(
        f"{base_url.rstrip('/')}/state",
        headers={"x-crowdtensor-observer-token": observer_token} if observer_token else {},
        method="GET",
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def real_llm_task_stage(task: dict[str, Any]) -> str:
    metadata = task.get("workload_metadata") if isinstance(task.get("workload_metadata"), dict) else {}
    claim_spec = task.get("claim_workload_spec") if isinstance(task.get("claim_workload_spec"), dict) else {}
    validation = task.get("validation") if isinstance(task.get("validation"), dict) else {}
    runtime_status = task.get("runtime_status") if isinstance(task.get("runtime_status"), dict) else {}
    for source in (metadata, claim_spec, validation, runtime_status):
        try:
            stage_id = int((source or {}).get("stage_id", -1))
        except (TypeError, ValueError):
            stage_id = -1
        if stage_id == 0:
            return "stage0"
        if stage_id == 1:
            return "stage1"
    try:
        runtime_stage_id = int(runtime_status.get("real_llm_stage_id", -1))
    except (TypeError, ValueError):
        runtime_stage_id = -1
    if runtime_stage_id == 0:
        return "stage0"
    if runtime_stage_id == 1:
        return "stage1"
    return ""


def find_real_llm_task(state: dict[str, Any], *, task_id: str = "", stage: str = "", miner_id: str = "") -> dict[str, Any]:
    for task in state.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        if task.get("workload_type") != WORKLOAD_TYPE:
            continue
        if task_id and str(task.get("task_id") or "") != task_id:
            continue
        if stage and real_llm_task_stage(task) != stage:
            continue
        if miner_id and str(task.get("miner_id") or "") != miner_id:
            continue
        return task
    return {}


def wait_for_live_claim(
    args: argparse.Namespace,
    *,
    coordinator_url: str,
    observer_token: str,
    target_stage: str,
    target_miner_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + float(args.claim_observe_timeout)
    last_status = ""
    last_state: dict[str, Any] = {}
    poll_seconds = min(0.25, float(args.kaggle_status_poll_seconds))
    while time.monotonic() <= deadline:
        try:
            state = fetch_coordinator_state(coordinator_url, observer_token=observer_token, timeout=args.http_timeout)
        except Exception as exc:
            last_status = f"{type(exc).__name__}: {exc}"
            time.sleep(poll_seconds)
            continue
        last_state = state
        task = find_real_llm_task(state, stage=target_stage, miner_id=target_miner_id)
        if task and task.get("status") == "leased":
            return {
                "ok": True,
                "stage": target_stage,
                "task_id": str(task.get("task_id") or ""),
                "attempt": task.get("attempt"),
                "miner_id": target_miner_id,
                "status": task.get("status"),
            }, state
        staged = find_real_llm_task(state, stage=target_stage)
        if staged:
            last_status = f"{staged.get('status')}:{staged.get('miner_id')}"
        time.sleep(poll_seconds)
    return {
        "ok": False,
        "stage": target_stage,
        "miner_id": target_miner_id,
        "error": last_status or "claim timeout",
    }, last_state


def wait_for_task_status(
    args: argparse.Namespace,
    *,
    coordinator_url: str,
    observer_token: str,
    task_id: str,
    status: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + float(timeout_seconds)
    last_task: dict[str, Any] = {}
    last_state: dict[str, Any] = {}
    poll_seconds = min(0.5, float(args.kaggle_status_poll_seconds))
    while time.monotonic() <= deadline:
        try:
            state = fetch_coordinator_state(coordinator_url, observer_token=observer_token, timeout=args.http_timeout)
        except Exception:
            time.sleep(poll_seconds)
            continue
        last_state = state
        task = find_real_llm_task(state, task_id=task_id)
        if task:
            last_task = task
            if task.get("status") == status:
                return {
                    "ok": True,
                    "task_id": task_id,
                    "status": status,
                    "attempt": task.get("attempt"),
                    "miner_id": task.get("miner_id"),
                }, state
        time.sleep(poll_seconds)
    return {
        "ok": False,
        "task_id": task_id,
        "expected_status": status,
        "last_status": last_task.get("status"),
        "last_miner_id": last_task.get("miner_id"),
        "last_attempt": last_task.get("attempt"),
    }, last_state


def wait_for_rescue_completion(
    args: argparse.Namespace,
    *,
    coordinator_url: str,
    observer_token: str,
    task_id: str,
    rescue_id: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + float(timeout_seconds)
    last_task: dict[str, Any] = {}
    last_state: dict[str, Any] = {}
    poll_seconds = min(0.5, float(args.kaggle_status_poll_seconds))
    while time.monotonic() <= deadline:
        try:
            state = fetch_coordinator_state(coordinator_url, observer_token=observer_token, timeout=args.http_timeout)
        except Exception:
            time.sleep(poll_seconds)
            continue
        last_state = state
        task = find_real_llm_task(state, task_id=task_id)
        if task:
            last_task = task
            if task.get("status") == "completed" and str(task.get("miner_id") or "") == rescue_id:
                return {
                    "ok": True,
                    "task_id": task_id,
                    "status": "completed",
                    "miner_id": rescue_id,
                    "attempt": task.get("attempt"),
                }, state
        time.sleep(poll_seconds)
    return {
        "ok": False,
        "task_id": task_id,
        "expected_miner_id": rescue_id,
        "last_status": last_task.get("status"),
        "last_miner_id": last_task.get("miner_id"),
        "last_attempt": last_task.get("attempt"),
    }, last_state


def effective_peer_secret(args: argparse.Namespace) -> str:
    if args.peer_secret:
        return str(args.peer_secret)
    if not hasattr(args, "_generated_peer_secret"):
        setattr(args, "_generated_peer_secret", f"real-p2p-rc-{secrets.token_hex(16)}")
    return str(getattr(args, "_generated_peer_secret"))


def provider_counts(catalog: dict[str, Any]) -> tuple[int, int, int]:
    peers = catalog.get("peers") if isinstance(catalog.get("peers"), list) else []
    coordinators = 0
    stage0 = 0
    stage1 = 0
    for peer in peers:
        if not isinstance(peer, dict):
            continue
        if peer.get("role") == "coordinator":
            coordinators += 1
        caps = peer.get("capabilities") if isinstance(peer.get("capabilities"), dict) else {}
        values = caps.get("real_llm_sharded_stage_capabilities") if isinstance(caps.get("real_llm_sharded_stage_capabilities"), list) else []
        if "real_llm_sharded_stage0" in values or "real_llm_sharded_cuda_stage0" in values:
            stage0 += 1
        if "real_llm_sharded_stage1" in values or "real_llm_sharded_cuda_stage1" in values:
            stage1 += 1
    return coordinators, stage0, stage1


def target_stage_for_failure(failure_mode: str) -> str:
    if failure_mode == FAILURE_KILL_STAGE0_AFTER_CLAIM:
        return "stage0"
    if failure_mode == FAILURE_KILL_STAGE1_AFTER_CLAIM:
        return "stage1"
    return ""


def opposite_stage(stage: str) -> str:
    return "stage1" if stage == "stage0" else "stage0"


def kernel_key(stage: str, role: str = "primary") -> str:
    return stage if role == "primary" else f"{stage}-{role}"


def kernel_role_from_key(key: str) -> str:
    if key.endswith("-victim"):
        return "victim"
    if key.endswith("-rescue"):
        return "rescue"
    return "primary"


def stage_from_key(key: str) -> str:
    return "stage1" if key.startswith("stage1") else "stage0"


def victim_miner_id(args: argparse.Namespace, stage: str) -> str:
    return f"real-p2p-rc-kaggle-{stage}-victim"


def rescue_miner_id(args: argparse.Namespace, stage: str) -> str:
    return f"real-p2p-rc-kaggle-{stage}-rescue"


def local_victim_miner_id(args: argparse.Namespace, stage: str) -> str:
    return f"real-p2p-rc-local-{stage}-victim"


def local_rescue_miner_id(args: argparse.Namespace, stage: str) -> str:
    return f"real-p2p-rc-local-{stage}-rescue"


def default_requeue_summary(args: argparse.Namespace) -> dict[str, Any]:
    target_stage = target_stage_for_failure(getattr(args, "failure_mode", FAILURE_NONE))
    return {
        "enabled": bool(target_stage),
        "failure_mode": getattr(args, "failure_mode", FAILURE_NONE),
        "target_stage": target_stage,
        "victim_miner_id": victim_miner_id(args, target_stage) if target_stage else "",
        "rescue_miner_id": rescue_miner_id(args, target_stage) if target_stage else "",
        "claim_observed": False,
        "victim_kernel_deleted": False,
        "lease_expired": False,
        "rescue_miner_used": False,
        "rescued_result": False,
        "accepted_result_after_requeue": False,
        "victim_result_accepted": False,
    }


def local_stage_join_command(
    args: argparse.Namespace,
    *,
    p2p_url: str,
    peer_secret: str,
    stage: str,
    miner_id: str,
    max_tasks: int,
    compute_seconds: float,
    max_request_attempts: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "join",
        "--p2p",
        "--p2p-backend",
        "real",
        "--peer-bootstrap",
        p2p_url,
        "--swarm-id",
        args.swarm_id,
        "--miner-id",
        miner_id,
        "--stage",
        stage,
        "--backend",
        args.backend,
        "--miner-token",
        MINER_TOKEN,
        "--hf-model-id",
        args.hf_model_id,
        "--peer-secret",
        peer_secret,
        "--max-tasks",
        str(max(1, int(max_tasks))),
        "--ttl-seconds",
        str(max(60.0, float(args.generate_timeout) + 60.0)),
        "--idle-sleep",
        "0.25",
        "--max-request-attempts",
        str(max(1, int(max_request_attempts))),
        "--retry-max-sleep",
        "1.0",
        "--run",
        "--json",
    ]
    if compute_seconds > 0:
        command.extend(["--compute-seconds", str(float(compute_seconds))])
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    return command


def safe_requeue_observation(value: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "ok",
        "stage",
        "task_id",
        "attempt",
        "miner_id",
        "status",
        "expected_status",
        "expected_miner_id",
        "last_status",
        "last_miner_id",
        "last_attempt",
        "error",
    }
    return {key: value.get(key) for key in allowed if key in value}


def extract_kernel_ref(text: str) -> str:
    return p2p_v06.extract_kernel_ref(text)


def default_kaggle_owner() -> str:
    return p2p_v06.default_kaggle_owner()


def safe_slug(value: str) -> str:
    return p2p_v06.safe_slug(value)


def kaggle_kernel_slug(prefix: str, suffix: str, *, max_length: int = 49) -> str:
    suffix = safe_slug(suffix)
    prefix = safe_slug(prefix)
    if not suffix:
        return prefix[:max_length].strip("-") or "crowdtensor"
    reserved = len(suffix) + 1
    if reserved >= max_length:
        return suffix[:max_length].strip("-") or "crowdtensor"
    prefix = prefix[: max_length - reserved].strip("-")
    return f"{prefix}-{suffix}" if prefix else suffix


def is_libp2p_backend(args: argparse.Namespace) -> bool:
    return str(getattr(args, "discovery_backend", "")) in LIBP2P_BACKENDS


def p2p_daemon_command(
    args: argparse.Namespace,
    *,
    host: str,
    port: int,
    public_host: str = "",
    peer_secret: str,
    peer_key_file: str = "",
    libp2p_host: str = "127.0.0.1",
    libp2p_port: int = 0,
    libp2p_public_host: str = "",
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "real_p2p_daemon.py"),
        "--host",
        host,
        "--port",
        str(port),
        "--swarm-id",
        args.swarm_id,
        "--record-secret",
        peer_secret,
        "--require-signed",
        "--discovery-backend",
        args.discovery_backend,
    ]
    if public_host:
        command.extend(["--public-host", public_host])
    if is_libp2p_backend(args):
        command.extend([
            "--libp2p-host",
            libp2p_host,
            "--libp2p-port",
            str(libp2p_port),
        ])
        if libp2p_public_host:
            command.extend(["--libp2p-public-host", libp2p_public_host])
        if peer_key_file:
            command.extend(["--peer-key-file", peer_key_file])
    return command


def real_p2p_source_files() -> list[Path]:
    files = p2p_v06.p2p_v06_source_files()
    files.extend([
        ROOT / "package.json",
        ROOT / "package-lock.json",
        ROOT / "scripts" / "real_p2p_daemon.py",
        ROOT / "scripts" / "libp2p_node20_polyfill.mjs",
        ROOT / "scripts" / "libp2p_kad_daemon.mjs",
    ])
    unique: dict[str, Path] = {}
    for path in files:
        if path.is_file() and "__pycache__" not in path.parts:
            unique[str(path.resolve())] = path
    return list(unique.values())


def build_real_p2p_source_tarball(path: Path) -> dict[str, Any]:
    import tarfile

    path.parent.mkdir(parents=True, exist_ok=True)
    included: list[str] = []
    with tarfile.open(path, "w:gz") as tar:
        for file_path in real_p2p_source_files():
            arcname = file_path.resolve().relative_to(ROOT.resolve()).as_posix()
            tar.add(file_path, arcname=arcname)
            included.append(arcname)
    included_set = set(included)
    return {
        "path": str(path),
        "file_count": len(included),
        "included_roots": sorted({item.split("/", 1)[0] for item in included}),
        "libp2p_runtime_files_included": all(
            item in included_set
            for item in {
                "package.json",
                "package-lock.json",
                "scripts/real_p2p_daemon.py",
                "scripts/libp2p_node20_polyfill.mjs",
                "scripts/libp2p_kad_daemon.mjs",
            }
        ),
    }


def extract_libp2p_bootstrap_multiaddr(health_payload: dict[str, Any]) -> str:
    libp2p = health_payload.get("libp2p") if isinstance(health_payload.get("libp2p"), dict) else {}
    addrs = libp2p.get("listen_multiaddrs") if isinstance(libp2p.get("listen_multiaddrs"), list) else []
    for item in addrs:
        text = str(item)
        if "/p2p/" in text:
            return text
    return str(addrs[0] if addrs else "")


def validate_public_report(report: dict[str, Any], *, secret_values: list[str] | None = None) -> list[str]:
    encoded = json.dumps(report, sort_keys=True)
    errors: list[str] = []
    for fragment in SECRET_FRAGMENTS:
        if fragment and fragment in encoded:
            errors.append(f"sensitive_fragment:{fragment}")
    for secret in secret_values or []:
        if secret and secret in encoded:
            errors.append("sensitive_fragment:peer_secret")
    for path in public_leak_paths(report):
        if path.endswith(".prompt_hash") or ".safety." in path:
            continue
        errors.append(f"public_leak:{path}")
    return sorted(set(errors))


def degraded_report(args: argparse.Namespace, output_dir: Path, missing: list[str]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": False,
        "mode": args.mode,
        "degraded": True,
        "output_dir": str(output_dir),
        "backend": args.backend,
        "hf_model_id": args.hf_model_id,
        "max_new_tokens": args.max_new_tokens,
        "missing_dependencies": missing,
        "diagnosis_codes": [
            "hf_dependencies_missing",
            "real_p2p_core_rc_hf_runtime_missing",
            "real_p2p_swarm_inference_core_rc_blocked",
        ],
        "prompt_scope": prompt_scope_summary(args),
        "operator_action": "Install optional runtime dependencies with: python -m pip install -e '.[hf]'",
        "safety": safety_block(discovery_backend=args.discovery_backend),
        "not_completed": [
            "Local real-P2P tiny-GPT generation",
            "External/Kaggle real-P2P proof",
            "libp2p/Kademlia production backend",
            "NAT traversal and relay",
        ],
    }


def safety_block(*, discovery_backend: str) -> dict[str, Any]:
    return {
        "real_p2p_provider_core": True,
        "replaceable_discovery_backend": True,
        "discovery_backend": discovery_backend,
        "coordinator_result_fallback": True,
        "tokens_gossiped": False,
        "raw_prompts_gossiped": False,
        "activations_gossiped": False,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "read_only_workload": WORKLOAD_TYPE,
        "not_production": True,
        "not_hivemind_petals_parity": True,
        "not_economic_system": True,
        "not_anti_sybil_complete": True,
        "not_large_model_throughput": True,
    }


def append_generate_prompt_options(command: list[str], args: argparse.Namespace, *, allow_batch: bool = False, allow_stream: bool = False) -> None:
    if allow_batch and getattr(args, "prompt_texts", ""):
        command.extend(["--prompt-texts", str(args.prompt_texts)])
    else:
        command.extend(["--prompt-text", str(args.prompt_text)])
    if allow_stream and getattr(args, "stream_generation", False):
        command.append("--stream")


def run_local_smoke(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    secret = effective_peer_secret(args)
    secret_values = [secret, ADMIN_TOKEN, MINER_TOKEN, OBSERVER_TOKEN]
    p2p_url = f"http://127.0.0.1:{args.p2p_port}"
    coordinator_url = f"http://127.0.0.1:{args.coordinator_port}"
    state_dir = output_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {}
    p2p_proc: subprocess.Popen[str] | None = None
    serve_proc: subprocess.Popen[str] | None = None
    generate_proc: subprocess.Popen[str] | None = None
    stage_miner_procs: list[tuple[str, subprocess.Popen[str]]] = []
    stopped_stage_miner_keys: set[str] = set()
    p2p_process: dict[str, Any] = {}
    serve_process: dict[str, Any] = {}
    try:
        p2p_cmd = p2p_daemon_command(
            args,
            host="127.0.0.1",
            port=args.p2p_port,
            peer_secret=secret,
            peer_key_file=str(output_dir / "libp2p-bootstrap-peer-key.json") if is_libp2p_backend(args) else "",
        )
        p2p_proc = popen_process(p2p_cmd)
        p2p_healthy, p2p_error = wait_real_p2p(p2p_url, p2p_proc, timeout=args.startup_timeout)
        steps.append({"name": "p2p_daemon", "ok": p2p_healthy, "error": p2p_error, "command": "crowdtensor p2p-daemon --run"})
        if not p2p_healthy:
            return finalize_report(
                args,
                output_dir=output_dir,
                steps=steps,
                payloads=payloads,
                p2p_url=p2p_url,
                coordinator_url=coordinator_url,
                p2p_process=stop_process(p2p_proc, secret_values=secret_values),
                serve_process={},
                secret_values=secret_values,
            )
        if is_libp2p_backend(args):
            try:
                libp2p_health = request_json(p2p_url, "/real-p2p/health", timeout=args.http_timeout)
            except Exception:
                libp2p_health = {}
            bootstrap_multiaddr = extract_libp2p_bootstrap_multiaddr(libp2p_health)
            steps.append({
                "name": "libp2p_bootstrap_multiaddr",
                "ok": bool(bootstrap_multiaddr),
                "multiaddr_present": bool(bootstrap_multiaddr),
            })

        shape_step, shape_payload = run_json_step(
            "p2p_daemon_command_shape",
            [
                sys.executable,
                "-m",
                "crowdtensor.cli",
                "p2p-daemon",
                "--port",
                str(args.p2p_port),
                "--record-secret",
                secret,
                "--require-signed",
                "--json",
            ],
            runner=runner,
            timeout_seconds=args.timeout_seconds,
            secret_values=secret_values,
        )
        steps.append(shape_step)
        payloads["p2p_daemon_command"] = shape_payload

        serve_cmd = [
            sys.executable,
            "-m",
            "crowdtensor.cli",
            "serve",
            "--p2p",
            "--p2p-backend",
            "real",
            "--peer-bootstrap",
            p2p_url,
            "--swarm-id",
            args.swarm_id,
            "--profile",
            "gpu-generation" if args.backend == "cuda" else "cpu-real-llm",
            "--bind-host",
            "127.0.0.1",
            "--public-host",
            "127.0.0.1",
            "--port",
            str(args.coordinator_port),
            "--state-dir",
            str(state_dir),
            "--admin-token",
            ADMIN_TOKEN,
            "--miner-token",
            MINER_TOKEN,
            "--observer-token",
            OBSERVER_TOKEN,
            "--hf-model-id",
            args.hf_model_id,
            "--lease-seconds",
            str(args.lease_seconds),
            "--peer-secret",
            secret,
            "--run",
            "--json",
        ]
        if args.hf_cache_dir:
            serve_cmd.extend(["--hf-cache-dir", args.hf_cache_dir])
        serve_proc = popen_process(serve_cmd)
        coordinator_healthy, coordinator_error = product_mvp.wait_health(coordinator_url, serve_proc, args.startup_timeout)
        steps.append({"name": "serve_real_p2p", "ok": coordinator_healthy, "error": coordinator_error, "command": "crowdtensor serve --p2p --p2p-backend real --run"})
        if not coordinator_healthy:
            return finalize_report(
                args,
                output_dir=output_dir,
                steps=steps,
                payloads=payloads,
                p2p_url=p2p_url,
                coordinator_url=coordinator_url,
                p2p_process={},
                serve_process=stop_process(serve_proc, secret_values=secret_values),
                secret_values=secret_values,
            )

        target_stage = target_stage_for_failure(args.failure_mode)
        if target_stage:
            stable_stage = opposite_stage(target_stage)
            stage_specs = [
                {
                    "key": stable_stage,
                    "stage": stable_stage,
                    "role": "primary",
                    "miner_id": f"real-p2p-rc-{stable_stage}",
                    "max_tasks": args.max_new_tokens,
                    "compute_seconds": args.compute_seconds,
                    "max_request_attempts": max(4, int(args.max_request_attempts)),
                },
                {
                    "key": kernel_key(target_stage, "victim"),
                    "stage": target_stage,
                    "role": "victim",
                    "miner_id": local_victim_miner_id(args, target_stage),
                    "max_tasks": 1,
                    "compute_seconds": args.victim_compute_seconds,
                    "max_request_attempts": 1,
                },
            ]
        else:
            stage_specs = [
                {
                    "key": stage,
                    "stage": stage,
                    "role": "primary",
                    "miner_id": f"real-p2p-rc-{stage}",
                    "max_tasks": args.max_new_tokens,
                    "compute_seconds": args.compute_seconds,
                    "max_request_attempts": max(4, int(args.max_request_attempts)),
                }
                for stage in ("stage0", "stage1")
            ]
        payloads["local_stage_specs"] = [
            {
                "key": str(spec["key"]),
                "stage": str(spec["stage"]),
                "role": str(spec["role"]),
                "miner_id": str(spec["miner_id"]),
                "max_tasks": int(spec["max_tasks"]),
                "compute_seconds": float(spec["compute_seconds"]),
            }
            for spec in stage_specs
        ]
        for spec in stage_specs:
            stage = str(spec["stage"])
            key = str(spec["key"])
            command = local_stage_join_command(
                args,
                p2p_url=p2p_url,
                peer_secret=secret,
                stage=stage,
                miner_id=str(spec["miner_id"]),
                max_tasks=int(spec["max_tasks"]),
                compute_seconds=float(spec["compute_seconds"]),
                max_request_attempts=int(spec["max_request_attempts"]),
            )
            proc = popen_process(command, capture_output=True)
            stage_miner_procs.append((key, proc))
            steps.append({
                "name": f"persistent_join_real_p2p_{key}_started",
                "ok": proc.poll() is None,
                "stage": stage,
                "role": str(spec["role"]),
                "miner_id": str(spec["miner_id"]),
            })

        miners_ready = False
        miner_error = ""
        miner_catalog: dict[str, Any] = {}
        deadline = time.monotonic() + float(args.startup_timeout)
        while time.monotonic() <= deadline:
            try:
                miner_catalog = request_json(p2p_url, "/real-p2p/providers", timeout=args.http_timeout)
                coordinator_count, stage0_count, stage1_count = provider_counts(miner_catalog)
                miners_ready = bool(coordinator_count >= 1 and stage0_count >= 1 and stage1_count >= 1)
                if miners_ready:
                    break
                miner_error = f"provider_counts coordinator={coordinator_count} stage0={stage0_count} stage1={stage1_count}"
            except Exception as exc:
                miner_error = f"{type(exc).__name__}: {exc}"
            if any(proc.poll() is not None for _stage, proc in stage_miner_procs):
                miner_error = "persistent stage miner exited before provider discovery"
                break
            time.sleep(0.5)
        steps.append({
            "name": "persistent_real_p2p_stage_miners_discovered",
            "ok": miners_ready,
            "error": miner_error,
        })
        payloads["persistent_stage_catalog"] = support_bundle.sanitize(redact_values(miner_catalog, secret_values))
        if not miners_ready:
            return finalize_report(
                args,
                output_dir=output_dir,
                steps=steps,
                payloads=payloads,
                p2p_url=p2p_url,
                coordinator_url=coordinator_url,
                p2p_process={},
                serve_process={},
                secret_values=secret_values,
            )

        generate_cmd = [
            sys.executable,
            "-m",
            "crowdtensor.cli",
            "generate",
            "--p2p",
            "--p2p-backend",
            "real",
            "--peer-bootstrap",
            p2p_url,
            "--admin-token",
            ADMIN_TOKEN,
            "--backend",
            args.backend,
            "--hf-model-id",
            args.hf_model_id,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--timeout-seconds",
            str(args.generate_timeout),
            "--http-timeout",
            str(args.http_timeout),
            "--json",
        ]
        append_generate_prompt_options(generate_cmd, args)
        generate_proc = popen_process(generate_cmd, capture_output=True)
        if target_stage:
            steps.append({
                "name": "generate_session_created_via_real_p2p",
                "ok": True,
                "observation": "live_claim_observation_enabled",
                "failure_mode": args.failure_mode,
            })
        else:
            session_queued, queue_error = product_mvp.wait_workload_queued(coordinator_url, timeout=args.session_queue_timeout)
            steps.append({"name": "generate_session_created_via_real_p2p", "ok": session_queued, "error": queue_error})
            if not session_queued:
                step, payload = product_mvp.finish_process_step("generate_real_p2p", generate_proc, timeout=1.0)
                steps.append(step)
                payloads["generate"] = payload
                return finalize_report(
                    args,
                    output_dir=output_dir,
                    steps=steps,
                    payloads=payloads,
                    p2p_url=p2p_url,
                    coordinator_url=coordinator_url,
                    p2p_process={},
                    serve_process={},
                    secret_values=secret_values,
                )

        requeue_summary = default_requeue_summary(args)
        if target_stage:
            victim_key = kernel_key(target_stage, "victim")
            victim_id = local_victim_miner_id(args, target_stage)
            rescue_id = local_rescue_miner_id(args, target_stage)
            requeue_summary.update({
                "scope": "local-smoke",
                "victim_miner_id": victim_id,
                "rescue_miner_id": rescue_id,
            })
            claim, _claim_state = wait_for_live_claim(
                args,
                coordinator_url=coordinator_url,
                observer_token=OBSERVER_TOKEN,
                target_stage=target_stage,
                target_miner_id=victim_id,
            )
            claim = safe_requeue_observation(claim)
            requeue_summary["claim"] = claim
            requeue_summary["claim_observed"] = bool(claim.get("ok"))
            requeue_summary["victim_task_id"] = str(claim.get("task_id") or "")
            steps.append({
                "name": f"local_{target_stage}_victim_claim_observed",
                "ok": bool(claim.get("ok")),
                "claim": claim,
            })
            if claim.get("ok"):
                victim_proc = next((proc for key, proc in stage_miner_procs if key == victim_key), None)
                stopped = stop_process(victim_proc, secret_values=secret_values) if victim_proc is not None else {}
                stopped_stage_miner_keys.add(victim_key)
                requeue_summary["victim_kernel_deleted"] = bool(stopped)
                requeue_summary["victim_process_terminated"] = bool(stopped)
                steps.append({
                    "name": f"local_{target_stage}_victim_process_terminated",
                    "ok": bool(stopped),
                    "returncode": stopped.get("returncode"),
                    "terminated": bool(stopped),
                })
                queued, _queued_state = wait_for_task_status(
                    args,
                    coordinator_url=coordinator_url,
                    observer_token=OBSERVER_TOKEN,
                    task_id=str(claim.get("task_id") or ""),
                    status="queued",
                    timeout_seconds=args.requeue_timeout,
                )
                queued = safe_requeue_observation(queued)
                requeue_summary["requeue_observation"] = queued
                requeue_summary["lease_expired"] = bool(queued.get("ok"))
                steps.append({
                    "name": f"local_{target_stage}_victim_task_requeued",
                    "ok": bool(queued.get("ok")),
                    "observation": queued,
                })
                if queued.get("ok"):
                    rescue_command = local_stage_join_command(
                        args,
                        p2p_url=p2p_url,
                        peer_secret=secret,
                        stage=target_stage,
                        miner_id=rescue_id,
                        max_tasks=args.max_new_tokens,
                        compute_seconds=args.compute_seconds,
                        max_request_attempts=max(4, int(args.max_request_attempts)),
                    )
                    rescue_proc = popen_process(rescue_command, capture_output=True)
                    stage_miner_procs.append((kernel_key(target_stage, "rescue"), rescue_proc))
                    requeue_summary["rescue_process_started"] = rescue_proc.poll() is None
                    steps.append({
                        "name": f"persistent_join_real_p2p_{kernel_key(target_stage, 'rescue')}_started",
                        "ok": rescue_proc.poll() is None,
                        "stage": target_stage,
                        "role": "rescue",
                        "miner_id": rescue_id,
                    })
                    if rescue_proc.poll() is None:
                        completed, _completed_state = wait_for_rescue_completion(
                            args,
                            coordinator_url=coordinator_url,
                            observer_token=OBSERVER_TOKEN,
                            task_id=str(requeue_summary["victim_task_id"]),
                            rescue_id=rescue_id,
                            timeout_seconds=args.requeue_timeout,
                        )
                        completed = safe_requeue_observation(completed)
                        requeue_summary["rescue_observation"] = completed
                        requeue_summary["rescue_miner_used"] = bool(completed.get("ok"))
                        requeue_summary["rescued_result"] = bool(completed.get("ok"))
                        requeue_summary["accepted_result_after_requeue"] = bool(completed.get("ok"))
                        try:
                            state = fetch_coordinator_state(coordinator_url, observer_token=OBSERVER_TOKEN, timeout=args.http_timeout)
                        except Exception:
                            state = {}
                        victim_task = find_real_llm_task(state, task_id=str(requeue_summary["victim_task_id"]))
                        requeue_summary["victim_result_accepted"] = bool(
                            victim_task.get("status") == "completed"
                            and str(victim_task.get("miner_id") or "") == victim_id
                        )
                        steps.append({
                            "name": f"local_{target_stage}_rescue_result_accepted",
                            "ok": bool(completed.get("ok")),
                            "observation": completed,
                        })
        payloads["live_requeue_summary"] = requeue_summary

        generate_step, generate_payload = product_mvp.finish_process_step(
            "generate_real_p2p",
            generate_proc,
            timeout=args.generate_timeout + 10.0,
        )
        generate_step = redact_values(generate_step, secret_values)
        generate_payload = redact_values(generate_payload, secret_values)
        steps.append(generate_step)
        payloads["generate"] = generate_payload

        completed_miner_keys: set[str] = set()
        for stage, proc in stage_miner_procs:
            if stage in completed_miner_keys:
                continue
            completed_miner_keys.add(stage)
            if stage in stopped_stage_miner_keys:
                continue
            miner_step, miner_payload = finish_json_process(
                f"persistent_join_real_p2p_{stage}",
                proc,
                timeout=5.0,
                secret_values=secret_values,
                terminate=True,
            )
            if _completed_persistent_miner(miner_payload, expected_tasks=args.max_new_tokens):
                miner_step["ok"] = True
                miner_step["completed_after_termination"] = bool(miner_step.get("returncode") not in {0, None})
            steps.append(miner_step)
            payloads[f"persistent_join_{stage}"] = redact_values(miner_payload, secret_values)

        route_command = [
            sys.executable,
            "-m",
            "crowdtensor.cli",
            "generate",
            "--p2p",
            "--p2p-backend",
            "real",
            "--peer-bootstrap",
            p2p_url,
            "--backend",
            args.backend,
            "--hf-model-id",
            args.hf_model_id,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--dry-run",
            "--json",
        ]
        append_generate_prompt_options(route_command, args)
        route_step, route_payload = run_json_step(
            "generate_real_p2p_route_dry_run",
            route_command,
            runner=runner,
            timeout_seconds=args.timeout_seconds,
            secret_values=secret_values,
        )
        steps.append(route_step)
        payloads["route_dry_run"] = route_payload

        return finalize_report(
            args,
            output_dir=output_dir,
            steps=steps,
            payloads=payloads,
            p2p_url=p2p_url,
            coordinator_url=coordinator_url,
            p2p_process={},
            serve_process={},
            secret_values=secret_values,
        )
    finally:
        if generate_proc is not None and generate_proc.poll() is None:
            stop_process(generate_proc, secret_values=secret_values if "secret_values" in locals() else [])
        for _stage, proc in stage_miner_procs:
            if proc.poll() is None:
                stop_process(proc, secret_values=secret_values if "secret_values" in locals() else [])
        if serve_proc is not None:
            serve_process.update(stop_process(serve_proc, secret_values=secret_values if "secret_values" in locals() else []))
        if p2p_proc is not None:
            p2p_process.update(stop_process(p2p_proc, secret_values=secret_values if "secret_values" in locals() else []))


def finalize_report(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    steps: list[dict[str, Any]],
    payloads: dict[str, Any],
    p2p_url: str,
    coordinator_url: str,
    p2p_process: dict[str, Any],
    serve_process: dict[str, Any],
    secret_values: list[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    ledger_error = ""
    session_id = str(((payloads.get("generate") or {}).get("session") or {}).get("session_id") or "")
    try:
        query = f"/admin/results?status=accepted&workload_type={WORKLOAD_TYPE}&limit=100"
        if session_id:
            query += f"&session_id={session_id}"
        ledger = product_mvp.request_json("GET", coordinator_url, query, admin_token=ADMIN_TOKEN, timeout=args.http_timeout)
        rows = ledger.get("results") if isinstance(ledger.get("results"), list) else []
    except Exception as exc:
        ledger_error = f"{type(exc).__name__}: {exc}"

    catalog: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    catalog_error = ""
    try:
        catalog = request_json(p2p_url, "/real-p2p/providers", timeout=args.http_timeout)
        write_json(output_dir / "real_p2p_provider_catalog.json", support_bundle.sanitize(redact_values(catalog, secret_values)))
    except Exception as exc:
        catalog_error = f"{type(exc).__name__}: {exc}"
    try:
        diagnostics = request_json(p2p_url, "/real-p2p/diagnostics", timeout=args.http_timeout)
        write_json(output_dir / "real_p2p_diagnostics.json", support_bundle.sanitize(redact_values(diagnostics, secret_values)))
    except Exception:
        diagnostics = {}

    route_payload = payloads.get("route_dry_run") if isinstance(payloads.get("route_dry_run"), dict) else {}
    route = route_payload.get("route") if isinstance(route_payload.get("route"), dict) else {}
    generation = (payloads.get("generate") or {}).get("generation") if isinstance((payloads.get("generate") or {}).get("generation"), dict) else {}
    requeue_summary = payloads.get("live_requeue_summary") if isinstance(payloads.get("live_requeue_summary"), dict) else default_requeue_summary(args)
    stages = product_mvp.stage_summary(rows)
    generated_tokens = int(generation.get("generated_token_count") or 0)
    performance = product_mvp.summarize_step_durations(steps, generated_tokens=generated_tokens, accepted_rows=len(rows))
    resources = product_mvp.runtime_resource_summary(args.backend)
    coordinator_count, stage0_count, stage1_count = provider_counts(catalog)
    registry = catalog.get("registry") if isinstance(catalog.get("registry"), dict) else {}
    peer_scoring = catalog.get("peer_scoring") if isinstance(catalog.get("peer_scoring"), dict) else {}
    signed_count = int(registry.get("signed_provider_record_count") or 0)
    step_ok = all(bool(step.get("ok")) for step in steps)
    generation_ready = bool((payloads.get("generate") or {}).get("ok") is True and generated_tokens >= args.max_new_tokens)
    stage_ready = bool(stages.get("distinct_stage_miners") and int(stages.get("completed_rows") or 0) >= args.max_new_tokens * 2)
    catalog_discovery_ready = bool(
        catalog.get("schema") == "real_p2p_provider_catalog_v1"
        and coordinator_count >= 1
        and stage0_count >= 1
        and stage1_count >= 1
        and signed_count >= 3
    )
    libp2p = catalog.get("libp2p") if isinstance(catalog.get("libp2p"), dict) else {}
    libp2p_backend_ready = bool(
        is_libp2p_backend(args)
        and libp2p.get("ok") is True
        and libp2p.get("peer_id")
        and registry.get("provider_record_transport") == "libp2p-stream"
    )
    route_ready = bool(route_payload.get("ok") and route.get("usable_now") and route.get("route_source") == "real-p2p-discovery")
    provider_core_ready = bool(
        catalog.get("schema") == "real_p2p_provider_catalog_v1"
        and signed_count >= 2
        and stage0_count >= 1
        and stage1_count >= 1
    )
    discovery_ready = bool(catalog_discovery_ready or (route_ready and provider_core_ready and stage_ready))
    peer_scoring_ready = bool(peer_scoring.get("schema") == "real_p2p_peer_scoring_v1" and peer_scoring.get("peer_count"))
    requeue_ready = bool(
        not requeue_summary.get("enabled")
        or (
            requeue_summary.get("claim_observed")
            and requeue_summary.get("victim_kernel_deleted")
            and requeue_summary.get("lease_expired")
            and requeue_summary.get("rescue_miner_used")
            and requeue_summary.get("accepted_result_after_requeue")
            and not requeue_summary.get("victim_result_accepted")
        )
    )
    ready = bool(step_ok and generation_ready and stage_ready and discovery_ready and route_ready and requeue_ready and not ledger_error and not catalog_error)
    codes = set()
    if ready:
        codes.update({
            "real_p2p_swarm_inference_core_rc_ready",
            "libp2p_or_real_p2p_discovery_ready",
            "real_p2p_provider_store_ready",
            "real_p2p_signed_provider_records_ready",
            "real_p2p_stage_discovery_ready",
            "real_p2p_generate_route_ready",
            "real_p2p_local_generate_ready",
            "generated_token_count_ready",
            "distinct_stage_miners",
            "stage_assignment_valid",
            "coordinator_result_fallback_ready",
        })
        if libp2p_backend_ready:
            codes.update({
                "libp2p_discovery_backend_ready",
                "p2p_peer_identity_ready",
                "p2p_provider_dht_ready",
                "hivemind_petals_class_alpha_local_ready",
            })
        if performance.get("stage_latency_ready"):
            codes.add("stage_latency_ready")
        if performance.get("throughput_summary_ready"):
            codes.add("throughput_summary_ready")
        if resources.get("memory_or_vram_summary_ready"):
            codes.add("memory_or_vram_summary_ready")
        if peer_scoring_ready:
            codes.add("peer_scoring_ready")
    else:
        codes.add("real_p2p_swarm_inference_core_rc_blocked")
        if not discovery_ready:
            codes.add("real_p2p_stage_discovery_blocked")
        if not route_ready:
            codes.add("real_p2p_generate_route_blocked")
        if not generation_ready:
            codes.add("real_p2p_generation_not_ready")
        if not stage_ready:
            codes.add("stage_assignment_incomplete")
        if not requeue_ready:
            codes.add("real_p2p_local_stage_requeue_blocked")
        if ledger_error:
            codes.add("admin_results_failed")
        if catalog_error:
            codes.add("real_p2p_catalog_unreachable")
        if is_libp2p_backend(args) and not libp2p_backend_ready:
            codes.add("libp2p_discovery_backend_blocked")
    if requeue_summary.get("enabled"):
        target_stage = str(requeue_summary.get("target_stage") or "")
        if requeue_summary.get("claim_observed"):
            codes.add("live_requeue_victim_claim_observed")
        if requeue_summary.get("victim_kernel_deleted"):
            codes.add("live_requeue_victim_kernel_deleted")
        if requeue_summary.get("victim_process_terminated"):
            codes.add("local_requeue_victim_process_terminated")
        if requeue_summary.get("lease_expired"):
            codes.add("live_requeue_lease_timeout_observed")
        if requeue_summary.get("rescue_miner_used"):
            codes.add("rescue_miner_used")
        if requeue_summary.get("accepted_result_after_requeue"):
            codes.add("accepted_result_after_requeue")
        if requeue_ready and target_stage:
            codes.update({
                "real_p2p_local_stage_requeue_ready",
                "local_stage_requeue_ready",
                "stage_requeue_ready",
                f"live_{target_stage}_requeue_ready",
            })
        else:
            codes.add("real_p2p_local_stage_requeue_blocked")

    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": MODE_LOCAL_SMOKE,
        "output_dir": str(output_dir),
        "backend": args.backend,
        "hf_model_id": args.hf_model_id,
        "max_new_tokens": args.max_new_tokens,
        "p2p": {
            "backend": "real",
            "discovery_backend": args.discovery_backend,
            "p2p_url": p2p_url,
            "catalog_schema": catalog.get("schema"),
            "provider_count": catalog.get("provider_count"),
            "signed_provider_record_count": signed_count,
            "coordinator_provider_count": coordinator_count,
            "stage0_provider_count": stage0_count,
            "stage1_provider_count": stage1_count,
            "route": route,
            "diagnostics": diagnostics,
            "libp2p": libp2p,
            "peer_scoring": peer_scoring,
            "catalog_error": catalog_error,
        },
        "session": {
            "session_id": session_id,
            "workload_type": WORKLOAD_TYPE,
        },
        "generation": generation,
        "stage_assignment": stages,
        "live_requeue_summary": requeue_summary,
        "performance": performance,
        "runtime_resources": resources,
        "prompt_scope": prompt_scope_summary(args),
        "ledger": {
            "accepted_rows": len(rows),
            "error": ledger_error,
        },
        "steps": support_bundle.sanitize(redact_values(steps, secret_values)),
        "processes": {
            "p2p_daemon": support_bundle.sanitize(redact_values(p2p_process, secret_values)),
            "serve": support_bundle.sanitize(redact_values(serve_process, secret_values)),
        },
        "diagnosis_codes": sorted(codes),
        "safety": safety_block(discovery_backend=args.discovery_backend),
        "completed": ["Real P2P provider-record local core proof"] if ready else [],
        "not_completed": [
            "External/Kaggle real-P2P proof",
            "External libp2p/Kad peer discovery proof",
            "Full libp2p/Kademlia provider-record value store",
            "NAT traversal and relay",
            "Economic system",
            "Anti-Sybil security model",
            "Large-model throughput",
        ],
        "operator_action": [
            "Run external-existing against a public p2p-daemon plus two external stage Miners before claiming external real-P2P readiness.",
            "Rotate runtime tokens and peer secrets after every temporary public proof.",
        ],
    }
    return sanitize_report(report, output_dir=output_dir, secret_values=secret_values)


def run_external_existing(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    secret_values = [args.peer_secret, args.admin_token]
    steps: list[dict[str, Any]] = []
    catalog: dict[str, Any] = {}
    catalog_error = ""
    try:
        catalog = request_json(args.peer_bootstrap, "/real-p2p/providers", timeout=args.http_timeout)
        write_json(output_dir / "external_real_p2p_provider_catalog.json", support_bundle.sanitize(redact_values(catalog, secret_values)))
    except Exception as exc:
        catalog_error = f"{type(exc).__name__}: {exc}"
    coordinator_count, stage0_count, stage1_count = provider_counts(catalog)
    peer_scoring = catalog.get("peer_scoring") if isinstance(catalog.get("peer_scoring"), dict) else {}
    route_payload: dict[str, Any] = {}
    if args.verify_generate:
        if not args.admin_token:
            steps.append({"name": "external_real_p2p_generate", "ok": False, "error": "admin_token_required"})
        else:
            command = [
                sys.executable,
                "-m",
                "crowdtensor.cli",
                "generate",
                "--p2p",
                "--p2p-backend",
                "real",
                "--peer-bootstrap",
                args.peer_bootstrap,
                "--admin-token",
                args.admin_token,
                "--backend",
                args.backend,
                "--hf-model-id",
                args.hf_model_id,
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--timeout-seconds",
                str(args.generate_timeout),
                "--http-timeout",
                str(args.http_timeout),
                "--json",
            ]
            append_generate_prompt_options(command, args, allow_batch=True, allow_stream=True)
            step, route_payload = run_json_step(
                "external_real_p2p_generate",
                command,
                runner=runner,
                timeout_seconds=args.generate_timeout + 30.0,
                secret_values=secret_values,
            )
            steps.append(step)
    discovery_ready = bool(catalog.get("schema") == "real_p2p_provider_catalog_v1" and coordinator_count >= 1 and stage0_count >= 1 and stage1_count >= 1)
    peer_scoring_ready = bool(peer_scoring.get("schema") == "real_p2p_peer_scoring_v1" and peer_scoring.get("peer_count"))
    generate_ready = bool(route_payload.get("ok")) if args.verify_generate else False
    generation = route_payload.get("generation") if isinstance(route_payload.get("generation"), dict) else {}
    batch = product_mvp.safe_batch_summary(args, generation) if route_payload else {
        "enabled": bool(str(getattr(args, "prompt_texts", "") or "").strip()),
        "batch_generation_ready": False,
        "raw_prompts_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }
    stream = product_mvp.safe_stream_summary(args, route_payload) if route_payload else {
        "enabled": bool(getattr(args, "stream_generation", False)),
        "requested": bool(getattr(args, "stream_generation", False)),
        "stream_generation_ready": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }
    ready = bool(discovery_ready and (generate_ready if args.verify_generate else True) and not catalog_error)
    codes = set()
    if discovery_ready:
        codes.update({"external_real_p2p_stage_discovery_ready", "libp2p_or_real_p2p_discovery_ready"})
    if generate_ready:
        codes.add("external_real_p2p_generate_ready")
        codes.add("real_p2p_core_rc_model_metadata_ready")
    if batch.get("batch_generation_ready"):
        codes.add("external_real_p2p_generate_batch_ready")
        codes.add("public_swarm_generate_batch_ready")
    if stream.get("stream_generation_ready"):
        codes.add("external_real_p2p_generate_stream_ready")
        codes.add("public_swarm_generate_stream_ready")
        if stream.get("endpoint_ready"):
            codes.add("public_swarm_generate_stream_endpoint_ready")
    if peer_scoring_ready:
        codes.add("peer_scoring_ready")
    if ready and args.verify_generate:
        codes.add("real_p2p_swarm_inference_core_rc_ready")
    if not ready:
        codes.add("real_p2p_external_existing_blocked")
        if catalog_error:
            codes.add("external_real_p2p_catalog_unreachable")
        if not discovery_ready:
            codes.add("external_real_p2p_stage_discovery_blocked")
        if args.verify_generate and not generate_ready:
            codes.add("external_real_p2p_generate_blocked")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": MODE_EXTERNAL_EXISTING,
        "output_dir": str(output_dir),
        "backend": args.backend,
        "hf_model_id": args.hf_model_id,
        "max_new_tokens": args.max_new_tokens,
        "p2p": {
            "backend": "real",
            "peer_bootstrap": args.peer_bootstrap,
            "catalog_schema": catalog.get("schema"),
            "provider_count": catalog.get("provider_count"),
            "coordinator_provider_count": coordinator_count,
            "stage0_provider_count": stage0_count,
            "stage1_provider_count": stage1_count,
            "peer_scoring": peer_scoring,
            "catalog_error": catalog_error,
        },
        "external": {
            "verify_generate": bool(args.verify_generate),
            "generate": route_payload,
            "batch": batch,
            "stream": stream,
        },
        "generation": generation,
        "batch": batch,
        "stream": stream,
        "prompt_scope": prompt_scope_summary(args),
        "steps": support_bundle.sanitize(redact_values(steps, secret_values)),
        "diagnosis_codes": sorted(codes),
        "safety": safety_block(discovery_backend=args.discovery_backend),
        "completed": ["External real-P2P provider discovery"] if discovery_ready else [],
        "not_completed": [
            "libp2p/Kademlia production backend",
            "NAT traversal and relay",
            "Economic system",
            "Anti-Sybil security model",
            "Large-model throughput",
        ],
    }
    return sanitize_report(report, output_dir=output_dir, secret_values=secret_values)


def render_real_p2p_kaggle_kernel(
    *,
    stage: str,
    p2p_url: str,
    p2p_http_port: int,
    p2p_libp2p_bootstrap: str,
    swarm_id: str,
    miner_id: str,
    backend: str,
    hf_model_id: str,
    hf_cache_dir: str,
    miner_token: str,
    peer_secret: str,
    max_tasks: int,
    source_tarball_b64: str,
    compute_seconds: float = 0.0,
    max_request_attempts: int = 180,
) -> str:
    peer_secret_line = f"PEER_SECRET = {json.dumps(peer_secret)}"
    p2p_libp2p_bootstrap_line = f"P2P_LIBP2P_BOOTSTRAP = {json.dumps(p2p_libp2p_bootstrap)}"
    return f'''from __future__ import annotations

import base64
import os
import signal
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from urllib.request import Request, urlopen
import json

STAGE = "{stage}"
P2P_URL = "{p2p_url}"
P2P_HTTP_PORT = {int(p2p_http_port)}
SWARM_ID = "{swarm_id}"
MINER_ID = "{miner_id}"
BACKEND = "{backend}"
HF_MODEL_ID = "{hf_model_id}"
HF_CACHE_DIR = "{hf_cache_dir}"
MINER_TOKEN = "{miner_token}"
{peer_secret_line}
{p2p_libp2p_bootstrap_line}
MAX_TASKS = {int(max_tasks)}
COMPUTE_SECONDS = {float(compute_seconds)}
MAX_REQUEST_ATTEMPTS = {int(max_request_attempts)}
SOURCE_TARBALL_B64 = """{source_tarball_b64}"""

src_dir = Path("/kaggle/working/crowdtensor-src")
src_dir.mkdir(parents=True, exist_ok=True)
archive = Path("/kaggle/working/crowdtensor_source.tar.gz")
archive.write_bytes(base64.b64decode(SOURCE_TARBALL_B64.encode("ascii")))
with tarfile.open(archive, "r:gz") as tar:
    tar.extractall(src_dir)

env = os.environ.copy()
env["PYTHONPATH"] = str(src_dir)
env["PYTHONUNBUFFERED"] = "1"
env["CROWDTENSOR_REMOTE_ENVIRONMENT"] = "kaggle-real-p2p-core-rc"
env["CROWDTENSOR_MINER_TOKEN"] = MINER_TOKEN
log_path = Path("/kaggle/working") / f"crowdtensor_real_p2p_rc_{{STAGE}}.log"
status_path = Path("/kaggle/working") / f"crowdtensor_real_p2p_rc_{{STAGE}}_status.json"


def redact_value(value: object) -> str:
    text = str(value)
    for secret in (MINER_TOKEN, PEER_SECRET):
        if secret:
            text = text.replace(secret, "<redacted>")
    return text


def write_status(step: str, ok: bool, **fields: object) -> None:
    payload = {{
        "schema": "real_p2p_kaggle_stage_status_v1",
        "stage": STAGE,
        "miner_id": MINER_ID,
        "step": step,
        "ok": bool(ok),
        "p2p_url": P2P_URL,
        "libp2p_bootstrap_configured": bool(P2P_LIBP2P_BOOTSTRAP),
        "timestamp": time.time(),
    }}
    for key, value in fields.items():
        if isinstance(value, str):
            payload[key] = redact_value(value)
        else:
            payload[key] = value
    encoded = json.dumps(payload, sort_keys=True)
    status_path.write_text(encoded + "\\n", encoding="utf-8")
    print("KAGGLE_STAGE_STATUS " + encoded, flush=True)


def fetch_json(url: str, timeout: float = 5.0) -> dict:
    req = Request(url, method="GET")
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def run_logged_step(step: str, command: list[str], log, **kwargs: object) -> None:
    write_status(step + "_started", True)
    try:
        subprocess.check_call(command, stdout=log, stderr=subprocess.STDOUT, **kwargs)
    except Exception as exc:
        write_status(step + "_failed", False, error_type=type(exc).__name__, error=str(exc)[:500])
        raise
    write_status(step + "_ready", True)


def cleanup_large_runtime_outputs() -> None:
    for path in (
        src_dir / "node_modules",
        src_dir / "__pycache__",
        src_dir / "crowdtensor" / "__pycache__",
        src_dir / "scripts" / "__pycache__",
    ):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


write_status(
    "source_extracted",
    True,
    source_files={{
        "package_json": (src_dir / "package.json").is_file(),
        "package_lock_json": (src_dir / "package-lock.json").is_file(),
        "real_p2p_daemon": (src_dir / "scripts" / "real_p2p_daemon.py").is_file(),
        "libp2p_kad_daemon": (src_dir / "scripts" / "libp2p_kad_daemon.mjs").is_file(),
    }},
)

with log_path.open("a", encoding="utf-8") as log:
    log.write("CrowdTensor Real P2P Core RC Kaggle stage miner start\\n")
    log.write(f"stage={{STAGE}} miner_id={{MINER_ID}} p2p={{P2P_URL}}\\n")
    log.flush()
    if P2P_LIBP2P_BOOTSTRAP:
        run_logged_step("npm_install", [
            "npm",
            "install",
            "--omit=dev",
            "--no-audit",
            "--no-fund",
        ], log, cwd=str(src_dir))
    run_logged_step("pip_install_hf", [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "transformers==4.40.2",
        "safetensors>=0.4,<1",
    ], log)


def wait_health(url: str, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() <= deadline:
        try:
            req = Request(url.rstrip("/") + "/real-p2p/health", method="GET")
            with urlopen(req, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("ok") is True:
                return
        except Exception as exc:
            last_error = f"{{type(exc).__name__}}: {{exc}}"
        time.sleep(1)
    raise RuntimeError("local libp2p sidecar did not become healthy: " + last_error)


sidecar = None
sidecar_url = P2P_URL
if P2P_LIBP2P_BOOTSTRAP:
    sidecar_url = f"http://127.0.0.1:{{P2P_HTTP_PORT}}"
    sidecar_cmd = [
        sys.executable,
        str(src_dir / "scripts" / "real_p2p_daemon.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(P2P_HTTP_PORT),
        "--swarm-id",
        SWARM_ID,
        "--node-id",
        MINER_ID,
        "--role",
        "miner",
        "--backend",
        BACKEND,
        "--stage-role",
        STAGE,
        "--stage-capability",
        "real_llm_sharded_stage0" if STAGE == "stage0" else "real_llm_sharded_stage1",
        "--record-secret",
        PEER_SECRET,
        "--require-signed",
        "--discovery-backend",
        "libp2p-kad",
        "--bootstrap",
        P2P_LIBP2P_BOOTSTRAP,
        "--peer-key-file",
        f"/kaggle/working/crowdtensor-libp2p-{{STAGE}}-peer-key.json",
    ]
    with log_path.open("a", encoding="utf-8") as log:
        write_status("sidecar_starting", True, bootstrap=P2P_LIBP2P_BOOTSTRAP)
        log.write("Starting local libp2p sidecar: " + redact_value(" ".join(sidecar_cmd)) + "\\n")
        log.flush()
        sidecar = subprocess.Popen(sidecar_cmd, cwd=str(src_dir), env=env, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        wait_health(sidecar_url)
        health = fetch_json(sidecar_url.rstrip("/") + "/real-p2p/health")
        write_status(
            "sidecar_ready",
            True,
            provider_count=health.get("provider_count"),
            diagnosis_codes=(health.get("libp2p") or {{}}).get("diagnosis_codes", []),
            provider_sync=(health.get("libp2p") or {{}}).get("provider_sync", {{}}),
        )
    except Exception as exc:
        write_status("sidecar_health_failed", False, error_type=type(exc).__name__, error=str(exc)[:500])
        raise

command = [
    sys.executable,
    "-m",
    "crowdtensor.cli",
    "join",
    "--p2p",
    "--p2p-backend",
    "real",
    "--peer-bootstrap",
    sidecar_url,
    "--swarm-id",
    SWARM_ID,
    "--miner-id",
    MINER_ID,
    "--stage",
    STAGE,
    "--backend",
    BACKEND,
    "--hf-model-id",
    HF_MODEL_ID,
    "--hf-cache-dir",
    HF_CACHE_DIR,
    "--miner-token",
    MINER_TOKEN,
    "--peer-secret",
    PEER_SECRET,
    "--once",
    "--max-tasks",
    str(MAX_TASKS),
    "--max-request-attempts",
    str(MAX_REQUEST_ATTEMPTS),
    "--compute-seconds",
    str(COMPUTE_SECONDS),
    "--retry-base-sleep",
    "1.0",
    "--retry-max-sleep",
    "5.0",
    "--idle-sleep",
    "1.0",
    "--http-timeout",
    "30",
    "--run",
    "--json",
]
print("Starting CrowdTensor Real P2P Core RC Kaggle stage miner:", redact_value(" ".join(command)), flush=True)
returncode = 1
with log_path.open("a", encoding="utf-8") as log:
    try:
        write_status("join_starting", True)
        log.write("Starting command: " + redact_value(" ".join(command)) + "\\n")
        log.flush()
        process = subprocess.Popen(command, cwd=str(src_dir), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
        write_status("join_finished", returncode == 0, returncode=returncode)
    finally:
        cleanup_large_runtime_outputs()
        if sidecar is not None:
            try:
                os.killpg(sidecar.pid, signal.SIGTERM)
            except Exception:
                pass
    raise SystemExit(returncode)
'''


def render_kaggle_connectivity_kernel(
    *,
    public_host: str,
    p2p_port: int,
    coordinator_port: int,
    libp2p_port: int,
) -> str:
    return f'''from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from urllib.request import urlopen

PUBLIC_HOST = {json.dumps(public_host)}
P2P_PORT = {int(p2p_port)}
COORDINATOR_PORT = {int(coordinator_port)}
LIBP2P_PORT = {int(libp2p_port)}
STATUS_PATH = Path("/kaggle/working/crowdtensor_real_p2p_connectivity_status.json")


def http_get(url: str) -> dict:
    started = time.monotonic()
    try:
        with urlopen(url, timeout=10) as response:
            body = response.read(4096).decode("utf-8", errors="replace")
        payload = json.loads(body) if body else {{}}
        return {{
            "ok": True,
            "status": 200,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            "schema": payload.get("schema"),
            "service": payload.get("service"),
            "provider_count": payload.get("provider_count"),
        }}
    except Exception as exc:
        return {{
            "ok": False,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }}


def tcp_probe(host: str, port: int) -> dict:
    started = time.monotonic()
    try:
        with socket.create_connection((host, int(port)), timeout=10):
            return {{
                "ok": True,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            }}
    except Exception as exc:
        return {{
            "ok": False,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }}


report = {{
    "schema": "real_p2p_kaggle_connectivity_probe_v1",
    "ok": False,
    "public_host": PUBLIC_HOST,
    "p2p_port": P2P_PORT,
    "coordinator_port": COORDINATOR_PORT,
    "libp2p_port": LIBP2P_PORT,
    "probes": {{}},
    "timestamp": time.time(),
}}
report["probes"]["p2p_http_health"] = http_get(f"http://{{PUBLIC_HOST}}:{{P2P_PORT}}/real-p2p/health")
report["probes"]["coordinator_http_health"] = http_get(f"http://{{PUBLIC_HOST}}:{{COORDINATOR_PORT}}/health")
report["probes"]["libp2p_tcp"] = tcp_probe(PUBLIC_HOST, LIBP2P_PORT)
report["ok"] = all(bool(row.get("ok")) for row in report["probes"].values())
report["diagnosis_codes"] = (
    ["kaggle_real_p2p_connectivity_ready"]
    if report["ok"]
    else ["kaggle_real_p2p_connectivity_blocked"]
)
STATUS_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print("KAGGLE_CONNECTIVITY_STATUS " + json.dumps(report, sort_keys=True), flush=True)
raise SystemExit(0 if report["ok"] else 2)
'''


def render_kaggle_runtime_smoke_kernel(
    *,
    p2p_libp2p_bootstrap: str,
    source_tarball_b64: str,
) -> str:
    return f'''from __future__ import annotations

import base64
import json
import os
import signal
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from urllib.request import Request, urlopen

P2P_LIBP2P_BOOTSTRAP = {json.dumps(p2p_libp2p_bootstrap)}
SOURCE_TARBALL_B64 = """{source_tarball_b64}"""
SRC_DIR = Path("/kaggle/working/crowdtensor-src")
STATUS_PATH = Path("/kaggle/working/crowdtensor_real_p2p_runtime_smoke_status.json")
LOG_PATH = Path("/kaggle/working/crowdtensor_real_p2p_runtime_smoke.log")


def write_status(step: str, ok: bool, **fields: object) -> None:
    payload = {{
        "schema": "real_p2p_kaggle_runtime_smoke_status_v1",
        "step": step,
        "ok": bool(ok),
        "libp2p_bootstrap_configured": bool(P2P_LIBP2P_BOOTSTRAP),
        "timestamp": time.time(),
    }}
    payload.update(fields)
    encoded = json.dumps(payload, sort_keys=True)
    STATUS_PATH.write_text(encoded + "\\n", encoding="utf-8")
    print("KAGGLE_RUNTIME_SMOKE_STATUS " + encoded, flush=True)


def run_cmd(step: str, command: list[str], *, cwd: Path | None = None, timeout: float = 300.0) -> None:
    write_status(step + "_started", True, command=command[:3])
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd or SRC_DIR),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        write_status(step + "_failed", False, error_type=type(exc).__name__, error=str(exc)[:500])
        raise
    with LOG_PATH.open("a", encoding="utf-8") as log:
        log.write(f"\\n$ {{' '.join(command)}}\\n")
        log.write(completed.stdout or "")
    write_status(
        step + "_ready",
        completed.returncode == 0,
        returncode=completed.returncode,
        output_tail=(completed.stdout or "")[-1200:],
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def fetch_json(url: str, timeout: float = 5.0) -> dict:
    req = Request(url, method="GET")
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_health(url: str, timeout: float = 90.0) -> dict:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() <= deadline:
        try:
            payload = fetch_json(url.rstrip("/") + "/real-p2p/health", timeout=5)
            if payload.get("ok") is True:
                return payload
        except Exception as exc:
            last_error = f"{{type(exc).__name__}}: {{exc}}"
        time.sleep(1)
    raise RuntimeError("local libp2p runtime smoke sidecar did not become healthy: " + last_error)


def log_tail(path: Path, limit: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except Exception as exc:
        return f"<log unavailable: {{type(exc).__name__}}: {{exc}}>"


def cleanup_large_runtime_outputs() -> None:
    for path in (SRC_DIR / "node_modules", SRC_DIR / "__pycache__"):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def stop_sidecar(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        subprocess.run(["pkill", "-TERM", "-P", str(process.pid)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    except Exception:
        pass
    try:
        process.terminate()
        process.wait(timeout=8)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass
        try:
            process.wait(timeout=5)
        except Exception:
            pass


write_status("started", True)
SRC_DIR.mkdir(parents=True, exist_ok=True)
archive = Path("/kaggle/working/crowdtensor_source.tar.gz")
archive.write_bytes(base64.b64decode(SOURCE_TARBALL_B64.encode("ascii")))
with tarfile.open(archive, "r:gz") as tar:
    tar.extractall(SRC_DIR)

source_files = {{
    "package_json": (SRC_DIR / "package.json").is_file(),
    "package_lock_json": (SRC_DIR / "package-lock.json").is_file(),
    "real_p2p_daemon": (SRC_DIR / "scripts" / "real_p2p_daemon.py").is_file(),
    "libp2p_kad_daemon": (SRC_DIR / "scripts" / "libp2p_kad_daemon.mjs").is_file(),
    "crowdtensor_cli": (SRC_DIR / "crowdtensor" / "cli.py").is_file(),
    "real_llm": (SRC_DIR / "crowdtensor" / "real_llm.py").is_file(),
}}
write_status("source_extracted", all(source_files.values()), source_files=source_files)
if not all(source_files.values()):
    raise SystemExit(2)

env = os.environ.copy()
env["PYTHONPATH"] = str(SRC_DIR)
env["PYTHONUNBUFFERED"] = "1"
env["CROWDTENSOR_REMOTE_ENVIRONMENT"] = "kaggle-real-p2p-runtime-smoke"

run_cmd("python_version", [sys.executable, "--version"], cwd=SRC_DIR, timeout=30)
run_cmd("node_version", ["node", "--version"], cwd=SRC_DIR, timeout=30)
run_cmd("npm_version", ["npm", "--version"], cwd=SRC_DIR, timeout=30)
run_cmd("npm_install", ["npm", "install", "--omit=dev", "--no-audit", "--no-fund"], cwd=SRC_DIR, timeout=360)
run_cmd("node_check_libp2p_daemon", ["node", "--check", "scripts/libp2p_kad_daemon.mjs"], cwd=SRC_DIR, timeout=60)
run_cmd("crowdtensor_cli_help", [sys.executable, "-m", "crowdtensor.cli", "p2p-daemon", "--help"], cwd=SRC_DIR, timeout=60)
run_cmd(
    "pip_install_hf",
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "transformers==4.40.2",
        "safetensors>=0.4,<1",
    ],
    cwd=SRC_DIR,
    timeout=420,
)
run_cmd(
    "hf_dependency_check",
    [
        sys.executable,
        "-c",
        "from crowdtensor.real_llm import missing_hf_dependencies; import json; missing=missing_hf_dependencies(); print(json.dumps({{'missing': missing}})); raise SystemExit(1 if missing else 0)",
    ],
    cwd=SRC_DIR,
    timeout=60,
)

sidecar = None
if P2P_LIBP2P_BOOTSTRAP:
    sidecar_cmd = [
        sys.executable,
        str(SRC_DIR / "scripts" / "real_p2p_daemon.py"),
        "--host",
        "127.0.0.1",
        "--port",
        "9879",
        "--swarm-id",
        "real-p2p-runtime-smoke",
        "--node-id",
        "kaggle-runtime-smoke",
        "--role",
        "miner",
        "--backend",
        "cpu",
        "--stage-role",
        "stage0",
        "--stage-capability",
        "real_llm_sharded_stage0",
        "--record-secret",
        "runtime-smoke-local-secret",
        "--require-signed",
        "--discovery-backend",
        "libp2p-kad",
        "--bootstrap",
        P2P_LIBP2P_BOOTSTRAP,
        "--peer-key-file",
        "/kaggle/working/crowdtensor-runtime-smoke-peer-key.json",
    ]
    write_status("libp2p_sidecar_starting", True, bootstrap_present=True)
    with LOG_PATH.open("a", encoding="utf-8") as log:
        sidecar = subprocess.Popen(sidecar_cmd, cwd=str(SRC_DIR), env=env, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        health = wait_health("http://127.0.0.1:9879", timeout=120)
        write_status(
            "libp2p_sidecar_ready",
            True,
            peer_id=(health.get("libp2p") or {{}}).get("peer_id"),
            diagnosis_codes=(health.get("libp2p") or {{}}).get("diagnosis_codes", []),
            listen_multiaddr_count=len((health.get("libp2p") or {{}}).get("listen_multiaddrs") or []),
        )
    except Exception as exc:
        tail = log_tail(LOG_PATH)
        stop_sidecar(sidecar)
        sidecar = None
        cleanup_large_runtime_outputs()
        write_status(
            "libp2p_sidecar_failed",
            False,
            error_type=type(exc).__name__,
            error=str(exc)[:500],
            sidecar_returncode=sidecar.poll() if sidecar is not None else None,
            log_tail=tail,
        )
        raise
    finally:
        stop_sidecar(sidecar)

cleanup_large_runtime_outputs()
write_status("runtime_smoke_ready", True)
raise SystemExit(0)
'''


def build_kaggle_runtime_smoke_package(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    p2p_libp2p_bootstrap: str = "",
) -> dict[str, Any]:
    owner = args.kaggle_owner or default_kaggle_owner()
    if not owner:
        return {"schema": "real_p2p_kaggle_runtime_smoke_package_v1", "ok": False, "diagnosis_codes": ["kaggle_owner_missing"]}
    package_dir = output_dir / "kaggle-runtime-smoke"
    kernel_dir = package_dir / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    source = build_real_p2p_source_tarball(package_dir / "crowdtensor_source.tar.gz")
    source_b64 = base64.b64encode((package_dir / "crowdtensor_source.tar.gz").read_bytes()).decode("ascii")
    prefix = safe_slug(args.kernel_slug_prefix or f"ct-real-p2p-runtime-{args.p2p_port}")
    kernel_slug = f"{prefix}-smoke"
    code = render_kaggle_runtime_smoke_kernel(
        p2p_libp2p_bootstrap=p2p_libp2p_bootstrap,
        source_tarball_b64=source_b64,
    )
    (kernel_dir / "kernel.py").write_text(code, encoding="utf-8")
    write_json(kernel_dir / "kernel-metadata.json", {
        "id": f"{owner}/{kernel_slug}",
        "title": kernel_slug.replace("-", " ").title(),
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    })
    report = {
        "schema": "real_p2p_kaggle_runtime_smoke_package_v1",
        "ok": bool(source.get("libp2p_runtime_files_included")),
        "output_dir": str(package_dir),
        "kernel_ref": f"{owner}/{kernel_slug}",
        "kernel_dir": str(kernel_dir),
        "source": source,
        "p2p_libp2p_bootstrap": p2p_libp2p_bootstrap,
        "diagnosis_codes": ["real_p2p_kaggle_runtime_smoke_package_ready"] if source.get("libp2p_runtime_files_included") else ["real_p2p_kaggle_runtime_smoke_package_blocked"],
        "safety": {
            "private_tokens_embedded": False,
            "source_tarball_embedded": True,
            "read_only_runtime_probe": True,
            "not_generation_proof": True,
            "not_production": True,
        },
    }
    write_json(package_dir / "real_p2p_kaggle_runtime_smoke_package.json", report)
    return report


def build_kaggle_connectivity_package(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    owner = args.kaggle_owner or default_kaggle_owner()
    if not owner:
        return {"schema": "real_p2p_kaggle_connectivity_package_v1", "ok": False, "diagnosis_codes": ["kaggle_owner_missing"]}
    package_dir = output_dir / "kaggle-connectivity"
    kernel_dir = package_dir / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    prefix = safe_slug(args.kernel_slug_prefix or f"ct-real-p2p-connectivity-{args.p2p_port}")
    kernel_slug = f"{prefix}-probe"
    code = render_kaggle_connectivity_kernel(
        public_host=args.public_host,
        p2p_port=args.p2p_port,
        coordinator_port=args.coordinator_port,
        libp2p_port=args.libp2p_port,
    )
    (kernel_dir / "kernel.py").write_text(code, encoding="utf-8")
    write_json(kernel_dir / "kernel-metadata.json", {
        "id": f"{owner}/{kernel_slug}",
        "title": kernel_slug.replace("-", " ").title(),
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    })
    report = {
        "schema": "real_p2p_kaggle_connectivity_package_v1",
        "ok": True,
        "output_dir": str(package_dir),
        "kernel_ref": f"{owner}/{kernel_slug}",
        "kernel_dir": str(kernel_dir),
        "public_host": args.public_host,
        "p2p_port": args.p2p_port,
        "coordinator_port": args.coordinator_port,
        "libp2p_port": args.libp2p_port,
        "diagnosis_codes": ["real_p2p_kaggle_connectivity_package_ready"],
        "safety": {
            "private_tokens_embedded": False,
            "read_only_network_probe": True,
            "not_production": True,
        },
    }
    write_json(package_dir / "real_p2p_kaggle_connectivity_package.json", report)
    return report


def build_real_p2p_kaggle_package(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    miner_token: str,
    peer_secret: str,
    p2p_libp2p_bootstrap: str = "",
) -> dict[str, Any]:
    owner = args.kaggle_owner or default_kaggle_owner()
    if not owner:
        return {"schema": "real_p2p_core_rc_kaggle_package_v1", "ok": False, "diagnosis_codes": ["kaggle_owner_missing"]}
    package_dir = output_dir / "kaggle-package"
    kernels_dir = package_dir / "kernels"
    package_dir.mkdir(parents=True, exist_ok=True)
    source = build_real_p2p_source_tarball(package_dir / "crowdtensor_source.tar.gz")
    source_b64 = base64.b64encode((package_dir / "crowdtensor_source.tar.gz").read_bytes()).decode("ascii")
    prefix = safe_slug(args.kernel_slug_prefix or f"ct-real-p2p-{args.p2p_port}")
    hf_cache_dir = "/kaggle/working/crowdtensor-hf-cache-real-p2p-rc"
    p2p_url = f"http://{args.public_host}:{args.p2p_port}"
    stages: list[dict[str, Any]] = []
    failure_stage = target_stage_for_failure(getattr(args, "failure_mode", FAILURE_NONE))
    kernel_specs: list[dict[str, Any]] = []
    if failure_stage:
        stable_stage = opposite_stage(failure_stage)
        kernel_specs.extend([
            {
                "stage": failure_stage,
                "role": "victim",
                "key": kernel_key(failure_stage, "victim"),
                "miner_id": victim_miner_id(args, failure_stage),
                "max_tasks": 1,
                "compute_seconds": args.victim_compute_seconds,
                "http_port": 9870 if failure_stage == "stage0" else 9871,
            },
            {
                "stage": stable_stage,
                "role": "primary",
                "key": kernel_key(stable_stage),
                "miner_id": f"real-p2p-rc-kaggle-{stable_stage}",
                "max_tasks": max(1, int(args.max_new_tokens)),
                "compute_seconds": args.compute_seconds,
                "http_port": 9871 if stable_stage == "stage1" else 9870,
            },
            {
                "stage": failure_stage,
                "role": "rescue",
                "key": kernel_key(failure_stage, "rescue"),
                "miner_id": rescue_miner_id(args, failure_stage),
                "max_tasks": max(1, int(args.max_new_tokens)),
                "compute_seconds": args.compute_seconds,
                "http_port": 9872,
            },
        ])
    else:
        for stage in ("stage0", "stage1"):
            kernel_specs.append({
                "stage": stage,
                "role": "primary",
                "key": kernel_key(stage),
                "miner_id": f"real-p2p-rc-kaggle-{stage}",
                "max_tasks": max(1, int(args.max_new_tokens)),
                "compute_seconds": args.compute_seconds,
                "http_port": 9870 if stage == "stage0" else 9871,
            })
    for spec in kernel_specs:
        stage = str(spec["stage"])
        role = str(spec["role"])
        key = str(spec["key"])
        kernel_dir = kernels_dir / key
        kernel_dir.mkdir(parents=True, exist_ok=True)
        kernel_slug = kaggle_kernel_slug(prefix, key)
        miner_id = str(spec["miner_id"])
        code = render_real_p2p_kaggle_kernel(
            stage=stage,
            p2p_url=p2p_url,
            p2p_http_port=int(spec["http_port"]),
            p2p_libp2p_bootstrap=p2p_libp2p_bootstrap,
            swarm_id=args.swarm_id,
            miner_id=miner_id,
            backend=args.backend,
            hf_model_id=args.hf_model_id,
            hf_cache_dir=hf_cache_dir,
            miner_token=miner_token,
            peer_secret=peer_secret,
            source_tarball_b64=source_b64,
            max_tasks=int(spec["max_tasks"]),
            compute_seconds=float(spec["compute_seconds"]),
            max_request_attempts=int(args.max_request_attempts),
        )
        (kernel_dir / "kernel.py").write_text(code, encoding="utf-8")
        write_json(kernel_dir / "kernel-metadata.json", {
            "id": f"{owner}/{kernel_slug}",
            "title": kernel_slug.replace("-", " ").title(),
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        })
        stages.append({
            "stage": stage,
            "role": role,
            "key": key,
            "miner_id": miner_id,
            "kernel_ref": f"{owner}/{kernel_slug}",
            "kernel_dir": str(kernel_dir),
            "max_tasks": int(spec["max_tasks"]),
            "compute_seconds": float(spec["compute_seconds"]),
            "real_p2p_join_present": "crowdtensor.cli" in code and "--p2p-backend" in code and "real" in code,
            "libp2p_sidecar_present": bool(p2p_libp2p_bootstrap and "libp2p-kad" in code and "P2P_LIBP2P_BOOTSTRAP" in code),
            "inline_private_token": True,
            "inline_private_peer_secret": True,
        })
    ok = bool(all(item.get("real_p2p_join_present") for item in stages))
    report = {
        "schema": "real_p2p_core_rc_kaggle_package_v1",
        "ok": ok,
        "output_dir": str(package_dir),
        "p2p_url": p2p_url,
        "p2p_libp2p_bootstrap": p2p_libp2p_bootstrap,
        "source": source,
        "stages": stages,
        "failure_mode": getattr(args, "failure_mode", FAILURE_NONE),
        "diagnosis_codes": ["real_p2p_core_rc_kaggle_package_ready"] if ok else ["real_p2p_core_rc_kaggle_package_blocked"],
        "safety": {
            "private_kernel_payload_contains_miner_token": True,
            "private_kernel_payload_contains_peer_secret": True,
            "source_tarball_excludes_git_and_dist": True,
            "not_production": True,
        },
    }
    write_json(package_dir / "real_p2p_core_rc_kaggle_package.json", report)
    return report


def run_kaggle_step(name: str, command: list[str], *, timeout_seconds: float, secret_values: list[str]) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {"name": name, "ok": False, "error": "timeout", "duration_seconds": round(time.monotonic() - started, 3)}
    output = f"{completed.stdout or ''}\n{completed.stderr or ''}"
    return {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout_tail": redact_text((completed.stdout or "")[-1200:], secret_values),
        "stderr_tail": redact_text((completed.stderr or "")[-1200:], secret_values),
        "actual_kernel_ref": extract_kernel_ref(output),
    }


def wait_kaggle_kernel_terminal(
    kernel_ref: str,
    *,
    timeout_seconds: float,
    poll_seconds: float,
    secret_values: list[str],
) -> dict[str, Any]:
    started = time.monotonic()
    last_step: dict[str, Any] = {}
    while time.monotonic() - started <= timeout_seconds:
        last_step = run_kaggle_step(
            "kaggle_kernel_status",
            ["kaggle", "kernels", "status", kernel_ref],
            timeout_seconds=min(60.0, max(5.0, timeout_seconds)),
            secret_values=secret_values,
        )
        output = f"{last_step.get('stdout_tail') or ''}\n{last_step.get('stderr_tail') or ''}"
        upper = output.upper()
        if any(status in upper for status in ("COMPLETE", "ERROR", "CANCELED", "CANCELLED")):
            last_step["duration_seconds"] = round(time.monotonic() - started, 3)
            last_step["terminal"] = True
            last_step["kernel_ref"] = kernel_ref
            last_step["ok"] = bool(last_step.get("ok") and "COMPLETE" in upper)
            return last_step
        time.sleep(max(1.0, float(poll_seconds)))
    last_step["duration_seconds"] = round(time.monotonic() - started, 3)
    last_step["terminal"] = False
    last_step["kernel_ref"] = kernel_ref
    last_step["ok"] = False
    last_step["error"] = "timeout_waiting_for_terminal_kernel_status"
    return last_step


def cleanup_pushed_kaggle_kernels(
    args: argparse.Namespace,
    *,
    pushed_refs: dict[str, str],
    cleanup_steps: list[dict[str, Any]],
    secret_values: list[str],
) -> None:
    for stage, ref in list(pushed_refs.items()):
        if any(step.get("name") == f"kaggle_delete_{stage}" for step in cleanup_steps):
            continue
        if args.skip_kaggle_cleanup:
            cleanup_steps.append({"name": f"kaggle_delete_{stage}", "ok": False, "skipped": True, "kernel_ref": ref})
            continue
        cleanup = run_kaggle_step(
            f"kaggle_delete_{stage}",
            ["kaggle", "kernels", "delete", ref, "-y"],
            timeout_seconds=args.kaggle_delete_timeout_seconds,
            secret_values=secret_values,
        )
        cleanup["stage"] = stage
        cleanup["kernel_ref"] = ref
        cleanup_steps.append(cleanup)


def collect_pushed_kaggle_outputs(
    *,
    output_dir: Path,
    pushed_refs: dict[str, str],
    secret_values: list[str],
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for key, ref in sorted(pushed_refs.items()):
        output_path = output_dir / "kaggle-output" / safe_slug(key)
        step = run_kaggle_step(
            f"kaggle_output_{key}",
            ["kaggle", "kernels", "output", ref, "-p", str(output_path), "--force"],
            timeout_seconds=timeout_seconds,
            secret_values=secret_values,
        )
        step["key"] = key
        step["kernel_ref"] = ref
        step["output_path"] = str(output_path)
        step["artifact_count"] = len([path for path in output_path.rglob("*") if path.is_file()]) if output_path.exists() else 0
        status_files = sorted(
            str(path.relative_to(output_path))
            for path in output_path.rglob("*status*.json")
            if path.is_file()
        ) if output_path.exists() else []
        log_files = sorted(
            str(path.relative_to(output_path))
            for path in output_path.rglob("*.log")
            if path.is_file()
        ) if output_path.exists() else []
        step["status_files"] = status_files[:20]
        step["log_files"] = log_files[:20]
        steps.append(step)
    return steps


def collect_kaggle_auto_failure_diagnostics(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    pushed_refs: dict[str, str],
    payloads: dict[str, Any],
    steps: list[dict[str, Any]],
    secret_values: list[str],
    reason: str,
) -> None:
    if not pushed_refs:
        return
    terminal_steps: list[dict[str, Any]] = []
    status_timeout = max(10.0, min(90.0, float(args.kaggle_status_poll_seconds) * 2.0))
    for key, ref in sorted(pushed_refs.items()):
        step = wait_kaggle_kernel_terminal(
            ref,
            timeout_seconds=status_timeout,
            poll_seconds=args.kaggle_status_poll_seconds,
            secret_values=secret_values,
        )
        step["name"] = f"kaggle_{key}_terminal_status"
        step["key"] = key
        step["kernel_ref"] = ref
        step["diagnostic_reason"] = reason
        terminal_steps.append(step)
    steps.extend(terminal_steps)
    payloads["kaggle_terminal_status_steps"] = terminal_steps
    payloads["kaggle_output_steps"] = collect_pushed_kaggle_outputs(
        output_dir=output_dir,
        pushed_refs=pushed_refs,
        secret_values=secret_values,
        timeout_seconds=args.kaggle_delete_timeout_seconds,
    )


def cleanup_local_kaggle_private_artifacts(payloads: dict[str, Any]) -> dict[str, Any]:
    package = payloads.get("kaggle_package") if isinstance(payloads.get("kaggle_package"), dict) else {}
    if not package:
        package = payloads.get("connectivity_package") if isinstance(payloads.get("connectivity_package"), dict) else {}
    if not package:
        package = payloads.get("runtime_smoke_package") if isinstance(payloads.get("runtime_smoke_package"), dict) else {}
    package_dir = Path(str(package.get("output_dir") or ""))
    removed: list[str] = []
    errors: list[str] = []
    for path in [package_dir / "kernels", package_dir / "kernel"]:
        if not path.exists():
            continue
        try:
            shutil.rmtree(path)
            removed.append(str(path))
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return {
        "schema": "real_p2p_core_rc_kaggle_private_cleanup_v1",
        "ok": not errors,
        "removed_paths": removed,
        "errors": errors,
        "private_kernel_payloads_removed": True,
    }


def finalize_kaggle_auto(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    steps: list[dict[str, Any]],
    payloads: dict[str, Any],
    pushed_refs: dict[str, str],
    cleanup_steps: list[dict[str, Any]],
    secret_values: list[str],
    p2p_process: dict[str, Any],
    serve_process: dict[str, Any],
) -> dict[str, Any]:
    cleanup_pushed_kaggle_kernels(args, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values)
    payloads["local_private_cleanup"] = cleanup_local_kaggle_private_artifacts(payloads)
    discovery_catalog = payloads.get("stage_catalog") if isinstance(payloads.get("stage_catalog"), dict) else {}
    catalog = discovery_catalog
    catalog_error = ""
    final_catalog: dict[str, Any] = {}
    try:
        final_catalog = request_json(f"http://127.0.0.1:{args.p2p_port}", "/real-p2p/providers", timeout=args.http_timeout)
        write_json(output_dir / "external_real_p2p_provider_catalog_final.json", support_bundle.sanitize(redact_values(final_catalog, secret_values)))
    except Exception as exc:
        catalog_error = f"{type(exc).__name__}: {exc}"
    if not catalog:
        catalog = final_catalog
    if discovery_catalog:
        write_json(output_dir / "external_real_p2p_provider_catalog.json", support_bundle.sanitize(redact_values(discovery_catalog, secret_values)))
    diagnostics: dict[str, Any] = {}
    try:
        diagnostics = request_json(f"http://127.0.0.1:{args.p2p_port}", "/real-p2p/diagnostics", timeout=args.http_timeout)
        write_json(output_dir / "real_p2p_diagnostics.json", support_bundle.sanitize(redact_values(diagnostics, secret_values)))
    except Exception:
        diagnostics = {}
    rows: list[dict[str, Any]] = []
    ledger_error = ""
    session_id = str(((payloads.get("generate") or {}).get("session") or {}).get("session_id") or "")
    try:
        query = f"/admin/results?status=accepted&workload_type={WORKLOAD_TYPE}&limit=100"
        if session_id:
            query += f"&session_id={session_id}"
        ledger = product_mvp.request_json("GET", f"http://127.0.0.1:{args.coordinator_port}", query, admin_token=str(payloads.get("admin_token") or ADMIN_TOKEN), timeout=args.http_timeout)
        rows = ledger.get("results") if isinstance(ledger.get("results"), list) else []
    except Exception as exc:
        ledger_error = f"{type(exc).__name__}: {exc}"
    coordinator_count, stage0_count, stage1_count = provider_counts(catalog)
    registry = catalog.get("registry") if isinstance(catalog.get("registry"), dict) else {}
    peer_scoring = catalog.get("peer_scoring") if isinstance(catalog.get("peer_scoring"), dict) else {}
    signed_count = int(registry.get("signed_provider_record_count") or 0)
    final_coordinator_count, final_stage0_count, final_stage1_count = provider_counts(final_catalog)
    final_registry = final_catalog.get("registry") if isinstance(final_catalog.get("registry"), dict) else {}
    final_peer_scoring = final_catalog.get("peer_scoring") if isinstance(final_catalog.get("peer_scoring"), dict) else {}
    final_signed_count = int(final_registry.get("signed_provider_record_count") or 0)
    generation = (payloads.get("generate") or {}).get("generation") if isinstance((payloads.get("generate") or {}).get("generation"), dict) else {}
    route = (payloads.get("generate") or {}).get("route") if isinstance((payloads.get("generate") or {}).get("route"), dict) else {}
    requeue_summary = payloads.get("live_requeue_summary") if isinstance(payloads.get("live_requeue_summary"), dict) else default_requeue_summary(args)
    generated_tokens = int(generation.get("generated_token_count") or 0)
    stage_assignment = product_mvp.stage_summary(rows)
    cleanup_ok = bool(pushed_refs) and all(step.get("ok") for step in cleanup_steps)
    local_private_cleanup = payloads.get("local_private_cleanup") if isinstance(payloads.get("local_private_cleanup"), dict) else {}
    local_private_cleanup_ok = bool(local_private_cleanup.get("ok", True))
    output_steps = payloads.get("kaggle_output_steps") if isinstance(payloads.get("kaggle_output_steps"), list) else []
    terminal_status_steps = payloads.get("kaggle_terminal_status_steps") if isinstance(payloads.get("kaggle_terminal_status_steps"), list) else []
    terminal_status_ready = bool(terminal_status_steps and all(step.get("terminal") for step in terminal_status_steps))
    terminal_status_blocked = bool(terminal_status_steps and any(not step.get("ok") for step in terminal_status_steps))
    output_probe_ready = bool(output_steps and all(step.get("ok") for step in output_steps))
    output_probe_attempted = bool(output_steps)
    discovery_ready = bool(coordinator_count >= 1 and stage0_count >= 1 and stage1_count >= 1 and signed_count >= 3)
    libp2p = catalog.get("libp2p") if isinstance(catalog.get("libp2p"), dict) else {}
    libp2p_backend_ready = bool(
        is_libp2p_backend(args)
        and libp2p.get("ok") is True
        and libp2p.get("peer_id")
        and registry.get("provider_record_transport") == "libp2p-stream"
    )
    external_generate_ready = bool((payloads.get("generate") or {}).get("ok") and generated_tokens >= args.max_new_tokens)
    stage_ready = bool(stage_assignment.get("distinct_stage_miners") and int(stage_assignment.get("completed_rows") or 0) >= args.max_new_tokens * 2)
    peer_scoring_ready = bool(
        peer_scoring.get("schema") == "real_p2p_peer_scoring_v1"
        and (peer_scoring.get("peer_count") or final_peer_scoring.get("peer_count"))
    )
    requeue_ready = bool(
        not requeue_summary.get("enabled")
        or (
            requeue_summary.get("claim_observed")
            and requeue_summary.get("victim_kernel_deleted")
            and requeue_summary.get("lease_expired")
            and requeue_summary.get("rescue_miner_used")
            and requeue_summary.get("accepted_result_after_requeue")
            and not requeue_summary.get("victim_result_accepted")
        )
    )
    ready = bool(
        all(step.get("ok") for step in steps)
        and (payloads.get("kaggle_package") or {}).get("ok")
        and discovery_ready
        and external_generate_ready
        and stage_ready
        and requeue_ready
        and cleanup_ok
        and local_private_cleanup_ok
        and not catalog_error
        and not ledger_error
    )
    codes = set()
    if discovery_ready:
        codes.update({
            "external_real_p2p_stage_discovery_ready",
            "libp2p_or_real_p2p_discovery_ready",
            "real_p2p_signed_provider_records_ready",
            "real_p2p_provider_store_ready",
        })
        if libp2p_backend_ready:
            codes.update({
                "libp2p_discovery_backend_ready",
                "p2p_peer_identity_ready",
                "p2p_provider_dht_ready",
                "external_libp2p_stage_discovery_ready",
            })
    if external_generate_ready:
        codes.add("external_real_p2p_generate_ready")
        if libp2p_backend_ready:
            codes.add("external_libp2p_generate_ready")
    if peer_scoring_ready:
        codes.add("peer_scoring_ready")
    if requeue_summary.get("enabled"):
        target_stage = str(requeue_summary.get("target_stage") or "")
        if requeue_summary.get("claim_observed"):
            codes.add("live_requeue_victim_claim_observed")
        if requeue_summary.get("victim_kernel_deleted"):
            codes.add("live_requeue_victim_kernel_deleted")
        if requeue_summary.get("lease_expired"):
            codes.add("live_requeue_lease_timeout_observed")
        if requeue_summary.get("rescue_miner_used"):
            codes.add("rescue_miner_used")
        if requeue_summary.get("accepted_result_after_requeue"):
            codes.add("accepted_result_after_requeue")
        if requeue_ready and target_stage:
            codes.update({
                "external_stage_requeue_ready",
                f"live_{target_stage}_requeue_ready",
            })
        else:
            codes.add("external_stage_requeue_blocked")
    if cleanup_ok:
        codes.add("kaggle_kernels_deleted")
    else:
        codes.add("kaggle_cleanup_failed")
    if local_private_cleanup_ok:
        codes.add("real_p2p_kaggle_private_artifacts_cleaned")
    else:
        codes.add("real_p2p_kaggle_private_artifacts_cleanup_failed")
    if terminal_status_ready:
        codes.add("kaggle_stage_terminal_status_ready")
    if terminal_status_blocked:
        codes.add("kaggle_stage_kernel_terminal_blocked")
    if output_probe_ready:
        codes.add("kaggle_stage_output_probe_ready")
    elif output_probe_attempted:
        codes.add("kaggle_stage_output_probe_blocked")
    if ready:
        codes.update({
            "real_p2p_swarm_inference_core_rc_ready",
            "real_p2p_kaggle_auto_ready",
            "coordinator_result_fallback_ready",
            "distinct_stage_miners",
            "stage_assignment_valid",
            "generated_token_count_ready",
            "token_rotation_required",
        })
        if libp2p_backend_ready:
            codes.add("hivemind_petals_class_alpha_ready")
    else:
        codes.add("real_p2p_kaggle_auto_blocked")
        if not discovery_ready:
            codes.add("external_real_p2p_stage_discovery_blocked")
        if not external_generate_ready:
            codes.add("external_real_p2p_generate_blocked")
        if not requeue_ready:
            codes.add("external_stage_requeue_blocked")
        if is_libp2p_backend(args) and not libp2p_backend_ready:
            codes.add("external_libp2p_discovery_backend_blocked")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": MODE_KAGGLE_AUTO,
        "output_dir": str(output_dir),
        "backend": args.backend,
        "hf_model_id": args.hf_model_id,
        "max_new_tokens": args.max_new_tokens,
        "p2p": {
            "backend": "real",
            "discovery_backend": args.discovery_backend,
            "p2p_url": f"http://{args.public_host}:{args.p2p_port}",
            "catalog_schema": catalog.get("schema"),
            "provider_count": catalog.get("provider_count"),
            "signed_provider_record_count": signed_count,
            "coordinator_provider_count": coordinator_count,
            "stage0_provider_count": stage0_count,
            "stage1_provider_count": stage1_count,
            "final_catalog_schema": final_catalog.get("schema"),
            "final_provider_count": final_catalog.get("provider_count"),
            "final_signed_provider_record_count": final_signed_count,
            "final_coordinator_provider_count": final_coordinator_count,
            "final_stage0_provider_count": final_stage0_count,
            "final_stage1_provider_count": final_stage1_count,
            "route": route,
            "diagnostics": diagnostics,
            "libp2p": libp2p,
            "peer_scoring": peer_scoring,
            "final_peer_scoring": final_peer_scoring,
            "catalog_error": catalog_error,
            "discovery_catalog_captured_before_cleanup": bool(discovery_catalog),
        },
        "external": {
            "kaggle_auto": True,
            "external_runtime_verified": discovery_ready,
            "external_generate_verified": external_generate_ready,
        },
        "generation": generation,
        "stage_assignment": stage_assignment,
        "live_requeue_summary": requeue_summary,
        "ledger": {"accepted_rows": len(rows), "error": ledger_error},
        "kaggle_lifecycle": {
            "pushed_refs": pushed_refs,
            "terminal_status_steps": terminal_status_steps,
            "output_steps": output_steps,
            "cleanup_steps": cleanup_steps,
            "kernels_deleted": cleanup_ok,
            "local_private_cleanup": local_private_cleanup,
            "local_private_artifacts_cleaned": local_private_cleanup_ok,
            "token_rotation_required": True,
        },
        "steps": support_bundle.sanitize(redact_values(steps, secret_values)),
        "payload_summaries": {
            "kaggle_package": support_bundle.sanitize(redact_values(payloads.get("kaggle_package") or {}, secret_values)),
            "kaggle_terminal_status_steps": support_bundle.sanitize(redact_values(terminal_status_steps, secret_values)),
            "kaggle_output_steps": support_bundle.sanitize(redact_values(output_steps, secret_values)),
            "generate": {
                "schema": (payloads.get("generate") or {}).get("schema"),
                "ok": (payloads.get("generate") or {}).get("ok"),
                "generation": generation,
            },
        },
        "processes": {
            "p2p_daemon": p2p_process,
            "serve": serve_process,
        },
        "diagnosis_codes": sorted(codes),
        "safety": safety_block(discovery_backend=args.discovery_backend) | {
            "temporary_public_http": True,
            "token_rotation_required": True,
        },
        "completed": ["External Kaggle real-P2P tiny-GPT split generation"] if ready else [],
        "not_completed": [
            "libp2p/Kademlia production backend",
            "NAT traversal and relay",
            "Economic system",
            "Anti-Sybil security model",
            "Large-model throughput",
        ],
        "operator_action": [
            "Rotate runtime tokens and peer secret after this temporary public proof.",
            "Delete any Kaggle kernels manually if cleanup failed.",
        ],
    }
    return sanitize_report(report, output_dir=output_dir, secret_values=secret_values)


def finalize_kaggle_connectivity(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    steps: list[dict[str, Any]],
    payloads: dict[str, Any],
    pushed_refs: dict[str, str],
    cleanup_steps: list[dict[str, Any]],
    secret_values: list[str],
    p2p_process: dict[str, Any],
    serve_process: dict[str, Any],
) -> dict[str, Any]:
    cleanup_pushed_kaggle_kernels(args, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values)
    payloads["local_private_cleanup"] = cleanup_local_kaggle_private_artifacts(payloads)
    probe_step = next((step for step in steps if step.get("name") == "kaggle_connectivity_probe_status"), {})
    local_private_cleanup = payloads.get("local_private_cleanup") if isinstance(payloads.get("local_private_cleanup"), dict) else {}
    cleanup_ok = bool(pushed_refs) and all(step.get("ok") for step in cleanup_steps)
    local_private_cleanup_ok = bool(local_private_cleanup.get("ok", True))
    package_ok = bool((payloads.get("connectivity_package") or {}).get("ok"))
    connectivity_ready = bool(probe_step.get("ok"))
    ready = bool(
        all(step.get("ok") for step in steps if step.get("name") != "kaggle_connectivity_probe_status")
        and package_ok
        and connectivity_ready
        and cleanup_ok
        and local_private_cleanup_ok
    )
    codes = {
        "real_p2p_kaggle_connectivity_ready" if connectivity_ready else "real_p2p_kaggle_connectivity_blocked",
        "kaggle_kernels_deleted" if cleanup_ok else "kaggle_cleanup_failed",
        "real_p2p_kaggle_private_artifacts_cleaned" if local_private_cleanup_ok else "real_p2p_kaggle_private_artifacts_cleanup_failed",
    }
    if ready:
        codes.update({
            "real_p2p_kaggle_connectivity_probe_ready",
            "libp2p_external_tcp_reachable",
            "external_p2p_http_reachable",
            "coordinator_public_http_reachable",
        })
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": MODE_KAGGLE_CONNECTIVITY,
        "output_dir": str(output_dir),
        "backend": args.backend,
        "p2p": {
            "backend": "real",
            "discovery_backend": args.discovery_backend,
            "p2p_url": f"http://{args.public_host}:{args.p2p_port}",
            "libp2p_port": args.libp2p_port,
        },
        "external": {
            "kaggle_connectivity_probe": support_bundle.sanitize(redact_values(probe_step, secret_values)),
        },
        "kaggle_lifecycle": {
            "pushed_refs": pushed_refs,
            "cleanup_steps": cleanup_steps,
            "kernels_deleted": cleanup_ok,
            "local_private_cleanup": local_private_cleanup,
            "local_private_artifacts_cleaned": local_private_cleanup_ok,
        },
        "steps": support_bundle.sanitize(redact_values(steps, secret_values)),
        "payload_summaries": {
            "connectivity_package": support_bundle.sanitize(redact_values(payloads.get("connectivity_package") or {}, secret_values)),
        },
        "processes": {
            "p2p_daemon": p2p_process,
            "serve": serve_process,
        },
        "diagnosis_codes": sorted(codes),
        "safety": safety_block(discovery_backend=args.discovery_backend) | {
            "read_only_network_probe": True,
            "temporary_public_http": True,
        },
        "completed": ["Kaggle external connectivity probe"] if ready else [],
        "not_completed": [
            "External Kaggle real-P2P tiny-GPT split generation",
            "NAT traversal and relay",
            "Economic system",
            "Anti-Sybil security model",
            "Large-model throughput",
        ],
        "operator_action": [
            "Use kaggle-auto only after connectivity probe is ready.",
            "Delete any Kaggle probe kernel manually if cleanup failed.",
        ],
    }
    return sanitize_report(report, output_dir=output_dir, secret_values=secret_values)


def finalize_kaggle_runtime_smoke(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    steps: list[dict[str, Any]],
    payloads: dict[str, Any],
    pushed_refs: dict[str, str],
    cleanup_steps: list[dict[str, Any]],
    secret_values: list[str],
    p2p_process: dict[str, Any],
) -> dict[str, Any]:
    cleanup_pushed_kaggle_kernels(args, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values)
    payloads["local_private_cleanup"] = cleanup_local_kaggle_private_artifacts(payloads)
    smoke_step = next((step for step in steps if step.get("name") == "kaggle_runtime_smoke_status"), {})
    output_steps = payloads.get("kaggle_output_steps") if isinstance(payloads.get("kaggle_output_steps"), list) else []
    local_private_cleanup = payloads.get("local_private_cleanup") if isinstance(payloads.get("local_private_cleanup"), dict) else {}
    cleanup_ok = bool(pushed_refs) and all(step.get("ok") for step in cleanup_steps)
    local_private_cleanup_ok = bool(local_private_cleanup.get("ok", True))
    package = payloads.get("runtime_smoke_package") if isinstance(payloads.get("runtime_smoke_package"), dict) else {}
    package_ok = bool(package.get("ok"))
    runtime_ready = bool(smoke_step.get("ok"))
    ready = bool(
        all(step.get("ok") for step in steps if step.get("name") != "kaggle_runtime_smoke_status")
        and package_ok
        and runtime_ready
        and cleanup_ok
        and local_private_cleanup_ok
    )
    codes = {
        "real_p2p_kaggle_runtime_smoke_ready" if runtime_ready else "real_p2p_kaggle_runtime_smoke_blocked",
        "kaggle_kernels_deleted" if cleanup_ok else "kaggle_cleanup_failed",
        "real_p2p_kaggle_private_artifacts_cleaned" if local_private_cleanup_ok else "real_p2p_kaggle_private_artifacts_cleanup_failed",
    }
    if ready:
        codes.update({
            "real_p2p_kaggle_source_runtime_ready",
            "real_p2p_kaggle_node_runtime_ready",
            "real_p2p_kaggle_hf_runtime_ready",
            "kaggle_libp2p_sidecar_start_ready",
        })
        if is_libp2p_backend(args):
            codes.update({
                "libp2p_discovery_backend_ready",
                "p2p_peer_identity_ready",
            })
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": MODE_KAGGLE_RUNTIME_SMOKE,
        "output_dir": str(output_dir),
        "backend": args.backend,
        "p2p": {
            "backend": "real",
            "discovery_backend": args.discovery_backend,
            "p2p_url": f"http://{args.public_host}:{args.p2p_port}",
            "libp2p_port": args.libp2p_port,
            "libp2p_bootstrap_configured": bool(package.get("p2p_libp2p_bootstrap")),
        },
        "external": {
            "kaggle_runtime_smoke": support_bundle.sanitize(redact_values(smoke_step, secret_values)),
        },
        "kaggle_lifecycle": {
            "pushed_refs": pushed_refs,
            "output_steps": output_steps,
            "cleanup_steps": cleanup_steps,
            "kernels_deleted": cleanup_ok,
            "local_private_cleanup": local_private_cleanup,
            "local_private_artifacts_cleaned": local_private_cleanup_ok,
        },
        "steps": support_bundle.sanitize(redact_values(steps, secret_values)),
        "payload_summaries": {
            "runtime_smoke_package": support_bundle.sanitize(redact_values(package, secret_values)),
        },
        "processes": {
            "p2p_daemon": p2p_process,
        },
        "diagnosis_codes": sorted(codes),
        "safety": safety_block(discovery_backend=args.discovery_backend) | {
            "read_only_runtime_probe": True,
            "no_miner_token_embedded": True,
            "not_generation_proof": True,
            "temporary_public_http": True,
        },
        "completed": ["Kaggle libp2p runtime smoke"] if ready else [],
        "not_completed": [
            "External Kaggle real-P2P tiny-GPT split generation",
            "External libp2p provider discovery with stage0/stage1 Miners",
            "Hivemind/Petals production parity",
            "NAT traversal and relay",
            "Economic system",
            "Anti-Sybil security model",
            "Large-model throughput",
        ],
        "operator_action": [
            "Inspect kaggle-output/runtime-smoke files when runtime smoke is blocked.",
            "Use kaggle-auto only after runtime smoke and connectivity probes are ready.",
            "Delete any Kaggle smoke kernel manually if cleanup failed.",
        ],
    }
    return sanitize_report(report, output_dir=output_dir, secret_values=secret_values)


def run_kaggle_connectivity(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    peer_secret = effective_peer_secret(args)
    miner_token = secrets.token_urlsafe(32)
    admin_token = secrets.token_urlsafe(32)
    observer_token = secrets.token_urlsafe(32)
    secret_values = [peer_secret, miner_token, admin_token, observer_token]
    payloads: dict[str, Any] = {}
    steps: list[dict[str, Any]] = []
    pushed_refs: dict[str, str] = {}
    cleanup_steps: list[dict[str, Any]] = []
    p2p_proc: subprocess.Popen[str] | None = None
    serve_proc: subprocess.Popen[str] | None = None
    state_dir = output_dir / "state"
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    try:
        p2p_proc = popen_process(p2p_daemon_command(
            args,
            host="0.0.0.0",
            port=args.p2p_port,
            public_host=args.public_host,
            peer_secret=peer_secret,
            peer_key_file=str(output_dir / "libp2p-connectivity-peer-key.json") if is_libp2p_backend(args) else "",
            libp2p_host="0.0.0.0" if is_libp2p_backend(args) else "127.0.0.1",
            libp2p_port=int(args.libp2p_port) if is_libp2p_backend(args) else 0,
            libp2p_public_host=args.public_host if is_libp2p_backend(args) else "",
        ))
        p2p_ready, p2p_error = wait_real_p2p(f"http://127.0.0.1:{args.p2p_port}", p2p_proc, timeout=args.startup_timeout)
        steps.append({"name": "real_p2p_daemon_public", "ok": p2p_ready, "error": p2p_error})
        if not p2p_ready:
            return finalize_kaggle_connectivity(args, output_dir=output_dir, steps=steps, payloads=payloads, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values, p2p_process=stop_process(p2p_proc, secret_values=secret_values), serve_process={})

        serve_cmd = [
            sys.executable,
            "-m",
            "crowdtensor.cli",
            "serve",
            "--p2p",
            "--p2p-backend",
            "real",
            "--peer-bootstrap",
            f"http://127.0.0.1:{args.p2p_port}",
            "--swarm-id",
            args.swarm_id,
            "--profile",
            "cpu-real-llm",
            "--bind-host",
            "0.0.0.0",
            "--public-host",
            args.public_host,
            "--port",
            str(args.coordinator_port),
            "--state-dir",
            str(state_dir),
            "--admin-token",
            admin_token,
            "--miner-token",
            miner_token,
            "--observer-token",
            observer_token,
            "--hf-model-id",
            args.hf_model_id,
            "--peer-secret",
            peer_secret,
            "--i-understand-public-bind",
            "--run",
            "--json",
        ]
        serve_proc = popen_process(serve_cmd)
        serve_ready, serve_error = product_mvp.wait_health(f"http://127.0.0.1:{args.coordinator_port}", serve_proc, args.startup_timeout)
        steps.append({"name": "serve_public_real_p2p", "ok": serve_ready, "error": serve_error})
        if not serve_ready:
            return finalize_kaggle_connectivity(args, output_dir=output_dir, steps=steps, payloads=payloads, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values, p2p_process={}, serve_process=stop_process(serve_proc, secret_values=secret_values))

        package = build_kaggle_connectivity_package(args, output_dir=output_dir)
        payloads["connectivity_package"] = package
        ref = str(package.get("kernel_ref") or "")
        kernel_dir = str(package.get("kernel_dir") or "")
        push_step = run_kaggle_step(
            "kaggle_push_connectivity_probe",
            ["kaggle", "kernels", "push", "-p", kernel_dir],
            timeout_seconds=args.kaggle_push_timeout_seconds,
            secret_values=secret_values,
        )
        push_step["kernel_ref"] = ref
        steps.append(push_step)
        if push_step.get("ok"):
            pushed_refs["probe"] = str(push_step.get("actual_kernel_ref") or ref)

        status_ref = pushed_refs.get("probe") or ref
        status_step = wait_kaggle_kernel_terminal(
            status_ref,
            timeout_seconds=args.kaggle_stage_timeout_seconds,
            poll_seconds=args.kaggle_status_poll_seconds,
            secret_values=secret_values,
        )
        status_step["name"] = "kaggle_connectivity_probe_status"
        steps.append(status_step)
        return finalize_kaggle_connectivity(args, output_dir=output_dir, steps=steps, payloads=payloads, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values, p2p_process={}, serve_process={})
    finally:
        if pushed_refs and not cleanup_steps:
            cleanup_pushed_kaggle_kernels(args, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values)
        if serve_proc is not None:
            stop_process(serve_proc, secret_values=secret_values)
        if p2p_proc is not None:
            stop_process(p2p_proc, secret_values=secret_values)


def run_kaggle_runtime_smoke(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    peer_secret = effective_peer_secret(args)
    secret_values = [peer_secret]
    payloads: dict[str, Any] = {}
    steps: list[dict[str, Any]] = []
    pushed_refs: dict[str, str] = {}
    cleanup_steps: list[dict[str, Any]] = []
    p2p_proc: subprocess.Popen[str] | None = None
    try:
        p2p_proc = popen_process(p2p_daemon_command(
            args,
            host="0.0.0.0",
            port=args.p2p_port,
            public_host=args.public_host,
            peer_secret=peer_secret,
            peer_key_file=str(output_dir / "libp2p-runtime-smoke-peer-key.json") if is_libp2p_backend(args) else "",
            libp2p_host="0.0.0.0" if is_libp2p_backend(args) else "127.0.0.1",
            libp2p_port=int(args.libp2p_port) if is_libp2p_backend(args) else 0,
            libp2p_public_host=args.public_host if is_libp2p_backend(args) else "",
        ))
        p2p_ready, p2p_error = wait_real_p2p(f"http://127.0.0.1:{args.p2p_port}", p2p_proc, timeout=args.startup_timeout)
        steps.append({"name": "real_p2p_daemon_public", "ok": p2p_ready, "error": p2p_error})
        if not p2p_ready:
            return finalize_kaggle_runtime_smoke(args, output_dir=output_dir, steps=steps, payloads=payloads, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values, p2p_process=stop_process(p2p_proc, secret_values=secret_values))
        p2p_health = request_json(f"http://127.0.0.1:{args.p2p_port}", "/real-p2p/health", timeout=args.http_timeout)
        p2p_libp2p_bootstrap = extract_libp2p_bootstrap_multiaddr(p2p_health) if is_libp2p_backend(args) else ""
        if is_libp2p_backend(args):
            steps.append({
                "name": "public_libp2p_bootstrap_multiaddr",
                "ok": bool(p2p_libp2p_bootstrap),
                "multiaddr_present": bool(p2p_libp2p_bootstrap),
            })
            if not p2p_libp2p_bootstrap:
                return finalize_kaggle_runtime_smoke(args, output_dir=output_dir, steps=steps, payloads=payloads, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values, p2p_process={})

        package = build_kaggle_runtime_smoke_package(
            args,
            output_dir=output_dir,
            p2p_libp2p_bootstrap=p2p_libp2p_bootstrap,
        )
        payloads["runtime_smoke_package"] = package
        ref = str(package.get("kernel_ref") or "")
        kernel_dir = str(package.get("kernel_dir") or "")
        push_step = run_kaggle_step(
            "kaggle_push_runtime_smoke",
            ["kaggle", "kernels", "push", "-p", kernel_dir],
            timeout_seconds=args.kaggle_push_timeout_seconds,
            secret_values=secret_values,
        )
        push_step["kernel_ref"] = ref
        steps.append(push_step)
        if push_step.get("ok"):
            pushed_refs["runtime-smoke"] = str(push_step.get("actual_kernel_ref") or ref)

        status_ref = pushed_refs.get("runtime-smoke") or ref
        status_step = wait_kaggle_kernel_terminal(
            status_ref,
            timeout_seconds=args.kaggle_stage_timeout_seconds,
            poll_seconds=args.kaggle_status_poll_seconds,
            secret_values=secret_values,
        )
        status_step["name"] = "kaggle_runtime_smoke_status"
        steps.append(status_step)
        payloads["kaggle_output_steps"] = collect_pushed_kaggle_outputs(
            output_dir=output_dir,
            pushed_refs=pushed_refs,
            secret_values=secret_values,
            timeout_seconds=args.kaggle_delete_timeout_seconds,
        )
        return finalize_kaggle_runtime_smoke(args, output_dir=output_dir, steps=steps, payloads=payloads, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values, p2p_process={})
    finally:
        if pushed_refs and not cleanup_steps:
            cleanup_pushed_kaggle_kernels(args, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values)
        if p2p_proc is not None:
            stop_process(p2p_proc, secret_values=secret_values)


def run_kaggle_auto(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    missing = missing_hf_dependencies()
    if missing:
        return sanitize_report(degraded_report(args, output_dir, missing), output_dir=output_dir, secret_values=[])
    peer_secret = effective_peer_secret(args)
    miner_token = secrets.token_urlsafe(32)
    admin_token = secrets.token_urlsafe(32)
    observer_token = secrets.token_urlsafe(32)
    secret_values = [peer_secret, miner_token, admin_token, observer_token]
    payloads: dict[str, Any] = {"admin_token": admin_token}
    steps: list[dict[str, Any]] = []
    pushed_refs: dict[str, str] = {}
    deleted_refs: set[str] = set()
    cleanup_steps: list[dict[str, Any]] = []
    p2p_proc: subprocess.Popen[str] | None = None
    serve_proc: subprocess.Popen[str] | None = None
    generate_proc: subprocess.Popen[str] | None = None
    p2p_process: dict[str, Any] = {}
    serve_process: dict[str, Any] = {}
    requeue_summary = default_requeue_summary(args)
    state_dir = output_dir / "state"
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    try:
        p2p_proc = popen_process(p2p_daemon_command(
            args,
            host="0.0.0.0",
            port=args.p2p_port,
            public_host=args.public_host,
            peer_secret=peer_secret,
            peer_key_file=str(output_dir / "libp2p-public-peer-key.json") if is_libp2p_backend(args) else "",
            libp2p_host="0.0.0.0" if is_libp2p_backend(args) else "127.0.0.1",
            libp2p_port=int(args.libp2p_port) if is_libp2p_backend(args) else 0,
            libp2p_public_host=args.public_host if is_libp2p_backend(args) else "",
        ))
        p2p_ready, p2p_error = wait_real_p2p(f"http://127.0.0.1:{args.p2p_port}", p2p_proc, timeout=args.startup_timeout)
        steps.append({"name": "real_p2p_daemon_public", "ok": p2p_ready, "error": p2p_error})
        if not p2p_ready:
            p2p_process = stop_process(p2p_proc, secret_values=secret_values)
            p2p_proc = None
            return finalize_kaggle_auto(args, output_dir=output_dir, steps=steps, payloads=payloads, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values, p2p_process=p2p_process, serve_process={})
        p2p_health = request_json(f"http://127.0.0.1:{args.p2p_port}", "/real-p2p/health", timeout=args.http_timeout)
        p2p_libp2p_bootstrap = extract_libp2p_bootstrap_multiaddr(p2p_health) if is_libp2p_backend(args) else ""
        if is_libp2p_backend(args):
            steps.append({
                "name": "public_libp2p_bootstrap_multiaddr",
                "ok": bool(p2p_libp2p_bootstrap),
                "multiaddr_present": bool(p2p_libp2p_bootstrap),
            })
            if not p2p_libp2p_bootstrap:
                return finalize_kaggle_auto(args, output_dir=output_dir, steps=steps, payloads=payloads, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values, p2p_process={}, serve_process={})
        package = build_real_p2p_kaggle_package(
            args,
            output_dir=output_dir,
            miner_token=miner_token,
            peer_secret=peer_secret,
            p2p_libp2p_bootstrap=p2p_libp2p_bootstrap,
        )
        payloads["kaggle_package"] = package

        serve_cmd = [
            sys.executable,
            "-m",
            "crowdtensor.cli",
            "serve",
            "--p2p",
            "--p2p-backend",
            "real",
            "--peer-bootstrap",
            f"http://127.0.0.1:{args.p2p_port}",
            "--swarm-id",
            args.swarm_id,
            "--profile",
            "gpu-generation" if args.backend == "cuda" else "cpu-real-llm",
            "--bind-host",
            "0.0.0.0",
            "--public-host",
            args.public_host,
            "--port",
            str(args.coordinator_port),
            "--state-dir",
            str(state_dir),
            "--admin-token",
            admin_token,
            "--miner-token",
            miner_token,
            "--observer-token",
            observer_token,
            "--hf-model-id",
            args.hf_model_id,
            "--lease-seconds",
            str(args.lease_seconds),
            "--peer-secret",
            peer_secret,
            "--i-understand-public-bind",
            "--run",
            "--json",
        ]
        if args.hf_cache_dir:
            serve_cmd.extend(["--hf-cache-dir", args.hf_cache_dir])
        serve_proc = popen_process(serve_cmd)
        serve_ready, serve_error = product_mvp.wait_health(f"http://127.0.0.1:{args.coordinator_port}", serve_proc, args.startup_timeout)
        steps.append({"name": "serve_public_real_p2p", "ok": serve_ready, "error": serve_error})
        if not serve_ready:
            serve_process = stop_process(serve_proc, secret_values=secret_values)
            serve_proc = None
            return finalize_kaggle_auto(args, output_dir=output_dir, steps=steps, payloads=payloads, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values, p2p_process={}, serve_process=serve_process)

        for stage_report in package.get("stages") or []:
            if not isinstance(stage_report, dict):
                continue
            role = str(stage_report.get("role") or "primary")
            if args.failure_mode != FAILURE_NONE and role == "rescue":
                continue
            stage = str(stage_report.get("stage") or "")
            key = str(stage_report.get("key") or stage)
            ref = str(stage_report.get("kernel_ref") or "")
            kernel_dir = str(stage_report.get("kernel_dir") or "")
            step = run_kaggle_step(
                f"kaggle_push_{key}",
                ["kaggle", "kernels", "push", "-p", kernel_dir],
                timeout_seconds=args.kaggle_push_timeout_seconds,
                secret_values=secret_values,
            )
            step["stage"] = stage
            step["role"] = role
            step["key"] = key
            step["kernel_ref"] = ref
            steps.append(step)
            if step.get("ok"):
                pushed_refs[key] = str(step.get("actual_kernel_ref") or ref)
        initial_expected = 2
        if len(pushed_refs) < initial_expected:
            collect_kaggle_auto_failure_diagnostics(
                args,
                output_dir=output_dir,
                pushed_refs=pushed_refs,
                payloads=payloads,
                steps=steps,
                secret_values=secret_values,
                reason="initial_kaggle_stage_push_incomplete",
            )
            return finalize_kaggle_auto(args, output_dir=output_dir, steps=steps, payloads=payloads, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values, p2p_process={}, serve_process={})

        stage_ready, stage_catalog, stage_error = wait_real_p2p_stage_miners(
            f"http://127.0.0.1:{args.p2p_port}",
            timeout=args.kaggle_stage_timeout_seconds,
            http_timeout=args.http_timeout,
        )
        payloads["stage_catalog"] = stage_catalog
        steps.append({
            "name": "wait_kaggle_stage_miners_real_p2p",
            "ok": stage_ready,
            "error": stage_error,
            "provider_count": stage_catalog.get("provider_count"),
            "stage_counts": dict(zip(["coordinator", "stage0", "stage1"], provider_counts(stage_catalog))),
        })
        if not stage_ready:
            collect_kaggle_auto_failure_diagnostics(
                args,
                output_dir=output_dir,
                pushed_refs=pushed_refs,
                payloads=payloads,
                steps=steps,
                secret_values=secret_values,
                reason="stage_discovery_not_ready",
            )
            return finalize_kaggle_auto(args, output_dir=output_dir, steps=steps, payloads=payloads, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values, p2p_process={}, serve_process={})

        generate_cmd = [
            sys.executable,
            "-m",
            "crowdtensor.cli",
            "generate",
            "--p2p",
            "--p2p-backend",
            "real",
            "--peer-bootstrap",
            f"http://{args.public_host}:{args.p2p_port}",
            "--admin-token",
            admin_token,
            "--backend",
            args.backend,
            "--hf-model-id",
            args.hf_model_id,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--timeout-seconds",
            str(args.generate_timeout),
            "--http-timeout",
            str(max(float(args.http_timeout), 60.0)),
            "--json",
        ]
        append_generate_prompt_options(generate_cmd, args)
        if args.failure_mode == FAILURE_NONE:
            generate_step, generate_payload = run_json_step(
                "generate_real_p2p_kaggle",
                generate_cmd,
                runner=subprocess.run,
                timeout_seconds=args.generate_timeout + 120.0,
                secret_values=secret_values,
            )
        else:
            coordinator_url = f"http://127.0.0.1:{args.coordinator_port}"
            target_stage = target_stage_for_failure(args.failure_mode)
            victim_id = victim_miner_id(args, target_stage)
            rescue_id = rescue_miner_id(args, target_stage)
            generate_proc = popen_process(generate_cmd, capture_output=True)
            claim, _ = wait_for_live_claim(
                args,
                coordinator_url=coordinator_url,
                observer_token=observer_token,
                target_stage=target_stage,
                target_miner_id=victim_id,
            )
            requeue_summary["claim"] = claim
            requeue_summary["claim_observed"] = bool(claim.get("ok"))
            requeue_summary["victim_task_id"] = str(claim.get("task_id") or "")
            steps.append({"name": f"{target_stage}_victim_claim_observed", "ok": bool(claim.get("ok")), "claim": claim})
            if claim.get("ok"):
                victim_key = kernel_key(target_stage, "victim")
                victim_ref = pushed_refs.get(victim_key, "")
                delete_step = run_kaggle_step(
                    f"kaggle_delete_{victim_key}",
                    ["kaggle", "kernels", "delete", victim_ref, "-y"],
                    timeout_seconds=args.kaggle_delete_timeout_seconds,
                    secret_values=secret_values,
                )
                delete_step["stage"] = target_stage
                delete_step["role"] = "victim"
                delete_step["key"] = victim_key
                delete_step["kernel_ref"] = victim_ref
                cleanup_steps.append(delete_step)
                if delete_step.get("ok") and victim_ref:
                    deleted_refs.add(victim_ref)
                    pushed_refs.pop(victim_key, None)
                requeue_summary["victim_kernel_deleted"] = bool(delete_step.get("ok"))
                queued, _ = wait_for_task_status(
                    args,
                    coordinator_url=coordinator_url,
                    observer_token=observer_token,
                    task_id=str(claim.get("task_id") or ""),
                    status="queued",
                    timeout_seconds=args.requeue_timeout,
                )
                requeue_summary["requeue_observation"] = queued
                requeue_summary["lease_expired"] = bool(queued.get("ok"))
                steps.append({"name": f"{target_stage}_victim_task_requeued", "ok": bool(queued.get("ok")), "observation": queued})
                if queued.get("ok"):
                    rescue_report = next(
                        (
                            item for item in package.get("stages") or []
                            if isinstance(item, dict) and item.get("role") == "rescue" and item.get("stage") == target_stage
                        ),
                        {},
                    )
                    rescue_key = str(rescue_report.get("key") or kernel_key(target_stage, "rescue"))
                    rescue_ref = str(rescue_report.get("kernel_ref") or "")
                    rescue_dir = str(rescue_report.get("kernel_dir") or "")
                    rescue_push = run_kaggle_step(
                        f"kaggle_push_{rescue_key}",
                        ["kaggle", "kernels", "push", "-p", rescue_dir],
                        timeout_seconds=args.kaggle_push_timeout_seconds,
                        secret_values=secret_values,
                    )
                    rescue_push["stage"] = target_stage
                    rescue_push["role"] = "rescue"
                    rescue_push["key"] = rescue_key
                    rescue_push["kernel_ref"] = rescue_ref
                    steps.append(rescue_push)
                    if rescue_push.get("ok"):
                        pushed_refs[rescue_key] = str(rescue_push.get("actual_kernel_ref") or rescue_ref)
            generate_step, generate_payload = finish_json_process(
                "generate_real_p2p_kaggle",
                generate_proc,
                timeout=args.generate_timeout + 120.0,
                secret_values=secret_values,
            )
            generate_proc = None
            if requeue_summary.get("victim_task_id"):
                completed, _ = wait_for_rescue_completion(
                    args,
                    coordinator_url=coordinator_url,
                    observer_token=observer_token,
                    task_id=str(requeue_summary["victim_task_id"]),
                    rescue_id=rescue_id,
                    timeout_seconds=max(10.0, float(args.kaggle_status_poll_seconds) * 2.0),
                )
                requeue_summary["rescue_observation"] = completed
                requeue_summary["rescue_miner_used"] = bool(completed.get("ok"))
                requeue_summary["rescued_result"] = bool(completed.get("ok"))
                requeue_summary["accepted_result_after_requeue"] = bool(completed.get("ok"))
                try:
                    state = fetch_coordinator_state(coordinator_url, observer_token=observer_token, timeout=args.http_timeout)
                except Exception:
                    state = {}
                victim_task = find_real_llm_task(state, task_id=str(requeue_summary["victim_task_id"]))
                requeue_summary["victim_result_accepted"] = bool(
                    victim_task.get("status") == "completed"
                    and str(victim_task.get("miner_id") or "") == victim_id
                )
                steps.append({"name": f"{target_stage}_rescue_result_accepted", "ok": bool(completed.get("ok")), "observation": completed})
        steps.append(generate_step)
        payloads["generate"] = generate_payload
        payloads["live_requeue_summary"] = requeue_summary
        return finalize_kaggle_auto(args, output_dir=output_dir, steps=steps, payloads=payloads, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values, p2p_process={}, serve_process={})
    finally:
        if generate_proc is not None and generate_proc.poll() is None:
            stop_process(generate_proc, secret_values=secret_values)
        if pushed_refs and not cleanup_steps:
            cleanup_pushed_kaggle_kernels(args, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values)
        elif pushed_refs:
            remaining_refs = {
                key: ref for key, ref in pushed_refs.items()
                if ref and ref not in deleted_refs and not any(step.get("kernel_ref") == ref and step.get("name") == f"kaggle_delete_{key}" for step in cleanup_steps)
            }
            if remaining_refs:
                cleanup_pushed_kaggle_kernels(args, pushed_refs=remaining_refs, cleanup_steps=cleanup_steps, secret_values=secret_values)
        if serve_proc is not None:
            serve_process.update(stop_process(serve_proc, secret_values=secret_values))
        if p2p_proc is not None:
            p2p_process.update(stop_process(p2p_proc, secret_values=secret_values))


def write_runbook(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    path = output_dir / "REAL_P2P_SWARM_INFERENCE_CORE_RC.md"
    lines = [
        "# Real P2P Swarm Inference Core RC",
        "",
        "Run a public provider daemon, Coordinator, and two stage Miners:",
        "",
        "```bash",
        f"crowdtensor p2p-daemon --host 0.0.0.0 --public-host {args.public_host} --port {args.p2p_port} --record-secret \"$CROWDTENSOR_P2P_PEER_SECRET\" --require-signed --run",
        f"crowdtensor serve --p2p --p2p-backend real --peer-bootstrap http://{args.public_host}:{args.p2p_port} --peer-secret \"$CROWDTENSOR_P2P_PEER_SECRET\" --public-host {args.public_host} --port {args.coordinator_port} --run",
        f"crowdtensor join --p2p --p2p-backend real --peer-bootstrap http://{args.public_host}:{args.p2p_port} --peer-secret \"$CROWDTENSOR_P2P_PEER_SECRET\" --stage stage0 --miner-id real-p2p-stage0 --run",
        f"crowdtensor join --p2p --p2p-backend real --peer-bootstrap http://{args.public_host}:{args.p2p_port} --peer-secret \"$CROWDTENSOR_P2P_PEER_SECRET\" --stage stage1 --miner-id real-p2p-stage1 --run",
        f"crowdtensor generate --p2p --p2p-backend real --peer-bootstrap http://{args.public_host}:{args.p2p_port} --prompt 'CrowdTensor real P2P RC' --max-new-tokens {args.max_new_tokens} --json",
        "```",
        "",
        "This RC keeps Coordinator leases/results as the execution authority. It does not provide production libp2p/Kademlia, NAT traversal, relay, economics, anti-Sybil security, or large-model throughput.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"kind": "real_p2p_runbook", "path": path.name, "present": path.is_file()}


def build_package(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    runbook = write_runbook(args, output_dir)
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "mode": MODE_PACKAGE,
        "output_dir": str(output_dir),
        "diagnosis_codes": ["real_p2p_core_rc_runbook_ready"],
        "artifacts": {"runbook": runbook},
        "prompt_scope": prompt_scope_summary(args),
        "safety": safety_block(discovery_backend=args.discovery_backend),
        "not_completed": [
            "Local real-P2P tiny-GPT generation",
            "External/Kaggle real-P2P proof",
            "libp2p/Kademlia production backend",
            "NAT traversal and relay",
        ],
    }
    return sanitize_report(report, output_dir=output_dir, secret_values=[])


def build_evidence_import(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    source = Path(args.real_p2p_report).resolve()
    payload = {}
    if source.is_file():
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    model = model_compatibility(payload, args.hf_model_id)
    ready = bool(
        payload.get("schema") == SCHEMA
        and payload.get("ok") is True
        and "real_p2p_swarm_inference_core_rc_ready" in set(payload.get("diagnosis_codes") or [])
        and model["compatible"]
    )
    codes = set(payload.get("diagnosis_codes") or [])
    if ready:
        codes.add("real_p2p_core_rc_evidence_import_ready")
        codes.add("real_p2p_core_rc_model_metadata_ready")
    else:
        codes.add("real_p2p_core_rc_evidence_import_blocked")
        if not model["compatible"]:
            codes.add("real_p2p_core_rc_model_metadata_mismatch")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": MODE_EVIDENCE_IMPORT,
        "output_dir": str(output_dir),
        "hf_model_id": args.hf_model_id,
        "expected_hf_model_id": args.hf_model_id,
        "source_report": str(source),
        "diagnosis_codes": sorted(codes),
        "imported": {
            "schema": payload.get("schema"),
            "ok": payload.get("ok"),
            "mode": payload.get("mode"),
            "hf_model_id": payload.get("hf_model_id"),
            "model": model,
        },
        "external": payload.get("external") if isinstance(payload.get("external"), dict) else {},
        "p2p": payload.get("p2p") if isinstance(payload.get("p2p"), dict) else {},
        "generation": payload.get("generation") if isinstance(payload.get("generation"), dict) else {},
        "stage_assignment": payload.get("stage_assignment") if isinstance(payload.get("stage_assignment"), dict) else {},
        "live_requeue_summary": payload.get("live_requeue_summary") if isinstance(payload.get("live_requeue_summary"), dict) else {},
        "prompt_scope": inherited_prompt_scope(args, payload),
        "safety": safety_block(discovery_backend=args.discovery_backend),
    }
    return sanitize_report(report, output_dir=output_dir, secret_values=[])


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Real P2P Swarm Inference Core RC",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- diagnosis: {', '.join(report.get('diagnosis_codes') or [])}",
        "",
        "## Output Scope",
        "",
        f"- output request: `include_output={bool((report.get('output_request') or {}).get('include_output'))} raw_prompt_public={bool((report.get('output_request') or {}).get('raw_prompt_public'))} raw_generated_text_public={bool((report.get('output_request') or {}).get('raw_generated_text_public'))} generated_token_ids_public={bool((report.get('output_request') or {}).get('generated_token_ids_public'))} public_artifact_safe={bool((report.get('output_request') or {}).get('public_artifact_safe'))}`",
        f"- output request note: {(report.get('output_request') or {}).get('summary') or 'Public artifacts summarize inference evidence only and do not include answer text.'}",
        f"- prompt scope: `{prompt_scope_text((report.get('prompt_scope') or {}) if isinstance(report.get('prompt_scope'), dict) else {})}`",
        f"- prompt scope note: {(report.get('prompt_scope') or {}).get('summary') or 'Public artifacts exclude raw prompt text.'}",
        f"- answer scope: `state={(report.get('answer_scope') or {}).get('scope_state')} terminal_only={bool((report.get('answer_scope') or {}).get('terminal_only'))} visible_in_terminal={bool((report.get('answer_scope') or {}).get('visible_in_terminal'))} saved_json={(report.get('answer_scope') or {}).get('saved_json_display')} saved_markdown={(report.get('answer_scope') or {}).get('saved_markdown_display')} public_artifact_safe={bool((report.get('answer_scope') or {}).get('public_artifact_safe'))}`",
        f"- answer scope note: {(report.get('answer_scope') or {}).get('summary') or 'Public artifacts contain no local answer transcript or raw generated text.'}",
        f"- shareable: `saved_artifacts={bool((report.get('shareable_summary') or {}).get('saved_artifacts_public_safe'))} raw_prompt_public={bool((report.get('shareable_summary') or {}).get('raw_prompt_public'))} raw_generated_text_public={bool((report.get('shareable_summary') or {}).get('raw_generated_text_public'))} generated_token_ids_public={bool((report.get('shareable_summary') or {}).get('generated_token_ids_public'))} answer_scope_state={(report.get('shareable_summary') or {}).get('answer_scope_state')} local_answer_terminal_only={bool((report.get('shareable_summary') or {}).get('local_answer_terminal_only'))}`",
        f"- note: {(report.get('answer_scope') or {}).get('summary')}",
        "",
        "## Boundaries",
        "",
        "- Coordinator remains the lease/result authority.",
        "- This is not Hivemind/Petals production parity.",
        "- This is not production libp2p/Kademlia/NAT traversal unless separately proven.",
    ]
    return "\n".join(lines) + "\n"


def sanitize_report(report: dict[str, Any], *, output_dir: Path, secret_values: list[str]) -> dict[str, Any]:
    report.setdefault("output_request", output_request_summary())
    report.setdefault(
        "prompt_scope",
        normalize_prompt_scope({"source": "imported-or-built-in-validation-prompts", "prompt_count": 0}),
    )
    report.setdefault("answer_scope", answer_scope_summary())
    report.setdefault("shareable_summary", shareable_summary())
    report = support_bundle.sanitize(redact_values(report, secret_values))
    errors = validate_public_report(report, secret_values=secret_values)
    if errors:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"public_report_safety_failed"})
        report["safety_errors"] = errors
    artifacts = report.setdefault("artifacts", {})
    artifacts.setdefault("real_p2p_swarm_inference_core_rc_json", {
        "kind": "real_p2p_swarm_inference_core_rc",
        "path": "real_p2p_swarm_inference_core_rc.json",
        "present": True,
        "schema": SCHEMA,
        "ok": report.get("ok"),
    })
    artifacts.setdefault("real_p2p_swarm_inference_core_rc_markdown", {
        "kind": "real_p2p_swarm_inference_core_rc_markdown",
        "path": "real_p2p_swarm_inference_core_rc.md",
        "present": True,
    })
    write_json(output_dir / "real_p2p_swarm_inference_core_rc.json", report)
    (output_dir / "real_p2p_swarm_inference_core_rc.md").write_text(render_markdown(report), encoding="utf-8")
    bundle = support_bundle.sanitize({
        "schema": SUPPORT_SCHEMA,
        "ok": report.get("ok"),
        "diagnosis_codes": report.get("diagnosis_codes"),
        "p2p": report.get("p2p"),
        "external": report.get("external"),
        "generation": report.get("generation"),
        "stage_assignment": report.get("stage_assignment"),
        "output_request": report.get("output_request"),
        "prompt_scope": report.get("prompt_scope"),
        "answer_scope": report.get("answer_scope"),
        "shareable_summary": report.get("shareable_summary"),
        "safety": report.get("safety"),
    })
    write_json(output_dir / "support_bundle.json", bundle)
    report["artifacts"]["support_bundle_json"] = {
        "kind": "real_p2p_swarm_inference_core_rc_support_bundle",
        "path": "support_bundle.json",
        "present": True,
        "schema": SUPPORT_SCHEMA,
        "ok": bundle.get("ok"),
    }
    write_json(output_dir / "real_p2p_swarm_inference_core_rc.json", report)
    return report


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == MODE_PACKAGE:
        return build_package(args, output_dir=output_dir)
    if args.mode == MODE_EVIDENCE_IMPORT:
        return build_evidence_import(args, output_dir=output_dir)
    if args.mode == MODE_EXTERNAL_EXISTING:
        return run_external_existing(args, output_dir=output_dir, runner=runner)
    if args.mode == MODE_KAGGLE_CONNECTIVITY:
        return run_kaggle_connectivity(args, output_dir=output_dir)
    if args.mode == MODE_KAGGLE_RUNTIME_SMOKE:
        return run_kaggle_runtime_smoke(args, output_dir=output_dir)
    if args.mode == MODE_KAGGLE_AUTO:
        return run_kaggle_auto(args, output_dir=output_dir)
    missing = missing_hf_dependencies()
    if missing:
        return sanitize_report(degraded_report(args, output_dir, missing), output_dir=output_dir, secret_values=[])
    return run_local_smoke(args, output_dir=output_dir, runner=runner)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Real P2P Swarm Inference Core RC evidence.")
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--swarm-id", default="real-p2p-core-rc")
    parser.add_argument("--public-host", default="24.199.118.54")
    parser.add_argument("--p2p-port", type=int, default=DEFAULT_P2P_PORT)
    parser.add_argument("--coordinator-port", type=int, default=DEFAULT_COORDINATOR_PORT)
    parser.add_argument("--libp2p-port", type=int, default=0)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--hf-model-id", default=DEFAULT_HF_MODEL_ID)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--prompt-text", default="CrowdTensor real P2P core RC")
    parser.add_argument("--prompt-texts", default="", help="comma-separated bounded batch of up to 4 prompts for external verify-generate")
    parser.add_argument("--stream-generation", action="store_true", help="require safe generate --stream progress evidence for external verify-generate")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--session-queue-timeout", type=float, default=45.0)
    parser.add_argument("--miner-timeout", type=float, default=180.0)
    parser.add_argument("--generate-timeout", type=float, default=180.0)
    parser.add_argument("--http-timeout", type=float, default=10.0)
    parser.add_argument("--peer-secret", default=os.environ.get("CROWDTENSOR_P2P_PEER_SECRET", ""))
    parser.add_argument("--peer-bootstrap", default="")
    parser.add_argument("--admin-token", default=os.environ.get("CROWDTENSOR_ADMIN_TOKEN", ""))
    parser.add_argument("--verify-generate", action="store_true")
    parser.add_argument("--discovery-backend", choices=sorted(DISCOVERY_BACKENDS), default="http-provider-store")
    parser.add_argument("--real-p2p-report", default="")
    parser.add_argument("--kaggle-owner", default=default_kaggle_owner())
    parser.add_argument("--kernel-slug-prefix", default="")
    parser.add_argument("--kaggle-push-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--kaggle-stage-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--kaggle-status-poll-seconds", type=float, default=15.0)
    parser.add_argument("--failure-mode", choices=FAILURE_MODES, default=FAILURE_NONE)
    parser.add_argument("--lease-seconds", type=float, default=15.0)
    parser.add_argument("--compute-seconds", type=float, default=0.2)
    parser.add_argument("--victim-compute-seconds", type=float, default=45.0)
    parser.add_argument("--claim-observe-timeout", type=float, default=180.0)
    parser.add_argument("--requeue-timeout", type=float, default=120.0)
    parser.add_argument("--max-request-attempts", type=int, default=240)
    parser.add_argument("--skip-kaggle-cleanup", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.p2p_port < 1 or args.coordinator_port < 1:
        raise SystemExit("--p2p-port and --coordinator-port must be positive")
    if args.libp2p_port < 0:
        raise SystemExit("--libp2p-port must be non-negative")
    if args.mode in {MODE_KAGGLE_AUTO, MODE_KAGGLE_CONNECTIVITY, MODE_KAGGLE_RUNTIME_SMOKE} and args.discovery_backend in LIBP2P_BACKENDS and args.libp2p_port == 0:
        args.libp2p_port = int(args.p2p_port) + DEFAULT_LIBP2P_PORT_OFFSET
    if args.max_new_tokens < 2 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 2 and 32")
    if args.mode == MODE_EXTERNAL_EXISTING and not args.peer_bootstrap:
        raise SystemExit("external-existing requires --peer-bootstrap")
    if args.mode == MODE_EVIDENCE_IMPORT and not args.real_p2p_report:
        raise SystemExit("evidence-import requires --real-p2p-report")
    if (args.prompt_texts or args.stream_generation) and args.mode != MODE_EXTERNAL_EXISTING:
        raise SystemExit("--prompt-texts and --stream-generation are currently supported for external-existing only")
    if (args.prompt_texts or args.stream_generation) and not args.verify_generate:
        raise SystemExit("--prompt-texts and --stream-generation require --verify-generate")
    if args.mode in {MODE_KAGGLE_AUTO, MODE_KAGGLE_CONNECTIVITY, MODE_KAGGLE_RUNTIME_SMOKE} and not args.kaggle_owner:
        raise SystemExit(f"{args.mode} requires --kaggle-owner or KAGGLE_USERNAME/~/.kaggle/kaggle.json")
    for name in ["startup_timeout", "timeout_seconds", "session_queue_timeout", "miner_timeout", "generate_timeout", "http_timeout"]:
        if float(getattr(args, name)) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    for name in ["kaggle_push_timeout_seconds", "kaggle_delete_timeout_seconds", "kaggle_stage_timeout_seconds", "kaggle_status_poll_seconds"]:
        if float(getattr(args, name)) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    for name in ["lease_seconds", "victim_compute_seconds", "claim_observe_timeout", "requeue_timeout"]:
        if float(getattr(args, name)) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.compute_seconds < 0:
        raise SystemExit("--compute-seconds must be non-negative")
    if args.max_request_attempts < 1:
        raise SystemExit("--max-request-attempts must be at least 1")
    product_mvp.parse_prompt_texts_arg(args.prompt_text, args.prompt_texts)
    if args.failure_mode != FAILURE_NONE and args.victim_compute_seconds <= args.lease_seconds:
        args.victim_compute_seconds = args.lease_seconds + 30.0
    if args.failure_mode != FAILURE_NONE and args.requeue_timeout <= args.lease_seconds:
        args.requeue_timeout = args.lease_seconds + 45.0
    return args


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"Real P2P Swarm Inference Core RC ready: {report.get('ok')}")
        print(f"Diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
