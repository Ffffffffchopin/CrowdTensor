#!/usr/bin/env python3
"""Build the P2P Swarm Inference v0.6 prototype evidence artifact.

This is the first Coordinator-to-P2P transition artifact.  It makes P2P
discovery/routing real for the product commands while keeping Coordinator
leases, validation, and result ledgers as the execution authority.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import public_swarm_preview_v04_pack as preview_v04  # noqa: E402
import product_swarm_mvp_check as product_mvp  # noqa: E402
import support_bundle  # noqa: E402
from crowdtensor.session_protocol import build_session_request  # noqa: E402
from crowdtensor.session_protocol import public_leak_paths  # noqa: E402
from crowdtensor.real_llm import missing_hf_dependencies  # noqa: E402


SCHEMA = "p2p_swarm_inference_v06_v1"
MODE_LOCAL_SMOKE = "local-smoke"
MODE_PACKAGE = "package"
MODE_EVIDENCE_IMPORT = "evidence-import"
MODE_EXTERNAL_EXISTING = "external-existing"
MODE_KAGGLE_AUTO = "kaggle-auto"
MODES = [MODE_LOCAL_SMOKE, MODE_PACKAGE, MODE_EVIDENCE_IMPORT, MODE_EXTERNAL_EXISTING, MODE_KAGGLE_AUTO]
DEFAULT_OUTPUT_DIR = "dist/p2p-swarm-inference-v06"
DEFAULT_PREVIEW_V04_REPORT = "dist/public-swarm-preview-v04-final/public_swarm_preview_v04.json"
DEFAULT_PRODUCT_MVP_REPORT = "dist/public-swarm-preview-v04-distilgpt2-strict/product-mvp/product_swarm_mvp_check.json"
DEFAULT_OPTIONAL_MODEL_REPORT = "dist/public-swarm-preview-v04-distilgpt2-strict/optional-model-mvp/product_swarm_mvp_check.json"
DEFAULT_P2P_PORT = 9560
DEFAULT_COORDINATOR_PORT = 9561
DEFAULT_HF_MODEL_ID = "sshleifer/tiny-gpt2"
WORKLOAD_TYPE = "real_llm_sharded_infer"
ADMIN_TOKEN = "p2p-v06-admin"
MINER_TOKEN = "p2p-v06-miner"
OBSERVER_TOKEN = "p2p-v06-observer"
Runner = Callable[..., subprocess.CompletedProcess[str]]

SECRET_FRAGMENTS = (
    ADMIN_TOKEN,
    MINER_TOKEN,
    OBSERVER_TOKEN,
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "lease_token",
    "idempotency_key",
    "Bearer ",
    "hidden_state",
    "input_ids",
    "logits",
    "activation_results",
    '"generated_text":',
    '"generated_token_ids":',
    '"prompt_text":',
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
)
LOCAL_P2P_DISCOVERY_CODES = {
    "p2pd_daemon_ready",
    "local_three_process_p2p_discovery_ready",
    "p2p_stage_discovery_ready",
    "p2p_generate_route_ready",
    "p2p_stage_rescue_ready",
    "p2p_real_generate_ready",
    "p2p_real_generate_kv_cache_ready",
    "p2p_real_generate_stream_ready",
    "p2p_real_generate_hf_runtime_missing",
    "public_swarm_generate_stream_ready",
    "public_swarm_generate_stream_endpoint_ready",
    "p2p_real_stage_rescue_ready",
    "p2p_real_stage_rescue_hf_runtime_missing",
    "p2p_real_stage_rescue_blocked",
    "p2p_rescue_generation_completed",
    "stage0_rescue_generation_completed",
    "stage1_rescue_generation_completed",
    "stage0_victim_requeued",
    "stage1_victim_requeued",
    "stage0_rescue_peer_discovered",
    "stage1_rescue_peer_discovered",
    "hf_dependencies_missing",
    "coordinator_to_p2p_transition_ready",
    "coordinator_result_fallback_ready",
    "real_llm_stage0_kv_cache_v1_ready",
    "real_llm_stage1_kv_cache_v1_ready",
    "stage0_kv_cache_hits_ready",
    "stage1_kv_cache_hits_ready",
}
EXTERNAL_P2P_CODES = {
    "external_p2p_runtime_verified",
    "external_p2p_route_ready",
    "external_p2p_stage_discovery_ready",
    "external_p2p_generate_ready",
    "external_p2p_generate_not_requested",
    "admin_token_required",
}
KAGGLE_CODE_URL = re.compile(r"https://www\.kaggle\.com/code/([^/\s]+)/([^/\s]+)")
SUPPRESSED_INHERITED_CODES = {
    "not_p2p",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def stream_evidence_ready(stream: dict[str, Any], batch: dict[str, Any] | None = None) -> bool:
    if not isinstance(stream, dict) or not stream.get("stream_generation_ready"):
        return False
    progress = stream.get("progress") if isinstance(stream.get("progress"), dict) else {}
    expected_requests = safe_int(progress.get("expected_request_count") or (batch or {}).get("expected_request_count") or (batch or {}).get("request_count"), 1)
    if expected_requests > 1 or bool((batch or {}).get("enabled")):
        return bool(
            progress.get("per_request_progress")
            and progress.get("per_request_progress_complete") is True
            and progress.get("per_request_monotonic_progress") is True
        )
    return bool(progress.get("stream_progress_complete") is True and progress.get("monotonic_progress") is True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in str(stdout or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def p2p_generate_command(
    *,
    p2p_url: str,
    backend: str,
    hf_model_id: str,
    max_new_tokens: int | str,
    admin_token: str = "",
    timeout_seconds: float | str | None = None,
    http_timeout: float | str | None = None,
    prompt_text: str = "",
    prompt_texts: str = "",
    prompt_texts_file: str = "",
    dry_run: bool = False,
    stream: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "generate",
        "--p2p",
        "--peer-bootstrap",
        p2p_url,
    ]
    if prompt_texts_file:
        command.extend(["--prompt-texts-file", prompt_texts_file])
    elif prompt_texts:
        command.extend(["--prompt-texts", prompt_texts])
    elif prompt_text:
        command.extend(["--prompt", prompt_text])
    if admin_token:
        command.extend(["--admin-token", admin_token])
    command.extend([
        "--backend",
        backend,
        "--hf-model-id",
        hf_model_id,
        "--max-new-tokens",
        str(max_new_tokens),
    ])
    if timeout_seconds is not None:
        command.extend(["--timeout-seconds", str(timeout_seconds)])
    if http_timeout is not None:
        command.extend(["--http-timeout", str(http_timeout)])
    if dry_run:
        command.append("--dry-run")
    if stream:
        command.append("--stream")
    command.append("--json")
    return command


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


def diagnosis_codes(*payloads: dict[str, Any], extra: list[str] | None = None) -> list[str]:
    codes: set[str] = set(extra or [])
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for code in payload.get("diagnosis_codes") or []:
            if isinstance(code, str):
                codes.add(code)
        for summary in (payload.get("payload_summaries") or {}).values() if isinstance(payload.get("payload_summaries"), dict) else []:
            if isinstance(summary, dict):
                for code in summary.get("diagnosis_codes") or []:
                    if isinstance(code, str):
                        codes.add(code)
    return sorted(codes)


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


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    entry: dict[str, Any] = {"kind": kind, "path": relative, "present": path.is_file()}
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def output_request_summary() -> dict[str, Any]:
    return {
        "include_output": False,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "public_artifact_safe": True,
        "summary": (
            "P2P Swarm Inference v0.6 public artifacts summarize discovery, routing, "
            "safe generation hashes/counts, batch/stream readiness, KV-cache evidence, "
            "and stage rescue proof only. Run `crowdtensor generate --p2p` in local "
            "human mode to display answer text."
        ),
    }


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
            "This P2P Swarm Inference v0.6 report is shareable route and readiness "
            "evidence, not an answer transcript. Raw prompts, generated text, generated "
            "token ids, activations, lease tokens, peer secrets, private env files, "
            "Kaggle kernel payloads, and runtime state are excluded."
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
            "Share `p2p_swarm_inference_v06.json`, `p2p_swarm_inference_v06.md`, "
            "and `support_bundle.json`; they contain P2P route evidence, hashes, "
            "counts, and readiness summaries, not raw prompts or answers."
        ),
    }


def request_json(base_url: str, path: str, *, timeout: float = 5.0, bearer_token: str = "") -> dict[str, Any]:
    request = Request(f"{base_url.rstrip('/')}{path}", method="GET")
    if bearer_token:
        request.add_header("Authorization", f"Bearer {bearer_token}")
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def run_json_step(name: str, command: list[str], *, runner: Runner, timeout_seconds: float) -> tuple[dict[str, Any], dict[str, Any]]:
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
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "ok": False,
            "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": "timeout",
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
        step["stdout_tail"] = redact_text((completed.stdout or "")[-1200:])
        step["stderr_tail"] = redact_text((completed.stderr or "")[-1200:])
    return step, payload


def popen_process(command: list[str]) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def safe_slug(value: str) -> str:
    cleaned: list[str] = []
    last_dash = False
    for char in str(value or "").lower():
        if char.isalnum():
            cleaned.append(char)
            last_dash = False
        elif not last_dash:
            cleaned.append("-")
            last_dash = True
    return "".join(cleaned).strip("-") or "crowdtensor-p2p-v06"


def default_kaggle_owner() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return str(os.environ["KAGGLE_USERNAME"])
    config = Path.home() / ".kaggle" / "kaggle.json"
    if config.is_file():
        try:
            payload = json.loads(config.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return ""
        username = payload.get("username")
        return str(username) if username else ""
    return ""


def extract_kernel_ref(text: str) -> str:
    match = KAGGLE_CODE_URL.search(text or "")
    if not match:
        return ""
    return f"{match.group(1)}/{match.group(2)}"


def stop_process(proc: subprocess.Popen[str] | None) -> dict[str, Any]:
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
    return {
        "returncode": proc.returncode,
        "stdout_tail": redact_text((stdout or "")[-1000:]),
        "stderr_tail": redact_text((stderr or "")[-1000:]),
    }


def terminate_process_group(proc: subprocess.Popen[str], *, sig: int = signal.SIGTERM) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


def wait_p2pd(base_url: str, proc: subprocess.Popen[str], *, timeout: float) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() <= deadline:
        if proc.poll() is not None:
            details = stop_process(proc)
            return False, f"p2pd exited early: {details}"
        try:
            payload = request_json(base_url, "/peer/health", timeout=2.0)
            if payload.get("ok") is True:
                return True, ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.2)
    return False, f"p2pd did not become healthy: {last_error}"


def wait_v06_workload_queued(base_url: str, *, timeout: float) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() <= deadline:
        try:
            state = product_mvp.request_json(
                "GET",
                base_url,
                "/state",
                observer_token=OBSERVER_TOKEN,
                timeout=2.0,
            )
            tasks = state.get("tasks") if isinstance(state.get("tasks"), list) else []
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                metadata = task.get("workload_metadata") if isinstance(task.get("workload_metadata"), dict) else {}
                if (
                    task.get("status") == "queued"
                    and task.get("workload_type") == WORKLOAD_TYPE
                    and int(metadata.get("stage_id", -1)) == 0
                ):
                    return True, ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.2)
    return False, f"stage0 task did not become queued: {last_error}"


def wait_v06_workload_started(base_url: str, *, timeout: float) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout
    last_error = ""
    visible_statuses = {"queued", "leased", "completed"}
    while time.monotonic() <= deadline:
        try:
            state = product_mvp.request_json(
                "GET",
                base_url,
                "/state",
                observer_token=OBSERVER_TOKEN,
                timeout=2.0,
            )
            tasks = state.get("tasks") if isinstance(state.get("tasks"), list) else []
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                metadata = task.get("workload_metadata") if isinstance(task.get("workload_metadata"), dict) else {}
                if (
                    task.get("status") in visible_statuses
                    and task.get("workload_type") == WORKLOAD_TYPE
                    and int(metadata.get("stage_id", -1)) == 0
                ):
                    return True, ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.2)
    return False, f"stage0 task did not become visible: {last_error}"


def wait_v06_stage_queued(base_url: str, *, stage_id: int, timeout: float) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() <= deadline:
        try:
            state = product_mvp.request_json(
                "GET",
                base_url,
                "/state",
                observer_token=OBSERVER_TOKEN,
                timeout=2.0,
            )
            tasks = state.get("tasks") if isinstance(state.get("tasks"), list) else []
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                metadata = task.get("workload_metadata") if isinstance(task.get("workload_metadata"), dict) else {}
                if (
                    task.get("status") == "queued"
                    and task.get("workload_type") == WORKLOAD_TYPE
                    and int(metadata.get("stage_id", -1)) == int(stage_id)
                ):
                    return True, ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.2)
    return False, f"stage{stage_id} task did not become queued: {last_error}"


def wait_v06_task_status(base_url: str, task_id: str, status: str, *, timeout: float) -> tuple[bool, dict[str, Any]]:
    deadline = time.monotonic() + timeout
    last_state: dict[str, Any] = {}
    while time.monotonic() <= deadline:
        try:
            last_state = product_mvp.request_json(
                "GET",
                base_url,
                "/state",
                observer_token=OBSERVER_TOKEN,
                timeout=2.0,
            )
            tasks = last_state.get("tasks") if isinstance(last_state.get("tasks"), list) else []
            for task in tasks:
                if isinstance(task, dict) and task.get("task_id") == task_id and task.get("status") == status:
                    return True, last_state
        except Exception:
            pass
        time.sleep(0.2)
    return False, last_state


def v06_admin_results(
    base_url: str,
    *,
    session_id: str = "",
    limit: int = 100,
    timeout: float = 10.0,
    admin_token: str = ADMIN_TOKEN,
) -> list[dict[str, Any]]:
    query = {
        "status": "accepted",
        "workload_type": WORKLOAD_TYPE,
        "limit": int(limit),
    }
    if session_id:
        query["session_id"] = session_id
    payload = product_mvp.request_json(
        "GET",
        base_url,
        f"/admin/results?{urlencode(query)}",
        admin_token=admin_token,
        timeout=timeout,
    )
    rows = payload.get("results") if isinstance(payload.get("results"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def claim_from_stdout(stdout: str) -> dict[str, Any]:
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("claimed task="):
            continue
        fields: dict[str, str] = {}
        for part in line.split():
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            fields[key] = value
        if fields.get("task"):
            try:
                attempt = int(fields.get("attempt") or 0)
            except ValueError:
                attempt = 0
            return {"task_id": fields["task"], "attempt": attempt, "line": line}
    return {}


def wait_for_claim_line(proc: subprocess.Popen[str], *, timeout: float) -> tuple[str, dict[str, Any]]:
    import select

    deadline = time.monotonic() + timeout
    stdout_lines: list[str] = []
    assert proc.stdout is not None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        ready, _, _ = select.select([proc.stdout], [], [], 0.1)
        if not ready:
            continue
        line = proc.stdout.readline()
        if not line:
            continue
        stdout_lines.append(line)
        claim = claim_from_stdout("".join(stdout_lines))
        if claim:
            return "".join(stdout_lines), claim
    return "".join(stdout_lines), {}


def process_result(
    miner_id: str,
    proc: subprocess.Popen[str],
    *,
    expected_failure: bool = False,
    stdout_prefix: str = "",
) -> dict[str, Any]:
    stdout = stdout_prefix
    stderr = ""
    try:
        more_out, stderr = proc.communicate(timeout=2.0)
        stdout += more_out or ""
    except subprocess.TimeoutExpired:
        terminate_process_group(proc, sig=signal.SIGKILL)
        more_out, stderr = proc.communicate(timeout=2.0)
        stdout += more_out or ""
    payload: dict[str, Any] = {}
    for line in reversed(stdout.strip().splitlines()):
        if not line.strip().startswith("{"):
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "accepted_tasks" in candidate:
            payload = candidate
            break
    return {
        "miner_id": miner_id,
        "ok": bool(proc.returncode == 0 and int(payload.get("accepted_tasks") or 0) >= 1),
        "expected_failure": expected_failure,
        "returncode": proc.returncode,
        "accepted_tasks": int(payload.get("accepted_tasks") or 0),
        "stdout_tail": redact_text((stdout or "")[-1000:]),
        "stderr_tail": redact_text((stderr or "")[-1000:]),
    }


def stage_peer_count(catalog: dict[str, Any]) -> tuple[int, int, int]:
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
        cap_values = caps.get("real_llm_sharded_stage_capabilities") if isinstance(caps.get("real_llm_sharded_stage_capabilities"), list) else []
        if "real_llm_sharded_stage0" in cap_values or "real_llm_sharded_cuda_stage0" in cap_values:
            stage0 += 1
        if "real_llm_sharded_stage1" in cap_values or "real_llm_sharded_cuda_stage1" in cap_values:
            stage1 += 1
    return coordinators, stage0, stage1


def wait_p2p_stage_miners(base_url: str, *, timeout: float, http_timeout: float) -> tuple[bool, dict[str, Any], str]:
    deadline = time.monotonic() + timeout
    last_error = ""
    last_catalog: dict[str, Any] = {}
    while time.monotonic() <= deadline:
        try:
            last_catalog = request_json(base_url, "/peer/catalog", timeout=http_timeout)
            _coordinators, stage0, stage1 = stage_peer_count(last_catalog)
            if stage0 >= 1 and stage1 >= 1:
                return True, last_catalog, ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2.0)
    return False, last_catalog, last_error or "stage miners did not announce before timeout"


def route_from_catalog(
    catalog: dict[str, Any],
    *,
    prompt_text: str,
    backend: str,
    hf_model_id: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    from crowdtensor.session_protocol import build_route_decision

    session_request = build_session_request(
        prompt_text=prompt_text,
        backend=backend,
        hf_model_id=hf_model_id,
        stage_mode="split",
        max_new_tokens=max_new_tokens,
        route_source="p2p-discovery",
    )
    peers = [peer for peer in (catalog.get("peers") or []) if isinstance(peer, dict)]
    return build_route_decision(session_request, coordinator_url="", peer_catalog=peers)


def resolve_route(
    base_url: str,
    *,
    prompt_text: str,
    backend: str,
    hf_model_id: str,
    max_new_tokens: int,
    http_timeout: float,
) -> dict[str, Any]:
    payload = {
        "session_request": build_session_request(
            prompt_text=prompt_text,
            backend=backend,
            hf_model_id=hf_model_id,
            stage_mode="split",
            max_new_tokens=max_new_tokens,
            route_source="p2p-discovery",
        )
    }
    request = Request(
        f"{base_url.rstrip('/')}/peer/resolve",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=http_timeout) as response:
        raw = response.read().decode("utf-8")
        result = json.loads(raw) if raw else {}
    route = result.get("route") if isinstance(result, dict) and isinstance(result.get("route"), dict) else {}
    return route


def run_single_stage_rescue_probe(
    args: argparse.Namespace,
    *,
    stage: str,
    port: int,
) -> dict[str, Any]:
    from crowdtensor.cli import build_p2p_peer  # Imported lazily to keep this script usable as a standalone checker.
    from crowdtensor.p2p_lite import post_announce

    base_url = f"http://127.0.0.1:{port}"
    proc: subprocess.Popen[str] | None = None
    try:
        proc = popen_process([
            sys.executable,
            str(ROOT / "scripts" / "p2p_lite_daemon.py"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--swarm-id",
            args.swarm_id,
            "--role",
            "observer",
        ])
        healthy, health_error = wait_p2pd(base_url, proc, timeout=args.startup_timeout)
        if not healthy:
            return {
                "ok": False,
                "stage": stage,
                "error": health_error,
                "diagnosis_codes": [f"{stage}_rescue_probe_start_failed"],
                "process": stop_process(proc),
            }
        victim_id = f"p2p-v06-victim-{stage}"
        rescue_id = f"p2p-v06-rescue-{stage}"
        coordinator = build_p2p_peer(
            swarm_id=args.swarm_id,
            peer_id=f"p2p-v06-rescue-coordinator-{stage}",
            role="coordinator",
            coordinator_url=f"http://127.0.0.1:{args.coordinator_port}",
            backend=args.backend,
            hf_model_id=args.hf_model_id,
            ttl_seconds=max(10.0, float(args.startup_timeout)),
        )
        opposite_stage = "stage1" if stage == "stage0" else "stage0"
        opposite = build_p2p_peer(
            swarm_id=args.swarm_id,
            peer_id=f"p2p-v06-rescue-static-{opposite_stage}",
            role="miner",
            backend=args.backend,
            stage_role=opposite_stage,
            hf_model_id=args.hf_model_id,
            ttl_seconds=max(10.0, float(args.startup_timeout)),
        )
        victim = build_p2p_peer(
            swarm_id=args.swarm_id,
            peer_id=victim_id,
            role="miner",
            backend=args.backend,
            stage_role=stage,
            hf_model_id=args.hf_model_id,
            ttl_seconds=1.0,
        )
        rescue = build_p2p_peer(
            swarm_id=args.swarm_id,
            peer_id=rescue_id,
            role="miner",
            backend=args.backend,
            stage_role=stage,
            hf_model_id=args.hf_model_id,
            ttl_seconds=max(10.0, float(args.startup_timeout)),
        )
        post_announce(base_url, coordinator, timeout=args.http_timeout)
        post_announce(base_url, opposite, timeout=args.http_timeout)
        post_announce(base_url, victim, timeout=args.http_timeout)
        route_before = resolve_route(
            base_url,
            prompt_text=args.prompt_text,
            backend=args.backend,
            hf_model_id=args.hf_model_id,
            max_new_tokens=args.max_new_tokens,
            http_timeout=args.http_timeout,
        )
        time.sleep(1.3)
        post_announce(base_url, rescue, timeout=args.http_timeout)
        route_after = resolve_route(
            base_url,
            prompt_text=args.prompt_text,
            backend=args.backend,
            hf_model_id=args.hf_model_id,
            max_new_tokens=args.max_new_tokens,
            http_timeout=args.http_timeout,
        )
        cap = "real_llm_sharded_stage0" if stage == "stage0" else "real_llm_sharded_stage1"
        before_matches = route_before.get("matched_capabilities") if isinstance(route_before.get("matched_capabilities"), dict) else {}
        after_matches = route_after.get("matched_capabilities") if isinstance(route_after.get("matched_capabilities"), dict) else {}
        stage_ready = bool(before_matches.get(cap) == victim_id and after_matches.get(cap) == rescue_id and route_after.get("usable_now"))
        return {
            "ok": stage_ready,
            "stage": stage,
            "probe_url": base_url,
            "victim_peer_id": victim_id,
            "rescue_peer_id": rescue_id,
            "matched_before": before_matches.get(cap),
            "matched_after": after_matches.get(cap),
            "route_before_usable": bool(route_before.get("usable_now")),
            "route_after_usable": bool(route_after.get("usable_now")),
            "diagnosis_codes": [f"{stage}_rescue_peer_discovered"] if stage_ready else [f"{stage}_rescue_peer_blocked"],
            "process": stop_process(proc),
        }
    finally:
        if proc is not None and proc.poll() is None:
            stop_process(proc)


def run_stage_rescue_probe(args: argparse.Namespace) -> dict[str, Any]:
    rescue_results: dict[str, Any] = {}
    codes: set[str] = set()
    for stage in ("stage0", "stage1"):
        result = run_single_stage_rescue_probe(args, stage=stage, port=find_free_port())
        rescue_results[stage] = result
        codes.update(code for code in result.get("diagnosis_codes") or [] if isinstance(code, str))
    ready = bool(rescue_results.get("stage0", {}).get("ok") and rescue_results.get("stage1", {}).get("ok"))
    if ready:
        codes.add("p2p_stage_rescue_ready")
    return {
        "ok": ready,
        "schema": "p2p_stage_rescue_probe_v1",
        "results": rescue_results,
        "diagnosis_codes": sorted(codes or {"p2p_stage_rescue_blocked"}),
    }


def run_real_stage_rescue_probe(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    if product_mvp.missing_hf_dependencies():
        return support_bundle.sanitize({
            "schema": "p2p_real_stage_rescue_probe_v1",
            "ok": False,
            "degraded": True,
            "diagnosis_codes": ["hf_dependencies_missing", "p2p_real_stage_rescue_hf_runtime_missing"],
            "operator_action": "Install optional runtime dependencies with: python -m pip install -e '.[hf]'",
        })
    results: dict[str, Any] = {}
    codes: set[str] = set()
    for stage in ("stage0", "stage1"):
        stage_dir = output_dir / stage
        result = run_single_real_stage_rescue_probe(args, output_dir=stage_dir, failure_stage=stage)
        results[stage] = result
        codes.update(code for code in result.get("diagnosis_codes") or [] if isinstance(code, str))
    ready = bool(results.get("stage0", {}).get("ok") and results.get("stage1", {}).get("ok"))
    if ready:
        codes.update({
            "p2p_real_stage_rescue_ready",
            "p2p_rescue_generation_completed",
            "stage0_rescue_generation_completed",
            "stage1_rescue_generation_completed",
        })
    else:
        codes.add("p2p_real_stage_rescue_blocked")
    return support_bundle.sanitize(redact_values({
        "schema": "p2p_real_stage_rescue_probe_v1",
        "ok": ready,
        "output_dir": str(output_dir),
        "results": results,
        "diagnosis_codes": sorted(codes),
        "safety": {
            "p2p_discovery_routing": True,
            "coordinator_result_fallback": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "not_production": True,
        },
    }))


def run_single_real_stage_rescue_probe(args: argparse.Namespace, *, output_dir: Path, failure_stage: str) -> dict[str, Any]:
    from crowdtensor.cli import build_p2p_peer
    from crowdtensor.p2p_lite import post_announce
    from crowdtensor.session_protocol import safe_generation_summary

    hf_cache_dir = str(getattr(args, "hf_cache_dir", "") or "")
    p2p_port = find_free_port()
    coordinator_port = find_free_port()
    p2p_url = f"http://127.0.0.1:{p2p_port}"
    coordinator_url = f"http://127.0.0.1:{coordinator_port}"
    state_dir = output_dir / "state"
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    p2pd_proc: subprocess.Popen[str] | None = None
    serve_proc: subprocess.Popen[str] | None = None
    generate_proc: subprocess.Popen[str] | None = None
    victim_proc: subprocess.Popen[str] | None = None
    steps: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {}
    stage_id = 0 if failure_stage == "stage0" else 1
    victim_id = f"p2p-v06-rescue-{failure_stage}-victim"
    rescue_id = f"p2p-v06-rescue-{failure_stage}-rescue"
    stable_stage_id = f"p2p-v06-rescue-{'stage1' if failure_stage == 'stage0' else 'stage0'}"
    try:
        p2pd_proc = popen_process([
            sys.executable,
            str(ROOT / "scripts" / "p2p_lite_daemon.py"),
            "--host",
            "127.0.0.1",
            "--port",
            str(p2p_port),
            "--swarm-id",
            args.swarm_id,
            "--role",
            "observer",
        ])
        healthy, health_error = wait_p2pd(p2p_url, p2pd_proc, timeout=args.startup_timeout)
        steps.append({"name": "p2pd", "ok": healthy, "error": health_error})
        if not healthy:
            return finish_real_rescue_report(args, output_dir=output_dir, base_url=coordinator_url, failure_stage=failure_stage, steps=steps, payloads=payloads, p2pd_proc=p2pd_proc, serve_proc=serve_proc, generate_proc=generate_proc, victim_proc=victim_proc)

        serve_cmd = [
            sys.executable,
            "-m",
            "crowdtensor.cli",
            "serve",
            "--p2p",
            "--peer-bootstrap",
            p2p_url,
            "--swarm-id",
            args.swarm_id,
            "--profile",
            "cpu-real-llm",
            "--bind-host",
            "127.0.0.1",
            "--public-host",
            "127.0.0.1",
            "--port",
            str(coordinator_port),
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
            "--run",
            "--json",
        ]
        if hf_cache_dir:
            serve_cmd.extend(["--hf-cache-dir", hf_cache_dir])
        serve_proc = product_mvp.popen_command(serve_cmd)
        serve_healthy, serve_error = product_mvp.wait_health(coordinator_url, serve_proc, args.startup_timeout)
        steps.append({"name": "serve_p2p_run", "ok": serve_healthy, "error": serve_error})
        if not serve_healthy:
            return finish_real_rescue_report(args, output_dir=output_dir, base_url=coordinator_url, failure_stage=failure_stage, steps=steps, payloads=payloads, p2pd_proc=p2pd_proc, serve_proc=serve_proc, generate_proc=generate_proc, victim_proc=victim_proc)

        for stage in ("stage0", "stage1"):
            peer_id = victim_id if stage == failure_stage else stable_stage_id
            peer = build_p2p_peer(
                swarm_id=args.swarm_id,
                peer_id=peer_id,
                role="miner",
                backend=args.backend,
                stage_role=stage,
                hf_model_id=args.hf_model_id,
                ttl_seconds=1.0 if stage == failure_stage else 60.0,
            )
            post_announce(p2p_url, peer, timeout=args.http_timeout)
            steps.append({"name": f"announce_{stage}_{peer_id}", "ok": True})

        route_before = resolve_route(
            p2p_url,
            prompt_text=args.prompt_text,
            backend=args.backend,
            hf_model_id=args.hf_model_id,
            max_new_tokens=1,
            http_timeout=args.http_timeout,
        )
        payloads["route_before"] = {"route": route_before, "ok": bool(route_before.get("usable_now"))}
        steps.append({"name": "route_before_failure", "ok": bool(route_before.get("usable_now"))})

        generate_cmd = p2p_generate_command(
            p2p_url=p2p_url,
            admin_token=ADMIN_TOKEN,
            backend=args.backend,
            hf_model_id=args.hf_model_id,
            max_new_tokens=1,
            timeout_seconds=args.timeout_seconds,
            prompt_text=args.prompt_text,
            prompt_texts=getattr(args, "prompt_texts", ""),
            prompt_texts_file=getattr(args, "prompt_texts_file", ""),
        )
        generate_proc = product_mvp.popen_command(generate_cmd)
        queued, queued_error = wait_v06_stage_queued(coordinator_url, stage_id=0, timeout=args.timeout_seconds)
        steps.append({"name": "initial_stage0_queued", "ok": queued, "error": queued_error})
        if not queued:
            return finish_real_rescue_report(args, output_dir=output_dir, base_url=coordinator_url, failure_stage=failure_stage, steps=steps, payloads=payloads, p2pd_proc=p2pd_proc, serve_proc=serve_proc, generate_proc=generate_proc, victim_proc=victim_proc)

        if failure_stage == "stage0":
            victim_proc = start_real_miner_process(
                coordinator_url=coordinator_url,
                miner_id=victim_id,
                stage="stage0",
                hf_model_id=args.hf_model_id,
                hf_cache_dir=hf_cache_dir,
                compute_seconds=6.0,
            )
            stdout_prefix, claim = wait_for_claim_line(victim_proc, timeout=min(args.timeout_seconds, 20.0))
            payloads["victim_claim"] = claim
            steps.append({"name": "victim_stage0_claimed", "ok": bool(claim), "claim_line": claim.get("line")})
            if not claim:
                return finish_real_rescue_report(args, output_dir=output_dir, base_url=coordinator_url, failure_stage=failure_stage, steps=steps, payloads=payloads, p2pd_proc=p2pd_proc, serve_proc=serve_proc, generate_proc=generate_proc, victim_proc=victim_proc)
            terminate_process_group(victim_proc, sig=signal.SIGTERM)
            victim_process = process_result(victim_id, victim_proc, expected_failure=True, stdout_prefix=stdout_prefix)
            payloads["victim_process"] = victim_process
            victim_proc = None
            lease_expired, _ = wait_v06_task_status(coordinator_url, str(claim["task_id"]), "queued", timeout=30.0)
            steps.append({"name": "victim_stage0_requeued", "ok": lease_expired})
            post_announce(
                p2p_url,
                build_p2p_peer(
                    swarm_id=args.swarm_id,
                    peer_id=rescue_id,
                    role="miner",
                    backend=args.backend,
                    stage_role="stage0",
                    hf_model_id=args.hf_model_id,
                    ttl_seconds=60.0,
                ),
                timeout=args.http_timeout,
            )
            time.sleep(1.2)
            route_after = resolve_route(
                p2p_url,
                prompt_text=args.prompt_text,
                backend=args.backend,
                hf_model_id=args.hf_model_id,
                max_new_tokens=1,
                http_timeout=args.http_timeout,
            )
            payloads["route_after"] = {"route": route_after, "ok": bool(route_after.get("usable_now"))}
            steps.append({"name": "route_after_stage0_rescue", "ok": bool(route_after.get("usable_now"))})
            rescue_step, rescue_payload = run_p2p_join_miner_step(
                p2p_url=p2p_url,
                swarm_id=args.swarm_id,
                miner_id=rescue_id,
                stage="stage0",
                backend=args.backend,
                hf_model_id=args.hf_model_id,
                hf_cache_dir=hf_cache_dir,
                timeout=args.timeout_seconds,
                http_timeout=args.http_timeout,
            )
            steps.append(rescue_step)
            payloads["rescue_stage0"] = rescue_payload
            stage1_queued, stage1_queued_error = wait_v06_stage_queued(coordinator_url, stage_id=1, timeout=args.timeout_seconds)
            steps.append({"name": "stage1_after_stage0_rescue_queued", "ok": stage1_queued, "error": stage1_queued_error})
            if not stage1_queued:
                return finish_real_rescue_report(args, output_dir=output_dir, base_url=coordinator_url, failure_stage=failure_stage, steps=steps, payloads=payloads, p2pd_proc=p2pd_proc, serve_proc=serve_proc, generate_proc=generate_proc, victim_proc=victim_proc)
            stable_step, stable_payload = run_p2p_join_miner_step(
                p2p_url=p2p_url,
                swarm_id=args.swarm_id,
                miner_id=stable_stage_id,
                stage="stage1",
                backend=args.backend,
                hf_model_id=args.hf_model_id,
                hf_cache_dir=hf_cache_dir,
                timeout=args.timeout_seconds,
                http_timeout=args.http_timeout,
            )
            steps.append(stable_step)
            payloads["stable_stage1"] = stable_payload
        else:
            stable_step, stable_payload = run_p2p_join_miner_step(
                p2p_url=p2p_url,
                swarm_id=args.swarm_id,
                miner_id=stable_stage_id,
                stage="stage0",
                backend=args.backend,
                hf_model_id=args.hf_model_id,
                hf_cache_dir=hf_cache_dir,
                timeout=args.timeout_seconds,
                http_timeout=args.http_timeout,
            )
            steps.append(stable_step)
            payloads["stable_stage0"] = stable_payload
            stage1_queued, stage1_queued_error = wait_v06_stage_queued(coordinator_url, stage_id=1, timeout=args.timeout_seconds)
            steps.append({"name": "stage1_before_victim_queued", "ok": stage1_queued, "error": stage1_queued_error})
            if not stage1_queued:
                return finish_real_rescue_report(args, output_dir=output_dir, base_url=coordinator_url, failure_stage=failure_stage, steps=steps, payloads=payloads, p2pd_proc=p2pd_proc, serve_proc=serve_proc, generate_proc=generate_proc, victim_proc=victim_proc)
            victim_proc = start_real_miner_process(
                coordinator_url=coordinator_url,
                miner_id=victim_id,
                stage="stage1",
                hf_model_id=args.hf_model_id,
                hf_cache_dir=hf_cache_dir,
                compute_seconds=6.0,
            )
            stdout_prefix, claim = wait_for_claim_line(victim_proc, timeout=min(args.timeout_seconds, 20.0))
            payloads["victim_claim"] = claim
            steps.append({"name": "victim_stage1_claimed", "ok": bool(claim), "claim_line": claim.get("line")})
            if not claim:
                return finish_real_rescue_report(args, output_dir=output_dir, base_url=coordinator_url, failure_stage=failure_stage, steps=steps, payloads=payloads, p2pd_proc=p2pd_proc, serve_proc=serve_proc, generate_proc=generate_proc, victim_proc=victim_proc)
            terminate_process_group(victim_proc, sig=signal.SIGTERM)
            victim_process = process_result(victim_id, victim_proc, expected_failure=True, stdout_prefix=stdout_prefix)
            payloads["victim_process"] = victim_process
            victim_proc = None
            lease_expired, _ = wait_v06_task_status(coordinator_url, str(claim["task_id"]), "queued", timeout=30.0)
            steps.append({"name": "victim_stage1_requeued", "ok": lease_expired})
            post_announce(
                p2p_url,
                build_p2p_peer(
                    swarm_id=args.swarm_id,
                    peer_id=rescue_id,
                    role="miner",
                    backend=args.backend,
                    stage_role="stage1",
                    hf_model_id=args.hf_model_id,
                    ttl_seconds=60.0,
                ),
                timeout=args.http_timeout,
            )
            time.sleep(1.2)
            route_after = resolve_route(
                p2p_url,
                prompt_text=args.prompt_text,
                backend=args.backend,
                hf_model_id=args.hf_model_id,
                max_new_tokens=1,
                http_timeout=args.http_timeout,
            )
            payloads["route_after"] = {"route": route_after, "ok": bool(route_after.get("usable_now"))}
            steps.append({"name": "route_after_stage1_rescue", "ok": bool(route_after.get("usable_now"))})
            rescue_step, rescue_payload = run_p2p_join_miner_step(
                p2p_url=p2p_url,
                swarm_id=args.swarm_id,
                miner_id=rescue_id,
                stage="stage1",
                backend=args.backend,
                hf_model_id=args.hf_model_id,
                hf_cache_dir=hf_cache_dir,
                timeout=args.timeout_seconds,
                http_timeout=args.http_timeout,
            )
            steps.append(rescue_step)
            payloads["rescue_stage1"] = rescue_payload

        generate_step, generate_payload = product_mvp.finish_process_step("generate_p2p_rescue", generate_proc, timeout=args.timeout_seconds + 10.0)
        steps.append(generate_step)
        payloads["generate"] = generate_payload
        generate_proc = None
        return finish_real_rescue_report(args, output_dir=output_dir, base_url=coordinator_url, failure_stage=failure_stage, steps=steps, payloads=payloads, p2pd_proc=p2pd_proc, serve_proc=serve_proc, generate_proc=generate_proc, victim_proc=victim_proc)
    finally:
        if victim_proc is not None and victim_proc.poll() is None:
            terminate_process_group(victim_proc, sig=signal.SIGTERM)
            process_result(victim_id, victim_proc, expected_failure=True)
        if generate_proc is not None and generate_proc.poll() is None:
            stop_process(generate_proc)
        if serve_proc is not None:
            product_mvp.stop_process(serve_proc)
        if p2pd_proc is not None:
            stop_process(p2pd_proc)


def start_real_miner_process(
    *,
    coordinator_url: str,
    miner_id: str,
    stage: str,
    hf_model_id: str,
    hf_cache_dir: str,
    compute_seconds: float,
) -> subprocess.Popen[str]:
    command = real_miner_command(
        coordinator_url=coordinator_url,
        miner_id=miner_id,
        stage=stage,
        hf_model_id=hf_model_id,
        hf_cache_dir=hf_cache_dir,
        compute_seconds=compute_seconds,
    )
    return popen_process(command)


def real_miner_command(
    *,
    coordinator_url: str,
    miner_id: str,
    stage: str,
    hf_model_id: str,
    hf_cache_dir: str,
    compute_seconds: float = 0.0,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "miner_cli.py"),
        "--coordinator",
        coordinator_url,
        "--miner-id",
        miner_id,
        "--enable-hf-tiny-gpt-runtime",
        "--real-llm-backend",
        "hf_transformers_cpu",
        "--real-llm-stage-role",
        stage,
        "--real-llm-partition-mode",
        "stage-local",
        "--hf-model-id",
        hf_model_id,
        "--miner-token",
        MINER_TOKEN,
        "--once",
        "--max-tasks",
        "1",
        "--compute-seconds",
        str(compute_seconds),
        "--heartbeat-interval",
        "1.0",
        "--idle-sleep",
        "0.2",
    ]
    if hf_cache_dir:
        command.extend(["--hf-cache-dir", hf_cache_dir])
    return command


def run_real_miner_step(
    *,
    coordinator_url: str,
    miner_id: str,
    stage: str,
    hf_model_id: str,
    hf_cache_dir: str,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            real_miner_command(
                coordinator_url=coordinator_url,
                miner_id=miner_id,
                stage=stage,
                hf_model_id=hf_model_id,
                hf_cache_dir=hf_cache_dir,
            ),
            cwd=str(ROOT),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "name": f"run_{miner_id}",
            "ok": False,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": "timeout",
        }, {}
    payload: dict[str, Any] = {}
    for line in reversed((completed.stdout or "").strip().splitlines()):
        if not line.strip().startswith("{"):
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "accepted_tasks" in candidate:
            payload = candidate
            break
    ok = bool(completed.returncode == 0 and int(payload.get("accepted_tasks") or 0) >= 1)
    step = {
        "name": f"run_{miner_id}",
        "ok": ok,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "payload_ok": ok,
    }
    if not ok:
        step["stdout_tail"] = redact_text((completed.stdout or "")[-1000:])
        step["stderr_tail"] = redact_text((completed.stderr or "")[-1000:])
    return step, {"ok": ok, "miner_id": miner_id, "stage": stage, "accepted_tasks": int(payload.get("accepted_tasks") or 0)}


def run_p2p_join_miner_step(
    *,
    p2p_url: str,
    swarm_id: str,
    miner_id: str,
    stage: str,
    backend: str,
    hf_model_id: str,
    hf_cache_dir: str,
    timeout: float,
    http_timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "join",
        "--p2p",
        "--peer-bootstrap",
        p2p_url,
        "--swarm-id",
        swarm_id,
        "--miner-id",
        miner_id,
        "--stage",
        stage,
        "--backend",
        backend,
        "--miner-token",
        MINER_TOKEN,
        "--hf-model-id",
        hf_model_id,
        "--http-timeout",
        str(http_timeout),
        "--once",
        "--max-tasks",
        "1",
        "--max-runtime-seconds",
        str(max(1.0, float(timeout) - min(5.0, float(timeout) / 2.0))),
        "--idle-sleep",
        "0.2",
        "--run",
        "--json",
    ]
    if hf_cache_dir:
        command.extend(["--hf-cache-dir", hf_cache_dir])
    step, payload = product_mvp.run_step(f"join_p2p_rescue_{miner_id}", command, timeout=timeout)
    payload = payload if isinstance(payload, dict) else {}
    return step, {
        "ok": bool(step.get("ok") and payload.get("ok") is not False),
        "miner_id": miner_id,
        "stage": stage,
        "mode": "join_p2p_run",
        "peer_bootstrap_used": bool(payload.get("peer_bootstrap_used")),
        "p2p": payload.get("p2p") if isinstance(payload.get("p2p"), dict) else {},
        "diagnosis_codes": payload.get("diagnosis_codes") if isinstance(payload.get("diagnosis_codes"), list) else [],
        "returncode": step.get("returncode"),
    }


def finish_real_rescue_report(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    base_url: str,
    failure_stage: str,
    steps: list[dict[str, Any]],
    payloads: dict[str, Any],
    p2pd_proc: subprocess.Popen[str] | None,
    serve_proc: subprocess.Popen[str] | None,
    generate_proc: subprocess.Popen[str] | None,
    victim_proc: subprocess.Popen[str] | None,
) -> dict[str, Any]:
    if generate_proc is not None and generate_proc.poll() is None:
        stop_process(generate_proc)
    if victim_proc is not None and victim_proc.poll() is None:
        terminate_process_group(victim_proc, sig=signal.SIGTERM)
        process_result(str((payloads.get("victim_claim") or {}).get("miner_id") or "victim"), victim_proc, expected_failure=True)
    rows: list[dict[str, Any]] = []
    ledger_error = ""
    session_id = str(((payloads.get("generate") or {}).get("session") or {}).get("session_id") or "")
    try:
        rows = v06_admin_results(base_url, session_id=session_id, limit=100)
    except Exception as exc:
        ledger_error = f"{type(exc).__name__}: {exc}"
    from crowdtensor.session_protocol import safe_generation_summary

    generation = safe_generation_summary((payloads.get("generate") or {}), max_new_tokens=1)
    route_after = ((payloads.get("route_after") or {}).get("route") or {}) if isinstance(payloads.get("route_after"), dict) else {}
    matched = route_after.get("matched_capabilities") if isinstance(route_after.get("matched_capabilities"), dict) else {}
    expected_capability = "real_llm_sharded_stage0" if failure_stage == "stage0" else "real_llm_sharded_stage1"
    rescued_peer = f"p2p-v06-rescue-{failure_stage}-rescue"
    victim_process = payloads.get("victim_process") if isinstance(payloads.get("victim_process"), dict) else {}
    rescue_payload = payloads.get(f"rescue_{failure_stage}") if isinstance(payloads.get(f"rescue_{failure_stage}"), dict) else {}
    lease_expired = any(step.get("name") == f"victim_{failure_stage}_requeued" and step.get("ok") for step in steps)
    failed_stage_id = 0 if failure_stage == "stage0" else 1
    rescued_result = bool(
        rescue_payload.get("ok")
        and any(
            row.get("status") == "completed"
            and int(row.get("stage_id", -1)) == failed_stage_id
            and row.get("miner_id") == rescued_peer
            for row in rows
        )
    )
    generation_ready = bool((payloads.get("generate") or {}).get("ok") and generation.get("generated_token_count"))
    route_ready = bool(route_after.get("usable_now") and matched.get(expected_capability) == rescued_peer)
    victim_failed = bool(victim_process.get("expected_failure") and not victim_process.get("ok"))
    ok = bool(all(step.get("ok") for step in steps) and route_ready and lease_expired and rescued_result and generation_ready and not ledger_error and victim_failed)
    codes = set(diagnosis_codes(*payloads.values()))
    if ok:
        codes.update({
            "p2p_real_stage_rescue_ready",
            "p2p_rescue_generation_completed",
            f"{failure_stage}_rescue_generation_completed",
            f"{failure_stage}_victim_requeued",
            "p2p_generate_route_ready",
        })
    else:
        codes.add("p2p_real_stage_rescue_blocked")
    report = {
        "schema": "p2p_real_stage_rescue_case_v1",
        "ok": ok,
        "failure_stage": failure_stage,
        "session": {"session_id": session_id, "workload_type": WORKLOAD_TYPE},
        "route_after": route_after,
        "generation": generation,
        "requeue_summary": {
            "victim_peer_id": f"p2p-v06-rescue-{failure_stage}-victim",
            "rescue_peer_id": rescued_peer,
            "lease_expired": lease_expired,
            "victim_task_requeued": lease_expired,
            "rescued_result": rescued_result,
            "victim_result_accepted": False,
            "victim_process_failed": victim_failed,
        },
        "ledger": {"accepted_rows": len(rows), "error": ledger_error},
        "steps": steps,
        "processes": {
            "serve": product_mvp.stop_process(serve_proc) if serve_proc is not None else {},
            "p2pd": stop_process(p2pd_proc) if p2pd_proc is not None else {},
        },
        "diagnosis_codes": sorted(codes),
    }
    return support_bundle.sanitize(redact_values(report))


def finalize_real_generate_report(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    base_url: str,
    steps: list[dict[str, Any]],
    payloads: dict[str, Any],
    p2pd_process: dict[str, Any],
    serve_process: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    ledger_error = ""
    session_id = str(((payloads.get("generate") or {}).get("session") or {}).get("session_id") or "")
    try:
        query = f"/admin/results?status=accepted&workload_type={WORKLOAD_TYPE}&limit=100"
        if session_id:
            query += f"&session_id={session_id}"
        ledger = product_mvp.request_json("GET", base_url, query, admin_token=ADMIN_TOKEN, timeout=10.0)
        rows = ledger.get("results") if isinstance(ledger.get("results"), list) else []
    except Exception as exc:
        ledger_error = f"{type(exc).__name__}: {exc}"
    generation = (payloads.get("generate") or {}).get("generation") if isinstance((payloads.get("generate") or {}).get("generation"), dict) else {}
    batch = product_mvp.safe_batch_summary(args, generation)
    stream = product_mvp.safe_stream_summary(args, payloads.get("generate") or {})
    route = (payloads.get("generate") or {}).get("route") if isinstance((payloads.get("generate") or {}).get("route"), dict) else {}
    stages = product_mvp.stage_summary(rows)
    kv_cache = real_generate_kv_cache_summary(output_dir, required_tokens=args.max_new_tokens)
    generated_tokens = int(generation.get("generated_token_count") or 0)
    step_ok = all(bool(step.get("ok")) for step in steps)
    route_ready = bool(route.get("usable_now") and route.get("route_source") == "p2p-discovery")
    generation_ready = bool(
        (payloads.get("generate") or {}).get("ok") is True
        and generated_tokens >= args.max_new_tokens
        and batch.get("batch_generation_ready") is True
    )
    stream_ready = bool(not args.stream_generation or stream.get("stream_generation_ready") is True)
    stage_ready = bool(stages.get("distinct_stage_miners") and int(stages.get("completed_rows") or 0) >= args.max_new_tokens * 2)
    ok = bool(step_ok and route_ready and generation_ready and stream_ready and stage_ready and not ledger_error)
    codes = set(diagnosis_codes(*payloads.values()))
    if ok:
        codes.update({
            "p2p_real_generate_ready",
            "p2p_generate_route_ready",
            "tiny_gpt2_multi_token_ready",
            "distinct_stage_miners",
            "stage_assignment_valid",
        })
        if batch.get("enabled") and batch.get("batch_generation_ready"):
            codes.update({
                "p2p_real_generate_batch_ready",
                "public_swarm_generate_batch_ready",
            })
        if stream_ready and args.stream_generation:
            codes.update({
                "p2p_real_generate_stream_ready",
                "public_swarm_generate_stream_ready",
            })
            if stream.get("endpoint_ready"):
                codes.add("public_swarm_generate_stream_endpoint_ready")
        if kv_cache.get("ready"):
            codes.update({
                "p2p_real_generate_kv_cache_ready",
                "real_llm_stage0_kv_cache_v1_ready",
                "real_llm_stage1_kv_cache_v1_ready",
                "stage0_kv_cache_hits_ready",
                "stage1_kv_cache_hits_ready",
            })
    else:
        codes.add("p2p_real_generate_blocked")
        if not route_ready:
            codes.add("p2p_real_generate_route_blocked")
        if not generation_ready:
            codes.add("p2p_real_generate_token_count_blocked")
        if args.stream_generation and not stream_ready:
            codes.add("p2p_real_generate_stream_blocked")
        if not stage_ready:
            codes.add("p2p_real_generate_stage_assignment_blocked")
        if ledger_error:
            codes.add("p2p_real_generate_ledger_blocked")
    return support_bundle.sanitize(redact_values({
        "schema": "p2p_real_generate_probe_v1",
        "ok": ok,
        "output_dir": str(output_dir),
        "session": {
            "session_id": session_id,
            "workload_type": WORKLOAD_TYPE,
        },
        "route": route,
        "generation": generation,
        "batch": batch,
        "stream": stream,
        "stage_assignment": stages,
        "kv_cache": kv_cache,
        "ledger": {
            "accepted_rows": len(rows),
            "error": ledger_error,
        },
        "steps": steps,
        "processes": {
            "p2pd": p2pd_process,
            "serve": serve_process,
        },
        "diagnosis_codes": sorted(codes),
        "safety": {
            "p2p_discovery_routing": True,
            "coordinator_result_fallback": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "not_production": True,
        },
    }))


def finish_real_generate_probe(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    base_url: str,
    steps: list[dict[str, Any]],
    payloads: dict[str, Any],
    p2pd_proc: subprocess.Popen[str] | None,
    serve_proc: subprocess.Popen[str] | None,
    generate_proc: subprocess.Popen[str] | None,
    stage_miner_procs: list[tuple[str, subprocess.Popen[str]]] | None = None,
) -> dict[str, Any]:
    if generate_proc is not None and generate_proc.poll() is None:
        stop_process(generate_proc)
    stage_processes: dict[str, Any] = {}
    for name, proc in stage_miner_procs or []:
        stage_processes[name] = product_mvp.stop_process(proc)
    report = finalize_real_generate_report(
        args,
        output_dir=output_dir,
        base_url=base_url,
        steps=steps,
        payloads=payloads,
        p2pd_process={},
        serve_process={},
    )
    report.setdefault("processes", {})
    if stage_processes:
        report["processes"]["stage_miners"] = stage_processes
    report["processes"]["serve"] = product_mvp.stop_process(serve_proc) if serve_proc is not None else {}
    report["processes"]["p2pd"] = stop_process(p2pd_proc) if p2pd_proc is not None else {}
    return support_bundle.sanitize(redact_values(report))


def run_real_generate_probe(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    if product_mvp.missing_hf_dependencies():
        return support_bundle.sanitize({
            "schema": "p2p_real_generate_probe_v1",
            "ok": False,
            "degraded": True,
            "diagnosis_codes": ["hf_dependencies_missing", "p2p_real_generate_hf_runtime_missing"],
            "operator_action": "Install optional runtime dependencies with: python -m pip install -e '.[hf]'",
        })
    hf_cache_dir = str(getattr(args, "hf_cache_dir", "") or "")
    p2p_port = find_free_port()
    coordinator_port = find_free_port()
    p2p_url = f"http://127.0.0.1:{p2p_port}"
    coordinator_url = f"http://127.0.0.1:{coordinator_port}"
    state_dir = output_dir / "real-generate-state"
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    p2pd_proc: subprocess.Popen[str] | None = None
    serve_proc: subprocess.Popen[str] | None = None
    generate_proc: subprocess.Popen[str] | None = None
    stage_miner_procs: list[tuple[str, subprocess.Popen[str]]] = []
    steps: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {}
    try:
        p2pd_proc = popen_process([
            sys.executable,
            str(ROOT / "scripts" / "p2p_lite_daemon.py"),
            "--host",
            "127.0.0.1",
            "--port",
            str(p2p_port),
            "--swarm-id",
            args.swarm_id,
            "--role",
            "observer",
        ])
        healthy, health_error = wait_p2pd(p2p_url, p2pd_proc, timeout=args.startup_timeout)
        steps.append({"name": "p2pd", "ok": healthy, "error": health_error})
        if not healthy:
            return finish_real_generate_probe(args, output_dir=output_dir, base_url=coordinator_url, steps=steps, payloads=payloads, p2pd_proc=p2pd_proc, serve_proc=serve_proc, generate_proc=generate_proc, stage_miner_procs=stage_miner_procs)

        serve_cmd = [
            sys.executable,
            "-m",
            "crowdtensor.cli",
            "serve",
            "--p2p",
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
            str(coordinator_port),
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
            "--run",
            "--json",
        ]
        if hf_cache_dir:
            serve_cmd.extend(["--hf-cache-dir", hf_cache_dir])
        serve_proc = product_mvp.popen_command(serve_cmd)
        serve_healthy, serve_error = product_mvp.wait_health(coordinator_url, serve_proc, args.startup_timeout)
        steps.append({"name": "serve_p2p_run", "ok": serve_healthy, "error": serve_error})
        if not serve_healthy:
            return finish_real_generate_probe(args, output_dir=output_dir, base_url=coordinator_url, steps=steps, payloads=payloads, p2pd_proc=p2pd_proc, serve_proc=serve_proc, generate_proc=generate_proc, stage_miner_procs=stage_miner_procs)

        for stage in ("stage0", "stage1"):
            command = [
                sys.executable,
                "-m",
                "crowdtensor.cli",
                "join",
                "--p2p",
                "--peer-bootstrap",
                p2p_url,
                "--swarm-id",
                args.swarm_id,
                "--miner-id",
                f"p2p-v06-real-{stage}",
                "--stage",
                stage,
                "--backend",
                args.backend,
                "--miner-token",
                MINER_TOKEN,
                "--hf-model-id",
                args.hf_model_id,
                "--max-tasks",
                str(args.max_new_tokens),
                "--ttl-seconds",
                str(max(60.0, float(args.timeout_seconds) + 60.0)),
                "--idle-sleep",
                "0.25",
                "--max-request-attempts",
                "5",
                "--retry-max-sleep",
                "1.0",
                "--run",
                "--json",
            ]
            if hf_cache_dir:
                command.extend(["--hf-cache-dir", hf_cache_dir])
            proc = product_mvp.popen_command(command)
            stage_miner_procs.append((stage, proc))
            steps.append({"name": f"persistent_join_p2p_{stage}_started", "ok": proc.poll() is None})

        miners_ready, catalog, miner_error = wait_p2p_stage_miners(
            p2p_url,
            timeout=args.startup_timeout,
            http_timeout=args.http_timeout,
        )
        steps.append({
            "name": "persistent_stage_miners_discovered",
            "ok": miners_ready,
            "error": miner_error,
            "stage_counts": dict(zip(["coordinator", "stage0", "stage1"], stage_peer_count(catalog))),
        })
        payloads["persistent_stage_catalog"] = catalog
        if not miners_ready:
            return finish_real_generate_probe(args, output_dir=output_dir, base_url=coordinator_url, steps=steps, payloads=payloads, p2pd_proc=p2pd_proc, serve_proc=serve_proc, generate_proc=generate_proc, stage_miner_procs=stage_miner_procs)

        generate_cmd = p2p_generate_command(
            p2p_url=p2p_url,
            admin_token=ADMIN_TOKEN,
            backend=args.backend,
            hf_model_id=args.hf_model_id,
            max_new_tokens=args.max_new_tokens,
            timeout_seconds=args.timeout_seconds,
            http_timeout=max(float(args.http_timeout), min(float(args.timeout_seconds), 120.0)),
            prompt_text=args.prompt_text,
            prompt_texts=getattr(args, "prompt_texts", ""),
            prompt_texts_file=getattr(args, "prompt_texts_file", ""),
            stream=args.stream_generation,
        )
        generate_proc = product_mvp.popen_command(generate_cmd)
        session_started, session_error = wait_v06_workload_started(coordinator_url, timeout=args.timeout_seconds)
        steps.append({"name": "generate_p2p_session_started", "ok": session_started, "error": session_error})
        if not session_started:
            generate_step, generate_payload = product_mvp.finish_process_step("generate_p2p", generate_proc, timeout=1.0)
            steps.append(generate_step)
            payloads["generate"] = generate_payload
            return finish_real_generate_probe(args, output_dir=output_dir, base_url=coordinator_url, steps=steps, payloads=payloads, p2pd_proc=p2pd_proc, serve_proc=serve_proc, generate_proc=generate_proc, stage_miner_procs=stage_miner_procs)

        generate_step, generate_payload = product_mvp.finish_process_step("generate_p2p", generate_proc, timeout=args.timeout_seconds + 60.0)
        steps.append(generate_step)
        payloads["generate"] = generate_payload
        generate_proc = None
        return finish_real_generate_probe(args, output_dir=output_dir, base_url=coordinator_url, steps=steps, payloads=payloads, p2pd_proc=p2pd_proc, serve_proc=serve_proc, generate_proc=generate_proc, stage_miner_procs=stage_miner_procs)
    finally:
        if generate_proc is not None and generate_proc.poll() is None:
            stop_process(generate_proc)
        for _stage, proc in stage_miner_procs:
            if proc.poll() is None:
                product_mvp.stop_process(proc)
        if serve_proc is not None:
            product_mvp.stop_process(serve_proc)
        if p2pd_proc is not None:
            stop_process(p2pd_proc)


def run_local_p2p_discovery(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    base_url = f"http://127.0.0.1:{args.p2p_port}"
    peer_secret = str(getattr(args, "peer_secret", "") or "")
    require_signed = bool(getattr(args, "require_signed", False))
    p2pd_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "p2p_lite_daemon.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.p2p_port),
        "--swarm-id",
        args.swarm_id,
        "--role",
        "observer",
    ]
    if peer_secret:
        p2pd_cmd.extend(["--peer-secret", peer_secret])
    if require_signed:
        p2pd_cmd.append("--require-signed")
    steps: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {}
    proc: subprocess.Popen[str] | None = None
    try:
        proc = popen_process(p2pd_cmd)
        healthy, health_error = wait_p2pd(base_url, proc, timeout=args.startup_timeout)
        steps.append({"name": "p2pd", "ok": healthy, "error": health_error, "command": "crowdtensor p2pd --run"})
        if not healthy:
            return {"ok": False, "diagnosis_codes": ["p2pd_start_failed"], "process": stop_process(proc)}, steps, payloads

        p2pd_step, p2pd_payload = run_json_step(
            "p2pd_command_shape",
            [
                sys.executable,
                "-m",
                "crowdtensor.cli",
                "p2pd",
                "--port",
                str(args.p2p_port),
                *(["--peer-secret", peer_secret] if peer_secret else []),
                *(["--require-signed"] if require_signed else []),
                "--json",
            ],
            runner=runner,
            timeout_seconds=args.timeout_seconds,
        )
        steps.append(p2pd_step)
        payloads["p2pd_command"] = p2pd_payload

        serve_step, serve_payload = run_json_step(
            "serve_p2p_announce",
            [
                sys.executable,
                "-m",
                "crowdtensor.cli",
                "serve",
                "--p2p",
                "--peer-bootstrap",
                base_url,
                "--swarm-id",
                args.swarm_id,
                "--public-host",
                "127.0.0.1",
                "--port",
                str(args.coordinator_port),
                "--hf-model-id",
                args.hf_model_id,
                *(["--peer-secret", peer_secret] if peer_secret else []),
                "--json",
            ],
            runner=runner,
            timeout_seconds=args.timeout_seconds,
        )
        steps.append(serve_step)
        payloads["serve"] = serve_payload

        for stage in ("stage0", "stage1"):
            step, payload = run_json_step(
                f"join_p2p_announce_{stage}",
                [
                    sys.executable,
                    "-m",
                    "crowdtensor.cli",
                    "join",
                    "--p2p",
                    "--peer-bootstrap",
                    base_url,
                    "--swarm-id",
                    args.swarm_id,
                    "--miner-id",
                    f"p2p-v06-{stage}",
                    "--stage",
                    stage,
                    "--backend",
                    args.backend,
                    "--hf-model-id",
                    args.hf_model_id,
                    *(["--peer-secret", peer_secret] if peer_secret else []),
                    "--json",
                ],
                runner=runner,
                timeout_seconds=args.timeout_seconds,
            )
            steps.append(step)
            payloads[f"join_{stage}"] = payload

        generate_step, generate_payload = run_json_step(
            "generate_p2p_route",
            p2p_generate_command(
                p2p_url=base_url,
                backend=args.backend,
                hf_model_id=args.hf_model_id,
                max_new_tokens=args.max_new_tokens,
                prompt_text=args.prompt_text,
                prompt_texts=getattr(args, "prompt_texts", ""),
                prompt_texts_file=getattr(args, "prompt_texts_file", ""),
                dry_run=True,
            ),
            runner=runner,
            timeout_seconds=args.timeout_seconds,
        )
        steps.append(generate_step)
        payloads["generate"] = generate_payload

        catalog = request_json(base_url, "/peer/catalog", timeout=args.http_timeout)
        write_json(output_dir / "p2p_catalog.json", catalog)
        rescue_probe = run_stage_rescue_probe(args)
        real_generate_probe = run_real_generate_probe(args, output_dir=output_dir / "p2p-real-generate")
        real_stage_rescue_probe = run_real_stage_rescue_probe(args, output_dir=output_dir / "p2p-real-stage-rescue")
        payloads["stage_rescue_probe"] = rescue_probe
        payloads["real_generate_probe"] = real_generate_probe
        payloads["real_stage_rescue_probe"] = real_stage_rescue_probe
        steps.append({
            "name": "stage_rescue_probe",
            "ok": bool(rescue_probe.get("ok")),
            "payload_schema": rescue_probe.get("schema"),
            "diagnosis_codes": rescue_probe.get("diagnosis_codes") or [],
        })
        steps.append({
            "name": "real_generate_probe",
            "ok": bool(real_generate_probe.get("ok") or real_generate_probe.get("degraded")),
            "payload_schema": real_generate_probe.get("schema"),
            "diagnosis_codes": real_generate_probe.get("diagnosis_codes") or [],
        })
        steps.append({
            "name": "real_stage_rescue_probe",
            "ok": bool(real_stage_rescue_probe.get("ok") or real_stage_rescue_probe.get("degraded")),
            "payload_schema": real_stage_rescue_probe.get("schema"),
            "diagnosis_codes": real_stage_rescue_probe.get("diagnosis_codes") or [],
        })
        coordinator_count, stage0_count, stage1_count = stage_peer_count(catalog)
        route = generate_payload.get("route") if isinstance(generate_payload.get("route"), dict) else {}
        ready = bool(
            healthy
            and p2pd_payload.get("ok")
            and serve_payload.get("ok")
            and payloads.get("join_stage0", {}).get("ok")
            and payloads.get("join_stage1", {}).get("ok")
            and generate_payload.get("ok")
            and route.get("usable_now")
            and coordinator_count >= 1
            and stage0_count >= 1
            and stage1_count >= 1
            and rescue_probe.get("ok")
            and real_generate_probe.get("ok")
            and real_stage_rescue_probe.get("ok")
        )
        codes = ["p2pd_daemon_ready"] if healthy else ["p2pd_daemon_blocked"]
        codes.extend(code for code in rescue_probe.get("diagnosis_codes") or [] if isinstance(code, str))
        codes.extend(code for code in real_generate_probe.get("diagnosis_codes") or [] if isinstance(code, str))
        codes.extend(code for code in real_stage_rescue_probe.get("diagnosis_codes") or [] if isinstance(code, str))
        if ready:
            codes.extend([
                "local_three_process_p2p_discovery_ready",
                "p2p_stage_discovery_ready",
                "p2p_generate_route_ready",
                "p2p_stage_rescue_ready",
                "coordinator_to_p2p_transition_ready",
                "coordinator_result_fallback_ready",
            ])
        return {
            "ok": ready,
            "schema": "p2p_swarm_local_discovery_v1",
            "hf_model_id": args.hf_model_id,
            "p2p_url": base_url,
            "catalog_peer_count": len(catalog.get("peers") or []),
            "registry": catalog.get("registry") if isinstance(catalog.get("registry"), dict) else {},
            "signed_peer_count": (catalog.get("registry") or {}).get("signed_peer_count") if isinstance(catalog.get("registry"), dict) else 0,
            "healthy_peer_count": (catalog.get("registry") or {}).get("healthy_peer_count") if isinstance(catalog.get("registry"), dict) else 0,
            "coordinator_peer_count": coordinator_count,
            "stage0_peer_count": stage0_count,
            "stage1_peer_count": stage1_count,
            "generate_route": route,
            "rescue_probe": rescue_probe,
            "real_generate_probe": real_generate_probe,
            "real_stage_rescue_probe": real_stage_rescue_probe,
            "diagnosis_codes": codes if ready else sorted(set(codes + ["p2p_stage_discovery_blocked"])),
        }, steps, payloads
    finally:
        process = stop_process(proc)
        if process:
            payloads["p2pd_process"] = process


def summarize_preview_v04(payload: dict[str, Any]) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    preview = payload.get("preview") if isinstance(payload.get("preview"), dict) else {}
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "ready": bool(payload.get("ok") and "public_swarm_preview_v04_ready" in codes),
        "external_two_stage_generation_ready": "external_two_stage_generation_ready" in codes or bool(preview.get("external_two_stage_generation_ready")),
        "external_stage_requeue_ready": "external_stage_requeue_ready" in codes or bool(preview.get("external_stage_requeue_ready")),
        "multi_token_generation_ready": "multi_token_generation_ready" in codes or bool(preview.get("multi_token_generation_ready")),
        "tiny_gpt2_ci_fallback_ready": "tiny_gpt2_ci_fallback_ready" in codes or bool(preview.get("tiny_gpt2_ci_fallback_ready")),
        "optional_model_ready": "optional_distilgpt2_or_gpt2_strict_ready" in codes or bool(preview.get("optional_model_ready")),
        "diagnosis_codes": sorted(codes),
    }


def summarize_product(payload: dict[str, Any], *, required_tokens: int) -> dict[str, Any]:
    summary = preview_v04.summarize_product_mvp(payload, required_tokens=required_tokens) if payload else {}
    codes = set(summary.get("diagnosis_codes") or [])
    return {
        **summary,
        "tiny_gpt2_multi_token_ready": bool(summary.get("ready") and summary.get("generated_token_count", 0) >= required_tokens),
        "degraded_ready": "product_swarm_mvp_degraded_ready" in codes,
    }


def _cache_payload_from_event(event: dict[str, Any], *, stage_id: int) -> dict[str, Any]:
    if int(stage_id) == 0:
        payload = (event.get("activation_results") or [{}])[0]
    else:
        payload = (event.get("inference_results") or [{}])[0]
        if not isinstance(payload, dict) or not payload:
            result = event.get("sharded_inference_result")
            if isinstance(result, dict):
                payload = result.get("inference_result") or {}
    return payload if isinstance(payload, dict) else {}


def _task_log_path(state_dir: Path) -> Path:
    direct = state_dir / "tasks.jsonl"
    if direct.is_file():
        return direct
    nested = state_dir / "real-generate-state" / "tasks.jsonl"
    if nested.is_file():
        return nested
    return direct


def _stage_cache_rows(state_dir: Path, *, stage_id: int) -> list[dict[str, Any]]:
    path = _task_log_path(state_dir)
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "task_completed":
            continue
        validation = event.get("validation") if isinstance(event.get("validation"), dict) else {}
        if int(validation.get("stage_id", -1)) != int(stage_id):
            continue
        payload = _cache_payload_from_event(event, stage_id=stage_id)
        rows.append({
            "generation_step": int(validation.get("generation_step", payload.get("generation_step", 0)) or 0),
            "stage_id": int(stage_id),
            "miner_id": str(event.get("miner_id") or ""),
            "cache_schema": payload.get("kv_cache_schema"),
            "cache_stage": payload.get("kv_cache_stage"),
            "cache_ready": bool(payload.get("kv_cache_ready")),
            "cache_hit": bool(payload.get("kv_cache_hit")),
            "token_count_before": int(payload.get("kv_cache_tokens_before") or 0),
            "token_count_after": int(payload.get("kv_cache_tokens_after") or 0),
            "prefix_token_count": int(payload.get("generated_prefix_token_count") or 0),
            "generated_count": int(payload.get("generated_token_count") or 0),
        })
    return sorted(rows, key=lambda item: int(item.get("generation_step") or 0))


def _stage_cache_summary(*, schema: str, stage: str, expected_hits: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ready_rows = [row for row in rows if row.get("cache_ready")]
    hit_rows = [row for row in rows if row.get("cache_hit")]
    return {
        "schema": schema,
        "stage": stage,
        "row_count": len(rows),
        "ready_count": len(ready_rows),
        "hit_count": len(hit_rows),
        "expected_hit_count": expected_hits,
        "ready": bool(len(rows) >= expected_hits + 1 and len(ready_rows) >= expected_hits + 1 and len(hit_rows) >= expected_hits),
        "rows": rows,
    }


def real_generate_kv_cache_summary(state_dir: Path, *, required_tokens: int) -> dict[str, Any]:
    expected_hits = max(0, int(required_tokens) - 1)
    stage0 = _stage_cache_summary(
        schema="real_llm_stage0_kv_cache_v1",
        stage="stage0_prefix",
        expected_hits=expected_hits,
        rows=_stage_cache_rows(state_dir, stage_id=0),
    )
    stage1 = _stage_cache_summary(
        schema="real_llm_stage1_kv_cache_v1",
        stage="stage1_suffix",
        expected_hits=expected_hits,
        rows=_stage_cache_rows(state_dir, stage_id=1),
    )
    return {
        "schema": "p2p_real_generate_dual_stage_kv_cache_v1",
        "ready": bool(stage0.get("ready") and stage1.get("ready")),
        "expected_hit_count_per_stage": expected_hits,
        "stage0": stage0,
        "stage1": stage1,
        "raw_activations_public": False,
        "raw_token_inputs_public": False,
        "raw_generated_values_public": False,
        "process_scope": "single_miner_process_per_stage",
    }


def normalize_local_p2p_report(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or not payload:
        return {}
    if payload.get("schema") == "p2p_swarm_local_discovery_v1":
        return payload
    if payload.get("schema") != SCHEMA:
        return payload
    p2p = payload.get("p2p") if isinstance(payload.get("p2p"), dict) else {}
    if not p2p:
        return {}
    codes = [
        code for code in payload.get("diagnosis_codes") or []
        if isinstance(code, str) and code in LOCAL_P2P_DISCOVERY_CODES
    ]
    summary = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
    local_summary = summary.get("local_p2p_discovery") if isinstance(summary.get("local_p2p_discovery"), dict) else {}
    rescue_probe = local_summary.get("rescue_probe") if isinstance(local_summary.get("rescue_probe"), dict) else {}
    real_generate_probe = local_summary.get("real_generate_probe") if isinstance(local_summary.get("real_generate_probe"), dict) else {}
    real_stage_rescue_probe = local_summary.get("real_stage_rescue_probe") if isinstance(local_summary.get("real_stage_rescue_probe"), dict) else {}
    if not codes and p2p.get("ready"):
        codes = [
            "p2pd_daemon_ready",
            "local_three_process_p2p_discovery_ready",
            "p2p_stage_discovery_ready",
            "p2p_generate_route_ready",
            "coordinator_to_p2p_transition_ready",
            "coordinator_result_fallback_ready",
        ]
        if rescue_probe.get("ok"):
            codes.append("p2p_stage_rescue_ready")
        if real_generate_probe.get("ok"):
            codes.append("p2p_real_generate_ready")
        elif real_generate_probe.get("degraded"):
            codes.extend(["hf_dependencies_missing", "p2p_real_generate_hf_runtime_missing"])
        if real_stage_rescue_probe.get("ok"):
            codes.extend([
                "p2p_real_stage_rescue_ready",
                "p2p_rescue_generation_completed",
                "stage0_rescue_generation_completed",
                "stage1_rescue_generation_completed",
            ])
        elif real_stage_rescue_probe.get("degraded"):
            codes.extend(["hf_dependencies_missing", "p2p_real_stage_rescue_hf_runtime_missing"])
    local_ready = bool(
        payload.get("ok")
        and p2p.get("ready")
        and (real_generate_probe.get("ok") if real_generate_probe else True)
        and (real_stage_rescue_probe.get("ok") if real_stage_rescue_probe else False)
    )
    return {
        "schema": "p2p_swarm_local_discovery_v1",
        "ok": local_ready,
        "hf_model_id": p2p.get("hf_model_id") or payload.get("hf_model_id") or first_string_value(local_summary, "hf_model_id"),
        "p2p_url": p2p.get("p2p_url"),
        "catalog_peer_count": p2p.get("catalog_peer_count"),
        "coordinator_peer_count": p2p.get("coordinator_peer_count"),
        "stage0_peer_count": p2p.get("stage0_peer_count"),
        "stage1_peer_count": p2p.get("stage1_peer_count"),
        "generate_route": p2p.get("generate_route") if isinstance(p2p.get("generate_route"), dict) else {},
        "rescue_probe": rescue_probe,
        "real_generate_probe": real_generate_probe,
        "real_stage_rescue_probe": real_stage_rescue_probe,
        "kv_cache": real_generate_probe.get("kv_cache") if isinstance(real_generate_probe.get("kv_cache"), dict) else {},
        "stream": real_generate_probe.get("stream") if isinstance(real_generate_probe.get("stream"), dict) else {"enabled": False, "stream_generation_ready": False},
        "diagnosis_codes": sorted(set(codes)),
    }


def run_external_existing_probe(args: argparse.Namespace, *, output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {}
    if not args.peer_bootstrap:
        return {
            "schema": "p2p_external_existing_probe_v1",
            "ok": False,
            "diagnosis_codes": ["external_p2p_bootstrap_missing"],
        }, steps, payloads
    try:
        catalog = request_json(args.peer_bootstrap, "/peer/catalog", timeout=args.http_timeout)
    except Exception as exc:
        return {
            "schema": "p2p_external_existing_probe_v1",
            "ok": False,
            "p2p_url": args.peer_bootstrap,
            "diagnosis_codes": ["external_p2p_catalog_unreachable"],
            "error": type(exc).__name__,
            "detail": str(exc)[:240],
        }, steps, payloads
    write_json(output_dir / "external_p2p_catalog.json", catalog)
    coordinator_count, stage0_count, stage1_count = stage_peer_count(catalog)
    route = route_from_catalog(
        catalog,
        prompt_text=args.prompt_text,
        backend=args.backend,
        hf_model_id=args.hf_model_id,
        max_new_tokens=args.max_new_tokens,
    )
    route_ready = bool(route.get("usable_now") and route.get("route_source") == "p2p-discovery")
    codes = set(diagnosis_codes(catalog))
    generate_payload: dict[str, Any] = {}
    if args.verify_generate:
        if not args.admin_token:
            codes.add("admin_token_required")
            steps.append({"name": "external_generate_p2p", "ok": False, "error": "admin_token_required"})
        else:
            command = [
                sys.executable,
                "-m",
                "crowdtensor.cli",
                "generate",
                "--p2p",
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
                str(args.timeout_seconds),
                "--http-timeout",
                str(args.http_timeout),
                "--json",
            ]
            if getattr(args, "prompt_texts_file", ""):
                command.extend(["--prompt-texts-file", args.prompt_texts_file])
            elif getattr(args, "prompt_texts", ""):
                command.extend(["--prompt-texts", args.prompt_texts])
            else:
                command.extend(["--prompt", args.prompt_text])
            if getattr(args, "stream_generation", False):
                command.append("--stream")
            step, generate_payload = run_json_step(
                "external_generate_p2p",
                command,
                runner=subprocess.run,
                timeout_seconds=args.timeout_seconds + 30.0,
            )
            steps.append(step)
            payloads["external_generate"] = generate_payload
            if generate_payload.get("ok"):
                codes.add("external_p2p_generate_ready")
    else:
        codes.add("external_p2p_generate_not_requested")
    if route_ready:
        codes.update({
            "external_p2p_route_ready",
            "external_p2p_stage_discovery_ready",
            "coordinator_to_p2p_transition_ready",
            "coordinator_result_fallback_ready",
        })
    else:
        codes.add("external_p2p_route_blocked")
    ready = bool(route_ready and coordinator_count >= 1 and stage0_count >= 1 and stage1_count >= 1)
    generate_ready = bool(generate_payload.get("ok"))
    if ready:
        codes.add("external_p2p_runtime_verified")
    if generate_ready:
        codes.add("external_p2p_generate_verified")
    generation = generate_payload.get("generation") if isinstance(generate_payload.get("generation"), dict) else {}
    batch = product_mvp.safe_batch_summary(args, generation) if generate_payload else {
        "enabled": bool(str(getattr(args, "prompt_texts", "") or "").strip() or str(getattr(args, "prompt_texts_file", "") or "").strip()),
        "batch_generation_ready": False,
        "raw_prompts_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }
    stream = product_mvp.safe_stream_summary(args, generate_payload) if generate_payload else {
        "enabled": bool(getattr(args, "stream_generation", False)),
        "requested": bool(getattr(args, "stream_generation", False)),
        "stream_generation_ready": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }
    if batch.get("batch_generation_ready"):
        codes.add("p2p_external_generate_batch_ready")
        codes.add("public_swarm_generate_batch_ready")
    if stream.get("stream_generation_ready"):
        codes.add("p2p_external_generate_stream_ready")
        codes.add("public_swarm_generate_stream_ready")
        if stream.get("endpoint_ready"):
            codes.add("public_swarm_generate_stream_endpoint_ready")
    return {
        "schema": "p2p_external_existing_probe_v1",
        "ok": bool(ready and (generate_ready if args.verify_generate else True)),
        "hf_model_id": args.hf_model_id,
        "requested_hf_model_id": args.hf_model_id,
        "p2p_url": args.peer_bootstrap,
        "catalog_peer_count": len(catalog.get("peers") or []),
        "coordinator_peer_count": coordinator_count,
        "stage0_peer_count": stage0_count,
        "stage1_peer_count": stage1_count,
        "generate_route": route,
        "verify_generate": bool(args.verify_generate),
        "external_runtime_verified": ready,
        "external_generate_verified": generate_ready,
        "generate": generate_payload,
        "batch": batch,
        "stream": stream,
        "diagnosis_codes": sorted(codes or {"external_p2p_route_blocked"}),
    }, steps, payloads


def p2p_v06_source_files() -> list[Path]:
    files = [
        ROOT / "pyproject.toml",
        ROOT / "coordinator.py",
        ROOT / "miner_cli.py",
    ]
    files.extend(sorted((ROOT / "crowdtensor").rglob("*.py")))
    files.extend([
        ROOT / "scripts" / "p2p_lite_daemon.py",
        ROOT / "scripts" / "support_bundle.py",
        ROOT / "scripts" / "doctor.py",
        ROOT / "scripts" / "release_evidence_pack.py",
        ROOT / "scripts" / "release_gate.py",
    ])
    return [path for path in files if path.is_file() and "__pycache__" not in path.parts]


def build_p2p_v06_source_tarball(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    included: list[str] = []
    with tarfile.open(path, "w:gz") as tar:
        for file_path in p2p_v06_source_files():
            arcname = file_path.resolve().relative_to(ROOT.resolve()).as_posix()
            tar.add(file_path, arcname=arcname)
            included.append(arcname)
    return {
        "path": str(path),
        "file_count": len(included),
        "included_roots": sorted({item.split("/", 1)[0] for item in included}),
    }


def render_p2p_v06_kaggle_kernel(
    *,
    stage: str,
    p2p_url: str,
    swarm_id: str,
    miner_id: str,
    backend: str,
    hf_model_id: str,
    hf_cache_dir: str,
    miner_token: str,
    peer_secret: str,
    max_tasks: int,
    source_tarball_b64: str,
) -> str:
    peer_secret_line = f"PEER_SECRET = {json.dumps(peer_secret)}"
    peer_secret_command = '''
if PEER_SECRET:
    command.extend(["--peer-secret", PEER_SECRET])
'''
    return f'''from __future__ import annotations

import base64
import os
import subprocess
import sys
import tarfile
from pathlib import Path

STAGE = "{stage}"
P2P_URL = "{p2p_url}"
SWARM_ID = "{swarm_id}"
MINER_ID = "{miner_id}"
BACKEND = "{backend}"
HF_MODEL_ID = "{hf_model_id}"
HF_CACHE_DIR = "{hf_cache_dir}"
MINER_TOKEN = "{miner_token}"
{peer_secret_line}
MAX_TASKS = {int(max_tasks)}
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
env["CROWDTENSOR_REMOTE_ENVIRONMENT"] = "kaggle-p2p-v06"
env["CROWDTENSOR_MINER_TOKEN"] = MINER_TOKEN
log_path = Path("/kaggle/working") / f"crowdtensor_p2p_v06_{{STAGE}}.log"

with log_path.open("a", encoding="utf-8") as log:
    log.write("CrowdTensor P2P v0.6 Kaggle stage miner start\\n")
    log.write(f"stage={{STAGE}} miner_id={{MINER_ID}} p2p={{P2P_URL}}\\n")
    log.flush()
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "transformers==4.40.2",
    ], stdout=log, stderr=subprocess.STDOUT)

command = [
    sys.executable,
    "-m",
    "crowdtensor.cli",
    "join",
    "--p2p",
    "--peer-bootstrap",
    P2P_URL,
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
    "--once",
    "--max-tasks",
    str(MAX_TASKS),
    "--max-request-attempts",
    "180",
    "--retry-base-sleep",
    "1.0",
    "--retry-max-sleep",
    "5.0",
    "--idle-sleep",
    "1.0",
    "--run",
    "--json",
]
{peer_secret_command}
print("Starting CrowdTensor P2P v0.6 Kaggle stage miner:", " ".join(command), flush=True)
with log_path.open("a", encoding="utf-8") as log:
    log.write("Starting command: " + " ".join(command) + "\\n")
    log.flush()
    process = subprocess.Popen(command, cwd=str(src_dir), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    raise SystemExit(process.wait())
'''


def build_p2p_v06_kaggle_package(args: argparse.Namespace, *, output_dir: Path, miner_token: str) -> dict[str, Any]:
    owner = args.kaggle_owner or default_kaggle_owner()
    if not owner:
        return {"schema": "p2p_v06_kaggle_package_v1", "ok": False, "diagnosis_codes": ["kaggle_owner_missing"]}
    package_dir = output_dir / "kaggle-package"
    kernels_dir = package_dir / "kernels"
    package_dir.mkdir(parents=True, exist_ok=True)
    source = build_p2p_v06_source_tarball(package_dir / "crowdtensor_source.tar.gz")
    source_b64 = base64.b64encode((package_dir / "crowdtensor_source.tar.gz").read_bytes()).decode("ascii")
    prefix = safe_slug(args.kernel_slug_prefix or f"ct-p2p-v06-{args.p2p_port}")
    hf_cache_dir = "/kaggle/working/crowdtensor-hf-cache-p2p-v06"
    p2p_url = f"http://{args.public_host}:{args.p2p_port}"
    peer_secret = str(getattr(args, "peer_secret", "") or "")
    require_signed = bool(getattr(args, "require_signed", False))
    stages: list[dict[str, Any]] = []
    for stage in ("stage0", "stage1"):
        kernel_dir = kernels_dir / stage
        kernel_dir.mkdir(parents=True, exist_ok=True)
        kernel_slug = f"{prefix}-{stage}"
        miner_id = f"p2p-v06-kaggle-{stage}"
        code = render_p2p_v06_kaggle_kernel(
            stage=stage,
            p2p_url=p2p_url,
            swarm_id=args.swarm_id,
            miner_id=miner_id,
            backend=args.backend,
            hf_model_id=args.hf_model_id,
            hf_cache_dir=hf_cache_dir,
            miner_token=miner_token,
            peer_secret=peer_secret,
            max_tasks=max(1, int(args.max_new_tokens)),
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
        stages.append({
            "stage": stage,
            "miner_id": miner_id,
            "kernel_ref": f"{owner}/{kernel_slug}",
            "kernel_dir": str(kernel_dir),
            "p2p_join_present": "crowdtensor.cli" in code and "--p2p" in code and "--peer-bootstrap" in code,
            "signed_join_present": bool((not require_signed) or ("--peer-secret" in code and bool(peer_secret))),
            "inline_private_token": True,
            "inline_private_peer_secret": bool(peer_secret),
        })
    ok = bool(all(item.get("p2p_join_present") and item.get("signed_join_present") for item in stages))
    report = {
        "schema": "p2p_v06_kaggle_package_v1",
        "ok": ok,
        "output_dir": str(package_dir),
        "p2p_url": p2p_url,
        "source": source,
        "stages": stages,
        "diagnosis_codes": ["p2p_v06_kaggle_package_ready"] if ok else ["p2p_v06_kaggle_package_blocked"],
        "safety": {
            "private_kernel_payload_contains_miner_token": True,
            "private_kernel_payload_contains_peer_secret": bool(peer_secret),
            "source_tarball_excludes_git_and_dist": True,
            "not_production": True,
        },
    }
    write_json(package_dir / "p2p_v06_kaggle_package.json", report)
    return report


def poll_process_json(proc: subprocess.Popen[str] | None) -> dict[str, Any]:
    if proc is None or proc.poll() is None:
        return {}
    try:
        stdout, _stderr = proc.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        return {}
    return json_from_stdout(stdout or "")


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


def cleanup_local_kaggle_private_artifacts(payloads: dict[str, Any]) -> dict[str, Any]:
    package = payloads.get("kaggle_package") if isinstance(payloads.get("kaggle_package"), dict) else {}
    package_dir = Path(str(package.get("output_dir") or ""))
    removed: list[str] = []
    errors: list[str] = []
    for path in [package_dir / "kernels"]:
        if not path.exists():
            continue
        try:
            shutil.rmtree(path)
            removed.append(str(path))
        except Exception as exc:  # pragma: no cover - defensive cleanup path.
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return {
        "schema": "p2p_v06_kaggle_private_cleanup_v1",
        "ok": not errors,
        "removed_paths": removed,
        "errors": errors,
        "private_kernel_payloads_removed": True,
    }


def kaggle_auto_hf_preflight_report(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    missing: list[str],
) -> dict[str, Any]:
    report = {
        "schema": "p2p_swarm_inference_v06_kaggle_auto_v1",
        "ok": False,
        "output_dir": str(output_dir),
        "p2p_url": f"http://{args.public_host}:{args.p2p_port}",
        "coordinator_url": f"http://{args.public_host}:{args.coordinator_port}",
        "catalog_peer_count": 0,
        "coordinator_peer_count": 0,
        "stage0_peer_count": 0,
        "stage1_peer_count": 0,
        "generate_route": {},
        "verify_generate": True,
        "external_runtime_verified": False,
        "external_generate_verified": False,
        "generation": {},
        "stage_assignment": {},
        "ledger": {"accepted_rows": 0, "error": ""},
        "catalog_error": "",
        "kaggle_lifecycle": {
            "pushed_refs": {},
            "cleanup_steps": [],
            "kernels_deleted": True,
            "token_rotation_required": False,
        },
        "steps": [
            {
                "name": "host_hf_runtime_preflight",
                "ok": False,
                "missing_dependencies": missing,
                "error": "hf_dependencies_missing",
            }
        ],
        "payload_summaries": {},
        "processes": {},
        "diagnosis_codes": [
            "hf_dependencies_missing",
            "host_hf_runtime_missing",
            "p2p_swarm_inference_v06_kaggle_auto_blocked",
        ],
        "operator_action": "Install optional host runtime dependencies with: python -m pip install -e '.[hf]' before running kaggle-auto.",
        "safety": {
            "p2p_discovery_routing": True,
            "coordinator_result_fallback": True,
            "temporary_public_http": False,
            "token_rotation_required": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "not_production": True,
        },
    }
    write_json(output_dir / "p2p_v06_kaggle_auto.json", support_bundle.sanitize(redact_values(report)))
    return support_bundle.sanitize(redact_values(report))


def finalize_kaggle_auto(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    steps: list[dict[str, Any]],
    payloads: dict[str, Any],
    pushed_refs: dict[str, str],
    cleanup_steps: list[dict[str, Any]],
    secret_values: list[str],
    p2pd_proc: subprocess.Popen[str] | None,
    serve_proc: subprocess.Popen[str] | None,
    generate_proc: subprocess.Popen[str] | None,
) -> dict[str, Any]:
    cleanup_pushed_kaggle_kernels(
        args,
        pushed_refs=pushed_refs,
        cleanup_steps=cleanup_steps,
        secret_values=secret_values,
    )
    payloads["local_private_cleanup"] = cleanup_local_kaggle_private_artifacts(payloads)
    return finish_kaggle_auto_report(
        args,
        output_dir=output_dir,
        steps=steps,
        payloads=payloads,
        pushed_refs=pushed_refs,
        cleanup_steps=cleanup_steps,
        p2pd_proc=p2pd_proc,
        serve_proc=serve_proc,
        generate_proc=generate_proc,
    )


def run_kaggle_auto(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    missing = missing_hf_dependencies()
    if missing:
        return kaggle_auto_hf_preflight_report(args, output_dir=output_dir, missing=missing)
    miner_token = secrets.token_urlsafe(32)
    admin_token = secrets.token_urlsafe(32)
    observer_token = secrets.token_urlsafe(32)
    peer_secret = str(getattr(args, "peer_secret", "") or "")
    require_signed = bool(getattr(args, "require_signed", False))
    secret_values = [miner_token, admin_token, observer_token, peer_secret]
    p2p_url = f"http://{args.public_host}:{args.p2p_port}"
    coordinator_url = f"http://{args.public_host}:{args.coordinator_port}"
    local_coordinator_url = f"http://127.0.0.1:{args.coordinator_port}"
    state_dir = output_dir / "state"
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {}
    p2pd_proc: subprocess.Popen[str] | None = None
    serve_proc: subprocess.Popen[str] | None = None
    generate_proc: subprocess.Popen[str] | None = None
    pushed_refs: dict[str, str] = {}
    cleanup_steps: list[dict[str, Any]] = []
    package = build_p2p_v06_kaggle_package(args, output_dir=output_dir, miner_token=miner_token)
    payloads["kaggle_package"] = package
    payloads["runtime"] = {
        "admin_token": admin_token,
        "observer_token": observer_token,
    }
    try:
        p2pd_proc = popen_process([
            sys.executable,
            str(ROOT / "scripts" / "p2p_lite_daemon.py"),
            "--host",
            "0.0.0.0",
            "--port",
            str(args.p2p_port),
            "--swarm-id",
            args.swarm_id,
            "--role",
            "observer",
            *(["--peer-secret", peer_secret] if peer_secret else []),
            *(["--require-signed"] if require_signed else []),
        ])
        p2pd_ready, p2pd_error = wait_p2pd(f"http://127.0.0.1:{args.p2p_port}", p2pd_proc, timeout=args.startup_timeout)
        steps.append({"name": "p2pd_public", "ok": p2pd_ready, "error": p2pd_error})
        if not p2pd_ready:
            report = finalize_kaggle_auto(args, output_dir=output_dir, steps=steps, payloads=payloads, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values, p2pd_proc=p2pd_proc, serve_proc=serve_proc, generate_proc=generate_proc)
            p2pd_proc = None
            serve_proc = None
            generate_proc = None
            return report

        serve_cmd = [
            sys.executable,
            "-m",
            "crowdtensor.cli",
            "serve",
            "--p2p",
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
            "--i-understand-public-bind",
            "--run",
            "--json",
        ]
        if peer_secret:
            serve_cmd.extend(["--peer-secret", peer_secret])
        if args.hf_cache_dir:
            serve_cmd.extend(["--hf-cache-dir", args.hf_cache_dir])
        serve_proc = product_mvp.popen_command(serve_cmd)
        serve_ready, serve_error = product_mvp.wait_health(local_coordinator_url, serve_proc, args.startup_timeout)
        steps.append({"name": "serve_public_p2p", "ok": serve_ready, "error": serve_error})
        if not serve_ready:
            report = finalize_kaggle_auto(args, output_dir=output_dir, steps=steps, payloads=payloads, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values, p2pd_proc=p2pd_proc, serve_proc=serve_proc, generate_proc=generate_proc)
            p2pd_proc = None
            serve_proc = None
            generate_proc = None
            return report

        for stage_report in package.get("stages") or []:
            if not isinstance(stage_report, dict):
                continue
            stage = str(stage_report.get("stage") or "")
            ref = str(stage_report.get("kernel_ref") or "")
            kernel_dir = str(stage_report.get("kernel_dir") or "")
            step = run_kaggle_step(
                f"kaggle_push_{stage}",
                ["kaggle", "kernels", "push", "-p", kernel_dir],
                timeout_seconds=args.kaggle_push_timeout_seconds,
                secret_values=secret_values,
            )
            step["stage"] = stage
            step["kernel_ref"] = ref
            steps.append(step)
            if step.get("ok"):
                pushed_refs[stage] = str(step.get("actual_kernel_ref") or ref)
        if len(pushed_refs) < 2:
            report = finalize_kaggle_auto(args, output_dir=output_dir, steps=steps, payloads=payloads, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values, p2pd_proc=p2pd_proc, serve_proc=serve_proc, generate_proc=generate_proc)
            p2pd_proc = None
            serve_proc = None
            generate_proc = None
            return report

        stage_ready, stage_catalog, stage_error = wait_p2p_stage_miners(
            f"http://127.0.0.1:{args.p2p_port}",
            timeout=args.kaggle_stage_timeout_seconds,
            http_timeout=args.http_timeout,
        )
        steps.append({
            "name": "wait_kaggle_stage_miners_p2p",
            "ok": stage_ready,
            "error": stage_error,
            "catalog_peer_count": len(stage_catalog.get("peers") or []),
            "stage_counts": dict(zip(["coordinator", "stage0", "stage1"], stage_peer_count(stage_catalog))),
        })
        if not stage_ready:
            payloads["stage_catalog"] = stage_catalog
            report = finalize_kaggle_auto(args, output_dir=output_dir, steps=steps, payloads=payloads, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values, p2pd_proc=p2pd_proc, serve_proc=serve_proc, generate_proc=generate_proc)
            p2pd_proc = None
            serve_proc = None
            generate_proc = None
            return report
        payloads["stage_catalog"] = stage_catalog

        generate_cmd = p2p_generate_command(
            p2p_url=p2p_url,
            admin_token=admin_token,
            backend=args.backend,
            hf_model_id=args.hf_model_id,
            max_new_tokens=args.max_new_tokens,
            timeout_seconds=args.timeout_seconds,
            http_timeout=max(float(args.http_timeout), 60.0),
            prompt_text=args.prompt_text,
            prompt_texts=getattr(args, "prompt_texts", ""),
            prompt_texts_file=getattr(args, "prompt_texts_file", ""),
        )
        generate_proc = product_mvp.popen_command(generate_cmd)
        generate_step, generate_payload = product_mvp.finish_process_step("generate_p2p_kaggle", generate_proc, timeout=args.timeout_seconds + 120.0)
        if generate_payload:
            generate_step["payload_diagnosis_codes"] = generate_payload.get("diagnosis_codes") or []
            generate_step["payload_error"] = generate_payload.get("error") or ""
            generate_step["payload_detail"] = generate_payload.get("detail") or ""
        steps.append(generate_step)
        payloads["generate"] = generate_payload
        generate_proc = None
        report = finalize_kaggle_auto(args, output_dir=output_dir, steps=steps, payloads=payloads, pushed_refs=pushed_refs, cleanup_steps=cleanup_steps, secret_values=secret_values, p2pd_proc=p2pd_proc, serve_proc=serve_proc, generate_proc=generate_proc)
        p2pd_proc = None
        serve_proc = None
        generate_proc = None
        return report
    finally:
        if pushed_refs and not cleanup_steps:
            cleanup_pushed_kaggle_kernels(
                args,
                pushed_refs=pushed_refs,
                cleanup_steps=cleanup_steps,
                secret_values=secret_values,
            )
        if generate_proc is not None and generate_proc.poll() is None:
            stop_process(generate_proc)
        if serve_proc is not None:
            product_mvp.stop_process(serve_proc)
        if p2pd_proc is not None:
            stop_process(p2pd_proc)


def finish_kaggle_auto_report(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    steps: list[dict[str, Any]],
    payloads: dict[str, Any],
    pushed_refs: dict[str, str],
    cleanup_steps: list[dict[str, Any]],
    p2pd_proc: subprocess.Popen[str] | None,
    serve_proc: subprocess.Popen[str] | None,
    generate_proc: subprocess.Popen[str] | None,
) -> dict[str, Any]:
    if "local_private_cleanup" not in payloads:
        payloads["local_private_cleanup"] = cleanup_local_kaggle_private_artifacts(payloads)
    catalog: dict[str, Any] = {}
    catalog_error = ""
    try:
        catalog = request_json(f"http://127.0.0.1:{args.p2p_port}", "/peer/catalog", timeout=args.http_timeout)
        write_json(output_dir / "external_p2p_catalog.json", catalog)
    except Exception as exc:
        catalog_error = f"{type(exc).__name__}: {exc}"
    coordinator_count, stage0_count, stage1_count = stage_peer_count(catalog)
    rows: list[dict[str, Any]] = []
    ledger_error = ""
    session_id = str(((payloads.get("generate") or {}).get("session") or {}).get("session_id") or "")
    try:
        rows = v06_admin_results(
            f"http://127.0.0.1:{args.coordinator_port}",
            session_id=session_id,
            limit=100,
            admin_token=str((payloads.get("runtime") or {}).get("admin_token") or ADMIN_TOKEN),
        )
    except Exception as exc:
        ledger_error = f"{type(exc).__name__}: {exc}"
    generation = (payloads.get("generate") or {}).get("generation") if isinstance((payloads.get("generate") or {}).get("generation"), dict) else {}
    stage_summary = product_mvp.stage_summary(rows)
    generate_route = (payloads.get("generate") or {}).get("route") if isinstance((payloads.get("generate") or {}).get("route"), dict) else {}
    discovery_catalog = payloads.get("stage_catalog") if isinstance(payloads.get("stage_catalog"), dict) else {}
    discovery_coordinator_count, discovery_stage0_count, discovery_stage1_count = stage_peer_count(discovery_catalog)
    cleanup_ok = bool(pushed_refs) and all(step.get("ok") for step in cleanup_steps)
    local_private_cleanup = payloads.get("local_private_cleanup") if isinstance(payloads.get("local_private_cleanup"), dict) else {}
    local_private_cleanup_ok = bool(local_private_cleanup.get("ok", True))
    external_generate_verified = bool((payloads.get("generate") or {}).get("ok") and int(generation.get("generated_token_count") or 0) >= args.max_new_tokens)
    stage_seen = bool((stage0_count >= 1 and stage1_count >= 1) or (discovery_stage0_count >= 1 and discovery_stage1_count >= 1))
    distinct_stage_miners = bool(stage_summary.get("distinct_stage_miners"))
    ok = bool(
        all(step.get("ok") for step in steps)
        and bool((payloads.get("kaggle_package") or {}).get("ok"))
        and stage_seen
        and external_generate_verified
        and distinct_stage_miners
        and cleanup_ok
        and local_private_cleanup_ok
        and not ledger_error
        and not catalog_error
    )
    codes = set(diagnosis_codes(*payloads.values()))
    if stage_seen:
        codes.update({"external_p2p_runtime_verified", "external_p2p_stage_discovery_ready", "p2p_kaggle_stage_miners_discovered"})
    if external_generate_verified:
        codes.update({"external_p2p_generate_ready", "external_p2p_generate_verified", "p2p_kaggle_generate_ready"})
    if cleanup_ok:
        codes.add("kaggle_kernels_deleted")
    else:
        codes.add("kaggle_cleanup_failed")
    if local_private_cleanup_ok:
        codes.add("p2p_v06_kaggle_private_artifacts_cleaned")
    else:
        codes.add("p2p_v06_kaggle_private_artifacts_cleanup_failed")
    if ok:
        codes.update({
            "p2p_swarm_inference_v06_kaggle_auto_ready",
            "external_existing_p2p_verified",
            "coordinator_to_p2p_transition_ready",
            "coordinator_result_fallback_ready",
            "tiny_gpt2_multi_token_ready",
        })
        if getattr(args, "require_signed", False):
            codes.update({
                "signed_peer_announcement_ready",
                "peer_identity_ready",
                "peer_registry_health_ready",
            })
    else:
        codes.add("p2p_swarm_inference_v06_kaggle_auto_blocked")
    report = {
        "schema": "p2p_swarm_inference_v06_kaggle_auto_v1",
        "ok": ok,
        "output_dir": str(output_dir),
        "p2p_url": f"http://{args.public_host}:{args.p2p_port}",
        "coordinator_url": f"http://{args.public_host}:{args.coordinator_port}",
        "catalog_peer_count": len(catalog.get("peers") or []),
        "coordinator_peer_count": coordinator_count,
        "stage0_peer_count": stage0_count,
        "stage1_peer_count": stage1_count,
        "discovery_catalog_peer_count": len(discovery_catalog.get("peers") or []),
        "discovery_coordinator_peer_count": discovery_coordinator_count,
        "discovery_stage0_peer_count": discovery_stage0_count,
        "discovery_stage1_peer_count": discovery_stage1_count,
        "generate_route": generate_route,
        "verify_generate": True,
        "external_runtime_verified": stage_seen,
        "external_generate_verified": external_generate_verified,
        "generation": generation,
        "stage_assignment": stage_summary,
        "ledger": {"accepted_rows": len(rows), "error": ledger_error},
        "catalog_error": catalog_error,
        "kaggle_lifecycle": {
            "pushed_refs": pushed_refs,
            "cleanup_steps": cleanup_steps,
            "kernels_deleted": cleanup_ok,
            "local_private_cleanup": local_private_cleanup,
            "local_private_artifacts_cleaned": local_private_cleanup_ok,
            "token_rotation_required": True,
        },
        "steps": steps,
        "payload_summaries": {
            "kaggle_package": payloads.get("kaggle_package") or {},
            "generate": {
                "schema": (payloads.get("generate") or {}).get("schema"),
                "ok": (payloads.get("generate") or {}).get("ok"),
                "generation": generation,
            },
        },
        "processes": {
            "serve": product_mvp.stop_process(serve_proc) if serve_proc is not None else {},
            "p2pd": stop_process(p2pd_proc) if p2pd_proc is not None else {},
        },
        "diagnosis_codes": sorted(codes),
        "safety": {
            "p2p_discovery_routing": True,
            "coordinator_result_fallback": True,
            "temporary_public_http": True,
            "token_rotation_required": True,
            "signed_peer_announcement": bool(getattr(args, "require_signed", False)),
            "peer_secret_gossiped": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "not_production": True,
        },
    }
    write_json(output_dir / "p2p_v06_kaggle_auto.json", support_bundle.sanitize(redact_values(report)))
    return support_bundle.sanitize(redact_values(report))


def write_runbook(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    path = output_dir / "P2P_SWARM_INFERENCE_V06.md"
    lines = [
        "# CrowdTensor P2P Swarm Inference v0.6 Prototype",
        "",
        "This runbook uses P2P discovery for route selection while the Coordinator remains the lease/result-ledger authority.",
        "",
        "```bash",
        f"crowdtensor p2pd --host 0.0.0.0 --port {args.p2p_port} --run",
        f"crowdtensor serve --p2p --peer-bootstrap http://{args.public_host}:{args.p2p_port} --public-host {args.public_host} --port {args.coordinator_port} --run",
        f"crowdtensor join --p2p --peer-bootstrap http://{args.public_host}:{args.p2p_port} --stage stage0 --miner-id p2p-stage0 --run",
        f"crowdtensor join --p2p --peer-bootstrap http://{args.public_host}:{args.p2p_port} --stage stage1 --miner-id p2p-stage1 --run",
        f"crowdtensor generate --p2p --peer-bootstrap http://{args.public_host}:{args.p2p_port} --prompt 'CrowdTensor P2P v0.6' --max-new-tokens {args.max_new_tokens} --json",
        "```",
        "",
        "For Kaggle, expose the p2pd and Coordinator host from the operator machine, then run stage0 and stage1 joins from separate private Kaggle CPU/GPU notebooks.",
        "",
        "Rotate tokens after every temporary public HTTP/Kaggle proof. Do not publish raw prompts, generated text, token ids, activations, private env files, or runtime state.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return artifact_entry(path, output_dir, kind="p2p_swarm_inference_v06_runbook")


def build_common_report(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    mode: str,
    local_p2p: dict[str, Any],
    steps: list[dict[str, Any]],
    payloads: dict[str, Any],
    preview_payload: dict[str, Any],
    product_payload: dict[str, Any],
    optional_payload: dict[str, Any],
    runbook: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preview_summary = summarize_preview_v04(preview_payload)
    product_summary = summarize_product(product_payload, required_tokens=max(2, args.max_new_tokens))
    optional_summary = summarize_product(optional_payload, required_tokens=max(2, args.max_new_tokens))
    p2p_ready = bool(local_p2p.get("ok"))
    preview_ready = bool(preview_summary.get("ready"))
    tiny_ready = bool(product_summary.get("tiny_gpt2_multi_token_ready") or preview_summary.get("tiny_gpt2_ci_fallback_ready"))
    optional_ready = bool(optional_summary.get("ready") or preview_summary.get("optional_model_ready"))
    optional_diagnosis = bool(optional_ready or "hf_dependencies_missing" in set(optional_summary.get("diagnosis_codes") or []))
    external_ready = bool(preview_summary.get("external_two_stage_generation_ready"))
    requeue_ready = bool(preview_summary.get("external_stage_requeue_ready"))
    package_ready = bool(mode == MODE_PACKAGE and runbook and runbook.get("present"))
    p2p_required = mode in {MODE_LOCAL_SMOKE, MODE_EVIDENCE_IMPORT, MODE_EXTERNAL_EXISTING, MODE_KAGGLE_AUTO}
    observed_model_id = first_string_value(local_p2p, "hf_model_id")
    model_id_present = bool(observed_model_id)
    model_id_match = bool(model_id_present and observed_model_id == args.hf_model_id)
    ready = bool(
        (p2p_ready or not p2p_required)
        and (model_id_match or not p2p_required)
        and preview_ready
        and tiny_ready
        and optional_diagnosis
        and external_ready
        and requeue_ready
        and (mode != MODE_PACKAGE or package_ready)
    )

    codes = set(diagnosis_codes(preview_payload, product_payload, optional_payload, local_p2p)) - SUPPRESSED_INHERITED_CODES
    if ready:
        codes.update({
            "p2p_swarm_inference_v06_ready",
            "p2p_discovery_routing_prototype_ready",
            "coordinator_to_p2p_transition_ready",
            "coordinator_result_fallback_ready",
            "external_two_stage_generation_ready",
            "external_stage_requeue_ready",
            "tiny_gpt2_multi_token_ready",
            "distilgpt2_attempt_ready" if optional_diagnosis else "distilgpt2_resource_diagnosis_missing",
            "completed_p2p_discovery_routing_prototype",
            "not_production_nat_traversal",
            "not_decentralized_security",
            "not_economic_system",
            "not_large_model_throughput",
        })
    else:
        codes.add("p2p_swarm_inference_v06_blocked")
    if p2p_ready:
        codes.update(local_p2p.get("diagnosis_codes") or [])
    if package_ready:
        codes.update({"p2p_two_machine_kaggle_runbook_ready", "external_p2p_join_path_documented"})
    if mode == MODE_EXTERNAL_EXISTING and p2p_ready:
        codes.update({
            "external_existing_p2p_verified",
            "external_p2p_runtime_verified",
        })
    if mode == MODE_KAGGLE_AUTO and p2p_ready:
        codes.update({
            "external_existing_p2p_verified",
            "external_p2p_runtime_verified",
            "external_p2p_generate_verified",
            "p2p_swarm_inference_v06_kaggle_auto_ready",
        })

    p2p_catalog_path = output_dir / ("external_p2p_catalog.json" if mode == MODE_EXTERNAL_EXISTING else "p2p_catalog.json")
    if model_id_match:
        codes.add("p2p_v06_model_metadata_ready")
    elif p2p_required:
        codes.add("p2p_v06_model_metadata_mismatch")
    else:
        codes.add("p2p_v06_model_metadata_not_applicable")
    kv_cache = local_p2p.get("kv_cache") if isinstance(local_p2p.get("kv_cache"), dict) else {}
    if not kv_cache and isinstance(local_p2p.get("real_generate_probe"), dict):
        real_probe = local_p2p.get("real_generate_probe") or {}
        kv_cache = real_probe.get("kv_cache") if isinstance(real_probe.get("kv_cache"), dict) else {}
    if kv_cache.get("ready"):
        codes.update({
            "p2p_real_generate_kv_cache_ready",
            "real_llm_stage0_kv_cache_v1_ready",
            "real_llm_stage1_kv_cache_v1_ready",
            "stage0_kv_cache_hits_ready",
            "stage1_kv_cache_hits_ready",
        })
    stream = local_p2p.get("stream") if isinstance(local_p2p.get("stream"), dict) else {}
    if not stream and isinstance(local_p2p.get("real_generate_probe"), dict):
        real_probe = local_p2p.get("real_generate_probe") or {}
        stream = real_probe.get("stream") if isinstance(real_probe.get("stream"), dict) else {}
    batch = local_p2p.get("batch") if isinstance(local_p2p.get("batch"), dict) else {}
    if not batch and isinstance(local_p2p.get("real_generate_probe"), dict):
        real_probe = local_p2p.get("real_generate_probe") or {}
        batch = real_probe.get("batch") if isinstance(real_probe.get("batch"), dict) else {}
    stream_ready = stream_evidence_ready(stream, batch)
    if stream_ready:
        codes.update({
            "p2p_real_generate_stream_ready",
            "public_swarm_generate_stream_ready",
        })
        if stream.get("endpoint_ready"):
            codes.add("public_swarm_generate_stream_endpoint_ready")
    artifacts = {
        "p2p_swarm_inference_v06_json": artifact_entry(output_dir / "p2p_swarm_inference_v06.json", output_dir, kind="p2p_swarm_inference_v06", schema=SCHEMA, ok=ready),
        "p2p_swarm_inference_v06_markdown": artifact_entry(output_dir / "p2p_swarm_inference_v06.md", output_dir, kind="p2p_swarm_inference_v06_markdown"),
        "p2p_catalog_json": artifact_entry(p2p_catalog_path, output_dir, kind="p2p_lite_catalog", schema="p2p_lite_catalog_v1"),
        "preview_v04_source_json": artifact_entry(Path(args.preview_v04_report), output_dir, kind="public_swarm_preview_v04_source", schema=preview_v04.SCHEMA, ok=preview_payload.get("ok") if preview_payload else None),
        "product_mvp_source_json": artifact_entry(Path(args.product_mvp_report), output_dir, kind="product_swarm_mvp_source", schema=preview_v04.PRODUCT_MVP_SCHEMA, ok=product_payload.get("ok") if product_payload else None),
        "optional_model_source_json": artifact_entry(Path(args.optional_model_report), output_dir, kind="product_swarm_optional_source", schema=preview_v04.PRODUCT_MVP_SCHEMA, ok=optional_payload.get("ok") if optional_payload else None),
    }
    if runbook:
        artifacts["p2p_swarm_inference_v06_runbook"] = runbook

    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": mode,
        "output_dir": str(output_dir),
        "p2p": {
            "ready": p2p_ready,
            "hf_model_id": args.hf_model_id,
            "observed_hf_model_id": observed_model_id,
            "model_id_present": model_id_present,
            "model_id_match": model_id_match,
            "swarm_id": args.swarm_id,
            "p2p_url": local_p2p.get("p2p_url") or f"http://{args.public_host}:{args.p2p_port}",
            "catalog_peer_count": local_p2p.get("catalog_peer_count"),
            "coordinator_peer_count": local_p2p.get("coordinator_peer_count"),
            "stage0_peer_count": local_p2p.get("stage0_peer_count"),
            "stage1_peer_count": local_p2p.get("stage1_peer_count"),
            "discovery_catalog_peer_count": local_p2p.get("discovery_catalog_peer_count"),
            "discovery_coordinator_peer_count": local_p2p.get("discovery_coordinator_peer_count"),
            "discovery_stage0_peer_count": local_p2p.get("discovery_stage0_peer_count"),
            "discovery_stage1_peer_count": local_p2p.get("discovery_stage1_peer_count"),
            "generate_route": local_p2p.get("generate_route") or {},
            "generation": local_p2p.get("generation") or {},
            "stage_assignment": local_p2p.get("stage_assignment") or {},
            "ledger": local_p2p.get("ledger") or {},
            "kaggle_lifecycle": local_p2p.get("kaggle_lifecycle") or {},
            "stage_rescue_ready": bool((local_p2p.get("rescue_probe") or {}).get("ok")) if isinstance(local_p2p.get("rescue_probe"), dict) else False,
            "real_generate_ready": bool((local_p2p.get("real_generate_probe") or {}).get("ok")) if isinstance(local_p2p.get("real_generate_probe"), dict) else False,
            "real_generate_degraded": bool((local_p2p.get("real_generate_probe") or {}).get("degraded")) if isinstance(local_p2p.get("real_generate_probe"), dict) else False,
            "real_stage_rescue_ready": bool((local_p2p.get("real_stage_rescue_probe") or {}).get("ok")) if isinstance(local_p2p.get("real_stage_rescue_probe"), dict) else False,
            "real_stage_rescue_degraded": bool((local_p2p.get("real_stage_rescue_probe") or {}).get("degraded")) if isinstance(local_p2p.get("real_stage_rescue_probe"), dict) else False,
            "external_runtime_verified": bool(local_p2p.get("external_runtime_verified")),
            "external_generate_verified": bool(local_p2p.get("external_generate_verified")),
            "kaggle_auto_ready": bool(local_p2p.get("ok") and mode == MODE_KAGGLE_AUTO),
            "kaggle_kernels_deleted": bool(((local_p2p.get("kaggle_lifecycle") or {}).get("kernels_deleted")) if isinstance(local_p2p.get("kaggle_lifecycle"), dict) else False),
            "kv_cache": kv_cache,
            "stream": stream if stream else {"enabled": False, "stream_generation_ready": False},
        },
        "inference": {
            "preview_v04_ready": preview_ready,
            "external_two_stage_generation_ready": external_ready,
            "external_stage_requeue_ready": requeue_ready,
            "tiny_gpt2_multi_token_ready": tiny_ready,
            "optional_distilgpt2_or_gpt2_attempt_ready": optional_diagnosis,
            "optional_distilgpt2_or_gpt2_strict_ready": optional_ready,
            "workload_type": WORKLOAD_TYPE,
            "hf_model_id": args.hf_model_id,
            "max_new_tokens": args.max_new_tokens,
        },
        "steps": steps,
        "payload_summaries": {
            "local_p2p_discovery": local_p2p,
            "preview_v04": preview_summary,
            "product_mvp": product_summary,
            "optional_model": optional_summary,
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": artifacts,
        "output_request": output_request_summary(),
        "answer_scope": answer_scope_summary(),
        "shareable_summary": shareable_summary(),
        "safety": {
            "p2p_discovery_routing_prototype": True,
            "coordinator_result_fallback": True,
            "tokens_gossiped": False,
            "raw_prompts_gossiped": False,
            "activations_gossiped": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "read_only_workload": WORKLOAD_TYPE,
            "not_production": True,
            "not_nat_traversal": True,
            "not_decentralized_security": True,
            "not_economic_system": True,
            "not_large_model_throughput": True,
        },
        "completed": ["P2P discovery/routing prototype"],
        "not_completed": [
            "Production NAT traversal",
            "Decentralized security",
            "Economic system",
            "Large-model throughput",
        ],
        "limitations": [
            "v0.6 moves discovery/routing into p2pd, but Coordinator remains the lease/result-ledger authority.",
            "The external stage0/stage1 and rescue evidence is imported from retained controlled Live Preview/GPU artifacts unless a fresh external run is supplied.",
            "This is not production NAT traversal, decentralized security, payment/tokenomics, Hivemind/Petals parity, or large-model serving.",
        ],
        "operator_action": [
            "Use P2P_SWARM_INFERENCE_V06.md for a two-machine or Kaggle rehearsal.",
            "Run a fresh external-existing P2P proof before claiming external P2P runtime verification.",
            "Rotate tokens after temporary public HTTP/Kaggle proofs.",
        ],
    }
    return support_bundle.sanitize(redact_values(report))


def build_local_smoke(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    local_p2p, steps, payloads = run_local_p2p_discovery(args, output_dir=output_dir, runner=runner)
    return build_common_report(
        args,
        output_dir=output_dir,
        mode=MODE_LOCAL_SMOKE,
        local_p2p=local_p2p,
        steps=steps,
        payloads=payloads,
        preview_payload=load_json(args.preview_v04_report),
        product_payload=load_json(args.product_mvp_report),
        optional_payload=load_json(args.optional_model_report),
    )


def build_package(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    runbook = write_runbook(args, output_dir)
    return build_common_report(
        args,
        output_dir=output_dir,
        mode=MODE_PACKAGE,
        local_p2p={},
        steps=[],
        payloads={},
        preview_payload=load_json(args.preview_v04_report),
        product_payload=load_json(args.product_mvp_report),
        optional_payload=load_json(args.optional_model_report),
        runbook=runbook,
    )


def build_evidence_import(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    return build_common_report(
        args,
        output_dir=output_dir,
        mode=MODE_EVIDENCE_IMPORT,
        local_p2p=normalize_local_p2p_report(load_json(args.p2p_discovery_report)),
        steps=[],
        payloads={},
        preview_payload=load_json(args.preview_v04_report),
        product_payload=load_json(args.product_mvp_report),
        optional_payload=load_json(args.optional_model_report),
    )


def build_external_existing(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    local_p2p, steps, payloads = run_external_existing_probe(args, output_dir=output_dir)
    return build_common_report(
        args,
        output_dir=output_dir,
        mode=MODE_EXTERNAL_EXISTING,
        local_p2p=local_p2p,
        steps=steps,
        payloads=payloads,
        preview_payload=load_json(args.preview_v04_report),
        product_payload=load_json(args.product_mvp_report),
        optional_payload=load_json(args.optional_model_report),
    )


def build_kaggle_auto(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    local_p2p = run_kaggle_auto(args, output_dir=output_dir / "kaggle-auto")
    return build_common_report(
        args,
        output_dir=output_dir,
        mode=MODE_KAGGLE_AUTO,
        local_p2p=local_p2p,
        steps=local_p2p.get("steps") if isinstance(local_p2p.get("steps"), list) else [],
        payloads={},
        preview_payload=load_json(args.preview_v04_report),
        product_payload=load_json(args.product_mvp_report),
        optional_payload=load_json(args.optional_model_report),
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CrowdTensor P2P Swarm Inference v0.6",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- p2p ready: `{(report.get('p2p') or {}).get('ready')}`",
        f"- p2p stage rescue ready: `{(report.get('p2p') or {}).get('stage_rescue_ready')}`",
        f"- p2p real generate ready: `{(report.get('p2p') or {}).get('real_generate_ready')}`",
        f"- p2p real generate degraded: `{(report.get('p2p') or {}).get('real_generate_degraded')}`",
        f"- p2p real stage rescue ready: `{(report.get('p2p') or {}).get('real_stage_rescue_ready')}`",
        f"- p2p real stage rescue degraded: `{(report.get('p2p') or {}).get('real_stage_rescue_degraded')}`",
        f"- external two-stage generation ready: `{(report.get('inference') or {}).get('external_two_stage_generation_ready')}`",
        f"- external stage requeue ready: `{(report.get('inference') or {}).get('external_stage_requeue_ready')}`",
        f"- tiny-gpt2 multi-token ready: `{(report.get('inference') or {}).get('tiny_gpt2_multi_token_ready')}`",
        "",
        "## Output Scope",
        "",
        f"- output request: `include_output={bool((report.get('output_request') or {}).get('include_output'))} raw_prompt_public={bool((report.get('output_request') or {}).get('raw_prompt_public'))} raw_generated_text_public={bool((report.get('output_request') or {}).get('raw_generated_text_public'))} generated_token_ids_public={bool((report.get('output_request') or {}).get('generated_token_ids_public'))} public_artifact_safe={bool((report.get('output_request') or {}).get('public_artifact_safe'))}`",
        f"- answer scope: `state={(report.get('answer_scope') or {}).get('scope_state')} terminal_only={bool((report.get('answer_scope') or {}).get('terminal_only'))} visible_in_terminal={bool((report.get('answer_scope') or {}).get('visible_in_terminal'))} saved_json={(report.get('answer_scope') or {}).get('saved_json_display')} saved_markdown={(report.get('answer_scope') or {}).get('saved_markdown_display')} public_artifact_safe={bool((report.get('answer_scope') or {}).get('public_artifact_safe'))}`",
        f"- shareable: `saved_artifacts={bool((report.get('shareable_summary') or {}).get('saved_artifacts_public_safe'))} raw_prompt_public={bool((report.get('shareable_summary') or {}).get('raw_prompt_public'))} raw_generated_text_public={bool((report.get('shareable_summary') or {}).get('raw_generated_text_public'))} generated_token_ids_public={bool((report.get('shareable_summary') or {}).get('generated_token_ids_public'))} answer_scope_state={(report.get('shareable_summary') or {}).get('answer_scope_state')} local_answer_terminal_only={bool((report.get('shareable_summary') or {}).get('local_answer_terminal_only'))}`",
        f"- note: {(report.get('answer_scope') or {}).get('summary')}",
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`",
        "",
        "## Completed",
        "",
    ]
    lines.extend(f"- {item}" for item in report.get("completed") or [])
    lines.extend(["", "## Not Completed", ""])
    lines.extend(f"- {item}" for item in report.get("not_completed") or [])
    lines.extend(["", "## Artifacts", ""])
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        lines.append(f"- `{name}`: `{artifact.get('path')}` present=`{artifact.get('present')}`")
    return "\n".join(lines) + "\n"


def validate_public_report(report: dict[str, Any]) -> list[str]:
    encoded = json.dumps(report, sort_keys=True)
    errors: list[str] = []
    for fragment in SECRET_FRAGMENTS:
        if fragment in encoded:
            errors.append(f"sensitive_fragment:{fragment}")
    for path in public_leak_paths(report):
        if path.endswith(".prompt_hash") or ".safety." in path:
            continue
        errors.append(f"public_leak:{path}")
    return sorted(set(errors))


def persist_report(report: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    report.setdefault("output_request", output_request_summary())
    report.setdefault("answer_scope", answer_scope_summary())
    report.setdefault("shareable_summary", shareable_summary())
    errors = validate_public_report(report)
    if errors:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"public_report_safety_failed"})
        report["safety_errors"] = errors
    report = support_bundle.sanitize(redact_values(report))
    write_json(output_dir / "p2p_swarm_inference_v06.json", report)
    (output_dir / "p2p_swarm_inference_v06.md").write_text(render_markdown(report), encoding="utf-8")
    bundle = support_bundle.sanitize({
        "schema": "p2p_swarm_inference_v06_support_bundle_v1",
        "ok": report.get("ok"),
        "diagnosis_codes": report.get("diagnosis_codes"),
        "p2p": report.get("p2p"),
        "inference": report.get("inference"),
        "artifacts": report.get("artifacts"),
        "output_request": report.get("output_request"),
        "answer_scope": report.get("answer_scope"),
        "shareable_summary": report.get("shareable_summary"),
        "safety": report.get("safety"),
    })
    write_json(output_dir / "support_bundle.json", bundle)
    report["artifacts"]["p2p_swarm_inference_v06_json"] = artifact_entry(output_dir / "p2p_swarm_inference_v06.json", output_dir, kind="p2p_swarm_inference_v06", schema=SCHEMA, ok=report.get("ok"))
    report["artifacts"]["p2p_swarm_inference_v06_markdown"] = artifact_entry(output_dir / "p2p_swarm_inference_v06.md", output_dir, kind="p2p_swarm_inference_v06_markdown")
    report["artifacts"]["support_bundle_json"] = artifact_entry(output_dir / "support_bundle.json", output_dir, kind="p2p_swarm_inference_v06_support_bundle", schema=str(bundle.get("schema")), ok=bundle.get("ok"))
    write_json(output_dir / "p2p_swarm_inference_v06.json", report)
    return report


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == MODE_LOCAL_SMOKE:
        report = build_local_smoke(args, output_dir=output_dir, runner=runner)
    elif args.mode == MODE_PACKAGE:
        report = build_package(args, output_dir=output_dir)
    elif args.mode == MODE_EVIDENCE_IMPORT:
        report = build_evidence_import(args, output_dir=output_dir)
    elif args.mode == MODE_EXTERNAL_EXISTING:
        report = build_external_existing(args, output_dir=output_dir)
    else:
        report = build_kaggle_auto(args, output_dir=output_dir)
    return persist_report(report, output_dir=output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build P2P Swarm Inference v0.6 prototype evidence.")
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--swarm-id", default="p2p-v06")
    parser.add_argument("--public-host", default="24.199.118.54")
    parser.add_argument("--p2p-port", type=int, default=DEFAULT_P2P_PORT)
    parser.add_argument("--coordinator-port", type=int, default=DEFAULT_COORDINATOR_PORT)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--hf-model-id", default=DEFAULT_HF_MODEL_ID)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--prompt-text", default="CrowdTensor P2P v0.6")
    parser.add_argument("--prompt-texts", default="", help="comma-separated bounded batch of up to 4 prompts for real-generate probes")
    parser.add_argument("--prompt-texts-file", default="", help="UTF-8 batch prompt file with one non-empty prompt per line for real-generate probes")
    parser.add_argument("--stream-generation", action="store_true", help="require safe generate --stream progress evidence in real-generate probes")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--http-timeout", type=float, default=5.0)
    parser.add_argument("--preview-v04-report", default=DEFAULT_PREVIEW_V04_REPORT)
    parser.add_argument("--product-mvp-report", default=DEFAULT_PRODUCT_MVP_REPORT)
    parser.add_argument("--optional-model-report", default=DEFAULT_OPTIONAL_MODEL_REPORT)
    parser.add_argument("--p2p-discovery-report", default="")
    parser.add_argument("--peer-bootstrap", default="")
    parser.add_argument("--peer-secret", default=os.environ.get("CROWDTENSOR_P2P_PEER_SECRET", ""))
    parser.add_argument("--require-signed", action="store_true")
    parser.add_argument("--admin-token", default=os.environ.get("CROWDTENSOR_ADMIN_TOKEN", ""))
    parser.add_argument("--verify-generate", action="store_true")
    parser.add_argument("--kaggle-owner", default=default_kaggle_owner())
    parser.add_argument("--kernel-slug-prefix", default="")
    parser.add_argument("--kaggle-push-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--kaggle-stage-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--skip-kaggle-cleanup", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.p2p_port < 1 or args.coordinator_port < 1:
        raise SystemExit("--p2p-port and --coordinator-port must be positive")
    if args.max_new_tokens < 2 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 2 and 32")
    if args.prompt_texts and args.prompt_texts_file:
        raise SystemExit("p2p_swarm_inference_v06 accepts either --prompt-texts or --prompt-texts-file, not both")
    try:
        if args.prompt_texts_file:
            args.prompt_texts_list = product_mvp.read_prompt_texts_file(args.prompt_texts_file)
        else:
            args.prompt_texts_list = product_mvp.parse_prompt_texts_arg(args.prompt_text, args.prompt_texts)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.mode == MODE_EXTERNAL_EXISTING and not args.peer_bootstrap:
        raise SystemExit("external-existing requires --peer-bootstrap")
    if args.mode == MODE_KAGGLE_AUTO and not args.kaggle_owner:
        raise SystemExit("kaggle-auto requires --kaggle-owner or KAGGLE_USERNAME/~/.kaggle/kaggle.json")
    if args.require_signed and not args.peer_secret:
        raise SystemExit("--require-signed requires --peer-secret")
    for name in ["startup_timeout", "timeout_seconds", "http_timeout"]:
        if float(getattr(args, name)) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    for name in ["kaggle_push_timeout_seconds", "kaggle_delete_timeout_seconds", "kaggle_stage_timeout_seconds"]:
        if float(getattr(args, name)) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"P2P Swarm Inference v0.6 ready: {report.get('ok')}")
        print(f"Diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
