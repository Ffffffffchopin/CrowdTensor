#!/usr/bin/env python3
"""Build the Public Swarm Inference Beta RC artifact.

The RC layer proves the current product path as a Coordinator-backed remote
inference loop: product surface, bounded session protocol, P2P-lite route
discovery, serve/join/generate execution, and CPU fallback.  It is not a
production P2P network and does not claim libp2p, DHT, NAT traversal, or
large-model serving.
"""

from __future__ import annotations

import argparse
import json
import os
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

import public_swarm_inference_beta_pack as beta_pack  # noqa: E402
import support_bundle  # noqa: E402
from crowdtensor.real_llm import missing_hf_dependencies  # noqa: E402
from product_swarm_mvp_check import parse_prompt_texts_arg  # noqa: E402


SCHEMA = "public_swarm_inference_beta_rc_v1"
BETA_SCHEMA = "public_swarm_inference_beta_v1"
REMOTE_REAL_SCHEMA = "remote_real_llm_sharded_beta_v1"
CPU_BETA_SCHEMA = "cpu_inference_beta_v1"
PRODUCT_CLI_SCHEMA = "public_swarm_product_cli_v1"
P2P_CHECK_SCHEMA = "p2p_lite_discovery_check_v1"
WORKLOAD_TYPE = "real_llm_sharded_infer"
DEFAULT_OUTPUT_DIR = "dist/public-swarm-inference-beta-rc"
DEFAULT_GPU_REPORT = beta_pack.DEFAULT_GPU_REPORT
DEFAULT_PROMPT = "CrowdTensor public beta RC"
DEFAULT_ADMIN_TOKEN = "public-swarm-beta-rc-admin"
DEFAULT_OBSERVER_TOKEN = "public-swarm-beta-rc-observer"
SECRET_FRAGMENTS = (
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
    "activation_result",
    "real_llm_sharded_result",
    "sharded_inference_result",
    "inference_results",
    '"generated_text":',
    '"generated_token_ids":',
    '"prompt_text":',
    '"prompt_texts":',
)

Runner = Callable[..., subprocess.CompletedProcess[str]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("command emitted no JSON object")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def redact_text(text: str, secret_values: list[str] | None = None) -> str:
    redacted = str(text)
    for value in secret_values or []:
        if value:
            redacted = redacted.replace(value, "<redacted>")
    for fragment in SECRET_FRAGMENTS:
        redacted = redacted.replace(fragment, "<redacted>")
    return redacted


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
        summaries = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
        for item in summaries.values():
            if isinstance(item, dict):
                for code in item.get("diagnosis_codes") or []:
                    if isinstance(code, str):
                        codes.add(code)
    return sorted(codes)


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


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


STREAM_READY_CODES = {
    "public_swarm_generate_stream_ready",
    "public_swarm_generate_stream_endpoint_ready",
}


def drop_unproven_stream_codes(codes: set[str], *, stream_ready: bool) -> set[str]:
    if stream_ready:
        return codes
    return {code for code in codes if code not in STREAM_READY_CODES}


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
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "ok": False,
            "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": "timeout",
        }, {}
    step: dict[str, Any] = {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    payload: dict[str, Any] = {}
    try:
        payload = json_from_stdout(completed.stdout)
    except ValueError as exc:
        step["ok"] = False
        step["error"] = str(exc)
    else:
        step["payload_schema"] = payload.get("schema")
        step["payload_ok"] = payload.get("ok")
        step["ok"] = bool(step.get("ok") and payload.get("ok"))
    if not step.get("ok"):
        if completed.stdout and not payload:
            step["stdout_tail"] = redact_text(completed.stdout[-1600:], secret_values)
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1600:], secret_values)
    return step, payload


def request_json(
    method: str,
    base_url: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    admin_token: str = "",
    observer_token: str = "",
    timeout: float = 5.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers: dict[str, str] = {"content-type": "application/json"} if payload is not None else {}
    if admin_token:
        headers["x-crowdtensor-admin-token"] = admin_token
        headers.setdefault("content-type", "application/json")
    if observer_token:
        headers["x-crowdtensor-observer-token"] = observer_token
    request = Request(f"{base_url.rstrip('/')}{path}", data=data, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def wait_health(base_url: str, proc: subprocess.Popen[str], *, timeout_seconds: float, http_timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return {"ok": False, "error": f"coordinator_exited:{proc.returncode}"}
        try:
            payload = request_json("GET", base_url, "/health", timeout=http_timeout)
            if payload.get("ok") is True:
                return {"ok": True, "payload": payload}
        except Exception as exc:
            last_error = type(exc).__name__
        time.sleep(0.1)
    return {"ok": False, "error": last_error or "health_timeout"}


def terminate_process(proc: subprocess.Popen[str] | None, *, timeout: float = 5.0) -> dict[str, Any]:
    if proc is None:
        return {"terminated": False, "already_exited": True}
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout)
            return {"terminated": True, "killed": True, "returncode": proc.returncode}
        return {"terminated": True, "killed": False, "returncode": proc.returncode}
    return {"terminated": False, "already_exited": True, "returncode": proc.returncode}


def process_tail(proc: subprocess.Popen[str] | None, *, secret_values: list[str] | None = None) -> dict[str, Any]:
    if proc is None:
        return {}
    stdout = ""
    stderr = ""
    try:
        out, err = proc.communicate(timeout=1.0)
        stdout = out or ""
        stderr = err or ""
    except subprocess.TimeoutExpired:
        pass
    return {
        "returncode": proc.returncode,
        "stdout_tail": redact_text(stdout[-1200:], secret_values),
        "stderr_tail": redact_text(stderr[-1200:], secret_values),
    }


def command_json_path(output_dir: Path, name: str) -> Path:
    return output_dir / f"{name}.json"


def int_seconds(value: float | int | str) -> str:
    return str(max(1, int(float(value))))


def product_beta_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "public_swarm_inference_beta_pack.py"),
        "product-beta",
        "--output-dir",
        str(output_dir),
        "--base-port",
        str(args.base_port + 20),
        "--hf-model-id",
        args.hf_model_id,
        "--gpu-report",
        args.gpu_report,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--cpu-request-count",
        str(args.cpu_request_count),
        "--external-llm-request-count",
        str(args.external_llm_request_count),
        "--scenario-id",
        args.scenario_id,
        "--cpu-timeout-seconds",
        str(args.cpu_timeout_seconds),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--json",
    ]
    if args.prompt_texts:
        command.extend(["--prompt-texts", args.prompt_texts])
    else:
        command.extend(["--prompt-text", args.prompt_text])
    return command


def run_product_beta(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    return run_json_step(
        "public_swarm_product_beta",
        product_beta_command(args, output_dir),
        runner=runner,
        timeout_seconds=float(args.timeout_seconds) + float(args.cpu_timeout_seconds) + 240.0,
    )


def safe_batch_summary(payload: dict[str, Any]) -> dict[str, Any]:
    batch = payload.get("batch") if isinstance(payload.get("batch"), dict) else {}
    if not batch:
        return {"enabled": False, "batch_generation_ready": False}
    raw_results = batch.get("results") if isinstance(batch.get("results"), list) else []
    safe_results: list[dict[str, Any]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        safe_results.append({
            "request_id": item.get("request_id"),
            "prompt_hash": item.get("prompt_hash"),
            "generated_token_count": safe_int(item.get("generated_token_count")),
            "max_new_tokens": item.get("max_new_tokens"),
            "generated_text_hash": item.get("generated_text_hash"),
            "decoded_tokens_match": item.get("decoded_tokens_match"),
            "multi_token_generation_ready": bool(item.get("multi_token_generation_ready")),
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        })
    result_identity_keys = [
        str(item.get("request_id") or item.get("prompt_hash") or "")
        for item in safe_results
    ]
    expected_request_count = safe_int(batch.get("expected_request_count") or batch.get("request_count"))
    batch_identity_ready = bool(
        expected_request_count > 0
        and (
            expected_request_count <= 1
            or (
                len(result_identity_keys) >= expected_request_count
                and all(result_identity_keys[:expected_request_count])
                and len(set(result_identity_keys[:expected_request_count])) == expected_request_count
            )
        )
    )
    return {
        "enabled": bool(batch.get("enabled")),
        "request_count": int(batch.get("request_count") or 0),
        "expected_request_count": expected_request_count,
        "observed_request_count": safe_int(batch.get("observed_request_count") or batch.get("request_count")),
        "prompt_hashes": list(batch.get("prompt_hashes") or []),
        "prompt_char_counts": list(batch.get("prompt_char_counts") or []),
        "result_count": safe_int(batch.get("result_count") or len(safe_results)),
        "results": safe_results,
        "batch_identity_ready": batch_identity_ready,
        "batch_generation_ready": bool(batch.get("batch_generation_ready") and batch_identity_ready),
        "raw_prompts_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }


def safe_stream_summary(payload: dict[str, Any]) -> dict[str, Any]:
    stream = payload.get("stream") if isinstance(payload.get("stream"), dict) else {}
    if not stream:
        return {"enabled": False, "stream_generation_ready": False}
    progress = stream.get("progress") if isinstance(stream.get("progress"), dict) else {}
    events = stream.get("events") if isinstance(stream.get("events"), list) else []
    safe_events: list[dict[str, Any]] = []
    for item in events:
        if not isinstance(item, dict):
            continue
        safe_events.append({
            "schema": item.get("schema"),
            "session_id": item.get("session_id"),
            "task_id": item.get("task_id"),
            "miner_id": item.get("miner_id"),
            "stage_id": item.get("stage_id"),
            "request_id": item.get("request_id"),
            "prompt_hash": item.get("prompt_hash"),
            "generated_token_count": int(item.get("generated_token_count") or 0),
            "max_new_tokens": item.get("max_new_tokens"),
            "generation_step": item.get("generation_step"),
            "generated_text_hash": item.get("generated_text_hash"),
            "decoded_tokens_match": item.get("decoded_tokens_match"),
            "observed_at": item.get("observed_at"),
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        })
    batch = payload.get("batch") if isinstance(payload.get("batch"), dict) else {}
    generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else {}
    expected_request_count = max(
        1,
        safe_int(batch.get("expected_request_count") or batch.get("request_count") or generation.get("request_count") or progress.get("expected_request_count"), 1),
    )
    per_request_progress: list[dict[str, Any]] = []
    raw_per_request = progress.get("per_request_progress") if isinstance(progress.get("per_request_progress"), list) else []
    for item in raw_per_request:
        if not isinstance(item, dict):
            continue
        counts = item.get("observed_token_counts") if isinstance(item.get("observed_token_counts"), list) else []
        per_request_progress.append({
            "request_key": item.get("request_key"),
            "request_id": item.get("request_id"),
            "prompt_hash": item.get("prompt_hash"),
            "event_count": safe_int(item.get("event_count")),
            "observed_token_counts": [safe_int(value) for value in counts if safe_int(value, -1) >= 0],
            "max_observed_token_count": safe_int(item.get("max_observed_token_count")),
            "target_token_count": safe_int(item.get("target_token_count") or progress.get("max_new_tokens")),
            "monotonic_progress": bool(item.get("monotonic_progress")),
            "stream_progress_complete": bool(item.get("stream_progress_complete")),
        })
    stream_progress_complete = bool(progress.get("stream_progress_complete"))
    monotonic_progress = bool(progress.get("monotonic_progress"))
    per_request_complete = bool(progress.get("per_request_progress_complete"))
    per_request_monotonic = bool(progress.get("per_request_monotonic_progress"))
    stream_generation_ready = bool(stream.get("stream_generation_ready"))
    if expected_request_count > 1:
        stream_generation_ready = bool(
            stream_generation_ready
            and per_request_progress
            and per_request_complete
            and per_request_monotonic
        )
    elif not monotonic_progress:
        stream_generation_ready = False
    return {
        "enabled": bool(stream.get("enabled")),
        "requested": bool(stream.get("requested") or stream.get("enabled")),
        "event_count": int(stream.get("event_count") or len(safe_events)),
        "source": stream.get("source"),
        "endpoint_ready": bool(stream.get("endpoint_ready")),
        "progress": {
            "stream_progress_complete": stream_progress_complete,
            "all_token_events_ready": bool(progress.get("all_token_events_ready")),
            "monotonic_progress": monotonic_progress,
            "expected_request_count": expected_request_count,
            "per_request_progress": per_request_progress,
            "per_request_progress_complete": per_request_complete,
            "per_request_monotonic_progress": per_request_monotonic,
            "observed_token_counts": list(progress.get("observed_token_counts") or []),
            "max_observed_token_count": int(progress.get("max_observed_token_count") or 0),
            "max_new_tokens": progress.get("max_new_tokens"),
            "source": progress.get("source") or stream.get("source") or "",
        },
        "events": safe_events,
        "stream_generation_ready": stream_generation_ready,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }


def stream_evidence_ready(stream: dict[str, Any], batch: dict[str, Any] | None = None) -> bool:
    if not isinstance(stream, dict) or stream.get("stream_generation_ready") is not True:
        return False
    progress = stream.get("progress") if isinstance(stream.get("progress"), dict) else {}
    expected_request_count = max(
        1,
        safe_int(
            progress.get("expected_request_count")
            or (batch or {}).get("expected_request_count")
            or (batch or {}).get("request_count"),
            1,
        ),
    )
    if expected_request_count > 1 or bool((batch or {}).get("enabled")):
        return bool(
            progress.get("per_request_progress")
            and progress.get("per_request_progress_complete") is True
            and progress.get("per_request_monotonic_progress") is True
        )
    return bool(progress.get("stream_progress_complete") is True and progress.get("monotonic_progress") is True)


def run_p2p_route(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    del output_dir
    return run_json_step(
        "p2p_lite_discovery_check",
        [sys.executable, str(ROOT / "scripts" / "p2p_lite_discovery_check.py"), "--json"],
        runner=runner,
        timeout_seconds=max(60.0, float(args.timeout_seconds)),
    )


def run_cpu_fallback(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "cpu_inference_beta_pack.py"),
        "--mode",
        "local",
        "--output-dir",
        str(output_dir),
        "--base-port",
        str(args.base_port + 30),
        "--request-count",
        str(args.cpu_request_count),
        "--external-llm-request-count",
        str(args.external_llm_request_count),
        "--scenario-id",
        args.scenario_id,
        "--timeout-seconds",
        int_seconds(args.cpu_timeout_seconds),
        "--json",
    ]
    return run_json_step(
        "cpu_inference_beta_fallback",
        command,
        runner=runner,
        timeout_seconds=float(args.cpu_timeout_seconds) + 120.0,
    )


def run_remote_real_loopback(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "remote_real_llm_sharded_beta_pack.py"),
        "--mode",
        "remote-loopback",
        "--output-dir",
        str(output_dir),
        "--base-port",
        str(args.base_port + 40),
        "--request-count",
        "1",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--failure-mode",
        "none",
        "--stage-mode",
        "split",
        "--require-distinct-stage-miners",
        "--hf-model-id",
        args.hf_model_id,
        "--timeout-seconds",
        str(max(float(args.timeout_seconds), 240.0)),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    return run_json_step(
        "remote_real_llm_sharded_loopback",
        command,
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), 240.0) + 180.0,
    )


def start_coordinator_command(args: argparse.Namespace, *, state_dir: Path, admin_token: str, observer_token: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "coordinator.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.base_port),
        "--state-dir",
        str(state_dir),
        "--backlog",
        "0",
        "--task-lane",
        "python-cli:cpu:0:real_llm_sharded_infer",
        "--admin-token",
        admin_token,
        "--observer-token",
        observer_token,
        "--real-llm-model-id",
        args.hf_model_id,
        "--real-llm-backend",
        "hf_transformers_cpu",
        "--real-llm-partition-mode",
        "stage-local",
    ]


def join_command(
    args: argparse.Namespace,
    *,
    coordinator_url: str,
    stage: str,
    max_tasks: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "join",
        "--coordinator-url",
        coordinator_url,
        "--miner-id",
        f"{args.miner_id_prefix}-{stage}",
        "--stage",
        stage,
        "--backend",
        "cpu",
        "--hf-model-id",
        args.hf_model_id,
        "--max-tasks",
        str(max_tasks),
        "--run",
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    return command


def run_product_generate_loop(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    runtime_dir = output_dir / "serve-join-generate"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    secret_values = [
        args.admin_token or DEFAULT_ADMIN_TOKEN,
        args.observer_token or DEFAULT_OBSERVER_TOKEN,
    ]
    admin_token = secret_values[0]
    observer_token = secret_values[1]
    coordinator_url = f"http://127.0.0.1:{args.base_port}"
    state_tmp = tempfile.TemporaryDirectory(prefix="crowdtensor_public_swarm_beta_rc_state_")
    coordinator: subprocess.Popen[str] | None = None
    stage0: subprocess.Popen[str] | None = None
    stage1: subprocess.Popen[str] | None = None
    started = time.monotonic()
    missing_hf = missing_hf_dependencies()
    if missing_hf:
        return {
            "ok": False,
            "step": {
                "name": "serve_join_generate_loop",
                "ok": False,
                "duration_seconds": round(time.monotonic() - started, 3),
                "error": "hf_dependencies_missing",
            },
            "diagnosis_codes": ["hf_dependencies_missing"],
            "operator_action": "Install optional runtime dependencies with: python -m pip install -e '.[hf]'",
            "runtime": {
                "backend": "hf_transformers_cpu",
                "missing_dependencies": missing_hf,
            },
        }
    try:
        coordinator_command = start_coordinator_command(
            args,
            state_dir=Path(state_tmp.name),
            admin_token=admin_token,
            observer_token=observer_token,
        )
        coordinator = subprocess.Popen(
            coordinator_command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        health = wait_health(
            coordinator_url,
            coordinator,
            timeout_seconds=args.startup_timeout,
            http_timeout=args.http_timeout,
        )
        if health.get("ok") is not True:
            return {
                "ok": False,
                "step": {
                    "name": "serve_join_generate_loop",
                    "ok": False,
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "error": health.get("error"),
                },
                "diagnosis_codes": ["coordinator_unreachable"],
                "processes": {"coordinator": process_tail(coordinator, secret_values=secret_values)},
            }
        max_tasks = max(1, int(args.max_new_tokens))
        stage0 = subprocess.Popen(
            join_command(args, coordinator_url=coordinator_url, stage="stage0", max_tasks=max_tasks),
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        stage1 = subprocess.Popen(
            join_command(args, coordinator_url=coordinator_url, stage="stage1", max_tasks=max_tasks),
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        generate_command = [
            sys.executable,
            "-m",
            "crowdtensor.cli",
            "generate",
            "--coordinator-url",
            coordinator_url,
            "--scenario-id",
            "public-swarm-beta-rc",
            "--backend",
            "cpu",
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--admin-token",
            admin_token,
            "--timeout-seconds",
            str(args.remote_timeout_seconds),
            "--poll-interval",
            str(args.poll_interval),
            "--http-timeout",
            str(args.http_timeout),
            "--json",
        ]
        if args.prompt_texts:
            generate_command.extend(["--prompt-texts", args.prompt_texts])
        else:
            generate_command.extend(["--prompt-text", args.prompt_text])
        if args.stream_generation:
            generate_command.append("--stream")
        completed = subprocess.run(
            generate_command,
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(args.remote_timeout_seconds) + 60.0,
        )
        try:
            generate_payload = json_from_stdout(completed.stdout)
        except ValueError:
            generate_payload = {}
        write_json(command_json_path(runtime_dir, "generate"), redact_values(generate_payload, secret_values))
        miner_timeout = max(10.0, float(args.process_exit_timeout))
        for proc in [stage0, stage1]:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=miner_timeout)
                except subprocess.TimeoutExpired:
                    terminate_process(proc, timeout=2.0)
        batch = safe_batch_summary(generate_payload)
        stream = safe_stream_summary(generate_payload)
        batch_ready = bool(not batch.get("enabled") or batch.get("batch_generation_ready"))
        stream_ready = bool(not args.stream_generation or stream_evidence_ready(stream, batch))
        codes = drop_unproven_stream_codes(set(diagnosis_codes(generate_payload)), stream_ready=stream_ready)
        generate_ready = bool(
            completed.returncode == 0
            and generate_payload.get("schema") == PRODUCT_CLI_SCHEMA
            and generate_payload.get("ok") is True
            and "public_swarm_generate_ready" in codes
            and batch_ready
            and stream_ready
        )
        stage0_clean = stage0.poll() == 0 if stage0 else False
        stage1_clean = stage1.poll() == 0 if stage1 else False
        if generate_ready and stage0_clean and stage1_clean:
            codes.update({
                "serve_join_generate_loop_ready",
                "remote_generate_session_ready",
                "public_swarm_generate_ready",
            })
            if batch.get("enabled"):
                codes.add("public_swarm_generate_batch_ready")
            if stream_ready and args.stream_generation:
                codes.add("public_swarm_generate_stream_ready")
                if stream.get("endpoint_ready"):
                    codes.add("public_swarm_generate_stream_endpoint_ready")
        else:
            if not generate_ready:
                codes.add("generation_timeout")
            if not stage0_clean or not stage1_clean:
                codes.add("stage_miner_missing")
        return {
            "ok": bool(generate_ready and stage0_clean and stage1_clean),
            "step": {
                "name": "serve_join_generate_loop",
                "ok": bool(generate_ready and stage0_clean and stage1_clean),
                "returncode": completed.returncode,
                "duration_seconds": round(time.monotonic() - started, 3),
                "payload_schema": generate_payload.get("schema"),
                "payload_ok": generate_payload.get("ok"),
                "stderr_tail": redact_text(completed.stderr[-1200:], secret_values) if completed.stderr else "",
            },
            "coordinator_url": coordinator_url,
            "generation": {
                "ok": generate_payload.get("ok"),
                "schema": generate_payload.get("schema"),
                "session": generate_payload.get("session") if isinstance(generate_payload.get("session"), dict) else {},
                "generation": generate_payload.get("generation") if isinstance(generate_payload.get("generation"), dict) else {},
                "batch": batch,
                "stream": stream,
                "route": generate_payload.get("route") if isinstance(generate_payload.get("route"), dict) else {},
            },
            "diagnosis_codes": sorted(codes),
            "processes": {
                "stage0": process_tail(stage0, secret_values=secret_values),
                "stage1": process_tail(stage1, secret_values=secret_values),
            },
            "artifacts": {
                "generate_json": artifact_entry(
                    command_json_path(runtime_dir, "generate"),
                    output_dir,
                    kind="public_swarm_generate",
                    schema=PRODUCT_CLI_SCHEMA,
                    ok=generate_payload.get("ok") if generate_payload else None,
                )
            },
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "step": {
                "name": "serve_join_generate_loop",
                "ok": False,
                "duration_seconds": round(time.monotonic() - started, 3),
                "error": "timeout",
            },
            "diagnosis_codes": ["remote_generate_session_timeout"],
            "processes": {
                "stage0": process_tail(stage0, secret_values=secret_values),
                "stage1": process_tail(stage1, secret_values=secret_values),
            },
        }
    finally:
        terminate_process(stage0, timeout=2.0)
        terminate_process(stage1, timeout=2.0)
        terminate_process(coordinator, timeout=2.0)
        state_tmp.cleanup()


def remote_real_existing_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "remote_real_llm_sharded_beta_pack.py"),
        "--mode",
        "remote-existing",
        "--output-dir",
        str(output_dir),
        "--coordinator-url",
        args.coordinator_url,
        "--observer-token",
        args.observer_token,
        "--admin-token",
        args.admin_token,
        "--request-count",
        "1",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--stage-mode",
        "split",
        "--require-distinct-stage-miners",
        "--hf-model-id",
        args.hf_model_id,
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--http-timeout",
        str(args.http_timeout),
        "--timeout-seconds",
        str(max(float(args.timeout_seconds), 240.0)),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    return command


def generate_existing_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    del output_dir
    command = [
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "generate",
        "--coordinator-url",
        args.coordinator_url,
        "--scenario-id",
        "public-swarm-beta-rc",
        "--backend",
        "cpu",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--admin-token",
        args.admin_token,
        "--timeout-seconds",
        str(args.remote_timeout_seconds),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--json",
    ]
    if args.prompt_texts:
        command.extend(["--prompt-texts", args.prompt_texts])
    else:
        command.extend(["--prompt-text", args.prompt_text])
    if args.stream_generation:
        command.append("--stream")
    return command


def run_existing_generate(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    return run_json_step(
        "crowdtensor_generate_existing",
        generate_existing_command(args, output_dir),
        runner=runner,
        timeout_seconds=float(args.remote_timeout_seconds) + 60.0,
        secret_values=[args.admin_token],
    )


def run_existing_remote_evidence(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    return run_json_step(
        "remote_real_llm_sharded_existing",
        remote_real_existing_command(args, output_dir),
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), 240.0) + 180.0,
        secret_values=[args.admin_token, args.observer_token],
    )


def build_common_report(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    product_step: dict[str, Any],
    product_payload: dict[str, Any],
    p2p_step: dict[str, Any],
    p2p_payload: dict[str, Any],
    cpu_step: dict[str, Any],
    cpu_payload: dict[str, Any],
    mode_body: dict[str, Any],
    mode_steps: list[dict[str, Any]],
    mode_payloads: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    product_codes = set(diagnosis_codes(product_payload))
    p2p_codes = set(diagnosis_codes(p2p_payload))
    cpu_codes = set(diagnosis_codes(cpu_payload))
    body_codes = set(mode_body.get("diagnosis_codes") or [])
    mode_batch = {}
    mode_stream = {}
    if isinstance(mode_body.get("generation"), dict):
        generation_body = mode_body.get("generation") or {}
        mode_batch = safe_batch_summary({"batch": generation_body.get("batch")})
        mode_stream = safe_stream_summary({
            "stream": generation_body.get("stream"),
            "batch": mode_batch,
            "generation": generation_body.get("generation") if isinstance(generation_body.get("generation"), dict) else {},
        })
    batch_requested = bool(mode_batch.get("enabled"))
    stream_requested = bool(mode_stream.get("requested") or mode_stream.get("enabled"))
    batch_ready = bool(mode_batch.get("enabled") and mode_batch.get("batch_generation_ready") is True)
    stream_ready = stream_evidence_ready(mode_stream, mode_batch)
    codes = drop_unproven_stream_codes(set().union(product_codes, p2p_codes, cpu_codes, body_codes), stream_ready=stream_ready)
    product_ready = bool(
        product_step.get("ok")
        and product_payload.get("schema") == BETA_SCHEMA
        and "public_swarm_product_beta_ready" in product_codes
    )
    p2p_ready = bool(p2p_step.get("ok") and p2p_payload.get("schema") == P2P_CHECK_SCHEMA and "p2p_lite_discovery_ready" in p2p_codes)
    cpu_ready = bool(cpu_step.get("ok") and cpu_payload.get("schema") == CPU_BETA_SCHEMA and "cpu_inference_beta_ready" in cpu_codes)
    mode_ready = bool(
        mode_body.get("ok")
        and (not batch_requested or batch_ready)
        and (not stream_requested or stream_ready)
    )
    if product_ready:
        codes.add("public_swarm_product_beta_ready")
    else:
        codes.add("public_swarm_product_beta_blocked")
    if p2p_ready:
        codes.add("p2p_lite_route_ready")
    else:
        codes.add("p2p_lite_route_blocked")
    if cpu_ready:
        codes.update({"cpu_fallback_ready", "local_cpu_inference_ready"})
    else:
        codes.add("cpu_fallback_blocked")
    ready = bool(product_ready and p2p_ready and cpu_ready and mode_ready)
    if ready:
        codes.update({
            "public_swarm_inference_beta_rc_ready",
            "read_only_workload",
            "not_production",
        })
        if batch_ready:
            codes.add("public_swarm_generate_batch_ready")
        if stream_ready:
            codes.add("public_swarm_generate_stream_ready")
            if mode_stream.get("endpoint_ready"):
                codes.add("public_swarm_generate_stream_endpoint_ready")
    else:
        codes.add("public_swarm_inference_beta_rc_blocked")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "rc": {
            "ready": ready,
            "product_beta_ready": product_ready,
            "p2p_lite_route_ready": p2p_ready,
            "cpu_fallback_ready": cpu_ready,
            "mode_ready": mode_ready,
            "batch_requested": batch_requested,
            "stream_requested": stream_requested,
            "workload_type": WORKLOAD_TYPE,
            "hf_model_id": args.hf_model_id,
            "max_new_tokens": args.max_new_tokens,
            "batch": mode_batch if mode_batch else {"enabled": False, "batch_generation_ready": False},
            "stream": mode_stream if mode_stream else {"enabled": False, "stream_generation_ready": False},
        },
        "steps": [product_step, p2p_step, cpu_step] + mode_steps,
        "payload_summaries": {
            "product_beta": beta_pack.product_rc_summary(
                (product_payload.get("payload_summaries") or {}).get("public_swarm_product_rc", {})
                if isinstance(product_payload.get("payload_summaries"), dict)
                else {}
            ),
            "public_swarm_product_beta": {
                "schema": product_payload.get("schema"),
                "ok": product_payload.get("ok"),
                "mode": product_payload.get("mode"),
                "diagnosis_codes": diagnosis_codes(product_payload),
            },
            "p2p_lite": {
                "schema": p2p_payload.get("schema"),
                "ok": p2p_payload.get("ok"),
                "diagnosis_codes": diagnosis_codes(p2p_payload),
            },
            "cpu_fallback": beta_pack.cpu_beta_summary(cpu_payload),
            **(mode_payloads or {}),
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": {
            "public_swarm_product_beta_json": artifact_entry(
                output_dir / "product-beta" / "public_swarm_inference_beta.json",
                output_dir,
                kind="public_swarm_inference_beta",
                schema=BETA_SCHEMA,
                ok=product_payload.get("ok") if product_payload else None,
            ),
            "cpu_inference_beta_json": artifact_entry(
                output_dir / "cpu-fallback" / "cpu_inference_beta.json",
                output_dir,
                kind="cpu_inference_beta",
                schema=CPU_BETA_SCHEMA,
                ok=cpu_payload.get("ok") if cpu_payload else None,
            ),
            **(mode_body.get("artifacts") if isinstance(mode_body.get("artifacts"), dict) else {}),
        },
        "safety": {
            "coordinator_backed_task_execution": True,
            "serve_join_generate_product_loop": args.mode == "local-loopback",
            "p2p_lite_discovery_only": True,
            "tokens_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_payloads_redacted": True,
            "read_only_workload": WORKLOAD_TYPE,
            "not_production": True,
            "not_libp2p": True,
            "not_dht": True,
            "not_nat_traversal": True,
            "not_gpu_pooling_marketplace": True,
            "not_large_model_serving": True,
            "not_public_prompt_serving": True,
        },
        "limitations": [
            "Public Swarm Inference Beta RC is Coordinator-backed and read-only; it is not production Swarm Inference.",
            "P2P-lite route evidence is HTTP-gossip discovery only; not libp2p, DHT, NAT traversal, or decentralized task execution.",
            "The default proof uses tiny GPT / CPU fallback paths and safe evidence summaries; not Hivemind-level serving or large-model public prompt serving.",
        ],
    }
    return persist_report(report, output_dir=output_dir, secret_values=[args.admin_token, args.observer_token])


def render_markdown(report: dict[str, Any]) -> str:
    rc = report.get("rc") if isinstance(report.get("rc"), dict) else {}
    lines = [
        "# CrowdTensor Public Swarm Inference Beta RC",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- output_dir: `{report.get('output_dir')}`",
        f"- ready: `{rc.get('ready')}`",
        f"- max_new_tokens: `{rc.get('max_new_tokens')}`",
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`",
        "",
        "## Steps",
        "",
    ]
    for step in report.get("steps") or []:
        lines.append(f"- `{step.get('name')}`: `{step.get('ok')}` schema=`{step.get('payload_schema')}`")
    lines.extend(["", "## Boundaries", ""])
    for item in report.get("limitations") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def persist_report(report: dict[str, Any], *, output_dir: Path, secret_values: list[str] | None = None) -> dict[str, Any]:
    report = support_bundle.sanitize(redact_values(report, secret_values))
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"sensitive_output_detected"})
        report["safety_error"] = "Public Swarm Inference Beta RC report contained secret-like fragments"
    json_path = output_dir / "public_swarm_inference_beta_rc.json"
    markdown_path = output_dir / "public_swarm_inference_beta_rc.md"
    report.setdefault("artifacts", {})
    report["artifacts"]["public_swarm_inference_beta_rc_json"] = artifact_entry(
        json_path,
        output_dir,
        kind="public_swarm_inference_beta_rc",
        schema=SCHEMA,
        ok=report.get("ok"),
    )
    report["artifacts"]["public_swarm_inference_beta_rc_markdown"] = artifact_entry(
        markdown_path,
        output_dir,
        kind="public_swarm_inference_beta_rc_markdown",
    )
    write_json(json_path, report)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    report["artifacts"]["public_swarm_inference_beta_rc_json"]["present"] = True
    report["artifacts"]["public_swarm_inference_beta_rc_markdown"]["present"] = True
    write_json(json_path, report)
    return report


def build_local_loopback(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    product_step, product_payload = run_product_beta(args, output_dir=output_dir / "product-beta", runner=runner)
    p2p_step, p2p_payload = run_p2p_route(args, output_dir=output_dir / "p2p-lite", runner=runner)
    cpu_step, cpu_payload = run_cpu_fallback(args, output_dir=output_dir / "cpu-fallback", runner=runner)
    product_loop = run_product_generate_loop(args, output_dir=output_dir)
    mode_payloads = {
        "serve_join_generate": {
            "ok": product_loop.get("ok"),
            "coordinator_url": product_loop.get("coordinator_url"),
            "diagnosis_codes": product_loop.get("diagnosis_codes") or [],
            "generation": (product_loop.get("generation") or {}).get("generation"),
            "batch": (product_loop.get("generation") or {}).get("batch"),
            "stream": (product_loop.get("generation") or {}).get("stream"),
        }
    }
    return build_common_report(
        args,
        output_dir=output_dir,
        product_step=product_step,
        product_payload=product_payload,
        p2p_step=p2p_step,
        p2p_payload=p2p_payload,
        cpu_step=cpu_step,
        cpu_payload=cpu_payload,
        mode_body=product_loop,
        mode_steps=[product_loop.get("step") or {"name": "serve_join_generate_loop", "ok": False}],
        mode_payloads=mode_payloads,
    )


def build_package(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    product_step, product_payload = run_product_beta(args, output_dir=output_dir / "product-beta", runner=runner)
    p2p_step, p2p_payload = run_p2p_route(args, output_dir=output_dir / "p2p-lite", runner=runner)
    cpu_step, cpu_payload = run_cpu_fallback(args, output_dir=output_dir / "cpu-fallback", runner=runner)
    package_dir = output_dir / "operator-package"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "public_swarm_inference_beta_pack.py"),
        "prepare",
        "--output-dir",
        str(package_dir),
        "--coordinator-url",
        args.coordinator_url or f"http://{args.public_host}:{args.port}",
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--bind-host",
        args.bind_host,
        "--miner-id-prefix",
        args.miner_id_prefix,
        "--hf-model-id",
        args.hf_model_id,
        "--request-count",
        "1",
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    package_step, package_payload = run_json_step(
        "public_swarm_beta_package",
        command,
        runner=runner,
        timeout_seconds=float(args.timeout_seconds) + 180.0,
        secret_values=[args.admin_token, args.observer_token],
    )
    package_codes = set(diagnosis_codes(package_payload))
    package_ready = bool(
        package_step.get("ok")
        and package_payload.get("schema") == BETA_SCHEMA
        and "public_swarm_beta_prepare_ready" in package_codes
    )
    codes = set(package_codes)
    if package_ready:
        codes.update({"public_swarm_beta_rc_package_ready", "miner_join_pack_ready", "private_artifacts_local_only"})
        if args.target == "kaggle":
            codes.add("kaggle_remote_miner_package_ready")
    else:
        codes.add("public_swarm_beta_rc_package_blocked")
    mode_body = {
        "ok": package_ready,
        "target": args.target,
        "diagnosis_codes": sorted(codes),
        "artifacts": {
            "operator_package_json": artifact_entry(
                package_dir / "public_swarm_inference_beta_prepare.json",
                output_dir,
                kind="public_swarm_beta_prepare",
                schema=BETA_SCHEMA,
                ok=package_payload.get("ok") if package_payload else None,
            )
        },
    }
    return build_common_report(
        args,
        output_dir=output_dir,
        product_step=product_step,
        product_payload=product_payload,
        p2p_step=p2p_step,
        p2p_payload=p2p_payload,
        cpu_step=cpu_step,
        cpu_payload=cpu_payload,
        mode_body=mode_body,
        mode_steps=[package_step],
        mode_payloads={
            "package": {
                "target": args.target,
                "schema": package_payload.get("schema"),
                "ok": package_payload.get("ok"),
                "diagnosis_codes": diagnosis_codes(package_payload),
            }
        },
    )


def build_external_existing(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    product_step, product_payload = run_product_beta(args, output_dir=output_dir / "product-beta", runner=runner)
    p2p_step, p2p_payload = run_p2p_route(args, output_dir=output_dir / "p2p-lite", runner=runner)
    cpu_step, cpu_payload = run_cpu_fallback(args, output_dir=output_dir / "cpu-fallback", runner=runner)
    generate_step, generate_payload = run_existing_generate(args, output_dir=output_dir / "external-generate", runner=runner)
    remote_step, remote_payload = run_existing_remote_evidence(args, output_dir=output_dir / "external-remote-real", runner=runner)
    remote_codes = set(diagnosis_codes(remote_payload))
    generate_batch = safe_batch_summary(generate_payload)
    generate_stream = safe_stream_summary(generate_payload)
    generate_batch_ready = bool(not generate_batch.get("enabled") or generate_batch.get("batch_generation_ready"))
    generate_stream_ready = bool(not args.stream_generation or stream_evidence_ready(generate_stream, generate_batch))
    generate_codes = drop_unproven_stream_codes(set(diagnosis_codes(generate_payload)), stream_ready=generate_stream_ready)
    generate_ready = bool(
        generate_step.get("ok")
        and generate_payload.get("schema") == PRODUCT_CLI_SCHEMA
        and "public_swarm_generate_ready" in generate_codes
        and generate_batch_ready
        and generate_stream_ready
    )
    remote_ready = bool(
        remote_step.get("ok")
        and remote_payload.get("schema") == REMOTE_REAL_SCHEMA
        and "remote_real_llm_sharded_existing_ready" in remote_codes
    )
    codes = drop_unproven_stream_codes(set(generate_codes) | set(remote_codes), stream_ready=generate_stream_ready)
    if generate_ready:
        codes.update({"remote_generate_session_ready", "serve_join_generate_loop_ready"})
        if generate_batch.get("enabled"):
            codes.add("public_swarm_generate_batch_ready")
        if generate_stream_ready and args.stream_generation:
            codes.add("public_swarm_generate_stream_ready")
            if generate_stream.get("endpoint_ready"):
                codes.add("public_swarm_generate_stream_endpoint_ready")
    else:
        codes.add("remote_generate_session_blocked")
    if remote_ready:
        codes.add("external_runtime_verified")
    else:
        codes.add("external_runtime_blocked")
    mode_body = {
        "ok": bool(generate_ready and remote_ready),
        "coordinator_url": args.coordinator_url,
        "generation": {"batch": generate_batch, "stream": generate_stream},
        "diagnosis_codes": sorted(codes),
        "artifacts": {
            "remote_real_llm_sharded_beta_json": artifact_entry(
                output_dir / "external-remote-real" / "remote_real_llm_sharded_beta.json",
                output_dir,
                kind="remote_real_llm_sharded_beta",
                schema=REMOTE_REAL_SCHEMA,
                ok=remote_payload.get("ok") if remote_payload else None,
            )
        },
    }
    return build_common_report(
        args,
        output_dir=output_dir,
        product_step=product_step,
        product_payload=product_payload,
        p2p_step=p2p_step,
        p2p_payload=p2p_payload,
        cpu_step=cpu_step,
        cpu_payload=cpu_payload,
        mode_body=mode_body,
        mode_steps=[generate_step, remote_step],
        mode_payloads={
            "external_generate": {
                "schema": generate_payload.get("schema"),
                "ok": generate_payload.get("ok"),
                "diagnosis_codes": diagnosis_codes(generate_payload),
                "generation": generate_payload.get("generation") if isinstance(generate_payload.get("generation"), dict) else {},
                "batch": generate_batch,
                "stream": generate_stream,
            },
            "remote_existing_real_llm": {
                "schema": remote_payload.get("schema"),
                "ok": remote_payload.get("ok"),
                "diagnosis_codes": diagnosis_codes(remote_payload),
            },
        },
    )


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "local-loopback":
        return build_local_loopback(args, output_dir=output_dir, runner=runner)
    if args.mode == "package":
        return build_package(args, output_dir=output_dir, runner=runner)
    if args.mode == "external-existing":
        return build_external_existing(args, output_dir=output_dir, runner=runner)
    raise SystemExit(f"unknown mode: {args.mode}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Public Swarm Inference Beta RC evidence.")
    parser.add_argument("mode", choices=["local-loopback", "package", "external-existing"])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-port", type=int, default=9310)
    parser.add_argument("--port", type=int, default=9310)
    parser.add_argument("--public-host", default="127.0.0.1")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--target", choices=["local", "kaggle"], default="local")
    parser.add_argument("--miner-id-prefix", default="public-swarm-beta-rc")
    parser.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--gpu-report", default=DEFAULT_GPU_REPORT)
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-texts", default="", help="comma-separated bounded batch of up to 4 prompts")
    parser.add_argument("--stream-generation", action="store_true")
    parser.add_argument("--scenario-id", default="route-baseline")
    parser.add_argument("--request-count", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--cpu-request-count", type=int, default=1)
    parser.add_argument("--external-llm-request-count", type=int, default=1)
    parser.add_argument("--admin-token", default="")
    parser.add_argument("--observer-token", default="")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--remote-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--cpu-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--process-exit-timeout", type=float, default=20.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--http-timeout", type=float, default=10.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.base_port < 1 or args.port < 1:
        raise SystemExit("--base-port and --port must be positive")
    if args.request_count < 1 or args.request_count > 4:
        raise SystemExit("--request-count must be between 1 and 4")
    if args.cpu_request_count < 1 or args.cpu_request_count > 4:
        raise SystemExit("--cpu-request-count must be between 1 and 4")
    if args.external_llm_request_count < 1 or args.external_llm_request_count > 4:
        raise SystemExit("--external-llm-request-count must be between 1 and 4")
    if args.max_new_tokens < 2 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 2 and 32")
    try:
        parse_prompt_texts_arg(args.prompt_text, args.prompt_texts)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for name in [
        "timeout_seconds",
        "remote_timeout_seconds",
        "cpu_timeout_seconds",
        "startup_timeout",
        "process_exit_timeout",
        "poll_interval",
        "http_timeout",
    ]:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.mode == "external-existing":
        missing = [
            name
            for name in ["coordinator_url", "observer_token", "admin_token"]
            if not getattr(args, name)
        ]
        if missing:
            raise SystemExit(f"external-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
    return args


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"Public Swarm Inference Beta RC ready: {report.get('ok')}")
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
