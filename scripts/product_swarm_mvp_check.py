#!/usr/bin/env python3
"""Validate the v0.3 product serve/join/generate MVP path.

The local check intentionally uses the public CLI commands instead of calling
the lower-level evidence pack directly:

1. start `crowdtensor serve --run`
2. start `crowdtensor generate` so the Coordinator creates the session
3. run `crowdtensor join --stage stage0 --run` and stage1 repeatedly

When optional Hugging Face dependencies are unavailable, the script exits
successfully by default with a clear degraded report.  Use
`--require-hf-runtime` for release or maintainer machines that must prove the
real tiny-GPT execution path.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
try:
    import resource
except ImportError:  # pragma: no cover - Windows fallback for local developers.
    resource = None  # type: ignore[assignment]
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import support_bundle  # noqa: E402
from crowdtensor.real_llm import cuda_runtime_summary, missing_hf_dependencies  # noqa: E402
from crowdtensor.session_protocol import MAX_BATCH_REQUESTS, MAX_PROMPT_CHARS, public_leak_paths, stable_hash_text  # noqa: E402


SCHEMA = "product_swarm_mvp_check_v1"
ADMIN_TOKEN = "product-mvp-admin"
MINER_TOKEN = "product-mvp-miner"
OBSERVER_TOKEN = "product-mvp-observer"
WORKLOAD_TYPE = "real_llm_sharded_infer"
SECRET_FRAGMENTS = (
    ADMIN_TOKEN,
    MINER_TOKEN,
    OBSERVER_TOKEN,
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
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def output_request_summary() -> dict[str, Any]:
    return {
        "include_output": False,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Product Swarm MVP check artifacts summarize serve/join/generate "
            "readiness, safe generation hashes/counts, batch and stream progress, "
            "and runtime diagnostics only. They do not include answer text."
        ),
    }


def prompt_scope_summary(args: argparse.Namespace | None = None) -> dict[str, Any]:
    prompt_file = str(getattr(args, "prompt_file", "") or "")
    prompt_texts_file = str(getattr(args, "prompt_texts_file", "") or "")
    prompt_texts = str(getattr(args, "prompt_texts", "") or "")
    prompt_text = str(
        getattr(args, "prompt_text", "CrowdTensor turns spare compute into open inference")
        or "CrowdTensor turns spare compute into open inference"
    )
    parsed_prompts = getattr(args, "prompt_texts_list", None)
    if isinstance(parsed_prompts, list) and parsed_prompts:
        prompt_count = len(parsed_prompts)
    elif prompt_file:
        prompt_count = 1
    elif prompt_texts_file:
        prompt_count = 1
    else:
        try:
            prompt_count = len(parse_prompt_texts_arg(prompt_text, prompt_texts))
        except ValueError:
            prompt_count = 0
    if prompt_texts_file:
        source = "prompt-texts-file"
    elif prompt_file:
        source = "prompt-file"
    elif prompt_texts:
        source = "prompt-texts"
    else:
        source = "prompt-text"
    inline_prompt_text = source in {"prompt-text", "prompt-texts"}
    return {
        "source": source,
        "prompt_count": prompt_count,
        "inline_prompt_text": inline_prompt_text,
        "terminal_next_commands_local_private": inline_prompt_text,
        "terminal_logs_local_private": inline_prompt_text,
        "saved_artifacts_prompt_placeholders": True,
        "saved_artifacts_public_safe": True,
        "prefer_prompt_file_or_stdin_for_shareable_logs": True,
        "prompt_file_path_public": False,
        "raw_prompt_public": False,
        "public_artifact_safe": True,
        "summary": (
            "This Product Swarm MVP report records prompt source/count and "
            "placeholder safety only; raw prompt text and prompt file paths are "
            "excluded from public JSON."
        ),
    }


def answer_scope_summary() -> dict[str, Any]:
    return {
        "scope_state": "no-local-answer",
        "terminal_only": False,
        "visible_in_terminal": False,
        "saved_json_display": "hash-only",
        "saved_markdown_display": "none",
        "json_stdout_display": "hash-only-json",
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "public_artifact_safe": True,
        "summary": (
            "The Product Swarm MVP check is shareable readiness evidence, not an "
            "answer transcript. Raw prompts, generated text, generated token ids, "
            "activations, credentials, and runtime state are excluded."
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
            "Share product_swarm_mvp_check.json for MVP readiness, hashes, counts, "
            "and diagnostics only; do not treat it as local answer output."
        ),
    }


def ensure_output_scope(report: dict[str, Any], args: argparse.Namespace | None = None) -> dict[str, Any]:
    report.setdefault("output_request", output_request_summary())
    report.setdefault("prompt_scope", prompt_scope_summary(args))
    report.setdefault("answer_scope", answer_scope_summary())
    report.setdefault("shareable_summary", shareable_summary())
    return report


def cleanup_private_runtime_state(output_dir: Path, *, keep_private_state: bool = False) -> dict[str, Any]:
    state_dir = output_dir / "state"
    summary = {
        "state_dir": "state",
        "kept": bool(keep_private_state),
        "removed": False,
        "present_after_cleanup": state_dir.exists(),
        "raw_runtime_state_public": False,
    }
    if keep_private_state:
        return summary
    if not state_dir.exists():
        summary["present_after_cleanup"] = False
        return summary
    try:
        shutil.rmtree(state_dir)
    except OSError as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    summary["removed"] = not state_dir.exists()
    summary["present_after_cleanup"] = state_dir.exists()
    return summary


def request_json(
    method: str,
    base_url: str,
    path: str,
    *,
    admin_token: str = "",
    observer_token: str = "",
    timeout: float = 5.0,
) -> dict[str, Any]:
    headers = {"content-type": "application/json"}
    if admin_token:
        headers["x-crowdtensor-admin-token"] = admin_token
    if observer_token:
        headers["x-crowdtensor-observer-token"] = observer_token
    request = Request(f"{base_url.rstrip('/')}{path}", headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def wait_health(base_url: str, proc: subprocess.Popen[str], timeout: float) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=1)
            return False, f"serve exited early rc={proc.returncode} stdout={stdout[-500:]} stderr={stderr[-500:]}"
        try:
            if request_json("GET", base_url, "/health", timeout=2.0).get("ok") is True:
                return True, ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.2)
    return False, f"coordinator did not become healthy: {last_error}"


def popen_command(command: list[str], *, cwd: Path = ROOT) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


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
        "stdout_tail": (stdout or "")[-1000:],
        "stderr_tail": (stderr or "")[-1000:],
    }


def summarize_serve_process(serve_process: dict[str, Any], *, ok: bool) -> dict[str, Any]:
    if not isinstance(serve_process, dict):
        return {}
    stdout_tail = str(serve_process.get("stdout_tail") or "")
    stderr_tail = str(serve_process.get("stderr_tail") or "")
    if not ok:
        return support_bundle.sanitize(serve_process)
    return {
        "returncode": serve_process.get("returncode"),
        "stdout_captured": bool(stdout_tail),
        "stderr_captured": bool(stderr_tail),
        "stdout_tail_omitted": bool(stdout_tail),
        "stderr_tail_omitted": bool(stderr_tail),
        "captured_output_redacted": True,
        "runtime_log_public": False,
    }


def parse_json_payload(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def finish_process_step(name: str, proc: subprocess.Popen[str], *, timeout: float) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        stop_process(proc)
        return {
            "name": name,
            "ok": False,
            "returncode": proc.returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": "timeout",
        }, {}
    step = {
        "name": name,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    payload = parse_json_payload(stdout or "")
    if not payload:
        step["ok"] = False
        step["error"] = "json_payload_missing"
    if not step["ok"]:
        step["stdout_tail"] = (stdout or "")[-1000:]
        step["stderr_tail"] = (stderr or "")[-1000:]
    return step, payload


def run_step(name: str, command: list[str], *, timeout: float) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "ok": False,
            "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": "timeout",
            "stdout_tail": (exc.stdout or "")[-1000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-1000:] if isinstance(exc.stderr, str) else "",
        }, {}
    step = {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    payload = parse_json_payload(completed.stdout)
    if not payload:
        step["ok"] = False
        step["error"] = "json_payload_missing"
    if not step["ok"]:
        step["stdout_tail"] = completed.stdout[-1000:]
        step["stderr_tail"] = completed.stderr[-1000:]
    return step, payload


def stage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed_rows = [row for row in rows if row.get("status") == "completed"]
    miners_by_stage: dict[str, str] = {}
    max_generation_step = -1
    for row in completed_rows:
        stage = row.get("stage_id")
        if stage in {0, 1}:
            miners_by_stage[str(stage)] = str(row.get("miner_id") or "")
        validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
        try:
            max_generation_step = max(max_generation_step, int(validation.get("generation_step", -1)))
        except (TypeError, ValueError):
            pass
    return {
        "completed_rows": len(completed_rows),
        "stage0_miner_id": miners_by_stage.get("0", ""),
        "stage1_miner_id": miners_by_stage.get("1", ""),
        "distinct_stage_miners": bool(miners_by_stage.get("0") and miners_by_stage.get("1") and miners_by_stage.get("0") != miners_by_stage.get("1")),
        "max_generation_step": max_generation_step,
    }


def summarize_step_durations(steps: list[dict[str, Any]], *, generated_tokens: int, accepted_rows: int) -> dict[str, Any]:
    stage_durations: dict[str, list[float]] = {"stage0": [], "stage1": []}
    for step in steps:
        name = str(step.get("name") or "")
        duration = step.get("duration_seconds")
        try:
            seconds = float(duration)
        except (TypeError, ValueError):
            continue
        if seconds <= 0:
            continue
        if name.startswith("join_stage0_step_") or name.startswith("join_real_p2p_stage0_step_"):
            stage_durations["stage0"].append(seconds)
        elif name.startswith("join_stage1_step_") or name.startswith("join_real_p2p_stage1_step_"):
            stage_durations["stage1"].append(seconds)

    per_stage: dict[str, dict[str, Any]] = {}
    for stage, durations in stage_durations.items():
        total = sum(durations)
        per_stage[stage] = {
            "count": len(durations),
            "total_seconds": round(total, 3),
            "avg_seconds": round(total / len(durations), 3) if durations else 0.0,
            "max_seconds": round(max(durations), 3) if durations else 0.0,
        }
    stage_total = sum(item["total_seconds"] for item in per_stage.values())
    stage_latency_ready = bool(per_stage["stage0"]["count"] > 0 and per_stage["stage1"]["count"] > 0)
    throughput_ready = bool(generated_tokens > 0 and stage_total > 0)
    return {
        "stage_latency_ready": stage_latency_ready,
        "throughput_summary_ready": throughput_ready,
        "generated_token_count": generated_tokens,
        "accepted_rows": accepted_rows,
        "stage_total_seconds": round(stage_total, 3),
        "generated_tokens_per_stage_second": round(generated_tokens / stage_total, 6) if throughput_ready else 0.0,
        "accepted_rows_per_stage_second": round(accepted_rows / stage_total, 6) if stage_total > 0 else 0.0,
        "per_stage": per_stage,
    }


def safe_batch_summary(args: argparse.Namespace, generation: dict[str, Any]) -> dict[str, Any]:
    prompts = prompt_list_from_args(args)
    expected_request_count = len(prompts)
    batch_requested = expected_request_count > 1
    observed_request_count = int(generation.get("request_count") or (0 if batch_requested else expected_request_count))
    results = generation.get("results") if isinstance(generation.get("results"), list) else []
    safe_results: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        safe_results.append({
            "request_id": item.get("request_id"),
            "prompt_hash": item.get("prompt_hash"),
            "generated_token_count": int(item.get("generated_token_count") or 0),
            "max_new_tokens": item.get("max_new_tokens"),
            "generated_text_hash": item.get("generated_text_hash"),
            "decoded_tokens_match": item.get("decoded_tokens_match"),
            "multi_token_generation_ready": bool(item.get("multi_token_generation_ready")),
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        })
    result_identity_keys = [
        str(item.get("request_id") or item.get("prompt_hash") or "")
        for item in safe_results[:expected_request_count]
    ]
    batch_identity_ready = bool(
        not batch_requested
        or (
            len(result_identity_keys) >= expected_request_count
            and all(result_identity_keys)
            and len(set(result_identity_keys)) == expected_request_count
        )
    )
    prompt_hashes = [stable_hash_text(prompt) for prompt in prompts]
    generated_count = int(generation.get("generated_token_count") or 0)
    token_ready = bool(generated_count >= int(args.max_new_tokens))
    ready = bool(
        token_ready
        and observed_request_count == expected_request_count
        and batch_identity_ready
        and (not batch_requested or generation.get("batch_generation_ready") is True)
    )
    return {
        "enabled": batch_requested,
        "expected_request_count": expected_request_count,
        "observed_request_count": observed_request_count,
        "max_request_count": MAX_BATCH_REQUESTS,
        "prompt_hashes": prompt_hashes,
        "prompt_char_counts": [len(prompt) for prompt in prompts],
        "result_count": len(safe_results),
        "results": safe_results,
        "batch_identity_ready": batch_identity_ready,
        "batch_generation_ready": ready,
        "raw_prompts_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }


def safe_stream_summary(args: argparse.Namespace, generate_payload: dict[str, Any]) -> dict[str, Any]:
    stream = generate_payload.get("stream") if isinstance(generate_payload.get("stream"), dict) else {}
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
    requested = bool(getattr(args, "stream_generation", False))
    complete = bool(progress.get("stream_progress_complete"))
    prompt_count = len(prompt_list_from_args(args))
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
            "event_count": int(item.get("event_count") or 0),
            "observed_token_counts": [int(value) for value in counts if isinstance(value, int)],
            "max_observed_token_count": int(item.get("max_observed_token_count") or 0),
            "target_token_count": int(item.get("target_token_count") or args.max_new_tokens),
            "monotonic_progress": bool(item.get("monotonic_progress")),
            "stream_progress_complete": bool(item.get("stream_progress_complete")),
        })
    per_request_complete = bool(progress.get("per_request_progress_complete"))
    per_request_monotonic = bool(progress.get("per_request_monotonic_progress"))
    if prompt_count > 1 and (not per_request_progress or not per_request_complete or not per_request_monotonic):
        complete = False
    if prompt_count <= 1 and not bool(progress.get("monotonic_progress")):
        complete = False
    ready = bool(
        requested
        and stream.get("enabled")
        and complete
        and int(stream.get("event_count") or len(safe_events)) >= int(args.max_new_tokens)
        and (
            prompt_count <= 1
            or int(stream.get("event_count") or len(safe_events)) >= int(args.max_new_tokens) * prompt_count
        )
    )
    return {
        "enabled": bool(stream.get("enabled")) if stream else requested,
        "requested": requested,
        "event_count": int(stream.get("event_count") or len(safe_events)),
        "source": stream.get("source") or ("disabled" if not requested else ""),
        "endpoint_ready": bool(stream.get("endpoint_ready")),
        "progress": {
            "stream_progress_complete": complete,
            "all_token_events_ready": bool(progress.get("all_token_events_ready")),
            "monotonic_progress": bool(progress.get("monotonic_progress")),
            "expected_request_count": int(progress.get("expected_request_count") or prompt_count),
            "per_request_progress": list(per_request_progress),
            "per_request_progress_complete": per_request_complete,
            "per_request_monotonic_progress": per_request_monotonic,
            "observed_token_counts": list(progress.get("observed_token_counts") or []),
            "max_observed_token_count": int(progress.get("max_observed_token_count") or 0),
            "max_new_tokens": int(progress.get("max_new_tokens") or args.max_new_tokens),
            "source": progress.get("source") or stream.get("source") or "",
        },
        "events": safe_events,
        "stream_generation_ready": ready,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }


def _rss_mb(kind: int) -> float:
    if resource is None:
        return 0.0
    try:
        raw = float(resource.getrusage(kind).ru_maxrss)
    except Exception:
        return 0.0
    if raw <= 0:
        return 0.0
    # Linux reports KiB, macOS reports bytes.
    divisor = 1024.0 * 1024.0 if platform.system().lower() == "darwin" else 1024.0
    return round(raw / divisor, 3)


def runtime_resource_summary(backend: str) -> dict[str, Any]:
    peak_self_mb = _rss_mb(resource.RUSAGE_SELF) if resource is not None else 0.0
    peak_children_mb = _rss_mb(resource.RUSAGE_CHILDREN) if resource is not None else 0.0
    cuda = cuda_runtime_summary()
    vram_values = [int(value) for value in cuda.get("vram_total_mb") or [] if int(value or 0) > 0]
    memory_ready = bool(peak_self_mb > 0 or peak_children_mb > 0 or vram_values)
    return {
        "backend": backend,
        "memory_or_vram_summary_ready": memory_ready,
        "peak_self_rss_mb": peak_self_mb,
        "peak_child_rss_mb": peak_children_mb,
        "cuda_available": bool(cuda.get("cuda_available")),
        "gpu_count": int(cuda.get("gpu_count") or 0),
        "vram_total_mb": vram_values,
        "diagnosis_codes": cuda.get("diagnosis_codes") or [],
    }


def parse_prompt_texts_arg(prompt_text: str, prompt_texts: str) -> list[str]:
    if str(prompt_texts or "").strip():
        prompts = [item.strip() for item in str(prompt_texts).split(",") if item.strip()]
    else:
        prompts = [str(prompt_text or "").strip()] if str(prompt_text or "").strip() else []
    return validate_prompt_texts(prompts)


def validate_prompt_texts(prompts: list[str]) -> list[str]:
    prompts = [str(prompt or "").strip() for prompt in prompts if str(prompt or "").strip()]
    if not prompts:
        raise ValueError("prompt_text is required")
    if len(prompts) > MAX_BATCH_REQUESTS:
        raise ValueError(f"prompt_texts must contain at most {MAX_BATCH_REQUESTS} prompts")
    for prompt in prompts:
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"prompt_text must be at most {MAX_PROMPT_CHARS} characters")
    return prompts


def read_prompt_texts_file(path_value: str) -> list[str]:
    path = Path(str(path_value or "")).expanduser()
    if not str(path_value or "").strip():
        raise ValueError("prompt_texts_file is required")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not read prompt texts file: {exc}") from exc
    line_prompts = [(lineno, line.strip()) for lineno, line in enumerate(text.splitlines(), start=1) if line.strip()]
    if not line_prompts:
        raise ValueError("prompt_texts_file is empty")
    if len(line_prompts) > MAX_BATCH_REQUESTS:
        raise ValueError(f"prompt_texts_file must contain at most {MAX_BATCH_REQUESTS} prompts")
    for lineno, prompt in line_prompts:
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"prompt_texts_file line {lineno} must be at most {MAX_PROMPT_CHARS} characters")
    return [prompt for _, prompt in line_prompts]


def read_prompt_file(path_value: str) -> str:
    path = Path(str(path_value or "")).expanduser()
    if not str(path_value or "").strip():
        raise ValueError("prompt_file is required")
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"could not read prompt_file: {exc}") from exc
    if not prompt:
        raise ValueError("prompt_file is empty")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"prompt_file must be at most {MAX_PROMPT_CHARS} characters")
    return prompt


def prompt_list_from_args(args: argparse.Namespace) -> list[str]:
    prompt_list = getattr(args, "prompt_texts_list", None)
    if isinstance(prompt_list, list) and prompt_list:
        return validate_prompt_texts([str(prompt) for prompt in prompt_list])
    prompt_file = str(getattr(args, "prompt_file", "") or "")
    if prompt_file:
        return validate_prompt_texts([read_prompt_file(prompt_file)])
    return parse_prompt_texts_arg(str(getattr(args, "prompt_text", "") or ""), str(getattr(args, "prompt_texts", "") or ""))


def wait_workload_queued(base_url: str, *, timeout: float) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() <= deadline:
        try:
            state = request_json(
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


def validate_public_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    if output_request.get("include_output") is not False:
        errors.append("output_request_include_output_mismatch")
    if output_request.get("raw_prompt_public") is not False:
        errors.append("output_request_raw_prompt_public_mismatch")
    if output_request.get("raw_generated_text_public") is not False:
        errors.append("output_request_raw_generated_text_public_mismatch")
    if output_request.get("generated_token_ids_public") is not False:
        errors.append("output_request_generated_token_ids_public_mismatch")
    if output_request.get("public_artifact_safe") is not True:
        errors.append("output_request_public_artifact_safe_mismatch")

    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    source = prompt_scope.get("source")
    if source not in {"prompt-text", "prompt-file", "prompt-texts", "prompt-texts-file"}:
        errors.append("prompt_scope_source_mismatch")
    if not isinstance(prompt_scope.get("prompt_count"), int) or prompt_scope.get("prompt_count") < 1:
        errors.append("prompt_scope_count_mismatch")
    expected_inline = source in {"prompt-text", "prompt-texts"}
    expected_prefer_file = source in {"prompt-text", "prompt-texts", "prompt-file", "prompt-texts-file"}
    if prompt_scope.get("inline_prompt_text") is not expected_inline:
        errors.append("prompt_scope_inline_prompt_text_mismatch")
    if prompt_scope.get("terminal_next_commands_local_private") is not expected_inline:
        errors.append("prompt_scope_terminal_next_commands_mismatch")
    if prompt_scope.get("terminal_logs_local_private") is not expected_inline:
        errors.append("prompt_scope_terminal_logs_mismatch")
    if prompt_scope.get("saved_artifacts_prompt_placeholders") is not True:
        errors.append("prompt_scope_saved_placeholders_mismatch")
    if prompt_scope.get("saved_artifacts_public_safe") is not True:
        errors.append("prompt_scope_saved_artifacts_public_safe_mismatch")
    if prompt_scope.get("prefer_prompt_file_or_stdin_for_shareable_logs") is not expected_prefer_file:
        errors.append("prompt_scope_shareable_log_guidance_mismatch")
    if prompt_scope.get("prompt_file_path_public") is not False:
        errors.append("prompt_scope_prompt_file_path_public_mismatch")
    if prompt_scope.get("raw_prompt_public") is not False:
        errors.append("prompt_scope_raw_prompt_public_mismatch")
    if prompt_scope.get("public_artifact_safe") is not True:
        errors.append("prompt_scope_public_artifact_safe_mismatch")

    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    if answer_scope.get("scope_state") != "no-local-answer":
        errors.append("answer_scope_state_mismatch")
    if answer_scope.get("terminal_only") is not False:
        errors.append("answer_scope_terminal_only_mismatch")
    if answer_scope.get("visible_in_terminal") is not False:
        errors.append("answer_scope_visible_in_terminal_mismatch")
    if answer_scope.get("saved_json_display") != "hash-only":
        errors.append("answer_scope_saved_json_display_mismatch")
    if answer_scope.get("saved_markdown_display") != "none":
        errors.append("answer_scope_saved_markdown_display_mismatch")
    if answer_scope.get("raw_prompt_public") is not False:
        errors.append("answer_scope_raw_prompt_public_mismatch")
    if answer_scope.get("raw_generated_text_public") is not False:
        errors.append("answer_scope_raw_generated_text_public_mismatch")
    if answer_scope.get("generated_token_ids_public") is not False:
        errors.append("answer_scope_generated_token_ids_public_mismatch")
    if answer_scope.get("public_artifact_safe") is not True:
        errors.append("answer_scope_public_artifact_safe_mismatch")

    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    if shareable.get("saved_artifacts_public_safe") is not True:
        errors.append("shareable_saved_artifacts_public_safe_mismatch")
    if shareable.get("raw_prompt_public") is not False:
        errors.append("shareable_raw_prompt_public_mismatch")
    if shareable.get("raw_generated_text_public") is not False:
        errors.append("shareable_raw_generated_text_public_mismatch")
    if shareable.get("generated_token_ids_public") is not False:
        errors.append("shareable_generated_token_ids_public_mismatch")
    if shareable.get("answer_scope_state") != "no-local-answer":
        errors.append("shareable_answer_scope_state_mismatch")
    if shareable.get("local_answer_terminal_only") is not False:
        errors.append("shareable_local_answer_terminal_only_mismatch")
    if shareable.get("public_artifact_safe") is not True:
        errors.append("shareable_public_artifact_safe_mismatch")

    encoded = json.dumps(report, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment and fragment in encoded:
            errors.append(f"sensitive_fragment:{fragment}")
    for path in public_leak_paths(report):
        if path.endswith(".prompt_hash") or ".safety." in path:
            continue
        errors.append(f"public_leak:{path}")
    return sorted(set(errors))


def attach_serve_process(report: dict[str, Any], serve_process: dict[str, Any]) -> dict[str, Any]:
    ensure_output_scope(report)
    report["serve_process"] = summarize_serve_process(serve_process, ok=bool(report.get("ok")))
    safety_errors = validate_public_report(report)
    if safety_errors:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"public_report_safety_failed"})
        report["safety_errors"] = safety_errors
    return support_bundle.sanitize(report)


def degraded_report(args: argparse.Namespace, missing: list[str], output_dir: Path) -> dict[str, Any]:
    ok = not bool(args.require_hf_runtime)
    codes = [
        "hf_dependencies_missing",
        "product_swarm_mvp_hf_runtime_missing",
    ]
    if ok:
        codes.append("product_swarm_mvp_degraded_ready")
    return ensure_output_scope({
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": "local-loopback",
        "degraded": True,
        "output_dir": str(output_dir),
        "missing_dependencies": missing,
        "diagnosis_codes": codes,
        "operator_action": "Install optional runtime dependencies with: python -m pip install -e '.[hf]'",
        "safety": {
            "coordinator_backed_task_execution": True,
            "read_only_workload": WORKLOAD_TYPE,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "raw_runtime_state_public": False,
            "raw_runtime_state_removed": True,
            "not_production": True,
            "not_p2p": True,
            "not_large_model_serving": True,
        },
    }, args)


def run_local_loopback(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    base_url = f"http://127.0.0.1:{args.port}"
    state_dir = output_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    serve_cmd = [
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "serve",
        "--profile",
        "gpu-generation" if args.backend == "cuda" else "cpu-real-llm",
        "--bind-host",
        "127.0.0.1",
        "--public-host",
        "127.0.0.1",
        "--port",
        str(args.port),
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
    if args.hf_cache_dir:
        serve_cmd.extend(["--hf-cache-dir", args.hf_cache_dir])
    serve_proc = popen_command(serve_cmd)
    serve_process: dict[str, Any] = {}
    steps: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {}
    pending_report: dict[str, Any] | None = None
    generate_proc: subprocess.Popen[str] | None = None
    try:
        healthy, health_error = wait_health(base_url, serve_proc, args.startup_timeout)
        steps.append({
            "name": "serve",
            "ok": healthy,
            "command": "crowdtensor serve --run",
            "error": health_error,
        })
        if not healthy:
            pending_report = {
                "schema": SCHEMA,
                "generated_at": utc_now(),
                "ok": False,
                "mode": "local-loopback",
                "output_dir": str(output_dir),
                "steps": steps,
                "diagnosis_codes": ["serve_start_failed"],
            }
            ensure_output_scope(pending_report, args)
            return pending_report

        generate_cmd = [
            sys.executable,
            "-m",
            "crowdtensor.cli",
            "generate",
            "--coordinator-url",
            base_url,
            "--admin-token",
            ADMIN_TOKEN,
            "--backend",
            args.backend,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--timeout-seconds",
            str(args.generate_timeout),
            "--json",
        ]
        if args.prompt_texts_file:
            generate_cmd.extend(["--prompt-texts-file", args.prompt_texts_file])
        elif args.prompt_file:
            generate_cmd.extend(["--prompt-file", args.prompt_file])
        elif args.prompt_texts:
            generate_cmd.extend(["--prompt-texts", args.prompt_texts])
        else:
            generate_cmd.extend(["--prompt-text", args.prompt_text])
        if args.stream_generation:
            generate_cmd.append("--stream")
        generate_proc = popen_command(generate_cmd)
        session_queued, queue_error = wait_workload_queued(base_url, timeout=args.session_queue_timeout)
        steps.append({
            "name": "generate_session_created",
            "ok": session_queued,
            "error": queue_error,
        })
        if not session_queued:
            generate_step, generate_payload = finish_process_step("generate", generate_proc, timeout=1.0)
            steps.append(generate_step)
            payloads["generate"] = generate_payload
            pending_report = finalize_report(args, output_dir, base_url, steps, payloads, serve_process)
            return pending_report

        for generation_step in range(args.max_new_tokens):
            for stage in ("stage0", "stage1"):
                name = f"join_{stage}_step_{generation_step}"
                command = [
                    sys.executable,
                    "-m",
                    "crowdtensor.cli",
                    "join",
                    "--coordinator-url",
                    base_url,
                    "--miner-id",
                    f"product-mvp-{stage}",
                    "--stage",
                    stage,
                    "--backend",
                    args.backend,
                    "--miner-token",
                    MINER_TOKEN,
                    "--hf-model-id",
                    args.hf_model_id,
                    "--once",
                    "--max-tasks",
                    "1",
                    "--run",
                    "--json",
                ]
                if args.hf_cache_dir:
                    command.extend(["--hf-cache-dir", args.hf_cache_dir])
                step, payload = run_step(name, command, timeout=args.miner_timeout)
                steps.append(step)
                payloads[name] = payload
                if not step.get("ok"):
                    stop_process(generate_proc)
                    pending_report = finalize_report(args, output_dir, base_url, steps, payloads, serve_process)
                    return pending_report

        generate_step, generate_payload = finish_process_step("generate", generate_proc, timeout=args.generate_timeout + 10.0)
        steps.append(generate_step)
        payloads["generate"] = generate_payload
        pending_report = finalize_report(args, output_dir, base_url, steps, payloads, serve_process)
        return pending_report
    finally:
        if generate_proc is not None and generate_proc.poll() is None:
            stop_process(generate_proc)
        serve_process.update(stop_process(serve_proc))
        if pending_report is not None:
            sanitized_report = attach_serve_process(pending_report, serve_process)
            pending_report.clear()
            pending_report.update(sanitized_report)


def finalize_report(
    args: argparse.Namespace,
    output_dir: Path,
    base_url: str,
    steps: list[dict[str, Any]],
    payloads: dict[str, Any],
    serve_process: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    ledger_error = ""
    session_id = str(((payloads.get("generate") or {}).get("session") or {}).get("session_id") or "")
    try:
        query = f"/admin/results?status=accepted&workload_type={WORKLOAD_TYPE}&limit=100"
        if session_id:
            query += f"&session_id={session_id}"
        ledger = request_json("GET", base_url, query, admin_token=ADMIN_TOKEN, timeout=10.0)
        rows = ledger.get("results") if isinstance(ledger.get("results"), list) else []
    except HTTPError as exc:
        ledger_error = f"http_{exc.code}"
    except Exception as exc:
        ledger_error = f"{type(exc).__name__}: {exc}"
    generation = (payloads.get("generate") or {}).get("generation") if isinstance((payloads.get("generate") or {}).get("generation"), dict) else {}
    batch = safe_batch_summary(args, generation)
    stream = safe_stream_summary(args, payloads.get("generate") or {})
    expected_request_count = int(batch.get("expected_request_count") or 1)
    stages = stage_summary(rows)
    generated_tokens = int(generation.get("generated_token_count") or 0)
    performance = summarize_step_durations(steps, generated_tokens=generated_tokens, accepted_rows=len(rows))
    resources = runtime_resource_summary(args.backend)
    step_ok = all(bool(step.get("ok")) for step in steps)
    generation_ready = bool(
        (payloads.get("generate") or {}).get("ok") is True
        and generated_tokens >= args.max_new_tokens
        and batch.get("batch_generation_ready") is True
    )
    stage_ready = bool(
        stages.get("distinct_stage_miners")
        and int(stages.get("completed_rows") or 0) >= args.max_new_tokens * 2
    )
    stream_ready = bool(not args.stream_generation or stream.get("stream_generation_ready") is True)
    ok = bool(step_ok and generation_ready and stage_ready and stream_ready and not ledger_error)
    codes = set()
    if ok:
        codes.update({
            "product_swarm_mvp_ready",
            "serve_join_generate_mvp_ready",
            "local_two_stage_real_llm_ready",
            "generated_token_count_ready",
            "distinct_stage_miners",
            "stage_assignment_valid",
        })
        if batch.get("enabled") and batch.get("batch_generation_ready"):
            codes.add("product_swarm_mvp_batch_ready")
            codes.add("public_swarm_generate_batch_ready")
        if stream.get("stream_generation_ready"):
            codes.add("product_swarm_mvp_stream_ready")
            codes.add("public_swarm_generate_stream_ready")
            if stream.get("endpoint_ready"):
                codes.add("public_swarm_generate_stream_endpoint_ready")
        if performance.get("stage_latency_ready"):
            codes.add("stage_latency_ready")
        if performance.get("throughput_summary_ready"):
            codes.add("throughput_summary_ready")
        if resources.get("memory_or_vram_summary_ready"):
            codes.add("memory_or_vram_summary_ready")
    else:
        codes.add("product_swarm_mvp_blocked")
        if not generation_ready:
            codes.add("generation_not_ready")
        if args.stream_generation and not stream_ready:
            codes.add("stream_generation_not_ready")
        if not stage_ready:
            codes.add("stage_assignment_incomplete")
        if ledger_error:
            codes.add("admin_results_failed")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": "local-loopback",
        "output_dir": str(output_dir),
        "backend": args.backend,
        "hf_model_id": args.hf_model_id,
        "max_new_tokens": args.max_new_tokens,
        "request_count": expected_request_count,
        "steps": steps,
        "session": {
            "session_id": session_id,
            "workload_type": WORKLOAD_TYPE,
        },
        "generation": generation,
        "batch": batch,
        "stream": stream,
        "stage_assignment": stages,
        "performance": performance,
        "runtime_resources": resources,
        "ledger": {
            "accepted_rows": len(rows),
            "error": ledger_error,
        },
        "serve_process": serve_process,
        "diagnosis_codes": sorted(codes),
        "safety": {
            "coordinator_backed_task_execution": True,
            "read_only_workload": WORKLOAD_TYPE,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "raw_runtime_state_public": False,
            "not_production": True,
            "not_p2p": True,
            "not_large_model_serving": True,
        },
    }
    ensure_output_scope(report, args)
    safety_errors = validate_public_report(report)
    if safety_errors:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report["diagnosis_codes"]) | {"public_report_safety_failed"})
        report["safety_errors"] = safety_errors
    return support_bundle.sanitize(report)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = missing_hf_dependencies()
    if missing:
        report = degraded_report(args, missing, output_dir)
    else:
        report = run_local_loopback(args, output_dir)
    ensure_output_scope(report, args)
    report = support_bundle.sanitize(report)
    cleanup_summary = cleanup_private_runtime_state(
        output_dir,
        keep_private_state=bool(getattr(args, "keep_private_state", False)),
    )
    report["private_runtime_state"] = cleanup_summary
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    safety["raw_runtime_state_public"] = False
    safety["raw_runtime_state_removed"] = bool(cleanup_summary.get("removed") or not cleanup_summary.get("present_after_cleanup"))
    safety["private_runtime_state_kept"] = bool(cleanup_summary.get("kept"))
    report["safety"] = safety
    if cleanup_summary.get("error"):
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"private_runtime_state_cleanup_failed"})
    elif cleanup_summary.get("kept"):
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"private_runtime_state_retained"})
    else:
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"private_runtime_state_cleaned"})
    report["artifacts"] = {
        "product_swarm_mvp_check_json": {
            "kind": "product_swarm_mvp_check",
            "path": "product_swarm_mvp_check.json",
            "present": True,
            "schema": SCHEMA,
            "ok": report.get("ok"),
        }
    }
    write_json(output_dir / "product_swarm_mvp_check.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the v0.3 product Swarm Inference MVP path.")
    parser.add_argument("--output-dir", default="dist/product-swarm-mvp")
    parser.add_argument("--port", type=int, default=9877)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--prompt-text", default="CrowdTensor turns spare compute into open inference")
    parser.add_argument("--prompt-file", default="", help="UTF-8 single prompt file")
    parser.add_argument("--prompt-texts", default="", help="comma-separated bounded batch of up to 4 prompts")
    parser.add_argument("--prompt-texts-file", default="", help="UTF-8 batch prompt file with one non-empty prompt per line")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--session-queue-timeout", type=float, default=30.0)
    parser.add_argument("--miner-timeout", type=float, default=180.0)
    parser.add_argument("--generate-timeout", type=float, default=180.0)
    parser.add_argument("--stream-generation", action="store_true", help="require safe generate --stream progress evidence")
    parser.add_argument("--require-hf-runtime", action="store_true")
    parser.add_argument("--keep-private-state", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.port < 1:
        raise SystemExit("--port must be positive")
    if args.max_new_tokens < 2 or args.max_new_tokens > 8:
        raise SystemExit("--max-new-tokens must be between 2 and 8 for this smoke")
    raw_argv = list(argv if argv is not None else sys.argv[1:])
    prompt_text_explicit = "--prompt-text" in raw_argv or any(item.startswith("--prompt-text=") for item in raw_argv)
    prompt_sources = [
        ("--prompt-text", prompt_text_explicit),
        ("--prompt-file", bool(args.prompt_file)),
        ("--prompt-texts", bool(args.prompt_texts)),
        ("--prompt-texts-file", bool(args.prompt_texts_file)),
    ]
    if sum(1 for _, present in prompt_sources if present) > 1:
        raise SystemExit(
            "product_swarm_mvp accepts one prompt source: --prompt-text, --prompt-file, "
            "--prompt-texts, or --prompt-texts-file"
        )
    try:
        if args.prompt_texts_file:
            args.prompt_texts_list = read_prompt_texts_file(args.prompt_texts_file)
        elif args.prompt_file:
            args.prompt_texts_list = [read_prompt_file(args.prompt_file)]
        else:
            args.prompt_texts_list = parse_prompt_texts_arg(args.prompt_text, args.prompt_texts)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for name in ["startup_timeout", "session_queue_timeout", "miner_timeout", "generate_timeout"]:
        if float(getattr(args, name)) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"Product Swarm MVP check ready: {report.get('ok')}")
        print(f"Diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
        output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
        prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
        answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
        if output_request:
            print(
                "Output request: "
                f"include_output={output_request.get('include_output')} "
                f"raw_prompt_public={output_request.get('raw_prompt_public')} "
                f"raw_generated_text_public={output_request.get('raw_generated_text_public')} "
                f"generated_token_ids_public={output_request.get('generated_token_ids_public')}"
            )
        if prompt_scope:
            print(
                "Prompt scope: "
                f"source={prompt_scope.get('source')} "
                f"count={prompt_scope.get('prompt_count')} "
                f"raw_prompt_public={prompt_scope.get('raw_prompt_public')}"
            )
        if answer_scope:
            print(
                "Answer scope: "
                f"{answer_scope.get('scope_state')} "
                f"saved_json={answer_scope.get('saved_json_display')} "
                f"public_artifact_safe={answer_scope.get('public_artifact_safe')}"
            )
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
