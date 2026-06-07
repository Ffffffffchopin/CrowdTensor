#!/usr/bin/env python3
"""Automate the real Internet Swarm Inference Beta Kaggle acceptance path."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import support_bundle  # noqa: E402
from create_miner_invite import create_invite  # noqa: E402
from crowdtensor.real_llm import BACKEND_CPU as REAL_LLM_BACKEND_CPU  # noqa: E402
from crowdtensor.real_llm import BACKEND_CUDA as REAL_LLM_BACKEND_CUDA  # noqa: E402
from crowdtensor.real_llm import DEFAULT_MODEL_ID, DEFAULT_PROMPTS  # noqa: E402
from crowdtensor.real_llm import normalize_backend as normalize_real_llm_backend  # noqa: E402
from crowdtensor.real_llm import normalize_partition_mode as normalize_real_llm_partition_mode  # noqa: E402
from kaggle_real_llm_live_package import DEFAULT_CUDA_TORCH_INDEX_URL, DEFAULT_CUDA_TORCH_RUNTIME_SPEC  # noqa: E402
from kaggle_real_llm_live_package import DEFAULT_TRANSFORMERS_SPEC  # noqa: E402


SCHEMA = "real_llm_internet_beta_v1"
MODE_KAGGLE_AUTO = "kaggle-auto"
MODE_EVIDENCE_IMPORT = "evidence-import"
MODES = [MODE_KAGGLE_AUTO, MODE_EVIDENCE_IMPORT]
DEFAULT_PUBLIC_HOST = "24.199.118.54"
DEFAULT_PORT = 9190
DEFAULT_BASE_PORT = 9191
WORKLOAD_TYPE = "real_llm_sharded_infer"
STAGES = ("stage0", "stage1")
ROLE_NORMAL = "normal"
ROLE_VICTIM = "victim"
ROLE_RESCUE = "rescue"
FAILURE_NONE = "none"
FAILURE_KILL_STAGE0_AFTER_CLAIM = "kill-stage0-after-claim"
FAILURE_KILL_STAGE1_AFTER_CLAIM = "kill-stage1-after-claim"
FAILURE_MODES = {
    FAILURE_NONE,
    FAILURE_KILL_STAGE0_AFTER_CLAIM,
    FAILURE_KILL_STAGE1_AFTER_CLAIM,
}
LIVE_REQUEUE_SUMMARY_FIELDS = (
    "enabled",
    "failure_mode",
    "target_stage",
    "victim_miner_id",
    "rescue_miner_id",
    "claim_observed",
    "victim_kernel_deleted",
    "lease_expired",
    "rescue_miner_used",
    "rescued_result",
    "accepted_result_after_requeue",
    "victim_result_accepted",
)
KAGGLE_CODE_URL = re.compile(r"https://www\.kaggle\.com/code/([^/\s]+)/([^/\s]+)")

Runner = Callable[..., subprocess.CompletedProcess[str]]
PopenFactory = Callable[..., subprocess.Popen[str]]
ReadyProbe = Callable[[str, float, float], dict[str, Any]]
StateProbe = Callable[[argparse.Namespace, str, list[str]], dict[str, Any]]

SECRET_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "lease_token",
    "idempotency_key",
    "activation_results",
    "activation_result",
    "hidden_state",
    "input_ids",
    "logits",
    "inference_results",
    "inference_result",
    "sharded_inference_result",
    "real_llm_sharded_result",
    "output_text",
    "Bearer ",
    "SOURCE_TARBALL_B64",
    "MINER_ENV_TEXT",
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
    "CrowdTensor routes",
    "A miner returns",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
            "Real Internet Swarm Inference Beta artifacts summarize external "
            "tiny GPT split readiness, Kaggle cleanup, live requeue evidence, "
            "generation hashes/counts, and diagnostics only. They do not include "
            "answer text."
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
            "This Beta report is shareable operator evidence, not a local answer "
            "transcript; raw prompts, generated text, token ids, activations, "
            "leases, credentials, private env files, and raw runtime state are "
            "excluded."
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
            "Share real_llm_internet_beta*.json/md artifacts; they contain "
            "readiness evidence, cleanup state, generation hashes/counts, and "
            "diagnostics, not raw prompts or answers."
        ),
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("command emitted no JSON object")


def redact_text(text: str, secret_values: list[str] | None = None) -> str:
    redacted = text
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


def parse_private_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        parsed = shlex.split(raw_value)
        values[key] = parsed[0] if parsed else ""
    return values


def secret_values_from_envs(*paths: Path) -> list[str]:
    values: list[str] = []
    for path in paths:
        for value in parse_private_env(path).values():
            if value:
                values.append(value)
    return values


def diagnosis_codes(*payloads: dict[str, Any], extra: list[str] | None = None) -> list[str]:
    codes: set[str] = set(extra or [])
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for code in payload.get("diagnosis_codes") or []:
            if isinstance(code, str):
                codes.add(code)
        summaries = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
        for summary in summaries.values():
            if isinstance(summary, dict):
                for code in summary.get("diagnosis_codes") or []:
                    if isinstance(code, str):
                        codes.add(code)
    return sorted(codes)


def summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    summary: dict[str, Any] = {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "diagnosis_codes": diagnosis_codes(payload),
    }
    generation = generation_summary(payload)
    if generation:
        summary["generation"] = generation
    runtime = payload.get("runtime_classification")
    if isinstance(runtime, dict):
        summary["runtime_classification"] = {
            "package_only": runtime.get("package_only"),
            "external_runtime_verified": runtime.get("external_runtime_verified"),
            "kaggle_auto": runtime.get("kaggle_auto"),
            "stage_requeue_verified": runtime.get("stage_requeue_verified"),
        }
    workload = payload.get("workload")
    if isinstance(workload, dict):
        summary["workload"] = {
            "workload_type": workload.get("workload_type"),
            "stage_mode": workload.get("stage_mode"),
            "request_count": workload.get("request_count"),
            "max_new_tokens": workload.get("max_new_tokens"),
            "hf_model_id": workload.get("hf_model_id"),
            "real_llm_backend": workload.get("real_llm_backend"),
            "real_llm_partition_mode": workload.get("real_llm_partition_mode"),
        }
    if payload.get("schema") == "kaggle_real_llm_live_package_v1":
        summary["dataset_ref"] = payload.get("dataset_ref")
        summary["hf_model_id"] = payload.get("hf_model_id")
        summary["real_llm_backend"] = payload.get("real_llm_backend")
        summary["max_tasks"] = payload.get("max_tasks")
        summary["max_request_attempts"] = payload.get("max_request_attempts")
        summary["stages"] = [
            {
                "stage": item.get("stage"),
                "role": item.get("role"),
                "key": item.get("key"),
                "kernel_ref": item.get("kernel_ref"),
                "inline_kernel_payload": item.get("inline_kernel_payload"),
                "hf_runtime_enabled": item.get("hf_runtime_enabled"),
                "real_llm_stage_role_present": item.get("real_llm_stage_role_present"),
            "real_llm_backend": item.get("real_llm_backend"),
            "real_llm_partition_mode": item.get("real_llm_partition_mode"),
            "gpu_accelerator_enabled": item.get("gpu_accelerator_enabled"),
                "cuda_preflight_present": item.get("cuda_preflight_present"),
            }
            for item in payload.get("stages") or []
            if isinstance(item, dict)
        ]
    if isinstance(payload.get("payload_summaries"), dict):
        for key, value in payload["payload_summaries"].items():
            if isinstance(value, dict):
                summary[key] = {
                    "schema": value.get("schema"),
                    "ok": value.get("ok"),
                    "diagnosis_codes": value.get("diagnosis_codes") or [],
                    "generation": generation_summary(value),
                    "stage_assignment": value.get("stage_assignment") or {},
                    "artifact": value.get("artifact") or {},
                }
    return summary


def generation_summary(payload: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_score = -1
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
        generation = item.get("generation")
        if isinstance(generation, dict) and generation:
            score = 0
            if generation.get("generated_text_hash"):
                score += 8
            if generation.get("generated_token_count"):
                score += 4
            if generation.get("multi_token_generation_ready"):
                score += 2
            if generation.get("generated_text_redacted"):
                score += 1
            if score > best_score:
                best = generation
                best_score = score
        for value in item.values():
            if isinstance(value, dict):
                pending.append(value)
            elif isinstance(value, list):
                pending.extend(entry for entry in value if isinstance(entry, dict))
    return best


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def imported_backend_from_payload(payload: dict[str, Any], fallback: str) -> str:
    observed = first_string_value(payload, "real_llm_backend")
    if not observed:
        return fallback
    try:
        return normalize_real_llm_backend(observed)
    except ValueError:
        return fallback


def imported_partition_mode_from_payload(payload: dict[str, Any], fallback: str) -> str:
    observed = first_string_value(payload, "real_llm_partition_mode")
    if not observed:
        return fallback
    try:
        return normalize_real_llm_partition_mode(observed)
    except ValueError:
        return fallback


def imported_string_from_payload(payload: dict[str, Any], key: str, fallback: str) -> str:
    observed = first_string_value(payload, key)
    return observed or fallback


def safety_summary_for_backend(args: argparse.Namespace, *, cleanup_ok: bool, real_llm_backend: str) -> dict[str, Any]:
    summary = safety_summary(args, cleanup_ok=cleanup_ok)
    gpu_backend = real_llm_backend == REAL_LLM_BACKEND_CUDA
    summary["cpu_only_workload"] = not gpu_backend
    summary["gpu_backend_selected"] = gpu_backend
    summary["coordinator_cuda_runtime_required"] = False if gpu_backend else None
    summary["miner_cuda_runtime_required"] = gpu_backend
    summary["not_gpu_tpu_pooling"] = not gpu_backend
    return summary


def safe_live_requeue_summary(payload: dict[str, Any]) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    best: dict[str, Any] = {}
    best_score = -1
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
        summary = item.get("live_requeue_summary") if isinstance(item.get("live_requeue_summary"), dict) else {}
        if summary:
            score = sum(1 for field in LIVE_REQUEUE_SUMMARY_FIELDS if field in summary)
            if score > best_score:
                best = summary
                best_score = score
        for value in item.values():
            if isinstance(value, dict):
                pending.append(value)
            elif isinstance(value, list):
                pending.extend(entry for entry in value if isinstance(entry, dict))
    safe = {field: best.get(field) for field in LIVE_REQUEUE_SUMMARY_FIELDS if field in best}
    if safe.get("lease_expired") == "<redacted>" and "live_requeue_lease_timeout_observed" in codes:
        safe["lease_expired"] = True
    return safe


def live_requeue_detail_ready(summary: dict[str, Any]) -> bool:
    return bool(
        summary.get("claim_observed") is True
        and summary.get("victim_kernel_deleted") is True
        and summary.get("lease_expired") is True
        and (summary.get("rescue_miner_used") is True or summary.get("rescue_miner_used") is None)
        and summary.get("rescued_result") is True
        and summary.get("victim_result_accepted") is False
    )


def model_summary(payloads: list[dict[str, Any]], expected_hf_model_id: str) -> dict[str, Any]:
    observed = sorted({
        str(value)
        for payload in payloads
        for value in [first_string_value(payload, "hf_model_id")]
        if value
    })
    observed_id = observed[0] if len(observed) == 1 else ""
    return {
        "expected_hf_model_id": expected_hf_model_id,
        "observed_hf_model_id": observed_id,
        "observed_hf_model_ids": observed,
        "model_id_present": bool(observed_id),
        "model_id_match": bool(observed_id and observed_id == expected_hf_model_id),
        "compatible": bool(observed_id and observed_id == expected_hf_model_id),
    }


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
        return (
            {
                "name": name,
                "ok": False,
                "returncode": None,
                "duration_seconds": round(time.monotonic() - started, 3),
                "error": "timeout",
            },
            {},
        )
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
        step["ok"] = bool(step["ok"] and payload.get("ok") is not False)
    if not step["ok"]:
        if completed.stdout and not payload:
            step["stdout_tail"] = redact_text(completed.stdout[-1200:], secret_values)
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1200:], secret_values)
    return step, payload


def run_step(
    name: str,
    command: list[str],
    *,
    runner: Runner,
    timeout_seconds: float,
    secret_values: list[str] | None = None,
) -> dict[str, Any]:
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
        }
    return {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout_tail": redact_text((completed.stdout or "")[-1200:], secret_values),
        "stderr_tail": redact_text((completed.stderr or "")[-1200:], secret_values),
    }


def wait_for_ready(url: str, timeout_seconds: float, poll_interval: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            request = Request(f"{url.rstrip('/')}/ready", method="GET")
            with urlopen(request, timeout=min(2.0, max(0.2, timeout_seconds))) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return {"ok": True, "payload": payload}
        except (OSError, URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(poll_interval)
    return {"ok": False, "error": last_error or "timeout"}


def terminate_process(process: subprocess.Popen[str], *, timeout: float) -> None:
    if process.poll() is not None:
        return
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            process.send_signal(signal.SIGTERM)
        process.wait(timeout=timeout)
    except Exception:
        try:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                process.kill()
            process.wait(timeout=timeout)
        except Exception:
            pass


def collect_process(
    process: subprocess.Popen[str],
    *,
    name: str,
    command: list[str],
    timeout: float,
    secret_values: list[str],
    terminate: bool,
) -> dict[str, Any]:
    if terminate:
        terminate_process(process, timeout=timeout)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        terminate_process(process, timeout=timeout)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", "process did not exit after termination"
    return {
        "name": name,
        "pid": process.pid,
        "returncode": process.returncode,
        "ok": process.returncode in {0, -signal.SIGTERM, 143, None},
        "command": command,
        "stdout_tail": redact_text((stdout or "")[-1200:], secret_values),
        "stderr_tail": redact_text((stderr or "")[-1200:], secret_values),
    }


def effective_coordinator_url(args: argparse.Namespace) -> str:
    if args.coordinator_url:
        return args.coordinator_url.rstrip("/")
    return f"http://{args.public_host}:{args.port}"


def local_ready_url(args: argparse.Namespace) -> str:
    return args.ready_url.rstrip("/") if args.ready_url else f"http://127.0.0.1:{args.port}"


def target_stage_for_failure(failure_mode: str) -> str:
    if failure_mode == FAILURE_KILL_STAGE0_AFTER_CLAIM:
        return "stage0"
    if failure_mode == FAILURE_KILL_STAGE1_AFTER_CLAIM:
        return "stage1"
    return ""


def stage_id(stage: str) -> int:
    return 1 if stage == "stage1" else 0


def kernel_key(stage: str, role: str = ROLE_NORMAL) -> str:
    return stage if role == ROLE_NORMAL else f"{stage}-{role}"


def kernel_role_from_key(key: str) -> str:
    if key.endswith("-victim"):
        return ROLE_VICTIM
    if key.endswith("-rescue"):
        return ROLE_RESCUE
    return ROLE_NORMAL


def stage_from_key(key: str) -> str:
    return key.split("-", 1)[0]


def victim_miner_id(args: argparse.Namespace, stage: str) -> str:
    return f"{args.miner_id}-{stage}-{ROLE_VICTIM}"


def rescue_miner_id(args: argparse.Namespace, stage: str) -> str:
    return f"{args.miner_id}-{stage}-{ROLE_RESCUE}"


def alpha_package_dir(output_dir: Path) -> Path:
    return output_dir / "alpha-package"


def alpha_live_dir(output_dir: Path) -> Path:
    return alpha_package_dir(output_dir) / "package-live-rc"


def kaggle_package_dir(output_dir: Path) -> Path:
    return output_dir / "kaggle-package"


def external_alpha_dir(output_dir: Path) -> Path:
    return output_dir / "external-alpha"


def stage_refs_from_package(payload: dict[str, Any]) -> dict[str, str]:
    refs: dict[str, str] = {}
    for item in payload.get("stages") or []:
        if isinstance(item, dict) and item.get("stage") in STAGES and item.get("kernel_ref"):
            key = str(item.get("key") or item["stage"])
            refs[key] = str(item["kernel_ref"])
    return refs


def kernel_entries_from_package(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in payload.get("stages") or []:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage") or "")
        ref = str(item.get("kernel_ref") or "")
        if stage not in STAGES or not ref:
            continue
        role = str(item.get("role") or kernel_role_from_key(str(item.get("key") or stage)))
        key = str(item.get("key") or kernel_key(stage, role))
        entries.append({
            "stage": stage,
            "role": role,
            "key": key,
            "kernel_ref": ref,
            "miner_id": str(item.get("miner_id") or ""),
        })
    return entries


def initial_push_entries(entries: list[dict[str, Any]], failure_mode: str) -> list[dict[str, Any]]:
    if failure_mode == FAILURE_NONE:
        return list(entries)
    target = target_stage_for_failure(failure_mode)
    selected = []
    for entry in entries:
        if entry["stage"] == target and entry["role"] == ROLE_RESCUE:
            continue
        selected.append(entry)
    return selected


def rescue_entry(entries: list[dict[str, Any]], failure_mode: str) -> dict[str, Any]:
    target = target_stage_for_failure(failure_mode)
    for entry in entries:
        if entry["stage"] == target and entry["role"] == ROLE_RESCUE:
            return entry
    return {}


def extract_status(text: str) -> str:
    upper = text.upper()
    for status in ["COMPLETE", "RUNNING", "QUEUED", "PENDING", "ERROR", "FAILED", "CANCELLED", "CANCELED"]:
        if status in upper:
            return status
    return "UNKNOWN"


def extract_kernel_ref(text: str) -> str:
    match = KAGGLE_CODE_URL.search(text or "")
    if not match:
        return ""
    return f"{match.group(1)}/{match.group(2)}"


def poll_kaggle_status(
    *,
    stage: str,
    ref: str,
    runner: Runner,
    timeout_seconds: float,
    poll_interval: float,
    secret_values: list[str],
) -> dict[str, Any]:
    del poll_interval
    step = run_step(
        f"kaggle_status_{stage}",
        ["kaggle", "kernels", "status", ref],
        runner=runner,
        timeout_seconds=min(60.0, max(1.0, timeout_seconds)),
        secret_values=secret_values,
    )
    step["stage"] = stage
    step["kernel_ref"] = ref
    step["status"] = extract_status(f"{step.get('stdout_tail', '')}\n{step.get('stderr_tail', '')}")
    step["attempts"] = 1
    step["snapshot_only"] = True
    return step


def collect_kaggle_output(
    *,
    key: str,
    ref: str,
    output_dir: Path,
    runner: Runner,
    timeout_seconds: float,
    secret_values: list[str],
) -> dict[str, Any]:
    output_path = output_dir / "kaggle-output" / key
    step = run_step(
        f"kaggle_output_{key}",
        ["kaggle", "kernels", "output", ref, "-p", str(output_path), "--force"],
        runner=runner,
        timeout_seconds=timeout_seconds,
        secret_values=secret_values,
    )
    step["key"] = key
    step["stage"] = stage_from_key(key)
    step["role"] = kernel_role_from_key(key)
    step["kernel_ref"] = ref
    step["output_path"] = str(output_path)
    step["artifact_count"] = len([path for path in output_path.rglob("*") if path.is_file()]) if output_path.exists() else 0
    return step


def fetch_state(args: argparse.Namespace, observer_token: str, secret_values: list[str]) -> dict[str, Any]:
    request = Request(
        f"{local_ready_url(args).rstrip('/')}/state",
        headers={"x-crowdtensor-observer-token": observer_token} if observer_token else {},
        method="GET",
    )
    with urlopen(request, timeout=max(1.0, float(args.http_timeout))) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return redact_values(payload if isinstance(payload, dict) else {}, secret_values)


def real_llm_task_stage(task: dict[str, Any]) -> str:
    metadata = task.get("workload_metadata") if isinstance(task.get("workload_metadata"), dict) else {}
    claim_spec = task.get("claim_workload_spec") if isinstance(task.get("claim_workload_spec"), dict) else {}
    for source in (metadata, claim_spec, task.get("validation") if isinstance(task.get("validation"), dict) else {}):
        try:
            sid = int((source or {}).get("stage_id", -1))
        except (TypeError, ValueError):
            sid = -1
        if sid == 0:
            return "stage0"
        if sid == 1:
            return "stage1"
    return ""


def find_task(state: dict[str, Any], *, task_id: str = "", stage: str = "", miner_id: str = "") -> dict[str, Any]:
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
    observer_token: str,
    secret_values: list[str],
    state_probe: StateProbe,
    target_stage: str,
    target_miner_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + float(args.claim_observe_timeout)
    last_status = ""
    last_state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            state = state_probe(args, observer_token, secret_values)
        except (OSError, URLError, json.JSONDecodeError) as exc:
            last_status = str(exc)
            time.sleep(args.poll_interval)
            continue
        last_state = state
        task = find_task(state, stage=target_stage, miner_id=target_miner_id)
        if task and task.get("status") == "leased":
            return {
                "ok": True,
                "stage": target_stage,
                "task_id": str(task.get("task_id") or ""),
                "attempt": task.get("attempt"),
                "miner_id": target_miner_id,
                "status": task.get("status"),
            }, state
        staged = find_task(state, stage=target_stage)
        if staged:
            last_status = f"{staged.get('status')}:{staged.get('miner_id')}"
        time.sleep(args.poll_interval)
    return {
        "ok": False,
        "stage": target_stage,
        "miner_id": target_miner_id,
        "error": last_status or "claim timeout",
    }, last_state


def wait_for_task_status(
    args: argparse.Namespace,
    *,
    observer_token: str,
    secret_values: list[str],
    state_probe: StateProbe,
    task_id: str,
    status: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + float(timeout_seconds)
    last_task: dict[str, Any] = {}
    last_state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            state = state_probe(args, observer_token, secret_values)
        except (OSError, URLError, json.JSONDecodeError):
            time.sleep(args.poll_interval)
            continue
        last_state = state
        task = find_task(state, task_id=task_id)
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
        time.sleep(args.poll_interval)
    return {
        "ok": False,
        "task_id": task_id,
        "expected_status": status,
        "last_status": last_task.get("status"),
        "last_miner_id": last_task.get("miner_id"),
    }, last_state


def wait_for_rescue_completion(
    args: argparse.Namespace,
    *,
    observer_token: str,
    secret_values: list[str],
    state_probe: StateProbe,
    task_id: str,
    rescue_id: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + float(timeout_seconds)
    last_task: dict[str, Any] = {}
    last_state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            state = state_probe(args, observer_token, secret_values)
        except (OSError, URLError, json.JSONDecodeError):
            time.sleep(args.poll_interval)
            continue
        last_state = state
        task = find_task(state, task_id=task_id)
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
        time.sleep(args.poll_interval)
    return {
        "ok": False,
        "task_id": task_id,
        "expected_miner_id": rescue_id,
        "last_status": last_task.get("status"),
        "last_miner_id": last_task.get("miner_id"),
        "last_attempt": last_task.get("attempt"),
    }, last_state


def build_alpha_package(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "real_llm_internet_alpha_pack.py"),
        "--mode",
        "package",
        "--output-dir",
        str(alpha_package_dir(output_dir)),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--miner-id",
        args.miner_id,
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--hf-model-id",
        args.hf_model_id,
        "--real-llm-backend",
        args.real_llm_backend,
        "--real-llm-partition-mode",
        args.real_llm_partition_mode,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--lease-seconds",
        str(args.lease_seconds),
        "--compute-seconds",
        str(args.compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        str(args.idle_sleep),
        "--max-request-attempts",
        str(args.max_request_attempts),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    return run_json_step(
        "real_llm_internet_alpha_package",
        command,
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), 300.0) + 120.0,
    )


def quote_env(value: str) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def write_miner_private_env(path: Path, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"export CROWDTENSOR_MINER_TOKEN={quote_env(token)}\n", encoding="utf-8")


def prepare_requeue_invites(
    args: argparse.Namespace,
    *,
    live_dir: Path,
    secret_values: list[str],
) -> dict[str, Any]:
    target_stage = target_stage_for_failure(args.failure_mode)
    if not target_stage:
        return {"enabled": False, "failure_mode": args.failure_mode}
    registry_path = live_dir / "remote-real-llm-runtime" / "miner_registry.json"
    invites: dict[str, Any] = {}
    for role in (ROLE_VICTIM, ROLE_RESCUE):
        miner_id = f"{args.miner_id}-{target_stage}-{role}"
        token = secrets.token_urlsafe(32)
        invite = create_invite(
            registry_path=registry_path,
            miner_id=miner_id,
            coordinator_url=effective_coordinator_url(args),
            label=f"real llm live {target_stage} {role}",
            token=token,
            replace=True,
        )
        upload = live_dir / f"kaggle-upload-real-llm-{target_stage}-{role}"
        write_miner_private_env(upload / "miner.private.env", token)
        secret_values.append(token)
        invites[role] = {
            "miner_id": miner_id,
            "env_path": str(upload / "miner.private.env"),
            "token_hash_prefix": str(invite.get("token_hash") or "").split(":", 1)[0],
        }
    return {
        "enabled": True,
        "failure_mode": args.failure_mode,
        "target_stage": target_stage,
        "victim_miner_id": invites[ROLE_VICTIM]["miner_id"],
        "rescue_miner_id": invites[ROLE_RESCUE]["miner_id"],
        "registry_path": str(registry_path),
    }


def build_kaggle_package(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    runner: Runner,
    secret_values: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    kernel_prefix = args.kernel_slug_prefix or f"crowdtensor-real-llm-beta-{args.port}"
    dataset_slug = args.dataset_slug or f"{kernel_prefix}-dataset"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "kaggle_real_llm_live_package.py"),
        "--real-llm-dir",
        str(alpha_live_dir(output_dir)),
        "--output-dir",
        str(kaggle_package_dir(output_dir)),
        "--owner",
        args.kaggle_owner,
        "--dataset-slug",
        dataset_slug,
        "--dataset-title",
        args.dataset_title,
        "--kernel-slug-prefix",
        kernel_prefix,
        "--kernel-title-prefix",
        args.kernel_title_prefix,
        "--coordinator-url",
        effective_coordinator_url(args),
        "--miner-id",
        args.miner_id,
        "--hf-model-id",
        args.hf_model_id,
        "--real-llm-backend",
        args.real_llm_backend,
        "--real-llm-partition-mode",
        args.real_llm_partition_mode,
        "--max-tasks",
        str(args.max_new_tokens),
        "--max-request-attempts",
        str(args.max_request_attempts),
        "--compute-seconds",
        str(args.compute_seconds),
        "--victim-compute-seconds",
        str(args.victim_compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        str(args.idle_sleep),
        "--failure-mode",
        args.failure_mode,
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.torch_spec:
        command.extend(["--torch-spec", args.torch_spec])
    if args.torch_index_url:
        command.extend(["--torch-index-url", args.torch_index_url])
    if args.transformers_spec:
        command.extend(["--transformers-spec", args.transformers_spec])
    if args.inline_kernel_payload:
        command.append("--inline-kernel-payload")
    return run_json_step(
        "kaggle_real_llm_live_package",
        command,
        runner=runner,
        timeout_seconds=max(60.0, args.timeout_seconds),
        secret_values=secret_values,
    )


def build_external_alpha(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    runner: Runner,
    observer_token: str,
    admin_token: str,
    secret_values: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    command = external_alpha_command(args, output_dir=output_dir, observer_token=observer_token, admin_token=admin_token)
    return run_json_step(
        "real_llm_internet_alpha_external_existing",
        command,
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds)) + 120.0,
        secret_values=secret_values,
    )


def external_alpha_command(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    observer_token: str,
    admin_token: str,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "real_llm_internet_alpha_pack.py"),
        "--mode",
        "external-existing",
        "--output-dir",
        str(external_alpha_dir(output_dir)),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--coordinator-url",
        effective_coordinator_url(args),
        "--miner-id",
        args.miner_id,
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--hf-model-id",
        args.hf_model_id,
        "--real-llm-backend",
        args.real_llm_backend,
        "--real-llm-partition-mode",
        args.real_llm_partition_mode,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--lease-seconds",
        str(args.lease_seconds),
        "--compute-seconds",
        str(args.compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        str(args.idle_sleep),
        "--max-request-attempts",
        str(args.max_request_attempts),
        "--observer-token",
        observer_token,
        "--admin-token",
        admin_token,
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    return command


def start_json_process(
    name: str,
    command: list[str],
    *,
    popen_factory: PopenFactory,
) -> tuple[subprocess.Popen[str], dict[str, Any]]:
    started = time.monotonic()
    process = popen_factory(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    return process, {
        "name": name,
        "ok": True,
        "pid": process.pid,
        "command": command,
        "started_monotonic": started,
    }


def collect_json_process(
    process: subprocess.Popen[str],
    *,
    step: dict[str, Any],
    timeout_seconds: float,
    secret_values: list[str],
    terminate: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if terminate:
        terminate_process(process, timeout=5.0)
    started = float(step.get("started_monotonic") or time.monotonic())
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        terminate_process(process, timeout=5.0)
        try:
            stdout, stderr = process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", "process did not exit after timeout"
    payload: dict[str, Any] = {}
    collected = {
        **{key: value for key, value in step.items() if key != "started_monotonic"},
        "returncode": process.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    try:
        payload = json_from_stdout(stdout or "")
    except ValueError as exc:
        collected["ok"] = False
        collected["error"] = str(exc)
    else:
        collected["payload_schema"] = payload.get("schema")
        collected["payload_ok"] = payload.get("ok")
        collected["ok"] = bool(process.returncode == 0 and payload.get("ok") is not False)
    if not collected.get("ok"):
        if stdout and not payload:
            collected["stdout_tail"] = redact_text(stdout[-1200:], secret_values)
        if stderr:
            collected["stderr_tail"] = redact_text(stderr[-1200:], secret_values)
    return collected, payload


def base_artifacts(output_dir: Path, *, ok: bool | None = None) -> dict[str, Any]:
    live_dir = alpha_live_dir(output_dir)
    external_dir = external_alpha_dir(output_dir)
    external_live = external_dir / "live-rc"
    return {
        "real_llm_internet_beta_json": artifact_entry(
            output_dir / "real_llm_internet_beta.json",
            output_dir,
            kind="real_llm_internet_beta",
            schema=SCHEMA,
            ok=ok,
        ),
        "real_llm_internet_beta_markdown": artifact_entry(
            output_dir / "real_llm_internet_beta.md",
            output_dir,
            kind="real_llm_internet_beta_markdown",
        ),
        "alpha_package_json": artifact_entry(
            alpha_package_dir(output_dir) / "real_llm_internet_alpha.json",
            output_dir,
            kind="real_llm_internet_alpha_package",
            schema="real_llm_internet_alpha_v1",
        ),
        "alpha_real_llm_live_rc_json": artifact_entry(
            live_dir / "real_llm_live_rc.json",
            output_dir,
            kind="real_llm_live_rc",
            schema="real_llm_live_rc_v1",
        ),
        "kaggle_live_package_json": artifact_entry(
            kaggle_package_dir(output_dir) / "kaggle_real_llm_live_package.json",
            output_dir,
            kind="kaggle_real_llm_live_package",
            schema="kaggle_real_llm_live_package_v1",
        ),
        "external_alpha_json": artifact_entry(
            external_dir / "real_llm_internet_alpha.json",
            output_dir,
            kind="real_llm_internet_alpha_external_existing",
            schema="real_llm_internet_alpha_v1",
        ),
        "external_real_llm_live_rc_json": artifact_entry(
            external_live / "real_llm_live_rc.json",
            output_dir,
            kind="real_llm_live_rc_external_existing",
            schema="real_llm_live_rc_v1",
        ),
        "external_remote_real_llm_sharded_beta_json": artifact_entry(
            external_live / "remote-real-llm-runtime" / "remote_real_llm_sharded_beta.json",
            output_dir,
            kind="remote_real_llm_sharded_beta",
            schema="remote_real_llm_sharded_beta_v1",
        ),
    }


def safety_summary(args: argparse.Namespace, *, cleanup_ok: bool) -> dict[str, Any]:
    gpu_backend = args.real_llm_backend == REAL_LLM_BACKEND_CUDA
    return {
        "read_only": True,
        "cpu_only_workload": not gpu_backend,
        "gpu_backend_selected": gpu_backend,
        "coordinator_cuda_runtime_required": False if gpu_backend else None,
        "miner_cuda_runtime_required": gpu_backend,
        "workload_type": WORKLOAD_TYPE,
        "stage_mode": "split",
        "request_count": args.request_count,
        "max_new_tokens": args.max_new_tokens,
        "captured_output_redacted": True,
        "summary_excludes_plaintext_tokens": True,
        "raw_activation_redacted": True,
        "temporary_http": effective_coordinator_url(args).startswith("http://"),
        "kaggle_cleanup_required": True,
        "kaggle_cleanup_ok": bool(cleanup_ok),
        "token_rotation_required": True,
        "inline_kernel_payload_private": bool(args.inline_kernel_payload),
        "inline_kernel_payload_excluded_from_report": True,
        "not_production": True,
        "not_p2p": True,
        "not_gpu_tpu_pooling": not gpu_backend,
        "not_gpu_pooling_marketplace": True,
        "not_gguf_llamacpp_serving": True,
        "not_large_model_serving": True,
        "not_public_prompt_serving": True,
    }


def render_markdown(report: dict[str, Any]) -> str:
    runtime = report.get("runtime_classification") or {}
    workload = report.get("workload") or {}
    lifecycle = report.get("kaggle_lifecycle") or {}
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    lines = [
        "# CrowdTensor Real Internet Swarm Inference Beta",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- coordinator_url: `{report.get('coordinator_url')}`",
        f"- request_count: `{workload.get('request_count')}`",
        f"- hf_model_id: `{workload.get('hf_model_id')}`",
        f"- kaggle_auto: `{runtime.get('kaggle_auto')}`",
        f"- external_runtime_verified: `{runtime.get('external_runtime_verified')}`",
        f"- kernels_deleted: `{lifecycle.get('kernels_deleted')}`",
        "",
        "## Output Scope",
        "",
        f"- include output: `{output_request.get('include_output')}`",
        f"- answer scope: `{answer_scope.get('scope_state')}`",
        f"- saved JSON display: `{answer_scope.get('saved_json_display')}`",
        f"- saved Markdown display: `{answer_scope.get('saved_markdown_display')}`",
        f"- shareable: `saved_artifacts={shareable.get('saved_artifacts_public_safe')} raw_prompt_public={shareable.get('raw_prompt_public')} raw_generated_text_public={shareable.get('raw_generated_text_public')} generated_token_ids_public={shareable.get('generated_token_ids_public')} answer_scope_state={shareable.get('answer_scope_state')} local_answer_terminal_only={shareable.get('local_answer_terminal_only')}`",
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`",
        "",
        "## Artifacts",
        "",
    ]
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        lines.append(f"- `{name}`: `{artifact.get('path')}` present=`{artifact.get('present')}`")
    lines.extend(["", "## Boundaries", ""])
    for limitation in report.get("limitations") or []:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def default_requeue_summary(args: argparse.Namespace) -> dict[str, Any]:
    target_stage = target_stage_for_failure(args.failure_mode)
    return {
        "enabled": args.failure_mode != FAILURE_NONE,
        "failure_mode": args.failure_mode,
        "target_stage": target_stage,
        "victim_miner_id": victim_miner_id(args, target_stage) if target_stage else "",
        "rescue_miner_id": rescue_miner_id(args, target_stage) if target_stage else "",
        "claim_observed": False,
        "victim_kernel_deleted": False,
        "lease_expired": False,
        "rescued_result": False,
        "victim_result_accepted": False,
    }


def persist_report(report: dict[str, Any], *, output_dir: Path, secret_values: list[str]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report.setdefault("output_request", output_request_summary())
    report.setdefault("answer_scope", answer_scope_summary())
    report.setdefault("shareable_summary", shareable_summary())
    report = support_bundle.sanitize(redact_values(report, secret_values))
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    leaks.extend(secret for secret in secret_values if secret and secret in encoded)
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"sensitive_output_detected"})
        report["safety_error"] = "real LLM Internet Beta report contained secret-like fragments"
    json_path = output_dir / "real_llm_internet_beta.json"
    md_path = output_dir / "real_llm_internet_beta.md"
    write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    if "artifacts" in report:
        report["artifacts"]["real_llm_internet_beta_json"]["present"] = True
        report["artifacts"]["real_llm_internet_beta_markdown"]["present"] = True
        write_json(json_path, report)
    return report


def build_evidence_import(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    generation_payload = load_json(Path(args.generation_report)) if args.generation_report else {}
    requeue_payload = load_json(Path(args.requeue_report)) if args.requeue_report else {}
    real_llm_backend = imported_backend_from_payload(generation_payload, args.real_llm_backend)
    real_llm_partition_mode = imported_partition_mode_from_payload(generation_payload, args.real_llm_partition_mode)
    torch_spec = imported_string_from_payload(generation_payload, "torch_spec", args.torch_spec)
    torch_index_url = imported_string_from_payload(generation_payload, "torch_index_url", args.torch_index_url)
    transformers_spec = imported_string_from_payload(generation_payload, "transformers_spec", args.transformers_spec)
    generation = generation_summary(generation_payload)
    generated_token_count = safe_int(generation.get("generated_token_count"))
    token_target_ready = generated_token_count >= int(args.max_new_tokens)
    generation_codes = set(diagnosis_codes(generation_payload))
    generation_required = {
        "external_runtime_verified",
        "decoded_tokens_match",
        "distinct_stage_miners",
        "stage_assignment_valid",
    }
    if args.max_new_tokens > 1:
        generation_required.add("multi_token_generation_ready")
    generation_ready = bool(
        generation_payload.get("ok") is True
        and generation_required.issubset(generation_codes)
        and token_target_ready
    )

    requeue_codes = set(diagnosis_codes(requeue_payload))
    requeue = safe_live_requeue_summary(requeue_payload)
    requeue_ready = bool(
        requeue_payload.get("ok") is True
        and "external_stage_requeue_ready" in requeue_codes
        and ("live_stage0_requeue_ready" in requeue_codes or "live_stage1_requeue_ready" in requeue_codes)
        and live_requeue_detail_ready(requeue)
    )
    model = model_summary([generation_payload, requeue_payload], args.hf_model_id)
    cleanup_ready = bool(
        "kaggle_kernels_deleted" in generation_codes
        and ("kaggle_kernels_deleted" in requeue_codes or requeue_payload.get("ok") is True)
    )
    ok = bool(generation_ready and requeue_ready and model["compatible"] and cleanup_ready)
    codes = set(diagnosis_codes(generation_payload, requeue_payload, extra=["token_rotation_required"]))
    if generation_ready:
        codes.update({
            "real_llm_internet_beta_generation_import_ready",
            "external_runtime_verified",
            "generation_complete",
        })
    else:
        codes.add("real_llm_internet_beta_generation_import_blocked")
    if token_target_ready:
        codes.add("external_generated_token_target_ready")
    else:
        codes.add("external_generated_token_target_missing")
    if requeue_ready:
        codes.update({
            "external_stage_requeue_ready",
            f"live_{requeue.get('target_stage')}_requeue_ready" if requeue.get("target_stage") else "live_stage_requeue_ready",
            "live_requeue_victim_claim_observed",
            "live_requeue_victim_kernel_deleted",
            "live_requeue_lease_timeout_observed",
            "live_requeue_rescue_result_accepted",
        })
    else:
        codes.add("external_stage_requeue_blocked")
    if model["compatible"]:
        codes.add("real_llm_internet_beta_model_metadata_ready")
    else:
        codes.add("real_llm_internet_beta_model_metadata_mismatch")
    if cleanup_ready:
        codes.add("kaggle_kernels_deleted")
    if ok:
        codes.update({"real_llm_internet_beta_ready", "real_llm_internet_beta_evidence_import_ready"})
    else:
        codes.add("real_llm_internet_beta_blocked")

    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "coordinator_url": args.coordinator_url,
        "public_host": args.public_host,
        "port": args.port,
        "base_port": args.base_port,
        "miner_id": args.miner_id,
        "workload": {
            "workload_type": WORKLOAD_TYPE,
            "stage_mode": "split",
            "request_count": args.request_count,
            "max_new_tokens": args.max_new_tokens,
            "hf_model_id": args.hf_model_id,
            "real_llm_backend": real_llm_backend,
            "real_llm_partition_mode": real_llm_partition_mode,
            "torch_spec": torch_spec,
            "torch_index_url": torch_index_url,
            "transformers_spec": transformers_spec,
            "prompt_text_count": len(DEFAULT_PROMPTS),
            "require_distinct_stage_miners": True,
        },
        "runtime_classification": {
            "kaggle_auto": False,
            "package_only": False,
            "external_runtime_verified": generation_ready,
            "kaggle_notebook_verified": generation_ready,
            "local_generated_stage_upload_standins": False,
            "stage_requeue_verified": requeue_ready,
            "evidence_import": True,
        },
        "payload_summaries": {
            "generation_report": summarize_payload(generation_payload),
            "requeue_report": summarize_payload(requeue_payload),
        },
        "generation": {
            **generation,
            "generated_token_count": generated_token_count,
            "required_generated_token_count": args.max_new_tokens,
            "token_target_ready": token_target_ready,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        },
        "model": model,
        "live_requeue_summary": requeue,
        "kaggle_lifecycle": {
            "owner": args.kaggle_owner,
            "kernel_slug_prefix": args.kernel_slug_prefix or f"crowdtensor-real-llm-beta-{args.port}",
            "kernels_deleted": cleanup_ready,
            "cleanup_required": True,
            "token_rotation_required": True,
        },
        "diagnosis_codes": sorted(set(codes)),
        "artifacts": {
            **base_artifacts(output_dir, ok=ok),
            "generation_report": artifact_entry(
                Path(args.generation_report),
                output_dir,
                kind="real_llm_internet_beta_generation_source",
                schema=str(generation_payload.get("schema") or ""),
                ok=generation_payload.get("ok") if generation_payload else None,
            ),
            "requeue_report": artifact_entry(
                Path(args.requeue_report),
                output_dir,
                kind="real_llm_internet_beta_requeue_source",
                schema=str(requeue_payload.get("schema") or ""),
                ok=requeue_payload.get("ok") if requeue_payload else None,
            ),
        },
        "source_reports": {
            "generation_report": str(Path(args.generation_report).resolve()) if args.generation_report else "",
            "requeue_report": str(Path(args.requeue_report).resolve()) if args.requeue_report else "",
        },
        "safety": safety_summary_for_backend(args, cleanup_ok=cleanup_ready, real_llm_backend=real_llm_backend),
        "operator_action": [
            "Import only retained external reports that expose safe generated_token_count evidence.",
            "Rotate generated Coordinator and Miner tokens after every temporary public HTTP run.",
            "Treat this as read-only Beta evidence, not production public serving.",
        ],
        "limitations": [
            "evidence-import combines retained external generation and requeue reports; it does not create a fresh Kaggle run.",
            "Read-only tiny Hugging Face GPT split proof; optional CUDA backend is not production Swarm Inference.",
            "No NAT traversal, GPU pooling marketplace, GGUF/llama.cpp serving, large-model serving, training, payments, or arbitrary prompt serving.",
        ],
        "not_completed": [] if ok else [
            item for item, ready in [
                ("external generation report", generation_ready),
                ("external generated token target", token_target_ready),
                ("external requeue report", requeue_ready),
                ("external model metadata", model["compatible"]),
                ("Kaggle cleanup evidence", cleanup_ready),
            ]
            if not ready
        ],
    }
    return persist_report(report, output_dir=output_dir, secret_values=[])


def build_report(
    args: argparse.Namespace,
    *,
    runner: Runner = subprocess.run,
    popen_factory: PopenFactory = subprocess.Popen,
    ready_probe: ReadyProbe = wait_for_ready,
    state_probe: StateProbe = fetch_state,
) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == MODE_EVIDENCE_IMPORT:
        return build_evidence_import(args, output_dir=output_dir)

    steps: list[dict[str, Any]] = []
    alpha_payload: dict[str, Any] = {}
    kaggle_payload: dict[str, Any] = {}
    external_payload: dict[str, Any] = {}
    secret_values: list[str] = []
    coordinator_process: subprocess.Popen[str] | None = None
    coordinator_lifecycle: dict[str, Any] = {}
    push_steps: list[dict[str, Any]] = []
    status_steps: list[dict[str, Any]] = []
    cleanup_steps: list[dict[str, Any]] = []
    output_steps: list[dict[str, Any]] = []
    pushed_refs: dict[str, str] = {}
    stage_refs: dict[str, str] = {}
    kernel_entries: list[dict[str, Any]] = []
    deleted_refs: set[str] = set()
    requeue_summary = default_requeue_summary(args)
    requeue_invites: dict[str, Any] = {}
    external_process: subprocess.Popen[str] | None = None
    external_process_step: dict[str, Any] = {}

    try:
        alpha_step, alpha_payload = build_alpha_package(args, output_dir=output_dir, runner=runner)
        steps.append(alpha_step)
        live_dir = alpha_live_dir(output_dir)
        secret_values = secret_values_from_envs(
            live_dir / "remote-real-llm-runtime" / "operator.private.env",
            live_dir / "kaggle-upload-real-llm-stage0" / "miner.private.env",
            live_dir / "kaggle-upload-real-llm-stage1" / "miner.private.env",
        )
        if args.failure_mode != FAILURE_NONE and alpha_step.get("ok") and alpha_payload.get("ok"):
            requeue_invites = prepare_requeue_invites(args, live_dir=live_dir, secret_values=secret_values)
        if alpha_step.get("ok") and alpha_payload.get("ok"):
            kaggle_step, kaggle_payload = build_kaggle_package(
                args,
                output_dir=output_dir,
                runner=runner,
                secret_values=secret_values,
            )
            steps.append(kaggle_step)
            stage_refs = stage_refs_from_package(kaggle_payload)
            kernel_entries = kernel_entries_from_package(kaggle_payload)

        expected_keys = [entry["key"] for entry in kernel_entries]
        if kaggle_payload.get("ok") and expected_keys and all(key in stage_refs for key in expected_keys):
            coordinator_command = ["bash", str(live_dir / "start_coordinator.sh")]
            try:
                coordinator_process = popen_factory(
                    coordinator_command,
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                coordinator_lifecycle = {
                    "ok": True,
                    "pid": coordinator_process.pid,
                    "command": coordinator_command,
                    "ready_url": local_ready_url(args),
                }
            except OSError as exc:
                coordinator_lifecycle = {
                    "ok": False,
                    "command": coordinator_command,
                    "error": str(exc),
                    "ready_url": local_ready_url(args),
                }

            if coordinator_process is not None:
                ready = ready_probe(local_ready_url(args), args.startup_timeout, args.poll_interval)
                coordinator_lifecycle["ready"] = redact_values(ready, secret_values)
                coordinator_lifecycle["ok"] = bool(coordinator_lifecycle.get("ok") and ready.get("ok"))

        if coordinator_lifecycle.get("ok"):
            for entry in initial_push_entries(kernel_entries, args.failure_mode):
                key = str(entry["key"])
                ref = stage_refs[key]
                kernel_dir = kaggle_package_dir(output_dir) / "kernels" / key
                push_step = run_step(
                    f"kaggle_push_{key}",
                    ["kaggle", "kernels", "push", "-p", str(kernel_dir)],
                    runner=runner,
                    timeout_seconds=args.kaggle_push_timeout_seconds,
                    secret_values=secret_values,
                )
                push_step["stage"] = entry["stage"]
                push_step["role"] = entry["role"]
                push_step["key"] = key
                push_step["kernel_ref"] = ref
                actual_ref = extract_kernel_ref(
                    f"{push_step.get('stdout_tail', '')}\n{push_step.get('stderr_tail', '')}"
                )
                if actual_ref:
                    push_step["actual_kernel_ref"] = actual_ref
                push_steps.append(push_step)
                if push_step.get("ok"):
                    pushed_refs[key] = actual_ref or ref

        operator_env = parse_private_env(alpha_live_dir(output_dir) / "remote-real-llm-runtime" / "operator.private.env")
        observer_token = operator_env.get("CROWDTENSOR_OBSERVER_TOKEN", "")
        admin_token = operator_env.get("CROWDTENSOR_ADMIN_TOKEN", "")
        initial_entries = initial_push_entries(kernel_entries, args.failure_mode)
        if all(step.get("ok") for step in push_steps) and len(push_steps) == len(initial_entries):
            if observer_token and admin_token:
                if args.failure_mode == FAILURE_NONE:
                    external_step, external_payload = build_external_alpha(
                        args,
                        output_dir=output_dir,
                        runner=runner,
                        observer_token=observer_token,
                        admin_token=admin_token,
                        secret_values=secret_values,
                    )
                    steps.append(external_step)
                else:
                    command = external_alpha_command(
                        args,
                        output_dir=output_dir,
                        observer_token=observer_token,
                        admin_token=admin_token,
                    )
                    try:
                        external_process, external_process_step = start_json_process(
                            "real_llm_internet_alpha_external_existing",
                            command,
                            popen_factory=popen_factory,
                        )
                    except OSError as exc:
                        steps.append({
                            "name": "real_llm_internet_alpha_external_existing",
                            "ok": False,
                            "error": str(exc),
                        })
                        external_process = None
                    if external_process is not None:
                        target_stage = target_stage_for_failure(args.failure_mode)
                        victim_id = victim_miner_id(args, target_stage)
                        rescue_id = rescue_miner_id(args, target_stage)
                        claim, _ = wait_for_live_claim(
                            args,
                            observer_token=observer_token,
                            secret_values=secret_values,
                            state_probe=state_probe,
                            target_stage=target_stage,
                            target_miner_id=victim_id,
                        )
                        requeue_summary["claim"] = claim
                        requeue_summary["claim_observed"] = bool(claim.get("ok"))
                        requeue_summary["victim_task_id"] = str(claim.get("task_id") or "")
                        if claim.get("ok"):
                            victim_key = kernel_key(target_stage, ROLE_VICTIM)
                            victim_ref = pushed_refs.get(victim_key, stage_refs.get(victim_key, ""))
                            delete_step = run_step(
                                f"kaggle_delete_{victim_key}",
                                ["kaggle", "kernels", "delete", victim_ref, "-y"],
                                runner=runner,
                                timeout_seconds=args.kaggle_delete_timeout_seconds,
                                secret_values=secret_values,
                            )
                            delete_step["stage"] = target_stage
                            delete_step["role"] = ROLE_VICTIM
                            delete_step["key"] = victim_key
                            delete_step["kernel_ref"] = victim_ref
                            cleanup_steps.append(delete_step)
                            if delete_step.get("ok") and victim_ref:
                                deleted_refs.add(victim_ref)
                            requeue_summary["victim_kernel_deleted"] = bool(delete_step.get("ok"))
                            queued, _ = wait_for_task_status(
                                args,
                                observer_token=observer_token,
                                secret_values=secret_values,
                                state_probe=state_probe,
                                task_id=str(claim.get("task_id") or ""),
                                status="queued",
                                timeout_seconds=args.requeue_timeout,
                            )
                            requeue_summary["requeue_observation"] = queued
                            requeue_summary["lease_expired"] = bool(queued.get("ok"))
                            if queued.get("ok"):
                                rescue = rescue_entry(kernel_entries, args.failure_mode)
                                rescue_key = str(rescue.get("key") or kernel_key(target_stage, ROLE_RESCUE))
                                rescue_ref = stage_refs.get(rescue_key, "")
                                rescue_dir = kaggle_package_dir(output_dir) / "kernels" / rescue_key
                                rescue_push = run_step(
                                    f"kaggle_push_{rescue_key}",
                                    ["kaggle", "kernels", "push", "-p", str(rescue_dir)],
                                    runner=runner,
                                    timeout_seconds=args.kaggle_push_timeout_seconds,
                                    secret_values=secret_values,
                                )
                                rescue_push["stage"] = target_stage
                                rescue_push["role"] = ROLE_RESCUE
                                rescue_push["key"] = rescue_key
                                rescue_push["kernel_ref"] = rescue_ref
                                actual_ref = extract_kernel_ref(
                                    f"{rescue_push.get('stdout_tail', '')}\n{rescue_push.get('stderr_tail', '')}"
                                )
                                if actual_ref:
                                    rescue_push["actual_kernel_ref"] = actual_ref
                                push_steps.append(rescue_push)
                                if rescue_push.get("ok"):
                                    pushed_refs[rescue_key] = actual_ref or rescue_ref
                        external_step, external_payload = collect_json_process(
                            external_process,
                            step=external_process_step,
                            timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds)) + 120.0,
                            secret_values=secret_values,
                        )
                        steps.append(external_step)
                        external_process = None
                        if requeue_summary.get("victim_task_id"):
                            completed, _ = wait_for_rescue_completion(
                                args,
                                observer_token=observer_token,
                                secret_values=secret_values,
                                state_probe=state_probe,
                                task_id=str(requeue_summary["victim_task_id"]),
                                rescue_id=rescue_id,
                                timeout_seconds=max(5.0, args.poll_interval * 2.0),
                            )
                            requeue_summary["rescue_observation"] = completed
                            requeue_summary["rescued_result"] = bool(completed.get("ok"))
                            victim_state = find_task(
                                state_probe(args, observer_token, secret_values),
                                task_id=str(requeue_summary["victim_task_id"]),
                            ) if observer_token else {}
                            requeue_summary["victim_result_accepted"] = bool(
                                victim_state.get("status") == "completed"
                                and victim_state.get("miner_id") == victim_id
                            )
            else:
                steps.append({
                    "name": "real_llm_internet_alpha_external_existing",
                    "ok": False,
                    "error": "missing operator observer/admin tokens",
                })

            for key, ref in pushed_refs.items():
                status_steps.append(poll_kaggle_status(
                    stage=key,
                    ref=ref,
                    runner=runner,
                timeout_seconds=args.kaggle_status_timeout_seconds,
                poll_interval=args.kaggle_status_poll_interval,
                    secret_values=secret_values,
                ))
    finally:
        if external_process is not None:
            steps.append(collect_json_process(
                external_process,
                step=external_process_step or {"name": "real_llm_internet_alpha_external_existing", "ok": False},
                timeout_seconds=5.0,
                secret_values=secret_values,
                terminate=True,
            )[0])
        if args.skip_kaggle_cleanup:
            cleanup_steps.append({"name": "kaggle_cleanup", "ok": False, "skipped": True, "error": "cleanup skipped"})
        else:
            for key, ref in pushed_refs.items():
                output_steps.append(collect_kaggle_output(
                    key=key,
                    ref=ref,
                    output_dir=output_dir,
                    runner=runner,
                    timeout_seconds=min(120.0, max(10.0, float(args.kaggle_delete_timeout_seconds))),
                    secret_values=secret_values,
                ))
            for key, ref in pushed_refs.items():
                if ref in deleted_refs:
                    cleanup_steps.append({
                        "name": f"kaggle_delete_{key}",
                        "ok": True,
                        "already_deleted": True,
                        "key": key,
                        "stage": stage_from_key(key),
                        "kernel_ref": ref,
                    })
                    continue
                cleanup_step = run_step(
                    f"kaggle_delete_{key}",
                    ["kaggle", "kernels", "delete", ref, "-y"],
                    runner=runner,
                    timeout_seconds=args.kaggle_delete_timeout_seconds,
                    secret_values=secret_values,
                )
                cleanup_step["stage"] = stage_from_key(key)
                cleanup_step["role"] = kernel_role_from_key(key)
                cleanup_step["key"] = key
                cleanup_step["kernel_ref"] = ref
                cleanup_steps.append(cleanup_step)
        if coordinator_process is not None:
            coordinator_lifecycle["process"] = collect_process(
                coordinator_process,
                name="coordinator",
                command=["bash", str(alpha_live_dir(output_dir) / "start_coordinator.sh")],
                timeout=args.process_exit_timeout,
                secret_values=secret_values,
                terminate=True,
            )

    codes = diagnosis_codes(alpha_payload, kaggle_payload, external_payload, extra=["token_rotation_required"])
    if args.skip_kaggle_cleanup:
        codes.append("kaggle_cleanup_skipped")
    cleanup_ok = bool(pushed_refs) and all(step.get("ok") for step in cleanup_steps)
    if cleanup_ok:
        codes.append("kaggle_kernels_deleted")
    elif pushed_refs or args.skip_kaggle_cleanup:
        codes.append("kaggle_cleanup_failed")

    required_external_codes = {
        "real_llm_internet_alpha_ready",
        "external_runtime_verified",
        "kaggle_real_llm_stage0_seen",
        "kaggle_real_llm_stage1_seen",
        "decoded_tokens_match",
        "distinct_stage_miners",
        "stage_assignment_valid",
    }
    if args.max_new_tokens > 1:
        required_external_codes.add("multi_token_generation_ready")
    if args.real_llm_partition_mode == "stage_local":
        required_external_codes.update({
            "stage_local_partition_ready",
            "stage0_partition_loaded",
            "stage1_partition_loaded",
            "partition_parameter_split_valid",
        })
    external_codes = set(diagnosis_codes(external_payload))
    external_verified = bool(external_payload.get("ok") and required_external_codes.issubset(external_codes))
    requeue_verified = bool(
        requeue_summary.get("enabled")
        and requeue_summary.get("claim_observed")
        and requeue_summary.get("victim_kernel_deleted")
        and requeue_summary.get("lease_expired")
        and requeue_summary.get("rescued_result")
        and not requeue_summary.get("victim_result_accepted")
    )
    if requeue_summary.get("enabled"):
        if requeue_summary.get("claim_observed"):
            codes.append("live_requeue_victim_claim_observed")
        if requeue_summary.get("victim_kernel_deleted"):
            codes.append("live_requeue_victim_kernel_deleted")
        if requeue_summary.get("lease_expired"):
            codes.append("live_requeue_lease_timeout_observed")
        if requeue_summary.get("rescued_result"):
            codes.append("live_requeue_rescue_result_accepted")
        if requeue_verified:
            codes.extend([
                "external_stage_requeue_ready",
                f"live_{requeue_summary.get('target_stage')}_requeue_ready",
            ])
        else:
            codes.append("external_stage_requeue_blocked")
    coordinator_ready = bool(coordinator_lifecycle.get("ok"))
    expected_push_count = len(kernel_entries) if args.failure_mode != FAILURE_NONE else len(STAGES)
    all_pushed = len(pushed_refs) == expected_push_count and all(step.get("ok") for step in push_steps)
    coordinator_stopped = bool((coordinator_lifecycle.get("process") or {}).get("ok")) if coordinator_process is not None else False
    ok = bool(
        alpha_payload.get("ok")
        and kaggle_payload.get("ok")
        and coordinator_ready
        and all_pushed
        and external_verified
        and (not requeue_summary.get("enabled") or requeue_verified)
        and cleanup_ok
        and coordinator_stopped
    )
    if ok:
        codes.extend(["real_llm_internet_beta_ready", "kaggle_auto_ready"])
        if args.real_llm_backend == REAL_LLM_BACKEND_CUDA:
            codes.extend([
                "public_swarm_gpu_beta_ready",
                "gpu_runtime_ready",
                "gpu_stage0_ready",
                "gpu_stage1_ready",
            ])
    else:
        codes.append("real_llm_internet_beta_blocked")

    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "coordinator_url": effective_coordinator_url(args),
        "public_host": args.public_host,
        "port": args.port,
        "base_port": args.base_port,
        "miner_id": args.miner_id,
        "workload": {
            "workload_type": WORKLOAD_TYPE,
            "stage_mode": "split",
            "request_count": args.request_count,
            "max_new_tokens": args.max_new_tokens,
            "hf_model_id": args.hf_model_id,
            "real_llm_backend": args.real_llm_backend,
            "real_llm_partition_mode": args.real_llm_partition_mode,
            "torch_spec": args.torch_spec,
            "torch_index_url": args.torch_index_url,
            "transformers_spec": args.transformers_spec,
            "prompt_text_count": len(DEFAULT_PROMPTS),
            "require_distinct_stage_miners": True,
        },
        "runtime_classification": {
            "kaggle_auto": True,
            "package_only": False,
            "external_runtime_verified": external_verified,
            "kaggle_notebook_verified": external_verified,
            "local_generated_stage_upload_standins": False,
            "stage_requeue_verified": requeue_verified,
        },
        "steps": steps,
        "payload_summaries": {
            "alpha_package": summarize_payload(alpha_payload),
            "kaggle_package": summarize_payload(kaggle_payload),
            "external_alpha": summarize_payload(external_payload),
        },
        "live_requeue_summary": requeue_summary,
        "live_requeue_invites": {
            "enabled": bool(requeue_invites.get("enabled")),
            "failure_mode": requeue_invites.get("failure_mode"),
            "target_stage": requeue_invites.get("target_stage"),
            "victim_miner_id": requeue_invites.get("victim_miner_id"),
            "rescue_miner_id": requeue_invites.get("rescue_miner_id"),
        },
        "kaggle_lifecycle": {
            "owner": args.kaggle_owner,
            "kernel_slug_prefix": args.kernel_slug_prefix or f"crowdtensor-real-llm-beta-{args.port}",
            "stage_refs": stage_refs,
            "pushed_refs": pushed_refs,
            "expected_push_count": expected_push_count,
            "push_steps": push_steps,
            "status_steps": status_steps,
            "output_steps": output_steps,
            "cleanup_steps": cleanup_steps,
            "kernels_deleted": cleanup_ok,
            "cleanup_required": True,
        },
        "coordinator_lifecycle": coordinator_lifecycle,
        "diagnosis_codes": sorted(set(codes)),
        "artifacts": base_artifacts(output_dir, ok=ok),
        "safety": safety_summary(args, cleanup_ok=cleanup_ok),
        "operator_action": [
            "Rotate generated Coordinator and Miner tokens after every temporary public HTTP run.",
            "If cleanup failed, delete temporary Kaggle kernels manually before retrying.",
            "Treat this as read-only Beta evidence, not production public serving.",
        ],
        "limitations": [
            "kaggle-auto creates temporary private Kaggle script kernels for two real stage Miners; it is still not P2P routing.",
            "Read-only tiny Hugging Face GPT split proof; optional CUDA backend is not production Swarm Inference.",
            "No NAT traversal, GPU pooling marketplace, GGUF/llama.cpp serving, large-model serving, training, payments, or arbitrary prompt serving.",
        ],
    }
    return persist_report(report, output_dir=output_dir, secret_values=secret_values)


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real Internet Swarm Inference Beta Kaggle automation.")
    parser.add_argument("--mode", choices=MODES, default=MODE_KAGGLE_AUTO)
    parser.add_argument("--output-dir", default="dist/real-llm-internet-beta-kaggle-auto")
    parser.add_argument("--generation-report", default="")
    parser.add_argument("--requeue-report", default="")
    parser.add_argument("--public-host", default=DEFAULT_PUBLIC_HOST)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT)
    parser.add_argument("--ready-url", default="")
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--miner-id", default="internet-real-llm-beta")
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--hf-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--real-llm-backend", choices=["hf_transformers_cpu", "hf_transformers_cuda", "cpu", "cuda", "auto"], default=REAL_LLM_BACKEND_CPU)
    parser.add_argument("--real-llm-partition-mode", choices=["full", "stage-local", "stage_local"], default="full")
    parser.add_argument("--torch-spec", default="")
    parser.add_argument("--torch-index-url", default="")
    parser.add_argument("--transformers-spec", default=DEFAULT_TRANSFORMERS_SPEC)
    parser.add_argument("--kaggle-owner", default=default_kaggle_owner())
    parser.add_argument("--dataset-slug", default="")
    parser.add_argument("--dataset-title", default="CrowdTensor Real LLM Internet Beta Package")
    parser.add_argument("--kernel-slug-prefix", default="")
    parser.add_argument("--kernel-title-prefix", default="CrowdTensor Real LLM Internet Beta Miner")
    parser.add_argument("--inline-kernel-payload", dest="inline_kernel_payload", action="store_true", default=True)
    parser.add_argument("--no-inline-kernel-payload", dest="inline_kernel_payload", action="store_false")
    parser.add_argument("--skip-kaggle-cleanup", action="store_true")
    parser.add_argument("--failure-mode", choices=sorted(FAILURE_MODES), default=FAILURE_NONE)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--remote-timeout-seconds", type=float, default=720.0)
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--process-exit-timeout", type=float, default=20.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--http-timeout", type=float, default=30.0)
    parser.add_argument("--kaggle-push-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--kaggle-status-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--kaggle-status-poll-interval", type=float, default=15.0)
    parser.add_argument("--lease-seconds", type=float, default=15.0)
    parser.add_argument("--compute-seconds", type=float, default=0.2)
    parser.add_argument("--victim-compute-seconds", type=float, default=45.0)
    parser.add_argument("--heartbeat-interval", type=float, default=0.1)
    parser.add_argument("--idle-sleep", type=float, default=0.5)
    parser.add_argument("--claim-observe-timeout", type=float, default=180.0)
    parser.add_argument("--requeue-timeout", type=float, default=120.0)
    parser.add_argument("--max-request-attempts", type=int, default=240)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.port < 1:
        raise SystemExit("--port must be positive")
    if args.base_port < 1:
        raise SystemExit("--base-port must be positive")
    if args.request_count < 1 or args.request_count > 4:
        raise SystemExit("--request-count must be between 1 and 4")
    if args.max_new_tokens < 1 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 1 and 32")
    args.real_llm_backend = normalize_real_llm_backend(args.real_llm_backend)
    args.real_llm_partition_mode = normalize_real_llm_partition_mode(args.real_llm_partition_mode)
    if args.real_llm_backend == REAL_LLM_BACKEND_CUDA and not args.torch_spec:
        args.torch_spec = DEFAULT_CUDA_TORCH_RUNTIME_SPEC
        args.torch_index_url = args.torch_index_url or DEFAULT_CUDA_TORCH_INDEX_URL
    if args.mode == MODE_EVIDENCE_IMPORT:
        if not args.generation_report:
            raise SystemExit("--generation-report is required in evidence-import mode")
        if not args.requeue_report:
            raise SystemExit("--requeue-report is required in evidence-import mode")
        args.kaggle_owner = args.kaggle_owner or "retained-evidence"
    if not args.kaggle_owner:
        raise SystemExit("--kaggle-owner is required or KAGGLE_USERNAME/~/.kaggle/kaggle.json must be configured")
    for name in [
        "timeout_seconds",
        "remote_timeout_seconds",
        "startup_timeout",
        "process_exit_timeout",
        "poll_interval",
        "http_timeout",
        "kaggle_push_timeout_seconds",
        "kaggle_delete_timeout_seconds",
        "kaggle_status_timeout_seconds",
        "kaggle_status_poll_interval",
        "lease_seconds",
        "victim_compute_seconds",
        "heartbeat_interval",
        "idle_sleep",
        "claim_observe_timeout",
        "requeue_timeout",
    ]:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.compute_seconds < 0:
        raise SystemExit("--compute-seconds must be non-negative")
    if args.failure_mode != FAILURE_NONE and args.victim_compute_seconds <= args.lease_seconds:
        args.victim_compute_seconds = args.lease_seconds + 30.0
    if args.failure_mode != FAILURE_NONE and args.requeue_timeout <= args.lease_seconds:
        args.requeue_timeout = args.lease_seconds + 45.0
    if args.max_request_attempts < 1:
        raise SystemExit("--max-request-attempts must be at least 1")
    args.coordinator_url = args.coordinator_url.rstrip("/") if args.coordinator_url else ""
    return args


def print_human(report: dict[str, Any]) -> None:
    print("CrowdTensor real Internet Swarm Inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  coordinator: {report.get('coordinator_url')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    runtime = report.get("runtime_classification") or {}
    lifecycle = report.get("kaggle_lifecycle") or {}
    print(f"  kaggle auto: {runtime.get('kaggle_auto')}")
    print(f"  external runtime: {runtime.get('external_runtime_verified')}")
    print(f"  kernels deleted: {lifecycle.get('kernels_deleted')}")


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print_human(report)
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
