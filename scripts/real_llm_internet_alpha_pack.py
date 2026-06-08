#!/usr/bin/env python3
"""Build the real Internet Swarm Inference Alpha evidence report."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import support_bundle  # noqa: E402
from crowdtensor.real_llm import BACKEND_CPU as REAL_LLM_BACKEND_CPU  # noqa: E402
from crowdtensor.real_llm import BACKEND_CUDA as REAL_LLM_BACKEND_CUDA  # noqa: E402
from crowdtensor.real_llm import DEFAULT_MODEL_ID, DEFAULT_PROMPTS  # noqa: E402
from crowdtensor.real_llm import normalize_backend as normalize_real_llm_backend  # noqa: E402
from crowdtensor.real_llm import normalize_partition_mode as normalize_real_llm_partition_mode  # noqa: E402


SCHEMA = "real_llm_internet_alpha_v1"
MODE_LOCAL_GENERATED = "local-generated"
MODE_PACKAGE = "package"
MODE_EXTERNAL_EXISTING = "external-existing"
DEFAULT_PUBLIC_HOST = "24.199.118.54"
DEFAULT_PORT = 9186
DEFAULT_REQUEUE_BASE_PORT = 9188
WORKLOAD_TYPE = "real_llm_sharded_infer"
Runner = Callable[..., subprocess.CompletedProcess[str]]

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
    '"generated_text":',
    '"generated_token_ids":',
    '"prompt_text":',
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


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("command emitted no JSON object")


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


def output_request_summary() -> dict[str, Any]:
    return {
        "include_output": False,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Real Internet Swarm Inference Alpha artifacts summarize tiny GPT split "
            "readiness, stage requeue evidence, generation hashes/counts, and "
            "diagnostics only. They do not include answer text."
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
            "This Alpha report is shareable operator evidence, not a local answer "
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
            "Share real_llm_internet_alpha*.json/md artifacts; they contain "
            "readiness evidence, generation hashes/counts, and diagnostics, not "
            "raw prompts or answers."
        ),
    }


def prompt_scope_summary(report: dict[str, Any] | None = None) -> dict[str, Any]:
    workload = report.get("workload") if isinstance(report, dict) and isinstance(report.get("workload"), dict) else {}
    try:
        prompt_count = int(workload.get("prompt_text_count") or len(DEFAULT_PROMPTS))
    except (TypeError, ValueError):
        prompt_count = len(DEFAULT_PROMPTS)
    return {
        "source": "built-in-default-prompts",
        "prompt_count": prompt_count,
        "inline_prompt_text": False,
        "terminal_next_commands_local_private": False,
        "terminal_logs_local_private": False,
        "saved_artifacts_prompt_placeholders": True,
        "saved_artifacts_public_safe": True,
        "prefer_prompt_file_or_stdin_for_shareable_logs": False,
        "prompt_file_path_public": False,
        "raw_prompt_public": False,
        "public_artifact_safe": True,
        "summary": (
            "Real Internet Swarm Inference Alpha uses fixed built-in validation prompts; "
            "public artifacts record prompt count/source only and exclude raw prompt text."
        ),
    }


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


def shell_command(parts: list[Any]) -> str:
    rendered: list[str] = []
    for part in parts:
        text = str(part)
        if not text:
            continue
        if text.startswith("$") and text[1:].replace("_", "").isalnum():
            rendered.append(text)
        else:
            rendered.append(shlex.quote(text))
    return " ".join(rendered)


def command_entry(
    label: str,
    command: list[Any],
    *,
    reason: str = "",
    requires_private_credentials: bool = False,
    side_effectful: bool = False,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "label": label,
        "command": [str(part) for part in command],
        "command_line": shell_command(command),
        "public_artifact_safe": True,
    }
    if reason:
        entry["reason"] = reason
    if requires_private_credentials:
        entry["requires_private_credentials"] = True
        entry["credential_note"] = (
            "Use local private Coordinator credentials when running this command; "
            "credential values are intentionally excluded from public artifacts."
        )
    if side_effectful:
        entry["side_effectful"] = True
    return entry


def artifact_command(output_dir: Path, filename: str, *, lines: str = "1,220p") -> list[str]:
    return ["sed", "-n", lines, str(output_dir / filename)]


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
    summary = {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "diagnosis_codes": diagnosis_codes(payload),
    }
    generation = generation_summary(payload)
    if generation:
        summary["generation"] = generation
    runtime = payload.get("runtime_classification") if isinstance(payload.get("runtime_classification"), dict) else {}
    if runtime:
        summary["runtime_classification"] = {
            "local_generated_stage_upload_standins": runtime.get("local_generated_stage_upload_standins"),
            "external_runtime_verified": runtime.get("external_runtime_verified"),
            "preparation_only": runtime.get("preparation_only"),
            "kaggle_notebook_verified": runtime.get("kaggle_notebook_verified"),
        }
    if payload.get("hf_model_id") is not None:
        summary["hf_model_id"] = payload.get("hf_model_id")
    if isinstance(payload.get("workload"), dict):
        workload = payload["workload"]
        summary["workload"] = {
            "workload_type": workload.get("workload_type"),
            "stage_mode": workload.get("stage_mode"),
            "request_count": workload.get("request_count"),
            "max_new_tokens": workload.get("max_new_tokens"),
            "hf_model_id": workload.get("hf_model_id"),
            "real_llm_partition_mode": workload.get("real_llm_partition_mode"),
        }
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


def effective_coordinator_url(args: argparse.Namespace) -> str:
    if args.coordinator_url:
        return args.coordinator_url.rstrip("/")
    if args.mode == MODE_LOCAL_GENERATED:
        return f"http://127.0.0.1:{args.port}"
    return f"http://{args.public_host}:{args.port}"


def live_child_mode(mode: str) -> str:
    if mode == MODE_PACKAGE:
        return "kaggle-generated"
    return mode


def run_live_rc(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    runner: Runner,
    secret_values: list[str],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    child_dir = output_dir / ("package-live-rc" if args.mode == MODE_PACKAGE else "live-rc")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "real_llm_live_rc_pack.py"),
        "--mode",
        live_child_mode(args.mode),
        "--output-dir",
        str(child_dir),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--port",
        str(args.port),
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
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    step, payload = run_json_step(
        f"real_llm_live_rc_{live_child_mode(args.mode)}",
        command,
        runner=runner,
        timeout_seconds=max(args.timeout_seconds, 1.0),
        secret_values=secret_values,
    )
    return step, payload, child_dir


def run_requeue_beta(
    args: argparse.Namespace,
    *,
    failure_mode: str,
    base_port: int,
    output_dir: Path,
    runner: Runner,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    child_dir = output_dir / f"requeue-{failure_mode}"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "remote_real_llm_sharded_beta_pack.py"),
        "--mode",
        "remote-loopback",
        "--output-dir",
        str(child_dir),
        "--json-out",
        str(child_dir / "remote_real_llm_sharded_beta.json"),
        "--markdown-out",
        str(child_dir / "remote_real_llm_sharded_beta.md"),
        "--base-port",
        str(base_port),
        "--request-count",
        "1",
        "--failure-mode",
        failure_mode,
        "--stage-mode",
        "split",
        "--require-distinct-stage-miners",
        "--hf-model-id",
        args.hf_model_id,
        "--real-llm-backend",
        args.real_llm_backend,
        "--real-llm-partition-mode",
        args.real_llm_partition_mode,
        "--timeout-seconds",
        str(max(int(args.timeout_seconds), 240)),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--prompt-texts",
        ",".join(DEFAULT_PROMPTS),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    step, payload = run_json_step(
        f"real_llm_{failure_mode}_requeue",
        command,
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), 240.0) + 30.0,
    )
    return step, payload, child_dir


def base_artifacts(output_dir: Path, *, ok: bool | None = None) -> dict[str, Any]:
    return {
        "real_llm_internet_alpha_json": artifact_entry(
            output_dir / "real_llm_internet_alpha.json",
            output_dir,
            kind="real_llm_internet_alpha",
            schema=SCHEMA,
            ok=ok,
        ),
        "real_llm_internet_alpha_markdown": artifact_entry(
            output_dir / "real_llm_internet_alpha.md",
            output_dir,
            kind="real_llm_internet_alpha_markdown",
        ),
    }


def artifact_summary(output_dir: Path) -> dict[str, Any]:
    paths = {
        "inspect_first": output_dir / "real_llm_internet_alpha.md",
        "summary_json": output_dir / "real_llm_internet_alpha.json",
        "summary_markdown": output_dir / "real_llm_internet_alpha.md",
        "support_bundle": output_dir / "support_bundle.json",
    }
    present = sum(1 for path in paths.values() if path.is_file())
    return {
        **{name: str(path) for name, path in paths.items()},
        "artifact_count": len(paths),
        "present_artifact_count": present,
        "shareable_paths": [
            str(paths["summary_json"]),
            str(paths["summary_markdown"]),
            str(paths["support_bundle"]),
        ],
        "public_artifact_safe": True,
    }


def child_artifacts(output_dir: Path, live_dir: Path, requeue_dirs: dict[str, Path], *, ok: bool | None = None) -> dict[str, Any]:
    artifacts = {
        "real_llm_live_rc_json": artifact_entry(
            live_dir / "real_llm_live_rc.json",
            output_dir,
            kind="real_llm_live_rc",
            schema="real_llm_live_rc_v1",
        ),
        "real_llm_live_rc_markdown": artifact_entry(
            live_dir / "real_llm_live_rc.md",
            output_dir,
            kind="real_llm_live_rc_markdown",
        ),
    }
    for failure_mode, child_dir in requeue_dirs.items():
        suffix = failure_mode.replace("-", "_")
        artifacts[f"{suffix}_remote_real_llm_sharded_beta_json"] = artifact_entry(
            child_dir / "remote_real_llm_sharded_beta.json",
            output_dir,
            kind="remote_real_llm_sharded_beta",
            schema="remote_real_llm_sharded_beta_v1",
            ok=ok,
        )
        artifacts[f"{suffix}_real_llm_sharded_evidence_json"] = artifact_entry(
            child_dir / "remote-loopback-real-llm-shard-infer" / "real_llm_sharded_evidence.json",
            output_dir,
            kind="real_llm_sharded_evidence",
            schema="real_llm_sharded_evidence_v1",
        )
    return artifacts


def safety_summary(args: argparse.Namespace, *, stage_requeue_verified: bool) -> dict[str, Any]:
    gpu_backend = args.real_llm_backend == REAL_LLM_BACKEND_CUDA
    return {
        "read_only": True,
        "cpu_only_workload": not gpu_backend,
        "gpu_backend_selected": gpu_backend,
        "coordinator_cuda_runtime_required": False if gpu_backend else None,
        "miner_cuda_runtime_required": gpu_backend,
        "workload_type": WORKLOAD_TYPE,
        "stage_mode": "split",
        "real_llm_partition_mode": args.real_llm_partition_mode,
        "require_distinct_stage_miners": True,
        "captured_output_redacted": True,
        "summary_excludes_plaintext_tokens": True,
        "raw_activation_redacted": True,
        "local_requeue_verified": bool(stage_requeue_verified),
        "temporary_http": effective_coordinator_url(args).startswith("http://"),
        "token_rotation_required": args.mode in {MODE_PACKAGE, MODE_EXTERNAL_EXISTING},
        "not_production": True,
        "not_p2p": True,
        "not_gpu_tpu_pooling": True,
        "not_gguf_llamacpp_serving": True,
        "not_large_model_serving": True,
        "not_public_prompt_serving": True,
    }


def real_llm_internet_alpha_command(args: argparse.Namespace, output_dir: Path, mode: str) -> list[Any]:
    command: list[Any] = [
        "crowdtensor",
        "real-llm-internet-alpha",
        "--mode",
        mode,
        "--output-dir",
        str(output_dir),
        "--public-host",
        getattr(args, "public_host", DEFAULT_PUBLIC_HOST),
        "--bind-host",
        getattr(args, "bind_host", "0.0.0.0"),
        "--port",
        str(getattr(args, "port", DEFAULT_PORT)),
        "--base-port",
        str(getattr(args, "base_port", DEFAULT_REQUEUE_BASE_PORT)),
        "--miner-id",
        getattr(args, "miner_id", "internet-real-llm"),
        "--request-count",
        str(getattr(args, "request_count", 1)),
        "--max-new-tokens",
        str(getattr(args, "max_new_tokens", 1)),
        "--hf-model-id",
        getattr(args, "hf_model_id", DEFAULT_MODEL_ID),
        "--real-llm-backend",
        getattr(args, "real_llm_backend", REAL_LLM_BACKEND_CPU),
        "--real-llm-partition-mode",
        getattr(args, "real_llm_partition_mode", "full"),
        "--timeout-seconds",
        str(getattr(args, "timeout_seconds", 300.0)),
        "--remote-timeout-seconds",
        str(getattr(args, "remote_timeout_seconds", 180.0)),
        "--startup-timeout",
        str(getattr(args, "startup_timeout", 30.0)),
        "--process-exit-timeout",
        str(getattr(args, "process_exit_timeout", 20.0)),
        "--poll-interval",
        str(getattr(args, "poll_interval", 1.0)),
        "--http-timeout",
        str(getattr(args, "http_timeout", 30.0)),
        "--lease-seconds",
        str(getattr(args, "lease_seconds", 15.0)),
        "--compute-seconds",
        str(getattr(args, "compute_seconds", 0.2)),
        "--heartbeat-interval",
        str(getattr(args, "heartbeat_interval", 0.1)),
        "--idle-sleep",
        str(getattr(args, "idle_sleep", 0.5)),
        "--max-request-attempts",
        str(getattr(args, "max_request_attempts", 120)),
    ]
    if getattr(args, "hf_cache_dir", ""):
        command.extend(["--hf-cache-dir", "HF_CACHE_DIR"])
    if mode == MODE_EXTERNAL_EXISTING:
        command.extend([
            "--coordinator-url",
            getattr(args, "coordinator_url", "") or "COORDINATOR_URL",
            "--observer-token",
            "$CROWDTENSOR_OBSERVER_TOKEN",
            "--admin-token",
            "$CROWDTENSOR_ADMIN_TOKEN",
        ])
    elif getattr(args, "coordinator_url", ""):
        command.extend(["--coordinator-url", "COORDINATOR_URL"])
    if getattr(args, "skip_requeue", False):
        command.append("--skip-requeue")
    command.append("--json")
    return command


def not_completed_items(report: dict[str, Any]) -> list[str]:
    runtime = report.get("runtime_classification") if isinstance(report.get("runtime_classification"), dict) else {}
    codes = set(report.get("diagnosis_codes") or [])
    mode = str(report.get("mode") or "")
    existing = [str(item) for item in (report.get("not_completed") or []) if str(item)]
    items: list[tuple[str, Any]] = [("Real Internet Alpha report ready", report.get("ok"))]
    if mode == MODE_LOCAL_GENERATED:
        items.extend([
            ("Live RC ready", "real_llm_live_rc_ready" in codes),
            ("local stage requeue ready", "real_llm_stage_requeue_ready" in codes and runtime.get("stage_requeue_verified") is True),
            ("stage0 requeue ready", "stage0_requeue_ready" in codes or "stage_requeue_ready" in codes),
            ("stage1 requeue ready", "stage1_requeue_ready" in codes or "stage_requeue_ready" in codes),
            ("local generated stand-ins classified", runtime.get("local_generated_stage_upload_standins") is True),
        ])
    elif mode == MODE_PACKAGE:
        items.extend([
            ("package artifacts ready", "real_llm_internet_alpha_package_ready" in codes),
            ("Live RC prepare ready", "real_llm_live_rc_prepare_ready" in codes),
            ("package-only classification present", runtime.get("package_only") is True),
            ("external verification still pending", "real_llm_internet_alpha_ready" in codes),
        ])
    else:
        items.extend([
            ("external runtime verified", runtime.get("external_runtime_verified") is True or "external_runtime_verified" in codes),
            ("real LLM Internet Alpha ready", "real_llm_internet_alpha_ready" in codes),
        ])
    for step in report.get("steps") or []:
        if isinstance(step, dict) and step.get("ok") is not True:
            items.append((f"step {step.get('name') or 'step'} passed", False))
    missing = list(existing)
    seen = set(missing)
    for label, ready in items:
        if ready is True or label in seen:
            continue
        missing.append(label)
        seen.add(label)
    return missing


def recommended_next_command(
    report: dict[str, Any],
    args: argparse.Namespace,
    *,
    output_dir: Path,
    missing: list[str],
) -> dict[str, Any]:
    mode = str(report.get("mode") or args.mode)
    if report.get("ok") and mode == MODE_LOCAL_GENERATED:
        return command_entry(
            "inspect Real Internet Alpha evidence",
            artifact_command(output_dir, "real_llm_internet_alpha.md"),
            reason="review_artifacts",
        )
    if report.get("ok") and mode == MODE_PACKAGE:
        return command_entry(
            "verify external Alpha runtime",
            real_llm_internet_alpha_command(args, output_dir, MODE_EXTERNAL_EXISTING),
            reason="run_external_existing_after_package",
            requires_private_credentials=True,
            side_effectful=True,
        )
    if report.get("ok"):
        return command_entry(
            "inspect Real Internet Alpha evidence",
            artifact_command(output_dir, "real_llm_internet_alpha.md"),
            reason="review_artifacts",
        )
    if mode == MODE_PACKAGE:
        return command_entry(
            "rerun Alpha package proof",
            real_llm_internet_alpha_command(args, output_dir, MODE_PACKAGE),
            reason="fix_package_blockers" if missing else "rerun_package",
            side_effectful=True,
        )
    if mode == MODE_EXTERNAL_EXISTING:
        return command_entry(
            "rerun external Alpha verification",
            real_llm_internet_alpha_command(args, output_dir, MODE_EXTERNAL_EXISTING),
            reason="fix_external_runtime_blockers" if missing else "rerun_external_existing",
            requires_private_credentials=True,
            side_effectful=True,
        )
    return command_entry(
        "rerun local Alpha proof",
        real_llm_internet_alpha_command(args, output_dir, MODE_LOCAL_GENERATED),
        reason="fix_local_requeue_blockers" if missing else "rerun_local_generated",
        side_effectful=True,
    )


def next_commands(
    report: dict[str, Any],
    args: argparse.Namespace,
    *,
    output_dir: Path,
    recommended: dict[str, Any],
) -> list[dict[str, Any]]:
    mode = str(report.get("mode") or args.mode)
    commands = [
        command_entry(
            "inspect shareable summary",
            artifact_command(output_dir, "real_llm_internet_alpha.md"),
            reason="review_artifacts",
        ),
        command_entry(
            "inspect support bundle",
            artifact_command(output_dir, "support_bundle.json", lines="1,220p"),
            reason="inspect_diagnostics",
        ),
    ]
    if report.get("ok"):
        commands.append(command_entry(
            f"refresh {mode} proof",
            real_llm_internet_alpha_command(args, output_dir, mode),
            reason="refresh_real_internet_alpha",
            requires_private_credentials=mode == MODE_EXTERNAL_EXISTING,
            side_effectful=True,
        ))
    else:
        commands.append(dict(recommended))
    if mode == MODE_PACKAGE:
        commands.append(dict(recommended))
    elif mode != MODE_EXTERNAL_EXISTING:
        commands.append(command_entry(
            "prepare external Alpha package",
            real_llm_internet_alpha_command(args, output_dir, MODE_PACKAGE),
            reason="prepare_external_stage_uploads",
            side_effectful=True,
        ))
    return commands


def user_status(*, ready: bool, mode: str, recommended: dict[str, Any], missing: list[str]) -> dict[str, Any]:
    if ready and mode == MODE_PACKAGE:
        state = "package-ready"
        headline = "Real Internet Swarm Inference Alpha package artifacts are ready; external verification is next."
        next_step = "run_external_existing_verification"
    elif ready:
        state = "ready"
        headline = "Real Internet Swarm Inference Alpha evidence is ready."
        next_step = "review_artifacts"
    elif mode == MODE_PACKAGE:
        state = "package-blocked"
        headline = "Real Internet Alpha package preparation needs attention."
        next_step = "fix_package_blockers"
    elif mode == MODE_EXTERNAL_EXISTING:
        state = "external-existing-blocked"
        headline = "Real Internet Alpha external verification needs attention."
        next_step = "fix_external_runtime_blockers"
    else:
        state = "local-generated-blocked"
        headline = "Real Internet Alpha local proof needs attention."
        next_step = "fix_local_requeue_blockers"
    return {
        "state": state,
        "headline": headline,
        "next_step": next_step,
        "recommended_label": recommended.get("label") or "none",
        "recommended_reason": recommended.get("reason") or "none",
        "not_completed_count": len(missing),
        "public_artifact_safe": True,
    }


def review_summary(
    report: dict[str, Any],
    *,
    output_dir: Path,
    recommended: dict[str, Any],
    missing: list[str],
) -> dict[str, Any]:
    ready = bool(report.get("ok"))
    mode = str(report.get("mode") or "")
    codes = [str(code) for code in (report.get("diagnosis_codes") or [])]
    return {
        "schema": "real_llm_internet_alpha_review_summary_v1",
        "state": "package-ready" if ready and mode == MODE_PACKAGE else ("ready" if ready else "blocked"),
        "headline": (
            "Real Internet Swarm Inference Alpha package artifacts are ready; external verification is next."
            if ready and mode == MODE_PACKAGE
            else (
                "Real Internet Swarm Inference Alpha evidence is ready."
                if ready
                else "Real Internet Swarm Inference Alpha evidence needs attention."
            )
        ),
        "mode": report.get("mode"),
        "next_step": "run_external_existing_verification" if ready and mode == MODE_PACKAGE else ("review_artifacts" if ready else "fix_blockers"),
        "inspect_first": str(output_dir / "real_llm_internet_alpha.md"),
        "support_bundle": str(output_dir / "support_bundle.json"),
        "recommended_label": recommended.get("label") or "none",
        "recommended_reason": recommended.get("reason") or "none",
        "next_command": recommended.get("command_line") or "",
        "primary_code": (
            "real_llm_internet_alpha_package_ready"
            if ready and mode == MODE_PACKAGE
            else ("real_llm_internet_alpha_ready" if ready else (codes[0] if codes else "real_llm_internet_alpha_blocked"))
        ),
        "attention": "external verification pending" if ready and mode == MODE_PACKAGE else ("none" if ready else (missing[0] if missing else "real_llm_internet_alpha_blocked")),
        "attention_detail": "; ".join(missing[:6]),
        "not_completed_count": len(missing),
        "public_artifact_safe": True,
    }


def attach_user_guidance(report: dict[str, Any], args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    missing = not_completed_items(report)
    recommended = recommended_next_command(report, args, output_dir=output_dir, missing=missing)
    report["not_completed"] = missing
    report["recommended_next_command"] = recommended
    report["next_commands"] = next_commands(report, args, output_dir=output_dir, recommended=recommended)
    report["user_status"] = user_status(
        ready=bool(report.get("ok")),
        mode=str(report.get("mode") or args.mode),
        recommended=recommended,
        missing=missing,
    )
    report["review_summary"] = review_summary(
        report,
        output_dir=output_dir,
        recommended=recommended,
        missing=missing,
    )
    report["artifact_summary"] = artifact_summary(output_dir)
    return report


def ensure_user_guidance(report: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    if (
        isinstance(report.get("recommended_next_command"), dict)
        and isinstance(report.get("next_commands"), list)
        and isinstance(report.get("user_status"), dict)
        and isinstance(report.get("review_summary"), dict)
    ):
        report.setdefault("not_completed", not_completed_items(report))
        report.setdefault("artifact_summary", artifact_summary(output_dir))
        return report
    missing = not_completed_items(report)
    recommended = command_entry(
        "inspect Real Internet Alpha evidence",
        artifact_command(output_dir, "real_llm_internet_alpha.md"),
        reason="review_artifacts" if report.get("ok") else "review_missing_evidence",
    )
    report["not_completed"] = missing
    report["recommended_next_command"] = recommended
    report["next_commands"] = [
        command_entry("inspect shareable summary", artifact_command(output_dir, "real_llm_internet_alpha.md"), reason="review_artifacts"),
        command_entry("inspect support bundle", artifact_command(output_dir, "support_bundle.json", lines="1,220p"), reason="inspect_diagnostics"),
    ]
    report["user_status"] = user_status(
        ready=bool(report.get("ok")),
        mode=str(report.get("mode") or ""),
        recommended=recommended,
        missing=missing,
    )
    report["review_summary"] = review_summary(
        report,
        output_dir=output_dir,
        recommended=recommended,
        missing=missing,
    )
    report["artifact_summary"] = artifact_summary(output_dir)
    return report


def support_bundle_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "real_llm_internet_alpha_support_bundle_v1",
        "ok": report.get("ok"),
        "mode": report.get("mode"),
        "output_dir": report.get("output_dir"),
        "coordinator_url": report.get("coordinator_url"),
        "diagnosis_codes": report.get("diagnosis_codes"),
        "runtime_classification": report.get("runtime_classification"),
        "steps": report.get("steps"),
        "payload_summaries": report.get("payload_summaries"),
        "review_summary": report.get("review_summary"),
        "user_status": report.get("user_status"),
        "recommended_next_command": report.get("recommended_next_command"),
        "next_commands": report.get("next_commands"),
        "artifact_summary": report.get("artifact_summary"),
        "not_completed": report.get("not_completed"),
        "output_request": report.get("output_request"),
        "prompt_scope": report.get("prompt_scope"),
        "answer_scope": report.get("answer_scope"),
        "shareable_summary": report.get("shareable_summary"),
        "safety": report.get("safety"),
        "limitations": report.get("limitations"),
        "public_artifact_safe": True,
    }


def persist_report(report: dict[str, Any], *, output_dir: Path, secret_values: list[str]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report.setdefault("output_request", output_request_summary())
    report.setdefault("prompt_scope", prompt_scope_summary(report))
    report.setdefault("answer_scope", answer_scope_summary())
    report.setdefault("shareable_summary", shareable_summary())
    report.setdefault("artifact_summary", artifact_summary(output_dir))
    report = ensure_user_guidance(report, output_dir=output_dir)
    report = support_bundle.sanitize(redact_values(report, secret_values))
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    leaks.extend(secret for secret in secret_values if secret and secret in encoded)
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"sensitive_output_detected"})
        report["safety_error"] = "real LLM Internet Alpha report contained secret-like fragments"
    json_path = output_dir / "real_llm_internet_alpha.json"
    md_path = output_dir / "real_llm_internet_alpha.md"
    support_path = output_dir / "support_bundle.json"
    report.setdefault("artifacts", {})
    report["artifacts"]["support_bundle_json"] = artifact_entry(
        support_path,
        output_dir,
        kind="real_llm_internet_alpha_support_bundle",
        schema="real_llm_internet_alpha_support_bundle_v1",
        ok=report.get("ok"),
    )
    report["artifact_summary"] = artifact_summary(output_dir)
    if isinstance(report.get("review_summary"), dict):
        report["review_summary"]["inspect_first"] = report["artifact_summary"]["inspect_first"]
        report["review_summary"]["support_bundle"] = report["artifact_summary"]["support_bundle"]
    write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    write_json(support_path, support_bundle_payload(report))
    if "artifacts" in report:
        report["artifacts"]["real_llm_internet_alpha_json"]["present"] = True
        report["artifacts"]["real_llm_internet_alpha_markdown"]["present"] = True
        report["artifacts"]["support_bundle_json"]["present"] = True
        report["artifact_summary"] = artifact_summary(output_dir)
        if isinstance(report.get("review_summary"), dict):
            report["review_summary"]["inspect_first"] = report["artifact_summary"]["inspect_first"]
            report["review_summary"]["support_bundle"] = report["artifact_summary"]["support_bundle"]
        write_json(json_path, report)
        md_path.write_text(render_markdown(report), encoding="utf-8")
        write_json(support_path, support_bundle_payload(report))
    return report


def build_local_generated(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    secret_values = [args.observer_token, args.admin_token]
    live_step, live_payload, live_dir = run_live_rc(args, output_dir=output_dir, runner=runner, secret_values=secret_values)
    steps.append(live_step)

    requeue_payloads: dict[str, dict[str, Any]] = {}
    requeue_dirs: dict[str, Path] = {}
    stage_requeue_verified = False
    if args.skip_requeue:
        steps.append({"name": "real_llm_stage_requeue", "ok": False, "skipped": True})
    else:
        for index, failure_mode in enumerate(["kill-stage0-after-claim", "kill-stage1-after-claim"]):
            step, payload, child_dir = run_requeue_beta(
                args,
                failure_mode=failure_mode,
                base_port=args.base_port + index,
                output_dir=output_dir,
                runner=runner,
            )
            steps.append(step)
            requeue_payloads[failure_mode] = payload
            requeue_dirs[failure_mode] = child_dir
        stage_requeue_verified = all(
            payload.get("ok") and "stage_requeue_ready" in diagnosis_codes(payload)
            for payload in requeue_payloads.values()
        )

    codes = diagnosis_codes(live_payload, *requeue_payloads.values())
    if stage_requeue_verified:
        codes.extend(["real_llm_stage_requeue_ready", "stage0_requeue_ready", "stage1_requeue_ready"])
    elif args.skip_requeue:
        codes.append("real_llm_stage_requeue_skipped")
    else:
        codes.append("real_llm_stage_requeue_blocked")
    if live_payload.get("ok") and "real_llm_live_rc_ready" in codes and stage_requeue_verified:
        codes.append("real_llm_internet_alpha_ready")
    else:
        codes.append("real_llm_internet_alpha_blocked")
    ok = "real_llm_internet_alpha_ready" in codes
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": MODE_LOCAL_GENERATED,
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
            "real_llm_partition_mode": args.real_llm_partition_mode,
            "prompt_text_count": len(DEFAULT_PROMPTS),
            "require_distinct_stage_miners": True,
        },
        "runtime_classification": {
            "local_generated_stage_upload_standins": True,
            "package_only": False,
            "external_runtime_verified": False,
            "stage_requeue_verified": stage_requeue_verified,
        },
        "steps": steps,
        "payload_summaries": {
            "live_rc": summarize_payload(live_payload),
            "stage0_requeue": summarize_payload(requeue_payloads.get("kill-stage0-after-claim", {})),
            "stage1_requeue": summarize_payload(requeue_payloads.get("kill-stage1-after-claim", {})),
        },
        "diagnosis_codes": sorted(set(codes)),
        "artifacts": {
            **base_artifacts(output_dir, ok=ok),
            **child_artifacts(output_dir, live_dir, requeue_dirs, ok=ok),
        },
        "safety": safety_summary(args, stage_requeue_verified=stage_requeue_verified),
        "operator_action": [
            "Use local-generated as the mandatory localhost proof before running real public Coordinator tests.",
            "Use package mode to create stage0/stage1 upload directories for two external CPU hosts or Kaggle sessions.",
            "After both external stage Miners are running, verify with real-llm-internet-alpha --mode external-existing.",
        ],
        "limitations": [
            "local-generated proves real tiny GPT weights and stage requeue semantics through local stand-ins; it is not external machine evidence.",
            "CPU-only read-only tiny Hugging Face GPT split proof; not production Swarm Inference.",
            "No P2P routing, NAT traversal, GPU/TPU pooling, GGUF/llama.cpp serving, large-model serving, training, or arbitrary prompt serving.",
        ],
    }
    report = attach_user_guidance(report, args, output_dir=output_dir)
    return persist_report(report, output_dir=output_dir, secret_values=secret_values)


def build_package(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    secret_values = [args.observer_token, args.admin_token]
    live_step, live_payload, live_dir = run_live_rc(args, output_dir=output_dir, runner=runner, secret_values=secret_values)
    codes = diagnosis_codes(live_payload)
    if live_payload.get("ok") and "real_llm_live_rc_prepare_ready" in codes:
        codes.append("real_llm_internet_alpha_package_ready")
    else:
        codes.append("real_llm_internet_alpha_package_blocked")
    ok = "real_llm_internet_alpha_package_ready" in codes
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": MODE_PACKAGE,
        "output_dir": str(output_dir),
        "coordinator_url": effective_coordinator_url(args),
        "public_host": args.public_host,
        "port": args.port,
        "miner_id": args.miner_id,
        "workload": {
            "workload_type": WORKLOAD_TYPE,
            "stage_mode": "split",
            "request_count": args.request_count,
            "max_new_tokens": args.max_new_tokens,
            "hf_model_id": args.hf_model_id,
            "real_llm_partition_mode": args.real_llm_partition_mode,
            "prompt_text_count": len(DEFAULT_PROMPTS),
            "require_distinct_stage_miners": True,
        },
        "runtime_classification": {
            "local_generated_stage_upload_standins": False,
            "package_only": True,
            "external_runtime_verified": False,
            "stage_requeue_verified": False,
        },
        "steps": [live_step],
        "payload_summaries": {"live_rc": summarize_payload(live_payload)},
        "diagnosis_codes": sorted(set(codes)),
        "artifacts": {
            **base_artifacts(output_dir, ok=ok),
            **child_artifacts(output_dir, live_dir, {}, ok=ok),
        },
        "safety": safety_summary(args, stage_requeue_verified=False),
        "operator_action": [
            "Start the generated Coordinator package on the public host.",
            "Upload the generated stage0 and stage1 directories to two external CPU hosts or Kaggle sessions.",
            "Run external-existing verification before claiming real Internet Alpha readiness.",
        ],
        "limitations": [
            "package mode prepares upload artifacts only; it does not claim live external runtime evidence.",
            "CPU-only read-only tiny Hugging Face GPT split proof; not production Swarm Inference.",
            "No P2P routing, NAT traversal, GPU/TPU pooling, GGUF/llama.cpp serving, large-model serving, training, or arbitrary prompt serving.",
        ],
    }
    report = attach_user_guidance(report, args, output_dir=output_dir)
    return persist_report(report, output_dir=output_dir, secret_values=secret_values)


def build_external_existing(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    secret_values = [args.observer_token, args.admin_token]
    live_step, live_payload, live_dir = run_live_rc(args, output_dir=output_dir, runner=runner, secret_values=secret_values)
    codes = diagnosis_codes(live_payload)
    external_verified = bool(live_payload.get("ok") and "external_runtime_verified" in codes)
    if external_verified:
        codes.append("real_llm_internet_alpha_ready")
    else:
        codes.append("real_llm_internet_alpha_blocked")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": external_verified,
        "mode": MODE_EXTERNAL_EXISTING,
        "output_dir": str(output_dir),
        "coordinator_url": effective_coordinator_url(args),
        "public_host": args.public_host,
        "port": args.port,
        "miner_id": args.miner_id,
        "workload": {
            "workload_type": WORKLOAD_TYPE,
            "stage_mode": "split",
            "request_count": args.request_count,
            "max_new_tokens": args.max_new_tokens,
            "hf_model_id": args.hf_model_id,
            "real_llm_partition_mode": args.real_llm_partition_mode,
            "prompt_text_count": len(DEFAULT_PROMPTS),
            "require_distinct_stage_miners": True,
        },
        "runtime_classification": {
            "local_generated_stage_upload_standins": False,
            "package_only": False,
            "external_runtime_verified": external_verified,
            "stage_requeue_verified": False,
        },
        "steps": [live_step],
        "payload_summaries": {"live_rc": summarize_payload(live_payload)},
        "diagnosis_codes": sorted(set(codes)),
        "artifacts": {
            **base_artifacts(output_dir, ok=external_verified),
            **child_artifacts(output_dir, live_dir, {}, ok=external_verified),
        },
        "safety": safety_summary(args, stage_requeue_verified=False),
        "operator_action": [
            "If blocked, confirm the public Coordinator is reachable and both stage Miners are running distinct --real-llm-stage-role values.",
            "Rotate temporary HTTP tokens after the public run.",
        ],
        "limitations": [
            "external-existing verifies already running external stage Miners; it does not create remote machines or notebooks.",
            "CPU-only read-only tiny Hugging Face GPT split proof; not production Swarm Inference.",
            "No P2P routing, NAT traversal, GPU/TPU pooling, GGUF/llama.cpp serving, large-model serving, training, or arbitrary prompt serving.",
        ],
    }
    report = attach_user_guidance(report, args, output_dir=output_dir)
    return persist_report(report, output_dir=output_dir, secret_values=secret_values)


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == MODE_LOCAL_GENERATED:
        return build_local_generated(args, output_dir=output_dir, runner=runner)
    if args.mode == MODE_PACKAGE:
        return build_package(args, output_dir=output_dir, runner=runner)
    return build_external_existing(args, output_dir=output_dir, runner=runner)


def render_markdown(report: dict[str, Any]) -> str:
    runtime = report.get("runtime_classification") or {}
    workload = report.get("workload") or {}
    review = report.get("review_summary") if isinstance(report.get("review_summary"), dict) else {}
    user = report.get("user_status") if isinstance(report.get("user_status"), dict) else {}
    recommended = report.get("recommended_next_command") if isinstance(report.get("recommended_next_command"), dict) else {}
    artifact_report = report.get("artifact_summary") if isinstance(report.get("artifact_summary"), dict) else {}
    next_items = report.get("next_commands") if isinstance(report.get("next_commands"), list) else []
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    lines = [
        "# CrowdTensor Real Internet Swarm Inference Alpha",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- coordinator_url: `{report.get('coordinator_url')}`",
        f"- request_count: `{workload.get('request_count')}`",
        f"- hf_model_id: `{workload.get('hf_model_id')}`",
        f"- partition_mode: `{workload.get('real_llm_partition_mode')}`",
        f"- local_generated_stage_upload_standins: `{runtime.get('local_generated_stage_upload_standins')}`",
        f"- package_only: `{runtime.get('package_only')}`",
        f"- external_runtime_verified: `{runtime.get('external_runtime_verified')}`",
        f"- stage_requeue_verified: `{runtime.get('stage_requeue_verified')}`",
        "",
        "## Review",
        "",
        f"- state: `{review.get('state')}`",
        f"- status: `{user.get('headline')}`",
        f"- next step: `{review.get('next_step')}`",
        f"- inspect first: `{review.get('inspect_first')}`",
        f"- recommended: `{recommended.get('label')}` reason=`{recommended.get('reason')}`",
        f"- recommended command: `{recommended.get('command_line')}`",
        f"- not completed count: `{review.get('not_completed_count')}`",
        "",
        "## What To Do Next",
        "",
    ]
    if next_items:
        lines.extend(
            (
                f"- {item.get('label')}: `{item.get('command_line')}`"
                + (" (requires private credentials; see runbook)" if item.get("requires_private_credentials") else "")
                + (" side_effectful=`True`" if item.get("side_effectful") else "")
            )
            for item in next_items
            if isinstance(item, dict)
        )
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Output Scope",
        "",
        f"- include output: `{output_request.get('include_output')}`",
        f"- output request note: {output_request.get('summary') or 'Public artifacts summarize inference evidence only and do not include answer text.'}",
        f"- prompt scope: `{prompt_scope_text(prompt_scope)}`",
        f"- prompt scope note: {prompt_scope.get('summary') or 'Public artifacts exclude raw prompt text.'}",
        f"- answer scope: `{answer_scope.get('scope_state')}`",
        f"- answer scope note: {answer_scope.get('summary') or 'Public artifacts contain no local answer transcript or raw generated text.'}",
        f"- saved JSON display: `{answer_scope.get('saved_json_display')}`",
        f"- saved Markdown display: `{answer_scope.get('saved_markdown_display')}`",
        f"- shareable: `saved_artifacts={shareable.get('saved_artifacts_public_safe')} raw_prompt_public={shareable.get('raw_prompt_public')} raw_generated_text_public={shareable.get('raw_generated_text_public')} generated_token_ids_public={shareable.get('generated_token_ids_public')} answer_scope_state={shareable.get('answer_scope_state')} local_answer_terminal_only={shareable.get('local_answer_terminal_only')}`",
        "",
        "## Artifact Summary",
        "",
        f"- inspect first: `{artifact_report.get('inspect_first')}`",
        f"- summary JSON: `{artifact_report.get('summary_json')}`",
        f"- summary Markdown: `{artifact_report.get('summary_markdown')}`",
        f"- support bundle: `{artifact_report.get('support_bundle')}`",
        f"- present: `{artifact_report.get('present_artifact_count')}` / `{artifact_report.get('artifact_count')}`",
        f"- public artifact safe: `{artifact_report.get('public_artifact_safe')}`",
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`",
        "",
        "## Steps",
        "",
    ])
    for step in report.get("steps") or []:
        lines.append(f"- `{step.get('name')}`: `{step.get('ok')}` schema=`{step.get('payload_schema')}`")
    lines.extend(["", "## Not Completed", ""])
    not_completed = report.get("not_completed") or []
    lines.extend(f"- {item}" for item in not_completed) if not_completed else lines.append("- none")
    lines.extend(["", "## Artifacts", ""])
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        lines.append(f"- `{name}`: `{artifact.get('path')}` present=`{artifact.get('present')}`")
    lines.extend(["", "## Boundaries", ""])
    for limitation in report.get("limitations") or []:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the real Internet Swarm Inference Alpha report.")
    parser.add_argument("--mode", choices=[MODE_LOCAL_GENERATED, MODE_PACKAGE, MODE_EXTERNAL_EXISTING], default=MODE_LOCAL_GENERATED)
    parser.add_argument("--output-dir", default="dist/real-llm-internet-alpha")
    parser.add_argument("--public-host", default=DEFAULT_PUBLIC_HOST)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--base-port", type=int, default=DEFAULT_REQUEUE_BASE_PORT)
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--miner-id", default="internet-real-llm")
    parser.add_argument("--request-count", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--hf-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--real-llm-backend", choices=["hf_transformers_cpu", "hf_transformers_cuda", "cpu", "cuda", "auto"], default=REAL_LLM_BACKEND_CPU)
    parser.add_argument("--real-llm-partition-mode", choices=["full", "stage-local", "stage_local"], default="full")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--remote-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--process-exit-timeout", type=float, default=20.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--http-timeout", type=float, default=30.0)
    parser.add_argument("--lease-seconds", type=float, default=15.0)
    parser.add_argument("--compute-seconds", type=float, default=0.2)
    parser.add_argument("--heartbeat-interval", type=float, default=0.1)
    parser.add_argument("--idle-sleep", type=float, default=0.5)
    parser.add_argument("--max-request-attempts", type=int, default=120)
    parser.add_argument("--observer-token", default="")
    parser.add_argument("--admin-token", default="")
    parser.add_argument("--skip-requeue", action="store_true")
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
    for name in [
        "timeout_seconds",
        "remote_timeout_seconds",
        "startup_timeout",
        "process_exit_timeout",
        "poll_interval",
        "http_timeout",
        "lease_seconds",
        "heartbeat_interval",
        "idle_sleep",
    ]:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.compute_seconds < 0:
        raise SystemExit("--compute-seconds must be non-negative")
    if args.max_request_attempts < 1:
        raise SystemExit("--max-request-attempts must be at least 1")
    if args.mode == MODE_EXTERNAL_EXISTING:
        missing = [
            name for name in ["coordinator_url", "observer_token", "admin_token"]
            if not getattr(args, name)
        ]
        if missing:
            raise SystemExit(f"external-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
    args.coordinator_url = args.coordinator_url.rstrip("/") if args.coordinator_url else ""
    return args


def print_human(report: dict[str, Any]) -> None:
    print("CrowdTensor real Internet Swarm Inference Alpha")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  coordinator: {report.get('coordinator_url')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    runtime = report.get("runtime_classification") or {}
    print(f"  local stand-in: {runtime.get('local_generated_stage_upload_standins')}")
    print(f"  package only: {runtime.get('package_only')}")
    print(f"  external runtime: {runtime.get('external_runtime_verified')}")
    print(f"  stage requeue: {runtime.get('stage_requeue_verified')}")


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
