#!/usr/bin/env python3
"""Run and package a local CPU-only pipeline-sharded inference proof."""

from __future__ import annotations

import argparse
import json
import os
import select
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from create_miner_invite import create_invite  # noqa: E402
import support_bundle  # noqa: E402


SCHEMA = "sharded_inference_evidence_v1"
OBSERVABILITY_SCHEMA = "sharded_inference_observability_v1"
WORKLOAD_TYPE = "sharded_model_bundle_infer"
DEFAULT_ADMIN_TOKEN = "sharded-inference-admin"
DEFAULT_OBSERVER_TOKEN = "sharded-inference-observer"
FAILURE_NONE = "none"
FAILURE_KILL_STAGE_AFTER_CLAIM = "kill-stage-after-claim"
FAILURE_KILL_STAGE0_AFTER_CLAIM = "kill-stage0-after-claim"
FAILURE_KILL_STAGE1_AFTER_CLAIM = "kill-stage1-after-claim"
FAILURE_MODES = {
    FAILURE_NONE,
    FAILURE_KILL_STAGE_AFTER_CLAIM,
    FAILURE_KILL_STAGE0_AFTER_CLAIM,
    FAILURE_KILL_STAGE1_AFTER_CLAIM,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def shell_command(parts: list[Any]) -> str:
    return shlex.join([str(part) for part in parts])


def command_entry(
    label: str,
    command: list[Any],
    *,
    reason: str = "",
    side_effectful: bool = False,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "label": label,
        "command": [str(part) for part in command],
        "command_line": shell_command(command),
        "public_artifact_safe": True,
        "side_effectful": bool(side_effectful),
    }
    if reason:
        entry["reason"] = reason
    return entry


def artifact_command(output_dir: Path, filename: str, *, lines: str = "1,220p") -> list[str]:
    return ["sed", "-n", lines, str(output_dir / filename)]


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


def artifact_summary(output_dir: Path) -> dict[str, Any]:
    paths = {
        "inspect_first": output_dir / "sharded_inference_evidence.md",
        "summary_json": output_dir / "sharded_inference_evidence.json",
        "summary_markdown": output_dir / "sharded_inference_evidence.md",
        "support_bundle": output_dir / "support_bundle.json",
    }
    return {
        "schema": "sharded_inference_artifact_summary_v1",
        "inspect_first": str(paths["inspect_first"]),
        "summary_json": str(paths["summary_json"]),
        "summary_markdown": str(paths["summary_markdown"]),
        "support_bundle": str(paths["support_bundle"]),
        "shareable_paths": [str(path) for path in paths.values()],
        "artifact_count": len(paths),
        "present_artifact_count": sum(1 for path in paths.values() if path.is_file()),
        "raw_prompt_public": False,
        "raw_result_public": False,
        "public_artifact_safe": True,
        "summary": (
            "Open inspect_first first, then support_bundle for diagnostics. "
            "Artifacts contain route/stage readiness, hashes, counts, and redaction status only."
        ),
    }


def output_request_summary() -> dict[str, Any]:
    return {
        "include_output": False,
        "raw_prompt_public": False,
        "raw_result_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Sharded inference evidence records fixed model-bundle route readiness and validation summaries only; "
            "raw prompts, raw outputs, and intermediate tensors are excluded."
        ),
    }


def prompt_scope_summary(report: dict[str, Any] | None = None) -> dict[str, Any]:
    session = report.get("session") if isinstance(report, dict) and isinstance(report.get("session"), dict) else {}
    try:
        prompt_count = int(session.get("request_count") or 0)
    except (TypeError, ValueError):
        prompt_count = 0
    return {
        "source": "fixed-model-bundle-scenario",
        "prompt_count": prompt_count,
        "scenario_id": session.get("scenario_id"),
        "inline_prompt_text": False,
        "terminal_next_commands_local_private": False,
        "terminal_logs_local_private": True,
        "saved_artifacts_prompt_placeholders": True,
        "saved_artifacts_public_safe": True,
        "prefer_prompt_file_or_stdin_for_shareable_logs": False,
        "prompt_file_path_public": False,
        "raw_prompt_public": False,
        "public_artifact_safe": True,
        "summary": (
            "This proof uses fixed model-bundle validation scenarios; public artifacts record scenario id "
            "and request count, not raw prompt text."
        ),
    }


def prompt_scope_text(prompt_scope: dict[str, Any]) -> str:
    return (
        f"source={prompt_scope.get('source') or 'unknown'} "
        f"count={prompt_scope.get('prompt_count')} "
        f"scenario_id={prompt_scope.get('scenario_id') or 'none'} "
        f"inline_prompt_text={bool(prompt_scope.get('inline_prompt_text'))} "
        f"terminal_next_commands_local_private={bool(prompt_scope.get('terminal_next_commands_local_private'))} "
        f"saved_artifacts_prompt_placeholders={bool(prompt_scope.get('saved_artifacts_prompt_placeholders'))} "
        f"prompt_file_path_public={bool(prompt_scope.get('prompt_file_path_public'))} "
        f"raw_prompt_public={bool(prompt_scope.get('raw_prompt_public'))} "
        f"public_artifact_safe={bool(prompt_scope.get('public_artifact_safe'))}"
    )


def answer_scope_summary() -> dict[str, Any]:
    return {
        "scope_state": "evidence-only",
        "terminal_only": False,
        "visible_in_terminal": False,
        "saved_json_display": "validation-summary-only",
        "saved_markdown_display": "validation-summary-only",
        "json_stdout_display": "validation-summary-only",
        "raw_result_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "public_artifact_safe": True,
        "summary": (
            "The report proves two-stage routing and baseline validation; it does not save or print raw outputs."
        ),
    }


def shareable_summary() -> dict[str, Any]:
    return {
        "saved_artifacts_public_safe": True,
        "raw_prompt_public": False,
        "raw_result_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "answer_scope_state": "evidence-only",
        "local_answer_terminal_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Share sharded_inference_evidence.json/md and support_bundle.json; they contain stage readiness, "
            "baseline match, redaction status, and diagnostics only."
        ),
    }


def request_json(
    method: str,
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
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


def wait_health(base_url: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"coordinator exited early with code {proc.returncode}")
        try:
            if request_json("GET", base_url, "/health", timeout=2.0).get("ok") is True:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"coordinator did not become healthy: {last_error}")


def start_coordinator(args: argparse.Namespace, state_dir: Path, registry_path: Path) -> subprocess.Popen:
    command = [
        sys.executable,
        str(ROOT / "coordinator.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--state-dir",
        str(state_dir),
        "--lease-seconds",
        str(args.lease_seconds),
        "--inner-steps",
        str(args.request_count),
        "--backlog",
        "0",
        "--task-lane",
        f"python-cli:cpu:0:{WORKLOAD_TYPE}",
        "--miner-token-registry",
        str(registry_path),
        "--observer-token",
        args.observer_token,
        "--admin-token",
        args.admin_token,
    ]
    micro_llm_artifact = str(getattr(args, "micro_llm_artifact", "") or "")
    if micro_llm_artifact:
        command.extend(["--micro-llm-artifact", micro_llm_artifact])
    if WORKLOAD_TYPE == "real_llm_sharded_infer":
        if getattr(args, "hf_model_id", ""):
            command.extend(["--real-llm-model-id", str(args.hf_model_id)])
        if getattr(args, "real_llm_backend", ""):
            command.extend(["--real-llm-backend", str(args.real_llm_backend)])
        if getattr(args, "real_llm_partition_mode", ""):
            command.extend(["--real-llm-partition-mode", str(args.real_llm_partition_mode)])
        if getattr(args, "hf_cache_dir", ""):
            command.extend(["--hf-cache-dir", str(args.hf_cache_dir)])
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    wait_health(args.base_url, proc, args.startup_timeout)
    return proc


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def terminate_process_group(proc: subprocess.Popen, *, sig: int = signal.SIGTERM) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


def parse_miner_summary(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "accepted_tasks" in payload:
            return payload
    return {}


def tail_text(value: str, *, limit: int = 2000) -> str:
    return value if len(value) <= limit else value[-limit:]


def miner_env(invite: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["CROWDTENSOR_MINER_TOKEN"] = invite["env"]["CROWDTENSOR_MINER_TOKEN"]
    return env


def miner_command(
    args: argparse.Namespace,
    invite: dict[str, Any],
    *,
    compute_seconds: float | None = None,
    stage_role: str = "",
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "miner_cli.py"),
        "--coordinator",
        args.base_url,
        "--miner-id",
        invite["miner_id"],
        "--once",
        "--compute-seconds",
        str(args.compute_seconds if compute_seconds is None else compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        "0.2",
    ]
    if stage_role:
        if WORKLOAD_TYPE == "real_llm_sharded_infer":
            command.extend(["--real-llm-stage-role", stage_role])
        else:
            command.extend(["--micro-llm-stage-role", stage_role])
    if WORKLOAD_TYPE == "real_llm_sharded_infer":
        command.append("--enable-hf-tiny-gpt-runtime")
        if getattr(args, "hf_model_id", ""):
            command.extend(["--hf-model-id", str(args.hf_model_id)])
        if getattr(args, "real_llm_backend", ""):
            command.extend(["--real-llm-backend", str(args.real_llm_backend)])
        if getattr(args, "real_llm_partition_mode", ""):
            command.extend(["--real-llm-partition-mode", str(args.real_llm_partition_mode)])
        if getattr(args, "hf_cache_dir", ""):
            command.extend(["--hf-cache-dir", str(args.hf_cache_dir)])
    return command


def run_invited_miner(args: argparse.Namespace, invite: dict[str, Any], *, stage_role: str = "") -> dict[str, Any]:
    completed = subprocess.run(
        miner_command(args, invite, stage_role=stage_role),
        cwd=ROOT,
        env=miner_env(invite),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.miner_timeout,
    )
    summary = parse_miner_summary(completed.stdout)
    if completed.returncode != 0 or int(summary.get("accepted_tasks", 0)) != 1:
        raise RuntimeError(
            f"miner {invite['miner_id']} failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return {
        "miner_id": invite["miner_id"],
        "stage_role": stage_role or None,
        "ok": True,
        "returncode": completed.returncode,
        "miner_summary": summary,
        "stdout_tail": tail_text(completed.stdout),
        "stderr_tail": tail_text(completed.stderr),
    }


def start_invited_miner(
    args: argparse.Namespace,
    invite: dict[str, Any],
    *,
    compute_seconds: float,
    stage_role: str = "",
) -> subprocess.Popen:
    return subprocess.Popen(
        miner_command(args, invite, compute_seconds=compute_seconds, stage_role=stage_role),
        cwd=ROOT,
        env=miner_env(invite),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


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
            return {"task_id": fields["task"], "attempt": int(fields.get("attempt") or 0), "line": line}
    return {}


def wait_for_claim_line(proc: subprocess.Popen, *, timeout: float) -> tuple[str, dict[str, Any]]:
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


def create_session(args: argparse.Namespace) -> dict[str, Any]:
    backend = "cuda" if str(getattr(args, "real_llm_backend", "") or "") in {"cuda", "hf_transformers_cuda"} else "cpu"
    payload = {
        "request_count": args.request_count,
        "scenario_id": args.scenario_id,
        "workload_type": WORKLOAD_TYPE,
        "backend": backend,
    }
    if WORKLOAD_TYPE == "real_llm_sharded_infer" and getattr(args, "real_llm_partition_mode", ""):
        payload["partition_mode"] = str(args.real_llm_partition_mode)
    if WORKLOAD_TYPE == "real_llm_sharded_infer" and getattr(args, "max_new_tokens", 1):
        payload["max_new_tokens"] = int(getattr(args, "max_new_tokens", 1))
    return request_json(
        "POST",
        args.base_url,
        "/admin/inference-sessions",
        payload=payload,
        admin_token=args.admin_token,
    )


def admin_results(args: argparse.Namespace, *, task_id: str = "", limit: int = 10) -> dict[str, Any]:
    query = urlencode({
        "status": "accepted",
        "workload_type": WORKLOAD_TYPE,
        "task_id": task_id,
        "limit": limit,
    })
    return request_json("GET", args.base_url, f"/admin/results?{query}", admin_token=args.admin_token)


def wait_for_stage_task(
    args: argparse.Namespace,
    session_id: str,
    *,
    stage_id: int,
    generation_step: int | None = None,
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    deadline = time.monotonic() + timeout
    last_state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last_state = request_json("GET", args.base_url, "/state", observer_token=args.observer_token)
        for task in last_state.get("tasks", []):
            metadata = task.get("workload_metadata") if isinstance(task, dict) else {}
            if (
                isinstance(metadata, dict)
                and task.get("workload_type") == WORKLOAD_TYPE
                and str(metadata.get("session_id")) == session_id
                and int(metadata.get("stage_id", -1)) == stage_id
                and (
                    generation_step is None
                    or int(metadata.get("generation_step", 0)) == int(generation_step)
                )
            ):
                return str(task.get("task_id")), last_state
        time.sleep(0.1)
    return "", last_state


def wait_for_stage1_task(args: argparse.Namespace, session_id: str, *, timeout: float) -> tuple[str, dict[str, Any]]:
    generation_step = 0 if WORKLOAD_TYPE == "real_llm_sharded_infer" else None
    return wait_for_stage_task(
        args,
        session_id,
        stage_id=1,
        generation_step=generation_step,
        timeout=timeout,
    )


def wait_for_task_status(args: argparse.Namespace, task_id: str, status: str, *, timeout: float) -> tuple[bool, dict[str, Any]]:
    deadline = time.monotonic() + timeout
    last_state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last_state = request_json("GET", args.base_url, "/state", observer_token=args.observer_token)
        for task in last_state.get("tasks", []):
            if isinstance(task, dict) and task.get("task_id") == task_id and task.get("status") == status:
                return True, last_state
        time.sleep(0.1)
    return False, last_state


def session_tasks(state: dict[str, Any], session_id: str) -> list[dict[str, Any]]:
    rows = []
    for task in state.get("tasks", []):
        metadata = task.get("workload_metadata") if isinstance(task, dict) else {}
        if (
            isinstance(metadata, dict)
            and task.get("workload_type") == WORKLOAD_TYPE
            and str(metadata.get("session_id")) == session_id
        ):
            rows.append(task)
    return sorted(
        rows,
        key=lambda item: (
            int((item.get("workload_metadata") or {}).get("generation_step", 0)),
            int((item.get("workload_metadata") or {}).get("stage_id", 0)),
        ),
    )


def process_result(
    miner_id: str,
    proc: subprocess.Popen,
    *,
    expected_failure: bool = False,
    stdout_prefix: str = "",
    stage_role: str = "",
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
    summary = parse_miner_summary(stdout) if proc.returncode == 0 else {}
    return {
        "miner_id": miner_id,
        "stage_role": stage_role or None,
        "ok": bool(proc.returncode == 0 and int(summary.get("accepted_tasks", 0)) == 1),
        "expected_failure": expected_failure,
        "returncode": proc.returncode,
        "miner_summary": {
            "accepted_tasks": summary.get("accepted_tasks"),
            "rejected_tasks": summary.get("rejected_tasks"),
            "workloads": summary.get("workloads"),
        },
        "stdout_tail": tail_text(stdout),
        "stderr_tail": tail_text(stderr),
    }


def safe_state_text(state: dict[str, Any]) -> str:
    return json.dumps(state.get("tasks", []), sort_keys=True)


def sharded_inference_command(args: argparse.Namespace, output_dir: Path) -> list[Any]:
    command: list[Any] = [
        "crowdtensor",
        "shard-infer",
        "--output-dir",
        str(output_dir),
        "--port",
        getattr(args, "port", 9820),
        "--request-count",
        getattr(args, "request_count", 4),
        "--scenario-id",
        getattr(args, "scenario_id", "route-baseline"),
        "--failure-mode",
        getattr(args, "failure_mode", FAILURE_NONE),
        "--stage-mode",
        getattr(args, "stage_mode", "both"),
        "--json",
    ]
    if getattr(args, "require_distinct_stage_miners", False):
        command.append("--require-distinct-stage-miners")
    return command


def not_completed_items(report: dict[str, Any]) -> list[str]:
    codes = set(str(code) for code in (report.get("diagnosis_codes") or []) if code)
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    requeue = (
        ((report.get("observability") if isinstance(report.get("observability"), dict) else {}).get("requeue_summary"))
        if isinstance((report.get("observability") if isinstance(report.get("observability"), dict) else {}).get("requeue_summary"), dict)
        else {}
    )
    items: list[tuple[str, Any]] = [
        ("sharded inference ready", report.get("ok") is True and "sharded_inference_ready" in codes),
        ("stage 0 accepted", "stage_0_accepted" in codes),
        ("stage 1 accepted", "stage_1_accepted" in codes),
        ("activation transport ready", "activation_transport_ready" in codes),
        ("baseline match ready", "baseline_match" in codes),
        ("read-only safety boundary present", safety.get("read_only") is True),
        ("redaction safety present", safety.get("redaction_ok") is True),
        ("raw activation redacted", safety.get("raw_activation_redacted") is True),
        ("not production boundary present", safety.get("not_production") is True),
    ]
    if requeue.get("enabled"):
        items.append(("stage requeue ready", "stage_requeue_ready" in codes))
    for process in ((report.get("observability") or {}).get("processes") or []):
        if isinstance(process, dict) and process.get("ok") is not True and process.get("expected_failure") is not True:
            items.append((f"miner process {process.get('miner_id') or 'unknown'} passed", False))
    missing: list[str] = []
    for label, ready in items:
        if ready is not True:
            missing.append(label)
    return missing


def recommended_next_command(
    report: dict[str, Any],
    args: argparse.Namespace,
    *,
    output_dir: Path,
    missing: list[str],
) -> dict[str, Any]:
    if report.get("ok") is True:
        return command_entry(
            "inspect sharded inference evidence",
            artifact_command(output_dir, "sharded_inference_evidence.md"),
            reason="review_artifacts",
        )
    return command_entry(
        "rerun sharded inference proof",
        sharded_inference_command(args, output_dir),
        reason="fix_sharded_inference_blockers" if missing else "rerun_sharded_inference",
        side_effectful=True,
    )


def next_commands(
    report: dict[str, Any],
    args: argparse.Namespace,
    *,
    output_dir: Path,
    recommended: dict[str, Any],
) -> list[dict[str, Any]]:
    commands = [
        command_entry(
            "inspect shareable summary",
            artifact_command(output_dir, "sharded_inference_evidence.md"),
            reason="review_artifacts",
        ),
        command_entry(
            "inspect support bundle",
            artifact_command(output_dir, "support_bundle.json"),
            reason="inspect_diagnostics",
        ),
    ]
    if recommended and all(item.get("command_line") != recommended.get("command_line") for item in commands):
        commands.append(dict(recommended))
    refresh = command_entry(
        "refresh sharded inference proof",
        sharded_inference_command(args, output_dir),
        reason="refresh_sharded_inference",
        side_effectful=True,
    )
    if all(item.get("command_line") != refresh.get("command_line") for item in commands):
        commands.append(refresh)
    return commands


def user_status(report: dict[str, Any], *, recommended: dict[str, Any], missing: list[str]) -> dict[str, Any]:
    if report.get("ok") is True and not missing:
        state = "ready"
        headline = "Pipeline-sharded model-bundle inference evidence is ready."
        next_step = "review_artifacts"
    else:
        state = "blocked"
        headline = "Pipeline-sharded model-bundle inference evidence needs attention."
        next_step = "fix_blockers"
    session = report.get("session") if isinstance(report.get("session"), dict) else {}
    return {
        "state": state,
        "headline": headline,
        "next_step": next_step,
        "recommended_label": recommended.get("label") or "none",
        "recommended_reason": recommended.get("reason") or "none",
        "scenario_id": session.get("scenario_id") or "unknown",
        "request_count": session.get("request_count"),
        "public_artifact_safe": True,
    }


def review_summary(
    report: dict[str, Any],
    *,
    output_dir: Path,
    recommended: dict[str, Any],
    missing: list[str],
) -> dict[str, Any]:
    artifacts = artifact_summary(output_dir)
    ready = bool(report.get("ok") is True and not missing)
    return {
        "schema": "sharded_inference_review_summary_v1",
        "state": "ready" if ready else "blocked",
        "ready": ready,
        "next_step": "review_artifacts" if ready else "fix_sharded_inference_blockers",
        "inspect_first": artifacts["inspect_first"],
        "support_bundle": artifacts["support_bundle"],
        "recommended_next_command": recommended,
        "recommended_label": recommended.get("label") or "none",
        "recommended_reason": recommended.get("reason") or "none",
        "next_command": recommended.get("command_line") or "",
        "primary_code": (report.get("diagnosis_codes") or ["none"])[0],
        "not_completed_count": len(missing),
        "not_completed": list(missing),
        "public_artifact_safe": True,
        "raw_prompt_public": False,
        "raw_result_public": False,
    }


def support_bundle_payload(report: dict[str, Any]) -> dict[str, Any]:
    return support_bundle.sanitize({
        "schema": "sharded_inference_support_bundle_v1",
        "generated_at": utc_now(),
        "ok": report.get("ok"),
        "diagnosis_codes": report.get("diagnosis_codes"),
        "not_completed": report.get("not_completed"),
        "review_summary": report.get("review_summary"),
        "user_status": report.get("user_status"),
        "recommended_next_command": report.get("recommended_next_command"),
        "next_commands": report.get("next_commands"),
        "artifact_summary": report.get("artifact_summary"),
        "output_request": report.get("output_request"),
        "prompt_scope": report.get("prompt_scope"),
        "answer_scope": report.get("answer_scope"),
        "shareable_summary": report.get("shareable_summary"),
        "session": report.get("session"),
        "stage_summary": report.get("stage_summary"),
        "observability": report.get("observability"),
        "safety": report.get("safety"),
        "limitations": report.get("limitations"),
    })


def attach_user_guidance(report: dict[str, Any], args: argparse.Namespace, *, output_dir: Path | None = None) -> dict[str, Any]:
    if output_dir is None:
        output_dir = Path.cwd()
    report.setdefault("output_request", output_request_summary())
    report.setdefault("prompt_scope", prompt_scope_summary(report))
    report.setdefault("answer_scope", answer_scope_summary())
    report.setdefault("shareable_summary", shareable_summary())
    missing = not_completed_items(report)
    report["not_completed"] = missing
    recommended = recommended_next_command(report, args, output_dir=output_dir, missing=missing)
    report["recommended_next_command"] = recommended
    report["next_commands"] = next_commands(report, args, output_dir=output_dir, recommended=recommended)
    report["user_status"] = user_status(report, recommended=recommended, missing=missing)
    report["review_summary"] = review_summary(
        report,
        output_dir=output_dir,
        recommended=recommended,
        missing=missing,
    )
    report["artifact_summary"] = artifact_summary(output_dir)
    return report


def build_report(
    *,
    args: argparse.Namespace,
    session: dict[str, Any],
    state: dict[str, Any],
    stage_processes: list[dict[str, Any]],
    requeue_summary: dict[str, Any],
    ledger_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    session_id = str(session.get("session_id") or "")
    tasks = session_tasks(state, session_id)
    stage_tasks = {int((task.get("workload_metadata") or {}).get("stage_id", -1)): task for task in tasks}
    rows = list(ledger_rows) if ledger_rows is not None else (admin_results(args, limit=10).get("results") or [])
    stage0 = stage_tasks.get(0, {})
    stage1 = stage_tasks.get(1, {})
    stage0_validation = stage0.get("validation") or {}
    stage1_validation = stage1.get("validation") or {}
    raw_public = safe_state_text(state)
    redaction_ok = all(fragment not in raw_public for fragment in [
        "activation_results",
        "activation_result",
        "logits",
        "sharded_inference_result",
        "inference_results",
        "inference_result",
        '"lease_token": "<redacted>"',
    ])
    bundle = state.get("model", {}).get("model_bundle", {})
    read_only = (
        bundle.get("version") == 0
        and bundle.get("optimizer_step") == 0
        and state.get("model", {}).get("global_step") == 0
        and state.get("model_updates") == 0
        and not any(row.get("model_updated") or row.get("model_bundle_updated") for row in rows)
    )
    stage0_ok = stage0.get("status") == "completed" and stage0_validation.get("code") == "ok"
    stage1_ok = stage1.get("status") == "completed" and stage1_validation.get("code") == "ok"
    baseline_match = bool(stage1_validation.get("baseline_match"))
    activation_ready = bool(stage0_validation.get("activation_transport_ready") and stage1_validation.get("activation_transport_ready"))
    codes = []
    if stage0_ok:
        codes.append("stage_0_accepted")
    if stage1_ok:
        codes.append("stage_1_accepted")
    if baseline_match:
        codes.append("baseline_match")
    if activation_ready:
        codes.append("activation_transport_ready")
    if requeue_summary.get("enabled") and requeue_summary.get("rescued_result"):
        codes.append("stage_requeue_ready")
    if stage0_ok and stage1_ok and baseline_match and activation_ready and read_only and redaction_ok:
        codes.append("sharded_inference_ready")
    else:
        if not stage0_ok:
            codes.append("stage_0_missing")
        if not stage1_ok:
            codes.append("stage_1_missing")
        if not baseline_match:
            codes.append("baseline_mismatch")
        if not activation_ready:
            codes.append("activation_transport_failed")
        if not read_only or not redaction_ok:
            codes.append("safety_failed")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": "sharded_inference_ready" in codes and (
            not requeue_summary.get("enabled") or "stage_requeue_ready" in codes
        ),
        "base_url": args.base_url,
        "workload_type": WORKLOAD_TYPE,
        "session": {
            "schema": session.get("schema"),
            "session_id": session_id,
            "stage_count": session.get("stage_count"),
            "stage_0_task_id": stage0.get("task_id") or session.get("stage_0_task_id"),
            "stage_1_task_id": stage1.get("task_id") or session.get("stage_1_task_id"),
            "request_count": session.get("request_count"),
            "scenario_id": session.get("scenario_id"),
        },
        "stage_summary": {
            "stage_0": {
                "task_id": stage0.get("task_id"),
                "miner_id": stage0.get("miner_id"),
                "attempt": stage0.get("attempt"),
                "accepted": stage0_ok,
                "activation_count": stage0_validation.get("activation_count"),
                "activation_bytes": stage0_validation.get("activation_bytes"),
                "activation_hashes": stage0_validation.get("activation_hashes"),
                "elapsed_ms": (stage0.get("metrics") or {}).get("elapsed_ms"),
            },
            "stage_1": {
                "task_id": stage1.get("task_id"),
                "miner_id": stage1.get("miner_id"),
                "attempt": stage1.get("attempt"),
                "accepted": stage1_ok,
                "baseline_match": baseline_match,
                "request_count": stage1_validation.get("request_count"),
                "correct_count": stage1_validation.get("correct_count"),
                "accuracy": stage1_validation.get("accuracy"),
                "request_trace": stage1_validation.get("request_trace"),
                "elapsed_ms": (stage1.get("metrics") or {}).get("elapsed_ms"),
            },
        },
        "observability": {
            "schema": OBSERVABILITY_SCHEMA,
            "stage_count": len(tasks),
            "accepted_ledger_rows": len(rows),
            "miner_ids": sorted({str(task.get("miner_id")) for task in tasks if task.get("miner_id")}),
            "processes": stage_processes,
            "requeue_summary": requeue_summary,
        },
        "diagnosis_codes": sorted(set(codes)),
        "safety": {
            "read_only": read_only,
            "redaction_ok": redaction_ok,
            "raw_activation_redacted": "activation_results" not in raw_public and "logits" not in raw_public,
            "not_production": True,
        },
        "limitations": [
            "CPU-only fixed two-stage pipeline; not production Swarm Inference",
            "Not GPU/TPU pooling, P2P routing, real LLM sharding, or arbitrary prompt serving",
        ],
    }
    return attach_user_guidance(report, args)


def run_evidence(args: argparse.Namespace) -> dict[str, Any]:
    temp_dir = None
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_sharded_inference_")
        state_dir = Path(temp_dir.name)
    registry_path = state_dir / "miner_registry.json"
    stage0_invite = create_invite(
        registry_path=registry_path,
        miner_id=f"{args.miner_prefix}-stage0",
        coordinator_url=args.base_url,
        label="sharded stage 0",
        token=f"{args.invite_token_prefix}-stage0",
        replace=True,
    )
    stage1_invite = create_invite(
        registry_path=registry_path,
        miner_id=f"{args.miner_prefix}-stage1",
        coordinator_url=args.base_url,
        label="sharded stage 1",
        token=f"{args.invite_token_prefix}-stage1",
        replace=True,
    )
    rescue_miner_id = f"{args.miner_prefix}-stage1-rescue" if getattr(args, "stage_mode", "both") == "split" else f"{args.miner_prefix}-rescue"
    if getattr(args, "failure_mode", FAILURE_NONE) in {FAILURE_KILL_STAGE_AFTER_CLAIM, FAILURE_KILL_STAGE0_AFTER_CLAIM}:
        rescue_miner_id = f"{args.miner_prefix}-stage0-rescue" if getattr(args, "stage_mode", "both") == "split" else rescue_miner_id
    rescue_invite = create_invite(
        registry_path=registry_path,
        miner_id=rescue_miner_id,
        coordinator_url=args.base_url,
        label="sharded rescue",
        token=f"{args.invite_token_prefix}-rescue",
        replace=True,
    )
    coordinator = None
    victim_proc: subprocess.Popen | None = None
    process_summaries: list[dict[str, Any]] = []
    failure_stage = 0
    if args.failure_mode == FAILURE_KILL_STAGE1_AFTER_CLAIM:
        failure_stage = 1
    requeue_summary = {
        "enabled": args.failure_mode != FAILURE_NONE,
        "failure_mode": args.failure_mode,
        "victim_stage_id": None,
        "victim_task_id": "",
        "rescue_miner_id": "",
        "lease_expired": False,
        "rescued_result": False,
        "victim_result_accepted": False,
    }
    try:
        coordinator = start_coordinator(args, state_dir, registry_path)
        session = create_session(args)
        stage0_role = "stage0" if getattr(args, "stage_mode", "both") == "split" else ""
        stage1_role = "stage1" if getattr(args, "stage_mode", "both") == "split" else ""
        rescue_role = "stage1" if failure_stage == 1 and getattr(args, "stage_mode", "both") == "split" else stage0_role
        if args.failure_mode in {FAILURE_KILL_STAGE_AFTER_CLAIM, FAILURE_KILL_STAGE0_AFTER_CLAIM}:
            victim_proc = start_invited_miner(
                args,
                stage0_invite,
                compute_seconds=args.victim_compute_seconds,
                stage_role=stage0_role,
            )
            stdout_prefix, claim = wait_for_claim_line(victim_proc, timeout=args.claim_observe_timeout)
            if not claim:
                raise RuntimeError("victim miner did not claim stage-0 task")
            terminate_process_group(victim_proc, sig=signal.SIGTERM)
            victim_process = process_result(
                stage0_invite["miner_id"],
                victim_proc,
                expected_failure=True,
                stdout_prefix=stdout_prefix,
                stage_role=stage0_role,
            )
            process_summaries.append(victim_process)
            victim_proc = None
            lease_expired, _ = wait_for_task_status(
                args,
                str(claim["task_id"]),
                "queued",
                timeout=args.requeue_timeout,
            )
            rescue_process = run_invited_miner(args, rescue_invite, stage_role=rescue_role)
            process_summaries.append(rescue_process)
            requeue_summary.update({
                "victim_stage_id": 0,
                "victim_task_id": str(claim["task_id"]),
                "rescue_miner_id": rescue_invite["miner_id"],
                "lease_expired": lease_expired,
                "rescued_result": rescue_process.get("ok") is True,
                "victim_result_accepted": False,
                "victim_process": victim_process,
                "rescue_process": rescue_process,
            })
        else:
            process_summaries.append(run_invited_miner(args, stage0_invite, stage_role=stage0_role))
        stage1_task_id, _ = wait_for_stage1_task(args, str(session.get("session_id")), timeout=args.stage_timeout)
        if not stage1_task_id:
            raise RuntimeError("stage-1 task was not created")
        if args.failure_mode == FAILURE_KILL_STAGE1_AFTER_CLAIM:
            victim_proc = start_invited_miner(
                args,
                stage1_invite,
                compute_seconds=args.victim_compute_seconds,
                stage_role=stage1_role,
            )
            stdout_prefix, claim = wait_for_claim_line(victim_proc, timeout=args.claim_observe_timeout)
            if not claim:
                raise RuntimeError("victim miner did not claim stage-1 task")
            terminate_process_group(victim_proc, sig=signal.SIGTERM)
            victim_process = process_result(
                stage1_invite["miner_id"],
                victim_proc,
                expected_failure=True,
                stdout_prefix=stdout_prefix,
                stage_role=stage1_role,
            )
            process_summaries.append(victim_process)
            victim_proc = None
            lease_expired, _ = wait_for_task_status(
                args,
                str(claim["task_id"]),
                "queued",
                timeout=args.requeue_timeout,
            )
            rescue_process = run_invited_miner(args, rescue_invite, stage_role=rescue_role)
            process_summaries.append(rescue_process)
            requeue_summary.update({
                "victim_stage_id": 1,
                "victim_task_id": str(claim["task_id"]),
                "rescue_miner_id": rescue_invite["miner_id"],
                "lease_expired": lease_expired,
                "rescued_result": rescue_process.get("ok") is True,
                "victim_result_accepted": False,
                "victim_process": victim_process,
                "rescue_process": rescue_process,
            })
        else:
            process_summaries.append(run_invited_miner(args, stage1_invite, stage_role=stage1_role))
        if WORKLOAD_TYPE == "real_llm_sharded_infer" and args.failure_mode == FAILURE_NONE:
            max_new_tokens = max(1, int(getattr(args, "max_new_tokens", 1)))
            session_id = str(session.get("session_id") or "")
            for generation_step in range(1, max_new_tokens):
                stage0_task_id, _ = wait_for_stage_task(
                    args,
                    session_id,
                    stage_id=0,
                    generation_step=generation_step,
                    timeout=args.stage_timeout,
                )
                if not stage0_task_id:
                    raise RuntimeError(f"stage-0 task for generation step {generation_step} was not created")
                process_summaries.append(run_invited_miner(args, stage0_invite, stage_role=stage0_role))
                stage1_task_id, _ = wait_for_stage_task(
                    args,
                    session_id,
                    stage_id=1,
                    generation_step=generation_step,
                    timeout=args.stage_timeout,
                )
                if not stage1_task_id:
                    raise RuntimeError(f"stage-1 task for generation step {generation_step} was not created")
                process_summaries.append(run_invited_miner(args, stage1_invite, stage_role=stage1_role))
        state = request_json("GET", args.base_url, "/state", observer_token=args.observer_token)
        return build_report(
            args=args,
            session=session,
            state=state,
            stage_processes=process_summaries,
            requeue_summary=requeue_summary,
            ledger_rows=admin_results(args, limit=10).get("results") or [],
        )
    finally:
        if victim_proc is not None:
            terminate_process_group(victim_proc, sig=signal.SIGTERM)
            try:
                victim_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                terminate_process_group(victim_proc, sig=signal.SIGKILL)
                victim_proc.wait(timeout=2.0)
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


def write_json(payload: dict[str, Any], path: str) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    session = report.get("session") or {}
    stage = report.get("stage_summary") or {}
    review = report.get("review_summary") if isinstance(report.get("review_summary"), dict) else {}
    user = report.get("user_status") if isinstance(report.get("user_status"), dict) else {}
    artifacts = report.get("artifact_summary") if isinstance(report.get("artifact_summary"), dict) else {}
    recommended = report.get("recommended_next_command") if isinstance(report.get("recommended_next_command"), dict) else {}
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    lines = [
        "# CrowdTensor Sharded Inference Evidence",
        "",
        f"Schema: `{report.get('schema')}`",
        f"OK: `{report.get('ok')}`",
        f"Session: `{session.get('session_id')}`",
        f"Workload: `{report.get('workload_type')}`",
        f"Scenario: `{session.get('scenario_id')}`",
        "",
        "## Review",
        "",
        f"- status: `{user.get('state')}` {user.get('headline') or ''}",
        f"- state: `{review.get('state')}`",
        f"- next step: `{review.get('next_step')}`",
        f"- inspect first: `{review.get('inspect_first')}`",
        f"- support bundle: `{review.get('support_bundle')}`",
        f"- recommended next: `{recommended.get('command_line') or 'none'}`",
        f"- public artifact safe: `{review.get('public_artifact_safe')}`",
        "",
        "## What To Do Next",
        "",
    ]
    for item in report.get("next_commands") or []:
        if isinstance(item, dict):
            lines.append(f"- {item.get('label')}: `{item.get('command_line')}`")
    if not report.get("next_commands"):
        lines.append("- none")
    lines.extend([
        "",
        "## Artifact Summary",
        "",
        f"- inspect first: `{artifacts.get('inspect_first')}`",
        f"- summary JSON: `{artifacts.get('summary_json')}`",
        f"- support bundle: `{artifacts.get('support_bundle')}`",
        f"- present artifacts: `{artifacts.get('present_artifact_count')}/{artifacts.get('artifact_count')}`",
        "",
        "## Not Completed",
        "",
    ])
    for item in report.get("not_completed") or []:
        lines.append(f"- {item}")
    if not report.get("not_completed"):
        lines.append("- none")
    lines.extend([
        "",
        "## Output Scope",
        "",
        f"- include output: `{output_request.get('include_output')}`",
        f"- prompt scope: `{prompt_scope_text(prompt_scope)}`",
        f"- prompt scope note: {prompt_scope.get('summary') or 'Public artifacts exclude raw prompt text.'}",
        f"- answer scope: `{answer_scope.get('scope_state')}`",
        f"- answer scope note: {answer_scope.get('summary') or 'Public artifacts contain no raw result.'}",
        f"- saved JSON display: `{answer_scope.get('saved_json_display')}`",
        f"- saved Markdown display: `{answer_scope.get('saved_markdown_display')}`",
        f"- shareable: `saved_artifacts={shareable.get('saved_artifacts_public_safe')} raw_prompt_public={shareable.get('raw_prompt_public')} raw_result_public={shareable.get('raw_result_public')} answer_scope_state={shareable.get('answer_scope_state')} local_answer_terminal_only={shareable.get('local_answer_terminal_only')}`",
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []),
        "",
        "## Stages",
        "",
        f"- Stage 0: task `{(stage.get('stage_0') or {}).get('task_id')}`, miner `{(stage.get('stage_0') or {}).get('miner_id')}`, activations `{(stage.get('stage_0') or {}).get('activation_count')}`",
        f"- Stage 1: task `{(stage.get('stage_1') or {}).get('task_id')}`, miner `{(stage.get('stage_1') or {}).get('miner_id')}`, baseline match `{(stage.get('stage_1') or {}).get('baseline_match')}`",
        "",
        "CPU-only fixed two-stage pipeline; not production Swarm Inference, GPU pooling, P2P routing, or real LLM sharding.",
        "",
    ])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CPU-only sharded model-bundle inference evidence.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9820)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--scenario-id", default="route-baseline")
    parser.add_argument("--failure-mode", choices=sorted(FAILURE_MODES), default=FAILURE_NONE)
    parser.add_argument("--stage-mode", choices=["both", "split"], default="both")
    parser.add_argument("--require-distinct-stage-miners", action="store_true")
    parser.add_argument("--miner-prefix", default="shard-miner")
    parser.add_argument("--invite-token-prefix", default="sharded-token")
    parser.add_argument("--lease-seconds", type=float, default=5.0)
    parser.add_argument("--compute-seconds", type=float, default=0.0)
    parser.add_argument("--victim-compute-seconds", type=float, default=8.0)
    parser.add_argument("--heartbeat-interval", type=float, default=0.1)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--miner-timeout", type=float, default=30.0)
    parser.add_argument("--stage-timeout", type=float, default=10.0)
    parser.add_argument("--claim-observe-timeout", type=float, default=5.0)
    parser.add_argument("--requeue-timeout", type=float, default=10.0)
    parser.add_argument("--admin-token", default=DEFAULT_ADMIN_TOKEN)
    parser.add_argument("--observer-token", default=DEFAULT_OBSERVER_TOKEN)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown-out", default="")
    args = parser.parse_args(argv)
    if args.request_count < 1 or args.request_count > 8:
        raise SystemExit("--request-count must be between 1 and 8")
    if args.requeue_timeout <= args.lease_seconds:
        args.requeue_timeout = args.lease_seconds + 5.0
    if args.victim_compute_seconds <= args.lease_seconds:
        args.victim_compute_seconds = args.lease_seconds + 3.0
    args.base_url = f"http://{args.host}:{args.port}"
    return args


def main() -> None:
    try:
        args = parse_args()
        report = run_evidence(args)
        output_dir = Path(args.json_out).resolve().parent if args.json_out else Path.cwd()
        report = attach_user_guidance(report, args, output_dir=output_dir)
        report.setdefault("artifacts", {})
        report["artifacts"]["sharded_inference_evidence_json"] = artifact_entry(
            Path(args.json_out) if args.json_out else output_dir / "sharded_inference_evidence.json",
            output_dir,
            kind="sharded_inference_evidence",
            schema=SCHEMA,
            ok=report.get("ok"),
        )
        report["artifacts"]["sharded_inference_evidence_markdown"] = artifact_entry(
            Path(args.markdown_out) if args.markdown_out else output_dir / "sharded_inference_evidence.md",
            output_dir,
            kind="sharded_inference_evidence_markdown",
        )
        report["artifacts"]["support_bundle_json"] = artifact_entry(
            output_dir / "support_bundle.json",
            output_dir,
            kind="sharded_inference_support_bundle",
            schema="sharded_inference_support_bundle_v1",
        )
        write_json(report, args.json_out)
        if args.markdown_out:
            output = Path(args.markdown_out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(render_markdown(report), encoding="utf-8")
        support_path = output_dir / "support_bundle.json"
        write_json(support_bundle_payload(report), str(support_path))
        if args.json_out:
            report["artifacts"]["sharded_inference_evidence_json"]["present"] = Path(args.json_out).is_file()
        if args.markdown_out:
            report["artifacts"]["sharded_inference_evidence_markdown"]["present"] = Path(args.markdown_out).is_file()
        report["artifacts"]["support_bundle_json"]["present"] = support_path.is_file()
        report["artifact_summary"] = artifact_summary(output_dir)
        write_json(support_bundle_payload(report), str(support_path))
        write_json(report, args.json_out)
        print(json.dumps(report, sort_keys=True))
        raise SystemExit(0 if report.get("ok") else 1)
    except SystemExit:
        raise
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
