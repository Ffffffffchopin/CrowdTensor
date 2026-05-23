#!/usr/bin/env python3
"""Run a controlled local multi-Miner scenario sweep."""

from __future__ import annotations

import argparse
import json
import os
import platform
import select
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
from crowdtensor.model_bundle import (  # noqa: E402
    DEFAULT_INFERENCE_SCENARIO_ID,
    MODEL_BUNDLE_INFERENCE_SCENARIOS,
    inference_scenario_summary,
    normalize_inference_scenario_id,
)
import support_bundle  # noqa: E402


SWEEP_SCHEMA = "multi_miner_scenario_sweep_v1"
OBSERVABILITY_SCHEMA = "multi_miner_scenario_sweep_observability_v1"
WORKLOAD_TYPE = "model_bundle_infer"
DEFAULT_SCENARIOS = ["route-baseline", "gradient-safety", "mixed-prompts"]
DEFAULT_OBSERVER_TOKEN = "multi-miner-sweep-observer"
DEFAULT_ADMIN_TOKEN = "multi-miner-sweep-admin"
DEFAULT_MINER_PREFIX = "sweep-miner"
DEFAULT_INVITE_PREFIX = "multi-miner-sweep-token"
EXECUTION_SEQUENTIAL = "sequential"
EXECUTION_CONCURRENT = "concurrent"
EXECUTION_MODES = {EXECUTION_SEQUENTIAL, EXECUTION_CONCURRENT}
FAILURE_NONE = "none"
FAILURE_KILL_AFTER_CLAIM = "kill-after-claim"
FAILURE_MODES = {FAILURE_NONE, FAILURE_KILL_AFTER_CLAIM}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def request_json(
    method: str,
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    observer_token: str = "",
    admin_token: str = "",
    timeout: float = 5.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json"} if payload is not None else {}
    if observer_token:
        headers["x-crowdtensor-observer-token"] = observer_token
    if admin_token:
        headers["x-crowdtensor-admin-token"] = admin_token
        headers.setdefault("content-type", "application/json")
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


def tail_text(value: str, *, limit: int = 2000) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


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
    raise RuntimeError(f"missing miner summary JSON in stdout:\n{stdout}")


def miner_command(args: argparse.Namespace, invite: dict[str, Any]) -> list[str]:
    compute_seconds = args.compute_seconds
    if invite.get("victim") is True:
        compute_seconds = args.victim_compute_seconds
    return [
        sys.executable,
        str(ROOT / "miner_cli.py"),
        "--coordinator",
        args.base_url,
        "--miner-id",
        invite["miner_id"],
        "--once",
        "--compute-seconds",
        str(compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        "0.2",
    ]


def miner_env(invite: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["CROWDTENSOR_MINER_TOKEN"] = invite["env"]["CROWDTENSOR_MINER_TOKEN"]
    return env


def process_summary_from_completed(
    *,
    miner_id: str,
    returncode: int | None,
    stdout: str,
    stderr: str,
    timed_out: bool = False,
) -> dict[str, Any]:
    miner_summary: dict[str, Any] = {}
    parse_error = ""
    if returncode == 0 and not timed_out:
        try:
            miner_summary = parse_miner_summary(stdout)
        except RuntimeError as exc:
            parse_error = str(exc)
    ok = bool(
        returncode == 0
        and not timed_out
        and int(miner_summary.get("accepted_tasks") or 0) == 1
    )
    summary = {
        "miner_id": miner_id,
        "ok": ok,
        "returncode": returncode,
        "timed_out": timed_out,
        "miner_summary": {
            "accepted_tasks": miner_summary.get("accepted_tasks"),
            "rejected_tasks": miner_summary.get("rejected_tasks"),
            "request_retries": miner_summary.get("request_retries"),
        },
        "stdout_tail": tail_text(stdout),
        "stderr_tail": tail_text(stderr),
    }
    if parse_error:
        summary["parse_error"] = tail_text(parse_error)
    return summary


def run_invited_miner(args: argparse.Namespace, invite: dict[str, Any]) -> dict[str, Any]:
    completed = subprocess.run(
        miner_command(args, invite),
        cwd=ROOT,
        env=miner_env(invite),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.miner_timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"invited miner {invite['miner_id']} failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return parse_miner_summary(completed.stdout)


def start_invited_miner(args: argparse.Namespace, invite: dict[str, Any]) -> subprocess.Popen:
    return subprocess.Popen(
        miner_command(args, invite),
        cwd=ROOT,
        env=miner_env(invite),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def terminate_process_group(proc: subprocess.Popen, *, sig: int = signal.SIGTERM) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


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
        task_id = fields.get("task")
        attempt = fields.get("attempt")
        if task_id:
            return {
                "task_id": task_id,
                "attempt": int(attempt or 0),
                "line": line,
            }
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


def wait_for_concurrent_miners(
    args: argparse.Namespace,
    processes: list[tuple[dict[str, Any], subprocess.Popen]],
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + args.miner_timeout
    summaries: list[dict[str, Any]] = []
    for invite, proc in processes:
        remaining = max(0.1, deadline - time.monotonic())
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = proc.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = proc.communicate(timeout=2.0)
        summaries.append(
            process_summary_from_completed(
                miner_id=str(invite["miner_id"]),
                returncode=proc.returncode,
                stdout=stdout or "",
                stderr=stderr or "",
                timed_out=timed_out,
            )
        )
    return summaries


def wait_for_task_status(
    args: argparse.Namespace,
    task_id: str,
    status: str,
    *,
    timeout: float,
) -> tuple[bool, dict[str, Any]]:
    deadline = time.monotonic() + timeout
    last_state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last_state = request_json("GET", args.base_url, "/state", observer_token=args.observer_token)
        for task in last_state.get("tasks", []):
            if isinstance(task, dict) and task.get("task_id") == task_id and task.get("status") == status:
                return True, last_state
        time.sleep(0.1)
    return False, last_state


def create_inference_session(args: argparse.Namespace, scenario_id: str) -> dict[str, Any]:
    return request_json(
        "POST",
        args.base_url,
        "/admin/inference-sessions",
        payload={"request_count": args.request_count, "scenario_id": scenario_id},
        admin_token=args.admin_token,
    )


def admin_results(
    base_url: str,
    *,
    admin_token: str,
    task_id: str,
    limit: int = 10,
) -> dict[str, Any]:
    query = urlencode({
        "status": "accepted",
        "workload_type": WORKLOAD_TYPE,
        "task_id": task_id,
        "limit": limit,
    })
    return request_json("GET", base_url, f"/admin/results?{query}", admin_token=admin_token)


def row_count(ledger: dict[str, Any]) -> int:
    rows = ledger.get("results") if isinstance(ledger, dict) else []
    return len(rows) if isinstance(rows, list) else 0


def completed_task_for(state: dict[str, Any], task_id: str) -> dict[str, Any] | None:
    for task in state.get("tasks", []):
        if (
            isinstance(task, dict)
            and task.get("task_id") == task_id
            and task.get("status") == "completed"
            and task.get("workload_type") == WORKLOAD_TYPE
        ):
            return task
    return None


def latest_ledger_row(ledger: dict[str, Any]) -> dict[str, Any] | None:
    rows = ledger.get("results") if isinstance(ledger, dict) else []
    if not isinstance(rows, list) or not rows:
        return None
    return rows[0]


def read_only_ok(state: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    model = state.get("model") or {}
    bundle = model.get("model_bundle") or {}
    return (
        model.get("global_step") == 0
        and state.get("model_updates") == 0
        and bundle.get("version") == 0
        and bundle.get("optimizer_step") == 0
        and all(not row.get("model_updated") and not row.get("model_bundle_updated") for row in rows)
    )


def redaction_ok(payload: Any) -> bool:
    encoded = json.dumps(payload, sort_keys=True)
    blocked_fragments = [
        "CROWDTENSOR_MINER_TOKEN",
        "multi-miner-sweep-token",
        "inference_result",
        "inference_results",
    ]
    return all(fragment not in encoded for fragment in blocked_fragments)


def build_session_summary(
    *,
    session: dict[str, Any],
    miner_id: str,
    miner_summary: dict[str, Any],
    state: dict[str, Any],
    ledger: dict[str, Any],
    request_count: int,
) -> dict[str, Any]:
    task_id = str(session.get("task_id") or "")
    expected_scenario = inference_scenario_summary(str(session.get("scenario_id") or ""))
    task = completed_task_for(state, task_id)
    row = latest_ledger_row(ledger)
    actual_miner_id = str((row or {}).get("miner_id") or (task or {}).get("miner_id") or miner_id)
    validation = (row or {}).get("validation") or (task or {}).get("validation") or {}
    metrics = (row or {}).get("session_metrics") or (task or {}).get("metrics") or {}
    expected_scenario_id = expected_scenario.get("scenario_id")
    actual_scenario_id = str(validation.get("scenario_id") or "")
    scenario_matches = bool(expected_scenario_id and actual_scenario_id == expected_scenario_id)
    profile = (state.get("miner_profiles") or {}).get(actual_miner_id) or {}
    capabilities = profile.get("last_capabilities") or {}
    workloads = list(capabilities.get("supported_workloads") or [])
    route_matches = [
        "runtime:python-cli" if profile.get("runtime") == "python-cli" else "",
        "backend:cpu" if profile.get("backend") == "cpu" else "",
        f"workload:{WORKLOAD_TYPE}" if WORKLOAD_TYPE in workloads else "",
        "accepted_result" if task and row else "",
        "validation:ok" if validation.get("code") == "ok" else "",
        "scenario_id" if scenario_matches else "",
    ]
    route_matches = [item for item in route_matches if item]
    missing = []
    if profile.get("runtime") != "python-cli":
        missing.append("runtime:python-cli")
    if profile.get("backend") != "cpu":
        missing.append("backend:cpu")
    if WORKLOAD_TYPE not in workloads:
        missing.append(f"workload:{WORKLOAD_TYPE}")
    if not (task and row):
        missing.append("accepted_result")
    if validation.get("code") != "ok":
        missing.append("validation:ok")
    if not scenario_matches:
        missing.append("scenario_id")
    ok = bool(
        not missing
        and int(validation.get("request_count") or 0) == int(request_count)
        and float(metrics.get("requests_per_second") or 0.0) > 0.0
    )
    return {
        "ok": ok,
        "task_id": task_id,
        "miner_id": actual_miner_id,
        "expected_miner_id": miner_id,
        "workload_type": WORKLOAD_TYPE,
        "route": "local_multi_miner_model_bundle_infer",
        "scenario": expected_scenario,
        "scenario_id": actual_scenario_id or expected_scenario_id,
        "expected_scenario_id": expected_scenario_id,
        "scenario_matches": scenario_matches,
        "request_count": validation.get("request_count"),
        "expected_request_count": request_count,
        "request_trace_count": validation.get("request_trace_count"),
        "accuracy": validation.get("accuracy"),
        "elapsed_ms": metrics.get("elapsed_ms"),
        "requests_per_second": metrics.get("requests_per_second"),
        "validation_code": validation.get("code"),
        "ledger_row_count": row_count(ledger),
        "miner_summary": {
            "accepted_tasks": miner_summary.get("accepted_tasks"),
            "rejected_tasks": miner_summary.get("rejected_tasks"),
            "request_retries": miner_summary.get("request_retries"),
        },
        "profile": {
            "runtime": profile.get("runtime"),
            "backend": profile.get("backend"),
            "accepted": profile.get("accepted"),
            "rejected": profile.get("rejected"),
            "supported_workloads": workloads,
        },
        "matched_capabilities": route_matches,
        "missing_capabilities": missing,
    }


def build_observability(
    *,
    state: dict[str, Any],
    sessions: list[dict[str, Any]],
    read_only: bool,
    redaction: bool,
    execution_mode: str,
    lease_summary: dict[str, Any],
    requeue_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    miners = sorted({str(session.get("miner_id")) for session in sessions if session.get("miner_id")})
    scenarios = sorted({str(session.get("scenario_id")) for session in sessions if session.get("scenario_id")})
    return {
        "schema": OBSERVABILITY_SCHEMA,
        "execution_mode": execution_mode,
        "route": "local_multi_miner_model_bundle_infer",
        "workload_type": WORKLOAD_TYPE,
        "session_count": len(sessions),
        "completed_sessions": sum(1 for session in sessions if session.get("ok")),
        "miner_count": len(miners),
        "miner_ids": miners,
        "scenario_ids": scenarios,
        "task_counts": state.get("task_counts", {}),
        "accepted_results": state.get("accepted_results"),
        "rejected_results": state.get("rejected_results"),
        "read_only": read_only,
        "redaction_ok": redaction,
        "lease_summary": dict(lease_summary),
        "requeue_summary": dict(requeue_summary or {}),
    }


def build_requeue_summary(
    *,
    failure_mode: str,
    victim_miner_id: str = "",
    rescue_miner_id: str = "",
    requeued_task_id: str = "",
    victim_process: dict[str, Any] | None = None,
    rescued_process: dict[str, Any] | None = None,
    lease_expired: bool = False,
    rescued_result: bool = False,
    victim_result_accepted: bool = False,
    requeue_observed_at: str = "",
) -> dict[str, Any]:
    enabled = failure_mode == FAILURE_KILL_AFTER_CLAIM
    return {
        "enabled": enabled,
        "failure_mode": failure_mode,
        "victim_miner_id": victim_miner_id,
        "rescue_miner_id": rescue_miner_id,
        "requeued_task_id": requeued_task_id,
        "lease_expired": bool(lease_expired),
        "rescued_result": bool(rescued_result),
        "victim_result_accepted": bool(victim_result_accepted),
        "requeue_observed_at": requeue_observed_at,
        "victim_process": dict(victim_process or {}),
        "rescued_process": dict(rescued_process or {}),
    }


def build_lease_summary(*, state: dict[str, Any], sessions: list[dict[str, Any]]) -> dict[str, Any]:
    task_ids: list[str] = []
    ledger_row_counts: dict[str, int] = {}
    for session in sessions:
        task_id = str((session.get("session") or {}).get("task_id") or "")
        if not task_id:
            continue
        task_ids.append(task_id)
        ledger_row_counts[task_id] = row_count(session.get("ledger") or {})
    duplicate_result_task_ids = [
        task_id for task_id, count in ledger_row_counts.items()
        if int(count) != 1
    ]
    task_counts = state.get("task_counts") or {}
    queued_remaining = int(task_counts.get("queued") or 0)
    leased_remaining = int(task_counts.get("leased") or 0)
    return {
        "expected_session_count": len(sessions),
        "task_ids": task_ids,
        "unique_task_id_count": len(set(task_ids)),
        "all_task_ids_unique": len(task_ids) == len(set(task_ids)),
        "ledger_row_counts": ledger_row_counts,
        "accepted_ledger_rows": sum(int(value) for value in ledger_row_counts.values()),
        "one_result_per_task": not duplicate_result_task_ids and len(ledger_row_counts) == len(sessions),
        "duplicate_or_missing_result_task_ids": duplicate_result_task_ids,
        "queued_tasks_remaining": queued_remaining,
        "leased_tasks_remaining": leased_remaining,
        "no_queued_or_leased_remaining": queued_remaining == 0 and leased_remaining == 0,
    }


def diagnosis_codes(report: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    sessions = report.get("sessions") if isinstance(report.get("sessions"), list) else []
    if not sessions:
        codes.append("no_sessions")
    if any(session.get("validation_code") != "ok" for session in sessions if isinstance(session, dict)):
        codes.append("validation_failed")
    if any(session.get("scenario_matches") is not True for session in sessions if isinstance(session, dict)):
        codes.append("scenario_mismatch")
    if not (report.get("distribution") or {}).get("all_expected_miners_seen"):
        codes.append("miner_distribution_failed")
    process_summary = report.get("process_summary") if isinstance(report.get("process_summary"), dict) else {}
    if process_summary and not process_summary.get("all_processes_ok"):
        codes.append("miner_process_failed")
    lease_summary = report.get("lease_summary") if isinstance(report.get("lease_summary"), dict) else {}
    if lease_summary and not (
        lease_summary.get("all_task_ids_unique")
        and lease_summary.get("one_result_per_task")
        and lease_summary.get("no_queued_or_leased_remaining")
    ):
        codes.append("lease_race_failed")
    requeue_summary = report.get("requeue_summary") if isinstance(report.get("requeue_summary"), dict) else {}
    if requeue_summary.get("enabled"):
        if not requeue_summary.get("lease_expired"):
            codes.append("requeue_not_observed")
        if not requeue_summary.get("rescued_result"):
            codes.append("rescue_result_missing")
        if requeue_summary.get("victim_result_accepted"):
            codes.append("victim_result_accepted_unexpectedly")
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    if not safety.get("read_only") or not safety.get("redaction_ok") or not safety.get("registry_hashed"):
        codes.append("safety_failed")
    if not codes:
        if requeue_summary.get("enabled"):
            codes.append("multi_miner_requeue_ready")
        elif report.get("execution_mode") == EXECUTION_CONCURRENT:
            codes.append("multi_miner_concurrent_ready")
        else:
            codes.append("multi_miner_sweep_ready")
    return sorted(set(codes))


def sanitize_public_report(value: Any) -> Any:
    sensitive_keys = {
        "lease_token",
        "idempotency_key",
        "result_idempotency_key_hash",
        "result_lease_token_hash",
        "token",
        "secret",
        "weights",
        "delta",
        "local_delta",
        "pseudo_gradient",
        "compressed_delta",
        "inference_result",
        "inference_results",
    }
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in sensitive_keys:
                sanitized[str(key)] = "<redacted>"
            else:
                sanitized[str(key)] = sanitize_public_report(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_public_report(item) for item in value]
    return value


def build_report(
    *,
    base_url: str,
    state: dict[str, Any],
    sessions: list[dict[str, Any]],
    invites: list[dict[str, Any]],
    request_count: int,
    execution_mode: str = EXECUTION_SEQUENTIAL,
    process_summaries: list[dict[str, Any]] | None = None,
    failure_mode: str = FAILURE_NONE,
    requeue_summary: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    rows = [latest_ledger_row(session.get("ledger") or {}) for session in sessions]
    ledger_rows = [row for row in rows if isinstance(row, dict)]
    session_summaries = [
        build_session_summary(
            session=session["session"],
            miner_id=session["miner_id"],
            miner_summary=session.get("miner_summary") or {},
            state=state,
            ledger=session.get("ledger") or {},
            request_count=request_count,
        )
        for session in sessions
    ]
    expected_miners = [str(invite.get("miner_id")) for invite in invites if invite.get("miner_id")]
    actual_miners = sorted({str(session.get("miner_id")) for session in session_summaries if session.get("miner_id")})
    registry_hashed = all(str(invite.get("token_hash", "")).startswith("sha256:") for invite in invites)
    read_only = read_only_ok(state, ledger_rows)
    lease_summary = build_lease_summary(state=state, sessions=sessions)
    requeue = requeue_summary or build_requeue_summary(failure_mode=failure_mode)
    expected_miners_for_distribution = list(expected_miners)
    if requeue.get("enabled") and requeue.get("victim_miner_id"):
        expected_miners_for_distribution = [
            miner_id for miner_id in expected_miners_for_distribution
            if miner_id != str(requeue.get("victim_miner_id"))
        ]
    distribution = {
        "expected_miner_ids": expected_miners_for_distribution,
        "actual_miner_ids": actual_miners,
        "distinct_miner_count": len(actual_miners),
        "all_expected_miners_seen": sorted(expected_miners_for_distribution) == actual_miners,
    }
    processes = list(process_summaries or [])
    process_ok = [
        item.get("ok") is True or item.get("expected_failure") is True
        for item in processes
    ]
    process_summary = {
        "started": len(processes),
        "completed": sum(1 for item in processes if item.get("returncode") is not None),
        "ok": sum(1 for item in processes if item.get("ok") is True),
        "expected_failures": sum(1 for item in processes if item.get("expected_failure") is True),
        "all_processes_ok": all(process_ok) if processes else True,
        "miners": processes,
    }
    public_probe = {
        "sessions": session_summaries,
        "distribution": distribution,
        "lease_summary": lease_summary,
        "requeue_summary": requeue,
        "process_summary": process_summary,
        "coordinator": {
            "task_counts": state.get("task_counts", {}),
            "accepted_results": state.get("accepted_results"),
            "rejected_results": state.get("rejected_results"),
        },
    }
    redaction = redaction_ok(public_probe)
    safety = {
        "read_only": read_only,
        "redaction_ok": redaction,
        "registry_hashed": registry_hashed,
        "raw_payloads_exposed": not redaction,
    }
    report = {
        "schema": SWEEP_SCHEMA,
        "generated_at": generated_at or utc_now(),
        "ok": False,
        "execution_mode": execution_mode,
        "failure_mode": failure_mode,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
        "coordinator": {
            "url": base_url,
            "task_counts": state.get("task_counts", {}),
            "accepted_results": state.get("accepted_results"),
            "rejected_results": state.get("rejected_results"),
        },
        "workload": {
            "type": WORKLOAD_TYPE,
            "request_count": request_count,
            "scenario_schema": "model_bundle_inference_scenario_v1",
            "scenario_ids": [str(session.get("scenario_id")) for session in session_summaries],
        },
        "sessions": session_summaries,
        "distribution": distribution,
        "lease_summary": lease_summary,
        "requeue_summary": requeue,
        "process_summary": process_summary,
        "safety": safety,
        "observability_summary": {},
        "diagnosis_codes": [],
        "limitations": [
            "Controlled local multi-Miner loopback evidence; not P2P routing or NAT traversal",
            "CPU-only model_bundle_infer sweep; not production LLM serving or GPU pooling",
            "Concurrent mode proves local lease uniqueness, not production throughput scaling",
            "Kill-after-claim mode proves local lease timeout rescue, not remote machine fault tolerance",
        ],
        "recommended_next_commands": [
            "python3 scripts/multi_miner_scenario_sweep_check.py --port 8916 --execution-mode concurrent",
            "python3 scripts/multi_miner_scenario_sweep_check.py --port 8916 --execution-mode concurrent --failure-mode kill-after-claim",
            "python3 scripts/runtime_acceptance_pack.py --include-multi-miner-sweep --include-multi-miner-requeue --base-port 8910",
        ],
    }
    report["observability_summary"] = build_observability(
        state=state,
        sessions=session_summaries,
        read_only=read_only,
        redaction=redaction,
        execution_mode=execution_mode,
        lease_summary=lease_summary,
        requeue_summary=requeue,
    )
    report["ok"] = bool(
        all(session.get("ok") for session in session_summaries)
        and distribution["all_expected_miners_seen"]
        and len(actual_miners) == len(session_summaries)
        and process_summary["all_processes_ok"]
        and lease_summary["all_task_ids_unique"]
        and lease_summary["one_result_per_task"]
        and lease_summary["no_queued_or_leased_remaining"]
        and safety["read_only"]
        and safety["redaction_ok"]
        and safety["registry_hashed"]
        and (
            not requeue.get("enabled")
            or (
                requeue.get("lease_expired")
                and requeue.get("rescued_result")
                and not requeue.get("victim_result_accepted")
            )
        )
    )
    report["diagnosis_codes"] = diagnosis_codes(report)
    return sanitize_public_report(report)


def run_local_loopback(args: argparse.Namespace) -> dict[str, Any]:
    temp_dir = None
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_multi_miner_sweep_")
        state_dir = Path(temp_dir.name)
    state_dir.mkdir(parents=True, exist_ok=True)
    registry_path = state_dir / "miner_registry.json"
    coordinator = None
    try:
        invites = []
        for index in range(len(args.scenario_ids)):
            miner_id = f"{args.miner_prefix}-{index + 1}"
            invites.append(
                create_invite(
                    registry_path=registry_path,
                    miner_id=miner_id,
                    coordinator_url=args.base_url,
                    label="multi Miner scenario sweep",
                    token=f"{args.invite_token_prefix}-{index + 1}",
                    replace=True,
                )
            )
        coordinator = start_coordinator(args, state_dir, registry_path)
        runs: list[dict[str, Any]] = []
        process_summaries: list[dict[str, Any]] = []
        sessions: list[dict[str, Any]] = []
        for scenario_id in args.scenario_ids:
            session = create_inference_session(args, scenario_id)
            if session.get("schema") != "inference_session_request_v1" or not session.get("task_id"):
                raise RuntimeError(f"invalid inference session response: {session}")
            sessions.append(session)

        if args.execution_mode == EXECUTION_CONCURRENT:
            processes = [(invite, start_invited_miner(args, invite)) for invite in invites]
            process_summaries = wait_for_concurrent_miners(args, processes)
            process_by_miner = {
                str(item.get("miner_id")): item
                for item in process_summaries
            }
            for session in sessions:
                ledger = admin_results(
                    args.base_url,
                    admin_token=args.admin_token,
                    task_id=str(session["task_id"]),
                )
                row = latest_ledger_row(ledger) or {}
                miner_id = str(row.get("miner_id") or "")
                process = process_by_miner.get(miner_id, {})
                runs.append({
                    "session": session,
                    "miner_id": miner_id,
                    "miner_summary": process.get("miner_summary") or {},
                    "ledger": ledger,
                })
        else:
            for session, invite in zip(sessions, invites):
                miner_summary = run_invited_miner(args, invite)
                ledger = admin_results(
                    args.base_url,
                    admin_token=args.admin_token,
                    task_id=str(session["task_id"]),
                )
                runs.append({
                    "session": session,
                    "miner_id": invite["miner_id"],
                    "miner_summary": miner_summary,
                    "ledger": ledger,
                })

        state = request_json("GET", args.base_url, "/state", observer_token=args.observer_token)
        return build_report(
            base_url=args.base_url,
            state=state,
            sessions=runs,
            invites=invites,
            request_count=args.request_count,
            execution_mode=args.execution_mode,
            process_summaries=process_summaries,
            failure_mode=args.failure_mode,
        )
    finally:
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


def run_kill_after_claim(args: argparse.Namespace) -> dict[str, Any]:
    temp_dir = None
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_multi_miner_requeue_")
        state_dir = Path(temp_dir.name)
    state_dir.mkdir(parents=True, exist_ok=True)
    registry_path = state_dir / "miner_registry.json"
    coordinator = None
    victim_proc: subprocess.Popen | None = None
    try:
        invites = []
        for index in range(len(args.scenario_ids)):
            miner_id = f"{args.miner_prefix}-{index + 1}"
            invites.append(
                create_invite(
                    registry_path=registry_path,
                    miner_id=miner_id,
                    coordinator_url=args.base_url,
                    label="multi Miner requeue sweep",
                    token=f"{args.invite_token_prefix}-{index + 1}",
                    replace=True,
                )
            )
        rescue_invite = create_invite(
            registry_path=registry_path,
            miner_id=f"{args.miner_prefix}-rescue",
            coordinator_url=args.base_url,
            label="multi Miner requeue rescue",
            token=f"{args.invite_token_prefix}-rescue",
            replace=True,
        )
        report_invites = invites + [rescue_invite]
        coordinator = start_coordinator(args, state_dir, registry_path)

        sessions: list[dict[str, Any]] = []
        for scenario_id in args.scenario_ids:
            session = create_inference_session(args, scenario_id)
            if session.get("schema") != "inference_session_request_v1" or not session.get("task_id"):
                raise RuntimeError(f"invalid inference session response: {session}")
            sessions.append(session)

        victim_invite = {**invites[0], "victim": True}
        victim_proc = start_invited_miner(args, victim_invite)
        victim_stdout, claim = wait_for_claim_line(victim_proc, timeout=args.claim_observe_timeout)
        requeued_task_id = str(claim.get("task_id") or "")
        terminate_process_group(victim_proc, sig=signal.SIGTERM)
        try:
            victim_tail, victim_stderr = victim_proc.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            terminate_process_group(victim_proc, sig=signal.SIGKILL)
            victim_tail, victim_stderr = victim_proc.communicate(timeout=2.0)
        victim_stdout += victim_tail or ""
        victim_process = process_summary_from_completed(
            miner_id=str(victim_invite["miner_id"]),
            returncode=victim_proc.returncode,
            stdout=victim_stdout,
            stderr=victim_stderr or "",
            timed_out=False,
        )
        victim_process["killed_after_claim"] = bool(requeued_task_id)
        victim_process["claimed_task_id"] = requeued_task_id
        victim_process["expected_failure"] = True
        victim_proc = None

        lease_expired = False
        requeue_state: dict[str, Any] = {}
        requeue_observed_at = ""
        if requeued_task_id:
            lease_expired, requeue_state = wait_for_task_status(
                args,
                requeued_task_id,
                "queued",
                timeout=args.requeue_timeout,
            )
            if lease_expired:
                requeue_observed_at = utc_now()

        rescue_summaries = wait_for_concurrent_miners(
            args,
            [(rescue_invite, start_invited_miner(args, rescue_invite))],
        )
        rescue_summaries.extend(
            wait_for_concurrent_miners(
                args,
                [(invite, start_invited_miner(args, invite)) for invite in invites[1:]],
            )
        )
        process_summaries = [victim_process] + rescue_summaries
        process_by_miner = {
            str(item.get("miner_id")): item
            for item in rescue_summaries
        }

        runs: list[dict[str, Any]] = []
        rescued_result = False
        victim_result_accepted = False
        rescue_miner_id = str(rescue_invite["miner_id"])
        for session in sessions:
            ledger = admin_results(
                args.base_url,
                admin_token=args.admin_token,
                task_id=str(session["task_id"]),
            )
            row = latest_ledger_row(ledger) or {}
            miner_id = str(row.get("miner_id") or "")
            if str(session["task_id"]) == requeued_task_id:
                rescued_result = bool(row and miner_id == rescue_miner_id)
                victim_result_accepted = bool(row and miner_id == victim_invite["miner_id"])
            process = process_by_miner.get(miner_id, {})
            runs.append({
                "session": session,
                "miner_id": miner_id,
                "miner_summary": process.get("miner_summary") or {},
                "ledger": ledger,
            })

        state = request_json("GET", args.base_url, "/state", observer_token=args.observer_token)
        if not requeue_state:
            requeue_state = state
        requeue_summary = build_requeue_summary(
            failure_mode=args.failure_mode,
            victim_miner_id=str(victim_invite["miner_id"]),
            rescue_miner_id=rescue_miner_id,
            requeued_task_id=requeued_task_id,
            victim_process=victim_process,
            rescued_process=process_by_miner.get(rescue_miner_id, {}),
            lease_expired=lease_expired,
            rescued_result=rescued_result,
            victim_result_accepted=victim_result_accepted,
            requeue_observed_at=requeue_observed_at,
        )
        return build_report(
            base_url=args.base_url,
            state=state,
            sessions=runs,
            invites=report_invites,
            request_count=args.request_count,
            execution_mode=args.execution_mode,
            process_summaries=process_summaries,
            failure_mode=args.failure_mode,
            requeue_summary=requeue_summary,
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


def parse_scenario_ids(value: str) -> list[str]:
    raw = [item.strip() for item in str(value or "").split(",") if item.strip()]
    scenario_ids = raw or list(DEFAULT_SCENARIOS)
    normalized: list[str] = []
    for scenario_id in scenario_ids:
        normalized.append(normalize_inference_scenario_id(scenario_id) or DEFAULT_INFERENCE_SCENARIO_ID)
    if len(normalized) < 2:
        raise ValueError("at least two scenario ids are required")
    return normalized


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a controlled local multi-Miner scenario sweep.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8916)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--scenario-ids", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--execution-mode", choices=sorted(EXECUTION_MODES), default=EXECUTION_SEQUENTIAL)
    parser.add_argument("--failure-mode", choices=sorted(FAILURE_MODES), default=FAILURE_NONE)
    parser.add_argument("--miner-prefix", default=DEFAULT_MINER_PREFIX)
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--compute-seconds", type=float, default=0.2)
    parser.add_argument("--victim-compute-seconds", type=float, default=5.0)
    parser.add_argument("--claim-observe-timeout", type=float, default=5.0)
    parser.add_argument("--requeue-timeout", type=float, default=10.0)
    parser.add_argument("--heartbeat-interval", type=float, default=0.1)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--miner-timeout", type=float, default=30.0)
    parser.add_argument("--observer-token", default=DEFAULT_OBSERVER_TOKEN)
    parser.add_argument("--admin-token", default=DEFAULT_ADMIN_TOKEN)
    parser.add_argument("--invite-token-prefix", default=DEFAULT_INVITE_PREFIX)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)
    if args.request_count < 1 or args.request_count > 8:
        raise SystemExit("--request-count must be between 1 and 8")
    if args.failure_mode == FAILURE_KILL_AFTER_CLAIM:
        args.execution_mode = EXECUTION_CONCURRENT
        if args.lease_seconds > args.victim_compute_seconds:
            args.victim_compute_seconds = args.lease_seconds + 2.0
        if args.requeue_timeout <= args.lease_seconds:
            args.requeue_timeout = args.lease_seconds + 5.0
    if args.victim_compute_seconds <= 0:
        raise SystemExit("--victim-compute-seconds must be positive")
    if args.claim_observe_timeout <= 0:
        raise SystemExit("--claim-observe-timeout must be positive")
    if args.requeue_timeout <= 0:
        raise SystemExit("--requeue-timeout must be positive")
    try:
        args.scenario_ids = parse_scenario_ids(args.scenario_ids)
    except ValueError as exc:
        allowed = ", ".join(sorted(MODEL_BUNDLE_INFERENCE_SCENARIOS))
        raise SystemExit(f"{exc}; allowed scenarios: {allowed}") from exc
    args.base_url = f"http://{args.host}:{args.port}"
    return args


def main() -> None:
    try:
        args = parse_args()
        if args.failure_mode == FAILURE_KILL_AFTER_CLAIM:
            payload = run_kill_after_claim(args)
        else:
            payload = run_local_loopback(args)
        write_json(payload, args.json_out)
        print(json.dumps(payload, sort_keys=True))
        raise SystemExit(0 if payload.get("ok") else 1)
    except SystemExit:
        raise
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
