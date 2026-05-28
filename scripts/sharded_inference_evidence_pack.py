#!/usr/bin/env python3
"""Run and package a local CPU-only pipeline-sharded inference proof."""

from __future__ import annotations

import argparse
import json
import os
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
    return report


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
    return "\n".join([
        "# CrowdTensor Sharded Inference Evidence",
        "",
        f"Schema: `{report.get('schema')}`",
        f"OK: `{report.get('ok')}`",
        f"Session: `{session.get('session_id')}`",
        f"Workload: `{report.get('workload_type')}`",
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
        write_json(report, args.json_out)
        if args.markdown_out:
            output = Path(args.markdown_out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(render_markdown(report), encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
        raise SystemExit(0 if report.get("ok") else 1)
    except SystemExit:
        raise
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
