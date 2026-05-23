#!/usr/bin/env python3
"""Build a safe, shareable remote-compute inference evidence report."""

from __future__ import annotations

import argparse
import json
import os
import platform
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
    inference_scenario_summary,
    normalize_inference_scenario_id,
)
import support_bundle  # noqa: E402


EVIDENCE_SCHEMA = "remote_compute_evidence_v1"
WORKLOAD_TYPE = "model_bundle_infer"
DEFAULT_MINER_ID = "remote-evidence-miner"
DEFAULT_OBSERVER_TOKEN = "remote-evidence-observer"
DEFAULT_ADMIN_TOKEN = "remote-evidence-admin"
DEFAULT_INVITE_TOKEN = "remote-evidence-token"


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
            health = request_json("GET", base_url, "/health", timeout=2.0)
            if health.get("ok") is True:
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


def run_invited_miner(args: argparse.Namespace, invite: dict[str, Any]) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["CROWDTENSOR_MINER_TOKEN"] = invite["env"]["CROWDTENSOR_MINER_TOKEN"]
    command = [
        sys.executable,
        str(ROOT / "miner_cli.py"),
        "--coordinator",
        args.base_url,
        "--miner-id",
        invite["miner_id"],
        "--once",
        "--compute-seconds",
        str(args.compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        "0.2",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.miner_timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "invited miner failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return parse_miner_summary(completed.stdout)


def admin_results(
    base_url: str,
    *,
    admin_token: str,
    miner_id: str,
    workload_type: str,
    limit: int = 10,
    task_id: str = "",
) -> dict[str, Any]:
    query_params = {
        "status": "accepted",
        "workload_type": workload_type,
        "limit": limit,
    }
    if task_id:
        query_params["task_id"] = task_id
    else:
        query_params["miner_id"] = miner_id
    query = urlencode(query_params)
    return request_json("GET", base_url, f"/admin/results?{query}", admin_token=admin_token)


def create_inference_session(args: argparse.Namespace) -> dict[str, Any]:
    return request_json(
        "POST",
        args.base_url,
        "/admin/inference-sessions",
        payload={"request_count": args.request_count, "scenario_id": args.scenario_id},
        admin_token=args.admin_token,
    )


def completed_task_for(state: dict[str, Any], miner_id: str, *, task_id: str = "") -> dict[str, Any] | None:
    completed = [
        task for task in state.get("tasks", [])
        if (
            task.get("status") == "completed"
            and task.get("workload_type") == WORKLOAD_TYPE
            and (task.get("task_id") == task_id if task_id else task.get("miner_id") == miner_id)
        )
    ]
    return completed[-1] if completed else None


def latest_ledger_row(ledger: dict[str, Any]) -> dict[str, Any] | None:
    rows = ledger.get("results") if isinstance(ledger, dict) else []
    if not isinstance(rows, list) or not rows:
        return None
    return rows[0]


def safe_profile(profile: dict[str, Any]) -> dict[str, Any]:
    capabilities = profile.get("last_capabilities") or {}
    hardware = capabilities.get("hardware_profile") or {}
    return {
        "runtime": profile.get("runtime"),
        "backend": profile.get("backend"),
        "accepted": profile.get("accepted"),
        "rejected": profile.get("rejected"),
        "hardware_profile": {
            "os": hardware.get("os"),
            "platform": hardware.get("platform"),
            "machine": hardware.get("machine"),
            "cpu_count": hardware.get("cpu_count"),
            "python_version": hardware.get("python_version"),
        },
        "supported_workloads": list(capabilities.get("supported_workloads") or []),
        "last_runtime_status": profile.get("last_runtime_status") or {},
    }


def route_decision(*, profile: dict[str, Any], has_task: bool) -> dict[str, Any]:
    capabilities = profile.get("last_capabilities") or {}
    supported = set(capabilities.get("supported_workloads") or [])
    matched: list[str] = []
    missing: list[str] = []
    if profile.get("runtime") == "python-cli":
        matched.append("runtime:python-cli")
    else:
        missing.append("runtime:python-cli")
    if profile.get("backend") == "cpu":
        matched.append("backend:cpu")
    else:
        missing.append("backend:cpu")
    if WORKLOAD_TYPE in supported:
        matched.append(f"workload:{WORKLOAD_TYPE}")
    else:
        missing.append(f"workload:{WORKLOAD_TYPE}")
    if has_task:
        matched.append("completed_result")
    else:
        missing.append("completed_result")
    return {
        "name": "remote_python_model_bundle_infer",
        "target": "remote_python_miner",
        "workload": WORKLOAD_TYPE,
        "status": "available" if not missing else "blocked",
        "usable_now": not missing,
        "confidence": "ready" if not missing else "blocked",
        "reason": "remote Python Miner completed read-only model_bundle_infer"
        if not missing else
        "remote Python Miner has not completed a compatible model_bundle_infer task",
        "matched_capabilities": matched,
        "missing_capabilities": missing,
    }


def build_inference_summary(
    *,
    task: dict[str, Any] | None,
    row: dict[str, Any] | None,
    request_count: int,
    scenario_id: str,
) -> dict[str, Any]:
    validation = (task or {}).get("validation") or (row or {}).get("validation") or {}
    metrics = (row or {}).get("session_metrics") or (task or {}).get("metrics") or {}
    expected_scenario = inference_scenario_summary(scenario_id)
    actual_scenario_id = str(validation.get("scenario_id") or "")
    expected_scenario_id = expected_scenario.get("scenario_id")
    scenario_matches = bool(expected_scenario_id and actual_scenario_id == expected_scenario_id)
    return {
        "present": bool(task and row),
        "ok": validation.get("code") == "ok",
        "workload_type": WORKLOAD_TYPE if task or row else None,
        "request_count": validation.get("request_count"),
        "expected_request_count": request_count,
        "scenario_schema": validation.get("scenario_schema") or expected_scenario.get("scenario_schema"),
        "scenario_id": actual_scenario_id or expected_scenario_id,
        "scenario_description": validation.get("scenario_description") or expected_scenario.get("scenario_description"),
        "scenario_request_count": validation.get("scenario_request_count") or expected_scenario.get("scenario_request_count"),
        "expected_scenario_id": expected_scenario_id,
        "scenario_matches": scenario_matches,
        "correct_count": validation.get("correct_count"),
        "accuracy": validation.get("accuracy"),
        "elapsed_ms": metrics.get("elapsed_ms"),
        "requests_per_second": metrics.get("requests_per_second"),
        "request_trace_count": validation.get("request_trace_count"),
        "request_trace_truncated": bool(validation.get("request_trace_truncated", False)),
        "predicted_token": validation.get("predicted_token"),
        "target_token": validation.get("target_token"),
    }


def build_observability_summary(
    *,
    mode: str,
    route: dict[str, Any],
    profile: dict[str, Any],
    state: dict[str, Any],
    ledger: dict[str, Any],
    summary: dict[str, Any],
    safety: dict[str, Any],
    miner_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": "remote_compute_observability_v1",
        "mode": mode,
        "route": {
            "name": route.get("name"),
            "confidence": route.get("confidence"),
            "usable_now": route.get("usable_now"),
            "matched_capabilities": list(route.get("matched_capabilities") or []),
            "missing_capabilities": list(route.get("missing_capabilities") or []),
        },
        "miner": {
            "runtime": profile.get("runtime"),
            "backend": profile.get("backend"),
            "accepted": profile.get("accepted"),
            "rejected": profile.get("rejected"),
            "summary_accepted_tasks": (miner_summary or {}).get("accepted_tasks"),
            "summary_rejected_tasks": (miner_summary or {}).get("rejected_tasks"),
        },
        "work_queue": {
            "task_counts": state.get("task_counts", {}),
            "accepted_results": state.get("accepted_results"),
            "rejected_results": state.get("rejected_results"),
            "ledger_rows": len(ledger.get("results") or []),
        },
        "inference": {
            "ok": summary.get("ok"),
            "request_count": summary.get("request_count"),
            "expected_request_count": summary.get("expected_request_count"),
            "scenario_schema": summary.get("scenario_schema"),
            "scenario_id": summary.get("scenario_id"),
            "expected_scenario_id": summary.get("expected_scenario_id"),
            "scenario_matches": summary.get("scenario_matches"),
            "request_trace_count": summary.get("request_trace_count"),
            "accuracy": summary.get("accuracy"),
            "elapsed_ms": summary.get("elapsed_ms"),
            "requests_per_second": summary.get("requests_per_second"),
        },
        "safety": {
            "read_only": safety.get("read_only"),
            "redaction_ok": safety.get("redaction_ok"),
            "registry_hashed": safety.get("registry_hashed"),
            "raw_payloads_exposed": safety.get("raw_payloads_exposed"),
        },
    }


def read_only_ok(state: dict[str, Any], row: dict[str, Any] | None) -> bool:
    bundle = (state.get("model") or {}).get("model_bundle") or {}
    return (
        bundle.get("version") == 0
        and bundle.get("optimizer_step") == 0
        and (state.get("model") or {}).get("global_step") == 0
        and state.get("model_updates") == 0
        and not (row or {}).get("model_updated")
        and not (row or {}).get("model_bundle_updated")
    )


def build_evidence(
    *,
    mode: str,
    base_url: str,
    miner_id: str,
    state: dict[str, Any],
    ledger: dict[str, Any],
    miner_summary: dict[str, Any] | None,
    request_count: int,
    invite: dict[str, Any] | None = None,
    registry_path: Path | None = None,
    scenario_id: str = DEFAULT_INFERENCE_SCENARIO_ID,
    task_id: str = "",
    generated_at: str | None = None,
) -> dict[str, Any]:
    scenario_id = normalize_inference_scenario_id(scenario_id) or DEFAULT_INFERENCE_SCENARIO_ID
    scenario = inference_scenario_summary(scenario_id)
    task = completed_task_for(state, miner_id, task_id=task_id)
    row = latest_ledger_row(ledger)
    profile = (state.get("miner_profiles") or {}).get(miner_id) or {}
    validation = (task or {}).get("validation") or (row or {}).get("validation") or {}
    trace = list(validation.get("request_trace") or [])
    route = route_decision(profile=profile, has_task=bool(task and row))
    registry_hashed = bool(invite and str(invite.get("token_hash", "")).startswith("sha256:"))
    if mode == "collect":
        registry_hashed = None
    raw_tasks = json.dumps(state.get("tasks", []), sort_keys=True)
    raw_ledger = json.dumps(ledger, sort_keys=True)
    redaction_ok = all(fragment not in raw_tasks and fragment not in raw_ledger for fragment in [
        "inference_result",
        "inference_results",
        "CROWDTENSOR_MINER_TOKEN",
        "remote-evidence-token",
    ])
    read_only = read_only_ok(state, row)
    summary = build_inference_summary(
        task=task,
        row=row,
        request_count=request_count,
        scenario_id=scenario_id,
    )
    safety = {
        "read_only": read_only,
        "redaction_ok": redaction_ok,
        "raw_payloads_exposed": not redaction_ok,
        "registry_hashed": registry_hashed,
    }
    observability = build_observability_summary(
        mode=mode,
        route=route,
        profile=profile,
        state=state,
        ledger=ledger,
        summary=summary,
        safety=safety,
        miner_summary=miner_summary,
    )
    ok = bool(
        route.get("usable_now")
        and summary.get("ok")
        and int(summary.get("request_count") or 0) == int(request_count)
        and summary.get("scenario_matches")
        and float(summary.get("requests_per_second") or 0.0) > 0.0
        and read_only
        and redaction_ok
        and (registry_hashed is not False)
    )
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "generated_at": generated_at or utc_now(),
        "mode": mode,
        "ok": ok,
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
        "miner": {
            "miner_id": miner_id,
            "profile": safe_profile(profile),
            "summary": miner_summary or {},
        },
        "invite": {
            "present": bool(invite),
            "registry": str(registry_path) if registry_path else "",
            "token_hash_prefix": str((invite or {}).get("token_hash", "")).split(":", 1)[0] if invite else "",
            "registry_hashed": registry_hashed,
        },
        "workload": {
            "type": WORKLOAD_TYPE,
            "request_count": request_count,
            "scenario_schema": scenario.get("scenario_schema"),
            "scenario_id": scenario.get("scenario_id"),
            "scenario_description": scenario.get("scenario_description"),
            "scenario_request_count": scenario.get("scenario_request_count"),
            "task_id": (task or {}).get("task_id"),
            "expected_task_id": task_id,
            "bundle_id": ((state.get("model") or {}).get("model_bundle") or {}).get("bundle_id"),
            "bundle_version": ((state.get("model") or {}).get("model_bundle") or {}).get("version"),
        },
        "route_decision": route,
        "inference_summary": summary,
        "observability_summary": observability,
        "request_trace": trace,
        "ledger_summary": {
            "rows": len(ledger.get("results") or []),
            "event_index": (row or {}).get("event_index"),
            "status": (row or {}).get("status"),
            "model_updated": bool((row or {}).get("model_updated")),
            "model_bundle_updated": bool((row or {}).get("model_bundle_updated")),
        },
        "safety": safety,
        "recommended_next_commands": [
            "python3 scripts/remote_compute_evidence_check.py --port 8912",
            "python3 scripts/runtime_acceptance_pack.py --include-remote-evidence --base-port 8910",
            "python3 scripts/remote_miner_readiness_check.py --port 8899",
        ],
        "limitations": [
            "Controlled remote Miner evidence; not public-internet hardening",
            "CPU-only model_bundle_infer evidence; not production LLM serving or GPU pooling",
            "No P2P/NAT traversal, decentralized identity, or incentives are claimed",
        ],
    }
    return support_bundle.sanitize(evidence)


def run_local_loopback(args: argparse.Namespace) -> dict[str, Any]:
    temp_dir = None
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_remote_evidence_")
        state_dir = Path(temp_dir.name)
    state_dir.mkdir(parents=True, exist_ok=True)
    registry_path = state_dir / "miner_registry.json"

    coordinator = None
    try:
        invite = create_invite(
            registry_path=registry_path,
            miner_id=args.miner_id,
            coordinator_url=args.base_url,
            label="remote compute evidence",
            token=args.invite_token,
            replace=True,
        )
        coordinator = start_coordinator(args, state_dir, registry_path)
        session = create_inference_session(args)
        task_id = str(session.get("task_id") or "")
        if not task_id:
            raise RuntimeError(f"admin inference session did not return task_id: {session}")
        miner_summary = run_invited_miner(args, invite)
        state = request_json("GET", args.base_url, "/state", observer_token=args.observer_token)
        ledger = admin_results(
            args.base_url,
            admin_token=args.admin_token,
            miner_id=args.miner_id,
            workload_type=WORKLOAD_TYPE,
            task_id=task_id,
        )
        return build_evidence(
            mode="local-loopback",
            base_url=args.base_url,
            miner_id=args.miner_id,
            state=state,
            ledger=ledger,
            miner_summary=miner_summary,
            request_count=args.request_count,
            invite=invite,
            registry_path=registry_path,
            scenario_id=args.scenario_id,
            task_id=task_id,
        )
    finally:
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


def run_collect(args: argparse.Namespace) -> dict[str, Any]:
    state = request_json("GET", args.coordinator_url, "/state", observer_token=args.observer_token)
    ledger = admin_results(
        args.coordinator_url,
        admin_token=args.admin_token,
        miner_id=args.miner_id,
        workload_type=WORKLOAD_TYPE,
    )
    return build_evidence(
        mode="collect",
        base_url=args.coordinator_url,
        miner_id=args.miner_id,
        state=state,
        ledger=ledger,
        miner_summary=None,
        request_count=args.request_count,
        scenario_id=args.scenario_id,
    )


def render_markdown(payload: dict[str, Any]) -> str:
    route = payload.get("route_decision") or {}
    summary = payload.get("inference_summary") or {}
    observability = payload.get("observability_summary") or {}
    observed_queue = observability.get("work_queue") or {}
    observed_inference = observability.get("inference") or {}
    observed_safety = observability.get("safety") or {}
    miner = payload.get("miner") or {}
    profile = miner.get("profile") or {}
    safety = payload.get("safety") or {}
    lines = [
        "# CrowdTensor Remote Compute Evidence",
        "",
        f"Generated: `{payload.get('generated_at', '')}`",
        f"Mode: `{payload.get('mode', '')}`",
        f"OK: `{payload.get('ok')}`",
        "",
        "## Miner",
        "",
        f"- Miner ID: `{miner.get('miner_id', '')}`",
        f"- Runtime: `{profile.get('runtime', '')}`",
        f"- Backend: `{profile.get('backend', '')}`",
        f"- Accepted: `{profile.get('accepted')}`",
        "",
        "## Route",
        "",
        f"- Route: `{route.get('name', '')}`",
        f"- Target: `{route.get('target', '')}`",
        f"- Workload: `{route.get('workload', '')}`",
        f"- Confidence: `{route.get('confidence', '')}`",
        f"- Matched capabilities: `{', '.join(route.get('matched_capabilities') or [])}`",
        f"- Missing capabilities: `{', '.join(route.get('missing_capabilities') or [])}`",
        "",
        "## Inference",
        "",
        f"- Requests: `{summary.get('request_count')}`",
        f"- Scenario: `{summary.get('scenario_id')}`",
        f"- Scenario schema: `{summary.get('scenario_schema')}`",
        f"- Scenario matches: `{summary.get('scenario_matches')}`",
        f"- Accuracy: `{summary.get('accuracy')}`",
        f"- Elapsed ms: `{summary.get('elapsed_ms')}`",
        f"- Requests/sec: `{summary.get('requests_per_second')}`",
        "",
        "## Observability",
        "",
        f"- Schema: `{observability.get('schema')}`",
        f"- Route usable now: `{(observability.get('route') or {}).get('usable_now')}`",
        f"- Accepted results: `{observed_queue.get('accepted_results')}`",
        f"- Rejected results: `{observed_queue.get('rejected_results')}`",
        f"- Ledger rows: `{observed_queue.get('ledger_rows')}`",
        f"- Request trace count: `{observed_inference.get('request_trace_count')}`",
        f"- Requests/sec: `{observed_inference.get('requests_per_second')}`",
        f"- Read-only: `{observed_safety.get('read_only')}`",
        f"- Redaction OK: `{observed_safety.get('redaction_ok')}`",
        "",
        "## Trace",
        "",
    ]
    for row in payload.get("request_trace") or []:
        lines.append(
            f"- `{row.get('request_id')}` prompt=`{row.get('prompt')}` "
            f"predicted=`{row.get('predicted_token')}` target=`{row.get('target_token')}` "
            f"correct=`{row.get('correct')}`"
        )
    if not payload.get("request_trace"):
        lines.append("- No request trace captured.")
    lines.extend([
        "",
        "## Safety",
        "",
        f"- Read-only: `{safety.get('read_only')}`",
        f"- Redaction OK: `{safety.get('redaction_ok')}`",
        f"- Registry hashed: `{safety.get('registry_hashed')}`",
        f"- Raw payloads exposed: `{safety.get('raw_payloads_exposed')}`",
        "",
        "## Limitations",
        "",
    ])
    for limitation in payload.get("limitations") or []:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def write_json(payload: dict[str, Any], path: str) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(payload: dict[str, Any], path: str) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(payload), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a safe CrowdTensor remote-compute evidence report.")
    parser.add_argument("--mode", choices=["local-loopback", "collect"], default="local-loopback")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8912)
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--miner-id", default=DEFAULT_MINER_ID)
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--scenario-id", default=DEFAULT_INFERENCE_SCENARIO_ID)
    parser.add_argument("--compute-seconds", type=float, default=0.2)
    parser.add_argument("--heartbeat-interval", type=float, default=0.1)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--miner-timeout", type=float, default=30.0)
    parser.add_argument("--observer-token", default=DEFAULT_OBSERVER_TOKEN)
    parser.add_argument("--admin-token", default=DEFAULT_ADMIN_TOKEN)
    parser.add_argument("--invite-token", default=DEFAULT_INVITE_TOKEN)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown-out", default="")
    args = parser.parse_args(argv)
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    args.scenario_id = normalize_inference_scenario_id(args.scenario_id) or DEFAULT_INFERENCE_SCENARIO_ID
    args.base_url = f"http://{args.host}:{args.port}"
    if args.mode == "collect" and not args.coordinator_url:
        raise SystemExit("--coordinator-url is required in collect mode")
    if args.mode == "collect":
        args.coordinator_url = args.coordinator_url.rstrip("/")
    return args


def main() -> None:
    try:
        args = parse_args()
        payload = run_collect(args) if args.mode == "collect" else run_local_loopback(args)
        write_json(payload, args.json_out)
        write_markdown(payload, args.markdown_out)
        print(json.dumps(payload, sort_keys=True))
        raise SystemExit(0 if payload.get("ok") else 1)
    except SystemExit:
        raise
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
