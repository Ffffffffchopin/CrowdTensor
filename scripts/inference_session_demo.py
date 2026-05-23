#!/usr/bin/env python3
"""Run a user-facing local CrowdTensorD inference session demo."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from auth_headers import activate_miner_token, activate_observer_token, add_miner_token_arg, add_observer_token_arg, coordinator_env, json_headers, observer_headers  # noqa: E402


ADMIN_HEADER = "x-crowdtensor-admin-token"
MINER_ID = "inference-session-demo"
WORKLOAD_TYPE = "model_bundle_infer"


def request_json(
    method: str,
    base_url: str,
    path: str,
    payload: dict | None = None,
    *,
    timeout: float = 5.0,
    admin_token: str = "",
) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json", ADMIN_HEADER: admin_token} if admin_token else (
        observer_headers() if method == "GET" else json_headers()
    )
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=headers,
        method=method,
    )
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


def start_coordinator(args: argparse.Namespace, state_dir: Path) -> subprocess.Popen:
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
        f"python-cli:cpu:{0 if args.scenario_id else 1}:{WORKLOAD_TYPE}",
    ]
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    env = coordinator_env()
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


def run_miner(args: argparse.Namespace) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "miner_cli.py"),
            "--coordinator",
            args.base_url,
            "--miner-id",
            MINER_ID,
            "--once",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        text=True,
        capture_output=True,
        timeout=args.miner_timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "miner_cli.py failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def admin_results(args: argparse.Namespace, **params: str | int) -> dict:
    query = urlencode({key: value for key, value in params.items() if value != ""})
    path = "/admin/results" + (f"?{query}" if query else "")
    return request_json("GET", args.base_url, path, admin_token=args.admin_token)


def latest_inference_task(state: dict) -> dict:
    completed = [
        task for task in state.get("tasks", [])
        if task.get("status") == "completed" and task.get("workload_type") == WORKLOAD_TYPE
    ]
    if not completed:
        raise RuntimeError(f"missing completed {WORKLOAD_TYPE} task")
    return completed[-1]


def build_demo_report(
    *,
    state: dict,
    ledger: dict,
    miner_summary: dict,
    expected_request_count: int,
) -> dict:
    task = latest_inference_task(state)
    bundle = state.get("model", {}).get("model_bundle", {})
    validation = task.get("validation") or {}
    metrics = task.get("metrics") or {}
    rows = ledger.get("results") or []
    if len(rows) != 1:
        raise RuntimeError(f"expected one {WORKLOAD_TYPE} ledger row")
    row = rows[0]
    session_metrics = row.get("session_metrics") or {}
    request_trace = list(validation.get("request_trace") or [])
    profiles = state.get("miner_profiles") or {}
    profile = profiles.get(MINER_ID) or {}
    hardware = (profile.get("last_capabilities") or {}).get("hardware_profile") or {}
    raw_state = json.dumps(state.get("tasks", []), sort_keys=True)
    redaction_ok = "inference_result" not in raw_state and "inference_results" not in raw_state
    read_only = (
        bundle.get("version") == 0
        and bundle.get("optimizer_step") == 0
        and state.get("model", {}).get("global_step") == 0
        and state.get("model_updates") == 0
        and not row.get("model_updated")
        and not row.get("model_bundle_updated")
    )
    report = {
        "ok": (
            validation.get("code") == "ok"
            and int(validation.get("request_count", 0)) == expected_request_count
            and int(metrics.get("request_count", 0)) == expected_request_count
            and int(session_metrics.get("request_count", 0)) == expected_request_count
            and float(session_metrics.get("requests_per_second", 0.0)) > 0.0
            and bool(redaction_ok)
            and bool(read_only)
        ),
        "task_id": task.get("task_id"),
        "workload_type": WORKLOAD_TYPE,
        "scenario_schema": validation.get("scenario_schema"),
        "scenario_id": validation.get("scenario_id"),
        "scenario_description": validation.get("scenario_description"),
        "scenario_request_count": validation.get("scenario_request_count"),
        "bundle_id": bundle.get("bundle_id"),
        "bundle_version": bundle.get("version"),
        "request_count": validation.get("request_count"),
        "correct_count": validation.get("correct_count"),
        "accuracy": validation.get("accuracy"),
        "elapsed_ms": session_metrics.get("elapsed_ms"),
        "requests_per_second": session_metrics.get("requests_per_second"),
        "predicted_token": validation.get("predicted_token"),
        "target_token": validation.get("target_token"),
        "request_trace": request_trace,
        "request_trace_count": validation.get("request_trace_count", len(request_trace)),
        "request_trace_truncated": bool(validation.get("request_trace_truncated", False)),
        "read_only": read_only,
        "redaction_ok": redaction_ok,
        "miner": {
            "miner_id": MINER_ID,
            "runtime": profile.get("runtime"),
            "backend": profile.get("backend"),
            "hardware_profile": hardware,
            "summary": miner_summary,
        },
        "ledger": {
            "rows": len(rows),
            "event_index": row.get("event_index"),
            "status": row.get("status"),
            "session_metrics": session_metrics,
        },
    }
    return report


def print_human_report(report: dict) -> None:
    hardware = report["miner"]["hardware_profile"]
    print("CrowdTensor local inference session demo")
    print(f"  ok: {report['ok']}")
    print(f"  task: {report['task_id']} ({report['workload_type']})")
    print(f"  bundle: {report['bundle_id']} v{report['bundle_version']}")
    print(
        "  session: "
        f"{report['request_count']} requests, "
        f"accuracy={float(report['accuracy'] or 0.0):.3f}, "
        f"elapsed_ms={float(report['elapsed_ms'] or 0.0):.3f}, "
        f"requests_per_second={float(report['requests_per_second'] or 0.0):.3f}"
    )
    print(f"  sample prediction: {report['predicted_token']!r} target={report['target_token']!r}")
    trace = report.get("request_trace") or []
    if trace:
        print("  trace:")
        for row in trace:
            print(
                "    "
                f"{row.get('request_id')} "
                f"prompt={row.get('prompt')!r} "
                f"predicted={row.get('predicted_token')!r} "
                f"target={row.get('target_token')!r} "
                f"correct={bool(row.get('correct'))}"
            )
        if report.get("request_trace_truncated"):
            print(f"    ... truncated at {report.get('request_trace_count')} rows")
    print(
        "  miner: "
        f"{report['miner']['runtime']}/{report['miner']['backend']} "
        f"cpu_count={hardware.get('cpu_count')} "
        f"os={hardware.get('os')} "
        f"python={hardware.get('python_version')}"
    )
    print(f"  safety: read_only={report['read_only']} redaction_ok={report['redaction_ok']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local CrowdTensorD inference session demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8904)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--scenario-id", default="")
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--miner-timeout", type=float, default=30.0)
    parser.add_argument("--admin-token", default="local-admin")
    parser.add_argument("--json", action="store_true", help="emit only the machine-readable demo report")
    add_miner_token_arg(parser)
    add_observer_token_arg(parser)
    args = parser.parse_args()
    activate_miner_token(args)
    activate_observer_token(args)
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    args.base_url = f"http://{args.host}:{args.port}"
    return args


def run_demo(args: argparse.Namespace) -> dict:
    temp_dir = None
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_inference_demo_")
        state_dir = Path(temp_dir.name)

    coordinator = None
    try:
        coordinator = start_coordinator(args, state_dir)
        if args.scenario_id:
            request_json(
                "POST",
                args.base_url,
                "/admin/inference-sessions",
                {"request_count": args.request_count, "scenario_id": args.scenario_id},
                admin_token=args.admin_token,
            )
        miner_summary = run_miner(args)
        state = request_json("GET", args.base_url, "/state")
        ledger = admin_results(args, status="accepted", workload_type=WORKLOAD_TYPE)
        return build_demo_report(
            state=state,
            ledger=ledger,
            miner_summary=miner_summary,
            expected_request_count=args.request_count,
        )
    finally:
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


def main() -> None:
    args = parse_args()
    report = run_demo(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print_human_report(report)
    if not report.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
