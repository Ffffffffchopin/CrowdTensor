#!/usr/bin/env python3
"""Validate the admin-created read-only inference session API."""

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

ADMIN_HEADER = "x-crowdtensor-admin-token"
WORKLOAD_TYPE = "model_bundle_infer"
MINER_ID = "admin-inference-session-miner"


def request_json(
    method: str,
    base_url: str,
    path: str,
    payload: dict | None = None,
    *,
    admin_token: str = "",
    timeout: float = 5.0,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json"}
    if admin_token:
        headers[ADMIN_HEADER] = admin_token
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
        "1",
        "--backlog",
        "0",
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


def run_miner(args: argparse.Namespace) -> dict:
    command = [
        sys.executable,
        str(ROOT / "miner_cli.py"),
        "--coordinator",
        args.base_url,
        "--miner-id",
        MINER_ID,
        "--once",
        "--idle-sleep",
        "0.2",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate admin read-only inference session API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8915)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--scenario-id", default="")
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--miner-timeout", type=float, default=30.0)
    parser.add_argument("--admin-token", default="local-admin")
    args = parser.parse_args()
    if args.request_count < 1 or args.request_count > 8:
        raise SystemExit("--request-count must be between 1 and 8")
    args.base_url = f"http://{args.host}:{args.port}"
    return args


def main() -> None:
    args = parse_args()
    temp_dir = None
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_admin_inference_")
        state_dir = Path(temp_dir.name)

    coordinator = None
    try:
        coordinator = start_coordinator(args, state_dir)
        session = request_json(
            "POST",
            args.base_url,
            "/admin/inference-sessions",
            {"request_count": args.request_count, "scenario_id": args.scenario_id},
            admin_token=args.admin_token,
        )
        if session.get("schema") != "inference_session_request_v1":
            raise SystemExit(f"unexpected session schema: {session}")
        if session.get("workload_type") != WORKLOAD_TYPE or session.get("status") != "queued":
            raise SystemExit(f"unexpected session response: {session}")
        if int(session.get("request_count", 0)) != args.request_count:
            raise SystemExit(f"session request_count mismatch: {session}")
        if args.scenario_id and session.get("scenario_id") != args.scenario_id:
            raise SystemExit(f"session scenario_id mismatch: {session}")
        requirements = session.get("task_requirements") or {}
        if requirements.get("runtime") != "python-cli" or requirements.get("backend") != "cpu":
            raise SystemExit(f"unexpected session requirements: {requirements}")

        miner_summary = run_miner(args)
        ledger = admin_results(
            args,
            status="accepted",
            workload_type=WORKLOAD_TYPE,
            task_id=session["task_id"],
        )
        rows = ledger.get("results") or []
        if len(rows) != 1:
            raise SystemExit(f"expected one ledger row for session task: {ledger}")
        row = rows[0]
        validation = row.get("validation") or {}
        metrics = row.get("session_metrics") or {}
        if row.get("task_id") != session["task_id"] or row.get("workload_type") != WORKLOAD_TYPE:
            raise SystemExit(f"unexpected ledger row identity: {row}")
        if row.get("model_updated") or row.get("model_bundle_updated"):
            raise SystemExit(f"admin inference session must be read-only: {row}")
        if validation.get("code") != "ok":
            raise SystemExit(f"unexpected validation: {validation}")
        if int(validation.get("request_count", 0)) != args.request_count:
            raise SystemExit(f"validation request_count mismatch: {validation}")
        if args.scenario_id and validation.get("scenario_id") != args.scenario_id:
            raise SystemExit(f"validation scenario_id mismatch: {validation}")
        if int(validation.get("request_trace_count", 0)) != min(args.request_count, 8):
            raise SystemExit(f"validation request_trace_count mismatch: {validation}")
        if float(metrics.get("requests_per_second", 0.0)) <= 0.0:
            raise SystemExit(f"invalid session throughput: {metrics}")
        state = request_json("GET", args.base_url, "/state")
        if state.get("model", {}).get("global_step") != 0 or state.get("model_updates") != 0:
            raise SystemExit(f"admin inference mutated dense model: {state.get('model')}")
        if state.get("model", {}).get("model_bundle", {}).get("version") != 0:
            raise SystemExit(f"admin inference mutated model bundle: {state.get('model')}")
        safe_payload = {"ledger": ledger, "state_tasks": state.get("tasks", [])}
        encoded = json.dumps(safe_payload, sort_keys=True)
        for fragment in ["inference_results", "CROWDTENSOR_MINER_TOKEN"]:
            if fragment in encoded:
                raise SystemExit(f"admin inference output leaked secret-like material: {fragment}")
        def assert_no_secret_field(value: object) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "lease_token" and child not in {"", None, "<redacted>"}:
                        raise SystemExit(f"admin inference output leaked secret-like material: {key}")
                    if key in {"idempotency_key", "result_idempotency_key_hash", "result_lease_token_hash"}:
                        raise SystemExit(f"admin inference output leaked secret-like material: {key}")
                    assert_no_secret_field(child)
            elif isinstance(value, list):
                for child in value:
                    assert_no_secret_field(child)

        assert_no_secret_field(safe_payload)

        print(json.dumps({
            "ok": True,
            "schema": session["schema"],
            "task_id": session["task_id"],
            "route": "admin_readonly_model_bundle_infer",
            "workload_type": WORKLOAD_TYPE,
            "request_count": validation.get("request_count"),
            "request_trace_count": validation.get("request_trace_count"),
            "requests_per_second": metrics.get("requests_per_second"),
            "scenario_id": validation.get("scenario_id"),
            "miner_id": MINER_ID,
            "miner_summary": {
                "accepted_tasks": miner_summary.get("accepted_tasks"),
                "rejected_tasks": miner_summary.get("rejected_tasks"),
            },
        }, sort_keys=True))
    finally:
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
