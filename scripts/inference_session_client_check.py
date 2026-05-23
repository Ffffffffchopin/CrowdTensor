#!/usr/bin/env python3
"""CI-safe check for the inference session client."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
ADMIN_TOKEN = "inference-session-client-admin"
MINER_ID = "inference-session-client-miner"


def request_json(base_url: str, path: str, *, timeout: float = 5.0) -> dict:
    request = Request(f"{base_url.rstrip('/')}{path}", method="GET")
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
            if request_json(base_url, "/health", timeout=2.0).get("ok") is True:
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
        "10",
        "--inner-steps",
        str(args.request_count),
        "--backlog",
        "0",
        "--task-lane",
        "python-cli:cpu:0:model_bundle_infer",
        "--admin-token",
        args.admin_token,
    ]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    wait_health(args.base_url, proc, args.startup_timeout)
    return proc


def start_miner(args: argparse.Namespace, log_dir: Path) -> subprocess.Popen:
    stdout = (log_dir / "miner_stdout.log").open("w", encoding="utf-8")
    stderr = (log_dir / "miner_stderr.log").open("w", encoding="utf-8")
    command = [
        sys.executable,
        str(ROOT / "miner_cli.py"),
        "--coordinator",
        args.base_url,
        "--miner-id",
        MINER_ID,
        "--max-tasks",
        "1",
        "--max-runtime-seconds",
        str(args.client_timeout),
        "--idle-sleep",
        "0.2",
        "--heartbeat-interval",
        "0.1",
    ]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
    proc._crowdtensor_stdout = stdout  # type: ignore[attr-defined]
    proc._crowdtensor_stderr = stderr  # type: ignore[attr-defined]
    return proc


def close_process_logs(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    for name in ("_crowdtensor_stdout", "_crowdtensor_stderr"):
        handle = getattr(proc, name, None)
        if handle is not None and not handle.closed:
            handle.close()


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def tail_text(path: Path, limit: int = 2000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


def run_client(args: argparse.Namespace) -> dict:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "inference_session_client.py"),
        "--coordinator-url",
        args.base_url,
        "--admin-token",
        args.admin_token,
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--timeout-seconds",
        str(args.client_timeout),
        "--poll-interval",
        "0.2",
        "--json",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=args.client_timeout + 10)
    if completed.returncode != 0:
        raise RuntimeError(
            "inference_session_client.py failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("inference_session_client.py emitted no JSON")
    return json.loads(lines[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference session client check.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8916)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--scenario-id", default="")
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--client-timeout", type=float, default=30.0)
    parser.add_argument("--admin-token", default=ADMIN_TOKEN)
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
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_session_client_")
        state_dir = Path(temp_dir.name) / "state"
    log_dir = state_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    coordinator = None
    miner = None
    try:
        coordinator = start_coordinator(args, state_dir)
        miner = start_miner(args, log_dir)
        try:
            report = run_client(args)
        except Exception as exc:
            stop_process(miner)
            close_process_logs(miner)
            raise RuntimeError(
                f"{exc}\n"
                f"miner stdout tail:\n{tail_text(log_dir / 'miner_stdout.log')}\n"
                f"miner stderr tail:\n{tail_text(log_dir / 'miner_stderr.log')}"
            ) from exc
        try:
            miner.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        close_process_logs(miner)
        if report.get("schema") != "inference_session_client_v1" or not report.get("ok"):
            raise SystemExit(f"unexpected client report: {json.dumps(report, sort_keys=True)}")
        if report.get("diagnosis", {}).get("primary_code") != "session_client_ready":
            raise SystemExit(f"client did not report ready: {json.dumps(report, sort_keys=True)}")
        session = report.get("session") or {}
        result = report.get("result") or {}
        validation = result.get("validation") or {}
        metrics = result.get("session_metrics") or {}
        if not session.get("created") or session.get("schema") != "inference_session_request_v1":
            raise SystemExit(f"unexpected session summary: {session}")
        if result.get("task_id") != session.get("task_id"):
            raise SystemExit(f"result did not match session task: result={result} session={session}")
        if validation.get("code") != "ok" or int(validation.get("request_count", 0)) != args.request_count:
            raise SystemExit(f"unexpected validation: {validation}")
        if args.scenario_id and validation.get("scenario_id") != args.scenario_id:
            raise SystemExit(f"scenario_id mismatch: {validation}")
        if float(metrics.get("requests_per_second", 0.0)) <= 0.0:
            raise SystemExit(f"invalid session throughput: {metrics}")
        encoded = json.dumps(report, sort_keys=True)
        for fragment in ["lease_token", "idempotency_key", "inference_results", args.admin_token]:
            if fragment in encoded:
                raise SystemExit(f"client report leaked sensitive fragment: {fragment}")
        print(json.dumps({
            "ok": True,
            "schema": report["schema"],
            "task_id": session.get("task_id"),
            "diagnosis": report.get("diagnosis", {}).get("primary_code"),
            "request_count": validation.get("request_count"),
            "request_trace_count": validation.get("request_trace_count"),
            "requests_per_second": metrics.get("requests_per_second"),
            "scenario_id": validation.get("scenario_id"),
            "miner_id": result.get("miner_id"),
        }, sort_keys=True))
    finally:
        stop_process(miner)
        close_process_logs(miner)
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
