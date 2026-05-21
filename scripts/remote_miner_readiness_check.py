#!/usr/bin/env python3
"""Subprocess smoke test for long-running remote-style Python Miners."""

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
MINER_ID = "remote-readiness-python"
WORKLOADS = {"diloco_train", "cpu_lora_mock", "micro_transformer_lm"}
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from auth_headers import activate_miner_token, activate_observer_token, add_miner_token_arg, add_observer_token_arg, coordinator_env, json_headers, observer_headers  # noqa: E402


def request_json(method: str, base_url: str, path: str, payload: dict | None = None, *, timeout: float = 5.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=observer_headers() if method == "GET" else json_headers(),
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
        str(args.inner_steps),
        "--backlog",
        "0",
        "--task-lane",
        "python-cli:cpu:1:diloco_train",
        "--task-lane",
        "python-cli:cpu:1:cpu_lora_mock",
        "--task-lane",
        "python-cli:cpu:1:micro_transformer_lm",
    ]
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


def parse_summary(stdout: str) -> dict:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "accepted_tasks" in payload and "workloads" in payload:
            return payload
    raise RuntimeError(f"missing miner summary JSON in stdout:\n{stdout}")


def run_miner(args: argparse.Namespace) -> dict:
    command = [
        sys.executable,
        str(ROOT / "miner_cli.py"),
        "--coordinator",
        args.base_url,
        "--miner-id",
        MINER_ID,
        "--max-tasks",
        str(args.max_tasks),
        "--max-runtime-seconds",
        str(args.max_runtime_seconds),
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
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.miner_timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "remote-style miner failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return parse_summary(completed.stdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run remote-style Python Miner readiness smoke.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--inner-steps", type=int, default=4)
    parser.add_argument("--compute-seconds", type=float, default=0.8)
    parser.add_argument("--heartbeat-interval", type=float, default=0.2)
    parser.add_argument("--max-tasks", type=int, default=3)
    parser.add_argument("--max-runtime-seconds", type=float, default=30.0)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--miner-timeout", type=float, default=40.0)
    add_miner_token_arg(parser)
    add_observer_token_arg(parser)
    args = parser.parse_args()
    activate_miner_token(args)
    activate_observer_token(args)
    args.base_url = f"http://{args.host}:{args.port}"
    return args


def main() -> None:
    args = parse_args()
    temp_dir = None
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_remote_miner_")
        state_dir = Path(temp_dir.name)

    coordinator = None
    try:
        coordinator = start_coordinator(args, state_dir)
        miner_summary = run_miner(args)
        state = request_json("GET", args.base_url, "/state")
        completed_workloads = {
            task.get("workload_type")
            for task in state.get("tasks", [])
            if task.get("status") == "completed" and task.get("miner_id") == MINER_ID
        }
        profile = state.get("miner_profiles", {}).get(MINER_ID, {})
        if int(miner_summary.get("accepted_tasks", 0)) != args.max_tasks:
            raise SystemExit(f"unexpected miner summary: {json.dumps(miner_summary, sort_keys=True)}")
        if int(miner_summary.get("heartbeat_failures", 0)) != 0:
            raise SystemExit(f"unexpected heartbeat failures: {json.dumps(miner_summary, sort_keys=True)}")
        if completed_workloads != WORKLOADS:
            raise SystemExit(f"expected workloads {sorted(WORKLOADS)}, got {sorted(completed_workloads)}")
        if profile.get("runtime") != "python-cli" or profile.get("backend") != "cpu":
            raise SystemExit(f"missing python-cli/cpu profile: {json.dumps(profile, sort_keys=True)}")
        if int(profile.get("accepted", 0)) != args.max_tasks:
            raise SystemExit(f"unexpected profile accepted count: {json.dumps(profile, sort_keys=True)}")
        runtime_status = profile.get("last_runtime_status") or {}
        if int(runtime_status.get("accepted_tasks", 0)) < 1:
            raise SystemExit(f"missing long-run runtime status: {json.dumps(profile, sort_keys=True)}")

        print(json.dumps({
            "accepted_results": state["accepted_results"],
            "adapter_updates": state["adapter_updates"],
            "dense_updates": state["model_updates"],
            "micro_transformer_updates": state["micro_transformer_updates"],
            "miner_summary": miner_summary,
            "profile_accepted": profile["accepted"],
            "profile_runtime": profile["runtime"],
            "workloads": sorted(completed_workloads),
        }, sort_keys=True))
    finally:
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
