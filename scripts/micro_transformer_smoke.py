#!/usr/bin/env python3
"""Subprocess smoke test for the CrowdTensorD micro_transformer_lm workload."""

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
        "--replay-audit",
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


def run_miner(args: argparse.Namespace) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "miner_cli.py"),
            "--coordinator",
            args.base_url,
            "--miner-id",
            "micro-transformer-smoke",
            "--once",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "miner_cli.py failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run micro_transformer_lm workload smoke test.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8896)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--inner-steps", type=int, default=4)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
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
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_micro_transformer_")
        state_dir = Path(temp_dir.name)

    coordinator = None
    try:
        coordinator = start_coordinator(args, state_dir)
        run_miner(args)
        state = request_json("GET", args.base_url, "/state")
        micro = state["model"]["micro_transformer"]
        completed = [
            task for task in state["tasks"]
            if task["status"] == "completed" and task["workload_type"] == "micro_transformer_lm"
        ]
        if micro["version"] != 1 or micro["optimizer_step"] != 1:
            raise SystemExit(f"unexpected micro_transformer state: {micro}")
        if state["model"]["global_step"] != 0:
            raise SystemExit(f"micro_transformer_lm should not update dense global_step: {state['model']}")
        if not completed:
            raise SystemExit(f"missing completed micro_transformer task: {json.dumps(state, sort_keys=True)}")
        metrics = completed[-1].get("metrics", {})
        if metrics.get("lm_loss_end", 0) >= metrics.get("lm_loss_start", 0):
            raise SystemExit(f"expected LM loss to decrease: {metrics}")
        if metrics.get("gradient_mode") != "analytic":
            raise SystemExit(f"expected analytic micro_transformer gradient mode: {metrics}")
        if state.get("audit_results") != 1:
            raise SystemExit(f"expected one replay audit result: {json.dumps(state, sort_keys=True)}")
        print(json.dumps({
            "accepted_results": state["accepted_results"],
            "audit_results": state["audit_results"],
            "gradient_mode": metrics["gradient_mode"],
            "lm_loss_end": metrics["lm_loss_end"],
            "lm_loss_start": metrics["lm_loss_start"],
            "micro_transformer_optimizer_step": micro["optimizer_step"],
            "micro_transformer_version": micro["version"],
            "task_id": completed[-1]["task_id"],
        }, sort_keys=True))
    finally:
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
