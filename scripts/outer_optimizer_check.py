#!/usr/bin/env python3
"""Subprocess smoke test for opt-in outer optimizer contracts."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor.diloco import run_inner_loop  # noqa: E402
from crowdtensor.outer_optimizer import OPTIMIZER_DILOCO_NESTEROV, compress_sign_delta  # noqa: E402


def request_json(method: str, base_url: str, path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json"} if payload is not None else {}
    request = Request(f"{base_url.rstrip('/')}{path}", data=data, headers=headers, method=method)
    with urlopen(request, timeout=5.0) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def wait_ready(base_url: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"coordinator exited early with code {proc.returncode}")
        try:
            ready = request_json("GET", base_url, "/ready")
            if ready.get("ok") is True:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"coordinator did not become ready: {last_error}")


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


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
        "1",
        "--replay-audit",
        "--outer-optimizer",
        OPTIMIZER_DILOCO_NESTEROV,
    ]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    wait_ready(args.base_url, proc, args.startup_timeout)
    return proc


def claim(base_url: str, miner_id: str) -> dict:
    return request_json(
        "POST",
        base_url,
        "/tasks/claim",
        {
            "miner_id": miner_id,
            "capabilities": {
                "runtime": "python-cli",
                "backend": "cpu",
                "protocol_version": "runtime_contract_v1",
                "supported_workloads": ["diloco_train"],
            },
        },
    )


def inner_result_for(claimed: dict, miner_id: str) -> dict:
    return run_inner_loop(
        claimed["weights"],
        task_id=claimed["task_id"],
        miner_id=miner_id,
        model_version=int(claimed["model_version"]),
        inner_steps=int(claimed["inner_steps"]),
        training_spec=claimed["training_spec"],
    )


def complete_dense(base_url: str, claimed: dict, miner_id: str) -> dict:
    inner = inner_result_for(claimed, miner_id)
    return request_json(
        "POST",
        base_url,
        f"/tasks/{claimed['task_id']}/result",
        {
            "lease_token": claimed["lease_token"],
            "attempt": claimed["attempt"],
            "local_delta": inner["local_delta"],
            "metrics": {**inner, "elapsed_ms": 1.0},
        },
    )


def complete_compressed(base_url: str, claimed: dict, miner_id: str) -> dict:
    inner = inner_result_for(claimed, miner_id)
    return request_json(
        "POST",
        base_url,
        f"/tasks/{claimed['task_id']}/result",
        {
            "lease_token": claimed["lease_token"],
            "attempt": claimed["attempt"],
            "compressed_delta": compress_sign_delta(inner["local_delta"]),
            "metrics": {"delta_format": "sign_compressed", "elapsed_ms": 1.0},
        },
    )


def assert_nesterov_result(claimed: dict, result: dict) -> None:
    if (claimed.get("optimizer_spec") or {}).get("optimizer_type") != OPTIMIZER_DILOCO_NESTEROV:
        raise RuntimeError(f"claim missing nesterov optimizer spec: {claimed}")
    optimizer = result.get("optimizer") or {}
    if optimizer.get("optimizer_type") != OPTIMIZER_DILOCO_NESTEROV:
        raise RuntimeError(f"result missing nesterov optimizer summary: {result}")
    if "outer_update_norm" not in optimizer:
        raise RuntimeError(f"result missing outer_update_norm: {result}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run outer optimizer contract smoke test.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--inner-steps", type=int, default=10)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    args = parser.parse_args()
    args.base_url = f"http://{args.host}:{args.port}"
    return args


def main() -> None:
    args = parse_args()
    temp_dir = None
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_outer_optimizer_")
        state_dir = Path(temp_dir.name)
    state_dir.mkdir(parents=True, exist_ok=True)

    coordinator = None
    try:
        coordinator = start_coordinator(args, state_dir)
        dense_claim = claim(args.base_url, "nesterov-dense")
        dense_result = complete_dense(args.base_url, dense_claim, "nesterov-dense")
        assert_nesterov_result(dense_claim, dense_result)

        compressed_claim = claim(args.base_url, "nesterov-compressed")
        compressed_result = complete_compressed(args.base_url, compressed_claim, "nesterov-compressed")
        assert_nesterov_result(compressed_claim, compressed_result)
        if (compressed_result.get("optimizer") or {}).get("delta_format") != "sign_compressed":
            raise RuntimeError(f"compressed result missing sign_compressed summary: {compressed_result}")

        state = request_json("GET", args.base_url, "/state")
        if state.get("audit_results") != 2 or state.get("audit_rejections") != 0:
            raise RuntimeError(f"unexpected replay audit counters: {state}")
        if (state.get("model") or {}).get("outer_optimizer_type") != OPTIMIZER_DILOCO_NESTEROV:
            raise RuntimeError(f"state did not persist nesterov optimizer: {state}")
        print(json.dumps({
            "accepted_results": state["accepted_results"],
            "audit_results": state["audit_results"],
            "compressed_delta_format": compressed_result["optimizer"]["delta_format"],
            "optimizer_type": OPTIMIZER_DILOCO_NESTEROV,
        }, sort_keys=True))
    finally:
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
