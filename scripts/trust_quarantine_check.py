#!/usr/bin/env python3
"""Subprocess smoke test for Miner workload quarantine routing."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
MINER_ID = "trust-quarantine-smoke"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from auth_headers import activate_miner_token, activate_observer_token, add_miner_token_arg, add_observer_token_arg, coordinator_env, json_headers, observer_headers  # noqa: E402


def request_json(
    method: str,
    base_url: str,
    path: str,
    payload: dict | None = None,
    *,
    expected_status: int = 200,
    timeout: float = 5.0,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=observer_headers() if method == "GET" else json_headers(),
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            if response.status != expected_status:
                raise RuntimeError(f"expected HTTP {expected_status}, got {response.status}: {raw}")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        raw = exc.read().decode("utf-8")
        if exc.code != expected_status:
            raise RuntimeError(f"expected HTTP {expected_status}, got {exc.code}: {raw}") from exc
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
        "python-cli:cpu:1:cpu_lora_mock",
        "--task-lane",
        "python-cli:cpu:1:diloco_train",
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


def lora_capabilities() -> dict:
    return {
        "runtime": "python-cli",
        "backend": "cpu",
        "protocol_version": "runtime_contract_v1",
        "supported_workloads": ["cpu_lora_mock"],
    }


def reject_bad_lora(base_url: str) -> None:
    claim = request_json(
        "POST",
        base_url,
        "/tasks/claim",
        {"miner_id": MINER_ID, "capabilities": lora_capabilities()},
    )
    if claim.get("workload_type") != "cpu_lora_mock":
        raise RuntimeError(f"expected cpu_lora_mock claim, got {claim}")

    rejected = request_json(
        "POST",
        base_url,
        f"/tasks/{claim['task_id']}/result",
        {
            "lease_token": claim["lease_token"],
            "attempt": claim["attempt"],
            "adapter_delta": {"values": [100.0, 0.0, 0.0]},
        },
        expected_status=422,
    )
    if rejected.get("detail", {}).get("code") != "adapter_delta_norm_too_large":
        raise RuntimeError(f"unexpected rejection payload: {rejected}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Miner trust quarantine smoke test.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8893)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--inner-steps", type=int, default=20)
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
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_trust_")
        state_dir = Path(temp_dir.name)

    coordinator = None
    try:
        coordinator = start_coordinator(args, state_dir)
        reject_bad_lora(args.base_url)
        reject_bad_lora(args.base_url)

        blocked = request_json(
            "POST",
            args.base_url,
            "/tasks/claim",
            {"miner_id": MINER_ID, "capabilities": lora_capabilities()},
            expected_status=503,
        )
        if blocked.get("detail") != "miner quarantined for workload":
            raise SystemExit(f"unexpected blocked claim response: {blocked}")

        dense_claim = request_json(
            "POST",
            args.base_url,
            "/tasks/claim",
            {
                "miner_id": MINER_ID,
                "capabilities": {
                    "runtime": "python-cli",
                    "backend": "cpu",
                    "protocol_version": "runtime_contract_v1",
                    "supported_workloads": ["cpu_lora_mock", "diloco_train"],
                },
            },
        )
        if dense_claim.get("workload_type") != "diloco_train":
            raise SystemExit(f"quarantine was not scoped by workload: {dense_claim}")

        state = request_json("GET", args.base_url, "/state")
        score = state["miner_workload_scores"][MINER_ID]["cpu_lora_mock"]
        if not score.get("quarantined"):
            raise SystemExit(f"miner was not quarantined: {json.dumps(state, sort_keys=True)}")
        if state.get("blocked_claims") != 1:
            raise SystemExit(f"blocked claim was not recorded: {json.dumps(state, sort_keys=True)}")

        print(json.dumps({
            "blocked_claims": state["blocked_claims"],
            "dense_claim_workload": dense_claim["workload_type"],
            "lora_quarantined": score["quarantined"],
            "lora_rejected": score["rejected"],
        }, sort_keys=True))
    finally:
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
