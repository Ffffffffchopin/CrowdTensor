#!/usr/bin/env python3
"""Subprocess smoke test for Operator control-plane trust overrides."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from crowdtensor.lora_mock import run_lora_inner_loop
from auth_headers import activate_miner_token, activate_observer_token, add_miner_token_arg, add_observer_token_arg, coordinator_env, json_headers, observer_headers  # noqa: E402

MINER_ID = "operator-control-smoke"


def request_json(
    method: str,
    base_url: str,
    path: str,
    payload: dict | None = None,
    *,
    expected_status: int = 200,
    token: str = "",
    timeout: float = 5.0,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = observer_headers() if method == "GET" else json_headers()
    if token:
        headers["x-crowdtensor-admin-token"] = token
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=headers,
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
        "--admin-token",
        args.admin_token,
        "--task-lane",
        "python-cli:cpu:1:cpu_lora_mock",
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


def claim_lora(base_url: str) -> dict:
    return request_json(
        "POST",
        base_url,
        "/tasks/claim",
        {"miner_id": MINER_ID, "capabilities": lora_capabilities()},
    )


def reject_bad_lora(base_url: str) -> None:
    claim = claim_lora(base_url)
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


def set_override(base_url: str, token: str, mode: str, reason: str) -> dict:
    return request_json(
        "POST",
        base_url,
        "/admin/trust-overrides",
        {
            "miner_id": MINER_ID,
            "workload_type": "cpu_lora_mock",
            "mode": mode,
            "reason": reason,
        },
        token=token,
    )


def complete_lora(base_url: str, claim: dict) -> None:
    result = run_lora_inner_loop(claim["workload_spec"], inner_steps=int(claim["inner_steps"]))
    accepted = request_json(
        "POST",
        base_url,
        f"/tasks/{claim['task_id']}/result",
        {
            "lease_token": claim["lease_token"],
            "attempt": claim["attempt"],
            "adapter_delta": result["adapter_delta"],
            "metrics": result,
        },
    )
    if accepted.get("adapter_updated") is not True:
        raise RuntimeError(f"expected adapter update, got {accepted}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Operator control-plane smoke test.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8895)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--admin-token", default="local-admin")
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
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_operator_")
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
            raise SystemExit(f"unexpected automatic block response: {blocked}")

        allow = set_override(args.base_url, args.admin_token, "allow", "smoke allow")
        if allow.get("mode") != "allow":
            raise SystemExit(f"allow override failed: {allow}")
        allowed_claim = claim_lora(args.base_url)
        complete_lora(args.base_url, allowed_claim)

        block = set_override(args.base_url, args.admin_token, "block", "smoke block")
        if block.get("mode") != "block":
            raise SystemExit(f"block override failed: {block}")
        manually_blocked = request_json(
            "POST",
            args.base_url,
            "/tasks/claim",
            {"miner_id": MINER_ID, "capabilities": lora_capabilities()},
            expected_status=503,
        )
        if manually_blocked.get("detail") != "miner manually blocked for workload":
            raise SystemExit(f"unexpected manual block response: {manually_blocked}")

        state = request_json("GET", args.base_url, "/state")
        events = request_json(
            "GET",
            args.base_url,
            "/admin/events?limit=50",
            token=args.admin_token,
        )
        if state["miner_trust_overrides"][MINER_ID]["cpu_lora_mock"]["mode"] != "block":
            raise SystemExit(f"missing block override: {json.dumps(state, sort_keys=True)}")
        if not any(event.get("type") == "trust_override_set" for event in events.get("events", [])):
            raise SystemExit(f"missing override event tail: {events}")
        if any(event.get("lease_token") and event.get("lease_token") != "<redacted>" for event in events.get("events", [])):
            raise SystemExit(f"unredacted lease token in events: {events}")

        print(json.dumps({
            "auto_blocked": blocked["detail"],
            "manual_blocked": manually_blocked["detail"],
            "override_mode": state["miner_trust_overrides"][MINER_ID]["cpu_lora_mock"]["mode"],
            "event_tail": len(events.get("events", [])),
        }, sort_keys=True))
    finally:
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
