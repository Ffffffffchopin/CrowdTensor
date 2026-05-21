#!/usr/bin/env python3
"""Smoke test for public Coordinator readiness/profile endpoints."""

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

from crowdtensor.auth import hash_token  # noqa: E402


MINER_TOKEN = "readiness-miner"
OBSERVER_TOKEN = "readiness-observer"
ADMIN_TOKEN = "readiness-admin"
REGISTRY_TOKEN = "readiness-registry"


def request_json(base_url: str, path: str, *, expected_status: int = 200, timeout: float = 5.0) -> dict:
    request = Request(f"{base_url.rstrip('/')}{path}", method="GET")
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


def wait_ready(base_url: str, proc: subprocess.Popen, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"coordinator exited early with code {proc.returncode}")
        try:
            ready = request_json(base_url, "/ready", timeout=2.0)
            if ready.get("ok") is True:
                return ready
        except Exception as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"coordinator did not become ready: {last_error}")


def write_registry(path: Path) -> Path:
    payload = {
        "miners": [
            {"miner_id": "readiness-registry-miner", "token": hash_token(REGISTRY_TOKEN), "enabled": True}
        ]
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def start_coordinator(args: argparse.Namespace, state_dir: Path, registry_path: Path | None = None) -> subprocess.Popen:
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
        "5",
        "--inner-steps",
        "5",
    ]
    if args.with_auth:
        command.extend([
            "--miner-token",
            hash_token(MINER_TOKEN),
            "--observer-token",
            hash_token(OBSERVER_TOKEN),
            "--admin-token",
            hash_token(ADMIN_TOKEN),
        ])
    if registry_path is not None:
        command.extend(["--miner-token-registry", str(registry_path)])
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    wait_ready(args.base_url, proc, args.startup_timeout)
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


def assert_public_profile(payload: dict, *, path: str) -> None:
    expected = {
        "service": "crowdtensord-coordinator",
        "version": "0.1.0a0",
        "protocol_version": "runtime_contract_v1",
        "default_workload_type": "diloco_train",
        "api_status": "alpha",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"{path}.{key} expected {value!r}, got {payload.get(key)!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Coordinator readiness/profile smoke.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--with-auth", action="store_true")
    args = parser.parse_args()
    args.base_url = f"http://{args.host}:{args.port}"
    return args


def main() -> None:
    args = parse_args()
    temp_dir = None
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_readiness_")
        state_dir = Path(temp_dir.name)

    registry_path = write_registry(state_dir / "miner_registry.json") if args.with_auth else None
    coordinator = None
    try:
        coordinator = start_coordinator(args, state_dir, registry_path)
        health = request_json(args.base_url, "/health")
        version = request_json(args.base_url, "/version")
        ready = request_json(args.base_url, "/ready")
        state_protected = request_json(args.base_url, "/state", expected_status=401) if args.with_auth else {}

        if health.get("ok") is not True:
            raise RuntimeError(f"health should be ok: {health}")
        assert_public_profile(version, path="/version")
        assert_public_profile(ready, path="/ready")
        if ready.get("ok") is not True:
            raise RuntimeError(f"ready should be ok: {ready}")
        if "queued" not in ready.get("task_counts", {}):
            raise RuntimeError(f"ready should expose task_counts: {ready}")
        auth = ready.get("auth") or {}
        expected_auth = {
            "miner_required": bool(args.with_auth),
            "observer_required": bool(args.with_auth),
            "admin_configured": bool(args.with_auth),
            "miner_registry_configured": bool(args.with_auth),
        }
        for key, value in expected_auth.items():
            if auth.get(key) is not value:
                raise RuntimeError(f"ready.auth.{key} expected {value}, got {auth.get(key)}")

        print(json.dumps({
            "health_ok": health.get("ok"),
            "ready_ok": ready.get("ok"),
            "version": version.get("version"),
            "protocol_version": ready.get("protocol_version"),
            "auth": auth,
            "state_public": state_protected == {},
            "task_counts": ready.get("task_counts"),
        }, sort_keys=True))
    finally:
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
