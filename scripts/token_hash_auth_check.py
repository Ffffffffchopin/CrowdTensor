#!/usr/bin/env python3
"""Smoke test for sha256 hashed token configuration."""

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

from auth_headers import observer_headers  # noqa: E402
from crowdtensor.auth import hash_token  # noqa: E402
from crowdtensor.diloco import run_inner_loop  # noqa: E402


ADMIN_TOKEN = "hash-admin-token"
MINER_TOKEN = "hash-miner-token"
OBSERVER_TOKEN = "hash-observer-token"
REGISTERED_MINER = "hash-registry-miner"
REGISTERED_TOKEN = "hash-registry-token"
DISABLED_MINER = "hash-disabled-miner"
DISABLED_TOKEN = "hash-disabled-token"
CLI_MINER = "hash-cli-miner"
CLI_TOKEN = "hash-cli-token"


def request_json(
    method: str,
    base_url: str,
    path: str,
    payload: dict | None = None,
    *,
    miner_token: str = "",
    observer_token: str = "",
    admin_token: str = "",
    expected_status: int = 200,
    timeout: float = 5.0,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = observer_headers() if method == "GET" else {"content-type": "application/json"}
    if miner_token:
        headers["x-crowdtensor-miner-token"] = miner_token
    if observer_token:
        headers["x-crowdtensor-observer-token"] = observer_token
    if admin_token:
        headers["x-crowdtensor-admin-token"] = admin_token
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


def write_registry(path: Path) -> Path:
    payload = {
        "miners": [
            {"miner_id": REGISTERED_MINER, "token": hash_token(REGISTERED_TOKEN), "enabled": True},
            {"miner_id": DISABLED_MINER, "token": hash_token(DISABLED_TOKEN), "enabled": False},
            {"miner_id": CLI_MINER, "token": hash_token(CLI_TOKEN), "enabled": True},
        ]
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


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
        str(args.inner_steps),
        "--backlog",
        "3",
        "--admin-token",
        hash_token(ADMIN_TOKEN),
        "--miner-token",
        hash_token(MINER_TOKEN),
        "--observer-token",
        hash_token(OBSERVER_TOKEN),
        "--miner-token-registry",
        str(registry_path),
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


def claim(args: argparse.Namespace, miner_id: str, token: str, *, expected_status: int = 200) -> dict:
    return request_json(
        "POST",
        args.base_url,
        "/tasks/claim",
        {"miner_id": miner_id},
        miner_token=token,
        expected_status=expected_status,
    )


def complete(args: argparse.Namespace, miner_id: str, token: str, task_claim: dict) -> None:
    inner_result = run_inner_loop(
        task_claim["weights"],
        task_id=task_claim["task_id"],
        miner_id=miner_id,
        model_version=int(task_claim["model_version"]),
        inner_steps=int(task_claim["inner_steps"]),
        training_spec=task_claim["training_spec"],
    )
    result = request_json(
        "POST",
        args.base_url,
        f"/tasks/{task_claim['task_id']}/result",
        {
            "lease_token": task_claim["lease_token"],
            "attempt": task_claim["attempt"],
            "local_delta": inner_result["local_delta"],
            "metrics": {**inner_result, "elapsed_ms": 1.0},
        },
        miner_token=token,
    )
    if result.get("accepted") is not True:
        raise RuntimeError(f"expected accepted result, got {result}")


def run_cli(args: argparse.Namespace) -> dict:
    env = dict(os.environ)
    env["CROWDTENSOR_MINER_TOKEN"] = CLI_TOKEN
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "miner_cli.py"),
            "--coordinator",
            args.base_url,
            "--miner-id",
            CLI_MINER,
            "--once",
        ],
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
            "miner_cli hash token path failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    for line in reversed(completed.stdout.strip().splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "accepted_tasks" in payload:
            return payload
    raise RuntimeError(f"missing miner_cli summary JSON:\n{completed.stdout}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hashed token auth smoke.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--inner-steps", type=int, default=10)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--miner-timeout", type=float, default=20.0)
    args = parser.parse_args()
    args.base_url = f"http://{args.host}:{args.port}"
    return args


def main() -> None:
    args = parse_args()
    temp_dir = None
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_hash_auth_")
        state_dir = Path(temp_dir.name)
    registry_path = write_registry(state_dir / "miner_registry.json")

    coordinator = None
    try:
        coordinator = start_coordinator(args, state_dir, registry_path)
        bad_claim = claim(args, "bad-miner", "bad-token", expected_status=401)
        shared_claim = claim(args, "shared-miner", MINER_TOKEN)
        registered_claim = claim(args, REGISTERED_MINER, REGISTERED_TOKEN)
        disabled_claim = claim(args, DISABLED_MINER, DISABLED_TOKEN, expected_status=401)

        complete(args, "shared-miner", MINER_TOKEN, shared_claim)
        complete(args, REGISTERED_MINER, REGISTERED_TOKEN, registered_claim)
        cli_summary = run_cli(args)

        state = request_json("GET", args.base_url, "/state", observer_token=OBSERVER_TOKEN)
        metrics = request_json("GET", args.base_url, "/metrics", observer_token="bad", expected_status=401)
        events = request_json("GET", args.base_url, "/admin/events?limit=1", admin_token=ADMIN_TOKEN)

        if state.get("accepted_results") != 3:
            raise RuntimeError(f"expected three accepted results: {json.dumps(state, sort_keys=True)}")
        if int(cli_summary.get("accepted_tasks", 0)) != 1:
            raise RuntimeError(f"expected CLI to accept one task: {cli_summary}")
        print(json.dumps({
            "accepted_results": state.get("accepted_results"),
            "admin_events": len(events.get("events", [])),
            "bad_claim_detail": bad_claim.get("detail"),
            "cli_accepted_tasks": cli_summary.get("accepted_tasks"),
            "disabled_detail": disabled_claim.get("detail"),
            "metrics_bad_detail": metrics.get("detail"),
            "registry_hash_prefix": hash_token(REGISTERED_TOKEN).split(":", 1)[0],
        }, sort_keys=True))
    finally:
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
