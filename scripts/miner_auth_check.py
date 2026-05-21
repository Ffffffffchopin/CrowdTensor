#!/usr/bin/env python3
"""Smoke test for shared Miner token admission control."""

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

from crowdtensor.diloco import run_inner_loop  # noqa: E402
from auth_headers import observer_headers  # noqa: E402


MINER_ID = "miner-auth-smoke"


def request_json(
    method: str,
    base_url: str,
    path: str,
    payload: dict | None = None,
    *,
    token: str = "",
    expected_status: int = 200,
    timeout: float = 5.0,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = observer_headers() if method == "GET" else {"content-type": "application/json"}
    if token:
        headers["x-crowdtensor-miner-token"] = token
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
        "1",
        "--miner-token",
        args.miner_token,
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


def claim_with_token(args: argparse.Namespace) -> dict:
    return request_json(
        "POST",
        args.base_url,
        "/tasks/claim",
        {"miner_id": MINER_ID},
        token=args.miner_token,
    )


def complete_with_token(args: argparse.Namespace, claim: dict) -> None:
    inner_result = run_inner_loop(
        claim["weights"],
        task_id=claim["task_id"],
        miner_id=MINER_ID,
        model_version=int(claim["model_version"]),
        inner_steps=int(claim["inner_steps"]),
        training_spec=claim["training_spec"],
    )
    result = request_json(
        "POST",
        args.base_url,
        f"/tasks/{claim['task_id']}/result",
        {
            "lease_token": claim["lease_token"],
            "attempt": claim["attempt"],
            "local_delta": inner_result["local_delta"],
            "metrics": {**inner_result, "elapsed_ms": 1.0},
        },
        token=args.miner_token,
    )
    if result.get("accepted") is not True:
        raise RuntimeError(f"expected accepted result, got {result}")


def run_cli_with_token(args: argparse.Namespace) -> dict:
    env = dict(os.environ)
    env["CROWDTENSOR_MINER_TOKEN"] = args.miner_token
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "miner_cli.py"),
            "--coordinator",
            args.base_url,
            "--miner-id",
            f"{MINER_ID}-cli",
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
            "miner_cli token path failed\n"
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
    parser = argparse.ArgumentParser(description="Run Miner token admission smoke test.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8892)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--inner-steps", type=int, default=10)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--miner-timeout", type=float, default=20.0)
    parser.add_argument("--miner-token", default=os.environ.get("CROWDTENSOR_MINER_TOKEN", "local-miner-token"))
    args = parser.parse_args()
    if not args.miner_token:
        raise SystemExit("--miner-token is required for miner_auth_check")
    args.base_url = f"http://{args.host}:{args.port}"
    return args


def main() -> None:
    args = parse_args()
    temp_dir = None
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_miner_auth_")
        state_dir = Path(temp_dir.name)

    coordinator = None
    try:
        coordinator = start_coordinator(args, state_dir)
        blocked_missing = request_json(
            "POST",
            args.base_url,
            "/tasks/claim",
            {"miner_id": "missing-token"},
            expected_status=401,
        )
        blocked_bad = request_json(
            "POST",
            args.base_url,
            "/tasks/claim",
            {"miner_id": "bad-token"},
            token="bad-token",
            expected_status=401,
        )
        claim = claim_with_token(args)
        request_json(
            "POST",
            args.base_url,
            f"/tasks/{claim['task_id']}/heartbeat",
            {
                "lease_token": claim["lease_token"],
                "attempt": claim["attempt"],
                "runtime_status": {"phase": "auth-smoke"},
            },
            expected_status=401,
        )
        request_json(
            "POST",
            args.base_url,
            f"/tasks/{claim['task_id']}/heartbeat",
            {
                "lease_token": claim["lease_token"],
                "attempt": claim["attempt"],
                "runtime_status": {"phase": "auth-smoke"},
            },
            token=args.miner_token,
        )
        complete_with_token(args, claim)
        cli_summary = run_cli_with_token(args)
        state = request_json("GET", args.base_url, "/state")
        if state.get("accepted_results") != 2:
            raise SystemExit(f"expected two accepted results: {json.dumps(state, sort_keys=True)}")
        if int(cli_summary.get("accepted_tasks", 0)) != 1:
            raise SystemExit(f"unexpected miner_cli summary: {json.dumps(cli_summary, sort_keys=True)}")

        print(json.dumps({
            "accepted_results": state["accepted_results"],
            "bad_token_detail": blocked_bad.get("detail"),
            "cli_accepted_tasks": cli_summary["accepted_tasks"],
            "missing_token_detail": blocked_missing.get("detail"),
            "state_tasks": state["task_counts"],
        }, sort_keys=True))
    finally:
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
