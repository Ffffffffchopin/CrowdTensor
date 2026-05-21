#!/usr/bin/env python3
"""Smoke test for Observer token read-endpoint admission control."""

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


def request_text(
    method: str,
    base_url: str,
    path: str,
    *,
    token: str = "",
    expected_status: int = 200,
    timeout: float = 5.0,
) -> tuple[str, dict[str, str]]:
    headers: dict[str, str] = {}
    if token:
        headers["x-crowdtensor-observer-token"] = token
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            if response.status != expected_status:
                raise RuntimeError(f"expected HTTP {expected_status}, got {response.status}: {raw}")
            return raw, dict(response.headers.items())
    except HTTPError as exc:
        raw = exc.read().decode("utf-8")
        if exc.code != expected_status:
            raise RuntimeError(f"expected HTTP {expected_status}, got {exc.code}: {raw}") from exc
        return raw, dict(exc.headers.items())


def request_json(
    method: str,
    base_url: str,
    path: str,
    *,
    token: str = "",
    expected_status: int = 200,
    timeout: float = 5.0,
) -> dict:
    raw, _headers = request_text(
        method,
        base_url,
        path,
        token=token,
        expected_status=expected_status,
        timeout=timeout,
    )
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
        "--observer-token",
        args.observer_token,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Observer token read-endpoint smoke test.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--inner-steps", type=int, default=10)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--observer-token", default=os.environ.get("CROWDTENSOR_OBSERVER_TOKEN", "local-observer-token"))
    args = parser.parse_args()
    if not args.observer_token:
        raise SystemExit("--observer-token is required for observer_auth_check")
    args.base_url = f"http://{args.host}:{args.port}"
    return args


def main() -> None:
    args = parse_args()
    temp_dir = None
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_observer_auth_")
        state_dir = Path(temp_dir.name)

    coordinator = None
    try:
        coordinator = start_coordinator(args, state_dir)
        health = request_json("GET", args.base_url, "/health")
        missing_state = request_json("GET", args.base_url, "/state", expected_status=401)
        bad_metrics = request_json("GET", args.base_url, "/metrics", token="bad-token", expected_status=401)
        state = request_json("GET", args.base_url, "/state", token=args.observer_token)
        metrics_text, metrics_headers = request_text("GET", args.base_url, "/metrics", token=args.observer_token)
        if health.get("ok") is not True:
            raise SystemExit(f"health should remain public: {health}")
        if missing_state.get("detail") != "invalid observer token":
            raise SystemExit(f"unexpected missing-token state response: {missing_state}")
        if bad_metrics.get("detail") != "invalid observer token":
            raise SystemExit(f"unexpected bad-token metrics response: {bad_metrics}")
        if "task_counts" not in state or "model" not in state:
            raise SystemExit(f"unexpected state payload: {state}")
        content_type = metrics_headers.get("Content-Type", metrics_headers.get("content-type", ""))
        if "text/plain" not in content_type or "crowdtensord_task_count" not in metrics_text:
            raise SystemExit(f"unexpected metrics response: content_type={content_type} body={metrics_text}")

        print(json.dumps({
            "health_public": True,
            "metrics_content_type": content_type,
            "metrics_protected": bad_metrics["detail"],
            "state_protected": missing_state["detail"],
            "state_task_counts": state["task_counts"],
        }, sort_keys=True))
    finally:
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
