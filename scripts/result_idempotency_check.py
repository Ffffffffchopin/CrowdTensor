#!/usr/bin/env python3
"""Smoke test for idempotent result uploads."""

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

from crowdtensor.diloco import run_inner_loop  # noqa: E402


MINER_ID = "result-idempotency-smoke"


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
    headers = {"content-type": "application/json"} if payload is not None else {}
    request = Request(f"{base_url.rstrip('/')}{path}", data=data, headers=headers, method=method)
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


def wait_ready(base_url: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"coordinator exited early with code {proc.returncode}")
        try:
            ready = request_json("GET", base_url, "/ready", timeout=2.0)
            if ready.get("ok") is True:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"coordinator did not become ready: {last_error}")


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
        "8",
        "--backlog",
        "1",
    ]
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


def claim_task(args: argparse.Namespace) -> dict:
    return request_json("POST", args.base_url, "/tasks/claim", {"miner_id": MINER_ID})


def result_payload(claim: dict, *, idempotency_key: str) -> dict:
    inner_result = run_inner_loop(
        claim["weights"],
        task_id=claim["task_id"],
        miner_id=MINER_ID,
        model_version=int(claim["model_version"]),
        inner_steps=int(claim["inner_steps"]),
        training_spec=claim["training_spec"],
    )
    return {
        "lease_token": claim["lease_token"],
        "attempt": claim["attempt"],
        "idempotency_key": idempotency_key,
        "local_delta": inner_result["local_delta"],
        "metrics": {**inner_result, "elapsed_ms": 1.0},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run result idempotency smoke.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8896)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    args = parser.parse_args()
    args.base_url = f"http://{args.host}:{args.port}"
    return args


def main() -> None:
    args = parse_args()
    temp_dir = None
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_result_idempotency_")
        state_dir = Path(temp_dir.name)

    coordinator = None
    try:
        coordinator = start_coordinator(args, state_dir)
        claim = claim_task(args)
        payload = result_payload(claim, idempotency_key="result-key-1")
        first = request_json("POST", args.base_url, f"/tasks/{claim['task_id']}/result", payload)
        duplicate = request_json("POST", args.base_url, f"/tasks/{claim['task_id']}/result", payload)
        wrong_key_payload = dict(payload)
        wrong_key_payload["idempotency_key"] = "result-key-2"
        wrong_key = request_json(
            "POST",
            args.base_url,
            f"/tasks/{claim['task_id']}/result",
            wrong_key_payload,
            expected_status=409,
        )
        state = request_json("GET", args.base_url, "/state")
        events = request_json("GET", args.base_url, "/admin/events?limit=10", expected_status=403)

        if first != duplicate:
            raise SystemExit(f"duplicate response changed: first={first} duplicate={duplicate}")
        if int(state.get("accepted_results", 0)) != 1 or int(state.get("model", {}).get("global_step", 0)) != 1:
            raise SystemExit(f"duplicate result changed state: {json.dumps(state, sort_keys=True)}")
        public_text = json.dumps(state, sort_keys=True)
        if "result-key-1" in public_text or "result_idempotency_key_hash" in public_text:
            raise SystemExit(f"public state leaked idempotency fields: {public_text}")

        print(json.dumps({
            "accepted_results": state["accepted_results"],
            "duplicate_response_stable": True,
            "forbidden_admin_events": events.get("detail"),
            "global_step": state["model"]["global_step"],
            "wrong_key_detail": wrong_key.get("detail"),
        }, sort_keys=True))
    finally:
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
