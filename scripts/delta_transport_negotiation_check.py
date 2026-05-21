#!/usr/bin/env python3
"""Smoke test for claim-time delta transport negotiation."""

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

from crowdtensor.outer_optimizer import DELTA_FORMAT_SIGN_COMPRESSED  # noqa: E402


ADMIN_TOKEN = "delta-transport-admin"
LEGACY_MINER = "delta-transport-legacy"
AUTO_MINER = "delta-transport-auto"


def request_json(
    method: str,
    base_url: str,
    path: str,
    payload: dict | None = None,
    *,
    admin_token: str = "",
    expected_status: int = 200,
    timeout: float = 5.0,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json"} if payload is not None else {}
    if admin_token:
        headers["x-crowdtensor-admin-token"] = admin_token
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
        str(args.lease_seconds),
        "--inner-steps",
        str(args.inner_steps),
        "--backlog",
        "1",
        "--admin-token",
        ADMIN_TOKEN,
        "--delta-format",
        DELTA_FORMAT_SIGN_COMPRESSED,
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


def legacy_incompatible_claim(args: argparse.Namespace) -> dict:
    payload = request_json(
        "POST",
        args.base_url,
        "/tasks/claim",
        {
            "miner_id": LEGACY_MINER,
            "capabilities": {
                "runtime": "python-cli",
                "backend": "cpu",
                "protocol_version": "runtime_contract_v1",
                "supported_workloads": ["diloco_train"],
            },
        },
        expected_status=503,
    )
    if payload.get("detail") != "no compatible queued task available":
        raise RuntimeError(f"unexpected legacy incompatible response: {payload}")
    return payload


def run_auto_miner(args: argparse.Namespace) -> dict:
    command = [
        sys.executable,
        str(ROOT / "miner_cli.py"),
        "--coordinator",
        args.base_url,
        "--miner-id",
        AUTO_MINER,
        "--once",
        "--delta-format",
        "auto",
        "--claim-timeout",
        str(args.claim_timeout),
        "--result-timeout",
        str(args.result_timeout),
        "--preflight-timeout",
        str(args.preflight_timeout),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        text=True,
        capture_output=True,
        timeout=args.miner_timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"auto miner failed with code {completed.returncode}: "
            f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("auto miner produced no stdout")
    summary = json.loads(lines[-1])
    if summary.get("accepted_tasks") != 1:
        raise RuntimeError(f"auto miner did not accept exactly one task: {summary}")
    return summary


def assert_no_transport_leaks(payload: dict) -> None:
    text = json.dumps(payload, sort_keys=True)
    for forbidden in ["compressed_delta", '"signs"', '"lease_token"']:
        if forbidden in text:
            raise RuntimeError(f"ledger leaked private transport detail {forbidden}: {text}")


def verify_negotiated_state(args: argparse.Namespace) -> dict:
    state = request_json("GET", args.base_url, "/state")
    if state.get("accepted_results") != 1:
        raise RuntimeError(f"expected one accepted negotiated result: {state}")
    if state.get("incompatible_claims") != 1:
        raise RuntimeError(f"expected one incompatible legacy claim: {state}")
    incompatible = state.get("last_incompatible_claim") or {}
    if incompatible.get("miner_id") != LEGACY_MINER:
        raise RuntimeError(f"unexpected incompatible claim summary: {state}")
    last_completed = state.get("last_completed") or {}
    optimizer = last_completed.get("optimizer") or {}
    if optimizer.get("delta_format") != DELTA_FORMAT_SIGN_COMPRESSED:
        raise RuntimeError(f"last completed task did not use negotiated sign compression: {state}")

    ledger = request_json("GET", args.base_url, "/admin/results?status=accepted", admin_token=ADMIN_TOKEN)
    rows = ledger.get("results") or []
    if len(rows) != 1:
        raise RuntimeError(f"expected one accepted ledger row: {ledger}")
    row_optimizer = rows[0].get("optimizer") or {}
    if row_optimizer.get("delta_format") != DELTA_FORMAT_SIGN_COMPRESSED:
        raise RuntimeError(f"ledger missing negotiated sign compression summary: {ledger}")
    assert_no_transport_leaks(ledger)
    return {
        "accepted_results": state["accepted_results"],
        "incompatible_claims": state["incompatible_claims"],
        "delta_format": row_optimizer["delta_format"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run delta transport negotiation smoke.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8901)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--inner-steps", type=int, default=10)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--claim-timeout", type=float, default=5.0)
    parser.add_argument("--result-timeout", type=float, default=5.0)
    parser.add_argument("--preflight-timeout", type=float, default=5.0)
    parser.add_argument("--miner-timeout", type=float, default=20.0)
    args = parser.parse_args()
    args.base_url = f"http://{args.host}:{args.port}"
    return args


def main() -> None:
    args = parse_args()
    temp_dir = None
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_delta_transport_")
        state_dir = Path(temp_dir.name)
    state_dir.mkdir(parents=True, exist_ok=True)

    coordinator = None
    try:
        coordinator = start_coordinator(args, state_dir)
        legacy = legacy_incompatible_claim(args)
        miner = run_auto_miner(args)
        state = verify_negotiated_state(args)
        print(json.dumps({"legacy": legacy, "miner": miner, **state}, sort_keys=True))
    finally:
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
