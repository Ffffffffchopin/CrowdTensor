#!/usr/bin/env python3
"""Subprocess smoke test for sign-compressed error-feedback delta transport."""

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
from crowdtensor.outer_optimizer import (  # noqa: E402
    DELTA_FORMAT_DENSE_FLOAT,
    DELTA_FORMAT_SIGN_COMPRESSED_EF,
    compress_sign_delta_with_error_feedback,
)


ADMIN_TOKEN = "compressed-error-feedback-admin"


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


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def start_coordinator(
    args: argparse.Namespace,
    state_dir: Path,
    *,
    replay_audit: bool,
) -> subprocess.Popen:
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
    ]
    if replay_audit:
        command.append("--replay-audit")
    else:
        command.extend(["--delta-format", DELTA_FORMAT_SIGN_COMPRESSED_EF])
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
                "supported_delta_formats": [
                    DELTA_FORMAT_DENSE_FLOAT,
                    DELTA_FORMAT_SIGN_COMPRESSED_EF,
                ],
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


def submit_error_feedback(
    base_url: str,
    claimed: dict,
    miner_id: str,
    residual: list[float] | None,
) -> tuple[dict, list[float]]:
    inner = inner_result_for(claimed, miner_id)
    compressed, next_residual = compress_sign_delta_with_error_feedback(
        inner["local_delta"],
        residual=residual,
    )
    result = request_json(
        "POST",
        base_url,
        f"/tasks/{claimed['task_id']}/result",
        {
            "lease_token": claimed["lease_token"],
            "attempt": claimed["attempt"],
            "compressed_delta": compressed,
            "metrics": {"delta_format": DELTA_FORMAT_SIGN_COMPRESSED_EF, "elapsed_ms": 1.0},
        },
    )
    return result, next_residual


def accepted_flow(args: argparse.Namespace, state_dir: Path) -> dict:
    coordinator = None
    try:
        coordinator = start_coordinator(args, state_dir, replay_audit=False)
        residual: list[float] | None = None
        first_claim = claim(args.base_url, "compressed-ef-1")
        first_result, residual = submit_error_feedback(args.base_url, first_claim, "compressed-ef-1", residual)
        second_claim = claim(args.base_url, "compressed-ef-2")
        second_result, residual = submit_error_feedback(args.base_url, second_claim, "compressed-ef-2", residual)

        if (first_result.get("optimizer") or {}).get("delta_format") != DELTA_FORMAT_SIGN_COMPRESSED_EF:
            raise RuntimeError(f"first result missing error-feedback optimizer summary: {first_result}")
        optimizer = second_result.get("optimizer") or {}
        if optimizer.get("delta_format") != DELTA_FORMAT_SIGN_COMPRESSED_EF or optimizer.get("error_feedback") is not True:
            raise RuntimeError(f"second result missing error-feedback optimizer summary: {second_result}")
        if "residual_norm" not in optimizer or "corrected_delta_norm" not in optimizer:
            raise RuntimeError(f"second result missing error-feedback norms: {second_result}")

        state = request_json("GET", args.base_url, "/state")
        ledger = request_json("GET", args.base_url, "/admin/results?status=accepted", admin_token=ADMIN_TOKEN)
        rows = ledger.get("results") or []
        if state.get("accepted_results") != 2 or len(rows) != 2:
            raise RuntimeError(f"unexpected accepted error-feedback state: state={state} ledger={ledger}")
        public_text = json.dumps(ledger, sort_keys=True)
        for forbidden in ["compressed_delta", "signs", "lease_token"]:
            if forbidden in public_text:
                raise RuntimeError(f"ledger leaked private transport detail {forbidden}: {public_text}")
        return {
            "accepted_results": state["accepted_results"],
            "delta_format": optimizer["delta_format"],
            "residual_norm": optimizer["residual_norm"],
            "residual_length": len(residual or []),
        }
    finally:
        stop_process(coordinator)


def replay_audit_rejects_flow(args: argparse.Namespace, state_dir: Path) -> dict:
    coordinator = None
    try:
        coordinator = start_coordinator(args, state_dir, replay_audit=True)
        claimed = claim(args.base_url, "compressed-ef-audit")
        inner = inner_result_for(claimed, "compressed-ef-audit")
        compressed, _residual = compress_sign_delta_with_error_feedback(inner["local_delta"])
        rejected = request_json(
            "POST",
            args.base_url,
            f"/tasks/{claimed['task_id']}/result",
            {
                "lease_token": claimed["lease_token"],
                "attempt": claimed["attempt"],
                "compressed_delta": compressed,
                "metrics": {"delta_format": DELTA_FORMAT_SIGN_COMPRESSED_EF},
            },
            expected_status=422,
        )
        detail = rejected.get("detail") or {}
        if detail.get("code") != "local_delta_replay_mismatch":
            raise RuntimeError(f"unexpected replay audit rejection code: {rejected}")
        if detail.get("audit_code") != "error_feedback_replay_unsupported":
            raise RuntimeError(f"unexpected replay audit error-feedback code: {rejected}")
        state = request_json("GET", args.base_url, "/state")
        if state.get("audit_results") != 1 or state.get("audit_rejections") != 1:
            raise RuntimeError(f"unexpected replay audit counters: {state}")
        return {
            "replay_audit_rejected": True,
            "audit_code": detail["audit_code"],
        }
    finally:
        stop_process(coordinator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sign-compressed error-feedback transport smoke test.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8900)
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
    state_root = Path(args.state_dir) if args.state_dir else None
    if state_root is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_compressed_ef_")
        state_root = Path(temp_dir.name)
    state_root.mkdir(parents=True, exist_ok=True)

    try:
        accepted = accepted_flow(args, state_root / "accepted")
        replay = replay_audit_rejects_flow(args, state_root / "replay")
        print(json.dumps({**accepted, **replay}, sort_keys=True))
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
