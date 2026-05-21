#!/usr/bin/env python3
"""Smoke test for the admin result traceability ledger."""

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
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor.diloco import run_inner_loop  # noqa: E402


ADMIN_TOKEN = "result-ledger-admin"
ACCEPTED_MINER = "result-ledger-accepted"
REJECTED_MINER = "result-ledger-rejected"


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
        "10",
        "--inner-steps",
        "8",
        "--backlog",
        "1",
        "--replay-audit",
        "--admin-token",
        ADMIN_TOKEN,
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


def claim_task(args: argparse.Namespace, miner_id: str) -> dict:
    return request_json("POST", args.base_url, "/tasks/claim", {"miner_id": miner_id})


def result_payload(claim: dict, miner_id: str, *, idempotency_key: str) -> dict:
    inner_result = run_inner_loop(
        claim["weights"],
        task_id=claim["task_id"],
        miner_id=miner_id,
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


def submit_accepted(args: argparse.Namespace) -> dict:
    claim = claim_task(args, ACCEPTED_MINER)
    payload = result_payload(claim, ACCEPTED_MINER, idempotency_key="accepted-key")
    result = request_json("POST", args.base_url, f"/tasks/{claim['task_id']}/result", payload)
    if result.get("accepted") is not True:
        raise RuntimeError(f"expected accepted result, got {result}")
    return claim


def submit_rejected(args: argparse.Namespace) -> dict:
    claim = claim_task(args, REJECTED_MINER)
    payload = result_payload(claim, REJECTED_MINER, idempotency_key="rejected-key")
    payload["local_delta"] = [value * 0.25 for value in payload["local_delta"]]
    rejected = request_json(
        "POST",
        args.base_url,
        f"/tasks/{claim['task_id']}/result",
        payload,
        expected_status=422,
    )
    detail = rejected.get("detail") or {}
    if detail.get("audit_accepted") is not False:
        raise RuntimeError(f"expected replay audit rejection, got {rejected}")
    return claim


def admin_results(args: argparse.Namespace, **params: str | int) -> dict:
    query = urlencode({key: value for key, value in params.items() if value != ""})
    path = "/admin/results" + (f"?{query}" if query else "")
    return request_json("GET", args.base_url, path, admin_token=ADMIN_TOKEN)


def assert_no_secret_leaks(payload: dict) -> None:
    text = json.dumps(payload, sort_keys=True)
    forbidden = [
        "accepted-key",
        "rejected-key",
        '"lease_token"',
        '"result_idempotency_key_hash"',
        '"result_lease_token_hash"',
        '"result_response"',
        '"local_delta"',
        '"adapter_delta"',
    ]
    leaked = [fragment for fragment in forbidden if fragment in text]
    if leaked:
        raise RuntimeError(f"result ledger leaked sensitive fields {leaked}: {text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run admin result ledger smoke.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8897)
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
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_result_ledger_")
        state_dir = Path(temp_dir.name)

    coordinator = None
    try:
        coordinator = start_coordinator(args, state_dir)
        accepted_claim = submit_accepted(args)
        rejected_claim = submit_rejected(args)

        unauthorized = request_json("GET", args.base_url, "/admin/results", expected_status=403)
        invalid = request_json(
            "GET",
            args.base_url,
            "/admin/results?status=broken",
            admin_token=ADMIN_TOKEN,
            expected_status=422,
        )
        all_results = admin_results(args, limit=10)
        accepted = admin_results(args, status="accepted")
        rejected = admin_results(args, status="rejected")
        filtered = admin_results(args, miner_id=ACCEPTED_MINER, workload_type="diloco_train")

        assert_no_secret_leaks(all_results)
        rows = all_results.get("results") or []
        if len(rows) != 2:
            raise SystemExit(f"expected two ledger rows: {json.dumps(all_results, sort_keys=True)}")
        if [row["event_index"] for row in rows] != sorted([row["event_index"] for row in rows], reverse=True):
            raise SystemExit(f"ledger rows are not newest-first: {json.dumps(all_results, sort_keys=True)}")
        if len(accepted.get("results") or []) != 1 or len(rejected.get("results") or []) != 1:
            raise SystemExit(f"accepted/rejected filters failed: accepted={accepted} rejected={rejected}")
        if len(filtered.get("results") or []) != 1:
            raise SystemExit(f"miner/workload filter failed: {json.dumps(filtered, sort_keys=True)}")

        accepted_row = accepted["results"][0]
        rejected_row = rejected["results"][0]
        if accepted_row["task_id"] != accepted_claim["task_id"] or accepted_row["idempotent"] is not True:
            raise SystemExit(f"accepted ledger row mismatch: {json.dumps(accepted_row, sort_keys=True)}")
        if rejected_row["task_id"] != rejected_claim["task_id"] or rejected_row["audit"].get("audit_accepted") is not False:
            raise SystemExit(f"rejected ledger row mismatch: {json.dumps(rejected_row, sort_keys=True)}")
        if accepted_row["validation"].get("code") != "ok":
            raise SystemExit(f"accepted validation summary missing: {json.dumps(accepted_row, sort_keys=True)}")

        print(json.dumps({
            "accepted_rows": len(accepted["results"]),
            "forbidden_admin_results": unauthorized.get("detail"),
            "invalid_status": invalid.get("detail"),
            "latest_status": rows[0]["status"],
            "rejected_audit_code": rejected_row["audit"].get("audit_code"),
            "rows": len(rows),
        }, sort_keys=True))
    finally:
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
