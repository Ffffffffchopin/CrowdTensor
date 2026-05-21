#!/usr/bin/env python3
"""Executable HTTP API contract smoke for the CrowdTensorD Coordinator."""

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
from crowdtensor.diloco import run_inner_loop  # noqa: E402


ADMIN_TOKEN = "api-contract-admin"
MINER_TOKEN = "api-contract-miner"
OBSERVER_TOKEN = "api-contract-observer"
REGISTERED_MINER = "api-contract-registered"
REGISTERED_TOKEN = "api-contract-registered-token"
DISABLED_MINER = "api-contract-disabled"
DISABLED_TOKEN = "api-contract-disabled-token"
SHARED_MINER = "api-contract-shared"
BAD_MINER = "api-contract-bad"


def request_text(
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
) -> tuple[str, dict[str, str]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers: dict[str, str] = {}
    if payload is not None:
        headers["content-type"] = "application/json"
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
    payload: dict | None = None,
    *,
    miner_token: str = "",
    observer_token: str = "",
    admin_token: str = "",
    expected_status: int = 200,
    timeout: float = 5.0,
) -> dict:
    raw, _headers = request_text(
        method,
        base_url,
        path,
        payload,
        miner_token=miner_token,
        observer_token=observer_token,
        admin_token=admin_token,
        expected_status=expected_status,
        timeout=timeout,
    )
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


def write_registry(path: Path) -> Path:
    payload = {
        "miners": [
            {"miner_id": REGISTERED_MINER, "token": hash_token(REGISTERED_TOKEN), "enabled": True},
            {"miner_id": DISABLED_MINER, "token": hash_token(DISABLED_TOKEN), "enabled": False},
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
        "2",
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


def assert_keys(payload: dict, keys: set[str], *, path: str) -> None:
    missing = sorted(keys.difference(payload))
    if missing:
        raise RuntimeError(f"{path} missing fields: {missing}; payload={json.dumps(payload, sort_keys=True)}")


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


def claim_task(args: argparse.Namespace, miner_id: str, token: str, *, expected_status: int = 200) -> dict:
    return request_json(
        "POST",
        args.base_url,
        "/tasks/claim",
        {
            "miner_id": miner_id,
            "capabilities": {
                "runtime": "python-cli",
                "backend": "cpu",
                "protocol_version": "runtime_contract_v1",
                "supported_workloads": ["diloco_train"],
            },
        },
        miner_token=token,
        expected_status=expected_status,
    )


def complete_claim(args: argparse.Namespace, claim: dict, miner_id: str, token: str) -> dict:
    inner_result = run_inner_loop(
        claim["weights"],
        task_id=claim["task_id"],
        miner_id=miner_id,
        model_version=int(claim["model_version"]),
        inner_steps=int(claim["inner_steps"]),
        training_spec=claim["training_spec"],
    )
    return request_json(
        "POST",
        args.base_url,
        f"/tasks/{claim['task_id']}/result",
        {
            "lease_token": claim["lease_token"],
            "attempt": claim["attempt"],
            "local_delta": inner_result["local_delta"],
            "metrics": {**inner_result, "elapsed_ms": 1.0},
        },
        miner_token=token,
    )


def verify_public_endpoints(args: argparse.Namespace) -> dict:
    health = request_json("GET", args.base_url, "/health")
    version = request_json("GET", args.base_url, "/version")
    ready = request_json("GET", args.base_url, "/ready")

    assert_keys(health, {"ok", "service", "version"}, path="/health")
    if health.get("ok") is not True:
        raise RuntimeError(f"/health should be ok: {health}")
    assert_public_profile(version, path="/version")
    assert_public_profile(ready, path="/ready")
    assert_keys(ready, {"ok", "event_index", "task_counts", "task_lanes", "auth"}, path="/ready")
    auth = ready.get("auth") or {}
    expected_auth = {
        "miner_required": True,
        "observer_required": True,
        "admin_configured": True,
        "miner_registry_configured": True,
    }
    for key, value in expected_auth.items():
        if auth.get(key) is not value:
            raise RuntimeError(f"/ready.auth.{key} expected {value}, got {auth.get(key)}")
    if "queued" not in ready.get("task_counts", {}):
        raise RuntimeError(f"/ready should include queued task count: {ready}")
    return {
        "health": True,
        "version": version.get("version"),
        "ready_auth": auth,
    }


def verify_observer_endpoints(args: argparse.Namespace) -> dict:
    missing_state = request_json("GET", args.base_url, "/state", expected_status=401)
    bad_metrics = request_json("GET", args.base_url, "/metrics", observer_token="bad", expected_status=401)
    state = request_json("GET", args.base_url, "/state", observer_token=OBSERVER_TOKEN)
    metrics_text, metrics_headers = request_text("GET", args.base_url, "/metrics", observer_token=OBSERVER_TOKEN)

    if missing_state.get("detail") != "invalid observer token":
        raise RuntimeError(f"unexpected missing observer response: {missing_state}")
    if bad_metrics.get("detail") != "invalid observer token":
        raise RuntimeError(f"unexpected bad observer response: {bad_metrics}")
    assert_keys(state, {"model", "event_index", "task_counts", "tasks"}, path="/state")
    content_type = metrics_headers.get("Content-Type", metrics_headers.get("content-type", ""))
    if "text/plain" not in content_type or "crowdtensord_task_count" not in metrics_text:
        raise RuntimeError(f"unexpected /metrics response: {content_type} {metrics_text}")
    return {
        "state_protected": True,
        "metrics_protected": True,
        "metrics_content_type": content_type,
    }


def verify_admin_endpoints(args: argparse.Namespace) -> dict:
    missing_events = request_json("GET", args.base_url, "/admin/events?limit=1", expected_status=403)
    missing_results = request_json("GET", args.base_url, "/admin/results?limit=1", expected_status=403)
    bad_override = request_json(
        "POST",
        args.base_url,
        "/admin/trust-overrides",
        {"miner_id": BAD_MINER, "workload_type": "diloco_train", "mode": "block"},
        admin_token="bad",
        expected_status=403,
    )
    block = request_json(
        "POST",
        args.base_url,
        "/admin/trust-overrides",
        {
            "miner_id": BAD_MINER,
            "workload_type": "diloco_train",
            "mode": "block",
            "reason": "api contract block",
        },
        admin_token=ADMIN_TOKEN,
    )
    blocked_claim = claim_task(args, BAD_MINER, MINER_TOKEN, expected_status=503)
    reset = request_json(
        "POST",
        args.base_url,
        "/admin/trust-overrides",
        {
            "miner_id": BAD_MINER,
            "workload_type": "diloco_train",
            "mode": "none",
            "reason": "api contract reset",
        },
        admin_token=ADMIN_TOKEN,
    )
    invalid_mode = request_json(
        "POST",
        args.base_url,
        "/admin/trust-overrides",
        {"miner_id": BAD_MINER, "workload_type": "diloco_train", "mode": "invalid"},
        admin_token=ADMIN_TOKEN,
        expected_status=422,
    )
    events = request_json("GET", args.base_url, "/admin/events?limit=50", admin_token=ADMIN_TOKEN)
    invalid_results = request_json(
        "GET",
        args.base_url,
        "/admin/results?status=broken",
        admin_token=ADMIN_TOKEN,
        expected_status=422,
    )
    results = request_json("GET", args.base_url, "/admin/results?limit=10", admin_token=ADMIN_TOKEN)

    if missing_events.get("detail") != "invalid admin token":
        raise RuntimeError(f"unexpected missing admin response: {missing_events}")
    if missing_results.get("detail") != "invalid admin token":
        raise RuntimeError(f"unexpected missing admin results response: {missing_results}")
    if bad_override.get("detail") != "invalid admin token":
        raise RuntimeError(f"unexpected bad admin response: {bad_override}")
    if block.get("mode") != "block" or reset.get("mode") != "none":
        raise RuntimeError(f"unexpected trust override response: block={block} reset={reset}")
    if blocked_claim.get("detail") != "miner manually blocked for workload":
        raise RuntimeError(f"unexpected blocked claim response: {blocked_claim}")
    if "allow, block, or none" not in str(invalid_mode.get("detail")):
        raise RuntimeError(f"unexpected invalid mode response: {invalid_mode}")
    if not any(event.get("type") == "trust_override_set" for event in events.get("events", [])):
        raise RuntimeError(f"missing trust override event in admin tail: {events}")
    if any(event.get("lease_token") and event.get("lease_token") != "<redacted>" for event in events.get("events", [])):
        raise RuntimeError(f"unredacted lease token in admin events: {events}")
    if "accepted, or rejected" not in str(invalid_results.get("detail")):
        raise RuntimeError(f"unexpected invalid result ledger status response: {invalid_results}")
    assert_keys(results, {"results", "limit", "status", "miner_id", "workload_type"}, path="/admin/results")
    if "result_idempotency_key_hash" in json.dumps(results, sort_keys=True):
        raise RuntimeError(f"admin result ledger leaked idempotency hash: {results}")
    return {
        "admin_events": len(events.get("events", [])),
        "admin_results": len(results.get("results", [])),
        "blocked_claim_detail": blocked_claim.get("detail"),
        "invalid_mode_status": 422,
    }


def verify_miner_endpoints(args: argparse.Namespace) -> dict:
    missing_claim = claim_task(args, "missing-token", "", expected_status=401)
    bad_claim = claim_task(args, "bad-token", "bad", expected_status=401)
    disabled_claim = claim_task(args, DISABLED_MINER, DISABLED_TOKEN, expected_status=401)
    wrong_registered = claim_task(args, REGISTERED_MINER, MINER_TOKEN, expected_status=401)
    registered_claim = claim_task(args, REGISTERED_MINER, REGISTERED_TOKEN)

    assert_keys(
        registered_claim,
        {
            "task_id",
            "attempt",
            "lease_token",
            "lease_expires_at",
            "model_version",
            "weights",
            "inner_steps",
            "workload_type",
            "workload_spec",
            "audit_mode",
            "heartbeat_interval",
            "schema_version",
            "optimizer_step",
            "task_requirements",
            "training_spec",
        },
        path="/tasks/claim",
    )
    if registered_claim.get("workload_type") != "diloco_train":
        raise RuntimeError(f"expected diloco_train claim: {registered_claim}")
    if missing_claim.get("detail") != "invalid miner token":
        raise RuntimeError(f"unexpected missing miner response: {missing_claim}")
    if bad_claim.get("detail") != "invalid miner token":
        raise RuntimeError(f"unexpected bad miner response: {bad_claim}")
    if disabled_claim.get("detail") != "miner token is disabled":
        raise RuntimeError(f"unexpected disabled miner response: {disabled_claim}")
    if wrong_registered.get("detail") != "invalid miner token":
        raise RuntimeError(f"unexpected wrong registered miner response: {wrong_registered}")

    bad_heartbeat = request_json(
        "POST",
        args.base_url,
        f"/tasks/{registered_claim['task_id']}/heartbeat",
        {
            "lease_token": registered_claim["lease_token"],
            "attempt": registered_claim["attempt"],
            "runtime_status": {"phase": "bad-token"},
        },
        miner_token="bad",
        expected_status=401,
    )
    heartbeat = request_json(
        "POST",
        args.base_url,
        f"/tasks/{registered_claim['task_id']}/heartbeat",
        {
            "lease_token": registered_claim["lease_token"],
            "attempt": registered_claim["attempt"],
            "runtime_status": {"phase": "api-contract"},
        },
        miner_token=REGISTERED_TOKEN,
    )
    if bad_heartbeat.get("detail") != "invalid miner token":
        raise RuntimeError(f"unexpected bad heartbeat token response: {bad_heartbeat}")
    if heartbeat.get("task_id") != registered_claim["task_id"]:
        raise RuntimeError(f"unexpected heartbeat response: {heartbeat}")

    bad_result = request_json(
        "POST",
        args.base_url,
        f"/tasks/{registered_claim['task_id']}/result",
        {
            "lease_token": "stale",
            "attempt": registered_claim["attempt"],
            "local_delta": [0.0, 0.0, 0.0],
        },
        miner_token=REGISTERED_TOKEN,
        expected_status=409,
    )
    result = complete_claim(args, registered_claim, REGISTERED_MINER, REGISTERED_TOKEN)
    duplicate_result = request_json(
        "POST",
        args.base_url,
        f"/tasks/{registered_claim['task_id']}/result",
        {
            "lease_token": registered_claim["lease_token"],
            "attempt": registered_claim["attempt"],
            "local_delta": [0.0, 0.0, 0.0],
        },
        miner_token=REGISTERED_TOKEN,
        expected_status=409,
    )
    shared_claim = claim_task(args, SHARED_MINER, MINER_TOKEN)
    shared_result = complete_claim(args, shared_claim, SHARED_MINER, MINER_TOKEN)

    if "stale" not in str(bad_result.get("detail")):
        raise RuntimeError(f"unexpected stale result response: {bad_result}")
    if result.get("accepted") is not True or shared_result.get("accepted") is not True:
        raise RuntimeError(f"expected accepted results: registered={result} shared={shared_result}")
    if "not leased" not in str(duplicate_result.get("detail")):
        raise RuntimeError(f"unexpected duplicate result response: {duplicate_result}")

    state = request_json("GET", args.base_url, "/state", observer_token=OBSERVER_TOKEN)
    tasks = state.get("tasks") or []
    if any(task.get("lease_token") and task.get("lease_token") != "<redacted>" for task in tasks):
        raise RuntimeError(f"unredacted lease token in state tasks: {state}")
    if int(state.get("accepted_results", 0)) != 2:
        raise RuntimeError(f"expected two accepted results: {state}")
    return {
        "accepted_results": state.get("accepted_results"),
        "disabled_detail": disabled_claim.get("detail"),
        "registered_bad_detail": wrong_registered.get("detail"),
        "stale_result_status": 409,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Coordinator HTTP API contract smoke.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8891)
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
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_api_contract_")
        state_dir = Path(temp_dir.name)
    state_dir.mkdir(parents=True, exist_ok=True)
    registry_path = write_registry(state_dir / "miner_registry.json")

    coordinator = None
    try:
        coordinator = start_coordinator(args, state_dir, registry_path)
        public = verify_public_endpoints(args)
        observer = verify_observer_endpoints(args)
        admin = verify_admin_endpoints(args)
        miner = verify_miner_endpoints(args)
        print(json.dumps({
            "admin": admin,
            "miner": miner,
            "observer": observer,
            "public": public,
            "registry_hash_prefix": hash_token(REGISTERED_TOKEN).split(":", 1)[0],
        }, sort_keys=True))
    finally:
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
