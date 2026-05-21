#!/usr/bin/env python3
"""Subprocess smoke test for the CrowdTensorD external_llm_infer workload."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from auth_headers import activate_miner_token, activate_observer_token, add_miner_token_arg, add_observer_token_arg, coordinator_env, json_headers, observer_headers  # noqa: E402


ADMIN_HEADER = "x-crowdtensor-admin-token"


def request_json(
    method: str,
    base_url: str,
    path: str,
    payload: dict | None = None,
    *,
    timeout: float = 5.0,
    admin_token: str = "",
) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    if admin_token:
        headers = {"content-type": "application/json", ADMIN_HEADER: admin_token}
    else:
        headers = observer_headers() if method == "GET" else json_headers()
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
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
        str(args.request_count),
        "--backlog",
        "0",
        "--task-lane",
        "python-cli:cpu:1:external_llm_infer",
    ]
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
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


def run_miner(args: argparse.Namespace) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "miner_cli.py"),
            "--coordinator",
            args.base_url,
            "--miner-id",
            "external-llm-infer-smoke",
            "--once",
            "--enable-mock-llm-runtime",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        text=True,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "miner_cli.py failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def admin_results(args: argparse.Namespace, **params: str | int) -> dict:
    query = urlencode({key: value for key, value in params.items() if value != ""})
    path = "/admin/results" + (f"?{query}" if query else "")
    return request_json("GET", args.base_url, path, admin_token=args.admin_token)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external_llm_infer workload smoke test.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8906)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--request-count", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--admin-token", default="local-admin")
    add_miner_token_arg(parser)
    add_observer_token_arg(parser)
    args = parser.parse_args()
    activate_miner_token(args)
    activate_observer_token(args)
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    args.base_url = f"http://{args.host}:{args.port}"
    return args


def main() -> None:
    args = parse_args()
    temp_dir = None
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_external_llm_infer_")
        state_dir = Path(temp_dir.name)

    coordinator = None
    try:
        coordinator = start_coordinator(args, state_dir)
        miner_summary = run_miner(args)
        state = request_json("GET", args.base_url, "/state")
        completed = [
            task for task in state["tasks"]
            if task["status"] == "completed" and task["workload_type"] == "external_llm_infer"
        ]
        if not completed:
            raise SystemExit(f"missing completed external_llm_infer task: {json.dumps(state, sort_keys=True)}")
        if state["model"]["global_step"] != 0 or state.get("model_updates") != 0:
            raise SystemExit(f"external_llm_infer should not update dense model: {state['model']}")
        task = completed[-1]
        validation = task.get("validation", {})
        if validation.get("code") != "ok":
            raise SystemExit(f"unexpected validation: {validation}")
        if int(validation.get("request_count", 0)) != args.request_count:
            raise SystemExit(f"expected {args.request_count} LLM requests: {validation}")
        if int(validation.get("completion_count", 0)) != args.request_count:
            raise SystemExit(f"completion_count mismatch: {validation}")
        if validation.get("adapter_kind") != "mock":
            raise SystemExit(f"unexpected adapter kind: {validation}")
        if int(validation.get("output_chars", 0)) <= 0:
            raise SystemExit(f"missing output_chars: {validation}")
        metrics = task.get("metrics", {})
        if int(metrics.get("request_count", 0)) != args.request_count:
            raise SystemExit(f"task metrics request_count mismatch: {metrics}")
        if float(metrics.get("elapsed_ms", -1.0)) < 0.0:
            raise SystemExit(f"task metrics elapsed_ms invalid: {metrics}")
        if float(metrics.get("requests_per_second", 0.0)) <= 0.0:
            raise SystemExit(f"task metrics requests_per_second invalid: {metrics}")
        public_task_payload = json.dumps(state["tasks"], sort_keys=True)
        if "external_llm_results" in public_task_payload or "output_text" in public_task_payload:
            raise SystemExit(f"state leaked raw external LLM output: {public_task_payload}")
        profiles = state.get("miner_profiles", {})
        profile = profiles.get("external-llm-infer-smoke")
        if not profile:
            raise SystemExit(f"missing miner profile: {profiles}")
        capabilities = profile.get("last_capabilities") or {}
        if "external_llm_infer" not in capabilities.get("supported_workloads", []):
            raise SystemExit(f"miner did not advertise external_llm_infer: {profile}")
        if (capabilities.get("external_llm_runtime") or {}).get("adapter_kind") != "mock":
            raise SystemExit(f"missing mock external runtime capability: {profile}")
        ledger = admin_results(args, status="accepted", workload_type="external_llm_infer")
        results = ledger.get("results") or []
        if len(results) != 1:
            raise SystemExit(f"expected one external_llm_infer ledger row: {ledger}")
        row = results[0]
        if row.get("model_updated") or row.get("model_bundle_updated"):
            raise SystemExit(f"external LLM ledger row must be read-only: {row}")
        row_validation = row.get("validation", {})
        if row_validation.get("request_count") != args.request_count:
            raise SystemExit(f"ledger request_count mismatch: {row}")
        if row_validation.get("adapter_kind") != "mock":
            raise SystemExit(f"ledger adapter summary mismatch: {row}")
        if "output_text" in json.dumps(row, sort_keys=True):
            raise SystemExit(f"ledger leaked raw output_text: {row}")
        session_metrics = row.get("session_metrics", {})
        if int(session_metrics.get("request_count", 0)) != args.request_count:
            raise SystemExit(f"ledger session_metrics request_count mismatch: {row}")
        if int(session_metrics.get("completion_count", 0)) != args.request_count:
            raise SystemExit(f"ledger session_metrics completion_count mismatch: {row}")
        if float(session_metrics.get("requests_per_second", 0.0)) <= 0.0:
            raise SystemExit(f"ledger session_metrics requests_per_second invalid: {row}")
        print(json.dumps({
            "accepted_results": state["accepted_results"],
            "adapter_kind": validation.get("adapter_kind"),
            "completion_count": validation.get("completion_count"),
            "ledger_rows": len(results),
            "miner_summary": miner_summary,
            "model_id": validation.get("model_id"),
            "output_chars": validation.get("output_chars"),
            "request_count": validation.get("request_count"),
            "requests_per_second": session_metrics.get("requests_per_second"),
            "task_id": task["task_id"],
        }, sort_keys=True))
    finally:
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
