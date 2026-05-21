#!/usr/bin/env python3
"""Smoke test for Miner preflight and retry behavior against transient faults."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


class FaultState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.counts: dict[str, int] = {}
        self.claim: dict[str, Any] | None = None
        self.result_payload: dict[str, Any] | None = None

    def bump(self, key: str) -> int:
        with self.lock:
            self.counts[key] = self.counts.get(key, 0) + 1
            return self.counts[key]

    def set_claim(self, claim: dict[str, Any]) -> None:
        with self.lock:
            self.claim = dict(claim)

    def set_result(self, payload: dict[str, Any]) -> None:
        with self.lock:
            self.result_payload = dict(payload)


def make_claim() -> dict[str, Any]:
    return {
        "task_id": "task-resilience-1",
        "attempt": 1,
        "lease_token": "lease-resilience-1",
        "lease_expires_at": 4102444800.0,
        "model_version": 0,
        "weights": [0.0, 0.0, 0.0],
        "inner_steps": 4,
        "workload_type": "diloco_train",
        "workload_spec": {"type": "diloco_train"},
        "audit_mode": "none",
        "heartbeat_interval": 0.1,
        "schema_version": "diloco_mock_v1",
        "optimizer_step": 0,
        "task_requirements": {
            "runtime": "python-cli",
            "backend": "cpu",
            "protocol_version": "runtime_contract_v1",
        },
        "training_spec": {
            "schema_version": "diloco_mock_v1",
            "features": [
                [1.0, 0.0, 0.5],
                [0.5, -1.0, 1.0],
                [-1.0, 0.25, 0.75],
                [0.25, 1.0, -0.5],
                [1.5, -0.5, 0.25],
                [-0.75, -1.25, 1.0],
            ],
            "targets": [1.25, 3.5, 1.875, -2.25, 2.625, 3.125],
            "inner_lr": 0.03,
            "local_delta_scale": 0.1,
            "sample_offset": 0,
        },
    }


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("content-length", "0") or "0")
    raw = handler.rfile.read(length).decode("utf-8") if length else "{}"
    return json.loads(raw) if raw else {}


class FaultHandler(BaseHTTPRequestHandler):
    server: "FaultServer"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path != "/ready":
            json_response(self, 404, {"detail": "not found"})
            return
        count = self.server.state.bump("ready")
        if count == 1:
            json_response(self, 500, {"detail": "temporary ready failure"})
            return
        json_response(self, 200, {
            "ok": True,
            "service": "crowdtensord-coordinator",
            "version": "0.1.0a0",
            "protocol_version": "runtime_contract_v1",
            "default_workload_type": "diloco_train",
            "api_status": "alpha",
            "event_index": 1,
            "task_counts": {"queued": 1, "leased": 0, "completed": 0, "rejected": 0},
            "task_lanes": [],
            "auth": {
                "miner_required": False,
                "observer_required": False,
                "admin_configured": False,
                "miner_registry_configured": False,
            },
        })

    def do_POST(self) -> None:
        if self.path == "/tasks/claim":
            _payload = read_json_body(self)
            count = self.server.state.bump("claim")
            if count == 1:
                json_response(self, 500, {"detail": "temporary claim failure"})
                return
            claim = make_claim()
            self.server.state.set_claim(claim)
            json_response(self, 200, claim)
            return

        if self.path == "/tasks/task-resilience-1/heartbeat":
            payload = read_json_body(self)
            count = self.server.state.bump("heartbeat")
            if count == 1:
                json_response(self, 500, {"detail": "temporary heartbeat failure"})
                return
            if payload.get("lease_token") != "lease-resilience-1":
                json_response(self, 409, {"detail": "stale lease"})
                return
            json_response(self, 200, {
                "task_id": "task-resilience-1",
                "attempt": 1,
                "lease_expires_at": 4102444800.0,
            })
            return

        if self.path == "/tasks/task-resilience-1/result":
            payload = read_json_body(self)
            count = self.server.state.bump("result")
            if payload.get("lease_token") != "lease-resilience-1":
                json_response(self, 409, {"detail": "stale lease"})
                return
            if count == 1:
                json_response(self, 500, {"detail": "temporary result failure"})
                return
            if not isinstance(payload.get("local_delta"), list):
                json_response(self, 422, {"detail": "missing local_delta"})
                return
            if not payload.get("idempotency_key"):
                json_response(self, 422, {"detail": "missing idempotency_key"})
                return
            self.server.state.set_result(payload)
            json_response(self, 200, {
                "accepted": True,
                "model_version": 1,
                "global_step": 1,
                "optimizer_step": 1,
                "weights": [0.1, -0.1, 0.1],
                "outer_velocity": [0.0, 0.0, 0.0],
                "loss": 1.0,
                "staleness": 0,
            })
            return

        json_response(self, 404, {"detail": "not found"})


class FaultServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], state: FaultState) -> None:
        super().__init__(address, FaultHandler)
        self.state = state


def parse_summary(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "accepted_tasks" in payload:
            return payload
    raise RuntimeError(f"missing miner summary JSON:\n{stdout}")


def run_miner(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "miner_cli.py"),
        "--coordinator",
        f"http://{args.host}:{args.port}",
        "--miner-id",
        "miner-resilience-smoke",
        "--once",
        "--compute-seconds",
        str(args.compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--preflight-timeout",
        "2.0",
        "--claim-timeout",
        "2.0",
        "--heartbeat-timeout",
        "2.0",
        "--result-timeout",
        "2.0",
        "--max-request-attempts",
        "3",
        "--retry-base-sleep",
        "0.01",
        "--retry-max-sleep",
        "0.05",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=args.miner_timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "miner resilience smoke failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return parse_summary(completed.stdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Miner resilience retry smoke.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8894)
    parser.add_argument("--state-dir", default="", help="accepted for runtime acceptance compatibility")
    parser.add_argument("--compute-seconds", type=float, default=0.35)
    parser.add_argument("--heartbeat-interval", type=float, default=0.05)
    parser.add_argument("--miner-timeout", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = FaultState()
    server = FaultServer((args.host, args.port), state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        miner_summary = run_miner(args)
        counts = dict(state.counts)
        if int(miner_summary.get("accepted_tasks", 0)) != 1:
            raise SystemExit(f"expected one accepted task: {json.dumps(miner_summary, sort_keys=True)}")
        if int(miner_summary.get("request_retries", 0)) < 4:
            raise SystemExit(f"expected ready/claim/heartbeat/result retries: {json.dumps(miner_summary, sort_keys=True)}")
        if counts.get("ready", 0) < 2 or counts.get("claim", 0) < 2 or counts.get("heartbeat", 0) < 2:
            raise SystemExit(f"expected retryable endpoint hits: {json.dumps(counts, sort_keys=True)}")
        if counts.get("result", 0) != 2:
            raise SystemExit(f"result upload should retry exactly once: {json.dumps(counts, sort_keys=True)}")
        if not (state.result_payload or {}).get("idempotency_key"):
            raise SystemExit(f"result upload must include idempotency_key: {state.result_payload}")

        print(json.dumps({
            "endpoint_counts": counts,
            "miner_summary": miner_summary,
            "result_uploads": counts.get("result", 0),
        }, sort_keys=True))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
