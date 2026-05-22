#!/usr/bin/env python3
"""Smoke test for external_llm_infer through an OpenAI-compatible HTTP runtime."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from auth_headers import activate_miner_token, activate_observer_token, add_miner_token_arg, add_observer_token_arg, coordinator_env, json_headers, observer_headers  # noqa: E402


ADMIN_HEADER = "x-crowdtensor-admin-token"


class MockOpenAIHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        prompt = ""
        messages = payload.get("messages") or []
        if messages and isinstance(messages[-1], dict):
            prompt = str(messages[-1].get("content") or "")
        self.__class__.requests.append({
            "path": self.path,
            "model": payload.get("model"),
            "max_tokens": payload.get("max_tokens"),
            "prompt": prompt,
            "authorization": self.headers.get("authorization", ""),
        })
        body = json.dumps({
            "id": "chatcmpl-crowdtensor-smoke",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"http completion for: {prompt}",
                    },
                    "finish_reason": "stop",
                }
            ],
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


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


def start_mock_runtime(args: argparse.Namespace) -> ThreadingHTTPServer:
    MockOpenAIHandler.requests = []
    server = ThreadingHTTPServer((args.host, args.runtime_port), MockOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, name="external-llm-http-smoke", daemon=True)
    thread.start()
    return server


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
            "external-llm-http-smoke",
            "--once",
            "--llm-runtime-url",
            args.runtime_url,
            "--llm-runtime-model-id",
            args.model_id,
            "--llm-runtime-api-key",
            args.runtime_api_key,
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
    parser = argparse.ArgumentParser(description="Run external_llm_infer HTTP adapter smoke test.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8907)
    parser.add_argument("--runtime-port", type=int, default=8908)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--request-count", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--admin-token", default="local-admin")
    parser.add_argument("--model-id", default="http-smoke-model")
    parser.add_argument("--runtime-api-key", default="local-runtime-key")
    add_miner_token_arg(parser)
    add_observer_token_arg(parser)
    args = parser.parse_args()
    activate_miner_token(args)
    activate_observer_token(args)
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    args.base_url = f"http://{args.host}:{args.port}"
    args.runtime_url = f"http://{args.host}:{args.runtime_port}/v1/chat/completions"
    return args


def main() -> None:
    args = parse_args()
    temp_dir = None
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_external_llm_http_")
        state_dir = Path(temp_dir.name)

    coordinator = None
    runtime = None
    try:
        runtime = start_mock_runtime(args)
        coordinator = start_coordinator(args, state_dir)
        miner_summary = run_miner(args)
        state = request_json("GET", args.base_url, "/state")
        completed = [
            task for task in state["tasks"]
            if task["status"] == "completed" and task["workload_type"] == "external_llm_infer"
        ]
        if not completed:
            raise SystemExit(f"missing completed external_llm_infer task: {json.dumps(state, sort_keys=True)}")
        task = completed[-1]
        validation = task.get("validation", {})
        if validation.get("code") != "ok":
            raise SystemExit(f"unexpected validation: {validation}")
        if validation.get("adapter_kind") != "http_openai_chat":
            raise SystemExit(f"unexpected adapter kind: {validation}")
        if validation.get("model_id") != args.model_id:
            raise SystemExit(f"unexpected model_id: {validation}")
        if int(validation.get("request_count", 0)) != args.request_count:
            raise SystemExit(f"request_count mismatch: {validation}")
        if state["model"]["global_step"] != 0 or state.get("model_updates") != 0:
            raise SystemExit(f"external_llm_infer should not update dense model: {state['model']}")
        if len(MockOpenAIHandler.requests) != args.request_count:
            raise SystemExit(f"runtime request count mismatch: {MockOpenAIHandler.requests}")
        if any(row.get("model") != args.model_id for row in MockOpenAIHandler.requests):
            raise SystemExit(f"runtime model_id mismatch: {MockOpenAIHandler.requests}")
        if any(row.get("authorization") != f"Bearer {args.runtime_api_key}" for row in MockOpenAIHandler.requests):
            raise SystemExit("runtime authorization header missing")
        public_task_payload = json.dumps(state["tasks"], sort_keys=True)
        if "output_text" in public_task_payload or "external_llm_results" in public_task_payload:
            raise SystemExit(f"state leaked raw external LLM output: {public_task_payload}")
        for row in MockOpenAIHandler.requests:
            prompt = str(row.get("prompt") or "")
            if prompt and prompt in public_task_payload:
                raise SystemExit(f"state leaked raw external LLM prompt: {public_task_payload}")
        if "<redacted>" not in public_task_payload or "prompt_hash" not in public_task_payload:
            raise SystemExit(f"state missing redacted prompt metadata: {public_task_payload}")
        profile = (state.get("miner_profiles") or {}).get("external-llm-http-smoke") or {}
        capabilities = profile.get("last_capabilities") or {}
        runtime_capability = capabilities.get("external_llm_runtime") or {}
        if runtime_capability.get("adapter_kind") != "http_openai_chat":
            raise SystemExit(f"missing HTTP runtime capability: {profile}")
        if "runtime_url" in json.dumps(runtime_capability, sort_keys=True):
            raise SystemExit(f"runtime URL leaked in capability: {runtime_capability}")
        if args.runtime_api_key in json.dumps(state, sort_keys=True):
            raise SystemExit("runtime API key leaked in state")
        ledger = admin_results(args, status="accepted", workload_type="external_llm_infer")
        results = ledger.get("results") or []
        if len(results) != 1:
            raise SystemExit(f"expected one external_llm_infer ledger row: {ledger}")
        row = results[0]
        if row.get("model_updated") or row.get("model_bundle_updated"):
            raise SystemExit(f"external LLM ledger row must be read-only: {row}")
        if "output_text" in json.dumps(row, sort_keys=True):
            raise SystemExit(f"ledger leaked raw output_text: {row}")
        print(json.dumps({
            "accepted_results": state["accepted_results"],
            "adapter_kind": validation.get("adapter_kind"),
            "completion_count": validation.get("completion_count"),
            "ledger_rows": len(results),
            "miner_summary": miner_summary,
            "model_id": validation.get("model_id"),
            "request_count": validation.get("request_count"),
            "runtime_requests": len(MockOpenAIHandler.requests),
            "task_id": task["task_id"],
        }, sort_keys=True))
    finally:
        stop_process(coordinator)
        if runtime is not None:
            runtime.shutdown()
            runtime.server_close()
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
