#!/usr/bin/env python3
"""Playwright smoke test for the CrowdTensorD browser_probe workload."""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from auth_headers import activate_miner_token, activate_observer_token, add_miner_token_arg, add_observer_token_arg, coordinator_env, json_headers, observer_headers, resolve_miner_token  # noqa: E402


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return


class ReusableThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_web_server(host: str, port: int) -> ReusableThreadingHTTPServer:
    handler = functools.partial(QuietHandler, directory=str(WEB_DIR))
    server = ReusableThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, name="browser-probe-web", daemon=True)
    thread.start()
    return server


def request_json(method: str, base_url: str, path: str, payload: dict | None = None, *, timeout: float = 5.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=observer_headers() if method == "GET" else json_headers(),
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
    origin = f"http://{args.host}:{args.web_port}"
    command = [
        sys.executable,
        str(ROOT / "coordinator.py"),
        "--host",
        args.host,
        "--port",
        str(args.coordinator_port),
        "--state-dir",
        str(state_dir),
        "--lease-seconds",
        str(args.lease_seconds),
        "--inner-steps",
        str(args.inner_steps),
        "--backlog",
        "0",
        "--task-lane",
        "browser:js-worker:1:browser_probe",
        "--cors-origin",
        origin,
    ]
    env = coordinator_env()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    wait_health(args.coordinator_url, proc, args.startup_timeout)
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
    parser = argparse.ArgumentParser(description="Run the browser_probe workload smoke test.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--coordinator-port", type=int, default=8896)
    parser.add_argument("--web-port", type=int, default=8771)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--inner-steps", type=int, default=500)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--browser", default=shutil.which("google-chrome") or shutil.which("chromium") or "")
    parser.add_argument("--headful", action="store_true")
    add_miner_token_arg(parser)
    add_observer_token_arg(parser)
    args = parser.parse_args()
    activate_miner_token(args)
    activate_observer_token(args)
    args.coordinator_url = f"http://{args.host}:{args.coordinator_port}"
    args.web_url = f"http://{args.host}:{args.web_port}"
    return args


def main() -> None:
    args = parse_args()
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Playwright Python is not installed. Run: pip install playwright\n"
            "If browsers are not installed, either run playwright install chromium "
            "or pass --browser /path/to/google-chrome."
        ) from exc

    temp_dir = None
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_browser_probe_")
        state_dir = Path(temp_dir.name)

    web_server = None
    coordinator = None
    browser = None
    try:
        web_server = start_web_server(args.host, args.web_port)
        coordinator = start_coordinator(args, state_dir)
        page_url = (
            f"{args.web_url}/browser_miner.html?"
            f"coordinator={quote(args.coordinator_url, safe='')}"
            "&miner_id=browser-probe-smoke"
            "&autorun=1"
        )
        token = resolve_miner_token()
        if token:
            page_url += f"&miner_token={quote(token, safe='')}"

        with sync_playwright() as playwright:
            launch_options = {
                "headless": not args.headful,
                "args": ["--no-sandbox", "--disable-dev-shm-usage"],
            }
            if args.browser:
                launch_options["executable_path"] = args.browser
            browser = playwright.chromium.launch(**launch_options)
            page = browser.new_page()
            page.goto(page_url, wait_until="domcontentloaded")
            try:
                page.wait_for_function(
                    "() => window.__crowdTensorBrowserMinerStatus "
                    "&& window.__crowdTensorBrowserMinerStatus.accepted === true "
                    "&& window.__crowdTensorBrowserMinerStatus.workloadType === 'browser_probe'",
                    timeout=args.timeout * 1000,
                )
            except PlaywrightTimeoutError as exc:
                status = page.evaluate("() => window.__crowdTensorBrowserMinerStatus || null")
                raise SystemExit(f"browser_probe smoke timed out\nstatus={json.dumps(status, sort_keys=True)}") from exc

            status = page.evaluate("() => window.__crowdTensorBrowserMinerStatus")
            state = request_json("GET", args.coordinator_url, "/state")
            if state.get("accepted_results") != 1 or state.get("model_updates") != 0:
                raise SystemExit(f"unexpected coordinator state: {json.dumps(state, sort_keys=True)}")
            if not status.get("probeHash") or float(status.get("probeGops", 0)) <= 0:
                raise SystemExit(f"invalid probe metrics: {status}")

            print(json.dumps({
                "accepted": bool(status["accepted"]),
                "task_id": status["taskId"],
                "workload_type": status["workloadType"],
                "probe_hash": status["probeHash"],
                "probe_gops": status["probeGops"],
                "worker_elapsed_ms": round(status["workerElapsedMs"], 3),
                "state_accepted_results": state["accepted_results"],
                "state_model_updates": state["model_updates"],
            }, sort_keys=True))
            browser.close()
            browser = None
    finally:
        if browser is not None:
            browser.close()
        stop_process(coordinator)
        if web_server is not None:
            web_server.shutdown()
            web_server.server_close()
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
