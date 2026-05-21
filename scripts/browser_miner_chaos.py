#!/usr/bin/env python3
"""Playwright chaos validation for CrowdTensorD browser Miners."""

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
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from auth_headers import activate_miner_token, activate_observer_token, add_miner_token_arg, add_observer_token_arg, coordinator_env, json_headers, observer_headers, resolve_miner_token  # noqa: E402


class BrowserChaosError(RuntimeError):
    pass


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return


class ReusableThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_web_server(host: str, port: int) -> ReusableThreadingHTTPServer:
    handler = functools.partial(QuietHandler, directory=str(WEB_DIR))
    server = ReusableThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, name="browser-miner-chaos-web", daemon=True)
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


def get_state(base_url: str) -> dict:
    return request_json("GET", base_url, "/state")


def wait_health(base_url: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise BrowserChaosError(f"coordinator exited early with code {proc.returncode}")
        try:
            health = request_json("GET", base_url, "/health", timeout=2.0)
            if health.get("ok") is True:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.1)
    raise BrowserChaosError(f"coordinator did not become healthy: {last_error}")


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
        str(max(1, args.parallel_browsers)),
        "--reaper-interval",
        str(args.reaper_interval),
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


def miner_url(args: argparse.Namespace, miner_id: str, *, hold_ms: int = 0, result_mode: str = "normal") -> str:
    query_payload = {
        "coordinator": args.coordinator_url,
        "miner_id": miner_id,
        "autorun": "1",
        "hold_ms": str(hold_ms),
        "result_mode": result_mode,
    }
    token = resolve_miner_token()
    if token:
        query_payload["miner_token"] = token
    query = urlencode(query_payload)
    return f"{args.web_url}/browser_miner.html?{query}"


def page_status(page) -> dict:
    return page.evaluate("() => window.__crowdTensorBrowserMinerStatus || null")


def wait_status(page, expression: str, timeout: float) -> dict:
    try:
        page.wait_for_function(expression, timeout=timeout * 1000)
    except Exception as exc:
        raise BrowserChaosError(f"page wait failed: {json.dumps(page_status(page), sort_keys=True)}") from exc
    return page_status(page)


def wait_state(args: argparse.Namespace, predicate, description: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last_state = None
    while time.monotonic() < deadline:
        state = get_state(args.coordinator_url)
        last_state = state
        if predicate(state):
            return state
        time.sleep(0.1)
    raise BrowserChaosError(f"timed out waiting for {description}: {json.dumps(last_state, sort_keys=True)}")


def task_by_id(state: dict, task_id: str) -> dict | None:
    for task in state.get("tasks", []):
        if task.get("task_id") == task_id:
            return task
    return None


def run_slow_heartbeat_case(args: argparse.Namespace, context) -> dict:
    page = context.new_page()
    page.goto(miner_url(args, "browser-slow", hold_ms=args.hold_ms), wait_until="domcontentloaded")
    status = wait_status(
        page,
        "() => window.__crowdTensorBrowserMinerStatus "
        "&& window.__crowdTensorBrowserMinerStatus.accepted === true",
        args.timeout,
    )
    page.close()
    if status.get("heartbeats", 0) < 1:
        raise BrowserChaosError(f"expected slow browser Miner to heartbeat: {status}")
    print(f"ok slow browser miner heartbeats={status['heartbeats']} accepted task={status['taskId']}")
    return status


def run_drop_requeue_case(args: argparse.Namespace, context) -> dict:
    page = context.new_page()
    page.goto(miner_url(args, "browser-drop", hold_ms=args.drop_hold_ms), wait_until="domcontentloaded")
    dropped = wait_status(
        page,
        "() => window.__crowdTensorBrowserMinerStatus "
        "&& window.__crowdTensorBrowserMinerStatus.claimed === true",
        args.timeout,
    )
    task_id = dropped["taskId"]
    attempt = int(dropped["attempt"])
    page.close()

    wait_state(
        args,
        lambda state: (
            (task := task_by_id(state, task_id)) is not None
            and task.get("status") == "queued"
            and int(task.get("attempt", 0)) == attempt
        ),
        f"dropped task {task_id} to be requeued",
        args.timeout,
    )

    rescue = context.new_page()
    rescue.goto(miner_url(args, "browser-rescue"), wait_until="domcontentloaded")
    rescued = wait_status(
        rescue,
        "() => window.__crowdTensorBrowserMinerStatus "
        "&& window.__crowdTensorBrowserMinerStatus.accepted === true",
        args.timeout,
    )
    rescue.close()
    if rescued["taskId"] != task_id or int(rescued["attempt"]) <= attempt:
        raise BrowserChaosError(f"rescue did not reclaim dropped task: dropped={dropped} rescued={rescued}")
    print(f"ok dropped browser task requeued task={task_id} rescue_attempt={rescued['attempt']}")
    return {
        "dropped_task_id": task_id,
        "dropped_attempt": attempt,
        "rescue_attempt": int(rescued["attempt"]),
    }


def run_parallel_case(args: argparse.Namespace, context) -> list[dict]:
    before = get_state(args.coordinator_url)
    before_accepted = int(before.get("accepted_results", 0))
    pages = []
    try:
        for index in range(args.parallel_browsers):
            page = context.new_page()
            page.goto(
                miner_url(args, f"browser-parallel-{index}", hold_ms=args.hold_ms),
                wait_until="domcontentloaded",
            )
            pages.append(page)
        statuses = [
            wait_status(
                page,
                "() => window.__crowdTensorBrowserMinerStatus "
                "&& window.__crowdTensorBrowserMinerStatus.accepted === true",
                args.timeout,
            )
            for page in pages
        ]
    finally:
        for page in pages:
            if not page.is_closed():
                page.close()

    state = wait_state(
        args,
        lambda item: int(item.get("accepted_results", 0)) >= before_accepted + args.parallel_browsers,
        "parallel browser accepted results",
        args.timeout,
    )
    if int(state.get("max_staleness", 0)) <= 0:
        raise BrowserChaosError(f"expected browser parallel staleness > 0: {state}")
    print(
        f"ok parallel browser miners={args.parallel_browsers} "
        f"accepted_results={state['accepted_results']} max_staleness={state['max_staleness']}"
    )
    return statuses


def run_malicious_case(args: argparse.Namespace, context) -> dict:
    before = get_state(args.coordinator_url)
    before_rejected = int(before.get("rejected_results", 0))
    page = context.new_page()
    page.goto(
        miner_url(args, "browser-malicious", result_mode="large_delta"),
        wait_until="domcontentloaded",
    )
    status = wait_status(
        page,
        "() => window.__crowdTensorBrowserMinerStatus "
        "&& window.__crowdTensorBrowserMinerStatus.rejected === true",
        args.timeout,
    )
    page.close()
    state = wait_state(
        args,
        lambda item: int(item.get("rejected_results", 0)) >= before_rejected + 1,
        "browser malicious rejection",
        args.timeout,
    )
    score = state.get("miner_scores", {}).get("browser-malicious", {})
    if score.get("rejected") != 1:
        raise BrowserChaosError(f"browser malicious score missing rejection: {score}")
    print("ok malicious browser miner rejected and scored")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run browser Miner chaos validation.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--coordinator-port", type=int, default=8897)
    parser.add_argument("--web-port", type=int, default=8768)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--lease-seconds", type=float, default=1.2)
    parser.add_argument("--inner-steps", type=int, default=500)
    parser.add_argument("--parallel-browsers", type=int, default=4)
    parser.add_argument("--hold-ms", type=int, default=2500)
    parser.add_argument("--drop-hold-ms", type=int, default=5000)
    parser.add_argument("--reaper-interval", type=float, default=0.1)
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
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_browser_chaos_")
        state_dir = Path(temp_dir.name)

    web_server = None
    coordinator = None
    browser = None
    try:
        web_server = start_web_server(args.host, args.web_port)
        coordinator = start_coordinator(args, state_dir)
        with sync_playwright() as playwright:
            launch_options = {
                "headless": not args.headful,
                "args": ["--no-sandbox", "--disable-dev-shm-usage"],
            }
            if args.browser:
                launch_options["executable_path"] = args.browser
            browser = playwright.chromium.launch(**launch_options)
            context = browser.new_context()
            try:
                slow = run_slow_heartbeat_case(args, context)
                dropped = run_drop_requeue_case(args, context)
                parallel = run_parallel_case(args, context)
                malicious = run_malicious_case(args, context)
                final_state = get_state(args.coordinator_url)
                profiles = final_state.get("miner_profiles", {})
                slow_profile = profiles.get("browser-slow", {})
                malicious_profile = profiles.get("browser-malicious", {})
                if slow_profile.get("runtime") != "browser" or slow_profile.get("backend") != "js-worker":
                    raise BrowserChaosError(f"missing browser-slow capability profile: {profiles}")
                if malicious_profile.get("rejected") != 1:
                    raise BrowserChaosError(f"missing browser-malicious rejected profile: {profiles}")

                print(json.dumps({
                    "accepted_results": final_state["accepted_results"],
                    "browser_backend": slow_profile.get("backend"),
                    "browser_runtime": slow_profile.get("runtime"),
                    "dropped_task_id": dropped["dropped_task_id"],
                    "max_staleness": final_state["max_staleness"],
                    "parallel_browsers": args.parallel_browsers,
                    "rejected_results": final_state["rejected_results"],
                    "rescue_attempt": dropped["rescue_attempt"],
                    "slow_heartbeats": slow["heartbeats"],
                    "malicious_rejected": malicious["rejected"],
                    "parallel_tasks": [item["taskId"] for item in parallel],
                }, sort_keys=True))
            finally:
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
