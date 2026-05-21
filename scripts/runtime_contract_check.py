#!/usr/bin/env python3
"""Cross-runtime parity check for the CrowdTensorD DiLoCo math contract."""

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
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from crowdtensor.diloco import run_inner_loop  # noqa: E402
from auth_headers import activate_miner_token, activate_observer_token, add_miner_token_arg, add_observer_token_arg, coordinator_env, json_headers  # noqa: E402


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return


class ReusableThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_web_server(host: str, port: int) -> ReusableThreadingHTTPServer:
    handler = functools.partial(QuietHandler, directory=str(WEB_DIR))
    server = ReusableThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, name="runtime-contract-web", daemon=True)
    thread.start()
    return server


def request_json(method: str, base_url: str, path: str, payload: dict | None = None, *, timeout: float = 5.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=json_headers(),
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
        "10",
        "--inner-steps",
        str(args.inner_steps),
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


def run_browser_worker(page, claim: dict, miner_id: str) -> dict:
    return page.evaluate(
        """async ({ claim, minerId }) => {
          return await new Promise((resolve, reject) => {
            const worker = new Worker(new URL("./diloco_worker.js", window.location.href));
            worker.onmessage = (event) => {
              const message = event.data || {};
              worker.terminate();
              if (message.type === "training-result") {
                resolve(message);
              } else {
                reject(new Error(message.error || "worker training failed"));
              }
            };
            worker.onerror = (event) => {
              worker.terminate();
              reject(new Error(event.message || "worker crashed"));
            };
            worker.postMessage({ type: "train", claim, minerId, holdMs: 0 });
          });
        }""",
        {"claim": claim, "minerId": miner_id},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Python/browser runtime math contract parity.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--coordinator-port", type=int, default=8896)
    parser.add_argument("--web-port", type=int, default=8769)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--inner-steps", type=int, default=500)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--tolerance", type=float, default=1e-9)
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
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_contract_")
        state_dir = Path(temp_dir.name)

    web_server = None
    coordinator = None
    browser = None
    try:
        web_server = start_web_server(args.host, args.web_port)
        coordinator = start_coordinator(args, state_dir)
        miner_id = "contract-probe"
        claim = request_json("POST", args.coordinator_url, "/tasks/claim", {"miner_id": miner_id})
        python_result = run_inner_loop(
            claim["weights"],
            task_id=claim["task_id"],
            miner_id=miner_id,
            model_version=int(claim["model_version"]),
            inner_steps=int(claim["inner_steps"]),
            training_spec=claim["training_spec"],
        )

        with sync_playwright() as playwright:
            launch_options = {
                "headless": not args.headful,
                "args": ["--no-sandbox", "--disable-dev-shm-usage"],
            }
            if args.browser:
                launch_options["executable_path"] = args.browser
            browser = playwright.chromium.launch(**launch_options)
            page = browser.new_page()
            page.goto(f"{args.web_url}/browser_miner.html", wait_until="domcontentloaded")
            browser_result = run_browser_worker(page, claim, miner_id)
            browser.close()
            browser = None

        diffs = [
            abs(float(left) - float(right))
            for left, right in zip(python_result["local_delta"], browser_result["local_delta"])
        ]
        max_abs_diff = max(diffs) if diffs else 0.0
        if max_abs_diff > args.tolerance:
            raise SystemExit(
                f"runtime contract mismatch max_abs_delta_diff={max_abs_diff} "
                f"python={python_result['local_delta']} browser={browser_result['local_delta']}"
            )
        if python_result["sample_offset"] != browser_result["metrics"]["sample_offset"]:
            raise SystemExit(f"sample_offset mismatch: python={python_result} browser={browser_result}")
        if python_result["local_delta_scale"] != browser_result["metrics"]["local_delta_scale"]:
            raise SystemExit(f"local_delta_scale mismatch: python={python_result} browser={browser_result}")

        print(json.dumps({
            "task_id": claim["task_id"],
            "model_version": claim["model_version"],
            "sample_offset": python_result["sample_offset"],
            "local_delta_scale": python_result["local_delta_scale"],
            "max_abs_delta_diff": max_abs_diff,
            "python_loss_end": python_result["inner_loss_end"],
            "browser_loss_end": browser_result["metrics"]["inner_loss_end"],
        }, sort_keys=True))
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
