#!/usr/bin/env python3
"""Playwright smoke test for the CrowdTensorD browser WebRTC tunnel."""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import shutil
import socketserver
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return


class ReusableThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_server(port: int) -> ReusableThreadingHTTPServer:
    handler = functools.partial(QuietHandler, directory=str(WEB_DIR))
    server = ReusableThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, name="webrtc-smoke-http", daemon=True)
    thread.start()
    return server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the browser WebRTC 5MB tensor tunnel smoke test.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bytes", type=int, default=5 * 1024 * 1024)
    parser.add_argument("--room", default="")
    parser.add_argument("--compute", choices=("worker", "off"), default="worker")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--browser", default=shutil.which("google-chrome") or shutil.which("chromium") or "")
    parser.add_argument("--headful", action="store_true")
    return parser.parse_args()


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

    if not WEB_DIR.exists():
        raise SystemExit(f"web directory not found: {WEB_DIR}")

    room = args.room or f"smoke-{int(time.time() * 1000)}"
    base_url = (
        f"http://{args.host}:{args.port}/index.html?"
        f"room={room}&bytes={args.bytes}&compute={args.compute}"
    )
    server = start_server(args.port)
    browser = None

    try:
        with sync_playwright() as playwright:
            launch_options = {
                "headless": not args.headful,
                "args": ["--no-sandbox", "--disable-dev-shm-usage"],
            }
            if args.browser:
                launch_options["executable_path"] = args.browser
            browser = playwright.chromium.launch(**launch_options)
            context = browser.new_context()
            receiver = context.new_page()
            sender = context.new_page()

            receiver.goto(f"{base_url}&role=receiver", wait_until="domcontentloaded")
            sender.goto(f"{base_url}&role=sender", wait_until="domcontentloaded")

            try:
                wait_expression = (
                    "() => window.__crowdTensorWebrtcStatus "
                    "&& window.__crowdTensorWebrtcStatus.verified === true"
                )
                if args.compute == "worker":
                    wait_expression = (
                        "() => window.__crowdTensorWebrtcStatus "
                        "&& window.__crowdTensorWebrtcStatus.verified === true "
                        "&& window.__crowdTensorWebrtcStatus.compute "
                        "&& window.__crowdTensorWebrtcStatus.compute.verified === true"
                    )
                receiver.wait_for_function(
                    wait_expression,
                    timeout=args.timeout * 1000,
                )
            except PlaywrightTimeoutError as exc:
                receiver_status = receiver.evaluate("() => window.__crowdTensorWebrtcStatus || null")
                sender_status = sender.evaluate("() => window.__crowdTensorWebrtcStatus || null")
                raise SystemExit(
                    "WebRTC smoke timed out\n"
                    f"receiver={json.dumps(receiver_status, sort_keys=True)}\n"
                    f"sender={json.dumps(sender_status, sort_keys=True)}"
                ) from exc

            receiver_status = receiver.evaluate("() => window.__crowdTensorWebrtcStatus")
            sender_status = sender.evaluate("() => window.__crowdTensorWebrtcStatus")

            if receiver_status["bytesReceived"] != args.bytes:
                raise SystemExit(f"expected {args.bytes} bytes, got {receiver_status['bytesReceived']}")
            if not receiver_status["checksumMatch"]:
                raise SystemExit(f"checksum mismatch: {receiver_status}")
            if receiver_status["elapsedMs"] <= 0 or receiver_status["mbps"] <= 0:
                raise SystemExit(f"invalid transfer metrics: {receiver_status}")
            compute_status = receiver_status.get("compute", {})
            if args.compute == "worker":
                if not compute_status.get("verified"):
                    raise SystemExit(f"compute probe failed: {receiver_status}")
                if compute_status.get("elapsedMs", 0) <= 0 or compute_status.get("ops", 0) <= 0:
                    raise SystemExit(f"invalid compute metrics: {receiver_status}")
                if not compute_status.get("hash"):
                    raise SystemExit(f"missing compute hash: {receiver_status}")

            print(json.dumps({
                "room": room,
                "bytes": receiver_status["bytesReceived"],
                "chunks": receiver_status["chunksReceived"],
                "elapsed_ms": round(receiver_status["elapsedMs"], 3),
                "mbps": round(receiver_status["mbps"], 3),
                "checksum": receiver_status["checksum"],
                "compute_backend": compute_status.get("backend", "off"),
                "compute_elapsed_ms": round(compute_status.get("elapsedMs", 0), 3),
                "compute_gops": round(compute_status.get("gops", 0), 6),
                "compute_hash": compute_status.get("hash", ""),
                "compute_verified": bool(compute_status.get("verified", False)),
                "sender_chunks": sender_status["chunksSent"],
                "verified": receiver_status["verified"],
            }, sort_keys=True))
            browser.close()
            browser = None
    finally:
        if browser is not None:
            browser.close()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
