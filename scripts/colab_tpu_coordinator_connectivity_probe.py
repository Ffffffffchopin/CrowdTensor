#!/usr/bin/env python3
"""Probe Colab TPU runtime connectivity to a local/public Coordinator URL."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import colab_cli_runtime  # noqa: E402


SCHEMA = "colab_tpu_coordinator_connectivity_probe_v1"


def sha256_short(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def load_session(config: Path, session_name: str) -> dict[str, Any]:
    data = json.loads(config.read_text())
    session = data.get(session_name)
    if not isinstance(session, dict):
        raise SystemExit(f"Session {session_name!r} not found")
    return session


def parse_report(outputs: list[dict[str, Any]], marker: str) -> dict[str, Any]:
    text = "\n".join(str(output.get("text") or "") for output in outputs if isinstance(output, dict))
    for line in text.splitlines()[::-1]:
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {"ok": False, "error": "json_decode_failed"}
    return {"ok": False, "error": "marker_missing"}


class ProbeServer:
    def __init__(self, port: int, token: str) -> None:
        token_value = token

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                if self.path.split("?", 1)[0] != "/ping":
                    self.send_response(404)
                    self.end_headers()
                    return
                if self.headers.get("X-CrowdTensor-Probe-Token") != token_value:
                    self.send_response(403)
                    self.end_headers()
                    return
                body = json.dumps({"ok": True, "schema": "ct_ping_v1"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("0.0.0.0", int(port)), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def run_colab(session: dict[str, Any], target_url: str, token: str, timeout: float) -> dict[str, Any]:
    ColabRuntime = colab_cli_runtime.load_colab_runtime_class()

    marker = "CT_COLAB_COORDINATOR_CONNECTIVITY"
    runtime = ColabRuntime(session["url"], session["token"], kernel_id=session.get("kernel_id"), session_id=session.get("session_id"))
    code = f"""
import json, time, urllib.request
marker = {marker!r}
target_url = {target_url!r}
token = {token!r}
started = time.time()
out = {{"schema": "colab_tpu_coordinator_connectivity_runtime_v1", "ok": False}}
try:
    req = urllib.request.Request(target_url.rstrip("/") + "/ping", headers={{"X-CrowdTensor-Probe-Token": token}})
    with urllib.request.urlopen(req, timeout={float(timeout)!r}) as resp:
        body = resp.read(200).decode("utf-8", "replace")
        out.update({{"ok": int(resp.status) == 200, "status": int(resp.status), "body_hash": "sha256:" + __import__("hashlib").sha256(body.encode()).hexdigest()}})
except Exception as exc:
    out.update({{"ok": False, "error_type": type(exc).__name__, "error_hash": "sha256:" + __import__("hashlib").sha256(str(exc).encode()).hexdigest()}})
out["elapsed_seconds"] = round(time.time() - started, 3)
print(marker + " " + json.dumps(out, sort_keys=True))
"""
    try:
        outputs = runtime.execute_code(code, timeout=timeout + 30)
        return parse_report(outputs, marker)
    finally:
        try:
            runtime.stop()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="ct-colab-tpu-v5e1")
    parser.add_argument("--config", default=os.path.expanduser("~/.config/colab-cli/sessions.json"))
    parser.add_argument("--public-host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    token = hashlib.sha256(f"{time.time()}:{os.getpid()}".encode()).hexdigest()
    server = ProbeServer(args.port, token)
    started = time.time()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": False,
        "session_name": args.session,
        "public_artifact_safe": True,
        "runtime_proxy_token_public": False,
        "runtime_proxy_url_public": False,
        "probe_token_public": False,
        "public_host_hash": sha256_short(args.public_host),
        "port": int(args.port),
    }
    try:
        session = load_session(Path(args.config), args.session)
        server.start()
        target_url = f"http://{args.public_host}:{int(args.port)}"
        runtime_report = run_colab(session, target_url, token, args.timeout)
        report.update(
            {
                "ok": runtime_report.get("ok") is True,
                "colab_to_coordinator_connectivity_verified": runtime_report.get("ok") is True,
                "runtime_report": runtime_report,
                "endpoint_hash": sha256_short(str(session.get("endpoint") or "")),
                "runtime_proxy_host_hash": sha256_short(urlparse(str(session.get("url") or "")).netloc),
            }
        )
    except Exception as exc:  # noqa: BLE001
        report.update({"ok": False, "error_type": type(exc).__name__, "error_digest": "sha256:" + hashlib.sha256(str(exc).encode()).hexdigest()})
    finally:
        server.stop()
        report["duration_seconds"] = round(time.time() - started, 3)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "colab_tpu_coordinator_connectivity_probe.json"
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(output_path)
        if not report.get("ok"):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
