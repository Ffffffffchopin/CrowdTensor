"""Loopback-only status and graceful control surface for a native Cell."""

from __future__ import annotations

import html
import json
import signal
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator


class VolunteerAgentStatusServer:
    def __init__(self, cell: Any, *, port: int = 8765) -> None:
        self.cell = cell
        self.stop_event = threading.Event()
        self._server = self._create_server(int(port))
        self.port = int(self._server.server_address[1])
        self.endpoint = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="crowdtensor-agent-status",
            daemon=True,
        )

    def _create_server(self, requested_port: int) -> ThreadingHTTPServer:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "CrowdTensorAgent/1"

            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def _headers(self, status: int, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'",
                )
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                status = owner.cell.local_status()
                if self.path == "/status.json":
                    self._headers(200, "application/json; charset=utf-8")
                    self.wfile.write(
                        (json.dumps(status, sort_keys=True) + "\n").encode("utf-8")
                    )
                    return
                if self.path != "/":
                    self._headers(404, "text/plain; charset=utf-8")
                    self.wfile.write(b"not found\n")
                    return
                state = html.escape(str(status.get("state") or status.get("last_state") or "ready"))
                completed = html.escape(
                    str(status.get("completed_in_run") or status.get("completed_work_units") or 0)
                )
                page = f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><meta http-equiv=\"refresh\" content=\"2\"><title>CrowdTensor Agent</title><style>
body{{margin:0;background:#f7faf7;color:#13251d;font:15px system-ui,sans-serif;letter-spacing:0}}main{{width:min(680px,calc(100% - 32px));margin:56px auto}}header{{display:flex;align-items:center;gap:12px;margin-bottom:28px}}b.mark{{display:grid;place-items:center;width:34px;height:34px;border-radius:4px;background:#176b4a;color:white;font-size:12px}}section{{border:1px solid #d7ddd8;border-radius:8px;background:white;padding:26px}}.state{{font-size:28px;font-weight:750}}dl{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:24px 0}}dt{{color:#607067;font-size:12px}}dd{{margin:3px 0 0;font-weight:700}}form{{display:inline-block;margin-right:8px}}button{{min-height:40px;padding:0 15px;border:1px solid #176b4a;border-radius:5px;background:white;color:#176b4a;font-weight:700}}button.stop{{background:#176b4a;color:white}}small{{display:block;margin-top:20px;color:#607067}}@media(max-width:480px){{dl{{grid-template-columns:1fr}}}}
</style></head><body><main><header><b class=\"mark\">CT</b><strong>CrowdTensor native Agent</strong></header><section><div class=\"state\">{state}</div><dl><div><dt>Completed this run</dt><dd>{completed}</dd></div><div><dt>Binding</dt><dd>127.0.0.1 only</dd></div></dl><form method=\"post\" action=\"/pause\"><button>Pause</button></form><form method=\"post\" action=\"/resume\"><button>Resume</button></form><form method=\"post\" action=\"/stop\"><button class=\"stop\">Stop safely</button></form><small>No credential, dataset, tensor, or private path is served here.</small></section></main></body></html>"""
                self._headers(200, "text/html; charset=utf-8")
                self.wfile.write(page.encode("utf-8"))

            def do_POST(self) -> None:  # noqa: N802
                if self.path == "/pause":
                    owner.cell.pause()
                elif self.path == "/resume":
                    owner.cell.resume()
                elif self.path == "/stop":
                    owner.stop_event.set()
                else:
                    self._headers(404, "text/plain; charset=utf-8")
                    self.wfile.write(b"not found\n")
                    return
                self.send_response(303)
                self.send_header("Location", "/")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()

        try:
            return ThreadingHTTPServer(("127.0.0.1", requested_port), Handler)
        except OSError:
            return ThreadingHTTPServer(("127.0.0.1", 0), Handler)

    def __enter__(self) -> "VolunteerAgentStatusServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)


@contextmanager
def graceful_agent_signals(stop_event: threading.Event) -> Iterator[None]:
    """Convert SIGINT/SIGTERM into a stop-after-current-work request."""

    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous: dict[int, Any] = {}

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

