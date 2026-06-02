from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import time
import unittest
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Libp2pKadDaemonTests(unittest.TestCase):
    def test_health_and_provider_catalog_do_not_500_or_422(self) -> None:
        if shutil.which("node") is None:
            self.skipTest("node runtime is not installed")
        if not (ROOT / "node_modules" / "libp2p").exists():
            self.skipTest("node_modules/libp2p is not installed")

        port = free_port()
        command = [
            "node",
            "--import",
            (ROOT / "scripts" / "libp2p_node20_polyfill.mjs").as_uri(),
            str(ROOT / "scripts" / "libp2p_kad_daemon.mjs"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--swarm-id",
            "libp2p-kad-daemon-test",
            "--record-secret",
            "test-secret",
            "--require-signed",
            "--discovery-backend",
            "libp2p-kad",
            "--libp2p-host",
            "127.0.0.1",
            "--libp2p-port",
            "0",
        ]
        proc = subprocess.Popen(
            command,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 20
            health: dict[str, object] = {}
            last_error = ""
            while time.monotonic() <= deadline:
                if proc.poll() is not None:
                    stdout = proc.communicate(timeout=1)[0]
                    self.fail(f"libp2p daemon exited early rc={proc.returncode}: {stdout[-1000:]}")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/real-p2p/health", timeout=2) as response:
                        health = json.loads(response.read().decode("utf-8"))
                    if health.get("ok") is True:
                        break
                except Exception as exc:  # pragma: no cover - diagnostic path
                    last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.2)
            else:
                self.fail(f"libp2p daemon did not become healthy: {last_error}")

            with urlopen(f"http://127.0.0.1:{port}/real-p2p/providers", timeout=5) as response:
                catalog = json.loads(response.read().decode("utf-8"))

            self.assertEqual(health.get("schema"), "real_p2p_health_v1")
            self.assertTrue(catalog.get("ok"), catalog)
            self.assertEqual(catalog.get("schema"), "real_p2p_provider_catalog_v1")
            self.assertGreaterEqual(int(catalog.get("provider_count") or 0), 1)
            self.assertIn("peer_scoring", catalog)
        finally:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
