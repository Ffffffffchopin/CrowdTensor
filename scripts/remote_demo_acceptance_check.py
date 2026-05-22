#!/usr/bin/env python3
"""CI-safe check for the remote demo acceptance pack."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from create_miner_invite import create_invite  # noqa: E402


MINER_ID = "remote-acceptance-miner"
OBSERVER_TOKEN = "remote-acceptance-observer"
ADMIN_TOKEN = "remote-acceptance-admin"
INVITE_TOKEN = "remote-acceptance-miner-token"


def request_json(base_url: str, path: str, *, timeout: float = 5.0) -> dict:
    request = Request(f"{base_url.rstrip('/')}{path}", method="GET")
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
            if request_json(base_url, "/health", timeout=2.0).get("ok") is True:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"coordinator did not become healthy: {last_error}")


def start_coordinator(args: argparse.Namespace, state_dir: Path, registry_path: Path) -> subprocess.Popen:
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
        "10",
        "--inner-steps",
        str(args.request_count),
        "--backlog",
        "0",
        "--task-lane",
        "python-cli:cpu:1:model_bundle_infer",
        "--miner-token-registry",
        str(registry_path),
        "--observer-token",
        OBSERVER_TOKEN,
        "--admin-token",
        ADMIN_TOKEN,
    ]
    env = dict(os.environ)
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


def run_invited_miner(args: argparse.Namespace, invite: dict) -> None:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["CROWDTENSOR_MINER_TOKEN"] = invite["env"]["CROWDTENSOR_MINER_TOKEN"]
    command = [
        sys.executable,
        str(ROOT / "miner_cli.py"),
        "--coordinator",
        args.base_url,
        "--miner-id",
        invite["miner_id"],
        "--once",
        "--compute-seconds",
        "0.2",
        "--heartbeat-interval",
        "0.1",
        "--idle-sleep",
        "0.2",
    ]
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=args.miner_timeout)
    if completed.returncode != 0:
        raise RuntimeError(
            "invited miner failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def run_acceptance(args: argparse.Namespace, output_dir: Path) -> dict:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "remote_demo_acceptance_pack.py"),
        "--coordinator-url",
        args.base_url,
        "--miner-id",
        MINER_ID,
        "--observer-token",
        OBSERVER_TOKEN,
        "--admin-token",
        ADMIN_TOKEN,
        "--request-count",
        str(args.request_count),
        "--timeout-seconds",
        "20",
        "--poll-interval",
        "0.2",
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=args.acceptance_timeout)
    if completed.returncode != 0:
        raise RuntimeError(
            "remote_demo_acceptance_pack.py failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("remote_demo_acceptance_pack.py emitted no JSON")
    return json.loads(lines[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run remote demo acceptance pack check.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8913)
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--miner-timeout", type=float, default=30.0)
    parser.add_argument("--acceptance-timeout", type=float, default=90.0)
    args = parser.parse_args()
    args.base_url = f"http://{args.host}:{args.port}"
    return args


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="crowdtensor_remote_acceptance_") as temp:
        state_dir = Path(temp) / "state"
        output_dir = Path(temp) / "acceptance"
        registry_path = state_dir / "miner_registry.json"
        state_dir.mkdir(parents=True, exist_ok=True)
        coordinator = None
        try:
            invite = create_invite(
                registry_path=registry_path,
                miner_id=MINER_ID,
                coordinator_url=args.base_url,
                label="remote acceptance check",
                token=INVITE_TOKEN,
                replace=True,
            )
            coordinator = start_coordinator(args, state_dir, registry_path)
            run_invited_miner(args, invite)
            report = run_acceptance(args, output_dir)
            if report.get("schema") != "remote_demo_acceptance_v1" or not report.get("ok"):
                raise SystemExit(f"unexpected acceptance report: {json.dumps(report, sort_keys=True)}")
            evidence_path = output_dir / "remote_compute_evidence.json"
            support_path = output_dir / "support_bundle.json"
            if not evidence_path.is_file() or not support_path.is_file():
                raise SystemExit("acceptance pack did not write evidence and support bundle artifacts")
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            support = json.loads(support_path.read_text(encoding="utf-8"))
            if evidence.get("schema") != "remote_compute_evidence_v1" or not evidence.get("ok"):
                raise SystemExit(f"unexpected evidence artifact: {json.dumps(evidence, sort_keys=True)}")
            if not (support.get("online") or {}).get("enabled"):
                raise SystemExit(f"support bundle did not collect online data: {json.dumps(support, sort_keys=True)}")
            encoded = json.dumps(report, sort_keys=True) + evidence_path.read_text(encoding="utf-8")
            for fragment in [INVITE_TOKEN, "CROWDTENSOR_MINER_TOKEN=", "lease_token", "idempotency_key"]:
                if fragment in encoded:
                    raise SystemExit(f"acceptance report leaked sensitive fragment: {fragment}")
            print(json.dumps({
                "ok": True,
                "schema": report["schema"],
                "miner_id": report.get("miner_id"),
                "route": report.get("route"),
                "evidence_schema": evidence.get("schema"),
                "support_online": (support.get("online") or {}).get("enabled"),
            }, sort_keys=True))
        finally:
            stop_process(coordinator)


if __name__ == "__main__":
    main()
