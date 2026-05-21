#!/usr/bin/env python3
"""Smoke test for remote Miner invite and registry join flow."""

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


MINER_ID = "remote-join-miner"
OBSERVER_TOKEN = "remote-join-observer"


def request_json(
    method: str,
    base_url: str,
    path: str,
    payload: dict | None = None,
    *,
    observer_token: str = "",
    timeout: float = 5.0,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json"} if payload is not None else {}
    if observer_token:
        headers["x-crowdtensor-observer-token"] = observer_token
    request = Request(f"{base_url.rstrip('/')}{path}", data=data, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def wait_ready(base_url: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"coordinator exited early with code {proc.returncode}")
        try:
            ready = request_json("GET", base_url, "/ready", timeout=2.0)
            if ready.get("ok") is True:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"coordinator did not become ready: {last_error}")


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
        str(args.lease_seconds),
        "--inner-steps",
        str(args.inner_steps),
        "--backlog",
        "1",
        "--miner-token-registry",
        str(registry_path),
        "--observer-token",
        OBSERVER_TOKEN,
    ]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    wait_ready(args.base_url, proc, args.startup_timeout)
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


def parse_summary(stdout: str) -> dict:
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
    raise RuntimeError(f"missing miner summary JSON in stdout:\n{stdout}")


def run_invited_miner(args: argparse.Namespace, invite: dict) -> dict:
    env = dict(os.environ)
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
        str(args.compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        "0.2",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.miner_timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "invited miner failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return parse_summary(completed.stdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run remote Miner invite/join smoke.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8898)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--inner-steps", type=int, default=8)
    parser.add_argument("--compute-seconds", type=float, default=0.2)
    parser.add_argument("--heartbeat-interval", type=float, default=0.1)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--miner-timeout", type=float, default=30.0)
    args = parser.parse_args()
    args.base_url = f"http://{args.host}:{args.port}"
    return args


def main() -> None:
    args = parse_args()
    temp_dir = None
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_remote_join_")
        state_dir = Path(temp_dir.name)
    state_dir.mkdir(parents=True, exist_ok=True)
    registry_path = state_dir / "miner_registry.json"

    coordinator = None
    try:
        invite = create_invite(
            registry_path=registry_path,
            miner_id=MINER_ID,
            coordinator_url=args.base_url,
            label="remote join smoke",
            token="remote-join-token",
        )
        registry_text = registry_path.read_text(encoding="utf-8")
        if "remote-join-token" in registry_text or "sha256:" not in registry_text:
            raise SystemExit(f"registry did not store hashed token only: {registry_text}")

        coordinator = start_coordinator(args, state_dir, registry_path)
        miner_summary = run_invited_miner(args, invite)
        state = request_json("GET", args.base_url, "/state", observer_token=OBSERVER_TOKEN)
        profile = (state.get("miner_profiles") or {}).get(MINER_ID) or {}
        completed = [
            task for task in state.get("tasks", [])
            if task.get("status") == "completed" and task.get("miner_id") == MINER_ID
        ]
        if int(miner_summary.get("accepted_tasks", 0)) != 1:
            raise SystemExit(f"unexpected miner summary: {json.dumps(miner_summary, sort_keys=True)}")
        if len(completed) != 1:
            raise SystemExit(f"expected one completed invited Miner task: {json.dumps(state, sort_keys=True)}")
        if profile.get("runtime") != "python-cli" or profile.get("backend") != "cpu":
            raise SystemExit(f"missing invited Miner profile: {json.dumps(profile, sort_keys=True)}")

        print(json.dumps({
            "accepted_results": state["accepted_results"],
            "completed_task_id": completed[0]["task_id"],
            "invite_token_hash_prefix": invite["token_hash"].split(":", 1)[0],
            "miner_id": MINER_ID,
            "profile_runtime": profile["runtime"],
            "registry": str(registry_path),
            "request_retries": miner_summary.get("request_retries", 0),
        }, sort_keys=True))
    finally:
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
