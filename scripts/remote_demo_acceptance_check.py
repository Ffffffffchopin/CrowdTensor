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
        "python-cli:cpu:0:model_bundle_infer",
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


def start_invited_miner(args: argparse.Namespace, invite: dict, log_dir: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["CROWDTENSOR_MINER_TOKEN"] = invite["env"]["CROWDTENSOR_MINER_TOKEN"]
    stdout = (log_dir / "miner_stdout.log").open("w", encoding="utf-8")
    stderr = (log_dir / "miner_stderr.log").open("w", encoding="utf-8")
    command = [
        sys.executable,
        str(ROOT / "miner_cli.py"),
        "--coordinator",
        args.base_url,
        "--miner-id",
        invite["miner_id"],
        "--max-tasks",
        "1",
        "--max-runtime-seconds",
        str(args.acceptance_timeout),
        "--compute-seconds",
        "0.2",
        "--heartbeat-interval",
        "0.1",
        "--idle-sleep",
        "0.2",
    ]
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
    proc._crowdtensor_stdout = stdout  # type: ignore[attr-defined]
    proc._crowdtensor_stderr = stderr  # type: ignore[attr-defined]
    return proc


def close_process_logs(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    for name in ("_crowdtensor_stdout", "_crowdtensor_stderr"):
        handle = getattr(proc, name, None)
        if handle is not None and not handle.closed:
            handle.close()


def tail_text(path: Path, limit: int = 2000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


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
        "--scenario-id",
        args.scenario_id,
        "--timeout-seconds",
        "20",
        "--poll-interval",
        "0.2",
        "--output-dir",
        str(output_dir),
        "--create-session",
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
    parser.add_argument("--scenario-id", default="route-baseline")
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
        log_dir = Path(temp) / "logs"
        registry_path = state_dir / "miner_registry.json"
        state_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        coordinator = None
        miner = None
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
            miner = start_invited_miner(args, invite, log_dir)
            try:
                report = run_acceptance(args, output_dir)
            except Exception as exc:
                stop_process(miner)
                close_process_logs(miner)
                raise RuntimeError(
                    f"{exc}\n"
                    f"miner stdout tail:\n{tail_text(log_dir / 'miner_stdout.log')}\n"
                    f"miner stderr tail:\n{tail_text(log_dir / 'miner_stderr.log')}"
                ) from exc
            try:
                miner.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            close_process_logs(miner)
            if report.get("schema") != "remote_demo_acceptance_v1" or not report.get("ok"):
                raise SystemExit(f"unexpected acceptance report: {json.dumps(report, sort_keys=True)}")
            session = report.get("session_request") or {}
            if not session.get("created") or session.get("schema") != "inference_session_request_v1":
                raise SystemExit(f"acceptance did not create an admin inference session: {session}")
            observed_queue = (report.get("observability_summary") or {}).get("work_queue") or {}
            if observed_queue.get("task_id") != session.get("task_id"):
                raise SystemExit(f"acceptance did not complete the created task: {observed_queue} session={session}")
            observability = report.get("observability_summary") or {}
            if observability.get("schema") != "remote_demo_observability_v1":
                raise SystemExit(f"missing remote demo observability summary: {observability}")
            availability = observability.get("availability") or {}
            observed_inference = observability.get("inference") or {}
            observed_artifacts = observability.get("artifacts") or {}
            if not availability.get("health_ok") or not availability.get("state_ok"):
                raise SystemExit(f"acceptance observability did not capture healthy endpoints: {availability}")
            if int(observed_inference.get("request_count", 0)) != args.request_count:
                raise SystemExit(f"acceptance observability request count mismatch: {observed_inference}")
            if observed_inference.get("scenario_id") != args.scenario_id or observed_inference.get("scenario_matches") is not True:
                raise SystemExit(f"acceptance observability scenario mismatch: {observed_inference}")
            if float(observed_inference.get("requests_per_second", 0.0)) <= 0.0:
                raise SystemExit(f"acceptance observability throughput is invalid: {observed_inference}")
            evidence_path = output_dir / "remote_compute_evidence.json"
            support_path = output_dir / "support_bundle.json"
            if not evidence_path.is_file() or not support_path.is_file():
                raise SystemExit("acceptance pack did not write evidence and support bundle artifacts")
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            support = json.loads(support_path.read_text(encoding="utf-8"))
            if evidence.get("schema") != "remote_compute_evidence_v1" or not evidence.get("ok"):
                raise SystemExit(f"unexpected evidence artifact: {json.dumps(evidence, sort_keys=True)}")
            evidence_summary = evidence.get("inference_summary") or {}
            if evidence_summary.get("scenario_id") != args.scenario_id or evidence_summary.get("scenario_matches") is not True:
                raise SystemExit(f"unexpected evidence scenario: {evidence_summary}")
            evidence_observability = evidence.get("observability_summary") or {}
            if evidence_observability.get("schema") != "remote_compute_observability_v1":
                raise SystemExit(f"unexpected evidence observability: {evidence_observability}")
            if observed_artifacts.get("evidence_observability_schema") != "remote_compute_observability_v1":
                raise SystemExit(f"acceptance did not summarize evidence observability: {observed_artifacts}")
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
                "scenario_id": (report.get("scenario") or {}).get("scenario_id"),
                "task_id": session.get("task_id"),
                "observability_schema": observability.get("schema"),
                "evidence_schema": evidence.get("schema"),
                "evidence_observability_schema": evidence_observability.get("schema"),
                "support_online": (support.get("online") or {}).get("enabled"),
            }, sort_keys=True))
        finally:
            stop_process(miner)
            close_process_logs(miner)
            stop_process(coordinator)


if __name__ == "__main__":
    main()
