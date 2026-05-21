#!/usr/bin/env python3
"""Subprocess chaos smoke for CrowdTensorD V2.2."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from auth_headers import activate_miner_token, activate_observer_token, add_miner_token_arg, add_observer_token_arg, coordinator_env, json_headers, observer_headers  # noqa: E402


class ChaosError(RuntimeError):
    pass


class HttpStatusError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


def request_json(method: str, base_url: str, path: str, payload: dict | None = None, *, timeout: float = 5.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=observer_headers() if method == "GET" else json_headers(),
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise HttpStatusError(exc.code, body) from exc
    except URLError as exc:
        raise ChaosError(f"coordinator unreachable: {exc}") from exc


def get_json(base_url: str, path: str) -> dict:
    return request_json("GET", base_url, path)


def post_json(base_url: str, path: str, payload: dict) -> dict:
    return request_json("POST", base_url, path, payload)


def wait_health(base_url: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise ChaosError(f"coordinator exited early with code {proc.returncode}")
        try:
            health = get_json(base_url, "/health")
            if health.get("ok") is True:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.1)
    raise ChaosError(f"coordinator did not become healthy: {last_error}")


def start_coordinator(args: argparse.Namespace, state_dir: Path) -> subprocess.Popen:
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
        str(args.backlog),
        "--reaper-interval",
        str(args.reaper_interval),
    ]
    env = coordinator_env()
    env["PYTHONUNBUFFERED"] = "1"
    output = None if args.verbose_server else subprocess.DEVNULL
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
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


def run_miner_once(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(ROOT / "miner_cli.py"),
        "--coordinator",
        args.base_url,
        "--miner-id",
        "chaos-miner",
        "--once",
        "--compute-seconds",
        "0.1",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=20)
    if completed.returncode != 0:
        raise ChaosError(
            "miner_cli.py failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    print("ok normal miner result accepted")


def run_parallel_miners(args: argparse.Namespace) -> None:
    if args.parallel_miners <= 1:
        return

    processes: list[tuple[str, subprocess.Popen]] = []
    for index in range(args.parallel_miners):
        miner_id = f"parallel-miner-{index}"
        command = [
            sys.executable,
            str(ROOT / "miner_cli.py"),
            "--coordinator",
            args.base_url,
            "--miner-id",
            miner_id,
            "--once",
            "--compute-seconds",
            str(args.parallel_compute_seconds),
        ]
        processes.append((
            miner_id,
            subprocess.Popen(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE),
        ))

    failures: list[str] = []
    for miner_id, proc in processes:
        try:
            stdout, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate(timeout=5)
            failures.append(f"{miner_id} timed out\nstdout:\n{stdout}\nstderr:\n{stderr}")
            continue

        if proc.returncode != 0:
            failures.append(
                f"{miner_id} failed with code {proc.returncode}\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )

    if failures:
        raise ChaosError("\n\n".join(failures))

    state = get_json(args.base_url, "/state")
    if state["max_staleness"] <= 0:
        raise ChaosError(f"expected async staleness > 0, got {state}")
    print(
        f"ok parallel miners={args.parallel_miners} "
        f"accepted_results={state['accepted_results']} "
        f"max_staleness={state['max_staleness']} "
        f"avg_staleness={state['avg_staleness']:.3f}"
    )


def assert_malicious_delta_rejected(args: argparse.Namespace) -> None:
    before = get_json(args.base_url, "/state")
    claim = post_json(args.base_url, "/tasks/claim", {"miner_id": "malicious-probe"})
    bad_delta = [1000.0 for _ in claim["weights"]]
    try:
        post_json(
            args.base_url,
            f"/tasks/{claim['task_id']}/result",
            {
                "lease_token": claim["lease_token"],
                "attempt": claim["attempt"],
                "local_delta": bad_delta,
            },
        )
    except HttpStatusError as exc:
        if exc.status != 422:
            raise
    else:
        raise ChaosError("malicious delta was accepted")

    after = get_json(args.base_url, "/state")
    if after["model"]["global_step"] != before["model"]["global_step"]:
        raise ChaosError("rejected malicious delta changed global_step")
    if after["rejected_results"] != before["rejected_results"] + 1:
        raise ChaosError(f"expected rejected_results to increase: before={before} after={after}")
    malicious_score = after["miner_scores"].get("malicious-probe", {})
    if malicious_score.get("rejected") != 1 or malicious_score.get("score", 0) >= 0:
        raise ChaosError(f"malicious miner score was not penalized: {malicious_score}")
    print("ok malicious delta rejected and penalized")


def compute_delta(claim: dict, miner_id: str) -> list[float]:
    from crowdtensor.diloco import run_inner_loop

    result = run_inner_loop(
        claim["weights"],
        task_id=claim["task_id"],
        miner_id=miner_id,
        model_version=claim["model_version"],
        inner_steps=claim["inner_steps"],
    )
    return [value * 0.1 for value in result["local_delta"]]


def assert_stale_result_rejected(args: argparse.Namespace) -> None:
    stale = post_json(args.base_url, "/tasks/claim", {"miner_id": "stale-probe"})
    time.sleep(args.lease_seconds + 0.3)
    state = get_json(args.base_url, "/state")
    if state["task_counts"]["queued"] < 1:
        raise ChaosError(f"expected queued task after timeout, got {state['task_counts']}")

    try:
        post_json(
            args.base_url,
            f"/tasks/{stale['task_id']}/result",
            {
                "lease_token": stale["lease_token"],
                "attempt": stale["attempt"],
                "local_delta": compute_delta(stale, "stale-probe"),
            },
        )
    except HttpStatusError as exc:
        if exc.status != 409:
            raise
    else:
        raise ChaosError("stale result was accepted")

    fresh = post_json(args.base_url, "/tasks/claim", {"miner_id": "fresh-probe"})
    if fresh["task_id"] != stale["task_id"] or fresh["attempt"] != stale["attempt"] + 1:
        raise ChaosError("timed out task was not reclaimed with incremented attempt")

    post_json(
        args.base_url,
        f"/tasks/{fresh['task_id']}/result",
        {
            "lease_token": fresh["lease_token"],
            "attempt": fresh["attempt"],
            "local_delta": compute_delta(fresh, "fresh-probe"),
        },
    )
    print("ok timeout requeue and stale result rejection")


def assert_restart_recovers_inflight(args: argparse.Namespace, state_dir: Path, proc: subprocess.Popen) -> subprocess.Popen:
    claim = post_json(args.base_url, "/tasks/claim", {"miner_id": "restart-probe"})
    stop_process(proc)

    restarted = start_coordinator(args, state_dir)
    state = get_json(args.base_url, "/state")
    if state["task_counts"]["leased"] != 0 or state["task_counts"]["queued"] < 1:
        stop_process(restarted)
        raise ChaosError(f"restart did not recover in-flight task: {state['task_counts']}")

    fresh = post_json(args.base_url, "/tasks/claim", {"miner_id": "restart-fresh"})
    if fresh["task_id"] != claim["task_id"] or fresh["attempt"] != claim["attempt"] + 1:
        stop_process(restarted)
        raise ChaosError("restart did not requeue the in-flight task with incremented attempt")

    post_json(
        args.base_url,
        f"/tasks/{fresh['task_id']}/result",
        {
            "lease_token": fresh["lease_token"],
            "attempt": fresh["attempt"],
            "local_delta": compute_delta(fresh, "restart-fresh"),
        },
    )
    print("ok coordinator restart recovered in-flight task")
    return restarted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CrowdTensorD V2.2 chaos smoke.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--lease-seconds", type=float, default=1.0)
    parser.add_argument("--inner-steps", type=int, default=20)
    parser.add_argument("--backlog", type=int, default=1)
    parser.add_argument("--parallel-miners", type=int, default=1)
    parser.add_argument("--parallel-compute-seconds", type=float, default=0.4)
    parser.add_argument("--reaper-interval", type=float, default=0.1)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--verbose-server", action="store_true")
    add_miner_token_arg(parser)
    add_observer_token_arg(parser)
    args = parser.parse_args()
    activate_miner_token(args)
    activate_observer_token(args)
    if args.parallel_miners > args.backlog:
        args.backlog = args.parallel_miners
    args.base_url = f"http://{args.host}:{args.port}"
    return args


def main() -> None:
    args = parse_args()
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.state_dir:
        state_dir = Path(args.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_chaos_")
        state_dir = Path(temp_dir.name)

    proc: subprocess.Popen | None = None
    try:
        proc = start_coordinator(args, state_dir)
        run_miner_once(args)
        assert_malicious_delta_rejected(args)
        assert_stale_result_rejected(args)
        proc = assert_restart_recovers_inflight(args, state_dir, proc)
        run_parallel_miners(args)
        final_state = get_json(args.base_url, "/state")
        expected_results = 3 + max(0, args.parallel_miners if args.parallel_miners > 1 else 0)
        if final_state["accepted_results"] < expected_results:
            raise ChaosError(f"expected accepted_results >= {expected_results}, got {final_state}")
        if final_state["model"].get("optimizer_step", 0) != final_state["accepted_results"]:
            raise ChaosError(f"optimizer_step did not match accepted_results: {final_state}")
        if final_state["rejected_results"] < 1:
            raise ChaosError(f"expected rejected_results >= 1, got {final_state}")
        print(
            f"ok final global_step={final_state['model']['global_step']} "
            f"optimizer_step={final_state['model']['optimizer_step']} "
            f"accepted_results={final_state['accepted_results']} "
            f"rejected_results={final_state['rejected_results']} "
            f"max_staleness={final_state['max_staleness']} "
            f"counts={final_state['task_counts']}"
        )
    finally:
        stop_process(proc)
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
