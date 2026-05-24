#!/usr/bin/env python3
"""CI-safe local-loopback check for the high-level remote home-compute demo."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]

MINER_ID = "remote-home-demo-miner"


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


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


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


def parse_private_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        parsed = shlex.split(raw_value)
        values[key] = parsed[0] if parsed else ""
    return values


def run_json(command: list[str], *, timeout: float) -> dict:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    for line in reversed([line for line in completed.stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError(f"command emitted no JSON: {' '.join(command)}\nstdout:\n{completed.stdout}")


def run_prepare(args: argparse.Namespace, output_dir: Path) -> dict:
    command = [
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "remote-demo",
        "prepare",
        "--workload",
        args.workload,
        "--coordinator-url",
        args.base_url,
        "--miner-id",
        MINER_ID,
        "--output-dir",
        str(output_dir),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--timeout-seconds",
        str(int(args.command_timeout)),
        "--replace",
        "--json",
    ]
    if args.workload == "external-llm":
        command.append("--mock")
    return run_json(command, timeout=args.command_timeout)


def start_coordinator(args: argparse.Namespace, state_dir: Path, registry_path: Path, operator_env: dict[str, str]) -> subprocess.Popen:
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
        f"python-cli:cpu:0:{'external_llm_infer' if args.workload == 'external-llm' else 'model_bundle_infer'}",
        "--miner-token-registry",
        str(registry_path),
        "--observer-token",
        operator_env["CROWDTENSOR_OBSERVER_TOKEN"],
        "--admin-token",
        operator_env["CROWDTENSOR_ADMIN_TOKEN"],
    ]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    wait_health(args.base_url, proc, args.startup_timeout)
    return proc


def start_miner(args: argparse.Namespace, miner_env: dict[str, str], log_dir: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["CROWDTENSOR_MINER_TOKEN"] = miner_env["CROWDTENSOR_MINER_TOKEN"]
    stdout = (log_dir / "miner_stdout.log").open("w", encoding="utf-8")
    stderr = (log_dir / "miner_stderr.log").open("w", encoding="utf-8")
    command = [
        sys.executable,
        str(ROOT / "miner_cli.py"),
        "--coordinator",
        args.base_url,
        "--miner-id",
        MINER_ID,
        "--max-tasks",
        "1",
        "--max-runtime-seconds",
        str(args.verify_timeout),
        "--compute-seconds",
        "0.2",
        "--heartbeat-interval",
        "0.1",
        "--idle-sleep",
        "0.2",
    ]
    if args.workload == "external-llm":
        command.extend(["--enable-mock-llm-runtime", "--llm-runtime-model-id", "loopback-mock-llm"])
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
    proc._crowdtensor_stdout = stdout  # type: ignore[attr-defined]
    proc._crowdtensor_stderr = stderr  # type: ignore[attr-defined]
    return proc


def run_verify(args: argparse.Namespace, output_dir: Path, operator_env: dict[str, str]) -> dict:
    command = [
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "remote-demo",
        "verify",
        "--workload",
        args.workload,
        "--coordinator-url",
        args.base_url,
        "--miner-id",
        MINER_ID,
        "--observer-token",
        operator_env["CROWDTENSOR_OBSERVER_TOKEN"],
        "--admin-token",
        operator_env["CROWDTENSOR_ADMIN_TOKEN"],
        "--output-dir",
        str(output_dir),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--timeout-seconds",
        str(int(args.command_timeout)),
        "--remote-timeout-seconds",
        str(args.verify_timeout),
        "--poll-interval",
        "0.2",
        "--create-session",
        "--json",
    ]
    if args.workload == "external-llm":
        command.append("--mock")
    return run_json(command, timeout=args.command_timeout)


def run_doctor(args: argparse.Namespace, output_dir: Path, operator_env: dict[str, str], *, require_result: bool) -> dict:
    command = [
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "remote-demo",
        "doctor",
        "--workload",
        args.workload,
        "--coordinator-url",
        args.base_url,
        "--miner-id",
        MINER_ID,
        "--observer-token",
        operator_env["CROWDTENSOR_OBSERVER_TOKEN"],
        "--admin-token",
        operator_env["CROWDTENSOR_ADMIN_TOKEN"],
        "--output-dir",
        str(output_dir),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--json",
    ]
    if require_result:
        command.append("--require-result")
    return run_json(command, timeout=args.command_timeout)


def run_collect(args: argparse.Namespace, output_dir: Path, operator_env: dict[str, str], task_id: str) -> dict:
    command = [
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "remote-demo",
        "collect",
        "--workload",
        args.workload,
        "--coordinator-url",
        args.base_url,
        "--miner-id",
        MINER_ID,
        "--observer-token",
        operator_env["CROWDTENSOR_OBSERVER_TOKEN"],
        "--admin-token",
        operator_env["CROWDTENSOR_ADMIN_TOKEN"],
        "--output-dir",
        str(output_dir),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--task-id",
        task_id,
        "--json",
    ]
    if args.workload == "external-llm":
        command.append("--mock")
    return run_json(command, timeout=args.command_timeout)


def run_clean(args: argparse.Namespace, output_dir: Path) -> dict:
    return run_json([
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "remote-demo",
        "clean",
        "--output-dir",
        str(output_dir),
        "--json",
    ], timeout=args.command_timeout)


def validate_public_report(payload: dict, *, mode: str, secret_values: list[str]) -> None:
    schema_by_mode = {
        "prepare": "remote_home_compute_demo_v1",
        "verify": "remote_home_compute_demo_v1",
        "doctor": "remote_home_compute_doctor_v1",
        "collect": "remote_home_compute_collect_v1",
        "clean": "remote_home_compute_cleanup_v1",
    }
    if payload.get("schema") != schema_by_mode[mode] or not payload.get("ok"):
        raise SystemExit(f"unexpected {mode} report: {json.dumps(payload, sort_keys=True)}")
    if mode in {"prepare", "verify"} and payload.get("mode") != mode:
        raise SystemExit(f"unexpected {mode} mode: {json.dumps(payload, sort_keys=True)}")
    encoded = json.dumps(payload, sort_keys=True)
    for fragment in [
        "lease_token",
        "idempotency_key",
        "inference_results",
        "external_llm_results",
        "output_text",
        "Bearer ",
        *secret_values,
    ]:
        if fragment and fragment in encoded:
            raise SystemExit(f"{mode} report leaked sensitive fragment: {fragment}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run remote home-compute demo loopback check.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8920)
    parser.add_argument("--workload", choices=["model-bundle", "external-llm"], default="model-bundle")
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--scenario-id", default="route-baseline")
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--verify-timeout", type=float, default=30.0)
    parser.add_argument("--command-timeout", type=float, default=90.0)
    args = parser.parse_args()
    args.base_url = f"http://{args.host}:{args.port}"
    return args


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="crowdtensor_remote_home_demo_") as temp:
        temp_dir = Path(temp)
        output_dir = temp_dir / "remote-home-compute"
        state_dir = temp_dir / "state"
        log_dir = temp_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        coordinator = None
        miner = None
        try:
            prepare = run_prepare(args, output_dir)
            operator_env = parse_private_env(output_dir / "operator.private.env")
            miner_env = parse_private_env(output_dir / "miner.private.env")
            registry_path = output_dir / "miner_registry.json"
            required_env = {"CROWDTENSOR_OBSERVER_TOKEN", "CROWDTENSOR_ADMIN_TOKEN"}
            if set(operator_env) & required_env != required_env:
                raise SystemExit(f"operator.private.env missing required values: {operator_env.keys()}")
            if "CROWDTENSOR_MINER_TOKEN" not in miner_env:
                raise SystemExit("miner.private.env missing CROWDTENSOR_MINER_TOKEN")
            validate_public_report(prepare, mode="prepare", secret_values=list(operator_env.values()) + list(miner_env.values()))
            coordinator = start_coordinator(args, state_dir, registry_path, operator_env)
            doctor_before = run_doctor(args, output_dir, operator_env, require_result=False)
            validate_public_report(doctor_before, mode="doctor", secret_values=list(operator_env.values()) + list(miner_env.values()))
            miner = start_miner(args, miner_env, log_dir)
            try:
                verify = run_verify(args, output_dir, operator_env)
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
            validate_public_report(verify, mode="verify", secret_values=list(operator_env.values()) + list(miner_env.values()))
            task_id = str((verify.get("acceptance_summary") or {}).get("task_id") or "")
            doctor_after = run_doctor(args, output_dir, operator_env, require_result=True)
            validate_public_report(doctor_after, mode="doctor", secret_values=list(operator_env.values()) + list(miner_env.values()))
            collect = run_collect(args, output_dir, operator_env, task_id)
            validate_public_report(collect, mode="collect", secret_values=list(operator_env.values()) + list(miner_env.values()))
            clean = run_clean(args, output_dir)
            validate_public_report(clean, mode="clean", secret_values=list(operator_env.values()) + list(miner_env.values()))
            acceptance = verify.get("acceptance_summary") or {}
            if args.workload == "external-llm":
                expected_acceptance_schema = "remote_external_llm_acceptance_v1"
                expected_evidence_schema = "remote_external_llm_evidence_v1"
                expected_observability_schema = "remote_external_llm_observability_v1"
                expected_ready_code = "remote_external_llm_ready"
            else:
                expected_acceptance_schema = "remote_demo_acceptance_v1"
                expected_evidence_schema = "remote_compute_evidence_v1"
                expected_observability_schema = "remote_demo_observability_v1"
                expected_ready_code = "remote_home_compute_ready"
            if expected_ready_code not in verify.get("diagnosis_codes", []):
                raise SystemExit(f"verify report did not emit {expected_ready_code}: {verify.get('diagnosis_codes')}")
            if acceptance.get("schema") != expected_acceptance_schema or acceptance.get("evidence_schema") != expected_evidence_schema:
                raise SystemExit(f"verify summary did not carry expected acceptance/evidence schemas: {acceptance}")
            if acceptance.get("observability_schema") != expected_observability_schema:
                raise SystemExit(f"verify summary did not carry remote demo observability: {acceptance}")
            artifact_names = [
                "remote_home_compute_demo_json",
                "support_bundle_json",
            ]
            if args.workload == "external-llm":
                artifact_names.extend([
                    "remote_external_llm_runbook_json",
                    "remote_external_llm_acceptance_json",
                    "remote_external_llm_evidence_json",
                ])
            else:
                artifact_names.extend([
                    "remote_demo_runbook_json",
                    "remote_demo_acceptance_json",
                    "remote_compute_evidence_json",
                ])
            for artifact_name in artifact_names:
                artifact = (verify.get("artifacts") or {}).get(artifact_name) or {}
                if artifact.get("present") is not True:
                    raise SystemExit(f"verify report missing artifact {artifact_name}: {artifact}")
            print(json.dumps({
                "ok": True,
                "schema": "remote_home_compute_demo_check_v1",
                "demo_schema": verify.get("schema"),
                "miner_id": verify.get("miner_id"),
                "workload": args.workload,
                "scenario_id": (verify.get("demo") or {}).get("scenario_id"),
                "diagnosis_codes": verify.get("diagnosis_codes"),
                "doctor_schema": doctor_after.get("schema"),
                "collect_schema": collect.get("schema"),
                "cleanup_schema": clean.get("schema"),
                "acceptance_schema": acceptance.get("schema"),
                "evidence_schema": acceptance.get("evidence_schema"),
                "observability_schema": acceptance.get("observability_schema"),
            }, sort_keys=True))
        finally:
            stop_process(miner)
            close_process_logs(miner)
            stop_process(coordinator)


if __name__ == "__main__":
    main()
