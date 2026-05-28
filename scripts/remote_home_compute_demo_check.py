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
PRIVATE_OUTPUT_FILENAMES = (
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
)


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


def tail_logs(log_dir: Path, *, limit: int = 4000) -> str:
    chunks: list[str] = []
    for path in sorted(log_dir.glob("miner_*.log")):
        chunks.append(f"== {path.name} ==\n{tail_text(path, limit=limit)}")
    return "\n".join(chunks)


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
    if args.workload == "real-llm-sharded":
        command.extend(["--hf-model-id", args.hf_model_id])
        if args.hf_cache_dir:
            command.extend(["--hf-cache-dir", args.hf_cache_dir])
        if args.prompt_texts:
            command.extend(["--prompt-texts", args.prompt_texts])
    return run_json(command, timeout=args.command_timeout)


def start_coordinator(
    args: argparse.Namespace,
    state_dir: Path,
    registry_path: Path,
    operator_env: dict[str, str],
    log_dir: Path,
) -> subprocess.Popen:
    workload_type = {
        "external-llm": "external_llm_infer",
        "sharded-model-bundle": "sharded_model_bundle_infer",
        "micro-llm-sharded": "micro_llm_sharded_infer",
        "real-llm-sharded": "real_llm_sharded_infer",
    }.get(args.workload, "model_bundle_infer")
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
        f"python-cli:cpu:0:{workload_type}",
        "--miner-token-registry",
        str(registry_path),
        "--observer-token",
        operator_env["CROWDTENSOR_OBSERVER_TOKEN"],
        "--admin-token",
        operator_env["CROWDTENSOR_ADMIN_TOKEN"],
    ]
    if args.workload == "real-llm-sharded":
        command.extend(["--real-llm-model-id", args.hf_model_id])
        if args.hf_cache_dir:
            command.extend(["--hf-cache-dir", args.hf_cache_dir])
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    stdout = (log_dir / "coordinator_stdout.log").open("w", encoding="utf-8")
    stderr = (log_dir / "coordinator_stderr.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
    proc._crowdtensor_stdout = stdout  # type: ignore[attr-defined]
    proc._crowdtensor_stderr = stderr  # type: ignore[attr-defined]
    try:
        wait_health(args.base_url, proc, args.startup_timeout)
    except Exception as exc:
        stop_process(proc)
        close_process_logs(proc)
        raise RuntimeError(
            f"{exc}\n"
            f"coordinator stdout:\n{tail_text(log_dir / 'coordinator_stdout.log')}\n"
            f"coordinator stderr:\n{tail_text(log_dir / 'coordinator_stderr.log')}"
        ) from exc
    return proc


def start_miner(
    args: argparse.Namespace,
    miner_env: dict[str, str],
    log_dir: Path,
    *,
    miner_id: str = MINER_ID,
    stage_role: str = "",
) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["CROWDTENSOR_MINER_TOKEN"] = miner_env["CROWDTENSOR_MINER_TOKEN"]
    suffix = f"_{miner_id}" if miner_id != MINER_ID else ""
    stdout = (log_dir / f"miner_stdout{suffix}.log").open("w", encoding="utf-8")
    stderr = (log_dir / f"miner_stderr{suffix}.log").open("w", encoding="utf-8")
    command = [
        sys.executable,
        str(ROOT / "miner_cli.py"),
        "--coordinator",
        args.base_url,
        "--miner-id",
        miner_id,
        "--max-tasks",
        "2" if args.workload in {"sharded-model-bundle", "micro-llm-sharded", "real-llm-sharded"} else "1",
        "--max-runtime-seconds",
        str(args.verify_timeout),
        "--compute-seconds",
        "0.2",
        "--heartbeat-interval",
        "0.1",
        "--idle-sleep",
        "0.2",
    ]
    if args.workload in {"sharded-model-bundle", "micro-llm-sharded", "real-llm-sharded"}:
        command.extend(["--max-request-attempts", "200"])
    if stage_role and args.workload == "micro-llm-sharded":
        command.extend(["--micro-llm-stage-role", stage_role])
    if args.workload == "real-llm-sharded":
        command.extend(["--enable-hf-tiny-gpt-runtime", "--hf-model-id", args.hf_model_id])
        if args.hf_cache_dir:
            command.extend(["--hf-cache-dir", args.hf_cache_dir])
        if stage_role:
            command.extend(["--real-llm-stage-role", stage_role])
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
        "--decode-steps",
        str(args.decode_steps),
        "--stage-mode",
        args.stage_mode,
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
    if args.workload == "real-llm-sharded":
        command.extend(["--hf-model-id", args.hf_model_id])
        if args.hf_cache_dir:
            command.extend(["--hf-cache-dir", args.hf_cache_dir])
        if args.prompt_texts:
            command.extend(["--prompt-texts", args.prompt_texts])
    if args.require_distinct_stage_miners:
        command.append("--require-distinct-stage-miners")
    return run_json(command, timeout=args.command_timeout)


def run_doctor(
    args: argparse.Namespace,
    output_dir: Path,
    operator_env: dict[str, str],
    *,
    require_result: bool,
    miner_id: str = MINER_ID,
) -> dict:
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
        miner_id,
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
        "--decode-steps",
        str(args.decode_steps),
        "--json",
    ]
    if require_result:
        command.append("--require-result")
    if args.workload == "real-llm-sharded":
        command.extend(["--hf-model-id", args.hf_model_id])
        if args.hf_cache_dir:
            command.extend(["--hf-cache-dir", args.hf_cache_dir])
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
        "--decode-steps",
        str(args.decode_steps),
        "--task-id",
        task_id,
        "--json",
    ]
    if args.workload == "external-llm":
        command.append("--mock")
    if args.workload == "real-llm-sharded":
        command.extend(["--hf-model-id", args.hf_model_id])
        if args.hf_cache_dir:
            command.extend(["--hf-cache-dir", args.hf_cache_dir])
        if args.prompt_texts:
            command.extend(["--prompt-texts", args.prompt_texts])
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


def scrub_private_output(output_dir: Path) -> list[str]:
    removed: list[str] = []
    for name in PRIVATE_OUTPUT_FILENAMES:
        path = output_dir / name
        if not path.exists() and not path.is_symlink():
            continue
        if not path.is_file() and not path.is_symlink():
            raise RuntimeError(f"refusing to remove non-file private output: {path}")
        path.unlink()
        removed.append(name)
    return removed


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
        "sharded_inference_result",
        "real_llm_sharded_result",
        "activation_results",
        "activation_result",
        "hidden_state",
        "logits",
        "input_ids",
        "external_llm_results",
        "output_text",
        "Bearer ",
        *secret_values,
    ]:
        if fragment and fragment in encoded:
            raise SystemExit(f"{mode} report leaked sensitive fragment: {fragment}")


def validate_miner_join_pack(prepare: dict, output_dir: Path, *, secret_values: list[str]) -> None:
    join = (prepare.get("runbook_summary") or {})
    if join.get("miner_join_pack_schema") != "miner_join_pack_v1" or join.get("miner_join_pack_ready") is not True:
        raise SystemExit(f"prepare report missing ready miner_join_pack_v1 summary: {join}")
    artifacts = prepare.get("artifacts") or {}
    for name in ["miner_join_script", "miner_join_runbook", "miner_private_env"]:
        artifact = artifacts.get(name) or {}
        if artifact.get("present") is not True:
            raise SystemExit(f"prepare report missing join artifact {name}: {artifact}")
    join_script = output_dir / "miner_join.sh"
    join_runbook = output_dir / "MINER_JOIN.md"
    if not join_script.is_file() or not join_runbook.is_file():
        raise SystemExit("Miner join pack files were not written")
    public_text = join_runbook.read_text(encoding="utf-8", errors="replace")
    for fragment in ["operator.private.env", "miner_registry.json"]:
        if fragment not in public_text:
            raise SystemExit(f"Miner join runbook must warn about {fragment}")
    for secret in secret_values:
        if secret and secret in public_text:
            raise SystemExit("Miner join runbook leaked a private token")
    if "operator.private.env" in join_script.read_text(encoding="utf-8", errors="replace"):
        raise SystemExit("Miner join script must not require operator.private.env")


def add_registry_aliases(registry_path: Path, aliases: list[str]) -> None:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    miners = payload.get("miners")
    if not isinstance(miners, list) or not miners:
        raise SystemExit("miner registry missing base entry")
    base_entry = next((item for item in miners if isinstance(item, dict) and item.get("miner_id") == MINER_ID), None)
    if base_entry is None:
        raise SystemExit(f"miner registry missing base miner {MINER_ID}")
    existing = {str(item.get("miner_id")) for item in miners if isinstance(item, dict)}
    for alias in aliases:
        if alias in existing:
            continue
        entry = dict(base_entry)
        entry["miner_id"] = alias
        entry["label"] = f"{base_entry.get('label', MINER_ID)} {alias.rsplit('-', 1)[-1]}"
        miners.append(entry)
    registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run remote home-compute demo loopback check.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8920)
    parser.add_argument("--workload", choices=["model-bundle", "external-llm", "sharded-model-bundle", "micro-llm-sharded", "real-llm-sharded"], default="model-bundle")
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--scenario-id", default="route-baseline")
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--stage-mode", choices=["both", "split"], default="both")
    parser.add_argument("--require-distinct-stage-miners", action="store_true")
    parser.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--prompt-texts", default="CrowdTensor routes home CPU,A miner returns one token")
    parser.add_argument("--output-dir", default="", help="keep generated check artifacts in this directory instead of a temporary directory")
    parser.add_argument("--preserve-output", action="store_true", help="require --output-dir and retain public evidence after the check")
    parser.add_argument("--keep-private-output", action="store_true", help="when --output-dir is used, keep private env and registry files")
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--verify-timeout", type=float, default=30.0)
    parser.add_argument("--command-timeout", type=float, default=90.0)
    args = parser.parse_args()
    if args.preserve_output and not args.output_dir:
        parser.error("--preserve-output requires --output-dir")
    args.base_url = f"http://{args.host}:{args.port}"
    if args.workload == "real-llm-sharded":
        if args.stage_mode == "both":
            args.stage_mode = "split"
        if args.stage_mode == "split":
            args.require_distinct_stage_miners = True
        if args.request_count > 2:
            args.request_count = 1
    return args


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="crowdtensor_remote_home_demo_") as temp:
        temp_dir = Path(temp)
        output_dir = Path(args.output_dir).resolve() if args.output_dir else temp_dir / "remote-home-compute"
        if args.output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        state_dir = temp_dir / "state"
        log_dir = temp_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        coordinator = None
        miner = None
        miners: list[subprocess.Popen] = []
        private_files_removed: list[str] = []
        private_files_scrubbed = False
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
            validate_miner_join_pack(prepare, output_dir, secret_values=list(operator_env.values()) + list(miner_env.values()))
            split_stage = args.workload in {"micro-llm-sharded", "real-llm-sharded"} and (
                args.stage_mode == "split" or args.require_distinct_stage_miners
            )
            if split_stage:
                add_registry_aliases(registry_path, [f"{MINER_ID}-stage0", f"{MINER_ID}-stage1"])
            coordinator = start_coordinator(args, state_dir, registry_path, operator_env, log_dir)
            doctor_before = run_doctor(args, output_dir, operator_env, require_result=False)
            validate_public_report(doctor_before, mode="doctor", secret_values=list(operator_env.values()) + list(miner_env.values()))
            if split_stage:
                miners = [
                    start_miner(args, miner_env, log_dir, miner_id=f"{MINER_ID}-stage0", stage_role="stage0"),
                    start_miner(args, miner_env, log_dir, miner_id=f"{MINER_ID}-stage1", stage_role="stage1"),
                ]
                miner = miners[0]
            else:
                miner = start_miner(args, miner_env, log_dir)
                miners = [miner]
            try:
                verify = run_verify(args, output_dir, operator_env)
            except Exception as exc:
                for proc in miners:
                    stop_process(proc)
                    close_process_logs(proc)
                beta_report = output_dir / "remote_real_llm_sharded_beta.json"
                beta_tail = tail_text(beta_report, limit=4000) if beta_report.is_file() else ""
                extra = ""
                if "hf_dependencies_missing" in beta_tail:
                    extra = (
                        "\nreal-llm-sharded loopback requires optional Hugging Face dependencies. "
                        "Install with: python -m pip install -e .[hf]"
                    )
                raise RuntimeError(
                    f"{exc}\n"
                    f"{extra}\n"
                    f"real llm beta report tail:\n{beta_tail}\n"
                    f"miner log tails:\n{tail_logs(log_dir)}"
                ) from exc
            for proc in miners:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                close_process_logs(proc)
            validate_public_report(verify, mode="verify", secret_values=list(operator_env.values()) + list(miner_env.values()))
            task_id = str((verify.get("acceptance_summary") or {}).get("task_id") or "")
            doctor_miner_id = f"{MINER_ID}-stage1" if split_stage else MINER_ID
            doctor_after = run_doctor(args, output_dir, operator_env, require_result=True, miner_id=doctor_miner_id)
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
            elif args.workload == "sharded-model-bundle":
                expected_acceptance_schema = "remote_sharded_inference_acceptance_v1"
                expected_evidence_schema = "remote_sharded_inference_beta_v1"
                expected_observability_schema = "remote_sharded_inference_observability_v1"
                expected_ready_code = "remote_sharded_inference_ready"
            elif args.workload == "micro-llm-sharded":
                expected_acceptance_schema = "remote_micro_llm_sharded_acceptance_v1"
                expected_evidence_schema = "remote_micro_llm_sharded_beta_v1"
                expected_observability_schema = "remote_micro_llm_sharded_observability_v1"
                expected_ready_code = "remote_micro_llm_sharded_ready"
            elif args.workload == "real-llm-sharded":
                expected_acceptance_schema = "remote_real_llm_sharded_acceptance_v1"
                expected_evidence_schema = "remote_real_llm_sharded_beta_v1"
                expected_observability_schema = "remote_real_llm_sharded_observability_v1"
                expected_ready_code = "remote_real_llm_sharded_ready"
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
            elif args.workload == "sharded-model-bundle":
                artifact_names.extend([
                    "remote_sharded_inference_runbook_json",
                    "remote_sharded_inference_acceptance_json",
                    "remote_sharded_inference_beta_json",
                ])
            elif args.workload == "micro-llm-sharded":
                artifact_names.extend([
                    "remote_micro_llm_sharded_runbook_json",
                    "remote_micro_llm_sharded_acceptance_json",
                    "remote_micro_llm_sharded_beta_json",
                ])
            elif args.workload == "real-llm-sharded":
                artifact_names.extend([
                    "remote_real_llm_sharded_runbook_json",
                    "remote_real_llm_sharded_acceptance_json",
                    "remote_real_llm_sharded_beta_json",
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
            if args.output_dir and not args.keep_private_output:
                private_files_removed = scrub_private_output(output_dir)
                private_files_scrubbed = True
            print(json.dumps({
                "ok": True,
                "schema": "remote_home_compute_demo_check_v1",
                "demo_schema": verify.get("schema"),
                "miner_id": verify.get("miner_id"),
                "workload": args.workload,
                "output_dir": str(output_dir) if args.output_dir else "",
                "private_files_removed": private_files_removed,
                "scenario_id": (verify.get("demo") or {}).get("scenario_id"),
                "diagnosis_codes": verify.get("diagnosis_codes"),
                "miner_join_pack_schema": (prepare.get("runbook_summary") or {}).get("miner_join_pack_schema"),
                "doctor_schema": doctor_after.get("schema"),
                "collect_schema": collect.get("schema"),
                "cleanup_schema": clean.get("schema"),
                "acceptance_schema": acceptance.get("schema"),
                "evidence_schema": acceptance.get("evidence_schema"),
                "observability_schema": acceptance.get("observability_schema"),
            }, sort_keys=True))
        finally:
            for proc in miners or ([miner] if miner is not None else []):
                stop_process(proc)
                close_process_logs(proc)
            stop_process(coordinator)
            close_process_logs(coordinator)
            if args.output_dir and not args.keep_private_output and not private_files_scrubbed:
                scrub_private_output(output_dir)


if __name__ == "__main__":
    main()
