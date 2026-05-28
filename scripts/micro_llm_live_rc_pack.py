#!/usr/bin/env python3
"""Build the stage-aware micro-LLM live two-node RC report."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import kaggle_real_runtime_acceptance_pack as kaggle_pack  # noqa: E402
import support_bundle  # noqa: E402
from crowdtensor.micro_llm_artifact import build_default_micro_llm_artifact, inspect_micro_llm_artifact  # noqa: E402


SCHEMA = "micro_llm_live_rc_v1"
MODE_LOCAL_GENERATED = "local-generated"
MODE_EXTERNAL_EXISTING = "external-existing"
WORKLOAD_KIND = "micro-llm-sharded"
WORKLOAD_TYPE = "micro_llm_sharded_infer"
DEFAULT_PUBLIC_HOST = "24.199.118.54"
DEFAULT_PORT = 9180
Runner = Callable[..., subprocess.CompletedProcess[str]]
PopenFactory = Callable[..., subprocess.Popen[str]]

SECRET_FRAGMENTS = tuple(sorted(set(kaggle_pack.SECRET_FRAGMENTS + (
    "CROWDTENSOR_MINER_TOKEN",
    "CROWDTENSOR_OBSERVER_TOKEN",
    "CROWDTENSOR_ADMIN_TOKEN",
    "hidden_state",
    "activation_results",
    "activation_result",
    "logits",
))))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def effective_coordinator_url(args: argparse.Namespace) -> str:
    if args.coordinator_url:
        return args.coordinator_url.rstrip("/")
    if args.mode == MODE_LOCAL_GENERATED:
        return f"http://127.0.0.1:{args.port}"
    return f"http://{args.public_host}:{args.port}"


def kaggle_output_dir(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.kaggle_output_dir:
        return Path(args.kaggle_output_dir).resolve()
    return output_dir / "kaggle-real-runtime"


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    entry: dict[str, Any] = {"kind": kind, "path": relative, "present": path.is_file()}
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("command emitted no JSON object")


def redact_text(text: str, secret_values: list[str] | None = None) -> str:
    redacted = text
    for value in secret_values or []:
        if value:
            redacted = redacted.replace(value, "<redacted>")
    for fragment in SECRET_FRAGMENTS:
        redacted = redacted.replace(fragment, "<redacted>")
    return redacted


def redact_values(value: Any, secret_values: list[str] | None = None) -> Any:
    if isinstance(value, str):
        return redact_text(value, secret_values)
    if isinstance(value, list):
        return [redact_values(item, secret_values) for item in value]
    if isinstance(value, dict):
        return {key: redact_values(item, secret_values) for key, item in value.items()}
    return value


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def secret_values_from_envs(*paths: Path) -> list[str]:
    values: list[str] = []
    for path in paths:
        for value in kaggle_pack.parse_private_env(path).values():
            if value:
                values.append(value)
    return values


def run_json_step(
    name: str,
    command: list[str],
    *,
    runner: Runner,
    timeout_seconds: float,
    secret_values: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    try:
        completed = runner(
            command,
            cwd=str(ROOT),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return (
            {
                "name": name,
                "ok": False,
                "returncode": None,
                "duration_seconds": round(time.monotonic() - started, 3),
                "error": "timeout",
            },
            {},
        )
    step: dict[str, Any] = {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    payload: dict[str, Any] = {}
    try:
        payload = json_from_stdout(completed.stdout)
    except ValueError as exc:
        step["ok"] = False
        step["error"] = str(exc)
    else:
        step["payload_schema"] = payload.get("schema")
        step["payload_ok"] = payload.get("ok")
        step["ok"] = bool(step["ok"] and payload.get("ok") is not False)
    if not step["ok"]:
        if completed.stdout and not payload:
            step["stdout_tail"] = redact_text(completed.stdout[-1200:], secret_values)
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1200:], secret_values)
    return step, payload


def diagnosis_codes(*payloads: dict[str, Any], extra: list[str] | None = None) -> list[str]:
    codes: set[str] = set(extra or [])
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for code in payload.get("diagnosis_codes") or []:
            if isinstance(code, str):
                codes.add(code)
        for nested in (payload.get("payloads") or {}).values() if isinstance(payload.get("payloads"), dict) else []:
            if isinstance(nested, dict):
                for code in nested.get("diagnosis_codes") or []:
                    if isinstance(code, str):
                        codes.add(code)
    return sorted(codes)


def wait_for_ready(url: str, *, timeout_seconds: float, poll_interval: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            request = Request(f"{url.rstrip('/')}/ready", method="GET")
            with urlopen(request, timeout=min(2.0, max(0.2, timeout_seconds))) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return {"ok": True, "payload": payload}
        except (OSError, URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(poll_interval)
    return {"ok": False, "error": last_error or "timeout"}


def process_command_for_report(command: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for item in command:
        if skip_next:
            cleaned.append("<redacted>")
            skip_next = False
            continue
        cleaned.append(item)
        if item in {"--miner-token", "--observer-token", "--admin-token"}:
            skip_next = True
    return cleaned


def terminate_process(process: subprocess.Popen[str], *, timeout: float = 5.0) -> None:
    if process.poll() is not None:
        return
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            process.send_signal(signal.SIGTERM)
        process.wait(timeout=timeout)
    except Exception:
        try:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                process.kill()
            process.wait(timeout=timeout)
        except Exception:
            pass


def collect_process(
    process: subprocess.Popen[str],
    *,
    name: str,
    command: list[str],
    timeout: float,
    secret_values: list[str],
    terminate: bool = False,
) -> dict[str, Any]:
    if terminate:
        terminate_process(process, timeout=timeout)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        terminate_process(process, timeout=timeout)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", "process did not exit after termination"
    return {
        "name": name,
        "pid": process.pid,
        "returncode": process.returncode,
        "ok": process.returncode == 0 if not terminate else process.returncode in {0, -signal.SIGTERM, 143, None},
        "command": process_command_for_report(command),
        "stdout_tail": redact_text((stdout or "")[-1200:], secret_values),
        "stderr_tail": redact_text((stderr or "")[-1200:], secret_values),
    }


def stage_env_path(kaggle_dir: Path, stage_role: str) -> Path:
    if stage_role == "stage0":
        return kaggle_dir / "remote-home-compute" / "miner.private.env"
    return kaggle_dir / "remote-home-compute" / "miner.stage1.private.env"


def stage_upload_path(kaggle_dir: Path, stage_role: str) -> Path:
    return kaggle_dir / f"kaggle-upload-{stage_role}"


def start_stage_miner(
    args: argparse.Namespace,
    *,
    stage_role: str,
    coordinator_url: str,
    kaggle_dir: Path,
    secret_values: list[str],
    popen_factory: PopenFactory,
) -> tuple[subprocess.Popen[str], list[str]]:
    env_values = kaggle_pack.parse_private_env(stage_env_path(kaggle_dir, stage_role))
    env = os.environ.copy()
    env.update(env_values)
    env["PYTHONUNBUFFERED"] = "1"
    env["CROWDTENSOR_REMOTE_ENVIRONMENT"] = "local-generated-stage-upload-standin"
    miner_id = kaggle_pack.stage_miner_id(args.miner_id, stage_role)
    command = [
        sys.executable,
        str(ROOT / "miner_cli.py"),
        "--coordinator",
        coordinator_url,
        "--miner-id",
        miner_id,
        "--max-tasks",
        "1",
        "--compute-seconds",
        str(args.compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        str(args.idle_sleep),
        "--max-request-attempts",
        str(args.max_request_attempts),
        "--micro-llm-stage-role",
        stage_role,
    ]
    process = popen_factory(
        command,
        cwd=str(ROOT),
        env=env,
        start_new_session=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process, command


def run_prepare(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    kaggle_dir: Path,
    coordinator_url: str,
    runner: Runner,
) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "kaggle_real_runtime_acceptance_pack.py"),
        "prepare",
        "--public-host",
        args.public_host,
        "--port",
        str(args.port),
        "--coordinator-url",
        coordinator_url,
        "--miner-id",
        args.miner_id,
        "--workload",
        WORKLOAD_KIND,
        "--output-dir",
        str(kaggle_dir),
        "--request-count",
        str(args.request_count),
        "--decode-steps",
        str(args.decode_steps),
        "--stage-mode",
        "split",
        "--micro-llm-artifact",
        str(args.micro_llm_artifact),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--lease-seconds",
        str(args.lease_seconds),
        "--replace",
        "--json",
    ]
    return run_json_step(
        "kaggle_real_runtime_prepare",
        command,
        runner=runner,
        timeout_seconds=args.timeout_seconds,
        secret_values=[],
    )


def run_verify(
    args: argparse.Namespace,
    *,
    kaggle_dir: Path,
    coordinator_url: str,
    runner: Runner,
    secret_values: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "kaggle_real_runtime_acceptance_pack.py"),
        "verify",
        "--public-host",
        args.public_host,
        "--port",
        str(args.port),
        "--coordinator-url",
        coordinator_url,
        "--miner-id",
        args.miner_id,
        "--workload",
        WORKLOAD_KIND,
        "--output-dir",
        str(kaggle_dir),
        "--request-count",
        str(args.request_count),
        "--decode-steps",
        str(args.decode_steps),
        "--stage-mode",
        "split",
        "--require-distinct-stage-miners",
        "--micro-llm-artifact",
        str(args.micro_llm_artifact),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--artifact-timeout",
        str(args.artifact_timeout),
        "--admin-results-limit",
        str(args.admin_results_limit),
        "--collect-on-failure",
        "--json",
    ]
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    return run_json_step(
        "kaggle_real_runtime_verify",
        command,
        runner=runner,
        timeout_seconds=args.timeout_seconds,
        secret_values=secret_values,
    )


def base_artifacts(output_dir: Path, kaggle_dir: Path, *, ok: bool | None = None, micro_llm_artifact: str = "") -> dict[str, Any]:
    return {
        "micro_llm_live_rc_json": artifact_entry(
            output_dir / "micro_llm_live_rc.json",
            output_dir,
            kind="micro_llm_live_rc",
            schema=SCHEMA,
            ok=ok,
        ),
        "micro_llm_live_rc_markdown": artifact_entry(
            output_dir / "micro_llm_live_rc.md",
            output_dir,
            kind="micro_llm_live_rc_markdown",
        ),
        "kaggle_real_runtime_acceptance_json": artifact_entry(
            kaggle_dir / "kaggle_real_runtime_acceptance.json",
            output_dir,
            kind="kaggle_real_runtime_acceptance",
            schema="kaggle_real_runtime_acceptance_v1",
        ),
        "kaggle_real_runtime_acceptance_markdown": artifact_entry(
            kaggle_dir / "kaggle_real_runtime_acceptance.md",
            output_dir,
            kind="kaggle_real_runtime_acceptance_markdown",
        ),
        "kaggle_upload_stage0_miner_env": artifact_entry(
            stage_upload_path(kaggle_dir, "stage0") / "miner.private.env",
            output_dir,
            kind="kaggle_upload_private_env",
        ),
        "kaggle_upload_stage0_miner_script": artifact_entry(
            stage_upload_path(kaggle_dir, "stage0") / "kaggle_remote_miner.py",
            output_dir,
            kind="kaggle_upload_miner_script",
        ),
        "kaggle_upload_stage1_miner_env": artifact_entry(
            stage_upload_path(kaggle_dir, "stage1") / "miner.private.env",
            output_dir,
            kind="kaggle_upload_private_env",
        ),
        "kaggle_upload_stage1_miner_script": artifact_entry(
            stage_upload_path(kaggle_dir, "stage1") / "kaggle_remote_miner.py",
            output_dir,
            kind="kaggle_upload_miner_script",
        ),
        "remote_micro_llm_sharded_beta_json": artifact_entry(
            kaggle_dir / "remote-home-compute" / "remote_micro_llm_sharded_beta.json",
            output_dir,
            kind="remote_micro_llm_sharded_beta",
            schema="remote_micro_llm_sharded_beta_v1",
        ),
        "remote_micro_llm_sharded_acceptance_json": artifact_entry(
            kaggle_dir / "remote-home-compute" / "remote_micro_llm_sharded_acceptance.json",
            output_dir,
            kind="remote_micro_llm_sharded_acceptance",
            schema="remote_micro_llm_sharded_acceptance_v1",
        ),
        "support_bundle_json": artifact_entry(
            kaggle_dir / "remote-home-compute" / "support_bundle.json",
            output_dir,
            kind="support_bundle",
            schema="support_bundle_v1",
        ),
        "micro_llm_artifact_manifest": artifact_entry(
            Path(micro_llm_artifact) / "manifest.json",
            output_dir,
            kind="micro_llm_artifact_manifest",
            schema="micro_llm_artifact_v1",
        ),
    }


def stage_package_summary(kaggle_dir: Path) -> list[dict[str, Any]]:
    packages: list[dict[str, Any]] = []
    for stage_role in ["stage0", "stage1"]:
        upload = stage_upload_path(kaggle_dir, stage_role)
        script = upload / "kaggle_remote_miner.py"
        script_text = script.read_text(encoding="utf-8", errors="replace") if script.is_file() else ""
        packages.append({
            "stage_role": stage_role,
            "path": str(upload),
            "miner_env_present": (upload / "miner.private.env").is_file(),
            "miner_script_present": script.is_file(),
            "runbook_present": (upload / "KAGGLE_RUN.md").is_file(),
            "operator_env_excluded": not (upload / "operator.private.env").exists(),
            "launcher_has_stage_role": "--micro-llm-stage-role" in script_text and stage_role in script_text,
        })
    return packages


def safety_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "read_only": True,
        "cpu_only_workload": True,
        "workload_type": WORKLOAD_TYPE,
        "stage_mode": "split",
        "require_distinct_stage_miners": True,
        "captured_output_redacted": True,
        "summary_excludes_plaintext_tokens": True,
        "raw_activation_redacted": True,
        "temporary_http": effective_coordinator_url(args).startswith("http://"),
        "token_rotation_required": args.mode == MODE_EXTERNAL_EXISTING,
        "local_generated_stage_upload_standins": args.mode == MODE_LOCAL_GENERATED,
        "not_production": True,
        "not_p2p": True,
        "not_gpu_tpu_pooling": True,
        "not_gguf_llamacpp_serving": True,
        "not_large_model_sharding": True,
        "not_public_prompt_serving": True,
    }


def persist_report(report: dict[str, Any], *, output_dir: Path, secret_values: list[str]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = support_bundle.sanitize(redact_values(report, secret_values))
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    leaks.extend(secret for secret in secret_values if secret and secret in encoded)
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"sensitive_output_detected"})
        report["safety_error"] = "micro-LLM live RC report contained secret-like fragments"
    json_path = output_dir / "micro_llm_live_rc.json"
    md_path = output_dir / "micro_llm_live_rc.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    if "artifacts" in report:
        report["artifacts"]["micro_llm_live_rc_json"]["present"] = True
        report["artifacts"]["micro_llm_live_rc_markdown"]["present"] = True
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_local_generated(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    kaggle_dir: Path,
    runner: Runner,
    popen_factory: PopenFactory,
) -> dict[str, Any]:
    coordinator_url = effective_coordinator_url(args)
    steps: list[dict[str, Any]] = []
    processes: list[dict[str, Any]] = []
    coordinator_process: subprocess.Popen[str] | None = None
    stage_processes: list[tuple[str, subprocess.Popen[str], list[str]]] = []
    secret_values: list[str] = []
    verify_payload: dict[str, Any] = {}
    prepare_payload: dict[str, Any] = {}
    if not args.micro_llm_artifact:
        args.micro_llm_artifact = str(output_dir / "micro-llm-artifact")
        artifact_summary = build_default_micro_llm_artifact(args.micro_llm_artifact)
    else:
        artifact_summary = inspect_micro_llm_artifact(args.micro_llm_artifact)
    try:
        prepare_step, prepare_payload = run_prepare(
            args,
            output_dir=output_dir,
            kaggle_dir=kaggle_dir,
            coordinator_url=coordinator_url,
            runner=runner,
        )
        steps.append(prepare_step)
        secret_values = secret_values_from_envs(
            kaggle_dir / "remote-home-compute" / "operator.private.env",
            stage_env_path(kaggle_dir, "stage0"),
            stage_env_path(kaggle_dir, "stage1"),
        )
        if not prepare_step.get("ok") or not prepare_payload.get("ok"):
            raise RuntimeError("kaggle_real_runtime_prepare_failed")

        launch_script = kaggle_dir / "start_coordinator.sh"
        coordinator_command = ["bash", str(launch_script)]
        coordinator_process = popen_factory(
            coordinator_command,
            cwd=str(ROOT),
            start_new_session=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        ready = wait_for_ready(
            coordinator_url,
            timeout_seconds=args.startup_timeout,
            poll_interval=args.poll_interval,
        )
        steps.append({
            "name": "local_generated_coordinator_start",
            "ok": bool(ready.get("ok")),
            "pid": coordinator_process.pid,
            "coordinator_url": coordinator_url,
            "ready": ready,
        })
        if not ready.get("ok"):
            raise RuntimeError("coordinator_start_failed")

        for stage_role in ["stage0", "stage1"]:
            process, command = start_stage_miner(
                args,
                stage_role=stage_role,
                coordinator_url=coordinator_url,
                kaggle_dir=kaggle_dir,
                secret_values=secret_values,
                popen_factory=popen_factory,
            )
            stage_processes.append((stage_role, process, command))
            steps.append({
                "name": f"local_generated_{stage_role}_miner_start",
                "ok": True,
                "pid": process.pid,
                "miner_id": kaggle_pack.stage_miner_id(args.miner_id, stage_role),
            })

        verify_step, verify_payload = run_verify(
            args,
            kaggle_dir=kaggle_dir,
            coordinator_url=coordinator_url,
            runner=runner,
            secret_values=secret_values,
        )
        steps.append(verify_step)

        for stage_role, process, command in stage_processes:
            processes.append(collect_process(
                process,
                name=f"{stage_role}_miner",
                command=command,
                timeout=args.process_exit_timeout,
                secret_values=secret_values,
            ))
        stage_processes = []
    except Exception as exc:
        steps.append({"name": "local_generated_exception", "ok": False, "error": str(exc)})
    finally:
        for stage_role, process, command in stage_processes:
            processes.append(collect_process(
                process,
                name=f"{stage_role}_miner",
                command=command,
                timeout=args.process_exit_timeout,
                secret_values=secret_values,
                terminate=True,
            ))
        if coordinator_process is not None:
            processes.append(collect_process(
                coordinator_process,
                name="coordinator",
                command=["bash", str(kaggle_dir / "start_coordinator.sh")],
                timeout=args.process_exit_timeout,
                secret_values=secret_values,
                terminate=True,
            ))

    packages = stage_package_summary(kaggle_dir)
    process_ok = all(item.get("ok") for item in processes if item.get("name", "").endswith("_miner"))
    package_ok = all(
        package.get("miner_env_present")
        and package.get("miner_script_present")
        and package.get("runbook_present")
        and package.get("operator_env_excluded")
        and package.get("launcher_has_stage_role")
        for package in packages
    )
    codes = diagnosis_codes(prepare_payload, verify_payload)
    if package_ok:
        codes.append("local_generated_stage_upload_packages_ready")
    if process_ok and package_ok and "kaggle_micro_llm_sharded_ready" in codes:
        codes.append("local_generated_stage_upload_standins_ready")
    if "kaggle_micro_llm_sharded_ready" in codes and "local_generated_stage_upload_standins_ready" in codes:
        codes.append("micro_llm_live_rc_ready")
    else:
        codes.append("micro_llm_live_rc_blocked")
    ok = "micro_llm_live_rc_ready" in codes and package_ok and process_ok
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": MODE_LOCAL_GENERATED,
        "output_dir": str(output_dir),
        "kaggle_output_dir": str(kaggle_dir),
        "coordinator_url": coordinator_url,
        "public_host": args.public_host,
        "port": args.port,
        "miner_id": args.miner_id,
        "workload": {
            "kind": WORKLOAD_KIND,
            "workload_type": WORKLOAD_TYPE,
            "stage_mode": "split",
            "request_count": args.request_count,
            "decode_steps": args.decode_steps,
            "require_distinct_stage_miners": True,
            "artifact_hash": artifact_summary.get("artifact_hash"),
            "artifact_id": artifact_summary.get("artifact_id"),
        },
        "artifact": artifact_summary,
        "runtime_classification": {
            "local_generated_stage_upload_standins": True,
            "external_runtime_verified": False,
            "kaggle_notebook_verified": False,
        },
        "stage_packages": packages,
        "process_summary": processes,
        "steps": steps,
        "payload_summaries": {
            "prepare": kaggle_pack.summarize_payload(prepare_payload),
            "verify": kaggle_pack.summarize_payload(verify_payload),
            "remote_micro_llm_beta": kaggle_pack.summarize_payload(
                load_json(kaggle_dir / "remote-home-compute" / "remote_micro_llm_sharded_beta.json")
            ),
        },
        "diagnosis_codes": sorted(set(codes)),
        "artifacts": base_artifacts(output_dir, kaggle_dir, ok=ok, micro_llm_artifact=args.micro_llm_artifact),
        "safety": safety_summary(args),
        "operator_action": [
            "Use this local-generated mode as the mandatory CI-safe live stand-in before attempting real Kaggle Notebooks.",
            "For a real external run, start the generated public Coordinator and two external stage Miners, then rerun with --mode external-existing.",
        ],
        "limitations": [
            "local-generated starts local processes from generated stage packages; it is not live Kaggle Notebook evidence.",
            "CPU-only read-only deterministic toy two-stage micro-LLM pipeline; not production Swarm Inference.",
            "No P2P routing, NAT traversal, GPU/TPU pooling, GGUF/llama.cpp serving, large-model sharding, training, or arbitrary prompt serving.",
        ],
    }
    return persist_report(report, output_dir=output_dir, secret_values=secret_values)


def build_external_existing(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    kaggle_dir: Path,
    runner: Runner,
) -> dict[str, Any]:
    coordinator_url = effective_coordinator_url(args)
    artifact_summary = inspect_micro_llm_artifact(args.micro_llm_artifact) if args.micro_llm_artifact else {}
    secret_values = secret_values_from_envs(
        kaggle_dir / "remote-home-compute" / "operator.private.env",
        stage_env_path(kaggle_dir, "stage0"),
        stage_env_path(kaggle_dir, "stage1"),
    )
    secret_values.extend([args.observer_token, args.admin_token])
    verify_step, verify_payload = run_verify(
        args,
        kaggle_dir=kaggle_dir,
        coordinator_url=coordinator_url,
        runner=runner,
        secret_values=secret_values,
    )
    codes = diagnosis_codes(verify_payload)
    if verify_payload.get("ok") and "kaggle_micro_llm_sharded_ready" in codes:
        codes.extend(["external_runtime_verified", "micro_llm_live_rc_ready"])
    else:
        codes.extend(["external_runtime_blocked", "micro_llm_live_rc_blocked"])
    ok = "micro_llm_live_rc_ready" in codes and bool(verify_step.get("ok"))
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": MODE_EXTERNAL_EXISTING,
        "output_dir": str(output_dir),
        "kaggle_output_dir": str(kaggle_dir),
        "coordinator_url": coordinator_url,
        "public_host": args.public_host,
        "port": args.port,
        "miner_id": args.miner_id,
        "workload": {
            "kind": WORKLOAD_KIND,
            "workload_type": WORKLOAD_TYPE,
            "stage_mode": "split",
            "request_count": args.request_count,
            "decode_steps": args.decode_steps,
            "require_distinct_stage_miners": True,
            "artifact_hash": artifact_summary.get("artifact_hash"),
            "artifact_id": artifact_summary.get("artifact_id"),
        },
        "artifact": artifact_summary,
        "runtime_classification": {
            "local_generated_stage_upload_standins": False,
            "external_runtime_verified": ok,
            "kaggle_notebook_verified": ok,
        },
        "stage_packages": stage_package_summary(kaggle_dir),
        "steps": [verify_step],
        "payload_summaries": {
            "verify": kaggle_pack.summarize_payload(verify_payload),
            "remote_micro_llm_beta": kaggle_pack.summarize_payload(
                load_json(kaggle_dir / "remote-home-compute" / "remote_micro_llm_sharded_beta.json")
            ),
        },
        "diagnosis_codes": sorted(set(codes)),
        "artifacts": base_artifacts(output_dir, kaggle_dir, ok=ok, micro_llm_artifact=args.micro_llm_artifact),
        "safety": safety_summary(args),
        "operator_action": [
            "If external_runtime_blocked is present, confirm both stage0 and stage1 Miners are running against the same Coordinator URL.",
            "Rotate tokens after any temporary HTTP public Coordinator run.",
        ],
        "limitations": [
            "external-existing verifies already running external Miners; it does not create Kaggle Notebooks.",
            "CPU-only read-only deterministic toy two-stage micro-LLM pipeline; not production Swarm Inference.",
            "No P2P routing, NAT traversal, GPU/TPU pooling, GGUF/llama.cpp serving, large-model sharding, training, or arbitrary prompt serving.",
        ],
    }
    return persist_report(report, output_dir=output_dir, secret_values=secret_values)


def build_report(
    args: argparse.Namespace,
    *,
    runner: Runner = subprocess.run,
    popen_factory: PopenFactory = subprocess.Popen,
) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    k_dir = kaggle_output_dir(args, output_dir)
    k_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == MODE_LOCAL_GENERATED:
        return build_local_generated(args, output_dir=output_dir, kaggle_dir=k_dir, runner=runner, popen_factory=popen_factory)
    return build_external_existing(args, output_dir=output_dir, kaggle_dir=k_dir, runner=runner)


def render_markdown(report: dict[str, Any]) -> str:
    runtime = report.get("runtime_classification") or {}
    lines = [
        "# CrowdTensor Micro-LLM Live Two-Node RC",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- coordinator_url: `{report.get('coordinator_url')}`",
        f"- request_count: `{(report.get('workload') or {}).get('request_count')}`",
        f"- decode_steps: `{(report.get('workload') or {}).get('decode_steps')}`",
        f"- local_generated_stage_upload_standins: `{runtime.get('local_generated_stage_upload_standins')}`",
        f"- external_runtime_verified: `{runtime.get('external_runtime_verified')}`",
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`",
        "",
        "## Stage Packages",
        "",
    ]
    for package in report.get("stage_packages") or []:
        lines.append(
            f"- `{package.get('stage_role')}`: env=`{package.get('miner_env_present')}` "
            f"script=`{package.get('miner_script_present')}` role=`{package.get('launcher_has_stage_role')}`"
        )
    lines.extend(["", "## Artifacts", ""])
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        lines.append(f"- `{name}`: `{artifact.get('path')}` present=`{artifact.get('present')}`")
    lines.extend(["", "## Boundaries", ""])
    for limitation in report.get("limitations") or []:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the micro-LLM live two-node RC report.")
    parser.add_argument("--mode", choices=[MODE_LOCAL_GENERATED, MODE_EXTERNAL_EXISTING], default=MODE_LOCAL_GENERATED)
    parser.add_argument("--output-dir", default="dist/micro-llm-live-rc")
    parser.add_argument("--kaggle-output-dir", default="")
    parser.add_argument("--public-host", default=DEFAULT_PUBLIC_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--miner-id", default="kaggle-cpu-1")
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--micro-llm-artifact", default="")
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--remote-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument("--process-exit-timeout", type=float, default=10.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--http-timeout", type=float, default=5.0)
    parser.add_argument("--artifact-timeout", type=float, default=60.0)
    parser.add_argument("--admin-results-limit", type=int, default=10)
    parser.add_argument("--lease-seconds", type=float, default=15.0)
    parser.add_argument("--compute-seconds", type=float, default=0.2)
    parser.add_argument("--heartbeat-interval", type=float, default=0.1)
    parser.add_argument("--idle-sleep", type=float, default=0.2)
    parser.add_argument("--max-request-attempts", type=int, default=20)
    parser.add_argument("--observer-token", default="")
    parser.add_argument("--admin-token", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.port < 1:
        raise SystemExit("--port must be positive")
    if args.request_count < 1 or args.request_count > 8:
        raise SystemExit("--request-count must be between 1 and 8")
    if args.decode_steps < 1 or args.decode_steps > 4:
        raise SystemExit("--decode-steps must be between 1 and 4")
    for name in [
        "timeout_seconds",
        "remote_timeout_seconds",
        "startup_timeout",
        "process_exit_timeout",
        "poll_interval",
        "http_timeout",
        "artifact_timeout",
        "lease_seconds",
        "heartbeat_interval",
        "idle_sleep",
    ]:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.compute_seconds < 0:
        raise SystemExit("--compute-seconds must be non-negative")
    if args.max_request_attempts < 1:
        raise SystemExit("--max-request-attempts must be at least 1")
    if args.admin_results_limit < 1:
        raise SystemExit("--admin-results-limit must be at least 1")
    args.coordinator_url = args.coordinator_url.rstrip("/") if args.coordinator_url else ""
    return args


def print_human(report: dict[str, Any]) -> None:
    print("CrowdTensor micro-LLM live two-node RC")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  coordinator: {report.get('coordinator_url')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    runtime = report.get("runtime_classification") or {}
    print(f"  local stand-in: {runtime.get('local_generated_stage_upload_standins')}")
    print(f"  external runtime: {runtime.get('external_runtime_verified')}")


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print_human(report)
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
