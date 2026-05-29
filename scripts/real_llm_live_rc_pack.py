#!/usr/bin/env python3
"""Build the real tiny-LLM live two-node RC report."""

from __future__ import annotations

import argparse
import ast
import json
import os
import secrets
import shlex
import signal
import stat
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

import support_bundle  # noqa: E402
from create_miner_invite import create_invite  # noqa: E402
from crowdtensor.auth import hash_token  # noqa: E402
from crowdtensor.real_llm import BACKEND_CPU as REAL_LLM_BACKEND_CPU  # noqa: E402
from crowdtensor.real_llm import BACKEND_CUDA as REAL_LLM_BACKEND_CUDA  # noqa: E402
from crowdtensor.real_llm import DEFAULT_MODEL_ID, DEFAULT_PROMPTS, inspect_real_llm_artifact  # noqa: E402
from crowdtensor.real_llm import normalize_backend as normalize_real_llm_backend  # noqa: E402
from crowdtensor.real_llm import normalize_partition_mode as normalize_real_llm_partition_mode  # noqa: E402


SCHEMA = "real_llm_live_rc_v1"
MODE_LOCAL_GENERATED = "local-generated"
MODE_KAGGLE_GENERATED = "kaggle-generated"
MODE_EXTERNAL_EXISTING = "external-existing"
WORKLOAD_TYPE = "real_llm_sharded_infer"
DEFAULT_PUBLIC_HOST = "24.199.118.54"
DEFAULT_PORT = 9184
Runner = Callable[..., subprocess.CompletedProcess[str]]
PopenFactory = Callable[..., subprocess.Popen[str]]

SECRET_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "lease_token",
    "idempotency_key",
    "activation_results",
    "activation_result",
    "hidden_state",
    "input_ids",
    "logits",
    "inference_results",
    "inference_result",
    "sharded_inference_result",
    "real_llm_sharded_result",
    "output_text",
    "Bearer ",
    "CrowdTensor routes",
    "A miner returns",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def effective_coordinator_url(args: argparse.Namespace) -> str:
    if args.coordinator_url:
        return args.coordinator_url.rstrip("/")
    if args.mode == MODE_LOCAL_GENERATED:
        return f"http://127.0.0.1:{args.port}"
    return f"http://{args.public_host}:{args.port}"


def remote_runtime_dir(output_dir: Path) -> Path:
    return output_dir / "remote-real-llm-runtime"


def coordinator_state_dir(output_dir: Path) -> Path:
    return output_dir / "coordinator-state"


def stage_upload_path(output_dir: Path, stage_role: str) -> Path:
    return output_dir / f"kaggle-upload-real-llm-{stage_role}"


def stage_miner_id(base_miner_id: str, stage_role: str) -> str:
    base = str(base_miner_id or "kaggle-real-llm").strip()
    suffix = f"-{stage_role}"
    return base if base.endswith(suffix) else f"{base}{suffix}"


def quote_env(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def write_private_env(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"export {key}={quote_env(value)}" for key, value in sorted(values.items()) if value]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def parse_private_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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


def secret_values_from_envs(*paths: Path) -> list[str]:
    values: list[str] = []
    for path in paths:
        for value in parse_private_env(path).values():
            if value:
                values.append(value)
    return values


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


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


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
        summaries = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
        for summary in summaries.values():
            if isinstance(summary, dict):
                for code in summary.get("diagnosis_codes") or []:
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


def render_coordinator_launch(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    observer_hash: str,
    admin_hash: str,
) -> str:
    state_dir = coordinator_state_dir(output_dir)
    registry = remote_runtime_dir(output_dir) / "miner_registry.json"
    runtime_backend = "cuda" if args.real_llm_backend == REAL_LLM_BACKEND_CUDA else "cpu"
    lane = f"python-cli:{runtime_backend}:0:{WORKLOAD_TYPE}"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(ROOT))}",
        f"{shlex.quote(sys.executable)} coordinator.py \\",
        f"  --host {shlex.quote(args.bind_host)} \\",
        f"  --port {args.port} \\",
        f"  --state-dir {shlex.quote(str(state_dir))} \\",
        f"  --lease-seconds {args.lease_seconds} \\",
        f"  --inner-steps {args.request_count} \\",
        "  --backlog 0 \\",
        f"  --task-lane {shlex.quote(lane)} \\",
        f"  --real-llm-model-id {shlex.quote(args.hf_model_id)} \\",
        f"  --real-llm-backend {shlex.quote(args.real_llm_backend)} \\",
        f"  --real-llm-partition-mode {shlex.quote(args.real_llm_partition_mode)} \\",
    ]
    if args.hf_cache_dir:
        lines.append(f"  --hf-cache-dir {shlex.quote(args.hf_cache_dir)} \\")
    lines.extend([
        f"  --miner-token-registry {shlex.quote(str(registry))} \\",
        f"  --observer-token {shlex.quote(observer_hash)} \\",
        f"  --admin-token {shlex.quote(admin_hash)}",
        "",
    ])
    return "\n".join(lines)


def render_stage_kaggle_miner(args: argparse.Namespace, *, miner_id: str, stage_role: str) -> str:
    coordinator_url = effective_coordinator_url(args)
    cache_lines = ""
    if args.hf_cache_dir:
        cache_lines = f'    "--hf-cache-dir",\n    "{args.hf_cache_dir}",\n'
    return f'''#!/usr/bin/env python3
"""Kaggle Remote Miner launcher for CrowdTensor real tiny-LLM sharded inference."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {{}}
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


env = os.environ.copy()
env.update(load_env(Path(os.environ.get("CROWDTENSOR_MINER_ENV_FILE", "miner.private.env"))))
env["CROWDTENSOR_REMOTE_ENVIRONMENT"] = "kaggle-real-llm"
env.setdefault("PYTHONUNBUFFERED", "1")
command = [
    "crowdtensor-miner",
    "--coordinator",
    "{coordinator_url}",
    "--miner-id",
    "{miner_id}",
    "--max-tasks",
    "{args.max_new_tokens}",
    "--compute-seconds",
    "0.2",
    "--heartbeat-interval",
    "0.1",
    "--enable-hf-tiny-gpt-runtime",
    "--hf-model-id",
    "{args.hf_model_id}",
    "--real-llm-backend",
    "{args.real_llm_backend}",
    "--real-llm-partition-mode",
    "{args.real_llm_partition_mode}",
{cache_lines}    "--real-llm-stage-role",
    "{stage_role}",
    "--max-request-attempts",
    "120",
    "--idle-sleep",
    "1.0",
]
print("Starting Kaggle real LLM remote Miner:", " ".join(command), flush=True)
raise SystemExit(subprocess.call(command, env=env))
'''


def render_kaggle_runbook(args: argparse.Namespace, *, output_dir: Path, stage_role: str) -> str:
    return "\n".join([
        "# CrowdTensor Real Small-LLM Stage Miner",
        "",
        f"Coordinator URL: `{effective_coordinator_url(args)}`",
        f"Miner ID: `{stage_miner_id(args.miner_id, stage_role)}`",
        f"Stage role: `{stage_role}`",
        f"Workload: `{WORKLOAD_TYPE}`",
        f"HF model: `{args.hf_model_id}`",
        "",
        "## Notebook Commands",
        "",
        "```bash",
        "python -m pip install -e '.[hf]'",
        "python kaggle_remote_miner.py --env-file miner.private.env",
        "```",
        "",
        "## Uploaded Files",
        "",
        "- `miner.private.env`",
        "- `kaggle_remote_miner.py`",
        "",
        "Do not upload `operator.private.env` or `miner_registry.json` to Kaggle.",
        "",
        "## Boundary",
        "",
        "- CPU-only read-only tiny Hugging Face GPT stage execution.",
        "- Not production Swarm Inference, not P2P, not GPU/TPU pooling, not GGUF/llama.cpp serving, and not large-model serving.",
        "- Rotate tokens after temporary public Coordinator tests.",
        "",
    ])


def render_operator_commands(args: argparse.Namespace, *, output_dir: Path) -> str:
    verify_command = (
        "crowdtensor real-llm-live-rc "
        "--mode external-existing "
        f"--coordinator-url {shlex.quote(effective_coordinator_url(args))} "
        f"--observer-token $(grep CROWDTENSOR_OBSERVER_TOKEN {remote_runtime_dir(output_dir) / 'operator.private.env'} | cut -d= -f2-) "
        f"--admin-token $(grep CROWDTENSOR_ADMIN_TOKEN {remote_runtime_dir(output_dir) / 'operator.private.env'} | cut -d= -f2-) "
        f"--output-dir {shlex.quote(str(output_dir))} "
        f"--request-count {args.request_count} "
        f"--max-new-tokens {args.max_new_tokens} "
        f"--hf-model-id {shlex.quote(args.hf_model_id)} "
        "--json"
    )
    return "\n".join([
        "# CrowdTensor Real Small-LLM Live RC Operator Commands",
        "",
        "## 1. Start the public Coordinator",
        "",
        "```bash",
        f"bash {output_dir / 'start_coordinator.sh'}",
        "```",
        "",
        "## 2. Upload two stage packages",
        "",
        f"- Stage 0 package: `{stage_upload_path(output_dir, 'stage0')}`",
        f"- Stage 1 package: `{stage_upload_path(output_dir, 'stage1')}`",
        "- Keep `operator.private.env` and `miner_registry.json` on the operator host.",
        "",
        "## 3. Run each Kaggle Notebook",
        "",
        "```bash",
        "python -m pip install -e '.[hf]'",
        "python kaggle_remote_miner.py --env-file miner.private.env",
        "```",
        "",
        "## 4. Verify from the operator host",
        "",
        "```bash",
        verify_command,
        "```",
        "",
        "## Boundary",
        "",
        "- CPU-only read-only `real_llm_sharded_infer`; not production Swarm Inference.",
        "- Not P2P, not GPU/TPU pooling, not GGUF/llama.cpp serving, and not large-model serving.",
        "- Rotate generated tokens after a temporary HTTP run.",
        "",
    ])


def write_stage_upload_package(args: argparse.Namespace, *, output_dir: Path, stage_role: str) -> dict[str, Any]:
    upload = stage_upload_path(output_dir, stage_role)
    upload.mkdir(parents=True, exist_ok=True)
    source_env = remote_runtime_dir(output_dir) / f"miner.{stage_role}.private.env"
    miner_env = upload / "miner.private.env"
    if source_env.is_file():
        miner_env.write_text(source_env.read_text(encoding="utf-8"), encoding="utf-8")
        miner_env.chmod(stat.S_IRUSR | stat.S_IWUSR)
    script = upload / "kaggle_remote_miner.py"
    script.write_text(
        render_stage_kaggle_miner(
            args,
            miner_id=stage_miner_id(args.miner_id, stage_role),
            stage_role=stage_role,
        ),
        encoding="utf-8",
    )
    script.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    runbook = upload / "KAGGLE_RUN.md"
    runbook.write_text(render_kaggle_runbook(args, output_dir=output_dir, stage_role=stage_role), encoding="utf-8")
    script_text = script.read_text(encoding="utf-8", errors="replace")
    return {
        "stage_role": stage_role,
        "miner_id": stage_miner_id(args.miner_id, stage_role),
        "path": str(upload),
        "miner_env_present": miner_env.is_file(),
        "miner_script_present": script.is_file(),
        "runbook_present": runbook.is_file(),
        "operator_env_excluded": not (upload / "operator.private.env").exists(),
        "launcher_has_stage_role": "--real-llm-stage-role" in script_text and stage_role in script_text,
        "hf_runtime_enabled": "--enable-hf-tiny-gpt-runtime" in script_text,
        "hf_model_id_present": args.hf_model_id in script_text,
        "launcher_syntax_valid": _python_syntax_valid(script_text, filename=str(script)),
    }


def prepare_generated_artifacts(args: argparse.Namespace, *, output_dir: Path) -> tuple[dict[str, Any], list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = remote_runtime_dir(output_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    coordinator_url = effective_coordinator_url(args)
    artifact_summary = inspect_real_llm_artifact(
        model_id=args.hf_model_id,
        cache_dir=args.hf_cache_dir,
        backend=args.real_llm_backend,
        require_runtime=args.real_llm_backend != REAL_LLM_BACKEND_CUDA,
    )
    artifact_summary["partition_mode"] = args.real_llm_partition_mode

    observer_token = args.observer_token or secrets.token_urlsafe(32)
    admin_token = args.admin_token or secrets.token_urlsafe(32)
    stage0_token = secrets.token_urlsafe(32)
    stage1_token = secrets.token_urlsafe(32)
    registry = runtime_dir / "miner_registry.json"
    for stage_role, token in [("stage0", stage0_token), ("stage1", stage1_token)]:
        create_invite(
            registry_path=registry,
            miner_id=stage_miner_id(args.miner_id, stage_role),
            coordinator_url=coordinator_url,
            label=f"Real LLM {stage_role} Miner",
            token=token,
            replace=True,
        )
        write_private_env(runtime_dir / f"miner.{stage_role}.private.env", {"CROWDTENSOR_MINER_TOKEN": token})

    write_private_env(
        runtime_dir / "operator.private.env",
        {
            "CROWDTENSOR_OBSERVER_TOKEN": observer_token,
            "CROWDTENSOR_ADMIN_TOKEN": admin_token,
        },
    )
    start_script = output_dir / "start_coordinator.sh"
    start_script.write_text(
        render_coordinator_launch(
            args=args,
            output_dir=output_dir,
            observer_hash=hash_token(observer_token),
            admin_hash=hash_token(admin_token),
        ),
        encoding="utf-8",
    )
    start_script.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    (output_dir / "operator_commands.md").write_text(render_operator_commands(args, output_dir=output_dir), encoding="utf-8")
    packages = [
        write_stage_upload_package(args, output_dir=output_dir, stage_role="stage0"),
        write_stage_upload_package(args, output_dir=output_dir, stage_role="stage1"),
    ]
    secret_values = [observer_token, admin_token, stage0_token, stage1_token]
    summary = {
        "artifact": artifact_summary,
        "stage_packages": packages,
        "observer_token": observer_token,
        "admin_token": admin_token,
    }
    return summary, secret_values


def stage_package_summary(output_dir: Path) -> list[dict[str, Any]]:
    packages: list[dict[str, Any]] = []
    for stage_role in ["stage0", "stage1"]:
        upload = stage_upload_path(output_dir, stage_role)
        script = upload / "kaggle_remote_miner.py"
        script_text = script.read_text(encoding="utf-8", errors="replace") if script.is_file() else ""
        packages.append({
            "stage_role": stage_role,
            "path": str(upload),
            "miner_env_present": (upload / "miner.private.env").is_file(),
            "miner_script_present": script.is_file(),
            "runbook_present": (upload / "KAGGLE_RUN.md").is_file(),
            "operator_env_excluded": not (upload / "operator.private.env").exists(),
            "launcher_has_stage_role": "--real-llm-stage-role" in script_text and stage_role in script_text,
            "hf_runtime_enabled": "--enable-hf-tiny-gpt-runtime" in script_text,
            "hf_model_id_present": "sshleifer/tiny-gpt2" in script_text or "--hf-model-id" in script_text,
            "launcher_syntax_valid": _python_syntax_valid(script_text, filename=str(script)) if script.is_file() else False,
        })
    return packages


def _python_syntax_valid(source: str, *, filename: str = "<generated>") -> bool:
    try:
        ast.parse(source, filename=filename)
    except SyntaxError:
        return False
    return True


def stage_package_ready(package: dict[str, Any]) -> bool:
    return bool(
        package.get("miner_env_present")
        and package.get("miner_script_present")
        and package.get("runbook_present")
        and package.get("operator_env_excluded")
        and package.get("launcher_has_stage_role")
        and package.get("hf_runtime_enabled")
        and package.get("launcher_syntax_valid")
    )


def start_stage_miner(
    args: argparse.Namespace,
    *,
    stage_role: str,
    coordinator_url: str,
    output_dir: Path,
    popen_factory: PopenFactory,
) -> tuple[subprocess.Popen[str], list[str]]:
    env_values = parse_private_env(stage_upload_path(output_dir, stage_role) / "miner.private.env")
    env = os.environ.copy()
    env.update(env_values)
    env["PYTHONUNBUFFERED"] = "1"
    env["CROWDTENSOR_REMOTE_ENVIRONMENT"] = "local-generated-real-llm-stage-upload-standin"
    command = [
        sys.executable,
        str(ROOT / "miner_cli.py"),
        "--coordinator",
        coordinator_url,
        "--miner-id",
        stage_miner_id(args.miner_id, stage_role),
        "--max-tasks",
        str(args.max_new_tokens),
        "--compute-seconds",
        str(args.compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        str(args.idle_sleep),
        "--max-request-attempts",
        str(args.max_request_attempts),
        "--enable-hf-tiny-gpt-runtime",
        "--hf-model-id",
        args.hf_model_id,
        "--real-llm-backend",
        args.real_llm_backend,
        "--real-llm-partition-mode",
        args.real_llm_partition_mode,
        "--real-llm-stage-role",
        stage_role,
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
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


def run_verify(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    coordinator_url: str,
    observer_token: str,
    admin_token: str,
    runner: Runner,
    secret_values: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_dir = remote_runtime_dir(output_dir)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "remote_real_llm_sharded_beta_pack.py"),
        "--mode",
        "remote-existing",
        "--output-dir",
        str(runtime_dir),
        "--json-out",
        str(runtime_dir / "remote_real_llm_sharded_beta.json"),
        "--markdown-out",
        str(runtime_dir / "remote_real_llm_sharded_beta.md"),
        "--base-port",
        str(args.port),
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--hf-model-id",
        args.hf_model_id,
        "--real-llm-backend",
        args.real_llm_backend,
        "--real-llm-partition-mode",
        args.real_llm_partition_mode,
        "--failure-mode",
        "none",
        "--stage-mode",
        "split",
        "--require-distinct-stage-miners",
        "--coordinator-url",
        coordinator_url,
        "--observer-token",
        observer_token,
        "--admin-token",
        admin_token,
        "--timeout-seconds",
        str(int(args.timeout_seconds)),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--prompt-texts",
        ",".join(DEFAULT_PROMPTS),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    return run_json_step(
        "remote_real_llm_sharded_verify",
        command,
        runner=runner,
        timeout_seconds=args.timeout_seconds,
        secret_values=secret_values,
    )


def base_artifacts(output_dir: Path, *, ok: bool | None = None) -> dict[str, Any]:
    runtime_dir = remote_runtime_dir(output_dir)
    return {
        "real_llm_live_rc_json": artifact_entry(
            output_dir / "real_llm_live_rc.json",
            output_dir,
            kind="real_llm_live_rc",
            schema=SCHEMA,
            ok=ok,
        ),
        "real_llm_live_rc_markdown": artifact_entry(
            output_dir / "real_llm_live_rc.md",
            output_dir,
            kind="real_llm_live_rc_markdown",
        ),
        "operator_commands": artifact_entry(output_dir / "operator_commands.md", output_dir, kind="operator_commands"),
        "coordinator_launch_script": artifact_entry(output_dir / "start_coordinator.sh", output_dir, kind="coordinator_launch_script"),
        "operator_private_env": artifact_entry(runtime_dir / "operator.private.env", output_dir, kind="private_env"),
        "miner_registry": artifact_entry(runtime_dir / "miner_registry.json", output_dir, kind="miner_registry"),
        "kaggle_upload_real_llm_stage0_miner_env": artifact_entry(
            stage_upload_path(output_dir, "stage0") / "miner.private.env",
            output_dir,
            kind="kaggle_upload_private_env",
        ),
        "kaggle_upload_real_llm_stage0_miner_script": artifact_entry(
            stage_upload_path(output_dir, "stage0") / "kaggle_remote_miner.py",
            output_dir,
            kind="kaggle_upload_miner_script",
        ),
        "kaggle_upload_real_llm_stage1_miner_env": artifact_entry(
            stage_upload_path(output_dir, "stage1") / "miner.private.env",
            output_dir,
            kind="kaggle_upload_private_env",
        ),
        "kaggle_upload_real_llm_stage1_miner_script": artifact_entry(
            stage_upload_path(output_dir, "stage1") / "kaggle_remote_miner.py",
            output_dir,
            kind="kaggle_upload_miner_script",
        ),
        "remote_real_llm_sharded_beta_json": artifact_entry(
            runtime_dir / "remote_real_llm_sharded_beta.json",
            output_dir,
            kind="remote_real_llm_sharded_beta",
            schema="remote_real_llm_sharded_beta_v1",
        ),
        "remote_existing_real_llm_sharded_evidence_json": artifact_entry(
            runtime_dir / "remote-existing-real-llm-shard-infer" / "real_llm_sharded_evidence.json",
            output_dir,
            kind="real_llm_sharded_evidence",
            schema="real_llm_sharded_evidence_v1",
        ),
    }


def summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "diagnosis_codes": diagnosis_codes(payload),
    }
    generation = generation_summary(payload)
    if generation:
        summary["generation"] = generation
    if payload.get("coordinator_url") is not None:
        summary["coordinator_url"] = payload.get("coordinator_url")
    if payload.get("hf_model_id") is not None:
        summary["hf_model_id"] = payload.get("hf_model_id")
    if isinstance(payload.get("workload"), dict):
        summary["workload"] = {
            "workload_type": payload["workload"].get("workload_type"),
            "stage_mode": payload["workload"].get("stage_mode"),
            "request_count": payload["workload"].get("request_count"),
            "max_new_tokens": payload["workload"].get("max_new_tokens"),
            "hf_model_id": payload["workload"].get("hf_model_id"),
            "real_llm_partition_mode": payload["workload"].get("real_llm_partition_mode"),
        }
    if isinstance(payload.get("payload_summaries"), dict):
        for key, value in payload["payload_summaries"].items():
            if isinstance(value, dict):
                summary[key] = {
                    "schema": value.get("schema"),
                    "ok": value.get("ok"),
                    "diagnosis_codes": value.get("diagnosis_codes") or [],
                    "generation": generation_summary(value),
                    "stage_assignment": value.get("stage_assignment") or {},
                    "artifact": value.get("artifact") or {},
                }
    return summary


def generation_summary(payload: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_score = -1
    pending: list[Any] = [payload]
    seen: set[int] = set()
    while pending:
        item = pending.pop(0)
        if not isinstance(item, dict):
            continue
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        generation = item.get("generation")
        if isinstance(generation, dict) and generation:
            score = 0
            if generation.get("generated_text_hash"):
                score += 8
            if generation.get("generated_token_count"):
                score += 4
            if generation.get("multi_token_generation_ready"):
                score += 2
            if generation.get("generated_text_redacted"):
                score += 1
            if score > best_score:
                best = generation
                best_score = score
        for value in item.values():
            if isinstance(value, dict):
                pending.append(value)
            elif isinstance(value, list):
                pending.extend(entry for entry in value if isinstance(entry, dict))
    return best


def safety_summary(args: argparse.Namespace) -> dict[str, Any]:
    gpu_backend = args.real_llm_backend == REAL_LLM_BACKEND_CUDA
    return {
        "read_only": True,
        "cpu_only_workload": not gpu_backend,
        "gpu_backend_selected": gpu_backend,
        "coordinator_cuda_runtime_required": False if gpu_backend else None,
        "miner_cuda_runtime_required": gpu_backend,
        "workload_type": WORKLOAD_TYPE,
        "stage_mode": "split",
        "real_llm_partition_mode": args.real_llm_partition_mode,
        "require_distinct_stage_miners": True,
        "captured_output_redacted": True,
        "summary_excludes_plaintext_tokens": True,
        "raw_activation_redacted": True,
        "temporary_http": effective_coordinator_url(args).startswith("http://"),
        "token_rotation_required": args.mode in {MODE_KAGGLE_GENERATED, MODE_EXTERNAL_EXISTING},
        "local_generated_stage_upload_standins": args.mode == MODE_LOCAL_GENERATED,
        "not_production": True,
        "not_p2p": True,
        "not_gpu_tpu_pooling": True,
        "not_gguf_llamacpp_serving": True,
        "not_large_model_serving": True,
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
        report["safety_error"] = "real LLM live RC report contained secret-like fragments"
    json_path = output_dir / "real_llm_live_rc.json"
    md_path = output_dir / "real_llm_live_rc.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    if "artifacts" in report:
        report["artifacts"]["real_llm_live_rc_json"]["present"] = True
        report["artifacts"]["real_llm_live_rc_markdown"]["present"] = True
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_kaggle_generated(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    generated, secret_values = prepare_generated_artifacts(args, output_dir=output_dir)
    packages = stage_package_summary(output_dir)
    package_ok = all(stage_package_ready(package) for package in packages)
    codes = ["real_llm_artifact_ready"] if generated.get("artifact", {}).get("artifact_hash") else []
    if package_ok:
        codes.extend(["kaggle_real_llm_stage_upload_packages_ready", "real_llm_live_rc_prepare_ready"])
    else:
        codes.append("real_llm_live_rc_prepare_blocked")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(package_ok),
        "mode": MODE_KAGGLE_GENERATED,
        "output_dir": str(output_dir),
        "coordinator_url": effective_coordinator_url(args),
        "public_host": args.public_host,
        "port": args.port,
        "miner_id": args.miner_id,
        "workload": {
            "workload_type": WORKLOAD_TYPE,
            "stage_mode": "split",
            "request_count": args.request_count,
            "max_new_tokens": args.max_new_tokens,
            "hf_model_id": args.hf_model_id,
            "real_llm_partition_mode": args.real_llm_partition_mode,
            "prompt_text_count": len(DEFAULT_PROMPTS),
            "require_distinct_stage_miners": True,
        },
        "artifact": generated.get("artifact") or {},
        "runtime_classification": {
            "local_generated_stage_upload_standins": False,
            "external_runtime_verified": False,
            "kaggle_notebook_verified": False,
            "preparation_only": True,
        },
        "stage_packages": packages,
        "diagnosis_codes": sorted(set(codes)),
        "artifacts": base_artifacts(output_dir, ok=package_ok),
        "safety": safety_summary(args),
        "operator_action": [
            "Start start_coordinator.sh on the public Coordinator host.",
            "Upload kaggle-upload-real-llm-stage0 and kaggle-upload-real-llm-stage1 to two private Kaggle CPU Notebooks.",
            "Run python -m pip install -e '.[hf]' and python kaggle_remote_miner.py --env-file miner.private.env in each Notebook.",
            "Then verify with crowdtensor real-llm-live-rc --mode external-existing.",
        ],
        "limitations": [
            "kaggle-generated prepares artifacts only; it is not live external runtime evidence.",
            "CPU-only read-only tiny Hugging Face GPT split proof; not production Swarm Inference.",
            "No P2P routing, NAT traversal, GPU/TPU pooling, GGUF/llama.cpp serving, large-model serving, training, or arbitrary prompt serving.",
        ],
    }
    return persist_report(report, output_dir=output_dir, secret_values=secret_values)


def build_local_generated(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    runner: Runner,
    popen_factory: PopenFactory,
) -> dict[str, Any]:
    coordinator_url = effective_coordinator_url(args)
    steps: list[dict[str, Any]] = []
    processes: list[dict[str, Any]] = []
    coordinator_process: subprocess.Popen[str] | None = None
    stage_processes: list[tuple[str, subprocess.Popen[str], list[str]]] = []
    verify_payload: dict[str, Any] = {}
    generated: dict[str, Any] = {}
    secret_values: list[str] = []
    try:
        generated, secret_values = prepare_generated_artifacts(args, output_dir=output_dir)
        operator_env = parse_private_env(remote_runtime_dir(output_dir) / "operator.private.env")
        observer_token = operator_env.get("CROWDTENSOR_OBSERVER_TOKEN", "")
        admin_token = operator_env.get("CROWDTENSOR_ADMIN_TOKEN", "")
        coordinator_command = ["bash", str(output_dir / "start_coordinator.sh")]
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
                output_dir=output_dir,
                popen_factory=popen_factory,
            )
            stage_processes.append((stage_role, process, command))
            steps.append({
                "name": f"local_generated_real_llm_{stage_role}_miner_start",
                "ok": True,
                "pid": process.pid,
                "miner_id": stage_miner_id(args.miner_id, stage_role),
            })

        verify_step, verify_payload = run_verify(
            args,
            output_dir=output_dir,
            coordinator_url=coordinator_url,
            observer_token=observer_token,
            admin_token=admin_token,
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
                command=["bash", str(output_dir / "start_coordinator.sh")],
                timeout=args.process_exit_timeout,
                secret_values=secret_values,
                terminate=True,
            ))

    packages = stage_package_summary(output_dir)
    process_ok = all(item.get("ok") for item in processes if item.get("name", "").endswith("_miner"))
    package_ok = all(stage_package_ready(package) for package in packages)
    codes = diagnosis_codes(verify_payload)
    if (generated.get("artifact") or {}).get("artifact_hash"):
        codes.append("real_llm_artifact_ready")
    if package_ok:
        codes.append("local_generated_real_llm_stage_upload_packages_ready")
    if process_ok and package_ok and "remote_real_llm_sharded_ready" in codes:
        codes.append("local_generated_real_llm_stage_upload_standins_ready")
    if "remote_real_llm_sharded_ready" in codes and "local_generated_real_llm_stage_upload_standins_ready" in codes:
        codes.append("real_llm_live_rc_ready")
    else:
        codes.append("real_llm_live_rc_blocked")
    ok = "real_llm_live_rc_ready" in codes and package_ok and process_ok
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": MODE_LOCAL_GENERATED,
        "output_dir": str(output_dir),
        "coordinator_url": coordinator_url,
        "public_host": args.public_host,
        "port": args.port,
        "miner_id": args.miner_id,
        "workload": {
            "workload_type": WORKLOAD_TYPE,
            "stage_mode": "split",
            "request_count": args.request_count,
            "max_new_tokens": args.max_new_tokens,
            "hf_model_id": args.hf_model_id,
            "real_llm_partition_mode": args.real_llm_partition_mode,
            "prompt_text_count": len(DEFAULT_PROMPTS),
            "require_distinct_stage_miners": True,
        },
        "artifact": generated.get("artifact") or {},
        "runtime_classification": {
            "local_generated_stage_upload_standins": True,
            "external_runtime_verified": False,
            "kaggle_notebook_verified": False,
        },
        "stage_packages": packages,
        "process_summary": processes,
        "steps": steps,
        "payload_summaries": {
            "verify": summarize_payload(verify_payload),
            "remote_real_llm_beta": summarize_payload(load_json(remote_runtime_dir(output_dir) / "remote_real_llm_sharded_beta.json")),
        },
        "diagnosis_codes": sorted(set(codes)),
        "artifacts": base_artifacts(output_dir, ok=ok),
        "safety": safety_summary(args),
        "operator_action": [
            "Use local-generated as the mandatory CI-safe stand-in before attempting two real external stage Miners.",
            "For real Kaggle or two-machine proof, start the generated Coordinator and stage packages, then rerun --mode external-existing.",
        ],
        "limitations": [
            "local-generated starts local processes from generated stage packages; it is not live Kaggle Notebook evidence.",
            "CPU-only read-only tiny Hugging Face GPT split proof; not production Swarm Inference.",
            "No P2P routing, NAT traversal, GPU/TPU pooling, GGUF/llama.cpp serving, large-model serving, training, or arbitrary prompt serving.",
        ],
    }
    return persist_report(report, output_dir=output_dir, secret_values=secret_values)


def build_external_existing(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    coordinator_url = effective_coordinator_url(args)
    secret_values = [args.observer_token, args.admin_token]
    artifact_summary = inspect_real_llm_artifact(
        model_id=args.hf_model_id,
        cache_dir=args.hf_cache_dir,
        backend=args.real_llm_backend,
        require_runtime=args.real_llm_backend != REAL_LLM_BACKEND_CUDA,
    )
    artifact_summary["partition_mode"] = args.real_llm_partition_mode
    verify_step, verify_payload = run_verify(
        args,
        output_dir=output_dir,
        coordinator_url=coordinator_url,
        observer_token=args.observer_token,
        admin_token=args.admin_token,
        runner=runner,
        secret_values=secret_values,
    )
    codes = diagnosis_codes(verify_payload)
    if "stage_0_accepted" in codes:
        codes.append("kaggle_real_llm_stage0_seen")
    if "stage_1_accepted" in codes:
        codes.append("kaggle_real_llm_stage1_seen")
    if verify_payload.get("ok") and "remote_real_llm_sharded_ready" in codes:
        codes.extend(["external_runtime_verified", "kaggle_real_llm_sharded_ready", "real_llm_live_rc_ready"])
    else:
        codes.extend(["external_runtime_blocked", "real_llm_live_rc_blocked"])
    ok = "real_llm_live_rc_ready" in codes and bool(verify_step.get("ok"))
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": MODE_EXTERNAL_EXISTING,
        "output_dir": str(output_dir),
        "coordinator_url": coordinator_url,
        "public_host": args.public_host,
        "port": args.port,
        "miner_id": args.miner_id,
        "workload": {
            "workload_type": WORKLOAD_TYPE,
            "stage_mode": "split",
            "request_count": args.request_count,
            "max_new_tokens": args.max_new_tokens,
            "hf_model_id": args.hf_model_id,
            "real_llm_partition_mode": args.real_llm_partition_mode,
            "prompt_text_count": len(DEFAULT_PROMPTS),
            "require_distinct_stage_miners": True,
        },
        "artifact": artifact_summary,
        "runtime_classification": {
            "local_generated_stage_upload_standins": False,
            "external_runtime_verified": ok,
            "kaggle_notebook_verified": ok,
        },
        "stage_packages": stage_package_summary(output_dir),
        "steps": [verify_step],
        "payload_summaries": {
            "verify": summarize_payload(verify_payload),
            "remote_real_llm_beta": summarize_payload(load_json(remote_runtime_dir(output_dir) / "remote_real_llm_sharded_beta.json")),
        },
        "diagnosis_codes": sorted(set(codes)),
        "artifacts": base_artifacts(output_dir, ok=ok),
        "safety": safety_summary(args),
        "operator_action": [
            "If external_runtime_blocked is present, confirm stage0 and stage1 Miners are running with --enable-hf-tiny-gpt-runtime and distinct --real-llm-stage-role values.",
            "Rotate tokens after any temporary HTTP public Coordinator run.",
        ],
        "limitations": [
            "external-existing verifies already running external Miners; it does not create Kaggle Notebooks.",
            "CPU-only read-only tiny Hugging Face GPT split proof; not production Swarm Inference.",
            "No P2P routing, NAT traversal, GPU/TPU pooling, GGUF/llama.cpp serving, large-model serving, training, or arbitrary prompt serving.",
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
    if args.mode == MODE_LOCAL_GENERATED:
        return build_local_generated(args, output_dir=output_dir, runner=runner, popen_factory=popen_factory)
    if args.mode == MODE_KAGGLE_GENERATED:
        return build_kaggle_generated(args, output_dir=output_dir)
    return build_external_existing(args, output_dir=output_dir, runner=runner)


def render_markdown(report: dict[str, Any]) -> str:
    runtime = report.get("runtime_classification") or {}
    workload = report.get("workload") or {}
    lines = [
        "# CrowdTensor Real Small-LLM Sharded Inference Live RC",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- coordinator_url: `{report.get('coordinator_url')}`",
        f"- request_count: `{workload.get('request_count')}`",
        f"- hf_model_id: `{workload.get('hf_model_id')}`",
        f"- partition_mode: `{workload.get('real_llm_partition_mode')}`",
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
            f"script=`{package.get('miner_script_present')}` role=`{package.get('launcher_has_stage_role')}` "
            f"hf=`{package.get('hf_runtime_enabled')}` syntax=`{package.get('launcher_syntax_valid')}`"
        )
    lines.extend(["", "## Artifacts", ""])
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        lines.append(f"- `{name}`: `{artifact.get('path')}` present=`{artifact.get('present')}`")
    lines.extend(["", "## Boundaries", ""])
    for limitation in report.get("limitations") or []:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the real small-LLM live two-node RC report.")
    parser.add_argument("--mode", choices=[MODE_LOCAL_GENERATED, MODE_KAGGLE_GENERATED, MODE_EXTERNAL_EXISTING], default=MODE_LOCAL_GENERATED)
    parser.add_argument("--output-dir", default="dist/real-llm-live-rc")
    parser.add_argument("--public-host", default=DEFAULT_PUBLIC_HOST)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--miner-id", default="kaggle-real-llm")
    parser.add_argument("--request-count", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--hf-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument(
        "--real-llm-backend",
        choices=["hf_transformers_cpu", "hf_transformers_cuda", "cpu", "cuda", "auto"],
        default=REAL_LLM_BACKEND_CPU,
    )
    parser.add_argument("--real-llm-partition-mode", choices=["full", "stage-local", "stage_local"], default="full")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--remote-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--process-exit-timeout", type=float, default=20.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--http-timeout", type=float, default=30.0)
    parser.add_argument("--lease-seconds", type=float, default=15.0)
    parser.add_argument("--compute-seconds", type=float, default=0.2)
    parser.add_argument("--heartbeat-interval", type=float, default=0.1)
    parser.add_argument("--idle-sleep", type=float, default=0.5)
    parser.add_argument("--max-request-attempts", type=int, default=120)
    parser.add_argument("--observer-token", default="")
    parser.add_argument("--admin-token", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.port < 1:
        raise SystemExit("--port must be positive")
    if args.request_count < 1 or args.request_count > 4:
        raise SystemExit("--request-count must be between 1 and 4")
    if args.max_new_tokens < 1 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 1 and 32")
    args.real_llm_backend = normalize_real_llm_backend(args.real_llm_backend)
    args.real_llm_partition_mode = normalize_real_llm_partition_mode(args.real_llm_partition_mode)
    for name in [
        "timeout_seconds",
        "remote_timeout_seconds",
        "startup_timeout",
        "process_exit_timeout",
        "poll_interval",
        "http_timeout",
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
    if args.mode == MODE_EXTERNAL_EXISTING:
        missing = [
            name for name in ["coordinator_url", "observer_token", "admin_token"]
            if not getattr(args, name)
        ]
        if missing:
            raise SystemExit(f"external-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
    args.coordinator_url = args.coordinator_url.rstrip("/") if args.coordinator_url else ""
    return args


def print_human(report: dict[str, Any]) -> None:
    print("CrowdTensor real small-LLM live two-node RC")
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
