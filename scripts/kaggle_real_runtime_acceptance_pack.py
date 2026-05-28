#!/usr/bin/env python3
"""Prepare and verify a real Kaggle CPU Miner against a public Coordinator."""

from __future__ import annotations

import argparse
import json
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import support_bundle  # noqa: E402
from crowdtensor.auth import hash_token  # noqa: E402
from crowdtensor.micro_llm_artifact import build_default_micro_llm_artifact, inspect_micro_llm_artifact  # noqa: E402
from create_miner_invite import create_invite  # noqa: E402


SCHEMA = "kaggle_real_runtime_acceptance_v1"
MODEL_BUNDLE_KIND = "model-bundle"
MICRO_LLM_SHARDED_KIND = "micro-llm-sharded"
WORKLOAD_CHOICES = [MODEL_BUNDLE_KIND, MICRO_LLM_SHARDED_KIND]
MODEL_BUNDLE_WORKLOAD_TYPE = "model_bundle_infer"
MODEL_BUNDLE_ROUTE_NAME = "remote_python_model_bundle_infer"
MICRO_LLM_SHARDED_WORKLOAD_TYPE = "micro_llm_sharded_infer"
MICRO_LLM_SHARDED_ROUTE_NAME = "remote_python_micro_llm_sharded_infer"
DEFAULT_PUBLIC_HOST = "24.199.118.54"
DEFAULT_PORT = 9180
Runner = Callable[..., subprocess.CompletedProcess[str]]

SECRET_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "lease_token",
    "idempotency_key",
    "inference_results",
    "output_text",
    "Bearer ",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def coordinator_url_for(args: argparse.Namespace) -> str:
    if getattr(args, "coordinator_url", ""):
        return str(args.coordinator_url).rstrip("/")
    return f"http://{args.public_host}:{args.port}"


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
        step["ok"] = bool(step.get("ok") and payload.get("ok") is not False)
    if not step["ok"]:
        if completed.stdout and not payload:
            step["stdout_tail"] = redact_text(completed.stdout[-1200:], secret_values)
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1200:], secret_values)
    return step, payload


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(payload), encoding="utf-8")


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


def secret_values_from_envs(*paths: Path) -> list[str]:
    values: list[str] = []
    for path in paths:
        for value in parse_private_env(path).values():
            if value:
                values.append(value)
    return values


def token_or_generated(value: str) -> str:
    return value or secrets.token_urlsafe(32)


def quote_env(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def write_private_env(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"export {key}={quote_env(value)}" for key, value in sorted(values.items()) if value]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def remote_demo_dir(output_dir: Path) -> Path:
    return output_dir / "remote-home-compute"


def upload_dir(output_dir: Path) -> Path:
    return output_dir / "kaggle-upload"


def stage_upload_dir(output_dir: Path, stage_role: str) -> Path:
    return output_dir / f"kaggle-upload-{stage_role}"


def coordinator_state_dir(output_dir: Path) -> Path:
    return output_dir / "coordinator-state"


def workload_kind_for(args: argparse.Namespace) -> str:
    return str(getattr(args, "workload", MODEL_BUNDLE_KIND) or MODEL_BUNDLE_KIND)


def workload_type_for(kind: str) -> str:
    return MICRO_LLM_SHARDED_WORKLOAD_TYPE if kind == MICRO_LLM_SHARDED_KIND else MODEL_BUNDLE_WORKLOAD_TYPE


def route_name_for(kind: str) -> str:
    return MICRO_LLM_SHARDED_ROUTE_NAME if kind == MICRO_LLM_SHARDED_KIND else MODEL_BUNDLE_ROUTE_NAME


def is_micro_llm_split(args: argparse.Namespace) -> bool:
    return workload_kind_for(args) == MICRO_LLM_SHARDED_KIND and str(getattr(args, "stage_mode", "both")) == "split"


def stage_miner_id(base_miner_id: str, stage_role: str) -> str:
    base = str(base_miner_id or "kaggle-cpu-1").strip()
    suffix = f"-{stage_role}"
    return base if base.endswith(suffix) else f"{base}{suffix}"


def remote_pack_miner_id(args: argparse.Namespace) -> str:
    return stage_miner_id(args.miner_id, "stage0") if is_micro_llm_split(args) else args.miner_id


def upload_summary_ready(upload_summary: dict[str, Any] | None) -> bool:
    if not upload_summary:
        return False
    if all(upload_summary.get(key) for key in ["miner_env_present", "miner_script_present", "runbook_present", "operator_env_excluded"]):
        return True
    packages = upload_summary.get("stage_packages")
    if not isinstance(packages, list) or not packages:
        return False
    return all(
        isinstance(package, dict)
        and package.get("miner_env_present")
        and package.get("miner_script_present")
        and package.get("runbook_present")
        and package.get("operator_env_excluded")
        for package in packages
    )


def render_coordinator_launch(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    observer_hash: str,
    admin_hash: str,
) -> str:
    state_dir = coordinator_state_dir(output_dir)
    registry = remote_demo_dir(output_dir) / "miner_registry.json"
    kind = workload_kind_for(args)
    micro_artifact = str(getattr(args, "micro_llm_artifact", "") or "")
    lane_backlog = 0 if kind == MICRO_LLM_SHARDED_KIND else args.backlog
    lane = f"python-cli:cpu:{lane_backlog}:{workload_type_for(kind)}"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(ROOT))}",
        "python3 coordinator.py \\",
        f"  --host {shlex.quote(args.bind_host)} \\",
        f"  --port {args.port} \\",
        f"  --state-dir {shlex.quote(str(state_dir))} \\",
        f"  --lease-seconds {args.lease_seconds} \\",
        f"  --inner-steps {args.request_count} \\",
        "  --backlog 0 \\",
        f"  --task-lane {shlex.quote(lane)} \\",
    ]
    if micro_artifact:
        lines.append(f"  --micro-llm-artifact {shlex.quote(micro_artifact)} \\")
    lines.extend([
        f"  --miner-token-registry {shlex.quote(str(registry))} \\",
        f"  --observer-token {shlex.quote(observer_hash)} \\",
        f"  --admin-token {shlex.quote(admin_hash)}",
        "",
    ])
    return "\n".join(lines)


def render_operator_commands(args: argparse.Namespace, *, output_dir: Path) -> str:
    coordinator_url = coordinator_url_for(args)
    kind = workload_kind_for(args)
    workload_type = workload_type_for(kind)
    if is_micro_llm_split(args):
        upload_step = "\n".join([
            f"Upload `{stage_upload_dir(output_dir, 'stage0')}` to one private Kaggle Notebook.",
            f"Upload `{stage_upload_dir(output_dir, 'stage1')}` to a second private Kaggle Notebook.",
            "Do not upload `operator.private.env` or `miner_registry.json` to either Notebook.",
        ])
        miner_step = "\n".join([
            "Run this in each Kaggle Notebook working directory:",
            "",
            "```bash",
            "python -m pip install -e .",
            "python kaggle_remote_miner.py --env-file miner.private.env",
            "```",
        ])
    else:
        upload_step = "\n".join([
            f"Upload only files from `{upload_dir(output_dir)}` to a private Kaggle Notebook input or working directory.",
            "Do not upload `operator.private.env`.",
        ])
        miner_step = "\n".join([
            "```bash",
            "python -m pip install -e .",
            "python kaggle_remote_miner.py --env-file miner.private.env",
            "```",
        ])
    verify_command = (
        f"crowdtensor remote-demo kaggle-real --action verify --workload {kind} "
        f"--public-host {args.public_host} --port {args.port} --output-dir {output_dir}"
    )
    if kind == MICRO_LLM_SHARDED_KIND:
        verify_command += f" --decode-steps {getattr(args, 'decode_steps', 4)} --stage-mode {getattr(args, 'stage_mode', 'both')}"
        if getattr(args, "micro_llm_artifact", ""):
            verify_command += f" --micro-llm-artifact {shlex.quote(str(args.micro_llm_artifact))}"
        if getattr(args, "prompt_texts", ""):
            verify_command += f" --prompt-texts {shlex.quote(str(args.prompt_texts))}"
        if getattr(args, "require_distinct_stage_miners", False):
            verify_command += " --require-distinct-stage-miners"
    verify_command += " --json"
    return "\n".join([
        "# CrowdTensor Kaggle Real Runtime Acceptance",
        "",
        f"Coordinator URL: `{coordinator_url}`",
        f"Miner ID: `{args.miner_id}`",
        f"Workload: `{workload_type}`",
        f"Stage mode: `{getattr(args, 'stage_mode', 'both')}`" if kind == MICRO_LLM_SHARDED_KIND else "",
        "",
        "## 1. Start the public Coordinator",
        "",
        "```bash",
        f"bash {output_dir / 'start_coordinator.sh'}",
        "```",
        "",
        "## 2. Upload Kaggle-only files",
        "",
        upload_step,
        "",
        "## 3. Run the Kaggle Miner",
        "",
        miner_step,
        "",
        "## 4. Verify from the operator host",
        "",
        "```bash",
        verify_command,
        "```",
        "",
        "## Boundaries",
        "",
        "- This uses temporary HTTP to the operator-owned public Coordinator URL.",
        "- Rotate generated tokens after the run.",
        f"- CPU-only read-only `{workload_type}`; not production Swarm Inference, not P2P, and not GPU/TPU workload routing.",
        "- Micro-LLM split mode is a deterministic two-stage toy pipeline proof, not large-model sharding or GGUF/llama.cpp serving." if kind == MICRO_LLM_SHARDED_KIND else "",
        "",
    ])


def render_kaggle_runbook(args: argparse.Namespace, *, output_dir: Path, stage_role: str = "") -> str:
    coordinator_url = coordinator_url_for(args)
    kind = workload_kind_for(args)
    workload_type = workload_type_for(kind)
    miner_id = stage_miner_id(args.miner_id, stage_role) if stage_role else args.miner_id
    return "\n".join([
        "# CrowdTensor Kaggle Real Runtime Miner",
        "",
        f"Coordinator URL: `{coordinator_url}`",
        f"Miner ID: `{miner_id}`",
        f"Workload: `{workload_type}`",
        f"Stage role: `{stage_role}`" if stage_role else "",
        "",
        "## Notebook Commands",
        "",
        "```bash",
        "python -m pip install -e .",
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
        "Kaggle is only an outbound temporary CPU Miner. This is not production P2P, GPU/TPU pooling, or arbitrary prompt serving.",
        "For micro-LLM split mode, this Notebook runs one deterministic toy pipeline stage only; it is not large-model sharding." if stage_role else "",
        "",
    ])


def render_stage_kaggle_miner(args: argparse.Namespace, *, miner_id: str, stage_role: str) -> str:
    coordinator_url = coordinator_url_for(args)
    stage_lines = ""
    if stage_role:
        stage_lines = f'    "--micro-llm-stage-role",\n    "{stage_role}",\n'
    return f'''#!/usr/bin/env python3
"""Kaggle Remote Miner launcher for CrowdTensor {workload_type_for(workload_kind_for(args))}."""

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
env["CROWDTENSOR_REMOTE_ENVIRONMENT"] = "kaggle"
env.setdefault("PYTHONUNBUFFERED", "1")
command = [
    "crowdtensor-miner",
    "--coordinator",
    "{coordinator_url}",
    "--miner-id",
    "{miner_id}",
    "--max-tasks",
    "1",
    "--compute-seconds",
    "0.2",
    "--heartbeat-interval",
    "0.1",
{stage_lines}    "--max-request-attempts",
    "120",
    "--idle-sleep",
    "1.0",
]
print("Starting Kaggle remote Miner:", " ".join(command), flush=True)
raise SystemExit(subprocess.call(command, env=env))
'''


def copy_kaggle_upload_files(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    remote_dir = remote_demo_dir(output_dir)
    if is_micro_llm_split(args):
        packages: list[dict[str, Any]] = []
        for stage_role in ["stage0", "stage1"]:
            upload = stage_upload_dir(output_dir, stage_role)
            upload.mkdir(parents=True, exist_ok=True)
            miner_id = stage_miner_id(args.miner_id, stage_role)
            miner_env = upload / "miner.private.env"
            source_env = remote_dir / ("miner.private.env" if stage_role == "stage0" else "miner.stage1.private.env")
            if source_env.is_file():
                shutil.copy2(source_env, miner_env)
            script = upload / "kaggle_remote_miner.py"
            script.write_text(render_stage_kaggle_miner(args, miner_id=miner_id, stage_role=stage_role), encoding="utf-8")
            script.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            runbook = upload / "KAGGLE_RUN.md"
            runbook.write_text(render_kaggle_runbook(args, output_dir=output_dir, stage_role=stage_role), encoding="utf-8")
            operator_env_in_upload = (upload / "operator.private.env").exists()
            packages.append({
                "stage_role": stage_role,
                "miner_id": miner_id,
                "path": upload.as_posix(),
                "miner_env_present": miner_env.is_file(),
                "miner_script_present": script.is_file(),
                "runbook_present": runbook.is_file(),
                "operator_env_excluded": not operator_env_in_upload,
            })
        return {
            "mode": "split",
            "stage_packages": packages,
            "operator_env_excluded": all(package.get("operator_env_excluded") for package in packages),
        }

    upload = upload_dir(output_dir)
    upload.mkdir(parents=True, exist_ok=True)
    for name in ["miner.private.env", "kaggle_remote_miner.py"]:
        source = remote_dir / name
        if source.is_file():
            shutil.copy2(source, upload / name)
    runbook = upload / "KAGGLE_RUN.md"
    runbook.write_text(render_kaggle_runbook(args, output_dir=output_dir), encoding="utf-8")
    operator_env_in_upload = (upload / "operator.private.env").exists()
    return {
        "path": upload.as_posix(),
        "miner_env_present": (upload / "miner.private.env").is_file(),
        "miner_script_present": (upload / "kaggle_remote_miner.py").is_file(),
        "runbook_present": runbook.is_file(),
        "operator_env_excluded": not operator_env_in_upload,
    }


def base_artifacts(output_dir: Path, *, ok: bool | None = None) -> dict[str, Any]:
    remote_dir = remote_demo_dir(output_dir)
    upload = upload_dir(output_dir)
    stage0_upload = stage_upload_dir(output_dir, "stage0")
    stage1_upload = stage_upload_dir(output_dir, "stage1")
    return {
        "kaggle_real_runtime_acceptance_json": artifact_entry(
            output_dir / "kaggle_real_runtime_acceptance.json",
            output_dir,
            kind="kaggle_real_runtime_acceptance",
            schema=SCHEMA,
            ok=ok,
        ),
        "kaggle_real_runtime_acceptance_markdown": artifact_entry(
            output_dir / "kaggle_real_runtime_acceptance.md",
            output_dir,
            kind="kaggle_real_runtime_acceptance_markdown",
        ),
        "operator_commands": artifact_entry(output_dir / "operator_commands.md", output_dir, kind="operator_commands"),
        "coordinator_launch_script": artifact_entry(output_dir / "start_coordinator.sh", output_dir, kind="coordinator_launch_script"),
        "remote_home_compute_demo_json": artifact_entry(remote_dir / "remote_home_compute_demo.json", output_dir, kind="remote_home_compute_demo", schema="remote_home_compute_demo_v1"),
        "remote_home_compute_doctor_json": artifact_entry(remote_dir / "remote_home_compute_doctor.json", output_dir, kind="remote_home_compute_doctor", schema="remote_home_compute_doctor_v1"),
        "remote_home_compute_collect_json": artifact_entry(remote_dir / "remote_home_compute_collect.json", output_dir, kind="remote_home_compute_collect", schema="remote_home_compute_collect_v1"),
        "remote_demo_acceptance_json": artifact_entry(remote_dir / "remote_demo_acceptance.json", output_dir, kind="remote_demo_acceptance", schema="remote_demo_acceptance_v1"),
        "remote_compute_evidence_json": artifact_entry(remote_dir / "remote_compute_evidence.json", output_dir, kind="remote_compute_evidence", schema="remote_compute_evidence_v1"),
        "support_bundle_json": artifact_entry(remote_dir / "support_bundle.json", output_dir, kind="support_bundle", schema="support_bundle_v1"),
        "operator_private_env": artifact_entry(remote_dir / "operator.private.env", output_dir, kind="private_env"),
        "miner_private_env": artifact_entry(remote_dir / "miner.private.env", output_dir, kind="private_env"),
        "miner_registry": artifact_entry(remote_dir / "miner_registry.json", output_dir, kind="miner_registry"),
        "kaggle_upload_miner_env": artifact_entry(upload / "miner.private.env", output_dir, kind="kaggle_upload_private_env"),
        "kaggle_upload_miner_script": artifact_entry(upload / "kaggle_remote_miner.py", output_dir, kind="kaggle_upload_miner_script"),
        "kaggle_upload_runbook": artifact_entry(upload / "KAGGLE_RUN.md", output_dir, kind="kaggle_upload_runbook"),
        "kaggle_upload_stage0_miner_env": artifact_entry(stage0_upload / "miner.private.env", output_dir, kind="kaggle_upload_private_env"),
        "kaggle_upload_stage0_miner_script": artifact_entry(stage0_upload / "kaggle_remote_miner.py", output_dir, kind="kaggle_upload_miner_script"),
        "kaggle_upload_stage0_runbook": artifact_entry(stage0_upload / "KAGGLE_RUN.md", output_dir, kind="kaggle_upload_runbook"),
        "kaggle_upload_stage1_miner_env": artifact_entry(stage1_upload / "miner.private.env", output_dir, kind="kaggle_upload_private_env"),
        "kaggle_upload_stage1_miner_script": artifact_entry(stage1_upload / "kaggle_remote_miner.py", output_dir, kind="kaggle_upload_miner_script"),
        "kaggle_upload_stage1_runbook": artifact_entry(stage1_upload / "KAGGLE_RUN.md", output_dir, kind="kaggle_upload_runbook"),
        "remote_micro_llm_sharded_beta_json": artifact_entry(remote_dir / "remote_micro_llm_sharded_beta.json", output_dir, kind="remote_micro_llm_sharded_beta", schema="remote_micro_llm_sharded_beta_v1"),
        "remote_micro_llm_sharded_acceptance_json": artifact_entry(remote_dir / "remote_micro_llm_sharded_acceptance.json", output_dir, kind="remote_micro_llm_sharded_acceptance", schema="remote_micro_llm_sharded_acceptance_v1"),
    }


def summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "diagnosis_codes": payload.get("diagnosis_codes") or [],
    }
    for key in ["coordinator_url", "miner_id", "workload_kind", "target"]:
        if payload.get(key) is not None:
            summary[key] = payload.get(key)
    if isinstance(payload.get("connectivity"), dict):
        connectivity = payload["connectivity"]
        summary["connectivity"] = {
            "lane_visible": connectivity.get("lane_visible"),
            "status_ready": connectivity.get("status_ready"),
            "matched_capabilities": connectivity.get("matched_capabilities") or [],
            "missing_capabilities": connectivity.get("missing_capabilities") or [],
        }
    if isinstance(payload.get("acceptance_summary"), dict):
        summary["acceptance_summary"] = payload.get("acceptance_summary")
    if isinstance(payload.get("stage_assignment"), dict):
        summary["stage_assignment"] = payload.get("stage_assignment")
    if isinstance(payload.get("status_summary"), dict):
        summary["status_summary"] = payload.get("status_summary")
    return summary


def micro_llm_stage_assignment_valid(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    assignment = payload.get("stage_assignment") if isinstance(payload.get("stage_assignment"), dict) else {}
    if assignment.get("stage_assignment_valid") is True:
        return True
    summaries = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
    for summary in summaries.values():
        if isinstance(summary, dict) and micro_llm_stage_assignment_valid(summary):
            return True
    return False


def diagnosis_from_payloads(
    *,
    prepare_payload: dict[str, Any] | None = None,
    doctor_payload: dict[str, Any] | None = None,
    verify_payload: dict[str, Any] | None = None,
    collect_payload: dict[str, Any] | None = None,
    upload_summary: dict[str, Any] | None = None,
    extra_payloads: list[dict[str, Any]] | None = None,
) -> list[str]:
    codes: set[str] = set()
    workload_kind = MODEL_BUNDLE_KIND
    for payload in [prepare_payload, doctor_payload, verify_payload, collect_payload, *(extra_payloads or [])]:
        if isinstance(payload, dict):
            codes.update(str(code) for code in payload.get("diagnosis_codes") or [] if isinstance(code, str))
            workload = payload.get("workload") if isinstance(payload.get("workload"), dict) else {}
            demo = payload.get("demo") if isinstance(payload.get("demo"), dict) else {}
            if payload.get("schema") == "remote_micro_llm_sharded_beta_v1":
                workload_kind = MICRO_LLM_SHARDED_KIND
            workload_kind = str(workload.get("kind") or demo.get("workload_kind") or payload.get("workload_kind") or workload_kind)
            summaries = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
            for summary in summaries.values():
                if isinstance(summary, dict):
                    codes.update(str(code) for code in summary.get("diagnosis_codes") or [] if isinstance(code, str))
            if micro_llm_stage_assignment_valid(payload):
                codes.add("stage_assignment_valid")
    if upload_summary and upload_summary_ready(upload_summary):
        codes.add("kaggle_artifacts_ready")
    if doctor_payload:
        connectivity = doctor_payload.get("connectivity") if isinstance(doctor_payload.get("connectivity"), dict) else {}
        observations = connectivity.get("observations") if isinstance(connectivity.get("observations"), dict) else {}
        health_ok = (observations.get("health") or {}).get("ok")
        ready_ok = (observations.get("ready") or {}).get("ok")
        state_ok = (observations.get("state") or {}).get("ok")
        if health_ok or ready_ok or state_ok or doctor_payload.get("ok"):
            codes.add("coordinator_public_ready")
        matched = set(str(item) for item in connectivity.get("matched_capabilities") or [])
        if {"runtime:python-cli", "backend:cpu", f"workload:{workload_type_for(workload_kind)}"}.issubset(matched):
            codes.add("kaggle_miner_seen")
        if workload_kind == MICRO_LLM_SHARDED_KIND and "stage_0_completed" in matched:
            codes.add("kaggle_micro_llm_stage0_seen")
        if workload_kind == MICRO_LLM_SHARDED_KIND and "stage_1_completed" in matched:
            codes.add("kaggle_micro_llm_stage1_seen")
    if verify_payload and verify_payload.get("ok"):
        codes.add("kaggle_result_accepted")
        codes.add("kaggle_miner_seen")
        codes.add("coordinator_public_ready")
        if workload_kind == MICRO_LLM_SHARDED_KIND:
            codes.add("kaggle_micro_llm_sharded_result_accepted")
            codes.add("kaggle_micro_llm_stage0_seen")
            codes.add("kaggle_micro_llm_stage1_seen")
            acceptance_summary = verify_payload.get("acceptance_summary") if isinstance(verify_payload.get("acceptance_summary"), dict) else {}
            if acceptance_summary.get("stage_assignment_valid") or "stage_assignment_valid" in codes:
                codes.add("kaggle_micro_llm_stage_assignment_valid")
    if collect_payload and collect_payload.get("ok"):
        codes.add("kaggle_result_accepted")
        if workload_kind == MICRO_LLM_SHARDED_KIND:
            codes.add("kaggle_micro_llm_sharded_result_accepted")
            acceptance_summary = collect_payload.get("acceptance_summary") if isinstance(collect_payload.get("acceptance_summary"), dict) else {}
            if acceptance_summary.get("stage_assignment_valid") or "stage_assignment_valid" in codes:
                codes.add("kaggle_micro_llm_stage_assignment_valid")
    if workload_kind == MICRO_LLM_SHARDED_KIND and "stage_assignment_valid" in codes:
        codes.add("kaggle_micro_llm_stage_assignment_valid")
    if {"kaggle_artifacts_ready", "coordinator_public_ready", "kaggle_miner_seen", "kaggle_result_accepted"}.issubset(codes):
        codes.add("kaggle_real_runtime_ready")
    if workload_kind == MICRO_LLM_SHARDED_KIND and {
        "kaggle_real_runtime_ready",
        "kaggle_micro_llm_stage0_seen",
        "kaggle_micro_llm_stage1_seen",
        "kaggle_micro_llm_sharded_result_accepted",
        "kaggle_micro_llm_stage_assignment_valid",
    }.issubset(codes):
        codes.add("kaggle_micro_llm_sharded_ready")
    elif doctor_payload is not None or verify_payload is not None or collect_payload is not None:
        codes.add("kaggle_runtime_blocked")
    return sorted(codes)


def safety_summary(args: argparse.Namespace) -> dict[str, Any]:
    coordinator_url = coordinator_url_for(args)
    kind = workload_kind_for(args)
    return {
        "temporary_http": coordinator_url.startswith("http://"),
        "temporary_http_boundary_confirmed": coordinator_url.startswith("http://"),
        "token_rotation_required": True,
        "public_http_not_production": coordinator_url.startswith("http://"),
        "operator_env_excluded_from_kaggle": True,
        "summary_excludes_plaintext_tokens": True,
        "read_only": True,
        "cpu_only_workload": True,
        "gpu_tpu_workload_enabled": False,
        "not_production": True,
        "not_p2p": True,
        "pipeline_sharded_toy_inference": kind == MICRO_LLM_SHARDED_KIND,
        "not_model_sharding": kind != MICRO_LLM_SHARDED_KIND,
        "not_large_model_sharding": True,
        "not_gguf_llamacpp_serving": True,
        "not_public_prompt_serving": True,
    }


def persist_report(report: dict[str, Any], output_dir: Path, secret_values: list[str]) -> dict[str, Any]:
    report = redact_values(report, secret_values)
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    leaks.extend(secret for secret in secret_values if secret and secret in encoded)
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"sensitive_output_detected"})
        report["safety_error"] = "kaggle real runtime report contained secret-like fragments"
    json_path = output_dir / "kaggle_real_runtime_acceptance.json"
    md_path = output_dir / "kaggle_real_runtime_acceptance.md"
    write_json(report, json_path)
    write_markdown(report, md_path)
    report["artifacts"] = base_artifacts(output_dir, ok=report.get("ok"))
    write_json(report, json_path)
    return report


def build_prepare(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = remote_demo_dir(output_dir)
    remote_dir.mkdir(parents=True, exist_ok=True)
    coordinator_url = coordinator_url_for(args)
    workload_kind = workload_kind_for(args)
    workload_type = workload_type_for(workload_kind)
    route_name = route_name_for(workload_kind)
    artifact_summary: dict[str, Any] = {}
    if workload_kind == MICRO_LLM_SHARDED_KIND:
        if not getattr(args, "micro_llm_artifact", ""):
            args.micro_llm_artifact = str(output_dir / "micro-llm-artifact")
            artifact_summary = build_default_micro_llm_artifact(args.micro_llm_artifact)
        else:
            artifact_summary = inspect_micro_llm_artifact(args.micro_llm_artifact)

    miner_token = token_or_generated(args.miner_token)
    stage1_miner_token = token_or_generated("")
    observer_token = token_or_generated(args.observer_token)
    admin_token = token_or_generated(args.admin_token)
    secret_values = [miner_token, stage1_miner_token, observer_token, admin_token]

    command = [
        sys.executable,
        str(ROOT / "scripts" / "remote_home_compute_demo_pack.py"),
        "prepare",
        "--workload",
        workload_kind,
        "--target",
        "kaggle",
        "--coordinator-url",
        coordinator_url,
        "--miner-id",
        remote_pack_miner_id(args),
        "--output-dir",
        str(remote_dir),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--decode-steps",
        str(getattr(args, "decode_steps", 4)),
        "--stage-role",
        "stage0" if is_micro_llm_split(args) else "both",
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--miner-token",
        miner_token,
        "--observer-token",
        observer_token,
        "--admin-token",
        admin_token,
        "--json",
    ]
    if workload_kind == MICRO_LLM_SHARDED_KIND:
        if getattr(args, "micro_llm_artifact", ""):
            command.extend(["--micro-llm-artifact", str(args.micro_llm_artifact)])
        if getattr(args, "prompt_texts", ""):
            command.extend(["--prompt-texts", str(args.prompt_texts)])
    if args.replace:
        command.append("--replace")
    step, prepare_payload = run_json_step(
        "kaggle_remote_prepare",
        command,
        runner=runner,
        timeout_seconds=args.timeout_seconds,
        secret_values=secret_values,
    )

    if is_micro_llm_split(args) and step.get("ok") and prepare_payload.get("ok"):
        registry_path = remote_dir / "miner_registry.json"
        stage1_miner_id = stage_miner_id(args.miner_id, "stage1")
        try:
            create_invite(
                registry_path=registry_path,
                miner_id=stage1_miner_id,
                coordinator_url=coordinator_url,
                label="Kaggle micro-LLM stage1 Miner",
                token=stage1_miner_token,
                replace=True,
            )
            write_private_env(remote_dir / "miner.stage1.private.env", {"CROWDTENSOR_MINER_TOKEN": stage1_miner_token})
        except Exception as exc:
            step["ok"] = False
            step["error"] = f"stage1_invite_failed: {exc}"
            prepare_payload["ok"] = False
            prepare_payload.setdefault("diagnosis_codes", []).append("kaggle_micro_llm_stage1_invite_failed")

    observer_hash = hash_token(observer_token)
    admin_hash = hash_token(admin_token)
    launch_path = output_dir / "start_coordinator.sh"
    launch_path.write_text(
        render_coordinator_launch(args=args, output_dir=output_dir, observer_hash=observer_hash, admin_hash=admin_hash),
        encoding="utf-8",
    )
    launch_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    operator_commands_path = output_dir / "operator_commands.md"
    operator_commands_path.write_text(render_operator_commands(args, output_dir=output_dir), encoding="utf-8")
    upload_summary = copy_kaggle_upload_files(args, output_dir=output_dir)

    ok = bool(step.get("ok") and prepare_payload.get("ok") and upload_summary.get("operator_env_excluded"))
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": "prepare",
        "output_dir": str(output_dir),
        "coordinator_url": coordinator_url,
        "public_host": args.public_host,
        "port": args.port,
        "miner_id": args.miner_id,
        "workload": {
            "kind": workload_kind,
            "workload_type": workload_type,
            "route": route_name,
            "request_count": args.request_count,
            "scenario_id": args.scenario_id,
            "decode_steps": getattr(args, "decode_steps", None) if workload_kind == MICRO_LLM_SHARDED_KIND else None,
            "stage_mode": getattr(args, "stage_mode", "both") if workload_kind == MICRO_LLM_SHARDED_KIND else None,
            "require_distinct_stage_miners": bool(getattr(args, "require_distinct_stage_miners", False)) if workload_kind == MICRO_LLM_SHARDED_KIND else None,
            "artifact_hash": artifact_summary.get("artifact_hash") if artifact_summary else None,
            "artifact_id": artifact_summary.get("artifact_id") if artifact_summary else None,
        },
        "artifact": artifact_summary,
        "step": step,
        "payloads": {"prepare": summarize_payload(prepare_payload)},
        "upload_package": upload_summary,
        "diagnosis_codes": diagnosis_from_payloads(prepare_payload=prepare_payload, upload_summary=upload_summary),
        "artifacts": base_artifacts(output_dir, ok=ok),
        "safety": safety_summary(args),
        "operator_action": [
            "Start the generated public Coordinator from start_coordinator.sh.",
            "Upload only the generated Kaggle upload package files to Kaggle; keep operator.private.env on the operator host.",
            "Run the Kaggle Notebook Miner, then run kaggle-real --action verify from the operator host.",
            "Rotate generated tokens after the temporary HTTP acceptance run.",
        ],
        "limitations": [
            "Preparation only; it does not prove that Kaggle has connected until verify reports kaggle_real_runtime_ready.",
            "Uses temporary HTTP for this operator-controlled acceptance run; not production public-internet hardening.",
            f"CPU-only read-only {workload_type}; not P2P, GPU/TPU pooling, training, or arbitrary prompt serving.",
            "micro-llm-sharded split mode is a toy two-stage pipeline proof, not production large-model sharding." if workload_kind == MICRO_LLM_SHARDED_KIND else "",
        ],
    }
    return persist_report(report, output_dir, secret_values)


def tokens_for_runtime(args: argparse.Namespace, output_dir: Path) -> tuple[str, str, list[str]]:
    operator_env = parse_private_env(remote_demo_dir(output_dir) / "operator.private.env")
    observer_token = args.observer_token or operator_env.get("CROWDTENSOR_OBSERVER_TOKEN", "")
    admin_token = args.admin_token or operator_env.get("CROWDTENSOR_ADMIN_TOKEN", "")
    secret_values = [observer_token, admin_token]
    secret_values.extend(secret_values_from_envs(remote_demo_dir(output_dir) / "miner.private.env"))
    secret_values.extend(secret_values_from_envs(remote_demo_dir(output_dir) / "miner.stage1.private.env"))
    secret_values.extend(secret_values_from_envs(stage_upload_dir(output_dir, "stage0") / "miner.private.env"))
    secret_values.extend(secret_values_from_envs(stage_upload_dir(output_dir, "stage1") / "miner.private.env"))
    return observer_token, admin_token, [secret for secret in secret_values if secret]


def extend_workload_command(command: list[str], args: argparse.Namespace) -> None:
    if workload_kind_for(args) != MICRO_LLM_SHARDED_KIND:
        return
    command.extend(["--decode-steps", str(getattr(args, "decode_steps", 4))])
    command.extend(["--stage-mode", str(getattr(args, "stage_mode", "both"))])
    if getattr(args, "micro_llm_artifact", ""):
        command.extend(["--micro-llm-artifact", str(args.micro_llm_artifact)])
    if getattr(args, "prompt_texts", ""):
        command.extend(["--prompt-texts", str(args.prompt_texts)])
    if getattr(args, "require_distinct_stage_miners", False):
        command.append("--require-distinct-stage-miners")


def upload_status_for(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    if is_micro_llm_split(args):
        packages: list[dict[str, Any]] = []
        for stage_role in ["stage0", "stage1"]:
            upload = stage_upload_dir(output_dir, stage_role)
            packages.append({
                "stage_role": stage_role,
                "miner_env_present": (upload / "miner.private.env").is_file(),
                "miner_script_present": (upload / "kaggle_remote_miner.py").is_file(),
                "runbook_present": (upload / "KAGGLE_RUN.md").is_file(),
                "operator_env_excluded": not (upload / "operator.private.env").exists(),
            })
        return {
            "mode": "split",
            "stage_packages": packages,
            "operator_env_excluded": all(package.get("operator_env_excluded") for package in packages),
        }
    return {
        "miner_env_present": (upload_dir(output_dir) / "miner.private.env").is_file(),
        "miner_script_present": (upload_dir(output_dir) / "kaggle_remote_miner.py").is_file(),
        "runbook_present": (upload_dir(output_dir) / "KAGGLE_RUN.md").is_file(),
        "operator_env_excluded": not (upload_dir(output_dir) / "operator.private.env").exists(),
    }


def build_doctor_step(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    observer_token: str,
    admin_token: str,
    runner: Runner,
) -> tuple[dict[str, Any], dict[str, Any]]:
    workload_kind = workload_kind_for(args)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "remote_home_compute_demo_pack.py"),
        "doctor",
        "--workload",
        workload_kind,
        "--coordinator-url",
        coordinator_url_for(args),
        "--miner-id",
        remote_pack_miner_id(args),
        "--observer-token",
        observer_token,
        "--admin-token",
        admin_token,
        "--output-dir",
        str(remote_demo_dir(output_dir)),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--http-timeout",
        str(args.http_timeout),
        "--admin-results-limit",
        str(args.admin_results_limit),
        "--json",
    ]
    extend_workload_command(command, args)
    if args.require_existing_result:
        command.append("--require-result")
    return run_json_step(
        "kaggle_real_runtime_doctor",
        command,
        runner=runner,
        timeout_seconds=args.timeout_seconds,
        secret_values=[observer_token, admin_token],
    )


def build_verify(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    workload_kind = workload_kind_for(args)
    workload_type = workload_type_for(workload_kind)
    route_name = route_name_for(workload_kind)
    observer_token, admin_token, secret_values = tokens_for_runtime(args, output_dir)
    if not observer_token or not admin_token:
        report = {
            "schema": SCHEMA,
            "generated_at": utc_now(),
            "ok": False,
            "mode": "verify",
            "output_dir": str(output_dir),
            "coordinator_url": coordinator_url_for(args),
            "miner_id": args.miner_id,
            "diagnosis_codes": ["operator_token_missing", "kaggle_runtime_blocked"],
            "artifacts": base_artifacts(output_dir, ok=False),
            "safety": safety_summary(args),
            "operator_action": ["Run kaggle-real --action prepare first, or pass --observer-token and --admin-token."],
            "limitations": ["Verification cannot run without operator observer/admin tokens."],
        }
        return persist_report(report, output_dir, secret_values)

    steps: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}

    doctor_step, doctor_payload = build_doctor_step(
        args,
        output_dir=output_dir,
        observer_token=observer_token,
        admin_token=admin_token,
        runner=runner,
    )
    steps.append(doctor_step)
    payloads["doctor"] = summarize_payload(doctor_payload)

    verify_command = [
        sys.executable,
        str(ROOT / "scripts" / "remote_home_compute_demo_pack.py"),
        "verify",
        "--workload",
        workload_kind,
        "--coordinator-url",
        coordinator_url_for(args),
        "--miner-id",
        remote_pack_miner_id(args),
        "--observer-token",
        observer_token,
        "--admin-token",
        admin_token,
        "--output-dir",
        str(remote_demo_dir(output_dir)),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
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
        "--create-session",
        "--json",
    ]
    extend_workload_command(verify_command, args)
    verify_step, verify_payload = run_json_step(
        "kaggle_real_runtime_verify",
        verify_command,
        runner=runner,
        timeout_seconds=args.timeout_seconds,
        secret_values=secret_values,
    )
    steps.append(verify_step)
    payloads["verify"] = summarize_payload(verify_payload)

    collect_payload: dict[str, Any] = {}
    if verify_payload.get("ok") or args.collect_on_failure:
        task_id = str(((verify_payload.get("acceptance_summary") or {}).get("task_id")) or "")
        collect_command = [
            sys.executable,
            str(ROOT / "scripts" / "remote_home_compute_demo_pack.py"),
            "collect",
            "--workload",
            workload_kind,
            "--coordinator-url",
            coordinator_url_for(args),
            "--miner-id",
            remote_pack_miner_id(args),
            "--observer-token",
            observer_token,
            "--admin-token",
            admin_token,
            "--output-dir",
            str(remote_demo_dir(output_dir)),
            "--request-count",
            str(args.request_count),
            "--scenario-id",
            args.scenario_id,
            "--http-timeout",
            str(args.http_timeout),
            "--artifact-timeout",
            str(args.artifact_timeout),
            "--admin-results-limit",
            str(args.admin_results_limit),
            "--json",
        ]
        extend_workload_command(collect_command, args)
        if task_id:
            collect_command.extend(["--task-id", task_id])
        collect_step, collect_payload = run_json_step(
            "kaggle_real_runtime_collect",
            collect_command,
            runner=runner,
            timeout_seconds=args.timeout_seconds,
            secret_values=secret_values,
        )
        steps.append(collect_step)
        payloads["collect"] = summarize_payload(collect_payload)

    micro_beta_payload = load_json(remote_demo_dir(output_dir) / "remote_micro_llm_sharded_beta.json") if workload_kind == MICRO_LLM_SHARDED_KIND else {}
    if micro_beta_payload:
        payloads["micro_llm_beta"] = summarize_payload(micro_beta_payload)

    diagnosis = diagnosis_from_payloads(
        doctor_payload=doctor_payload,
        verify_payload=verify_payload,
        collect_payload=collect_payload,
        upload_summary=upload_status_for(args, output_dir),
        extra_payloads=[micro_beta_payload] if micro_beta_payload else [],
    )
    ready_code = "kaggle_micro_llm_sharded_ready" if workload_kind == MICRO_LLM_SHARDED_KIND else "kaggle_real_runtime_ready"
    ok = ready_code in diagnosis and all(step.get("ok") for step in steps)
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": "verify",
        "output_dir": str(output_dir),
        "coordinator_url": coordinator_url_for(args),
        "public_host": args.public_host,
        "port": args.port,
        "miner_id": args.miner_id,
        "workload": {
            "kind": workload_kind,
            "workload_type": workload_type,
            "route": route_name,
            "request_count": args.request_count,
            "scenario_id": args.scenario_id,
            "decode_steps": getattr(args, "decode_steps", None) if workload_kind == MICRO_LLM_SHARDED_KIND else None,
            "stage_mode": getattr(args, "stage_mode", "both") if workload_kind == MICRO_LLM_SHARDED_KIND else None,
            "require_distinct_stage_miners": bool(getattr(args, "require_distinct_stage_miners", False)) if workload_kind == MICRO_LLM_SHARDED_KIND else None,
        },
        "steps": steps,
        "payloads": payloads,
        "diagnosis_codes": diagnosis,
        "artifacts": base_artifacts(output_dir, ok=ok),
        "safety": safety_summary(args),
        "operator_action": [
            "If kaggle_runtime_blocked is present, confirm the Coordinator is reachable from Kaggle over the public URL.",
            "Keep the Kaggle Miner Notebook running until kaggle_result_accepted appears.",
            "Rotate tokens after this temporary HTTP run.",
        ],
        "limitations": [
            "Real runtime verification requires a live Kaggle Notebook Miner to connect outbound to the public Coordinator.",
            "Temporary HTTP is accepted only for this controlled proof; production use needs TLS, VPN, or tunnel hardening.",
            "CPU-only read-only task-level inference; not model sharding, P2P routing, GPU/TPU pooling, training, or public prompt serving.",
        ],
    }
    return persist_report(report, output_dir, secret_values)


def build_collect(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    workload_kind = workload_kind_for(args)
    workload_type = workload_type_for(workload_kind)
    route_name = route_name_for(workload_kind)
    observer_token, admin_token, secret_values = tokens_for_runtime(args, output_dir)
    if not observer_token or not admin_token:
        report = {
            "schema": SCHEMA,
            "generated_at": utc_now(),
            "ok": False,
            "mode": "collect",
            "output_dir": str(output_dir),
            "coordinator_url": coordinator_url_for(args),
            "miner_id": args.miner_id,
            "diagnosis_codes": ["operator_token_missing", "kaggle_runtime_blocked"],
            "artifacts": base_artifacts(output_dir, ok=False),
            "safety": safety_summary(args),
        }
        return persist_report(report, output_dir, secret_values)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "remote_home_compute_demo_pack.py"),
        "collect",
        "--workload",
        workload_kind,
        "--coordinator-url",
        coordinator_url_for(args),
        "--miner-id",
        remote_pack_miner_id(args),
        "--observer-token",
        observer_token,
        "--admin-token",
        admin_token,
        "--output-dir",
        str(remote_demo_dir(output_dir)),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--http-timeout",
        str(args.http_timeout),
        "--artifact-timeout",
        str(args.artifact_timeout),
        "--admin-results-limit",
        str(args.admin_results_limit),
        "--json",
    ]
    extend_workload_command(command, args)
    if args.task_id:
        command.extend(["--task-id", args.task_id])
    step, collect_payload = run_json_step(
        "kaggle_real_runtime_collect",
        command,
        runner=runner,
        timeout_seconds=args.timeout_seconds,
        secret_values=secret_values,
    )
    micro_beta_payload = load_json(remote_demo_dir(output_dir) / "remote_micro_llm_sharded_beta.json") if workload_kind == MICRO_LLM_SHARDED_KIND else {}
    diagnosis = diagnosis_from_payloads(
        collect_payload=collect_payload,
        upload_summary=upload_status_for(args, output_dir),
        extra_payloads=[micro_beta_payload] if micro_beta_payload else [],
    )
    ready_code = "kaggle_micro_llm_sharded_ready" if workload_kind == MICRO_LLM_SHARDED_KIND else "kaggle_result_accepted"
    ok = bool(step.get("ok") and collect_payload.get("ok") and ready_code in diagnosis)
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": "collect",
        "output_dir": str(output_dir),
        "coordinator_url": coordinator_url_for(args),
        "public_host": args.public_host,
        "port": args.port,
        "miner_id": args.miner_id,
        "workload": {
            "kind": workload_kind,
            "workload_type": workload_type,
            "route": route_name,
            "request_count": args.request_count,
            "scenario_id": args.scenario_id,
            "decode_steps": getattr(args, "decode_steps", None) if workload_kind == MICRO_LLM_SHARDED_KIND else None,
            "stage_mode": getattr(args, "stage_mode", "both") if workload_kind == MICRO_LLM_SHARDED_KIND else None,
            "require_distinct_stage_miners": bool(getattr(args, "require_distinct_stage_miners", False)) if workload_kind == MICRO_LLM_SHARDED_KIND else None,
        },
        "step": step,
        "payloads": {
            "collect": summarize_payload(collect_payload),
            **({"micro_llm_beta": summarize_payload(micro_beta_payload)} if micro_beta_payload else {}),
        },
        "diagnosis_codes": diagnosis,
        "artifacts": base_artifacts(output_dir, ok=ok),
        "safety": safety_summary(args),
        "limitations": [
            "Collect only summarizes an already running real Kaggle runtime acceptance.",
            f"CPU-only read-only {workload_type}; not production Swarm Inference or P2P.",
            "micro-llm-sharded split mode is a toy two-stage pipeline proof, not production large-model sharding." if workload_kind == MICRO_LLM_SHARDED_KIND else "",
        ],
    }
    return persist_report(report, output_dir, secret_values)


def render_markdown(payload: dict[str, Any]) -> str:
    workload = payload.get("workload") or {}
    safety = payload.get("safety") or {}
    lines = [
        "# CrowdTensor Kaggle Real Runtime Acceptance",
        "",
        f"Generated: `{payload.get('generated_at', '')}`",
        f"OK: `{payload.get('ok')}`",
        f"Mode: `{payload.get('mode')}`",
        f"Coordinator URL: `{payload.get('coordinator_url', '')}`",
        f"Miner ID: `{payload.get('miner_id', '')}`",
        f"Workload: `{workload.get('workload_type', MODEL_BUNDLE_WORKLOAD_TYPE)}`",
        f"Route: `{workload.get('route', MODEL_BUNDLE_ROUTE_NAME)}`",
        f"Decode steps: `{workload.get('decode_steps')}`" if workload.get("decode_steps") is not None else "",
        f"Stage mode: `{workload.get('stage_mode')}`" if workload.get("stage_mode") else "",
        f"Diagnosis codes: `{', '.join(payload.get('diagnosis_codes') or [])}`",
        "",
        "## Safety",
        "",
        f"- Temporary HTTP: `{safety.get('temporary_http')}`",
        f"- Token rotation required: `{safety.get('token_rotation_required')}`",
        f"- Operator env excluded from Kaggle: `{safety.get('operator_env_excluded_from_kaggle')}`",
        f"- CPU-only: `{safety.get('cpu_only_workload')}`",
        f"- Not production: `{safety.get('not_production')}`",
        f"- Not P2P: `{safety.get('not_p2p')}`",
        "",
        "## Artifacts",
        "",
    ]
    for name, artifact in sorted((payload.get("artifacts") or {}).items()):
        lines.append(f"- `{name}`: `{artifact.get('path')}` present=`{artifact.get('present')}`")
    lines.extend(["", "## Operator Action", ""])
    for action in payload.get("operator_action") or []:
        lines.append(f"- {action}")
    lines.extend(["", "## Limitations", ""])
    for limitation in payload.get("limitations") or []:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def add_common_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--public-host", default=DEFAULT_PUBLIC_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--miner-id", default="kaggle-cpu-1")
    parser.add_argument("--workload", choices=WORKLOAD_CHOICES, default=MODEL_BUNDLE_KIND)
    parser.add_argument("--output-dir", default="dist/kaggle-real-runtime")
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument("--scenario-id", default="route-baseline")
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--stage-mode", choices=["both", "split"], default="both")
    parser.add_argument("--require-distinct-stage-miners", action="store_true")
    parser.add_argument("--micro-llm-artifact", default="")
    parser.add_argument("--prompt-texts", default="arn,ten")
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--json", action="store_true")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or verify a real Kaggle CPU Miner runtime acceptance.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    prepare = subparsers.add_parser("prepare", help="Generate public Coordinator and Kaggle upload artifacts.")
    add_common_runtime_args(prepare)
    prepare.add_argument("--bind-host", default="0.0.0.0")
    prepare.add_argument("--backlog", type=int, default=1)
    prepare.add_argument("--lease-seconds", type=float, default=15.0)
    prepare.add_argument("--miner-token", default="")
    prepare.add_argument("--observer-token", default="")
    prepare.add_argument("--admin-token", default="")
    prepare.add_argument("--replace", action="store_true")

    verify = subparsers.add_parser("verify", help="Verify a live Kaggle Miner against the public Coordinator.")
    add_common_runtime_args(verify)
    verify.add_argument("--observer-token", default="")
    verify.add_argument("--admin-token", default="")
    verify.add_argument("--remote-timeout-seconds", type=float, default=180.0)
    verify.add_argument("--poll-interval", type=float, default=2.0)
    verify.add_argument("--http-timeout", type=float, default=5.0)
    verify.add_argument("--artifact-timeout", type=float, default=60.0)
    verify.add_argument("--admin-results-limit", type=int, default=10)
    verify.add_argument("--require-existing-result", action="store_true")
    verify.add_argument("--collect-on-failure", action="store_true")

    collect = subparsers.add_parser("collect", help="Collect evidence from an already verified Kaggle runtime.")
    add_common_runtime_args(collect)
    collect.add_argument("--observer-token", default="")
    collect.add_argument("--admin-token", default="")
    collect.add_argument("--http-timeout", type=float, default=5.0)
    collect.add_argument("--artifact-timeout", type=float, default=60.0)
    collect.add_argument("--admin-results-limit", type=int, default=10)
    collect.add_argument("--task-id", default="")

    args = parser.parse_args(argv)
    if args.port < 1:
        raise SystemExit("--port must be positive")
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    if getattr(args, "decode_steps", 1) < 1 or getattr(args, "decode_steps", 1) > 4:
        raise SystemExit("--decode-steps must be between 1 and 4")
    if args.workload == MICRO_LLM_SHARDED_KIND and args.stage_mode == "split":
        args.require_distinct_stage_miners = True
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if getattr(args, "backlog", 1) < 1:
        raise SystemExit("--backlog must be at least 1")
    if getattr(args, "lease_seconds", 1) <= 0:
        raise SystemExit("--lease-seconds must be positive")
    if getattr(args, "remote_timeout_seconds", 0) < 0:
        raise SystemExit("--remote-timeout-seconds must be non-negative")
    if getattr(args, "poll_interval", 1) <= 0:
        raise SystemExit("--poll-interval must be positive")
    if getattr(args, "http_timeout", 1) <= 0:
        raise SystemExit("--http-timeout must be positive")
    if getattr(args, "artifact_timeout", 1) <= 0:
        raise SystemExit("--artifact-timeout must be positive")
    if getattr(args, "admin_results_limit", 1) < 1:
        raise SystemExit("--admin-results-limit must be at least 1")
    args.coordinator_url = args.coordinator_url.rstrip("/") if args.coordinator_url else ""
    return args


def main() -> None:
    args = parse_args()
    if args.action == "prepare":
        report = build_prepare(args)
    elif args.action == "verify":
        report = build_verify(args)
    elif args.action == "collect":
        report = build_collect(args)
    else:
        raise SystemExit(f"unknown action: {args.action}")
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"CrowdTensor Kaggle real runtime {args.action}")
        print(f"  ok: {report.get('ok')}")
        print(f"  schema: {report.get('schema')}")
        print(f"  output: {report.get('output_dir')}")
        print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
