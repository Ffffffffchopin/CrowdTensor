#!/usr/bin/env python3
"""Build a safe two-machine remote Miner demo runbook."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from crowdtensor.auth import hash_token  # noqa: E402
from crowdtensor.model_bundle import (  # noqa: E402
    DEFAULT_INFERENCE_SCENARIO_ID,
    inference_scenario_summary,
    normalize_inference_scenario_id,
)
from create_miner_invite import create_invite  # noqa: E402
import support_bundle  # noqa: E402


RUNBOOK_SCHEMA = "remote_demo_runbook_v1"
MINER_JOIN_SCHEMA = "miner_join_pack_v1"
WORKLOAD_TYPE = "model_bundle_infer"
TARGET_GENERIC = "generic"
TARGET_KAGGLE = "kaggle"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def quote_env(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def write_private_env(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"export {key}={quote_env(value)}" for key, value in sorted(values.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def write_json(payload: dict[str, Any], path: str) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(payload: dict[str, Any], path: str) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(payload), encoding="utf-8")


def token_or_generated(value: str) -> str:
    return value or secrets.token_urlsafe(32)


def build_commands(args: argparse.Namespace, *, observer_hash: str, admin_hash: str) -> dict[str, str]:
    state_dir = str(Path(args.state_dir))
    registry = str(Path(args.registry))
    coordinator_url = args.coordinator_url.rstrip("/")
    lane = f"python-cli:cpu:{args.backlog}:{WORKLOAD_TYPE}"
    preflight = (
        "python3 scripts/security_preflight.py "
        f"--host {args.bind_host} "
        f"--miner-token-registry {registry} "
        f"--observer-token {observer_hash} "
        f"--admin-token {admin_hash} "
        "--strict --json"
    )
    coordinator = (
        "crowdtensord "
        f"--host {args.bind_host} "
        f"--port {args.port} "
        f"--state-dir {state_dir} "
        f"--lease-seconds {args.lease_seconds} "
        f"--inner-steps {args.request_count} "
        "--backlog 0 "
        f"--task-lane {lane} "
        f"--miner-token-registry {registry} "
        f"--observer-token {observer_hash} "
        f"--admin-token {admin_hash}"
    )
    miner = (
        ". ./miner.private.env && "
        "crowdtensor-miner "
        f"--coordinator {coordinator_url} "
        f"--miner-id {args.miner_id} "
        f"--max-tasks {args.max_tasks} "
        f"--compute-seconds {args.compute_seconds} "
        f"--heartbeat-interval {args.heartbeat_interval} "
        f"--max-request-attempts {args.max_request_attempts}"
    )
    collect = (
        ". ./operator.private.env && "
        "python3 scripts/remote_compute_evidence_pack.py "
        "--mode collect "
        f"--coordinator-url {coordinator_url} "
        f"--miner-id {args.miner_id} "
        f"--request-count {args.request_count} "
        f"--scenario-id {args.scenario_id} "
        "--observer-token \"$CROWDTENSOR_OBSERVER_TOKEN\" "
        "--admin-token \"$CROWDTENSOR_ADMIN_TOKEN\" "
        "--json-out /tmp/crowdtensor_remote_evidence.json "
        "--markdown-out /tmp/crowdtensor_remote_evidence.md"
    )
    support = (
        ". ./operator.private.env && "
        "python3 scripts/support_bundle.py "
        f"--coordinator {coordinator_url} "
        "--observer-token \"$CROWDTENSOR_OBSERVER_TOKEN\" "
        "--admin-token \"$CROWDTENSOR_ADMIN_TOKEN\" "
        "--json-out /tmp/crowdtensor_support_bundle.json"
    )
    return {
        "security_preflight": preflight,
        "start_coordinator": coordinator,
        "start_miner": miner,
        "collect_remote_evidence": collect,
        "collect_support_bundle": support,
    }


def kaggle_commands(args: argparse.Namespace) -> dict[str, str]:
    coordinator_url = args.coordinator_url.rstrip("/")
    return {
        "upload_files": "Upload miner.private.env and kaggle_remote_miner.py to the Kaggle Notebook input or working directory.",
        "install_checkout": "python -m pip install -e .",
        "start_kaggle_miner": (
            "python kaggle_remote_miner.py "
            f"--coordinator {coordinator_url} "
            f"--miner-id {args.miner_id} "
            "--env-file miner.private.env"
        ),
    }


def render_miner_join_script(args: argparse.Namespace) -> str:
    coordinator_url = args.coordinator_url.rstrip("/")
    target = getattr(args, "target", TARGET_GENERIC)
    remote_environment = "kaggle" if target == TARGET_KAGGLE else "generic"
    return f'''#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${{CROWDTENSOR_MINER_ENV_FILE:-miner.private.env}}"
if [ ! -f "$ENV_FILE" ]; then
  echo "missing Miner env file: $ENV_FILE" >&2
  exit 2
fi

set -a
. "$ENV_FILE"
set +a

export CROWDTENSOR_REMOTE_ENVIRONMENT="${{CROWDTENSOR_REMOTE_ENVIRONMENT:-{remote_environment}}}"
exec crowdtensor-miner \\
  --coordinator {coordinator_url!r} \\
  --miner-id {args.miner_id!r} \\
  --max-tasks {args.max_tasks} \\
  --compute-seconds {args.compute_seconds} \\
  --heartbeat-interval {args.heartbeat_interval} \\
  --max-request-attempts {args.max_request_attempts}
'''


def miner_join_pack(args: argparse.Namespace, *, target_environment: dict[str, Any]) -> dict[str, Any]:
    target = getattr(args, "target", TARGET_GENERIC)
    return {
        "schema": MINER_JOIN_SCHEMA,
        "ready": True,
        "target": target_environment.get("name") or TARGET_GENERIC,
        "remote_environment": target_environment.get("remote_environment") or TARGET_GENERIC,
        "workload_type": WORKLOAD_TYPE,
        "route": "remote_python_model_bundle_infer",
        "miner_id": args.miner_id,
        "private_files_required": ["miner.private.env"],
        "operator_files_forbidden": ["operator.private.env", "miner_registry.json"],
        "generated_files": {
            "join_script": "miner_join.sh",
            "join_runbook": "MINER_JOIN.md",
            "kaggle_script": "kaggle_remote_miner.py" if target == TARGET_KAGGLE else "",
        },
        "recommended_command": "bash miner_join.sh",
        "kaggle_command": "python kaggle_remote_miner.py --env-file miner.private.env" if target == TARGET_KAGGLE else "",
        "safety": {
            "token_values_in_public_artifacts": False,
            "operator_env_required_on_miner": False,
            "miner_outbound_only": target == TARGET_KAGGLE,
            "requires_operator_provided_transport": True,
        },
        "boundaries": {
            "cpu_only": True,
            "read_only": True,
            "task_level_remote_inference": True,
            "not_model_sharding": True,
            "not_p2p": True,
            "not_production": True,
        },
    }


def render_miner_join_markdown(payload: dict[str, Any]) -> str:
    demo = payload.get("demo") or {}
    miner = payload.get("miner") or {}
    join = payload.get("miner_join_pack") or {}
    target = payload.get("target_environment") or {}
    lines = [
        "# CrowdTensor Miner Join Pack",
        "",
        f"Schema: `{join.get('schema', MINER_JOIN_SCHEMA)}`",
        f"Coordinator URL: `{demo.get('coordinator_url', '')}`",
        f"Miner ID: `{miner.get('miner_id', '')}`",
        f"Target: `{target.get('name', TARGET_GENERIC)}`",
        f"Workload: `{demo.get('workload_type', WORKLOAD_TYPE)}`",
        f"Route: `{demo.get('route', 'remote_python_model_bundle_infer')}`",
        "",
        "## Miner Host Steps",
        "",
        "1. Copy only `miner.private.env`, `miner_join.sh`, and this file to the Miner host.",
        "2. Install CrowdTensor from this checkout or package.",
        "3. Run the generated join command.",
        "",
        "```bash",
        join.get("recommended_command") or "bash miner_join.sh",
        "```",
        "",
    ]
    if join.get("kaggle_command"):
        lines.extend([
            "## Kaggle CPU Runtime",
            "",
            "Upload only `miner.private.env` and `kaggle_remote_miner.py`, then run:",
            "",
            "```bash",
            str(join.get("kaggle_command")),
            "```",
            "",
        ])
    lines.extend([
        "## Do Not Copy",
        "",
        "- `operator.private.env`",
        "- `miner_registry.json`",
        "",
        "## Boundaries",
        "",
        "- CPU-only, read-only task-level remote inference.",
        "- Not production Swarm Inference, not model sharding, not P2P, and not GPU/TPU pooling.",
        "- Real two-machine use still requires operator-provided TLS, VPN, tunnel, or trusted network.",
        "",
    ])
    return "\n".join(lines)


def render_kaggle_miner_script(args: argparse.Namespace) -> str:
    coordinator_url = args.coordinator_url.rstrip("/")
    return f'''#!/usr/bin/env python3
"""Kaggle Remote Miner Beta launcher for CrowdTensor.

Upload this file and miner.private.env into a Kaggle Notebook attached to the
CrowdTensor checkout, then run this script from the notebook. It only starts an
outbound CPU-only Miner; it does not expose a Coordinator from Kaggle.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_COORDINATOR = {coordinator_url!r}
DEFAULT_MINER_ID = {args.miner_id!r}


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Kaggle-hosted CrowdTensor remote Miner.")
    parser.add_argument("--coordinator", default=os.environ.get("CROWDTENSOR_COORDINATOR_URL", DEFAULT_COORDINATOR))
    parser.add_argument("--miner-id", default=os.environ.get("CROWDTENSOR_MINER_ID", DEFAULT_MINER_ID))
    parser.add_argument("--env-file", default=os.environ.get("CROWDTENSOR_MINER_ENV_FILE", "miner.private.env"))
    parser.add_argument("--max-tasks", type=int, default=int(os.environ.get("CROWDTENSOR_MAX_TASKS", "1")))
    parser.add_argument("--compute-seconds", type=float, default=float(os.environ.get("CROWDTENSOR_COMPUTE_SECONDS", "0.2")))
    parser.add_argument("--heartbeat-interval", type=float, default=float(os.environ.get("CROWDTENSOR_HEARTBEAT_INTERVAL", "0.1")))
    parser.add_argument("--max-request-attempts", type=int, default=int(os.environ.get("CROWDTENSOR_MAX_REQUEST_ATTEMPTS", "5")))
    args = parser.parse_args()

    env_file = Path(args.env_file)
    if not env_file.is_file():
        raise SystemExit(f"missing Miner env file: {{env_file}}")
    env = os.environ.copy()
    env.update(load_env(env_file))
    env["CROWDTENSOR_REMOTE_ENVIRONMENT"] = "kaggle"
    env.setdefault("PYTHONUNBUFFERED", "1")

    command = [
        "crowdtensor-miner",
        "--coordinator",
        args.coordinator,
        "--miner-id",
        args.miner_id,
        "--max-tasks",
        str(args.max_tasks),
        "--compute-seconds",
        str(args.compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--max-request-attempts",
        str(args.max_request_attempts),
    ]
    print("Starting Kaggle remote Miner:", " ".join(command), flush=True)
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
'''


def render_kaggle_markdown(payload: dict[str, Any]) -> str:
    demo = payload.get("demo") or {}
    target = payload.get("target_environment") or {}
    commands = payload.get("kaggle_commands") or {}
    lines = [
        "# CrowdTensor Kaggle Remote Miner Beta",
        "",
        f"Generated: `{payload.get('generated_at', '')}`",
        f"Coordinator URL: `{demo.get('coordinator_url', '')}`",
        f"Miner ID: `{(payload.get('miner') or {}).get('miner_id', '')}`",
        f"Target: `{target.get('name', '')}`",
        "",
        "## Kaggle Notebook Steps",
        "",
        "1. Create a Kaggle Notebook with Internet enabled.",
        "2. Upload only `miner.private.env` and `kaggle_remote_miner.py` to the notebook working directory or attach them as private input files.",
        "3. Install this CrowdTensor checkout in the notebook.",
        "4. Run the generated Kaggle Miner command.",
        "",
        "## Commands",
        "",
    ]
    for name, command in commands.items():
        lines.extend([f"### {name}", "", "```bash", command, "```", ""])
    lines.extend([
        "## Boundaries",
        "",
        "- Kaggle is used as an outbound remote Miner environment; it is not the production network substrate.",
        "- This Beta is CPU-only and read-only for `model_bundle_infer`.",
        "- GPU/TPU visibility may be recorded as runtime hints, but no GPU/TPU workload is enabled by this path.",
        "- Do not paste `operator.private.env` into Kaggle.",
        "",
    ])
    return "\n".join(lines)


def build_runbook(
    *,
    args: argparse.Namespace,
    invite: dict[str, Any],
    observer_hash: str,
    admin_hash: str,
    operator_env_path: Path,
    miner_env_path: Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    commands = build_commands(args, observer_hash=observer_hash, admin_hash=admin_hash)
    scenario = inference_scenario_summary(args.scenario_id)
    target = getattr(args, "target", TARGET_GENERIC)
    target_environment = {
        "name": target,
        "remote_environment": "kaggle" if target == TARGET_KAGGLE else "generic",
        "kaggle_remote_miner_beta": target == TARGET_KAGGLE,
        "coordinator_in_kaggle": False,
        "miner_outbound_only": target == TARGET_KAGGLE,
        "gpu_tpu_workload_enabled": False,
    }
    payload = {
        "schema": RUNBOOK_SCHEMA,
        "generated_at": generated_at or utc_now(),
        "ok": True,
        "demo": {
            "kind": "controlled_two_machine_remote_miner",
            "coordinator_url": args.coordinator_url.rstrip("/"),
            "bind_host": args.bind_host,
            "port": args.port,
            "workload_type": WORKLOAD_TYPE,
            "request_count": args.request_count,
            "scenario_schema": scenario.get("scenario_schema"),
            "scenario_id": scenario.get("scenario_id"),
            "scenario_description": scenario.get("scenario_description"),
            "scenario_request_count": scenario.get("scenario_request_count"),
            "route": "remote_python_model_bundle_infer",
        },
        "target_environment": target_environment,
        "files": {
            "registry": str(Path(args.registry)),
            "operator_private_env": str(operator_env_path),
            "miner_private_env": str(miner_env_path),
            "miner_join_script": str(Path(args.output_dir) / "miner_join.sh"),
            "miner_join_runbook": str(Path(args.output_dir) / "MINER_JOIN.md"),
            "kaggle_miner_script": str(Path(args.output_dir) / "kaggle_remote_miner.py") if target == TARGET_KAGGLE else "",
            "kaggle_runbook": str(Path(args.output_dir) / "kaggle_remote_miner.md") if target == TARGET_KAGGLE else "",
        },
        "tokens": {
            "miner_registry_hashed": str(invite.get("token_hash", "")).startswith("sha256:"),
            "observer_hash_prefix": observer_hash.split(":", 1)[0],
            "admin_hash_prefix": admin_hash.split(":", 1)[0],
        },
        "miner": {
            "miner_id": args.miner_id,
            "max_tasks": args.max_tasks,
            "compute_seconds": args.compute_seconds,
            "heartbeat_interval": args.heartbeat_interval,
            "max_request_attempts": args.max_request_attempts,
        },
        "commands": commands,
        "miner_join_pack": miner_join_pack(args, target_environment=target_environment),
        "kaggle_commands": kaggle_commands(args) if target == TARGET_KAGGLE else {},
        "operator_steps": [
            "Copy only miner.private.env to the remote Miner host.",
            "Run the security preflight command on the Coordinator host.",
            "Start the Coordinator with the model_bundle_infer task lane.",
            "Start the remote Miner with the generated Miner command.",
            "After at least one accepted result, collect remote_compute_evidence_v1.",
            f"Keep --scenario-id {args.scenario_id} consistent across acceptance and evidence collection.",
        ],
        "kaggle_steps": [
            "Keep operator.private.env on the Coordinator/operator side.",
            "Upload only miner.private.env and kaggle_remote_miner.py to the Kaggle Notebook.",
            "Run Kaggle as an outbound Miner; do not try to expose a Coordinator from Kaggle in this Beta.",
            "Run crowdtensor remote-demo doctor/verify/collect from the operator host.",
        ] if target == TARGET_KAGGLE else [],
        "safety": {
            "public_artifact_redacted": True,
            "private_env_files": True,
            "registry_hashed": str(invite.get("token_hash", "")).startswith("sha256:"),
            "requires_tls_or_vpn": True,
            "raw_tokens_in_public_report": False,
            "kaggle_operator_env_excluded": target == TARGET_KAGGLE,
        },
        "limitations": [
            "Controlled two-machine demo; not public-internet hardening",
            "CPU-only model_bundle_infer path; not GPU pooling or production LLM serving",
            "No P2P discovery, NAT traversal, decentralized identity, or incentives are claimed",
            "Kaggle target records accelerator visibility only as hints; GPU/TPU workloads are not enabled",
        ],
    }
    return support_bundle.sanitize(payload)


def render_markdown(payload: dict[str, Any]) -> str:
    demo = payload.get("demo") or {}
    files = payload.get("files") or {}
    tokens = payload.get("tokens") or {}
    safety = payload.get("safety") or {}
    target = payload.get("target_environment") or {}
    lines = [
        "# CrowdTensor Remote Demo Runbook",
        "",
        f"Generated: `{payload.get('generated_at', '')}`",
        f"OK: `{payload.get('ok')}`",
        "",
        "## Demo",
        "",
        f"- Coordinator URL: `{demo.get('coordinator_url', '')}`",
        f"- Workload: `{demo.get('workload_type', '')}`",
        f"- Route: `{demo.get('route', '')}`",
        f"- Request count: `{demo.get('request_count')}`",
        f"- Scenario: `{demo.get('scenario_id')}`",
        f"- Scenario schema: `{demo.get('scenario_schema')}`",
        f"- Target: `{target.get('name', 'generic')}`",
        "",
        "## Files",
        "",
        f"- Registry: `{files.get('registry', '')}`",
        f"- Operator env: `{files.get('operator_private_env', '')}`",
        f"- Miner env: `{files.get('miner_private_env', '')}`",
        f"- Miner join script: `{files.get('miner_join_script', '')}`",
        f"- Miner join runbook: `{files.get('miner_join_runbook', '')}`",
        f"- Kaggle Miner script: `{files.get('kaggle_miner_script', '')}`",
        f"- Kaggle runbook: `{files.get('kaggle_runbook', '')}`",
        "",
        "## Commands",
        "",
    ]
    for name, command in (payload.get("commands") or {}).items():
        lines.extend([f"### {name}", "", "```bash", command, "```", ""])
    if payload.get("kaggle_commands"):
        lines.extend(["## Kaggle Remote Miner", ""])
        for name, command in (payload.get("kaggle_commands") or {}).items():
            lines.extend([f"### {name}", "", "```bash", command, "```", ""])
    if payload.get("miner_join_pack"):
        lines.extend([
            "## Miner Join Pack",
            "",
            f"- Schema: `{(payload.get('miner_join_pack') or {}).get('schema')}`",
            f"- Command: `{(payload.get('miner_join_pack') or {}).get('recommended_command')}`",
            "- Copy only `miner.private.env` and generated join files to the Miner host.",
            "",
        ])
    lines.extend([
        "## Safety",
        "",
        f"- Registry hashed: `{safety.get('registry_hashed')}`",
        f"- Public artifact redacted: `{safety.get('public_artifact_redacted')}`",
        f"- Private env files: `{safety.get('private_env_files')}`",
        f"- Requires TLS or VPN: `{safety.get('requires_tls_or_vpn')}`",
        f"- Observer hash prefix: `{tokens.get('observer_hash_prefix', '')}`",
        f"- Admin hash prefix: `{tokens.get('admin_hash_prefix', '')}`",
        "",
        "## Limitations",
        "",
    ])
    for limitation in payload.get("limitations") or []:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def build_from_args(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = Path(args.registry)
    operator_env_path = Path(args.operator_env)
    miner_env_path = Path(args.miner_env)

    miner_token = token_or_generated(args.miner_token)
    observer_token = token_or_generated(args.observer_token)
    admin_token = token_or_generated(args.admin_token)
    observer_hash = hash_token(observer_token)
    admin_hash = hash_token(admin_token)
    invite = create_invite(
        registry_path=registry_path,
        miner_id=args.miner_id,
        coordinator_url=args.coordinator_url,
        label=args.label,
        token=miner_token,
        replace=args.replace,
    )
    write_private_env(operator_env_path, {
        "CROWDTENSOR_ADMIN_TOKEN": admin_token,
        "CROWDTENSOR_OBSERVER_TOKEN": observer_token,
    })
    write_private_env(miner_env_path, {
        "CROWDTENSOR_MINER_TOKEN": miner_token,
    })
    payload = build_runbook(
        args=args,
        invite=invite,
        observer_hash=observer_hash,
        admin_hash=admin_hash,
        operator_env_path=operator_env_path,
        miner_env_path=miner_env_path,
    )
    if getattr(args, "target", TARGET_GENERIC) == TARGET_KAGGLE:
        script_path = output_dir / "kaggle_remote_miner.py"
        runbook_path = output_dir / "kaggle_remote_miner.md"
        script_path.write_text(render_kaggle_miner_script(args), encoding="utf-8")
        script_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        runbook_path.write_text(render_kaggle_markdown(payload), encoding="utf-8")
    join_script = output_dir / "miner_join.sh"
    join_runbook = output_dir / "MINER_JOIN.md"
    join_script.write_text(render_miner_join_script(args), encoding="utf-8")
    join_script.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    join_runbook.write_text(render_miner_join_markdown(payload), encoding="utf-8")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a safe two-machine CrowdTensorD remote demo runbook.")
    parser.add_argument("--output-dir", default="dist/remote-demo")
    parser.add_argument("--registry", default="")
    parser.add_argument("--operator-env", default="")
    parser.add_argument("--miner-env", default="")
    parser.add_argument("--coordinator-url", default="http://127.0.0.1:8787")
    parser.add_argument("--target", choices=[TARGET_GENERIC, TARGET_KAGGLE], default=TARGET_GENERIC)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--state-dir", default="state")
    parser.add_argument("--miner-id", default="remote-linux-1")
    parser.add_argument("--label", default="remote demo miner")
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--scenario-id", default=DEFAULT_INFERENCE_SCENARIO_ID)
    parser.add_argument("--backlog", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--lease-seconds", type=float, default=15.0)
    parser.add_argument("--compute-seconds", type=float, default=0.2)
    parser.add_argument("--heartbeat-interval", type=float, default=0.1)
    parser.add_argument("--max-request-attempts", type=int, default=5)
    parser.add_argument("--miner-token", default="")
    parser.add_argument("--observer-token", default="")
    parser.add_argument("--admin-token", default="")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown-out", default="")
    args = parser.parse_args(argv)
    if args.request_count < 1:
        raise SystemExit("--request-count must be at least 1")
    args.scenario_id = normalize_inference_scenario_id(args.scenario_id) or DEFAULT_INFERENCE_SCENARIO_ID
    if args.backlog < 1:
        raise SystemExit("--backlog must be at least 1")
    if args.max_tasks < 1:
        raise SystemExit("--max-tasks must be at least 1")
    output_dir = Path(args.output_dir)
    if not args.registry:
        args.registry = str(output_dir / "miner_registry.json")
    if not args.operator_env:
        args.operator_env = str(output_dir / "operator.private.env")
    if not args.miner_env:
        args.miner_env = str(output_dir / "miner.private.env")
    if not args.json_out:
        args.json_out = str(output_dir / "remote_demo_runbook.json")
    if not args.markdown_out:
        args.markdown_out = str(output_dir / "remote_demo_runbook.md")
    args.coordinator_url = args.coordinator_url.rstrip("/")
    return args


def main() -> None:
    try:
        args = parse_args()
        payload = build_from_args(args)
        write_json(payload, args.json_out)
        write_markdown(payload, args.markdown_out)
        print(json.dumps(payload, sort_keys=True))
    except SystemExit:
        raise
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
