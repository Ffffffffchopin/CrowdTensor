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
WORKLOAD_TYPE = "model_bundle_infer"


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
        "files": {
            "registry": str(Path(args.registry)),
            "operator_private_env": str(operator_env_path),
            "miner_private_env": str(miner_env_path),
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
        "operator_steps": [
            "Copy only miner.private.env to the remote Miner host.",
            "Run the security preflight command on the Coordinator host.",
            "Start the Coordinator with the model_bundle_infer task lane.",
            "Start the remote Miner with the generated Miner command.",
            "After at least one accepted result, collect remote_compute_evidence_v1.",
            f"Keep --scenario-id {args.scenario_id} consistent across acceptance and evidence collection.",
        ],
        "safety": {
            "public_artifact_redacted": True,
            "private_env_files": True,
            "registry_hashed": str(invite.get("token_hash", "")).startswith("sha256:"),
            "requires_tls_or_vpn": True,
            "raw_tokens_in_public_report": False,
        },
        "limitations": [
            "Controlled two-machine demo; not public-internet hardening",
            "CPU-only model_bundle_infer path; not GPU pooling or production LLM serving",
            "No P2P discovery, NAT traversal, decentralized identity, or incentives are claimed",
        ],
    }
    return support_bundle.sanitize(payload)


def render_markdown(payload: dict[str, Any]) -> str:
    demo = payload.get("demo") or {}
    files = payload.get("files") or {}
    tokens = payload.get("tokens") or {}
    safety = payload.get("safety") or {}
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
        "",
        "## Files",
        "",
        f"- Registry: `{files.get('registry', '')}`",
        f"- Operator env: `{files.get('operator_private_env', '')}`",
        f"- Miner env: `{files.get('miner_private_env', '')}`",
        "",
        "## Commands",
        "",
    ]
    for name, command in (payload.get("commands") or {}).items():
        lines.extend([f"### {name}", "", "```bash", command, "```", ""])
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
    return build_runbook(
        args=args,
        invite=invite,
        observer_hash=observer_hash,
        admin_hash=admin_hash,
        operator_env_path=operator_env_path,
        miner_env_path=miner_env_path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a safe two-machine CrowdTensorD remote demo runbook.")
    parser.add_argument("--output-dir", default="dist/remote-demo")
    parser.add_argument("--registry", default="")
    parser.add_argument("--operator-env", default="")
    parser.add_argument("--miner-env", default="")
    parser.add_argument("--coordinator-url", default="http://127.0.0.1:8787")
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
