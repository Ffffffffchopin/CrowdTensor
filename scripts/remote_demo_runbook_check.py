#!/usr/bin/env python3
"""Acceptance check for the safe two-machine remote demo runbook."""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the CrowdTensorD remote demo runbook pack.")
    parser.add_argument("--coordinator-url", default="http://127.0.0.1:8787")
    parser.add_argument("--miner-id", default="remote-runbook-miner")
    parser.add_argument("--request-count", type=int, default=4)
    parser.add_argument("--scenario-id", default="route-baseline")
    return parser.parse_args()


def assert_not_contains(payload: str, fragments: list[str]) -> None:
    for fragment in fragments:
        if fragment and fragment in payload:
            raise SystemExit(f"public runbook leaked sensitive fragment: {fragment}")


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="crowdtensor_remote_runbook_") as temp:
        output_dir = Path(temp)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "remote_demo_runbook_pack.py"),
            "--output-dir",
            str(output_dir),
            "--coordinator-url",
            args.coordinator_url,
            "--miner-id",
            args.miner_id,
            "--request-count",
            str(args.request_count),
            "--scenario-id",
            args.scenario_id,
            "--miner-token",
            "runbook-miner-secret",
            "--observer-token",
            "runbook-observer-secret",
            "--admin-token",
            "runbook-admin-secret",
        ]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)
        if completed.returncode != 0:
            raise SystemExit(
                "remote_demo_runbook_pack.py failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise SystemExit("remote_demo_runbook_pack.py emitted no JSON")
        runbook = json.loads(lines[-1])
        if runbook.get("schema") != "remote_demo_runbook_v1" or not runbook.get("ok"):
            raise SystemExit(f"unexpected runbook payload: {runbook}")

        registry_path = output_dir / "miner_registry.json"
        operator_env_path = output_dir / "operator.private.env"
        miner_env_path = output_dir / "miner.private.env"
        markdown_path = output_dir / "remote_demo_runbook.md"
        json_path = output_dir / "remote_demo_runbook.json"
        for path in [registry_path, operator_env_path, miner_env_path, markdown_path, json_path]:
            if not path.is_file():
                raise SystemExit(f"expected runbook file was not created: {path}")

        registry_text = registry_path.read_text(encoding="utf-8")
        if "runbook-miner-secret" in registry_text or "sha256:" not in registry_text:
            raise SystemExit(f"registry did not store hashed token only: {registry_text}")

        operator_text = operator_env_path.read_text(encoding="utf-8")
        miner_text = miner_env_path.read_text(encoding="utf-8")
        if "runbook-admin-secret" not in operator_text or "runbook-observer-secret" not in operator_text:
            raise SystemExit("operator private env is missing operator tokens")
        if "runbook-miner-secret" in operator_text:
            raise SystemExit("operator private env must not contain Miner token")
        if "runbook-miner-secret" not in miner_text:
            raise SystemExit("miner private env is missing Miner token")
        if "runbook-admin-secret" in miner_text or "runbook-observer-secret" in miner_text:
            raise SystemExit("miner private env must not contain operator tokens")
        for path in [operator_env_path, miner_env_path]:
            mode = stat.S_IMODE(os.stat(path).st_mode)
            if mode != 0o600:
                raise SystemExit(f"private env file must be 0600: {path} mode={oct(mode)}")

        public_payload = json.dumps(runbook, sort_keys=True) + "\n" + markdown_path.read_text(encoding="utf-8")
        assert_not_contains(public_payload, [
            "runbook-miner-secret",
            "runbook-observer-secret",
            "runbook-admin-secret",
            "CROWDTENSOR_MINER_TOKEN=runbook",
            "lease_token",
            "idempotency_key",
        ])
        commands = runbook.get("commands") or {}
        coordinator_command = commands.get("start_coordinator", "")
        collect_command = commands.get("collect_remote_evidence", "")
        miner_command = commands.get("start_miner", "")
        required_fragments = [
            "--backlog 0",
            "--task-lane python-cli:cpu:1:model_bundle_infer",
            "--miner-token-registry",
            "--observer-token sha256:",
            "--admin-token sha256:",
        ]
        for fragment in required_fragments:
            if fragment not in coordinator_command:
                raise SystemExit(f"coordinator command missing {fragment}: {coordinator_command}")
        if "remote_compute_evidence_pack.py --mode collect" not in collect_command:
            raise SystemExit(f"collect command missing evidence collect mode: {collect_command}")
        if f"--scenario-id {args.scenario_id}" not in collect_command:
            raise SystemExit(f"collect command missing scenario id: {collect_command}")
        if "--observer-token \"$CROWDTENSOR_OBSERVER_TOKEN\"" not in collect_command:
            raise SystemExit(f"collect command must read observer token from operator env: {collect_command}")
        if ". ./miner.private.env" not in miner_command or args.miner_id not in miner_command:
            raise SystemExit(f"miner command missing private env or miner id: {miner_command}")

        print(json.dumps({
            "ok": True,
            "schema": runbook["schema"],
            "miner_id": runbook.get("miner", {}).get("miner_id"),
            "route": runbook.get("demo", {}).get("route"),
            "scenario_id": runbook.get("demo", {}).get("scenario_id"),
            "registry_hashed": runbook.get("safety", {}).get("registry_hashed"),
            "operator_env_mode": "0600",
            "miner_env_mode": "0600",
        }, sort_keys=True))


if __name__ == "__main__":
    main()
