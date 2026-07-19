"""CLI for the Community training golden path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

from .community_workflow import CommunityWorkflow, CommunityWorkflowError
from .model_adapter import (
    adapter_registry_report,
    check_model_adapter_conformance,
    get_model_adapter,
)


def _common(parser: argparse.ArgumentParser, *, dry_run: bool = True) -> None:
    parser.add_argument("workspace")
    if dry_run:
        parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="crowdtensor community",
        description="Versioned ordinary-user heterogeneous LoRA workflow.",
    )
    actions = parser.add_subparsers(dest="action", required=True)
    adapters = actions.add_parser("adapters")
    adapter_actions = adapters.add_subparsers(dest="adapter_action", required=True)
    adapter_list = adapter_actions.add_parser("list")
    adapter_list.add_argument("--json", action="store_true")
    adapter_check = adapter_actions.add_parser("check")
    adapter_check.add_argument("adapter_id")
    adapter_check.add_argument("--json", action="store_true")
    init = actions.add_parser("init")
    _common(init)
    init.add_argument("--adapter", default="qwen2_lora_v1")
    init.add_argument("--model", default="")
    init.add_argument("--revision", default="")
    init.add_argument("--accelerators", default="cpu,cuda")
    init.add_argument("--target-steps", type=int, default=100)
    init.add_argument("--force", action="store_true")
    for name in ("validate", "plan", "train", "status", "pause", "resume", "stop", "cleanup", "contract"):
        item = actions.add_parser(name)
        _common(item, dry_run=name in {"train", "pause", "resume", "stop", "cleanup"})
    export = actions.add_parser("export")
    _common(export)
    export.add_argument("--output-dir", default="")
    rebalance = actions.add_parser("rebalance")
    _common(rebalance)
    rebalance.add_argument(
        "--reason",
        choices=["owner_requested", "performance_rebalance", "health_degraded", "coordinator_recovery"],
        default="owner_requested",
    )
    coordinator = actions.add_parser("coordinator")
    coordinator_actions = coordinator.add_subparsers(dest="coordinator_action", required=True)
    up = coordinator_actions.add_parser("up")
    _common(up)
    up.add_argument("--run", action="store_true", help="run the foreground service after preparation")
    miner = actions.add_parser("miner")
    miner_actions = miner.add_subparsers(dest="miner_action", required=True)
    join = miner_actions.add_parser("join")
    _common(join)
    join.add_argument("--run", action="store_true", help="join in the foreground")
    join.add_argument("--device-policy", choices=["auto", "cpu", "cuda", "jax_tpu"], default="auto")
    return parser.parse_args(argv)


def _emit(value: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(value, indent=2, sort_keys=True))
        return
    print(f"community action={value.get('action', value.get('schema'))} ok={bool(value.get('ok', True))}")
    if value.get("run_id"):
        print(f"run_id={value['run_id']}")
    if value.get("next_command"):
        print(f"next={value['next_command']}")


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.action == "adapters":
            if args.adapter_action == "list":
                value = adapter_registry_report()
                value["ok"] = True
            else:
                value = check_model_adapter_conformance(
                    get_model_adapter(args.adapter_id)
                )
        elif args.action == "init":
            value = CommunityWorkflow.initialize(
                args.workspace,
                adapter_id=args.adapter,
                model_id=args.model,
                revision=args.revision,
                accelerators=[item.strip() for item in args.accelerators.split(",") if item.strip()],
                target_steps=args.target_steps,
                force=args.force,
                dry_run=args.dry_run,
            )
        else:
            workflow = CommunityWorkflow(args.workspace)
            if args.action == "coordinator":
                value = workflow.coordinator_up(dry_run=args.dry_run, run=args.run)
            elif args.action == "miner":
                value = workflow.miner_join(dry_run=args.dry_run, run=args.run, device_policy=args.device_policy)
            elif args.action == "rebalance":
                value = workflow.rebalance(reason=args.reason, dry_run=args.dry_run)
            elif args.action == "export":
                value = workflow.export(output_dir=args.output_dir or None, dry_run=args.dry_run)
            else:
                method = getattr(workflow, args.action)
                value = method(dry_run=args.dry_run) if hasattr(args, "dry_run") else method()
        _emit(value, json_output=bool(args.json))
        return 0 if value.get("ok", True) else 2
    except CommunityWorkflowError as exc:
        value = {
            "schema": "crowdtensor_community_cli_error_v1",
            "ok": False,
            "error": exc.reason,
            "exit_code": exc.exit_code,
            "credential_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        _emit(value, json_output=bool(getattr(args, "json", False)))
        return exc.exit_code
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        value = {
            "schema": "crowdtensor_community_cli_error_v1",
            "ok": False,
            "error": "community_runtime_operation_failed:" + type(exc).__name__,
            "exit_code": 5,
            "credential_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        _emit(value, json_output=bool(getattr(args, "json", False)))
        return 5


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run(argv))


if __name__ == "__main__":
    main()
