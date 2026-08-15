"""Thin command surface for framework-neutral Training Architecture v2 work."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from .contracts import TrainingMode
from .execution import PROVIDER_SNAPSHOT_SCHEMA, ProviderSnapshot, ResourceAvailability
from .workspace import (
    CONTROL_DIR,
    PROJECT_FILE,
    export_workspace,
    init_project,
    inspect_workspace,
    join_workspace,
    load_project,
    pause_workspace,
    record_plan,
    resume_workspace,
    run_workspace,
)


TRAINING_V2_ACTIONS = frozenset(
    {
        "backends",
        "init",
        "inspect",
        "plan",
        "run",
        "start",
        "join",
        "status",
        "pause",
        "resume",
        "export",
    }
)


def add_training_v2_run_arguments(
    parser: Any, *, include_dry_run: bool = True
) -> None:
    """Add bounded elastic and stable Session options to a run parser."""

    parser.add_argument(
        "--campaign-dir",
        default="",
        help="Existing Volunteer PEFT Campaign to serve through this v2 workspace.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8789)
    parser.add_argument("--public-url", default="")
    parser.add_argument(
        "--release-dir",
        default="",
        help="Directory containing the exact contributor release artifacts to serve.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    if include_dry_run:
        parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--require-https", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--trust-forwarded-headers", action="store_true")
    parser.add_argument("--trusted-proxy-id", default="")
    parser.add_argument("--upload-chunk-bytes", type=int, default=1024 * 1024)
    parser.add_argument(
        "--work-unit-steps",
        type=int,
        default=0,
        help="Stable mode steps per committed Work Unit; 0 uses all remaining steps.",
    )
    parser.add_argument(
        "--max-work-units",
        type=int,
        default=1,
        help="Maximum stable Work Units to execute in this invocation.",
    )
    parser.add_argument(
        "--execution-timeout-seconds",
        type=float,
        default=3600.0,
        help="Per-attempt stable rank-group timeout.",
    )


def add_training_v2_join_arguments(parser: Any) -> None:
    """Add bounded Volunteer PEFT contributor options to a join parser."""

    parser.add_argument("--invite", default="")
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--code", dest="pairing_code", default="")
    parser.add_argument("--cell-id", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-local-steps", type=int, default=64)
    parser.add_argument("--max-download-gib", type=float, default=8.0)
    parser.add_argument("--max-work-units", type=int, default=1)
    parser.add_argument("--poll-interval-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument(
        "--status-page", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--status-port", type=int, default=8765)
    parser.add_argument("--dry-run", action="store_true")


def add_training_v2_parsers(subparsers: Any, *, include_lifecycle: bool = True) -> None:
    initialize = subparsers.add_parser(
        "init",
        help="Initialize a small, framework-neutral Training Architecture v2 workspace.",
    )
    initialize.add_argument("workspace")
    initialize.add_argument("--model", required=True)
    initialize.add_argument("--model-revision", required=True)
    initialize.add_argument("--dataset", required=True)
    initialize.add_argument("--dataset-revision", required=True)
    initialize.add_argument("--model-adapter", required=True)
    initialize.add_argument(
        "--backend",
        dest="training_backend",
        default="auto",
        help="Numerical backend identifier; execution is supplied by a plugin.",
    )
    initialize.add_argument(
        "--mode",
        choices=["elastic-delta", "stable-sharded"],
        default="elastic-delta",
    )
    initialize.add_argument("--target-steps", type=int, default=100)
    initialize.add_argument("--project-id", default="")
    initialize.add_argument(
        "--optimization-plugin",
        action="append",
        default=[],
        dest="optimization_plugins",
    )
    initialize.add_argument("--json", action="store_true")

    inspect = subparsers.add_parser(
        "inspect",
        help="Inspect a Training Architecture v2 workspace without loading a model.",
    )
    inspect.add_argument("workspace")
    inspect.add_argument("--json", action="store_true")

    backends = subparsers.add_parser(
        "backends",
        help="List framework-neutral training backend integrations.",
    )
    backends.add_argument("--json", action="store_true")

    if include_lifecycle:
        run = subparsers.add_parser(
            "run", aliases=["start"],
            help="Validate v2 execution readiness without silently executing a backend.",
        )
        run.add_argument("workspace")
        add_training_v2_run_arguments(run)
        run.add_argument("--json", action="store_true")

        join = subparsers.add_parser(
            "join", help="Show the v2 contributor admission boundary for a workspace."
        )
        join.add_argument("workspace")
        add_training_v2_join_arguments(join)
        join.add_argument("--json", action="store_true")

        status = subparsers.add_parser(
            "status", help="Show v2 workspace lifecycle and checkpoint status."
        )
        status.add_argument("workspace")
        status.add_argument("--json", action="store_true")

        pause = subparsers.add_parser("pause", help="Pause v2 work at a durable local boundary.")
        pause.add_argument("workspace")
        pause.add_argument("--reason", default="owner_paused")
        pause.add_argument("--json", action="store_true")

        resume = subparsers.add_parser("resume", help="Resume a paused v2 workspace.")
        resume.add_argument("workspace")
        resume.add_argument("--json", action="store_true")

        export = subparsers.add_parser(
            "export", help="Export public v2 contracts without model weights or credentials."
        )
        export.add_argument("workspace")
        export.add_argument("--output-dir", default="")
        export.add_argument("--json", action="store_true")


def add_training_v2_plan_arguments(parser: Any) -> None:
    parser.add_argument(
        "--capability",
        action="append",
        default=[],
        help="Public-safe provider snapshot or legacy capability JSON.",
    )
    parser.add_argument(
        "--runtime-probe",
        action="store_true",
        help="Explicitly import and probe the selected numerical runtime.",
    )
    parser.add_argument("--trainer-entrypoint", default="")
    parser.add_argument("--trainer-contract-verified", action="store_true")
    parser.add_argument("--transformer-layer-class", default="")
    parser.add_argument(
        "--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16"
    )
    parser.add_argument("--max-restarts", type=int, default=1)
    parser.add_argument("--stable-group-id", default="stable-window")
    parser.add_argument("--materialize", action="store_true")


def is_training_v2_workspace(value: str | Path) -> bool:
    root = Path(value).expanduser()
    return (root / CONTROL_DIR / PROJECT_FILE).is_file()


def validate_training_v2_args(args: argparse.Namespace) -> None:
    if args.train_action not in TRAINING_V2_ACTIONS:
        return
    if args.train_action == "init":
        if args.target_steps < 1:
            raise SystemExit("--target-steps must be at least 1")
        if not args.model.strip() or not args.model_revision.strip():
            raise SystemExit("--model and --model-revision must not be empty")
        if not args.dataset.strip() or not args.dataset_revision.strip():
            raise SystemExit("--dataset and --dataset-revision must not be empty")
        if not args.model_adapter.strip() or not args.training_backend.strip():
            raise SystemExit("--model-adapter and --backend must not be empty")
        if any(not str(item).strip() for item in args.optimization_plugins):
            raise SystemExit("--optimization-plugin values must not be empty")
    if args.train_action == "plan":
        if args.max_restarts < 0:
            raise SystemExit("--max-restarts must be non-negative")
        if args.materialize and not str(args.job or "").strip():
            raise SystemExit("a v2 workspace is required for --materialize")
    if args.train_action in {"run", "start"} and getattr(args, "campaign_dir", ""):
        if args.port < 1 or args.port > 65535:
            raise SystemExit("--port must be in [1, 65535]")
        if args.upload_chunk_bytes < 64 * 1024 or args.upload_chunk_bytes > 64 * 1024**2:
            raise SystemExit("--upload-chunk-bytes must be in [65536, 67108864]")
        if bool(args.trust_forwarded_headers) != bool(str(args.trusted_proxy_id).strip()):
            raise SystemExit(
                "--trust-forwarded-headers and --trusted-proxy-id must be provided together"
            )
    if args.train_action in {"run", "start"}:
        if args.work_unit_steps < 0:
            raise SystemExit("--work-unit-steps must be non-negative")
        if args.max_work_units < 1 or args.max_work_units > 1024:
            raise SystemExit("--max-work-units must be in [1, 1024]")
        if (
            not math.isfinite(args.execution_timeout_seconds)
            or args.execution_timeout_seconds < 1
            or args.execution_timeout_seconds > 86400
        ):
            raise SystemExit("--execution-timeout-seconds must be in [1, 86400]")
    if args.train_action == "join" and (
        getattr(args, "invite", "")
        or getattr(args, "coordinator_url", "")
        or getattr(args, "pairing_code", "")
    ):
        if bool(args.invite) == bool(args.coordinator_url):
            raise SystemExit("provide exactly one of --invite or --coordinator-url")
        if args.pairing_code and not args.coordinator_url:
            raise SystemExit("--code requires --coordinator-url")
        if args.max_local_steps < 1 or args.max_local_steps > 1024:
            raise SystemExit("--max-local-steps must be in [1, 1024]")
        if (
            not math.isfinite(args.max_download_gib)
            or args.max_download_gib <= 0
            or args.max_download_gib > 1024
        ):
            raise SystemExit("--max-download-gib must be in (0, 1024]")
        if args.max_work_units < 0:
            raise SystemExit("--max-work-units must be non-negative")
        if (
            not math.isfinite(args.poll_interval_seconds)
            or args.poll_interval_seconds <= 0
        ):
            raise SystemExit("--poll-interval-seconds must be positive")
        if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
            raise SystemExit("--timeout-seconds must be positive")
        if args.status_page and (args.status_port < 0 or args.status_port > 65535):
            raise SystemExit("--status-port must be in [0, 65535]")


def _load_provider_snapshots(
    paths: list[str],
    *,
    mode: TrainingMode,
    stable_group_id: str,
) -> tuple[ProviderSnapshot, ...]:
    from crowdtensor.adapters.providers import legacy_capability_to_snapshots

    snapshots: list[ProviderSnapshot] = []
    for index, raw_path in enumerate(paths):
        path = Path(raw_path).expanduser()
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("provider_capability_object_required")
            if record.get("schema") == PROVIDER_SNAPSHOT_SCHEMA:
                snapshots.append(ProviderSnapshot.from_dict(record))
                continue
            availability = (
                ResourceAvailability.STABLE_WINDOW
                if mode is TrainingMode.STABLE_SHARDED
                else ResourceAvailability.INTERMITTENT
            )
            snapshots.extend(
                legacy_capability_to_snapshots(
                    record,
                    provider_id=f"capability-{index + 1}",
                    availability=availability,
                    stable_group_id=(
                        stable_group_id
                        if availability is ResourceAvailability.STABLE_WINDOW
                        else None
                    ),
                )
            )
    resource_ids = [item.resource_id for item in snapshots]
    if len(resource_ids) != len(set(resource_ids)):
        raise ValueError("provider_capability_resource_conflict")
    return tuple(snapshots)


def _build_training_v2_plan(args: argparse.Namespace) -> dict[str, Any]:
    from crowdtensor.backends.registry import get_training_backend
    from crowdtensor.core.plugins import StableShardedBackend

    project = load_project(args.job)
    providers = _load_provider_snapshots(
        list(args.capability),
        mode=project.mode,
        stable_group_id=str(args.stable_group_id),
    )
    backend = get_training_backend(project.training_backend)
    plan_options = {}
    if not args.runtime_probe:
        plan_options["runtime_probe"] = {"available": False, "version": ""}
    plan = backend.build_plan(project, providers, **plan_options)
    launch = None
    materialization = None
    if project.mode is TrainingMode.STABLE_SHARDED:
        if not isinstance(backend, StableShardedBackend):
            raise ValueError("stable_sharded_backend_launch_protocol_missing")
        launch = backend.build_launch_spec(
            plan,
            trainer_entrypoint=args.trainer_entrypoint,
            trainer_contract_verified=args.trainer_contract_verified,
            transformer_layer_class=args.transformer_layer_class or None,
            mixed_precision=args.mixed_precision,
            max_restarts=args.max_restarts,
        )
        if args.materialize:
            materialization = backend.materialize_launch(
                launch, workspace=Path(args.job)
            )
    execution_ready = launch.execution_ready if launch is not None else plan.execution_ready
    report = {
        "schema": "crowdtensor_training_plan_command_v2",
        "command_ok": True,
        "project_hash": project.content_hash,
        "mode": project.mode.value,
        "backend_id": project.training_backend,
        "provider_snapshot_count": len(providers),
        "runtime_probe_performed": bool(args.runtime_probe),
        "plan": plan.to_dict(),
        "launch": launch.to_dict() if launch is not None else None,
        "materialization": materialization,
        "execution_ready": execution_ready,
        "command_executed": False,
        "credential_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    from .contracts import stable_hash

    report["content_hash"] = stable_hash(report)
    record_plan(args.job, report)
    return report


def execute_training_v2_action(args: argparse.Namespace) -> dict[str, Any]:
    if args.train_action == "backends":
        from crowdtensor.backends.registry import backend_registry_report

        return backend_registry_report()
    if args.train_action == "init":
        mode = args.mode.replace("-", "_")
        backend = args.training_backend
        automatic_backend = backend == "auto"
        if backend == "auto":
            backend = (
                "volunteer_peft"
                if mode == TrainingMode.ELASTIC_DELTA.value
                else "accelerate_fsdp2"
            )
        optimization_plugins = list(args.optimization_plugins)
        if (
            automatic_backend
            and backend == "volunteer_peft"
            and "peft_lora_v1" not in optimization_plugins
        ):
            optimization_plugins.insert(0, "peft_lora_v1")
        return init_project(
            args.workspace,
            model=args.model,
            model_revision=args.model_revision,
            dataset=args.dataset,
            dataset_revision=args.dataset_revision,
            model_adapter=args.model_adapter,
            training_backend=backend,
            mode=mode,
            target_steps=args.target_steps,
            project_id=args.project_id or None,
            optimization_plugins=tuple(optimization_plugins),
        )
    if args.train_action == "inspect":
        return inspect_workspace(args.workspace)
    if args.train_action == "plan":
        return _build_training_v2_plan(args)
    if args.train_action in {"run", "start"}:
        if getattr(args, "campaign_dir", ""):
            from crowdtensor.backends.volunteer_session import run_volunteer_session

            return run_volunteer_session(
                args.workspace,
                campaign_dir=args.campaign_dir,
                host=args.host,
                port=args.port,
                public_url=args.public_url,
                public_release_dir=args.release_dir or None,
                prepare_only=bool(
                    getattr(args, "prepare_only", False)
                    or getattr(args, "dry_run", False)
                ),
                require_https=args.require_https,
                trust_forwarded_headers=args.trust_forwarded_headers,
                trusted_proxy_id=args.trusted_proxy_id,
                upload_chunk_bytes=args.upload_chunk_bytes,
            )
        project = load_project(args.workspace)
        if project.mode is TrainingMode.STABLE_SHARDED:
            from crowdtensor.backends.registry import get_training_backend

            backend = get_training_backend(project.training_backend)
            runner = getattr(backend, "run_session", None)
            if not callable(runner):
                raise ValueError("stable_sharded_backend_execution_protocol_missing")
            return runner(
                args.workspace,
                steps_per_work_unit=args.work_unit_steps,
                max_work_units=args.max_work_units,
                timeout_seconds=args.execution_timeout_seconds,
                dry_run=bool(getattr(args, "dry_run", False)),
            )
        return run_workspace(args.workspace)
    if args.train_action == "join":
        if getattr(args, "invite", "") or getattr(args, "coordinator_url", ""):
            from crowdtensor.backends.volunteer_session import join_volunteer_session

            return join_volunteer_session(
                args.workspace,
                invite=args.invite,
                coordinator_url=args.coordinator_url,
                pairing_code=args.pairing_code,
                cell_id=args.cell_id,
                device=args.device,
                max_local_steps=args.max_local_steps,
                max_download_bytes=int(args.max_download_gib * 1024**3),
                max_work_units=args.max_work_units,
                poll_interval_seconds=args.poll_interval_seconds,
                timeout_seconds=args.timeout_seconds,
                cache_dir=args.cache_dir or None,
                status_page=bool(args.status_page),
                status_port=int(args.status_port),
                dry_run=args.dry_run,
            )
        return join_workspace(args.workspace)
    if args.train_action == "status":
        return inspect_workspace(args.workspace)
    if args.train_action == "pause":
        return pause_workspace(args.workspace, reason=getattr(args, "reason", "owner_paused"))
    if args.train_action == "resume":
        return resume_workspace(args.workspace)
    if args.train_action == "export":
        return export_workspace(args.workspace, getattr(args, "output_dir", "") or None)
    raise ValueError("training_v2_action_unsupported")


def run_training_v2_action(args: argparse.Namespace) -> int:
    try:
        report = execute_training_v2_action(args)
    except Exception as exc:
        safe_detail = str(exc)
        public_error_types = {
            "SessionControllerError",
            "StableShardedSessionError",
            "VolunteerProtocolError",
            "VolunteerSessionError",
        }
        blocker = (
            safe_detail
            if type(exc).__name__ in public_error_types
            and re.fullmatch(r"[a-z0-9_:-]{1,160}", safe_detail)
            else f"workspace_{args.train_action}_failed:{type(exc).__name__}"
        )
        report = {
            "schema": "crowdtensor_training_workspace_command_error_v2",
            "command_ok": False,
            "action": args.train_action,
            "blocker": blocker,
            "public_artifact_safe": True,
            "private_paths_public": False,
            "credential_values_public": False,
        }
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print(
                f"training v2 action={args.train_action} ok=False "
                f"blocker={report['blocker']}"
            )
        return 1
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        command_ok = report.get("command_ok", True) is True
        print(f"training v2 action={args.train_action} ok={command_ok}")
        if args.train_action == "backends":
            print(
                "  backends="
                + ",".join(item["backend_id"] for item in report["backends"])
            )
        elif args.train_action == "plan":
            print(f"  workspace={args.job}")
            print(f"  mode={report.get('mode', '')}")
            print(f"  backend={report.get('backend_id', '')}")
            print(f"  execution_ready={report.get('execution_ready', False)}")
        elif args.train_action in {"run", "start", "join", "pause", "resume"}:
            print(f"  state={report.get('state', '')}")
            print(f"  blockers={','.join(report.get('blockers') or []) or 'none'}")
        elif args.train_action == "export":
            print(f"  artifacts={report.get('artifact_count', 0)}")
        else:
            status = report.get("status", report)
            project = status.get("project", {})
            print(f"  workspace={args.workspace}")
            print(f"  project={project.get('project_id', '')}")
            print(f"  mode={status.get('mode', '')}")
            print(f"  state={status.get('execution_state', '')}")
    return 0 if report.get("command_ok", True) is True else 1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="crowdtensor train")
    subparsers = parser.add_subparsers(dest="train_action", required=True)
    add_training_v2_parsers(subparsers)
    plan = subparsers.add_parser(
        "plan", help="Build a v2 backend and provider execution plan."
    )
    plan.add_argument("job")
    add_training_v2_plan_arguments(plan)
    plan.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    validate_training_v2_args(args)
    raise SystemExit(run_training_v2_action(args))


if __name__ == "__main__":
    main()
