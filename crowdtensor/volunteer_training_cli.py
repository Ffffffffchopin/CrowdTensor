"""CLI for operating and joining Volunteer Training Protocol Alpha campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any

import uvicorn
import httpx

from .community_security import TLSProxyPolicy
from .hf_lora_training import create_local_training_fixture
from .volunteer_training_api import create_volunteer_training_app, service_contract
from .volunteer_agent_status import (
    VolunteerAgentStatusServer,
    graceful_agent_signals,
)
from .volunteer_campaign_proposal import (
    load_and_validate_proposal,
    write_proposal_template,
)
from .volunteer_training_campaign import create_pinned_smollm_wikitext_fixture
from .volunteer_training_cell import (
    HTTPVolunteerTransport,
    VolunteerTrainingCell,
    detect_hardware,
)
from .volunteer_training_coordinator import VolunteerTrainingCoordinator
from .volunteer_training_storage import S3VolunteerBlobStore
from .volunteer_training_protocol import (
    INVITE_SCHEMA,
    VolunteerProtocolError,
    hash_cell_id,
    public_error,
    validate_campaign_manifest,
    with_public_safety,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="crowdtensor volunteer",
        description="Join or operate a low-frequency LoRA volunteer training campaign.",
    )
    actions = parser.add_subparsers(dest="action", required=True)

    join = actions.add_parser(
        "join",
        help="Join from a Coordinator URL plus one-time code, or a legacy private invite.",
    )
    join.add_argument("campaign", help="Coordinator HTTPS URL or legacy private invite JSON")
    join.add_argument("--code", default="", help="one-time Agent pairing code")
    join.add_argument("--workspace", default="")
    join.add_argument("--cell-id", default="")
    join.add_argument("--device", default="auto")
    join.add_argument("--max-local-steps", type=int, default=64)
    join.add_argument("--max-download-gib", type=float, default=8.0)
    join.add_argument("--max-work-units", type=int, default=0)
    join.add_argument("--cache-dir", default="")
    join.add_argument("--once", action="store_true")
    join.add_argument("--poll-interval-seconds", type=float, default=2.0)
    join.add_argument("--timeout-seconds", type=float, default=120.0)
    join.add_argument(
        "--status-page", action=argparse.BooleanOptionalAction, default=True
    )
    join.add_argument("--status-port", type=int, default=8765)
    join.add_argument("--dry-run", action="store_true")
    join.add_argument(
        "--test-proxy-id", default="", help=argparse.SUPPRESS
    )
    join.add_argument(
        "--test-interrupt-upload-after-chunks",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    join.add_argument("--json", action="store_true")

    for name in ("status", "pause", "resume", "cleanup"):
        item = actions.add_parser(name)
        item.add_argument("workspace")
        item.add_argument("--json", action="store_true")

    remote_status = actions.add_parser("campaign-status")
    remote_status.add_argument("campaign", help="private invite JSON file")
    remote_status.add_argument("--timeout-seconds", type=float, default=30.0)
    remote_status.add_argument("--json", action="store_true")

    pair_code = actions.add_parser(
        "pair-code", help="Create one short-lived, single-use contributor pairing code."
    )
    pair_code.add_argument("campaign_dir")
    pair_code.add_argument("--mode", choices=("agent", "browser"), default="agent")
    pair_code.add_argument("--ttl-seconds", type=int, default=3600)
    pair_code.add_argument("--json", action="store_true")

    campaign = actions.add_parser("campaign")
    campaign_actions = campaign.add_subparsers(dest="campaign_action", required=True)
    create = campaign_actions.add_parser(
        "create-local",
        help="Create a bounded local PEFT campaign for evaluation or private deployment.",
    )
    create.add_argument("campaign_dir")
    create.add_argument("--campaign-id", default="")
    create.add_argument("--target-rounds", type=int, default=2)
    create.add_argument("--local-steps", type=int, default=2)
    create.add_argument("--lease-seconds", type=float, default=300.0)
    create.add_argument("--json", action="store_true")

    imported = campaign_actions.add_parser(
        "import-smollm-wikitext",
        help="Import the pinned public SmolLM2-135M and WikiText-2 Campaign.",
    )
    imported.add_argument("campaign_dir")
    imported.add_argument("--campaign-id", default="")
    imported.add_argument("--target-rounds", type=int, default=3)
    imported.add_argument("--local-steps", type=int, default=1)
    imported.add_argument("--sequence-length", type=int, default=16)
    imported.add_argument("--train-sequences", type=int, default=12)
    imported.add_argument("--validation-sequences", type=int, default=4)
    imported.add_argument("--lease-seconds", type=float, default=900.0)
    imported.add_argument("--json", action="store_true")

    for name in ("validate", "start", "pause", "resume", "finalize", "evaluate"):
        operation = campaign_actions.add_parser(name)
        operation.add_argument("campaign_dir")
        operation.add_argument("--json", action="store_true")
    export = campaign_actions.add_parser("export")
    export.add_argument("campaign_dir")
    export.add_argument("destination")
    export.add_argument("--json", action="store_true")
    backup = campaign_actions.add_parser("backup")
    backup.add_argument("campaign_dir")
    backup.add_argument("destination")
    backup.add_argument("--json", action="store_true")
    restore = campaign_actions.add_parser("restore")
    restore.add_argument("backup")
    restore.add_argument("destination")
    restore.add_argument("--json", action="store_true")
    proposal_template = campaign_actions.add_parser(
        "proposal-template",
        help="Write a public Campaign proposal template with licensing and governance fields.",
    )
    proposal_template.add_argument("destination")
    proposal_template.add_argument("--json", action="store_true")
    validate_proposal = campaign_actions.add_parser(
        "validate-proposal",
        help="Validate a public Campaign proposal before contributor recruitment.",
    )
    validate_proposal.add_argument("proposal")
    validate_proposal.add_argument("--json", action="store_true")

    serve = actions.add_parser("serve")
    serve.add_argument("campaign_dir")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8789)
    serve.add_argument("--public-url", default="")
    serve.add_argument("--prepare-only", action="store_true")
    serve.add_argument(
        "--require-https", action=argparse.BooleanOptionalAction, default=None
    )
    serve.add_argument("--trust-forwarded-headers", action="store_true")
    serve.add_argument("--trusted-proxy-id", default="")
    serve.add_argument("--upload-chunk-bytes", type=int, default=1024 * 1024)
    serve.add_argument(
        "--upload-storage", choices=("local", "s3"), default="local"
    )
    serve.add_argument("--s3-bucket", default="")
    serve.add_argument("--s3-prefix", default="crowdtensor/volunteer")
    serve.add_argument("--s3-endpoint", default="")
    serve.add_argument("--s3-region", default="us-east-1")
    serve.add_argument("--s3-access-key-env", default="AWS_ACCESS_KEY_ID")
    serve.add_argument("--s3-secret-key-env", default="AWS_SECRET_ACCESS_KEY")
    serve.add_argument("--s3-session-token-env", default="AWS_SESSION_TOKEN")
    serve.add_argument("--json", action="store_true")

    operator = actions.add_parser(
        "operator",
        help="Create if needed and run one campaign Coordinator in one command.",
    )
    operator.add_argument("campaign_dir")
    operator.add_argument(
        "--profile", choices=("local", "smollm-wikitext"), default="local"
    )
    operator.add_argument("--campaign-id", default="")
    operator.add_argument("--target-rounds", type=int, default=2)
    operator.add_argument("--local-steps", type=int, default=1)
    operator.add_argument("--host", default="127.0.0.1")
    operator.add_argument("--port", type=int, default=8789)
    operator.add_argument("--public-url", default="")
    operator.add_argument("--prepare-only", action="store_true")
    operator.add_argument(
        "--require-https", action=argparse.BooleanOptionalAction, default=None
    )
    operator.add_argument("--trust-forwarded-headers", action="store_true")
    operator.add_argument("--trusted-proxy-id", default="")
    operator.add_argument("--upload-chunk-bytes", type=int, default=1024 * 1024)
    operator.add_argument(
        "--upload-storage", choices=("local", "s3"), default="local"
    )
    operator.add_argument("--s3-bucket", default="")
    operator.add_argument("--s3-prefix", default="crowdtensor/volunteer")
    operator.add_argument("--s3-endpoint", default="")
    operator.add_argument("--s3-region", default="us-east-1")
    operator.add_argument("--s3-access-key-env", default="AWS_ACCESS_KEY_ID")
    operator.add_argument("--s3-secret-key-env", default="AWS_SECRET_ACCESS_KEY")
    operator.add_argument("--s3-session-token-env", default="AWS_SESSION_TOKEN")
    operator.add_argument("--json", action="store_true")

    contract = actions.add_parser("contract")
    contract.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _read_invite(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != INVITE_SCHEMA:
        raise VolunteerProtocolError("volunteer_invite_schema_mismatch", status_code=400)
    return value


def _default_workspace(invite: dict[str, Any]) -> Path:
    campaign_id = str(invite.get("campaign_id") or "campaign")
    suffix = __import__("hashlib").sha256(campaign_id.encode("utf-8")).hexdigest()[:16]
    return Path.home() / ".cache" / "crowdtensor" / "volunteer" / suffix


def _is_coordinator_url(value: str) -> bool:
    text = str(value).strip().lower()
    return text.startswith("https://") or text.startswith("http://127.0.0.1") or text.startswith(
        "http://localhost"
    )


def _campaign_from_url(
    coordinator_url: str,
    *,
    timeout_seconds: float,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    response = httpx.get(
        str(coordinator_url).rstrip("/") + "/v1/volunteer/campaign",
        headers=dict(extra_headers or {}),
        timeout=float(timeout_seconds),
    )
    return HTTPVolunteerTransport._response(response)


def _cell_id_for_workspace(workspace: Path, requested: str) -> str:
    state_path = workspace / ".private" / "cell_state.json"
    if state_path.is_file():
        value = json.loads(state_path.read_text(encoding="utf-8"))
        cell_id = str(value.get("cell_id") or "") if isinstance(value, dict) else ""
        if cell_id:
            return cell_id
    return str(requested or "agent-" + secrets.token_hex(12))


def _load_agent_enrollment(
    workspace: Path, *, coordinator_url: str, cell_id: str
) -> dict[str, Any] | None:
    path = workspace / ".private" / "agent_enrollment.json"
    if not path.is_file() or path.stat().st_mode & 0o077:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        return None
    if (
        value.get("schema") != "crowdtensor_volunteer_agent_enrollment_v1"
        or str(value.get("coordinator_url") or "").rstrip("/")
        != str(coordinator_url).rstrip("/")
        or value.get("cell_id") != cell_id
        or float(value.get("expires_at") or 0.0) <= time.time() + 5.0
        or not str(value.get("credential_token") or "")
    ):
        return None
    return value


def _save_agent_enrollment(workspace: Path, value: dict[str, Any]) -> Path:
    path = workspace / ".private" / "agent_enrollment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _existing_cell(workspace: str | Path) -> VolunteerTrainingCell:
    class _OfflineTransport:
        def __getattr__(self, _name: str) -> Any:
            raise RuntimeError("volunteer transport is unavailable for this local action")

    return VolunteerTrainingCell(_OfflineTransport(), workspace)


def _emit(value: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(value, indent=2, sort_keys=True))
        return
    print(
        "volunteer"
        f" state={value.get('state', value.get('overall_state', 'ready'))}"
        f" ok={bool(value.get('ok', True))}"
    )
    if value.get("campaign_id"):
        print(f"campaign_id={value['campaign_id']}")
    if value.get("adapter_version") is not None:
        print(f"adapter_version={value['adapter_version']}")
    if value.get("error"):
        print(f"error={value['error']}")
    if value.get("pairing_code"):
        print(f"pairing_code={value['pairing_code']}")


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.action == "contract":
            value = service_contract()
        elif args.action == "campaign" and args.campaign_action == "create-local":
            root = Path(args.campaign_dir).expanduser().resolve()
            fixture = create_local_training_fixture(
                root / ".private" / "fixture",
                job_id=args.campaign_id or "volunteer-training-local-alpha",
                local_steps=int(args.local_steps),
            )
            coordinator = VolunteerTrainingCoordinator.create_from_fixture(
                root,
                fixture,
                campaign_id=args.campaign_id,
                target_rounds=int(args.target_rounds),
                lease_seconds=float(args.lease_seconds),
            )
            manifest = coordinator.campaign_manifest()
            value = with_public_safety(
                {
                    "schema": "crowdtensor_volunteer_training_campaign_create_v1",
                    "ok": True,
                    "state": "created",
                    "campaign_id": manifest["campaign_id"],
                    "campaign_manifest_hash": manifest["manifest_hash"],
                    "target_rounds": manifest["round_policy"]["target_rounds"],
                    "private_invite_created": coordinator.invite_path.is_file(),
                    "next_action": "serve_campaign_then_share_private_invite",
                }
            )
            _emit(value, json_output=bool(args.json))
            if not args.json:
                print(f"private_invite={coordinator.invite_path}")
            return 0
        elif (
            args.action == "campaign"
            and args.campaign_action == "import-smollm-wikitext"
        ):
            root = Path(args.campaign_dir).expanduser().resolve()
            state_path = root / ".private" / "coordinator_state.json"
            if state_path.is_file():
                coordinator = VolunteerTrainingCoordinator(root)
            else:
                fixture = create_pinned_smollm_wikitext_fixture(
                    root / ".private" / "fixture",
                    job_id=args.campaign_id or "volunteer-smollm2-wikitext-beta",
                    sequence_length=int(args.sequence_length),
                    train_sequence_count=int(args.train_sequences),
                    validation_sequence_count=int(args.validation_sequences),
                    local_steps=int(args.local_steps),
                )
                coordinator = VolunteerTrainingCoordinator.create_from_fixture(
                    root,
                    fixture,
                    campaign_id=args.campaign_id,
                    target_rounds=int(args.target_rounds),
                    lease_seconds=float(args.lease_seconds),
                    outer_lr=0.5,
                    momentum=0.0,
                )
            manifest = coordinator.campaign_manifest()
            value = with_public_safety(
                {
                    "schema": "crowdtensor_volunteer_training_campaign_import_result_v1",
                    "ok": True,
                    "state": "imported",
                    "campaign_id": manifest["campaign_id"],
                    "campaign_manifest_hash": manifest["manifest_hash"],
                    "import_profile": (manifest.get("campaign_import") or {}).get(
                        "profile"
                    ),
                    "model_adapter_id": manifest.get("model_adapter_id"),
                    "model_source_verified": (manifest.get("model_source") or {}).get(
                        "source_verified"
                    )
                    is True,
                    "dataset_source_verified": (
                        manifest.get("dataset_source") or {}
                    ).get("source_verified")
                    is True,
                    "target_rounds": manifest["round_policy"]["target_rounds"],
                    "private_invite_created": coordinator.invite_path.is_file(),
                    "next_action": "serve_campaign_then_share_private_invite",
                }
            )
            _emit(value, json_output=bool(args.json))
            if not args.json:
                print(f"private_invite={coordinator.invite_path}")
            return 0
        elif args.action == "campaign" and args.campaign_action == "restore":
            _coordinator, value = VolunteerTrainingCoordinator.restore_campaign(
                args.backup, args.destination
            )
        elif (
            args.action == "campaign"
            and args.campaign_action == "proposal-template"
        ):
            value = write_proposal_template(args.destination)
        elif (
            args.action == "campaign"
            and args.campaign_action == "validate-proposal"
        ):
            value = load_and_validate_proposal(args.proposal)
        elif args.action == "campaign" and args.campaign_action in {
            "validate",
            "start",
            "pause",
            "resume",
            "finalize",
            "evaluate",
            "export",
            "backup",
        }:
            coordinator = VolunteerTrainingCoordinator(args.campaign_dir)
            token = coordinator.private_invite()["invite_token"]
            if args.campaign_action == "validate":
                value = coordinator.validate_campaign()
            elif args.campaign_action == "start":
                value = coordinator.start_campaign(invite_token=token)
            elif args.campaign_action == "pause":
                value = coordinator.pause_campaign(invite_token=token)
            elif args.campaign_action == "resume":
                value = coordinator.resume_campaign(invite_token=token)
            elif args.campaign_action == "finalize":
                value = coordinator.finalize_campaign(invite_token=token)
            elif args.campaign_action == "evaluate":
                value = coordinator.evaluate_campaign()
            elif args.campaign_action == "export":
                value = coordinator.export_campaign(args.destination)
            else:
                value = coordinator.backup_campaign(args.destination)
        elif args.action in {"serve", "operator"}:
            if args.action == "operator":
                root = Path(args.campaign_dir).expanduser().resolve()
                if not (root / ".private" / "coordinator_state.json").is_file():
                    if args.profile == "smollm-wikitext":
                        fixture = create_pinned_smollm_wikitext_fixture(
                            root / ".private" / "fixture",
                            job_id=args.campaign_id
                            or "volunteer-smollm2-wikitext-operator-beta",
                            local_steps=int(args.local_steps),
                        )
                    else:
                        fixture = create_local_training_fixture(
                            root / ".private" / "fixture",
                            job_id=args.campaign_id
                            or "volunteer-training-operator-beta",
                            local_steps=int(args.local_steps),
                        )
                    VolunteerTrainingCoordinator.create_from_fixture(
                        root,
                        fixture,
                        campaign_id=args.campaign_id,
                        target_rounds=int(args.target_rounds),
                    )
            coordinator = VolunteerTrainingCoordinator(args.campaign_dir)
            public_url = str(args.public_url or f"http://{args.host}:{args.port}").rstrip("/")
            coordinator.write_invite(public_url)
            require_https = (
                bool(args.require_https)
                if args.require_https is not None
                else public_url.startswith("https://")
            )
            proxy_hashes = (
                (
                    "sha256:"
                    + hashlib.sha256(args.trusted_proxy_id.encode("utf-8")).hexdigest(),
                )
                if args.trusted_proxy_id
                else ()
            )
            if args.trust_forwarded_headers and not proxy_hashes:
                raise VolunteerProtocolError(
                    "volunteer_trusted_proxy_identity_required", status_code=400
                )
            tls_policy = TLSProxyPolicy(
                require_https=require_https,
                trust_forwarded_headers=bool(args.trust_forwarded_headers),
                trusted_proxy_hashes=proxy_hashes,
            )
            recovery = (
                with_public_safety(
                    {
                        "ok": True,
                        "coordinator_state_reloaded": False,
                        "prepare_only": True,
                    }
                )
                if args.prepare_only
                else coordinator.recover_after_restart()
            )
            value = with_public_safety(
                {
                    "schema": "crowdtensor_volunteer_training_serve_v1",
                    "ok": True,
                    "state": "prepared" if args.prepare_only else "serving",
                    "campaign_id": coordinator.campaign_manifest()["campaign_id"],
                    "bind_host": args.host,
                    "port": int(args.port),
                    "private_invite_updated": True,
                    "external_tls_termination_required": not public_url.startswith(
                        ("http://127.0.0.1", "http://localhost")
                    ),
                    "tls_required": require_https,
                    "trusted_forwarded_headers": bool(args.trust_forwarded_headers),
                    "trusted_proxy_identity_count": len(proxy_hashes),
                    "resumable_upload_chunk_bytes": int(args.upload_chunk_bytes),
                    "upload_storage_backend": args.upload_storage,
                    "s3_compatible_upload_storage": args.upload_storage == "s3",
                    "one_command_operator_workflow": args.action == "operator",
                    "coordinator_restart_recovery_verified": recovery.get("ok")
                    is True,
                }
            )
            _emit(value, json_output=bool(args.json))
            if not args.json:
                print(f"private_invite={coordinator.invite_path}")
            if args.prepare_only:
                return 0
            upload_store = None
            if args.upload_storage == "s3":
                if not args.s3_bucket:
                    raise VolunteerProtocolError(
                        "volunteer_s3_bucket_required", status_code=400
                    )
                upload_store = S3VolunteerBlobStore(
                    bucket=args.s3_bucket,
                    prefix=args.s3_prefix,
                    endpoint_url=args.s3_endpoint,
                    region_name=args.s3_region,
                    access_key_env=args.s3_access_key_env,
                    secret_key_env=args.s3_secret_key_env,
                    session_token_env=args.s3_session_token_env,
                )
            uvicorn.run(
                create_volunteer_training_app(
                    coordinator,
                    tls_policy=tls_policy,
                    upload_chunk_bytes=int(args.upload_chunk_bytes),
                    upload_blob_store=upload_store,
                ),
                host=args.host,
                port=int(args.port),
                log_level="info",
            )
            return 0
        elif args.action == "campaign-status":
            transport = HTTPVolunteerTransport.from_invite(
                args.campaign, timeout_seconds=float(args.timeout_seconds)
            )
            value = transport.status()
        elif args.action == "pair-code":
            coordinator = VolunteerTrainingCoordinator(args.campaign_dir)
            value = coordinator.create_pairing_code(
                invite_token=coordinator.private_invite()["invite_token"],
                mode=args.mode,
                ttl_seconds=int(args.ttl_seconds),
            )
        elif args.action == "join":
            url_join = _is_coordinator_url(args.campaign)
            test_headers = (
                {
                    "X-Forwarded-Proto": "https",
                    "X-CrowdTensor-Proxy-Id": args.test_proxy_id,
                }
                if args.test_proxy_id
                else {}
            )
            invite = (
                _campaign_from_url(
                    args.campaign,
                    timeout_seconds=float(args.timeout_seconds),
                    extra_headers=test_headers,
                )
                if url_join
                else _read_invite(args.campaign)
            )
            workspace = (
                Path(args.workspace).expanduser().resolve()
                if args.workspace
                else _default_workspace(invite)
            )
            selected_cell_id = _cell_id_for_workspace(workspace, args.cell_id)
            if url_join:
                enrollment = _load_agent_enrollment(
                    workspace,
                    coordinator_url=args.campaign,
                    cell_id=selected_cell_id,
                )
                if enrollment is not None:
                    transport = HTTPVolunteerTransport.from_cell_credential(
                        args.campaign,
                        cell_id=selected_cell_id,
                        credential_token=str(enrollment["credential_token"]),
                        credential_id=str(enrollment.get("credential_id") or ""),
                        expires_at=float(enrollment["expires_at"]),
                        timeout_seconds=float(args.timeout_seconds),
                        extra_headers=test_headers,
                        interrupt_after_chunks=int(
                            args.test_interrupt_upload_after_chunks
                        ),
                    )
                else:
                    if not args.code:
                        raise VolunteerProtocolError(
                            "volunteer_pairing_code_required", status_code=401
                        )
                    transport = HTTPVolunteerTransport.from_pairing_code(
                        args.campaign,
                        args.code,
                        cell_id=selected_cell_id,
                        timeout_seconds=float(args.timeout_seconds),
                        extra_headers=test_headers,
                        interrupt_after_chunks=int(
                            args.test_interrupt_upload_after_chunks
                        ),
                    )
                    _save_agent_enrollment(workspace, transport.private_enrollment())
            else:
                transport = HTTPVolunteerTransport.from_invite(
                    args.campaign,
                    timeout_seconds=float(args.timeout_seconds),
                    extra_headers=test_headers,
                    interrupt_after_chunks=int(
                        args.test_interrupt_upload_after_chunks
                    ),
                )
            campaign = transport.campaign()
            validate_campaign_manifest(campaign)
            if args.dry_run:
                hardware = detect_hardware()
                value = with_public_safety(
                    {
                        "schema": "crowdtensor_volunteer_training_join_preflight_v1",
                        "ok": True,
                        "state": "preflight_ready",
                        "campaign_id": campaign["campaign_id"],
                        "campaign_manifest_hash": campaign["manifest_hash"],
                        "hardware": hardware,
                        "requested_device": args.device,
                        "max_local_steps": int(args.max_local_steps),
                        "max_download_bytes": int(float(args.max_download_gib) * 1024**3),
                        "work_claimed": False,
                    }
                )
            else:
                cell = VolunteerTrainingCell(
                    transport,
                    workspace,
                    cell_id=selected_cell_id,
                    device=args.device,
                    max_local_steps=int(args.max_local_steps),
                    max_download_bytes=int(float(args.max_download_gib) * 1024**3),
                    cache_dir=args.cache_dir or None,
                )
                limit = (
                    1
                    if args.once or (url_join and int(args.max_work_units) == 0)
                    else int(args.max_work_units)
                )
                if args.status_page:
                    with VolunteerAgentStatusServer(
                        cell, port=int(args.status_port)
                    ) as control:
                        if not args.json:
                            print(f"local_status={control.endpoint}")
                        with graceful_agent_signals(control.stop_event):
                            value = cell.run(
                                max_work_units=limit,
                                poll_interval_seconds=float(
                                    args.poll_interval_seconds
                                ),
                                stop_requested=control.stop_event.is_set,
                            )
                        value = with_public_safety(
                            {
                                **value,
                                "local_status_endpoint": control.endpoint,
                                "local_status_page_enabled": True,
                                "graceful_signal_stop": True,
                            }
                        )
                else:
                    value = cell.run(
                        max_work_units=limit,
                        poll_interval_seconds=float(args.poll_interval_seconds),
                    )
        else:
            cell = _existing_cell(args.workspace)
            if args.action == "status":
                value = cell.local_status()
            elif args.action == "pause":
                value = cell.pause()
            elif args.action == "resume":
                value = cell.resume()
            elif args.action == "cleanup":
                value = cell.cleanup()
            else:  # pragma: no cover - argparse makes this unreachable
                raise VolunteerProtocolError("volunteer_cli_action_invalid", status_code=400)
        _emit(value, json_output=bool(args.json))
        return 0 if value.get("ok", True) else 2
    except VolunteerProtocolError as exc:
        value = public_error(exc.code)
        _emit(value, json_output=bool(getattr(args, "json", False)))
        return 4 if exc.status_code in {401, 403, 409} else 2
    except httpx.HTTPError:
        value = public_error("volunteer_coordinator_network_unavailable")
        _emit(value, json_output=bool(getattr(args, "json", False)))
        return 5
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        value = with_public_safety(
            {
                "schema": "crowdtensor_volunteer_training_cli_error_v1",
                "ok": False,
                "error": "volunteer_runtime_failed:" + type(exc).__name__,
            }
        )
        _emit(value, json_output=bool(getattr(args, "json", False)))
        return 5


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run(argv))


if __name__ == "__main__":
    main()
