"""Ordinary Miner entrypoint for Elastic Volunteer Training Beta."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import signal
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .qwen15b_four_gpu_worker import run_elastic_kernel_role
from .qwen15b_training import MODEL_ID, MODEL_REVISION, sha256_bytes, stable_hash
from .heterogeneous_training_manifest import validate_training_manifest
from .heterogeneous_training_miner import run_heterogeneous_miner


MINER_SCHEMA = "crowdtensor_elastic_training_beta_miner_join_v1"


def _request_json(
    coordinator: str,
    path: str,
    *,
    token: str,
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{str(coordinator).rstrip('/')}{path}",
        headers={
            "User-Agent": "crowdtensor-elastic-training-beta-miner/1",
            "x-crowdtensor-miner-token": token,
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:240]
        raise RuntimeError(
            f"elastic_training_beta_bootstrap_http_{exc.code}:{detail}"
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError("elastic_training_beta_coordinator_unreachable") from exc
    if not isinstance(value, dict):
        raise RuntimeError("elastic_training_beta_response_invalid")
    return value


def discover_training_capability() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("elastic_training_beta_miner_requires_torch") from exc
    cuda_available = bool(torch.cuda.is_available())
    gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
    names = [str(torch.cuda.get_device_name(index)) for index in range(gpu_count)]
    memory = [
        int(torch.cuda.get_device_properties(index).total_memory)
        for index in range(gpu_count)
    ]
    report = {
        "schema": "crowdtensor_elastic_training_beta_miner_capability_v1",
        "accelerator": "cuda" if cuda_available else "cpu",
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "gpu_name_hashes": [
            "sha256:" + hashlib.sha256(name.encode("utf-8")).hexdigest()
            for name in names
        ],
        "gpu_memory_bytes": memory,
        "two_stage_slots": min(2, gpu_count),
        "automatic_role_assignment": True,
        "raw_gpu_names_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    report["compatible"] = bool(cuda_available and gpu_count >= 2)
    return report


def _write_private(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, separators=(",", ":"), sort_keys=True), encoding="utf-8"
    )
    path.chmod(0o600)


def _run_heterogeneous_join(
    args: argparse.Namespace,
    *,
    coordinator: str,
    token: str,
    bootstrap: dict[str, Any],
    status: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    manifest = validate_training_manifest(bootstrap.get("training_manifest"))
    if (
        bootstrap.get("model_id") != manifest["model"]["model_id"]
        or bootstrap.get("model_revision") != manifest["model"]["model_revision"]
        or int(bootstrap.get("target_steps") or 0)
        != int(manifest["training"]["target_steps"])
        or int(bootstrap.get("microbatches_per_step") or 0)
        != int(manifest["training"]["microbatches_per_step"])
        or bootstrap.get("checkpoint_signatures_required") is not True
        or bootstrap.get("checkpoint_tensor_validation_required") is not True
    ):
        raise RuntimeError("heterogeneous_training_beta_bootstrap_contract_invalid")
    config = dict(bootstrap.get("config") or {})
    tokenized = dict(bootstrap.get("tokenized_payload") or {})
    if (
        stable_hash(config) != str(bootstrap.get("config_hash") or "")
        or stable_hash(tokenized)
        != str(bootstrap.get("tokenized_payload_hash") or "")
    ):
        raise RuntimeError("heterogeneous_training_beta_bootstrap_hash_mismatch")
    committed_step = int(status.get("committed_step") or 0)
    if committed_step >= int(manifest["training"]["target_steps"]):
        return {
            "schema": "crowdtensor_heterogeneous_training_miner_join_v1",
            "ok": True,
            "job_id": str(bootstrap["job_id"]),
            "joined": False,
            "training_already_completed": True,
            "committed_step": committed_step,
            "credential_values_public": False,
            "public_artifact_safe": True,
        }
    root = Path(args.private_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    session_root = root / ("heterogeneous-session-" + secrets.token_hex(8))
    session_root.mkdir(mode=0o700)
    drain_event = threading.Event()
    previous_handlers: dict[int, Any] = {}

    def request_drain(_signum: int, _frame: Any) -> None:
        drain_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_drain)
    drain_path = Path(args.drain_file).expanduser().resolve() if args.drain_file else None

    def drain_requested() -> bool:
        return drain_event.is_set() or bool(drain_path and drain_path.exists())

    miner_id_source = str(args.miner_id or f"{socket.gethostname()}:{os.getpid()}")
    miner_id_hash = stable_hash({"miner_id": miner_id_source})
    try:
        worker = run_heterogeneous_miner(
            coordinator_url=coordinator,
            coordinator_token=token,
            run_id=str(bootstrap["run_id"]),
            miner_id_hash=miner_id_hash,
            registration_nonce=secrets.token_urlsafe(32),
            training_manifest=manifest,
            config=config,
            tokenized_payload=tokenized,
            private_root=session_root,
            device_policy=args.device_policy,
            cuda_devices=list(args.cuda_device or []) or None,
            max_stage_count=int(args.max_stages),
            max_steps_per_session=int(args.max_steps),
            wait_timeout=float(args.wait_timeout),
            heartbeat_interval_seconds=float(args.heartbeat_interval),
            drain_requested=drain_requested,
            hf_token=str(os.environ.get(args.hf_token_env) or ""),
            attached_model_root=args.attached_model_root or None,
        )
        report = {
            "schema": "crowdtensor_heterogeneous_training_miner_join_v1",
            "ok": worker.get("ok") is True,
            "job_id": str(bootstrap["job_id"]),
            "joined": True,
            "device_policy": args.device_policy,
            "steps_completed": int(worker.get("steps_completed") or 0),
            "graceful_drain_applied": worker.get("graceful_drain_applied") is True,
            "all_completed_barriers_committed": worker.get(
                "all_completed_barriers_committed"
            )
            is True,
            "positive_lora_gradient_norms": worker.get(
                "positive_lora_gradient_norms"
            )
            is True,
            "optimizer_and_scheduler_steps_applied": worker.get(
                "optimizer_and_scheduler_steps_applied"
            )
            is True,
            "central_checkpoint_restore_count": int(
                worker.get("central_checkpoint_restore_count") or 0
            ),
            "capability": worker.get("capability"),
            "worker_report_hash": worker.get("content_hash"),
            "elapsed_seconds": time.time() - started,
            "credential_values_public": False,
            "coordinator_url_public": False,
            "raw_training_text_public": False,
            "token_ids_public": False,
            "activation_values_public": False,
            "gradient_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        report["content_hash"] = stable_hash(report)
        output = Path(args.output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "heterogeneous_training_miner_join.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if not args.keep_private_cache:
            shutil.rmtree(session_root, ignore_errors=True)


def run_training_join(args: argparse.Namespace) -> dict[str, Any]:
    invite: dict[str, Any] = {}
    if args.invite:
        try:
            invite = json.loads(
                Path(args.invite).expanduser().read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("elastic_training_beta_private_invite_invalid") from exc
        if (
            not isinstance(invite, dict)
            or invite.get("schema")
            != "crowdtensor_elastic_training_beta_private_miner_invite_v1"
        ):
            raise ValueError("elastic_training_beta_private_invite_invalid")
    token = str(
        args.token
        or os.environ.get(args.token_env)
        or invite.get("miner_token")
        or ""
    )
    coordinator = str(args.coordinator or invite.get("coordinator_url") or "")
    if not token:
        raise RuntimeError("elastic_training_beta_private_miner_token_required")
    if not coordinator:
        raise RuntimeError("elastic_training_beta_coordinator_url_required")
    started = time.time()
    bootstrap = _request_json(
        coordinator,
        "/elastic-training/bootstrap",
        token=token,
        timeout=args.http_timeout,
    )
    status = _request_json(
        coordinator,
        "/elastic-training/status",
        token=token,
        timeout=args.http_timeout,
    )
    if (
        bootstrap.get("schema")
        == "crowdtensor_heterogeneous_training_beta_miner_bootstrap_v1"
    ):
        return _run_heterogeneous_join(
            args,
            coordinator=coordinator,
            token=token,
            bootstrap=bootstrap,
            status=status,
            started=started,
        )
    if (
        bootstrap.get("schema")
        != "crowdtensor_elastic_training_beta_miner_bootstrap_v1"
        or bootstrap.get("model_id") != MODEL_ID
        or bootstrap.get("model_revision") != MODEL_REVISION
        or int(bootstrap.get("target_steps") or 0) != 8
        or int(bootstrap.get("microbatches_per_step") or 0) != 4
        or bootstrap.get("checkpoint_signatures_required") is not True
        or bootstrap.get("checkpoint_tensor_validation_required") is not True
    ):
        raise RuntimeError("elastic_training_beta_bootstrap_contract_invalid")
    capability = discover_training_capability()
    if capability["compatible"] is not True:
        raise RuntimeError("elastic_training_beta_two_cuda_devices_required")
    committed_step = int(status.get("committed_step") or 0)
    target_steps = int(bootstrap["target_steps"])
    if committed_step >= target_steps:
        return {
            "schema": MINER_SCHEMA,
            "ok": True,
            "job_id": str(bootstrap["job_id"]),
            "joined": False,
            "training_already_completed": True,
            "committed_step": committed_step,
            "capability": capability,
            "credential_values_public": False,
            "coordinator_url_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
    root = Path(args.private_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    session_root = root / (
        "session-" + secrets.token_hex(8)
    )
    session_root.mkdir(parents=True, exist_ok=False)
    session_root.chmod(0o700)
    config_path = session_root / "config.json"
    tokenized_path = session_root / "qwen15b_tokenized_private.json"
    config = dict(bootstrap.get("config") or {})
    tokenized = dict(bootstrap.get("tokenized_payload") or {})
    _write_private(config_path, config)
    _write_private(tokenized_path, tokenized)
    if (
        stable_hash(config) != str(bootstrap["config_hash"])
        or stable_hash(tokenized) != str(bootstrap["tokenized_payload_hash"])
    ):
        shutil.rmtree(session_root, ignore_errors=True)
        raise RuntimeError("elastic_training_beta_bootstrap_hash_mismatch")
    drain_event = threading.Event()
    previous_handlers: dict[int, Any] = {}

    def request_drain(_signum: int, _frame: Any) -> None:
        drain_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_drain)
    drain_path = Path(args.drain_file).expanduser().resolve() if args.drain_file else None

    def drain_requested() -> bool:
        return drain_event.is_set() or bool(drain_path and drain_path.exists())

    miner_id_source = str(args.miner_id or f"{socket.gethostname()}:{os.getpid()}")
    miner_id_hash = stable_hash({"miner_id": miner_id_source})
    try:
        worker = run_elastic_kernel_role(
            role=args.role,
            coordinator_url=coordinator,
            coordinator_token=token,
            run_id=str(bootstrap["run_id"]),
            miner_id_hash=miner_id_hash,
            registration_nonce=secrets.token_urlsafe(32),
            expected_start_step=committed_step,
            segment_end_step=target_steps,
            config=config,
            tokenized_payload_path=tokenized_path,
            private_root=session_root,
            target_steps=target_steps,
            microbatch_count=int(bootstrap["microbatches_per_step"]),
            wait_timeout=float(args.wait_timeout),
            heartbeat_interval_seconds=float(args.heartbeat_interval),
            drain_requested=drain_requested,
            max_steps_per_session=int(args.max_steps),
        )
        report = {
            "schema": MINER_SCHEMA,
            "ok": worker.get("ok") is True,
            "job_id": str(bootstrap["job_id"]),
            "joined": True,
            "role": str(worker.get("role") or ""),
            "requested_role": args.role,
            "expected_start_step": committed_step,
            "segment_end_step": int(worker.get("segment_end_step") or committed_step),
            "target_steps": target_steps,
            "graceful_drain_applied": worker.get("graceful_drain_applied") is True,
            "barrier_commit_count": len(worker.get("barrier_commits") or []),
            "all_completed_barriers_committed": worker.get(
                "all_completed_barriers_committed"
            )
            is True,
            "central_checkpoint_restore_verified": worker.get(
                "central_checkpoint_restore_verified"
            )
            is True,
            "base_weights_frozen": worker.get("base_weights_frozen") is True,
            "positive_lora_gradient_norms": worker.get(
                "positive_lora_gradient_norms"
            )
            is True,
            "standard_peft_export_verified": (
                (worker.get("export") or {}).get("standard_peft_format") is True
            ),
            "evaluation_verified": (
                (worker.get("evaluation") or {}).get("evaluation_verified") is True
            ),
            "capability": capability,
            "worker_report_hash": str(worker.get("content_hash") or ""),
            "elapsed_seconds": time.time() - started,
            "credential_values_public": False,
            "coordinator_url_public": False,
            "raw_training_text_public": False,
            "token_ids_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        report["content_hash"] = stable_hash(report)
        output = Path(args.output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "elastic_training_miner_join.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if not args.keep_private_cache:
            shutil.rmtree(session_root, ignore_errors=True)


def parse_training_join_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="crowdtensor-miner join",
        description="Join an Elastic Volunteer Training Beta job.",
    )
    parser.add_argument("--training", action="store_true", required=True)
    parser.add_argument("--coordinator", default="")
    parser.add_argument("--invite", default="")
    parser.add_argument("--token", default="")
    parser.add_argument("--token-env", default="CROWDTENSOR_MINER_TOKEN")
    parser.add_argument("--miner-id", default="")
    parser.add_argument("--role", choices=["auto", "kernel_a", "kernel_b"], default="auto")
    parser.add_argument(
        "--device-policy",
        choices=["auto", "cpu", "cuda", "jax_tpu", "mixed"],
        default="auto",
    )
    parser.add_argument("--cuda-device", action="append", type=int, default=[])
    parser.add_argument("--max-stages", type=int, default=0)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--attached-model-root", default="")
    parser.add_argument("--private-root", default="state/elastic-training-miner")
    parser.add_argument("--output-dir", default="dist/elastic-training-miner")
    parser.add_argument("--drain-file", default="")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--wait-timeout", type=float, default=900.0)
    parser.add_argument("--http-timeout", type=float, default=120.0)
    parser.add_argument("--heartbeat-interval", type=float, default=5.0)
    parser.add_argument("--keep-private-cache", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if not args.coordinator and not args.invite:
        parser.error("--coordinator or --invite is required")
    if args.max_steps < 0:
        parser.error("--max-steps must be non-negative")
    if args.max_stages < 0:
        parser.error("--max-stages must be non-negative")
    if any(index < 0 for index in args.cuda_device):
        parser.error("--cuda-device must be non-negative")
    if args.wait_timeout <= 0 or args.http_timeout <= 0 or args.heartbeat_interval <= 0:
        parser.error("timeouts and heartbeat interval must be positive")
    return args


def training_join_main(argv: list[str]) -> None:
    args = parse_training_join_args(argv)
    try:
        report = run_training_join(args)
    except BaseException as exc:
        report = {
            "schema": MINER_SCHEMA,
            "ok": False,
            "blockers": [f"elastic_training_miner_join_failed:{type(exc).__name__}"],
            "failure_detail_public": False,
            "credential_values_public": False,
            "coordinator_url_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
    if args.json:
        print(json.dumps(report, sort_keys=True), flush=True)
    else:
        print(
            f"elastic training join ok={report.get('ok')} "
            f"role={report.get('role', 'unassigned')} "
            f"step={report.get('segment_end_step', 0)}/{report.get('target_steps', 0)}",
            flush=True,
        )
        for blocker in report.get("blockers") or []:
            print(f"  blocker={blocker}", flush=True)
    raise SystemExit(0 if report.get("ok") else 1)
