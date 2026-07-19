#!/usr/bin/env python3
"""Run a bounded same-account two-Kaggle-kernel CUDA training gate."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coordinator import create_app  # noqa: E402
from crowdtensor.hf_lora_training import create_local_training_fixture, evaluate_adapter  # noqa: E402
from crowdtensor.named_tensor_optimizer import (  # noqa: E402
    compress_sign_with_error_feedback,
    decode_sign_transport,
    export_standard_peft_adapter,
    load_tensors,
)
from crowdtensor.training_contract import public_training_spec, sha256_file, sha256_json  # noqa: E402
from crowdtensor.training_allocation_budget import require_attempt_limit  # noqa: E402
from scripts.training_cuda_kaggle_common import (  # noqa: E402
    authenticated_owner,
    delete_succeeded_or_absent,
    extract_kernel_ref,
    kaggle_env,
    public_safety_errors,
    push_accepted,
    run_command,
    safe_slug,
    status_class,
    utc_now,
)
from scripts.training_cuda_two_node_package import build_package  # noqa: E402


SCHEMA = "crowdtensor_cuda_two_node_live_probe_v1"
WORKER_REPORT = "training_cuda_two_node_worker.json"
OUTPUT_PATTERN = r"training_cuda_two_node_(worker\.json|(stage0|stage1)_checkpoint_bundle\.zip)"
TERMINAL = {"complete", "failed"}
TUNNEL_URL_PATTERN = __import__("re").compile(r"https://[-a-z0-9]+\.trycloudflare\.com")


class CoordinatorRouteError(RuntimeError):
    """Public-safe failure raised before any Kaggle allocation is attempted."""

    def __init__(self, code: str, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(str(code))
        self.code = str(code)
        self.diagnostics = dict(diagnostics or {})


class RoutePreflightComplete(RuntimeError):
    """Internal control flow for a successful preflight-only run."""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def reserve_attempt(ledger_path: Path, *, limit: int) -> int:
    ledger = _load_json(ledger_path)
    limit = require_attempt_limit(ledger, kind="two_node", requested_limit=limit)
    attempts = list(ledger.get("two_node_attempts") or [])
    if len(attempts) >= int(limit):
        raise RuntimeError("two_node_allocation_attempt_limit_reached")
    attempt = len(attempts) + 1
    attempts.append(
        {
            "attempt": attempt,
            "started_at": utc_now(),
            "allocation_started": True,
            "completed": False,
        }
    )
    ledger.update(
        {
            "schema": "crowdtensor_cuda_training_allocation_attempts_v1",
            "two_node_attempts": attempts,
            "two_node_attempt_limit": int(limit),
        }
    )
    _write_json(ledger_path, ledger)
    return attempt


def finish_attempt(ledger_path: Path, *, attempt: int, outcome: str) -> None:
    ledger = _load_json(ledger_path)
    attempts = list(ledger.get("two_node_attempts") or [])
    for record in attempts:
        if int(record.get("attempt") or 0) == int(attempt):
            record.update({"completed": True, "finished_at": utc_now(), "outcome": str(outcome)})
    ledger["two_node_attempts"] = attempts
    _write_json(ledger_path, ledger)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _get_json(
    url: str,
    *,
    timeout: float = 10.0,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request_headers = {"User-Agent": "crowdtensor-cuda-route-probe/1"}
    request_headers.update(headers or {})
    with urlopen(Request(url, headers=request_headers, method="GET"), timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("expected JSON object")
    return value


def _wait_ready(url: str, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if _get_json(f"{url.rstrip('/')}/ready", timeout=10.0).get("ok") is True:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(1.0)
    raise TimeoutError(f"Coordinator readiness timed out: {type(last_error).__name__ if last_error else 'unknown'}")


def _public_error_class(exc: BaseException) -> str:
    reason = getattr(exc, "reason", None)
    return f"{type(exc).__name__}:{type(reason).__name__}" if reason is not None else type(exc).__name__


def _probe_authenticated_route(
    url: str,
    *,
    coordinator_token: str,
    run_id: str,
    timeout: float,
) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "ready_http_ok": False,
        "miner_auth_required": False,
        "authenticated_status_ok": False,
        "run_id_hash_match": False,
        "ok": False,
    }
    try:
        ready = _get_json(f"{url.rstrip('/')}/ready", timeout=timeout)
        observation["ready_http_ok"] = ready.get("ok") is True
        observation["miner_auth_required"] = (ready.get("auth") or {}).get("miner_required") is True
        status = _get_json(
            f"{url.rstrip('/')}/cuda-training/status",
            timeout=timeout,
            headers={"x-crowdtensor-miner-token": coordinator_token},
        )
        expected_run_hash = "sha256:" + hashlib.sha256(run_id.encode("utf-8")).hexdigest()
        observation["authenticated_status_ok"] = (
            status.get("schema") == "crowdtensor_cuda_training_rendezvous_v1"
        )
        observation["run_id_hash_match"] = status.get("run_id_hash") == expected_run_hash
        observation["ok"] = all(
            observation[key]
            for key in (
                "ready_http_ok",
                "miner_auth_required",
                "authenticated_status_ok",
                "run_id_hash_match",
            )
        )
    except Exception as exc:
        observation["error_class"] = _public_error_class(exc)
        status_code = getattr(exc, "code", None)
        if isinstance(status_code, int):
            observation["http_status"] = status_code
    return observation


def _wait_authenticated_route(
    url: str,
    *,
    coordinator_token: str,
    run_id: str,
    timeout: float,
    stable_successes: int = 2,
    poll_interval: float = 2.0,
    process: subprocess.Popen[Any] | None = None,
) -> dict[str, Any]:
    required = max(1, int(stable_successes))
    deadline = time.monotonic() + float(timeout)
    started = time.monotonic()
    consecutive = 0
    success_count = 0
    observation_count = 0
    errors: dict[str, int] = {}
    first_success_seconds: float | None = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise CoordinatorRouteError(
                "cloudflare_quick_tunnel_process_exited",
                {
                    "observation_count": observation_count,
                    "successful_observation_count": success_count,
                    "stable_successes_required": required,
                    "error_classes": errors,
                    "url_public": False,
                    "credentials_public": False,
                    "public_artifact_safe": True,
                },
            )
        observation = _probe_authenticated_route(
            url,
            coordinator_token=coordinator_token,
            run_id=run_id,
            timeout=min(15.0, max(1.0, float(timeout))),
        )
        observation_count += 1
        if observation.get("ok") is True:
            success_count += 1
            consecutive += 1
            if first_success_seconds is None:
                first_success_seconds = round(time.monotonic() - started, 3)
            if consecutive >= required:
                return {
                    "verified": True,
                    "observation_count": observation_count,
                    "successful_observation_count": success_count,
                    "stable_successes_observed": consecutive,
                    "stable_successes_required": required,
                    "first_success_seconds": first_success_seconds,
                    "authenticated_status_verified": True,
                    "miner_auth_required_verified": True,
                    "run_id_hash_verified": True,
                    "error_classes": errors,
                    "url_public": False,
                    "credentials_public": False,
                    "public_artifact_safe": True,
                }
        else:
            consecutive = 0
            error_class = str(observation.get("error_class") or "route_contract_incomplete")
            errors[error_class] = int(errors.get(error_class, 0)) + 1
        time.sleep(max(0.2, float(poll_interval)))
    raise CoordinatorRouteError(
        "cloudflare_quick_tunnel_authenticated_readiness_timeout",
        {
            "verified": False,
            "observation_count": observation_count,
            "successful_observation_count": success_count,
            "stable_successes_observed": consecutive,
            "stable_successes_required": required,
            "first_success_seconds": first_success_seconds,
            "authenticated_status_verified": False,
            "error_classes": errors,
            "url_public": False,
            "credentials_public": False,
            "public_artifact_safe": True,
        },
    )


def ensure_cloudflared(private_dir: Path) -> Path:
    installed = shutil.which("cloudflared")
    if installed:
        return Path(installed)
    binary = private_dir / "cloudflared"
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    with urlopen(url, timeout=120.0) as response:
        payload = response.read()
    if len(payload) < 1_000_000:
        raise RuntimeError("cloudflared_download_invalid")
    binary.write_bytes(payload)
    binary.chmod(0o700)
    return binary


def start_tunnel(
    binary: Path,
    local_url: str,
    private_dir: Path,
    *,
    log_name: str = "cloudflared.log",
) -> tuple[subprocess.Popen[Any], str, Path]:
    log_path = private_dir / log_name
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            str(binary),
            "tunnel",
            "--url",
            local_url,
            "--no-autoupdate",
            "--protocol",
            "http2",
            "--edge-ip-version",
            "4",
            "--loglevel",
            "info",
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process._crowdtensor_log_handle = log_handle  # type: ignore[attr-defined]
    deadline = time.monotonic() + 90.0
    tunnel_url = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        matches = TUNNEL_URL_PATTERN.findall(text)
        if matches:
            tunnel_url = matches[-1]
            break
        time.sleep(1.0)
    if not tunnel_url:
        process.terminate()
        process.wait(timeout=10.0)
        log_handle.close()
        raise RuntimeError("cloudflare_quick_tunnel_url_not_discovered")
    return process, tunnel_url, log_path


def start_verified_tunnel(
    binary: Path,
    local_url: str,
    private_dir: Path,
    *,
    coordinator_token: str,
    run_id: str,
    attempts: int = 2,
    readiness_timeout: float = 240.0,
    stable_successes: int = 2,
) -> tuple[subprocess.Popen[Any], str, int, dict[str, Any]]:
    attempt_diagnostics: list[dict[str, Any]] = []
    for attempt in range(1, max(1, int(attempts)) + 1):
        process: subprocess.Popen[Any] | None = None
        try:
            process, tunnel_url, _log_path = start_tunnel(
                binary,
                local_url,
                private_dir,
                log_name=f"cloudflared-{attempt}.log",
            )
            readiness = _wait_authenticated_route(
                tunnel_url,
                coordinator_token=coordinator_token,
                run_id=run_id,
                timeout=readiness_timeout,
                stable_successes=stable_successes,
                process=process,
            )
            readiness["tunnel_attempt"] = attempt
            readiness["tunnel_process_registered"] = True
            return process, tunnel_url, attempt, readiness
        except CoordinatorRouteError as exc:
            attempt_diagnostics.append(
                {
                    "tunnel_attempt": attempt,
                    "blocker": exc.code,
                    "readiness": exc.diagnostics,
                }
            )
            stop_process(process)
            time.sleep(min(5.0, float(attempt)))
        except Exception as exc:
            attempt_diagnostics.append(
                {
                    "tunnel_attempt": attempt,
                    "blocker": f"cloudflare_quick_tunnel_start_failed:{type(exc).__name__}",
                }
            )
            stop_process(process)
            time.sleep(min(5.0, float(attempt)))
    raise CoordinatorRouteError(
        "cloudflare_quick_tunnel_authenticated_route_unavailable",
        {
            "verified": False,
            "tunnel_attempt_count": len(attempt_diagnostics),
            "attempts": attempt_diagnostics,
            "url_public": False,
            "credentials_public": False,
            "public_artifact_safe": True,
        },
    )


def stop_process(process: subprocess.Popen[Any] | None) -> bool:
    if process is None:
        return True
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15.0)
    handle = getattr(process, "_crowdtensor_log_handle", None)
    if handle is not None:
        handle.close()
    return process.poll() is not None


def _prepare_cuda_fixture(private_dir: Path, *, run_id: str) -> dict[str, Any]:
    fixture = create_local_training_fixture(
        private_dir / "fixture",
        job_id=run_id,
        row_count=24,
        sequence_length=16,
        local_steps=8,
        learning_rate=0.08,
        batch_size=2,
    )
    job_path = private_dir / "fixture" / "training_job_private.json"
    job = _load_json(job_path)
    job["backend"] = "pytorch_transformers_peft_cuda"
    job["gpu_live_verified"] = False
    job["job_hash"] = sha256_json(public_training_spec(job))
    _write_json(job_path, job)
    _write_json(private_dir / "fixture" / "training_job_public.json", public_training_spec(job))
    return {**job, "job_manifest_path": str(job_path)}


def _cpu_logits(base_model_path: Path, adapter_path: Path | None, dataset_path: Path) -> Any:
    import torch
    from peft import PeftModel
    from transformers import LlamaForCausalLM

    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokens = torch.tensor([rows[index]["input_ids"] for index in range(4)], dtype=torch.long)
    base = LlamaForCausalLM.from_pretrained(base_model_path, local_files_only=True)
    model = PeftModel.from_pretrained(base, adapter_path, local_files_only=True) if adapter_path else base
    model.eval()
    with torch.no_grad():
        result = model(input_ids=tokens, labels=tokens, use_cache=False)
        loss = float(result.loss.item())
        logits = result.logits[0, -1].detach().cpu().contiguous()
    return loss, logits


def _evaluation_and_export(
    *,
    fixture: dict[str, Any],
    store: Any,
    rendezvous: Any,
    output: Path,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load

    state = dict(store.training_state)
    global_path = Path(state["global_adapter_path"])
    export = export_standard_peft_adapter(
        adapter_tensor_path=global_path,
        adapter_config_path=fixture["lora"]["adapter_config_path"],
        output_dir=output / "exported_adapter",
    )
    base_model_path = Path(fixture["model"]["base_model_path"])
    dataset_path = Path(fixture["dataset"]["private_dataset_path"])
    before_loss, before_logits = _cpu_logits(base_model_path, None, dataset_path)
    after_loss, after_logits = _cpu_logits(base_model_path, Path(export["adapter_dir"]), dataset_path)
    gpu_values = rendezvous.private_evaluations()
    comparisons: list[dict[str, Any]] = []
    for role in sorted(gpu_values):
        record = gpu_values[role]
        raw = base64.b64decode(str(record["logits_b64"]).encode("ascii"), validate=True)
        gpu_logits = load(raw)["logits"].cpu()
        max_difference = float((after_logits.float() - gpu_logits.float()).abs().max().item())
        comparisons.append(
            {
                "role": role,
                "shape_match": list(gpu_logits.shape) == list(after_logits.shape),
                "logits_close": bool(torch.allclose(after_logits, gpu_logits, atol=5e-4, rtol=5e-4)),
                "max_abs_difference": max_difference,
                "atol": 0.0005,
                "rtol": 0.0005,
                "cuda_logits_hash": record["logits_hash"],
            }
        )
    cpu_changed = not bool(torch.allclose(before_logits, after_logits, atol=1e-7, rtol=1e-6))
    return {
        "schema": "crowdtensor_cuda_two_node_evaluation_export_v1",
        "before_loss": before_loss,
        "after_loss": after_loss,
        "validation_loss_reduced": after_loss < before_loss,
        "cpu_adapter_changes_logits": cpu_changed,
        "cpu_before_logits_hash": sha256_json({"values": before_logits.tolist()}),
        "cpu_after_logits_hash": sha256_json({"values": after_logits.tolist()}),
        "cpu_cuda_comparisons": comparisons,
        "cpu_cuda_logits_close": len(comparisons) == 2 and all(item["logits_close"] for item in comparisons),
        "standard_peft_cpu_load": True,
        "standard_peft_cuda_load": len(gpu_values) == 2 and all(
            item.get("standard_peft_cuda_load") is True for item in gpu_values.values()
        ),
        "adapter_model_hash": export["adapter_model_hash"],
        "adapter_config_hash": export["adapter_config_hash"],
        "logits_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def _error_feedback(store: Any, private_dir: Path) -> dict[str, Any]:
    import torch

    tasks = [
        task for task in store._tasks.values()
        if task.get("workload_type") == "hf_lora_train" and task.get("status") == "completed"
    ]
    first = dict(tasks[0]["training_result"])
    delta = load_tensors(first["adapter_delta"]["delta_path"])
    manifest = compress_sign_with_error_feedback(
        delta,
        transport_path=private_dir / "transport" / "sign.safetensors",
        residual_path=private_dir / "transport" / "residual.safetensors",
    )
    decoded = decode_sign_transport(manifest)
    residual = load_tensors(manifest["residual_path"])
    reconstructed = all(
        torch.allclose(decoded[name] + residual[name], delta[name], atol=1e-7, rtol=1e-6)
        for name in delta
    )
    return {
        "schema": manifest["schema"],
        "tensor_count": manifest["tensor_count"],
        "compression_ratio": manifest["compression_ratio"],
        "error_feedback": manifest["error_feedback"],
        "dense_reconstruction_with_residual_verified": reconstructed,
        "transport_hash": manifest["transport_hash"],
        "residual_hash": manifest["residual_hash"],
        "tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def _build_embedded_single_gate(
    *,
    stage0_worker: dict[str, Any],
    stage0_bundle: dict[str, Any],
    stage0_kernel_ref_hash: str,
    attempt: int,
) -> dict[str, Any]:
    worker_report = dict(stage0_worker.get("embedded_single_kernel_gate") or {})
    worker_bundle = dict(worker_report.get("checkpoint_bundle") or {})
    checkpoint_bundle = {
        "preserved": stage0_bundle.get("preserved") is True,
        "worker_hash_match": stage0_bundle.get("worker_hash_match") is True,
        "file_hash": stage0_bundle.get("file_hash") or "",
        "byte_count": int(stage0_bundle.get("byte_count") or 0),
        "file_count": int(stage0_bundle.get("file_count") or 0),
        "contains_baseline_and_resumed_checkpoints": stage0_bundle.get(
            "contains_baseline_and_resumed_checkpoints"
        )
        is True,
        "checkpoint_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    binding_verified = bool(
        worker_report
        and stage0_worker.get("role") == "stage0"
        and stage0_worker.get("embedded_single_kernel_gate_verified") is True
        and worker_report.get("source_role") == "stage0"
        and worker_report.get("execution_order") == "before_cross_node_stage0"
        and worker_report.get("coallocated_with_two_node_attempt") is True
        and str(stage0_kernel_ref_hash).startswith("sha256:")
        and worker_bundle.get("file_hash") == checkpoint_bundle.get("file_hash")
        and checkpoint_bundle.get("preserved") is True
        and checkpoint_bundle.get("worker_hash_match") is True
        and checkpoint_bundle.get("contains_baseline_and_resumed_checkpoints") is True
    )
    return {
        "schema": "crowdtensor_cuda_single_kernel_live_probe_v1",
        "ok": False,
        "single_kernel_t4x2_verified": bool(
            binding_verified and worker_report.get("single_kernel_t4x2_verified") is True
        ),
        "worker_report": worker_report,
        "checkpoint_bundle": checkpoint_bundle,
        "cleanup": {
            "kernel_deleted": False,
            "private_package_removed": False,
            "checkpoint_preserved": checkpoint_bundle.get("preserved") is True,
            "private_cleanup_state_removed": False,
        },
        "evidence_source": "two_node_stage0_kernel_embedded_single_gate",
        "coallocated_with_two_node_attempt": True,
        "source_role": "stage0",
        "execution_order": "before_cross_node_stage0",
        "source_two_node_attempt": int(attempt),
        "source_kernel_ref_hash": stage0_kernel_ref_hash,
        "source_worker_report_hash": sha256_json(worker_report) if worker_report else "",
        "source_binding_verified": binding_verified,
        "allocation_started": True,
        "activation_values_public": False,
        "gradient_values_public": False,
        "checkpoint_values_public": False,
        "raw_training_text_public": False,
        "credentials_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-token-file", default="")
    parser.add_argument("--raw-token-username", default="")
    parser.add_argument("--attempt-ledger", default="dist/training-cuda-two-node-work/allocation_attempts.json")
    parser.add_argument("--attempt-limit", type=int, default=2)
    parser.add_argument("--allocation-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--push-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--status-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--output-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--delete-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=15.0)
    parser.add_argument("--route-readiness-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--route-stability-successes", type=int, default=2)
    parser.add_argument("--tunnel-attempts", type=int, default=2)
    parser.add_argument("--route-preflight-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.attempt_limit < 1 or args.attempt_limit > 3:
        parser.error("--attempt-limit must be in [1, 3]")
    if args.allocation_timeout_seconds <= 0 or args.allocation_timeout_seconds > 1800:
        parser.error("--allocation-timeout-seconds must be in (0, 1800]")
    if args.route_readiness_timeout_seconds <= 0 or args.route_readiness_timeout_seconds > 600:
        parser.error("--route-readiness-timeout-seconds must be in (0, 600]")
    if args.route_stability_successes < 1 or args.route_stability_successes > 5:
        parser.error("--route-stability-successes must be in [1, 5]")
    if args.tunnel_attempts < 1 or args.tunnel_attempts > 3:
        parser.error("--tunnel-attempts must be in [1, 3]")
    if not args.route_preflight_only and not str(args.raw_token_file).strip():
        parser.error("--raw-token-file is required unless --route-preflight-only is used")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    private_dir = output / ".private-runtime"
    private_dir.mkdir(parents=True, exist_ok=True)
    report_path = output / "training_cuda_two_node_live_probe.json"
    ledger_path = Path(args.attempt_ledger).resolve()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": False,
        "evidence_ready": False,
        "two_node_cuda_verified": False,
        "started_at": utc_now(),
        "attempt": 0,
        "allocation_started": False,
        "push_attempted": False,
        "attempt_limit": int(args.attempt_limit),
        "allocation_timeout_seconds": float(args.allocation_timeout_seconds),
        "worker_reports": [],
        "embedded_single_kernel_gate": {},
        "status_observations": [],
        "blockers": [],
        "cleanup": {
            "kernels_deleted": False,
            "private_packages_removed": False,
            "coordinator_stopped": False,
            "tunnel_stopped": False,
            "private_runtime_removed": False,
            "checkpoint_bundles_preserved": True,
            "private_cleanup_state_removed": True,
        },
        "same_authorized_account": True,
        "requested_kernel_count": 2,
        "used_gpu_per_kernel": 1,
        "all_four_t4_used_claimed": False,
        "tpu_used": False,
        "multi_account_used": False,
        "activation_values_public": False,
        "gradient_values_public": False,
        "evaluation_logits_public": False,
        "checkpoint_values_public": False,
        "credentials_public": False,
        "token_paths_public": False,
        "coordinator_url_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    attempt = 0
    outcome = "not_started"
    refs: list[str] = []
    cleanup_refs: list[str] = []
    private_cleanup_dir = output / ".private-cleanup"
    server = None
    server_thread = None
    tunnel_process = None
    app = None
    rendezvous_status: dict[str, Any] = {}
    rendezvous_cleanup: dict[str, Any] = {}
    try:
        run_id = f"cuda-two-node-{int(time.time())}-{secrets.token_hex(3)}"
        fixture = _prepare_cuda_fixture(private_dir, run_id=run_id)
        coordinator_token = secrets.token_urlsafe(32)
        port = _free_port()
        app = create_app(
            state_dir=private_dir / "coordinator-state",
            lease_seconds=1200.0,
            backlog=0,
            task_lanes=[],
            reaper_interval=1.0,
            hf_lora_job_manifest=fixture["job_manifest_path"],
            miner_token=coordinator_token,
        )
        import uvicorn

        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None
        server_thread = threading.Thread(target=server.run, name="cuda-training-coordinator", daemon=True)
        server_thread.start()
        local_url = f"http://127.0.0.1:{port}"
        _wait_ready(local_url, timeout=30.0)
        tunnel_binary = ensure_cloudflared(private_dir)
        tunnel_process, tunnel_url, tunnel_start_attempts, route_readiness = start_verified_tunnel(
            tunnel_binary,
            local_url,
            private_dir,
            coordinator_token=coordinator_token,
            run_id=run_id,
            attempts=int(args.tunnel_attempts),
            readiness_timeout=float(args.route_readiness_timeout_seconds),
            stable_successes=int(args.route_stability_successes),
        )
        report["route_preflight"] = route_readiness
        report["route_preflight_verified"] = True
        report["coordinator"] = {
            "local_ready": True,
            "authenticated_remote_tunnel_ready": True,
            "miner_auth_required": True,
            "url_hash": "sha256:" + hashlib.sha256(tunnel_url.encode("utf-8")).hexdigest(),
            "url_public": False,
            "tunnel_start_attempts": tunnel_start_attempts,
        }
        _write_json(report_path, report)
        if args.route_preflight_only:
            report["route_preflight_only"] = True
            report["blockers"].append("two_node_route_preflight_only_no_kaggle_allocation")
            outcome = "route_preflight_verified_without_kaggle_allocation"
            raise RoutePreflightComplete(outcome)
        with kaggle_env(args.raw_token_file, username_hint=args.raw_token_username) as env:
            owner = authenticated_owner(env)
            if not owner:
                raise RuntimeError("authorized_kaggle_account_authentication_failed")
            packages: list[dict[str, Any]] = []
            suffix = str(int(time.time()))[-8:]
            for role in ("stage0", "stage1"):
                package = build_package(
                    private_dir / f"package-{role}",
                    owner=owner,
                    slug=safe_slug(f"ct-cuda-two-node-{role}-{suffix}"),
                    role=role,
                    fixture_dir=private_dir / "fixture",
                    coordinator_url=tunnel_url,
                    coordinator_token=coordinator_token,
                    run_id=run_id,
                )
                packages.append(package)

            attempt = reserve_attempt(ledger_path, limit=args.attempt_limit)
            report["attempt"] = attempt
            report["allocation_started"] = True
            report["push_attempted"] = True
            cleanup_refs = [str(package["kernel_ref"]) for package in packages]
            _write_json(
                private_cleanup_dir / "active_resources.json",
                {
                    "schema": "crowdtensor_cuda_training_private_cleanup_resources_v1",
                    "provider": "kaggle",
                    "kernel_refs": cleanup_refs,
                    "push_attempted": True,
                    "credentials_embedded": False,
                },
            )
            _write_json(report_path, report)

            def push(package: dict[str, Any]) -> dict[str, Any]:
                step = run_command(
                    [
                        "kaggle",
                        "kernels",
                        "push",
                        "-p",
                        str(package["package_dir"]),
                        "-t",
                        str(int(args.allocation_timeout_seconds)),
                        "--accelerator",
                        "NvidiaTeslaT4",
                    ],
                    env=env,
                    timeout=float(args.push_timeout_seconds),
                )
                step["role"] = package["role"]
                step["accepted"] = push_accepted(step)
                step["ref"] = extract_kernel_ref(str(step.get("output_tail") or ""), package["kernel_ref"])
                return step

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                pushes = list(pool.map(push, packages))
            report["pushes"] = [
                {key: value for key, value in push.items() if key != "ref"}
                for push in pushes
            ]
            refs = [str(push["ref"]) for push in pushes if push.get("accepted")]
            if len(refs) != 2:
                raise RuntimeError("two_node_concurrent_kernel_push_incomplete")
            report["kernel_ref_hashes"] = [
                "sha256:" + hashlib.sha256(ref.encode("utf-8")).hexdigest() for ref in refs
            ]
            report["kernel_ref_hashes_by_role"] = {
                str(push["role"]): "sha256:"
                + hashlib.sha256(str(push["ref"]).encode("utf-8")).hexdigest()
                for push in pushes
                if push.get("accepted")
            }
            deadline = time.monotonic() + float(args.allocation_timeout_seconds)
            terminal_classes: dict[str, str] = {}
            max_running = 0
            while time.monotonic() < deadline:
                classes: dict[str, str] = {}
                for ref in refs:
                    status = run_command(
                        ["kaggle", "kernels", "status", ref],
                        env=env,
                        timeout=float(args.status_timeout_seconds),
                    )
                    classes[ref] = status_class(str(status.get("output_tail") or ""))
                running = sum(value == "running" for value in classes.values())
                max_running = max(max_running, running)
                report["status_observations"].append(
                    {
                        "observed_at": utc_now(),
                        "running_count": running,
                        "complete_count": sum(value == "complete" for value in classes.values()),
                        "failed_count": sum(value == "failed" for value in classes.values()),
                    }
                )
                _write_json(report_path, report)
                if all(value in TERMINAL for value in classes.values()):
                    terminal_classes = classes
                    break
                time.sleep(max(5.0, float(args.poll_interval_seconds)))
            if len(terminal_classes) != 2:
                raise RuntimeError("two_node_allocation_wait_timeout")
            report["max_observed_running_kernel_count"] = max_running
            worker_reports: list[dict[str, Any]] = []
            checkpoint_bundles: list[dict[str, Any]] = []
            role_by_ref = {str(push["ref"]): str(push["role"]) for push in pushes}
            for ref in refs:
                role = role_by_ref[ref]
                private_output = private_dir / f"output-{role}"
                output_step = run_command(
                    [
                        "kaggle",
                        "kernels",
                        "output",
                        ref,
                        "-p",
                        str(private_output),
                        "--force",
                        "--file-pattern",
                        OUTPUT_PATTERN,
                    ],
                    env=env,
                    timeout=float(args.output_timeout_seconds),
                )
                worker = _load_json(private_output / WORKER_REPORT)
                if worker:
                    _write_json(output / "workers" / f"{role}.json", worker)
                    worker_reports.append(worker)
                worker_bundle = dict(worker.get("checkpoint_bundle") or {})
                bundle_name = f"training_cuda_two_node_{role}_checkpoint_bundle.zip"
                bundle_source = private_output / bundle_name
                bundle_summary = {
                    "role": role,
                    "preserved": False,
                    "worker_hash_match": False,
                    "file_hash": "",
                    "byte_count": 0,
                    "file_count": int(worker_bundle.get("file_count") or 0),
                    "contains_pipeline_and_miner_checkpoints": worker_bundle.get(
                        "contains_pipeline_and_miner_checkpoints"
                    )
                    is True,
                    "contains_baseline_and_resumed_checkpoints": worker_bundle.get(
                        "contains_baseline_and_resumed_checkpoints"
                    )
                    is True,
                    "checkpoint_values_public": False,
                    "private_paths_public": False,
                    "public_artifact_safe": True,
                }
                if bundle_source.is_file():
                    checkpoint_dir = output / "checkpoints"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_destination = checkpoint_dir / bundle_name
                    checkpoint_destination.unlink(missing_ok=True)
                    shutil.move(str(bundle_source), checkpoint_destination)
                    actual_hash = sha256_file(checkpoint_destination)
                    bundle_summary.update(
                        {
                            "preserved": True,
                            "worker_hash_match": actual_hash == worker_bundle.get("file_hash"),
                            "file_hash": actual_hash,
                            "byte_count": checkpoint_destination.stat().st_size,
                        }
                    )
                checkpoint_bundles.append(bundle_summary)
                report.setdefault("outputs", []).append(
                    {"role": role, **output_step}
                )
            report["worker_reports"] = worker_reports
            report["checkpoint_bundles"] = checkpoint_bundles
            report["cleanup"]["checkpoint_bundles_preserved"] = bool(
                len(checkpoint_bundles) == 2
                and all(
                    item.get("preserved") is True and item.get("worker_hash_match") is True
                    for item in checkpoint_bundles
                )
            )
            worker_by_role = {str(item.get("role")): item for item in worker_reports}
            checkpoint_by_role = {
                str(item.get("role")): item for item in checkpoint_bundles
            }
            if "stage0" in worker_by_role and "stage0" in checkpoint_by_role:
                report["embedded_single_kernel_gate"] = _build_embedded_single_gate(
                    stage0_worker=worker_by_role["stage0"],
                    stage0_bundle=checkpoint_by_role["stage0"],
                    stage0_kernel_ref_hash=str(
                        (report.get("kernel_ref_hashes_by_role") or {}).get("stage0") or ""
                    ),
                    attempt=attempt,
                )

        rendezvous = app.state.cuda_training_rendezvous
        rendezvous_status = rendezvous.public_status() if rendezvous is not None else {}
        store = app.state.store
        training_state = dict(store.training_state)
        report["rendezvous"] = rendezvous_status
        report["training_state"] = {
            "schema": training_state.get("schema"),
            "round_status": training_state.get("round_status"),
            "adapter_version": int(training_state.get("adapter_version", 0)),
            "outer_step": int(training_state.get("outer_step", 0)),
            "accepted_result_count": len(training_state.get("accepted_result_ids") or []),
            "accepted_shard_indexes": training_state.get("accepted_shard_indexes") or [],
            "global_adapter_file_hash": training_state.get("global_adapter_file_hash", ""),
            "dense_diloco_aggregation": bool(training_state.get("aggregation")),
            "private_paths_public": False,
        }
        report["error_feedback"] = _error_feedback(store, private_dir)
        report["evaluation_export"] = _evaluation_and_export(
            fixture=fixture,
            store=store,
            rendezvous=rendezvous,
            output=output,
        )
        worker_by_role = {str(item.get("role")): item for item in report["worker_reports"]}
        pipeline_roles = set(rendezvous_status.get("registered_roles") or [])
        payloads = list(rendezvous_status.get("payloads") or [])
        checkpoint_by_role = {
            str(item.get("role")): item for item in report.get("checkpoint_bundles") or []
        }
        report["two_node_cuda_verified"] = bool(
            set(worker_by_role) == {"stage0", "stage1"}
            and all(item.get("ok") is True for item in worker_by_role.values())
            and pipeline_roles == {"stage0", "stage1"}
            and len([item for item in payloads if item.get("kind") == "activation"]) == 4
            and len([item for item in payloads if item.get("kind") == "gradient"]) == 4
            and len(rendezvous_status.get("completions") or []) == 2
            and int(report.get("max_observed_running_kernel_count") or 0) >= 2
            and training_state.get("round_status") == "aggregated"
            and int(training_state.get("adapter_version", 0)) == 1
            and int(training_state.get("outer_step", 0)) == 1
            and set(training_state.get("accepted_shard_indexes") or []) == {0, 1}
            and report["error_feedback"].get("dense_reconstruction_with_residual_verified") is True
            and report["evaluation_export"].get("validation_loss_reduced") is True
            and report["evaluation_export"].get("cpu_adapter_changes_logits") is True
            and report["evaluation_export"].get("cpu_cuda_logits_close") is True
            and report["evaluation_export"].get("standard_peft_cpu_load") is True
            and report["evaluation_export"].get("standard_peft_cuda_load") is True
            and set(checkpoint_by_role) == {"stage0", "stage1"}
            and all(
                checkpoint_by_role[role].get("preserved") is True
                and checkpoint_by_role[role].get("worker_hash_match") is True
                and checkpoint_by_role[role].get("contains_pipeline_and_miner_checkpoints") is True
                for role in ("stage0", "stage1")
            )
        )
        report["ok"] = report["two_node_cuda_verified"]
        if not report["ok"]:
            report["blockers"].append("two_node_cuda_acceptance_incomplete")
        outcome = "verified" if report["ok"] else "acceptance_incomplete"
    except RoutePreflightComplete:
        pass
    except CoordinatorRouteError as exc:
        report["blockers"].append(exc.code)
        report["route_preflight"] = exc.diagnostics
        report["route_preflight_verified"] = False
        report["error_class"] = type(exc).__name__
        outcome = exc.code
    except Exception as exc:
        code = str(exc)[:180] or type(exc).__name__
        report["blockers"].append(code)
        report["error_class"] = type(exc).__name__
        outcome = code
    finally:
        if cleanup_refs:
            deleted = 0
            try:
                with kaggle_env(args.raw_token_file, username_hint=args.raw_token_username) as cleanup_env:
                    for ref in cleanup_refs:
                        step = run_command(
                            ["kaggle", "kernels", "delete", ref, "-y"],
                            env=cleanup_env,
                            timeout=float(args.delete_timeout_seconds),
                        )
                        deleted += int(delete_succeeded_or_absent(step))
            except Exception:
                pass
            report["cleanup"]["kernels_deleted"] = deleted == len(cleanup_refs)
        else:
            report["cleanup"]["kernels_deleted"] = True
        if app is not None and getattr(app.state, "cuda_training_rendezvous", None) is not None:
            rendezvous_cleanup = app.state.cuda_training_rendezvous.cleanup()
        report["rendezvous_cleanup"] = rendezvous_cleanup
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=20.0)
        report["cleanup"]["coordinator_stopped"] = bool(server_thread is None or not server_thread.is_alive())
        report["cleanup"]["tunnel_stopped"] = stop_process(tunnel_process)
        package_dirs = [private_dir / "package-stage0", private_dir / "package-stage1"]
        for path in package_dirs:
            shutil.rmtree(path, ignore_errors=True)
        report["cleanup"]["private_packages_removed"] = all(not path.exists() for path in package_dirs)
        shutil.rmtree(private_dir, ignore_errors=True)
        report["cleanup"]["private_runtime_removed"] = not private_dir.exists()
        if report["cleanup"]["kernels_deleted"]:
            shutil.rmtree(private_cleanup_dir, ignore_errors=True)
        report["cleanup"]["private_cleanup_state_removed"] = not private_cleanup_dir.exists()
        embedded_single = report.get("embedded_single_kernel_gate") or {}
        if embedded_single:
            embedded_cleanup = embedded_single.setdefault("cleanup", {})
            embedded_cleanup.update(
                {
                    "kernel_deleted": report["cleanup"].get("kernels_deleted") is True,
                    "private_package_removed": report["cleanup"].get("private_packages_removed")
                    is True,
                    "checkpoint_preserved": (
                        (embedded_single.get("checkpoint_bundle") or {}).get("preserved") is True
                    ),
                    "private_cleanup_state_removed": report["cleanup"].get(
                        "private_cleanup_state_removed"
                    )
                    is True,
                }
            )
            embedded_single["ok"] = bool(
                embedded_single.get("single_kernel_t4x2_verified")
                and embedded_single.get("source_binding_verified") is True
                and all(embedded_cleanup.get(key) is True for key in embedded_cleanup)
            )
        report["finished_at"] = utc_now()
        report["blockers"] = sorted(set(report.get("blockers") or []))
        cleanup_ok = all(
            report["cleanup"].get(key) is True
            for key in (
                "kernels_deleted",
                "private_packages_removed",
                "coordinator_stopped",
                "tunnel_stopped",
                "private_runtime_removed",
                "private_cleanup_state_removed",
            )
        )
        safety_errors = public_safety_errors(report)
        report["public_artifact_safe"] = not safety_errors
        if safety_errors:
            report["safety_errors"] = safety_errors
        report["ok"] = bool(report.get("ok") and cleanup_ok and report["public_artifact_safe"])
        report["evidence_ready"] = bool(cleanup_ok and report["public_artifact_safe"])
        _write_json(report_path, report)
        if attempt:
            finish_attempt(ledger_path, attempt=attempt, outcome=outcome)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print(
                f"training_cuda_two_node_probe ok={report['ok']} attempt={report['attempt']} "
                f"blockers={','.join(report['blockers']) or 'none'}"
            )
    return 0 if report["ok"] else (1 if report["evidence_ready"] else 2)


if __name__ == "__main__":
    raise SystemExit(main())
