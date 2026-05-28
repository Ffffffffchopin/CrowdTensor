#!/usr/bin/env python3
"""Headless Miner CLI for CrowdTensorD Phase 1."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import threading
import time
import traceback
import uuid
from collections import Counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from crowdtensor.diloco import run_inner_loop
from crowdtensor.external_llm import WORKLOAD_TYPE as WORKLOAD_EXTERNAL_LLM_INFER
from crowdtensor.external_llm import run_external_llm_inference, run_mock_external_llm_inference
from crowdtensor.lora_mock import run_lora_inner_loop
from crowdtensor.micro_transformer import MICRO_LLM_SHARDED_WORKLOAD_TYPE as WORKLOAD_MICRO_LLM_SHARDED_INFER
from crowdtensor.micro_transformer import WORKLOAD_TYPE as WORKLOAD_MICRO_TRANSFORMER_LM
from crowdtensor.micro_transformer import run_micro_llm_sharded_inference, run_micro_transformer_inner_loop
from crowdtensor.model_bundle import INFERENCE_WORKLOAD_TYPE as WORKLOAD_MODEL_BUNDLE_INFER
from crowdtensor.model_bundle import SHARDED_INFERENCE_WORKLOAD_TYPE as WORKLOAD_SHARDED_MODEL_BUNDLE_INFER
from crowdtensor.model_bundle import WORKLOAD_TYPE as WORKLOAD_MODEL_BUNDLE_LM
from crowdtensor.model_bundle import (
    run_model_bundle_inference,
    run_model_bundle_inner_loop,
    run_sharded_model_bundle_inference,
)
from crowdtensor.outer_optimizer import (
    DELTA_FORMAT_DENSE_FLOAT,
    DELTA_FORMAT_SIGN_COMPRESSED,
    DELTA_FORMAT_SIGN_COMPRESSED_EF,
    compress_sign_delta,
    compress_sign_delta_with_error_feedback,
)
from crowdtensor.protocol import (
    MICRO_LLM_SHARDED_BOTH_CAPABILITY,
    MICRO_LLM_SHARDED_STAGE0_CAPABILITY,
    MICRO_LLM_SHARDED_STAGE1_CAPABILITY,
    REAL_LLM_SHARDED_BOTH_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_BOTH_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY,
    REAL_LLM_SHARDED_STAGE0_CAPABILITY,
    REAL_LLM_SHARDED_STAGE1_CAPABILITY,
)
from crowdtensor.real_llm import BACKEND_CPU as REAL_LLM_BACKEND_CPU
from crowdtensor.real_llm import BACKEND_CUDA as REAL_LLM_BACKEND_CUDA
from crowdtensor.real_llm import DEFAULT_MODEL_ID as DEFAULT_REAL_LLM_MODEL_ID
from crowdtensor.real_llm import PARTITION_MODE_FULL as REAL_LLM_PARTITION_MODE_FULL
from crowdtensor.real_llm import WORKLOAD_TYPE as WORKLOAD_REAL_LLM_SHARDED_INFER
from crowdtensor.real_llm import cuda_runtime_summary, normalize_backend as normalize_real_llm_backend
from crowdtensor.real_llm import normalize_partition_mode as normalize_real_llm_partition_mode
from crowdtensor.real_llm import run_real_llm_sharded_inference


RETRYABLE_HTTP_STATUSES = {500, 502, 504}
EXPECTED_PROTOCOL_VERSION = "runtime_contract_v1"
DELTA_FORMAT_AUTO = "auto"
SUPPORTED_MINER_DELTA_FORMATS = [
    DELTA_FORMAT_DENSE_FLOAT,
    DELTA_FORMAT_SIGN_COMPRESSED,
    DELTA_FORMAT_SIGN_COMPRESSED_EF,
]


class CoordinatorHTTPError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"coordinator returned HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class CoordinatorTransportError(RuntimeError):
    """Raised for transport failures before the Coordinator returns a response."""


def request_json(
    method: str,
    base_url: str,
    path: str,
    payload: dict | None = None,
    *,
    timeout: float = 10.0,
    miner_token: str = "",
) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json"} if payload is not None else {}
    if miner_token:
        headers["x-crowdtensor-miner-token"] = miner_token
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise CoordinatorHTTPError(exc.code, detail) from exc
    except URLError as exc:
        raise CoordinatorTransportError(f"coordinator is unreachable: {exc}") from exc
    except OSError as exc:
        raise CoordinatorTransportError(f"coordinator is unreachable: {exc}") from exc


def should_retry_error(exc: Exception) -> bool:
    if isinstance(exc, CoordinatorTransportError):
        return True
    return isinstance(exc, CoordinatorHTTPError) and exc.status in RETRYABLE_HTTP_STATUSES


def retry_sleep_seconds(args: argparse.Namespace, attempt: int) -> float:
    base = max(0.0, float(args.retry_base_sleep))
    cap = max(base, float(args.retry_max_sleep))
    return min(cap, base * (2 ** max(0, attempt - 1)))


def request_json_with_retries(
    method: str,
    base_url: str,
    path: str,
    payload: dict | None = None,
    *,
    timeout: float,
    miner_token: str = "",
    args: argparse.Namespace,
    counters: Counter | None = None,
    retry_result_upload: bool = False,
) -> dict:
    max_attempts = max(1, int(args.max_request_attempts))
    if method == "POST" and path.rsplit("/", 1)[-1] == "result" and not retry_result_upload:
        max_attempts = 1
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return request_json(
                method,
                base_url,
                path,
                payload,
                timeout=timeout,
                miner_token=miner_token,
            )
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts or not should_retry_error(exc):
                raise
            if counters is not None:
                counters["request_retries"] += 1
            sleep_for = retry_sleep_seconds(args, attempt)
            print(
                f"retrying {method} {path} after {exc} "
                f"(attempt {attempt + 1}/{max_attempts})",
                file=sys.stderr,
                flush=True,
            )
            if sleep_for > 0:
                time.sleep(sleep_for)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("request retry loop exited unexpectedly")


def post_json(
    base_url: str,
    path: str,
    payload: dict,
    *,
    timeout: float = 10.0,
    miner_token: str = "",
    args: argparse.Namespace | None = None,
    counters: Counter | None = None,
    retry_result_upload: bool = False,
) -> dict:
    if args is None:
        return request_json("POST", base_url, path, payload, timeout=timeout, miner_token=miner_token)
    return request_json_with_retries(
        "POST",
        base_url,
        path,
        payload,
        timeout=timeout,
        miner_token=miner_token,
        args=args,
        counters=counters,
        retry_result_upload=retry_result_upload,
    )


def preflight(args: argparse.Namespace, counters: Counter) -> None:
    if args.skip_preflight:
        return
    try:
        payload = request_json_with_retries(
            "GET",
            args.coordinator,
            "/ready",
            timeout=args.preflight_timeout,
            args=args,
            counters=counters,
        )
    except Exception:
        counters["preflight_failures"] += 1
        raise
    if payload.get("ok") is not True:
        counters["preflight_failures"] += 1
        raise RuntimeError(f"coordinator is not ready: {payload}")
    protocol_version = payload.get("protocol_version")
    if protocol_version != EXPECTED_PROTOCOL_VERSION:
        counters["preflight_failures"] += 1
        raise RuntimeError(
            f"coordinator protocol mismatch: expected {EXPECTED_PROTOCOL_VERSION}, got {protocol_version}"
        )
    print(
        f"preflight ok service={payload.get('service', 'unknown')} "
        f"version={payload.get('version', 'unknown')} protocol={protocol_version}",
        flush=True,
    )


def run_heartbeat(
    *,
    coordinator: str,
    claim: dict,
    interval: float,
    stop: threading.Event,
    runtime_status: dict,
    args: argparse.Namespace,
    counters: Counter,
    task_started_at: float,
) -> None:
    while not stop.wait(interval):
        try:
            heartbeat = post_json(
                coordinator,
                f"/tasks/{claim['task_id']}/heartbeat",
                {
                    "lease_token": claim["lease_token"],
                    "attempt": claim["attempt"],
                    "runtime_status": {
                        "runtime": "python-cli",
                        "phase": str(runtime_status.get("phase") or "training"),
                        "workload_type": claim.get("workload_type", "diloco_train"),
                        "pid": os_safe_pid(),
                        "accepted_tasks": int(counters.get("accepted_tasks", 0)),
                        "current_task_elapsed_seconds": round(time.monotonic() - task_started_at, 6),
                        "compute_seconds": float(args.compute_seconds),
                        "max_tasks": int(args.max_tasks),
                        **{
                            key: value
                            for key, value in runtime_status.items()
                            if key not in {"runtime", "workload_type", "pid", "accepted_tasks", "current_task_elapsed_seconds", "compute_seconds", "max_tasks"}
                        },
                    },
                },
                timeout=args.heartbeat_timeout,
                miner_token=args.miner_token,
                args=args,
                counters=counters,
            )
            print(
                f"heartbeat task={heartbeat['task_id']} "
                f"attempt={heartbeat['attempt']} expires={heartbeat['lease_expires_at']:.3f}",
                flush=True,
            )
        except Exception as exc:
            counters["heartbeat_failures"] += 1
            print(f"heartbeat failed: {exc}", file=sys.stderr, flush=True)


def os_safe_pid() -> int:
    return os.getpid()


def hardware_profile() -> dict:
    runtime_environment = "generic"
    if os.environ.get("CROWDTENSOR_REMOTE_ENVIRONMENT"):
        runtime_environment = str(os.environ.get("CROWDTENSOR_REMOTE_ENVIRONMENT") or "generic").strip() or "generic"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        runtime_environment = "kaggle"

    accelerator_hints: list[str] = []
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_KERNEL_INTEGRATIONS"):
        accelerator_hints.append("kaggle_runtime")
    if os.environ.get("TPU_NAME") or os.environ.get("COLAB_TPU_ADDR"):
        accelerator_hints.append("tpu_env_visible")
    cuda_visible = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip().lower()
    nvidia_visible = str(os.environ.get("NVIDIA_VISIBLE_DEVICES") or "").strip().lower()
    if cuda_visible and cuda_visible not in {"-1", "none", "void"}:
        accelerator_hints.append("cuda_env_visible")
    if nvidia_visible and nvidia_visible not in {"-1", "none", "void"}:
        accelerator_hints.append("nvidia_env_visible")

    cuda_summary = cuda_runtime_summary()
    return {
        "os": platform.system() or "unknown",
        "platform": platform.platform(aliased=True, terse=True) or "unknown",
        "machine": platform.machine() or "unknown",
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or 1,
        "python_version": platform.python_version(),
        "runtime_environment": runtime_environment,
        "accelerator_hints": sorted(set(accelerator_hints)),
        "cuda_available": bool(cuda_summary.get("cuda_available")),
        "gpu_count": int(cuda_summary.get("gpu_count") or 0),
        "gpu_names": list(cuda_summary.get("gpu_names") or []),
        "vram_total_mb": list(cuda_summary.get("vram_total_mb") or []),
        "torch_cuda_version": str(cuda_summary.get("torch_cuda_version") or ""),
        "gpu_tpu_workload_enabled": bool(cuda_summary.get("cuda_available")),
    }


def miner_capabilities(
    *,
    enable_mock_llm_runtime: bool = False,
    llm_runtime_cmd: str = "",
    llm_runtime_url: str = "",
    llm_runtime_model_id: str = "external-llm-runtime",
    micro_llm_stage_role: str = "both",
    enable_hf_tiny_gpt_runtime: bool = False,
    hf_model_id: str = DEFAULT_REAL_LLM_MODEL_ID,
    real_llm_backend: str = REAL_LLM_BACKEND_CPU,
    real_llm_stage_role: str = "both",
    real_llm_partition_mode: str = REAL_LLM_PARTITION_MODE_FULL,
) -> dict:
    supported_workloads = [
        "diloco_train",
        "cpu_lora_mock",
        WORKLOAD_MICRO_TRANSFORMER_LM,
        WORKLOAD_MICRO_LLM_SHARDED_INFER,
        WORKLOAD_MODEL_BUNDLE_LM,
        WORKLOAD_MODEL_BUNDLE_INFER,
        WORKLOAD_SHARDED_MODEL_BUNDLE_INFER,
    ]
    external_llm_runtime = {}
    if enable_mock_llm_runtime or str(llm_runtime_cmd or "").strip() or str(llm_runtime_url or "").strip():
        supported_workloads.append(WORKLOAD_EXTERNAL_LLM_INFER)
        adapter_kind = "mock"
        model_id = "mock-external-llm"
        if not enable_mock_llm_runtime and str(llm_runtime_cmd or "").strip():
            adapter_kind = "command"
            model_id = llm_runtime_model_id
        if not enable_mock_llm_runtime and not str(llm_runtime_cmd or "").strip() and str(llm_runtime_url or "").strip():
            adapter_kind = "http_openai_chat"
            model_id = llm_runtime_model_id
        external_llm_runtime = {
            "adapter_kind": adapter_kind,
            "model_id": model_id,
        }
    stage_role = str(micro_llm_stage_role or "both").strip().lower()
    if stage_role not in {"stage0", "stage1", "both"}:
        stage_role = "both"
    micro_llm_stage_capabilities = {
        "stage0": [MICRO_LLM_SHARDED_STAGE0_CAPABILITY],
        "stage1": [MICRO_LLM_SHARDED_STAGE1_CAPABILITY],
        "both": [
            MICRO_LLM_SHARDED_STAGE0_CAPABILITY,
            MICRO_LLM_SHARDED_STAGE1_CAPABILITY,
            MICRO_LLM_SHARDED_BOTH_CAPABILITY,
        ],
    }[stage_role]
    real_stage_role = str(real_llm_stage_role or "both").strip().lower()
    if real_stage_role not in {"stage0", "stage1", "both"}:
        real_stage_role = "both"
    resolved_real_backend = normalize_real_llm_backend(real_llm_backend)
    if resolved_real_backend == "auto":
        resolved_real_backend = REAL_LLM_BACKEND_CUDA if cuda_runtime_summary().get("cuda_available") else REAL_LLM_BACKEND_CPU
    if resolved_real_backend == REAL_LLM_BACKEND_CUDA:
        real_llm_stage_capabilities = {
            "stage0": [REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY],
            "stage1": [REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY],
            "both": [
                REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY,
                REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY,
                REAL_LLM_SHARDED_CUDA_BOTH_CAPABILITY,
            ],
        }[real_stage_role]
    else:
        real_llm_stage_capabilities = {
            "stage0": [REAL_LLM_SHARDED_STAGE0_CAPABILITY],
            "stage1": [REAL_LLM_SHARDED_STAGE1_CAPABILITY],
            "both": [
                REAL_LLM_SHARDED_STAGE0_CAPABILITY,
                REAL_LLM_SHARDED_STAGE1_CAPABILITY,
                REAL_LLM_SHARDED_BOTH_CAPABILITY,
            ],
        }[real_stage_role]
    real_llm_runtime: dict[str, object] = {}
    if enable_hf_tiny_gpt_runtime:
        supported_workloads.append(WORKLOAD_REAL_LLM_SHARDED_INFER)
        real_llm_runtime = {
            "adapter_kind": resolved_real_backend,
            "model_id": str(hf_model_id or DEFAULT_REAL_LLM_MODEL_ID),
            "stage_role": real_stage_role,
            "partition_mode": normalize_real_llm_partition_mode(real_llm_partition_mode),
        }
        if resolved_real_backend == REAL_LLM_BACKEND_CUDA:
            real_llm_runtime["cuda_runtime"] = cuda_runtime_summary()
    return {
        "runtime": "python-cli",
        "backend": "cuda" if enable_hf_tiny_gpt_runtime and resolved_real_backend == REAL_LLM_BACKEND_CUDA else "cpu",
        "hardware_profile": hardware_profile(),
        "supports_training_spec": True,
        "protocol_version": "runtime_contract_v1",
        "supported_workloads": supported_workloads,
        "micro_llm_sharded_stage_role": stage_role,
        "micro_llm_sharded_stage_capabilities": micro_llm_stage_capabilities,
        "real_llm_sharded_stage_role": real_stage_role,
        "real_llm_sharded_stage_capabilities": real_llm_stage_capabilities if enable_hf_tiny_gpt_runtime else [],
        "supported_delta_formats": SUPPORTED_MINER_DELTA_FORMATS,
        "external_llm_runtime": external_llm_runtime,
        "real_llm_runtime": real_llm_runtime,
        "pid": os_safe_pid(),
    }


def delta_format_for_claim(claim: dict, requested_format: str) -> str:
    if requested_format != DELTA_FORMAT_AUTO:
        return requested_format
    optimizer_spec = claim.get("optimizer_spec") or {}
    return str(optimizer_spec.get("delta_format") or DELTA_FORMAT_DENSE_FLOAT)


def build_result_payload(
    claim: dict,
    inner_result: dict,
    *,
    delta_format: str,
    elapsed_ms: float,
    residual: list[float] | None = None,
) -> tuple[dict, list[float] | None]:
    workload_type = claim.get("workload_type", "diloco_train")
    payload = {
        "lease_token": claim["lease_token"],
        "attempt": claim["attempt"],
        "idempotency_key": uuid.uuid4().hex,
        "metrics": {
            key: value
            for key, value in inner_result.items()
            if key not in {
                "local_delta",
                "pseudo_gradient",
                "compressed_delta",
                "adapter_delta",
                "bundle_delta",
                "inference_result",
                "inference_results",
            "sharded_inference_result",
            "micro_llm_sharded_result",
            "real_llm_sharded_result",
            "external_llm_result",
            "external_llm_results",
            }
        },
    }
    payload["metrics"]["elapsed_ms"] = elapsed_ms
    next_residual = None
    if workload_type == "cpu_lora_mock":
        payload["adapter_delta"] = inner_result["adapter_delta"]
    elif workload_type == WORKLOAD_MICRO_TRANSFORMER_LM:
        payload["local_delta"] = inner_result["local_delta"]
    elif workload_type == WORKLOAD_MODEL_BUNDLE_LM:
        payload["bundle_delta"] = inner_result["bundle_delta"]
    elif workload_type == WORKLOAD_MODEL_BUNDLE_INFER:
        payload["inference_result"] = inner_result["inference_result"]
        payload["inference_results"] = inner_result.get("inference_results", [inner_result["inference_result"]])
    elif workload_type == WORKLOAD_SHARDED_MODEL_BUNDLE_INFER:
        payload["sharded_inference_result"] = inner_result
    elif workload_type == WORKLOAD_MICRO_LLM_SHARDED_INFER:
        payload["sharded_inference_result"] = inner_result
    elif workload_type == WORKLOAD_REAL_LLM_SHARDED_INFER:
        payload["sharded_inference_result"] = inner_result
    elif workload_type == WORKLOAD_EXTERNAL_LLM_INFER:
        payload["external_llm_result"] = inner_result["external_llm_result"]
        payload["external_llm_results"] = inner_result.get(
            "external_llm_results",
            [inner_result["external_llm_result"]],
        )
    elif delta_format == DELTA_FORMAT_SIGN_COMPRESSED_EF:
        payload["compressed_delta"], next_residual = compress_sign_delta_with_error_feedback(
            inner_result["local_delta"],
            residual=residual,
        )
        payload["metrics"]["delta_format"] = DELTA_FORMAT_SIGN_COMPRESSED_EF
    elif delta_format == DELTA_FORMAT_SIGN_COMPRESSED:
        payload["compressed_delta"] = compress_sign_delta(inner_result["local_delta"])
        payload["metrics"]["delta_format"] = DELTA_FORMAT_SIGN_COMPRESSED
    else:
        payload["local_delta"] = inner_result["local_delta"]
    return payload, next_residual


def send_failure_heartbeat(
    args: argparse.Namespace,
    claim: dict,
    *,
    task_started_at: float,
    workload_type: str,
    phase: str,
    exc: Exception,
    counters: Counter,
) -> None:
    try:
        post_json(
            args.coordinator,
            f"/tasks/{claim['task_id']}/heartbeat",
            {
                "lease_token": claim["lease_token"],
                "attempt": claim["attempt"],
                "runtime_status": {
                    "runtime": "python-cli",
                    "phase": phase,
                    "workload_type": workload_type,
                    "pid": os_safe_pid(),
                    "accepted_tasks": int(counters.get("accepted_tasks", 0)),
                    "current_task_elapsed_seconds": round(time.monotonic() - task_started_at, 6),
                    "compute_seconds": float(args.compute_seconds),
                    "max_tasks": int(args.max_tasks),
                    "failure_class": exc.__class__.__name__,
                    "failure_message": str(exc)[:240],
                },
            },
            timeout=args.heartbeat_timeout,
            miner_token=args.miner_token,
            args=args,
            counters=counters,
        )
    except Exception as heartbeat_exc:
        counters["heartbeat_failures"] += 1
        print(f"failure heartbeat failed: {heartbeat_exc}", file=sys.stderr, flush=True)


def process_one(args: argparse.Namespace, counters: Counter, residual_state: dict[str, list[float]]) -> bool:
    claim = post_json(
        args.coordinator,
        "/tasks/claim",
        {
            "miner_id": args.miner_id,
            "capabilities": miner_capabilities(
                enable_mock_llm_runtime=args.enable_mock_llm_runtime,
                llm_runtime_cmd=args.llm_runtime_cmd,
                llm_runtime_url=args.llm_runtime_url,
                llm_runtime_model_id=args.llm_runtime_model_id,
                micro_llm_stage_role=args.micro_llm_stage_role,
                enable_hf_tiny_gpt_runtime=args.enable_hf_tiny_gpt_runtime,
                hf_model_id=args.hf_model_id,
                real_llm_backend=args.real_llm_backend,
                real_llm_stage_role=args.real_llm_stage_role,
                real_llm_partition_mode=getattr(args, "real_llm_partition_mode", REAL_LLM_PARTITION_MODE_FULL),
            ),
        },
        timeout=args.claim_timeout,
        miner_token=args.miner_token,
        args=args,
        counters=counters,
    )
    print(
        f"claimed task={claim['task_id']} attempt={claim['attempt']} "
        f"model_version={claim['model_version']}",
        flush=True,
    )
    workload_type = claim.get("workload_type", "diloco_train")
    if workload_type not in {
        "diloco_train",
        "cpu_lora_mock",
        WORKLOAD_MICRO_TRANSFORMER_LM,
        WORKLOAD_MICRO_LLM_SHARDED_INFER,
        WORKLOAD_MODEL_BUNDLE_LM,
        WORKLOAD_MODEL_BUNDLE_INFER,
        WORKLOAD_SHARDED_MODEL_BUNDLE_INFER,
        WORKLOAD_EXTERNAL_LLM_INFER,
        WORKLOAD_REAL_LLM_SHARDED_INFER,
    }:
        raise RuntimeError(f"python-cli miner does not support workload {workload_type}")

    stop = threading.Event()
    runtime_status: dict[str, object] = {"phase": "training"}
    heartbeat_interval = args.heartbeat_interval or float(claim.get("heartbeat_interval", 5.0))
    task_started_at = time.monotonic()
    thread = threading.Thread(
        target=run_heartbeat,
        kwargs={
            "coordinator": args.coordinator,
            "claim": claim,
            "interval": heartbeat_interval,
            "stop": stop,
            "runtime_status": runtime_status,
            "args": args,
            "counters": counters,
            "task_started_at": task_started_at,
        },
        daemon=True,
    )
    thread.start()

    try:
        try:
            if args.compute_seconds > 0 and workload_type in {
                WORKLOAD_MODEL_BUNDLE_INFER,
                WORKLOAD_SHARDED_MODEL_BUNDLE_INFER,
                WORKLOAD_MICRO_LLM_SHARDED_INFER,
                WORKLOAD_REAL_LLM_SHARDED_INFER,
                WORKLOAD_EXTERNAL_LLM_INFER,
            }:
                time.sleep(args.compute_seconds)
            if workload_type == "cpu_lora_mock":
                inner_result = run_lora_inner_loop(
                    claim["workload_spec"],
                    inner_steps=int(claim["inner_steps"]),
                    compute_seconds=args.compute_seconds,
                )
            elif workload_type == WORKLOAD_MICRO_TRANSFORMER_LM:
                inner_result = run_micro_transformer_inner_loop(
                    claim["workload_spec"],
                    inner_steps=int(claim["inner_steps"]),
                    compute_seconds=args.compute_seconds,
                )
            elif workload_type == WORKLOAD_MODEL_BUNDLE_LM:
                inner_result = run_model_bundle_inner_loop(
                    claim["workload_spec"],
                    inner_steps=int(claim["inner_steps"]),
                    compute_seconds=args.compute_seconds,
                )
            elif workload_type == WORKLOAD_MODEL_BUNDLE_INFER:
                inner_result = run_model_bundle_inference(claim["workload_spec"])
            elif workload_type == WORKLOAD_SHARDED_MODEL_BUNDLE_INFER:
                inner_result = run_sharded_model_bundle_inference(claim["workload_spec"])
            elif workload_type == WORKLOAD_MICRO_LLM_SHARDED_INFER:
                inner_result = run_micro_llm_sharded_inference(claim["workload_spec"])
            elif workload_type == WORKLOAD_REAL_LLM_SHARDED_INFER:
                if not args.enable_hf_tiny_gpt_runtime:
                    raise RuntimeError("real_llm_sharded_infer requires --enable-hf-tiny-gpt-runtime")
                runtime_status["phase"] = "real_llm_runtime"
                runtime_status["real_llm_stage_id"] = (claim.get("workload_spec") or {}).get("stage_id")
                runtime_status["real_llm_backend"] = (claim.get("workload_spec") or {}).get("backend")
                runtime_status["real_llm_partition_mode"] = (claim.get("workload_spec") or {}).get("partition_mode")
                inner_result = run_real_llm_sharded_inference(
                    claim["workload_spec"],
                    cache_dir=args.hf_cache_dir,
                )
            elif workload_type == WORKLOAD_EXTERNAL_LLM_INFER:
                if args.enable_mock_llm_runtime:
                    inner_result = run_mock_external_llm_inference(claim["workload_spec"])
                elif args.llm_runtime_cmd:
                    inner_result = run_external_llm_inference(
                        claim["workload_spec"],
                        adapter_kind="command",
                        model_id=args.llm_runtime_model_id,
                        runtime_command=args.llm_runtime_cmd,
                        timeout=args.llm_runtime_timeout,
                    )
                elif args.llm_runtime_url:
                    inner_result = run_external_llm_inference(
                        claim["workload_spec"],
                        adapter_kind="http_openai_chat",
                        model_id=args.llm_runtime_model_id,
                        runtime_url=args.llm_runtime_url,
                        api_key=args.llm_runtime_api_key,
                        timeout=args.llm_runtime_timeout,
                    )
                else:
                    raise RuntimeError(
                        "external_llm_infer requires --enable-mock-llm-runtime, "
                        "--llm-runtime-cmd, or --llm-runtime-url"
                    )
            else:
                inner_result = run_inner_loop(
                    claim["weights"],
                    task_id=claim["task_id"],
                    miner_id=args.miner_id,
                    model_version=int(claim["model_version"]),
                    inner_steps=int(claim["inner_steps"]),
                    compute_seconds=args.compute_seconds,
                    training_spec=claim.get("training_spec"),
                )
        except Exception as exc:
            runtime_status["phase"] = "workload_failed"
            runtime_status["failure_class"] = exc.__class__.__name__
            runtime_status["failure_message"] = str(exc)[:240]
            send_failure_heartbeat(
                args,
                claim,
                task_started_at=task_started_at,
                workload_type=workload_type,
                phase="workload_failed",
                exc=exc,
                counters=counters,
            )
            raise
        if stop.is_set():
            print("lease heartbeat stopped before result upload", file=sys.stderr, flush=True)
            return False
        payload, next_residual = build_result_payload(
            claim,
            inner_result,
            delta_format=delta_format_for_claim(claim, args.delta_format),
            elapsed_ms=round((time.monotonic() - task_started_at) * 1000.0, 6),
            residual=residual_state.get(workload_type),
        )
        stop.set()
        thread.join(timeout=max(0.1, args.heartbeat_timeout))
        result = post_json(
            args.coordinator,
            f"/tasks/{claim['task_id']}/result",
            payload,
            timeout=args.result_timeout,
            miner_token=args.miner_token,
            args=args,
            counters=counters,
            retry_result_upload=bool(payload.get("idempotency_key")),
        )
        counters["accepted_tasks"] += 1
        counters[f"workload:{workload_type}"] += 1
        if next_residual is not None and workload_type == "diloco_train":
            residual_state[workload_type] = next_residual
        if workload_type == "cpu_lora_mock":
            print(
                f"accepted adapter task={claim['task_id']} adapter_step={result['adapter_step']} "
                f"adapter_loss={result['adapter_loss']:.6f}",
                flush=True,
            )
            return True
        if workload_type == WORKLOAD_MICRO_TRANSFORMER_LM:
            print(
                f"accepted micro-transformer task={claim['task_id']} "
                f"lm_version={result['model_version']} "
                f"lm_step={result['micro_transformer_optimizer_step']} "
                f"lm_loss={inner_result['lm_loss_start']:.6f}->{inner_result['lm_loss_end']:.6f}",
                flush=True,
            )
            return True
        if workload_type == WORKLOAD_MODEL_BUNDLE_LM:
            print(
                f"accepted model-bundle task={claim['task_id']} "
                f"bundle_version={result['bundle_version']} "
                f"bundle_step={result['bundle_optimizer_step']} "
                f"bundle_loss={inner_result['bundle_loss_start']:.6f}->{inner_result['bundle_loss_end']:.6f}",
                flush=True,
            )
            return True
        if workload_type == WORKLOAD_MODEL_BUNDLE_INFER:
            print(
                f"accepted model-bundle-infer task={claim['task_id']} "
                f"bundle_version={result['bundle_version']} "
                f"requests={result.get('request_count', 1)} "
                f"accuracy={float(result.get('accuracy', 1.0 if result.get('correct') else 0.0)):.3f} "
                f"prediction={result['predicted_token']} "
                f"target={result['target_token']} "
                f"correct={result['correct']}",
                flush=True,
            )
            return True
        if workload_type == WORKLOAD_SHARDED_MODEL_BUNDLE_INFER:
            stage_id = int(result.get("stage_id", inner_result.get("stage_id", 0)))
            print(
                f"accepted sharded-model-bundle-infer task={claim['task_id']} "
                f"session={result.get('session_id')} "
                f"stage={stage_id}/2 "
                f"requests={result.get('request_count', inner_result.get('request_count', 0))} "
                f"activation_count={result.get('activation_count', inner_result.get('activation_count', 0))} "
                f"baseline_match={result.get('baseline_match')}",
                flush=True,
            )
            return True
        if workload_type == WORKLOAD_MICRO_LLM_SHARDED_INFER:
            stage_id = int(result.get("stage_id", inner_result.get("stage_id", 0)))
            print(
                f"accepted micro-llm-sharded-infer task={claim['task_id']} "
                f"session={result.get('session_id')} "
                f"stage={stage_id}/2 "
                f"requests={result.get('request_count', inner_result.get('request_count', 0))} "
                f"decode_steps={result.get('decode_steps', inner_result.get('decode_steps', 0))} "
                f"activation_count={result.get('activation_count', inner_result.get('activation_count', 0))} "
                f"baseline_match={result.get('baseline_match')} "
                f"decoded_tokens_match={result.get('decoded_tokens_match')}",
                flush=True,
            )
            return True
        if workload_type == WORKLOAD_REAL_LLM_SHARDED_INFER:
            stage_id = int(result.get("stage_id", inner_result.get("stage_id", 0)))
            print(
                f"accepted real-llm-sharded-infer task={claim['task_id']} "
                f"session={result.get('session_id')} "
                f"stage={stage_id}/2 "
                f"model={result.get('model_id')} "
                f"requests={result.get('request_count', inner_result.get('request_count', 0))} "
                f"activation_count={result.get('activation_count', inner_result.get('activation_count', 0))} "
                f"baseline_match={result.get('baseline_match')} "
                f"decoded_tokens_match={result.get('decoded_tokens_match')}",
                flush=True,
            )
            return True
        if workload_type == WORKLOAD_EXTERNAL_LLM_INFER:
            print(
                f"accepted external-llm task={claim['task_id']} "
                f"requests={result.get('request_count', 0)} "
                f"adapter={result.get('adapter_kind')} "
                f"model={result.get('model_id')} "
                f"output_chars={result.get('output_chars', 0)}",
                flush=True,
            )
            return True
        print(
            f"accepted task={claim['task_id']} global_step={result['global_step']} "
            f"model_version={result['model_version']} optimizer_step={result['optimizer_step']} "
            f"inner_loss={inner_result['inner_loss_start']:.6f}->{inner_result['inner_loss_end']:.6f} "
            f"outer_loss={result['loss']:.6f}",
            flush=True,
        )
        return True
    finally:
        stop.set()
        thread.join(timeout=2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CrowdTensorD Phase 1 Miner.")
    parser.add_argument("--coordinator", default="http://127.0.0.1:8787")
    parser.add_argument("--miner-id", default=f"{socket.gethostname()}-{os_safe_pid()}")
    parser.add_argument("--once", action="store_true", help="process one task and exit")
    parser.add_argument("--max-tasks", type=int, default=0, help="exit after accepting this many tasks; 0 runs forever")
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=0.0,
        help="exit after this many seconds once the current task completes; 0 disables the limit",
    )
    parser.add_argument("--compute-seconds", type=float, default=0.0, help="hold the lease for chaos testing")
    parser.add_argument("--heartbeat-interval", type=float, default=0.0)
    parser.add_argument("--claim-timeout", type=float, default=10.0)
    parser.add_argument("--result-timeout", type=float, default=10.0)
    parser.add_argument("--heartbeat-timeout", type=float, default=5.0)
    parser.add_argument(
        "--delta-format",
        choices=[
            DELTA_FORMAT_AUTO,
            DELTA_FORMAT_DENSE_FLOAT,
            DELTA_FORMAT_SIGN_COMPRESSED,
            DELTA_FORMAT_SIGN_COMPRESSED_EF,
        ],
        default=DELTA_FORMAT_AUTO,
        help="delta transport format for diloco_train results; auto follows claim optimizer_spec.delta_format",
    )
    parser.add_argument("--skip-preflight", action="store_true", help="skip the startup /ready protocol check")
    parser.add_argument("--preflight-timeout", type=float, default=5.0)
    parser.add_argument("--max-request-attempts", type=int, default=3)
    parser.add_argument("--retry-base-sleep", type=float, default=0.2)
    parser.add_argument("--retry-max-sleep", type=float, default=2.0)
    parser.add_argument(
        "--miner-token",
        default=os.environ.get("CROWDTENSOR_MINER_TOKEN", ""),
        help="shared Miner token for Coordinator task endpoints; falls back to CROWDTENSOR_MINER_TOKEN",
    )
    parser.add_argument(
        "--enable-mock-llm-runtime",
        action="store_true",
        help="advertise and execute external_llm_infer with the deterministic local mock runtime",
    )
    parser.add_argument(
        "--llm-runtime-cmd",
        default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_CMD", ""),
        help="optional external_llm_infer command; it receives prompt and max_tokens arguments",
    )
    parser.add_argument(
        "--llm-runtime-url",
        default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_URL", ""),
        help="optional OpenAI-compatible chat completions endpoint for external_llm_infer",
    )
    parser.add_argument(
        "--llm-runtime-api-key",
        default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_API_KEY", ""),
        help="optional bearer token for --llm-runtime-url; never advertised in Miner capabilities",
    )
    parser.add_argument(
        "--llm-runtime-model-id",
        default=os.environ.get("CROWDTENSOR_LLM_RUNTIME_MODEL_ID", "external-llm-runtime"),
        help="model id reported for --llm-runtime-cmd results",
    )
    parser.add_argument(
        "--llm-runtime-timeout",
        type=float,
        default=float(os.environ.get("CROWDTENSOR_LLM_RUNTIME_TIMEOUT", "30.0")),
        help="seconds before an external LLM command invocation is aborted",
    )
    parser.add_argument(
        "--micro-llm-stage-role",
        choices=["stage0", "stage1", "both"],
        default=os.environ.get("CROWDTENSOR_MICRO_LLM_STAGE_ROLE", "both"),
        help="which micro_llm_sharded_infer stage capability this Miner advertises",
    )
    parser.add_argument(
        "--enable-hf-tiny-gpt-runtime",
        action="store_true",
        help="advertise and execute real_llm_sharded_infer with the optional CPU Hugging Face tiny GPT runtime",
    )
    parser.add_argument(
        "--hf-model-id",
        default=os.environ.get("CROWDTENSOR_HF_MODEL_ID", DEFAULT_REAL_LLM_MODEL_ID),
        help="Hugging Face causal LM id for real_llm_sharded_infer; defaults to sshleifer/tiny-gpt2",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=os.environ.get("CROWDTENSOR_HF_CACHE_DIR", ""),
        help="optional Hugging Face cache directory for real_llm_sharded_infer",
    )
    parser.add_argument(
        "--real-llm-backend",
        choices=["hf_transformers_cpu", "hf_transformers_cuda", "cpu", "cuda", "auto"],
        default=os.environ.get("CROWDTENSOR_REAL_LLM_BACKEND", REAL_LLM_BACKEND_CPU),
        help="backend advertised for real_llm_sharded_infer; cuda requires a local torch CUDA runtime",
    )
    parser.add_argument(
        "--real-llm-stage-role",
        choices=["stage0", "stage1", "both"],
        default=os.environ.get("CROWDTENSOR_REAL_LLM_STAGE_ROLE", "both"),
        help="which real_llm_sharded_infer stage capability this Miner advertises",
    )
    parser.add_argument(
        "--real-llm-partition-mode",
        choices=["full", "stage-local", "stage_local"],
        default=os.environ.get("CROWDTENSOR_REAL_LLM_PARTITION_MODE", REAL_LLM_PARTITION_MODE_FULL),
        help="real_llm_sharded_infer partition mode; stage-local moves only stage-owned modules to the target device",
    )
    parser.add_argument("--idle-sleep", type=float, default=2.0)
    parser.add_argument(
        "--debug-tracebacks",
        action="store_true",
        help="print full exception tracebacks for operator debugging; secrets are not intentionally included",
    )
    args = parser.parse_args()
    if args.once and args.max_tasks <= 0:
        args.max_tasks = 1
    if args.max_tasks < 0:
        raise SystemExit("--max-tasks must be non-negative")
    if args.max_request_attempts < 1:
        raise SystemExit("--max-request-attempts must be at least 1")
    if args.retry_base_sleep < 0:
        raise SystemExit("--retry-base-sleep must be non-negative")
    if args.retry_max_sleep < 0:
        raise SystemExit("--retry-max-sleep must be non-negative")
    if args.llm_runtime_timeout <= 0:
        raise SystemExit("--llm-runtime-timeout must be positive")
    args.real_llm_partition_mode = normalize_real_llm_partition_mode(args.real_llm_partition_mode)
    return args


def summary_payload(args: argparse.Namespace, counters: Counter, started_at: float) -> dict:
    return {
        "miner_id": args.miner_id,
        "accepted_tasks": int(counters.get("accepted_tasks", 0)),
        "failed_claims": int(counters.get("failed_claims", 0)),
        "rejected_results": int(counters.get("rejected_results", 0)),
        "stale_leases": int(counters.get("stale_leases", 0)),
        "heartbeat_failures": int(counters.get("heartbeat_failures", 0)),
        "request_retries": int(counters.get("request_retries", 0)),
        "preflight_failures": int(counters.get("preflight_failures", 0)),
        "workloads": {
            key.split(":", 1)[1]: int(value)
            for key, value in sorted(counters.items())
            if key.startswith("workload:")
        },
        "elapsed_seconds": round(time.monotonic() - started_at, 6),
    }


def main() -> None:
    args = parse_args()
    counters: Counter = Counter()
    residual_state: dict[str, list[float]] = {}
    started_at = time.monotonic()
    try:
        preflight(args, counters)
        while True:
            if args.max_runtime_seconds > 0 and time.monotonic() - started_at >= args.max_runtime_seconds:
                break
            if args.max_tasks > 0 and counters["accepted_tasks"] >= args.max_tasks:
                break

            try:
                process_one(args, counters, residual_state)
            except CoordinatorHTTPError as exc:
                if exc.status == 409:
                    counters["stale_leases"] += 1
                    print(f"stale lease rejected: {exc.detail}", file=sys.stderr, flush=True)
                elif exc.status == 422:
                    counters["rejected_results"] += 1
                    print(str(exc), file=sys.stderr, flush=True)
                    time.sleep(args.idle_sleep)
                else:
                    counters["failed_claims"] += 1
                    print(str(exc), file=sys.stderr, flush=True)
                    time.sleep(args.idle_sleep)
            except KeyboardInterrupt:
                raise
            except CoordinatorTransportError as exc:
                counters["failed_claims"] += 1
                print(str(exc), file=sys.stderr, flush=True)
                time.sleep(args.idle_sleep)
            except Exception as exc:
                counters["failed_claims"] += 1
                print(str(exc), file=sys.stderr, flush=True)
                if args.debug_tracebacks:
                    traceback.print_exc(file=sys.stderr)
                time.sleep(args.idle_sleep)
    finally:
        print(json.dumps(summary_payload(args, counters, started_at), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
