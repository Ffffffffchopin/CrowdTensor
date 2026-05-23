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
import uuid
from collections import Counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from crowdtensor.diloco import run_inner_loop
from crowdtensor.external_llm import WORKLOAD_TYPE as WORKLOAD_EXTERNAL_LLM_INFER
from crowdtensor.external_llm import run_external_llm_inference, run_mock_external_llm_inference
from crowdtensor.lora_mock import run_lora_inner_loop
from crowdtensor.micro_transformer import WORKLOAD_TYPE as WORKLOAD_MICRO_TRANSFORMER_LM
from crowdtensor.micro_transformer import run_micro_transformer_inner_loop
from crowdtensor.model_bundle import INFERENCE_WORKLOAD_TYPE as WORKLOAD_MODEL_BUNDLE_INFER
from crowdtensor.model_bundle import WORKLOAD_TYPE as WORKLOAD_MODEL_BUNDLE_LM
from crowdtensor.model_bundle import run_model_bundle_inference, run_model_bundle_inner_loop
from crowdtensor.outer_optimizer import (
    DELTA_FORMAT_DENSE_FLOAT,
    DELTA_FORMAT_SIGN_COMPRESSED,
    DELTA_FORMAT_SIGN_COMPRESSED_EF,
    compress_sign_delta,
    compress_sign_delta_with_error_feedback,
)


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
                        "phase": "training",
                        "workload_type": claim.get("workload_type", "diloco_train"),
                        "pid": os_safe_pid(),
                        "accepted_tasks": int(counters.get("accepted_tasks", 0)),
                        "current_task_elapsed_seconds": round(time.monotonic() - task_started_at, 6),
                        "compute_seconds": float(args.compute_seconds),
                        "max_tasks": int(args.max_tasks),
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
            stop.set()


def os_safe_pid() -> int:
    return os.getpid()


def hardware_profile() -> dict:
    return {
        "os": platform.system() or "unknown",
        "platform": platform.platform(aliased=True, terse=True) or "unknown",
        "machine": platform.machine() or "unknown",
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or 1,
        "python_version": platform.python_version(),
    }


def miner_capabilities(
    *,
    enable_mock_llm_runtime: bool = False,
    llm_runtime_cmd: str = "",
    llm_runtime_url: str = "",
    llm_runtime_model_id: str = "external-llm-runtime",
) -> dict:
    supported_workloads = [
        "diloco_train",
        "cpu_lora_mock",
        WORKLOAD_MICRO_TRANSFORMER_LM,
        WORKLOAD_MODEL_BUNDLE_LM,
        WORKLOAD_MODEL_BUNDLE_INFER,
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
    return {
        "runtime": "python-cli",
        "backend": "cpu",
        "hardware_profile": hardware_profile(),
        "supports_training_spec": True,
        "protocol_version": "runtime_contract_v1",
        "supported_workloads": supported_workloads,
        "supported_delta_formats": SUPPORTED_MINER_DELTA_FORMATS,
        "external_llm_runtime": external_llm_runtime,
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
        WORKLOAD_MODEL_BUNDLE_LM,
        WORKLOAD_MODEL_BUNDLE_INFER,
        WORKLOAD_EXTERNAL_LLM_INFER,
    }:
        raise RuntimeError(f"python-cli miner does not support workload {workload_type}")

    stop = threading.Event()
    heartbeat_interval = args.heartbeat_interval or float(claim.get("heartbeat_interval", 5.0))
    task_started_at = time.monotonic()
    thread = threading.Thread(
        target=run_heartbeat,
        kwargs={
            "coordinator": args.coordinator,
            "claim": claim,
            "interval": heartbeat_interval,
            "stop": stop,
            "args": args,
            "counters": counters,
            "task_started_at": task_started_at,
        },
        daemon=True,
    )
    thread.start()

    try:
        if args.compute_seconds > 0 and workload_type in {WORKLOAD_MODEL_BUNDLE_INFER, WORKLOAD_EXTERNAL_LLM_INFER}:
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
    parser.add_argument("--idle-sleep", type=float, default=2.0)
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
                time.sleep(args.idle_sleep)
    finally:
        print(json.dumps(summary_payload(args, counters, started_at), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
