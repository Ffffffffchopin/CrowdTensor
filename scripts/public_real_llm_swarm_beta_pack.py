#!/usr/bin/env python3
"""Build the top-level Public Real-LLM Swarm Inference Beta artifact.

This is the product-facing release aggregate for the current Coordinator-backed
small-LLM split inference path.  It deliberately composes the existing
serve/join/generate product loop, retained external Kaggle evidence, a release-local
Petals-class P2P candidate local-smoke over retained external/requeue source reports,
optional CUDA diagnostics, and real-P2P discovery candidate evidence instead of
claiming Coordinator-free production P2P.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import support_bundle  # noqa: E402
from crowdtensor.session_protocol import public_leak_paths  # noqa: E402
from product_swarm_mvp_check import parse_prompt_texts_arg, read_prompt_texts_file  # noqa: E402


SCHEMA = "public_real_llm_swarm_beta_v1"
SUPPORT_SCHEMA = "public_real_llm_swarm_beta_support_bundle_v1"
ARTIFACT_SUMMARY_SCHEMA = "public_real_llm_swarm_beta_artifact_summary_v1"
REVIEW_SUMMARY_SCHEMA = "public_real_llm_swarm_beta_review_summary_v1"
RUNTIME_PROVENANCE_SCHEMA = "public_real_llm_swarm_beta_runtime_provenance_v1"
EVIDENCE_SCOPE_SCHEMA = "public_real_llm_swarm_beta_evidence_scope_v1"
INFERENCE_VERDICT_SCHEMA = "crowdtensor_inference_verdict_v1"
PRODUCT_SCHEMA = "public_swarm_product_beta_v1"
GPU_SCHEMA = "public_swarm_gpu_inference_beta_v1"
P2P_SCHEMA = "petals_class_p2p_candidate_v1"
PUBLIC_SWARM_V2_SCHEMA = "public_swarm_inference_v2"
REAL_LLM_INTERNET_BETA_SCHEMA = "real_llm_internet_beta_v1"
WORKLOAD_TYPE = "real_llm_sharded_infer"


def shell_command(parts: list[str]) -> str:
    return shlex.join([str(part) for part in parts])

MODE_RELEASE = "release"
MODE_LOCAL_SMOKE = "local-smoke"
MODE_LOCAL_MODEL_VARIANT = "local-model-variant"
MODE_PACKAGE = "package"
MODE_EVIDENCE_IMPORT = "evidence-import"
MODES = [MODE_RELEASE, MODE_LOCAL_SMOKE, MODE_LOCAL_MODEL_VARIANT, MODE_PACKAGE, MODE_EVIDENCE_IMPORT]

DEFAULT_OUTPUT_DIR = "dist/public-real-llm-swarm-beta"
DEFAULT_PRODUCT_REPORT = "dist/public-swarm-product-beta/public_swarm_product_beta.json"
DEFAULT_EXTERNAL_REPORT = "dist/goal-final-infer-real-llm-internet-beta-import-16tok-gpu-summary-20260602/real_llm_internet_beta.json"
DEFAULT_P2P_REPORT = "dist/goal-final-infer-petals-candidate-16tok-batch-stream-composed-20260602/petals_class_p2p_candidate.json"
DEFAULT_GPU_REPORT = "dist/public-swarm-gpu-beta-live-20260528-runtimepin/public_swarm_gpu_inference_beta_kaggle_auto.json"
DEFAULT_USABLE_REPORT = "dist/goal-final-infer-usable-swarm-16tok-kv-cache-20260601/usable_swarm_inference.json"
DEFAULT_PUBLIC_SWARM_V2_REPORT = "dist/public-swarm-inference-v2/public_swarm_inference_v2.json"
DEFAULT_PUBLIC_SWARM_V2_PREVIEW_REPORT = "dist/public-swarm-preview-v04-final/public_swarm_preview_v04.json"
DEFAULT_PUBLIC_SWARM_V2_REAL_P2P_REPORT = "dist/goal-final-infer-real-p2p-core-fresh-16tok-import-strict-20260601/real_p2p_swarm_inference_core_rc.json"
DEFAULT_P2P_RUNTIME_SMOKE_REPORT = "dist/real-p2p-libp2p-kaggle-runtime-smoke-20260531-r6/real_p2p_swarm_inference_core_rc.json"
DEFAULT_P2P_EXTERNAL_REPORT = "dist/goal-final-infer-fresh-real-p2p-kaggle-16tok-20260601/real_p2p_swarm_inference_core_rc.json"
DEFAULT_P2P_REQUEUE_REPORT = "dist/petals-p2p-candidate-live-stage0-20260531-r6/real_p2p_swarm_inference_core_rc.json"
DEFAULT_P2P_BATCH_STREAM_REPORT = "dist/goal-final-infer-public-swarm-v2-batch-stream-16tok-20260602/public_swarm_inference_v2.json"
DEFAULT_PROMPT = "CrowdTensor public real LLM swarm beta"
DEFAULT_HF_MODEL_ID = "sshleifer/tiny-gpt2"
DEFAULT_PUBLIC_SWARM_V2_REAL_P2P_PORT = 9890
DEFAULT_PUBLIC_SWARM_V2_REAL_P2P_COORDINATOR_PORT = 9891
DEFAULT_PUBLIC_SWARM_V2_REAL_P2P_LIBP2P_PORT = 0
DEFAULT_PUBLIC_SWARM_V2_REAL_P2P_DISCOVERY_BACKEND = "http-provider-store"

SECRET_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "CROWDTENSOR_P2P_PEER_SECRET=",
    "lease_token",
    "idempotency_key",
    "Bearer ",
    "hidden_state",
    "input_ids",
    "logits",
    "activation_results",
    "activation_result",
    "real_llm_sharded_result",
    "sharded_inference_result",
    "inference_results",
    '"generated_text":',
    '"generated_token_ids":',
    '"prompt_text":',
    '"prompt_texts":',
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
)
LIVE_REQUEUE_SUMMARY_FIELDS = (
    "enabled",
    "failure_mode",
    "target_stage",
    "victim_miner_id",
    "rescue_miner_id",
    "claim_observed",
    "victim_kernel_deleted",
    "lease_expired",
    "rescue_miner_used",
    "rescued_result",
    "accepted_result_after_requeue",
    "victim_result_accepted",
)

Runner = Callable[..., subprocess.CompletedProcess[str]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in str(stdout or "").splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def redact_text(value: str, secret_values: list[str] | None = None) -> str:
    result = str(value)
    for secret in secret_values or []:
        if secret:
            result = result.replace(secret, "<redacted>")
    for fragment in SECRET_FRAGMENTS:
        result = result.replace(fragment, "<redacted>")
    return result


def redact_values(value: Any, secret_values: list[str] | None = None) -> Any:
    if isinstance(value, str):
        return redact_text(value, secret_values)
    if isinstance(value, list):
        return [redact_values(item, secret_values) for item in value]
    if isinstance(value, dict):
        return {key: redact_values(item, secret_values) for key, item in value.items()}
    return value


def diagnosis_codes(*payloads: dict[str, Any], extra: list[str] | None = None) -> list[str]:
    codes: set[str] = set(extra or [])
    pending: list[Any] = list(payloads)
    seen: set[int] = set()
    while pending:
        item = pending.pop(0)
        if not isinstance(item, dict):
            continue
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        for code in item.get("diagnosis_codes") or []:
            if isinstance(code, str):
                codes.add(code)
        summary = item.get("diagnosis_summary") if isinstance(item.get("diagnosis_summary"), dict) else {}
        for code in summary.get("codes") or []:
            if isinstance(code, str):
                codes.add(code)
        for value in item.values():
            if isinstance(value, dict):
                pending.append(value)
            elif isinstance(value, list):
                pending.extend(entry for entry in value if isinstance(entry, dict))
    return sorted(codes)


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    entry: dict[str, Any] = {"kind": kind, "path": relative, "present": path.is_file()}
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def run_json_step(
    name: str,
    command: list[str],
    *,
    runner: Runner,
    timeout_seconds: float,
    secret_values: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    try:
        completed = runner(
            command,
            cwd=str(ROOT),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "ok": False,
            "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": "timeout",
        }, {}
    payload = json_from_stdout(completed.stdout)
    step = {
        "name": name,
        "ok": bool(completed.returncode == 0 and payload.get("ok") is not False),
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "payload_schema": payload.get("schema"),
        "payload_ok": payload.get("ok"),
    }
    if not payload:
        step["ok"] = False
        step["error"] = "json_payload_missing"
    if not step["ok"]:
        if completed.stdout and not payload:
            step["stdout_tail"] = redact_text(completed.stdout[-1600:], secret_values)
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1600:], secret_values)
    return step, redact_values(payload, secret_values)


def generation_summary(payload: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_score = -1
    pending: list[Any] = [payload]
    seen: set[int] = set()
    while pending:
        item = pending.pop(0)
        if not isinstance(item, dict):
            continue
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        generation = item.get("generation") if isinstance(item.get("generation"), dict) else {}
        validation = item.get("validation") if isinstance(item.get("validation"), dict) else {}
        for candidate in (generation, validation, item):
            if not isinstance(candidate, dict):
                continue
            score = 0
            if candidate.get("generated_text_hash"):
                score += 8
            if candidate.get("generated_token_count"):
                score += 4
            if candidate.get("multi_token_generation_ready") or candidate.get("decoded_tokens_match"):
                score += 2
            if score > best_score:
                best = candidate
                best_score = score
        for value in item.values():
            if isinstance(value, dict):
                pending.append(value)
            elif isinstance(value, list):
                pending.extend(entry for entry in value if isinstance(entry, dict))
    return {
        "generated_token_count": best.get("generated_token_count"),
        "max_new_tokens": best.get("max_new_tokens"),
        "generated_text_hash": best.get("generated_text_hash"),
        "decoded_tokens_match": best.get("decoded_tokens_match"),
        "multi_token_generation_ready": best.get("multi_token_generation_ready"),
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_per_request_progress(progress: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_rows = progress.get("per_request_progress") if isinstance(progress.get("per_request_progress"), list) else []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        counts = row.get("observed_token_counts") if isinstance(row.get("observed_token_counts"), list) else []
        rows.append({
            "request_key": row.get("request_key"),
            "request_id": row.get("request_id"),
            "prompt_hash": row.get("prompt_hash"),
            "event_count": safe_int(row.get("event_count")),
            "observed_token_counts": [safe_int(value) for value in counts if safe_int(value, -1) >= 0],
            "max_observed_token_count": safe_int(row.get("max_observed_token_count")),
            "target_token_count": safe_int(row.get("target_token_count")),
            "monotonic_progress": bool(row.get("monotonic_progress")),
            "stream_progress_complete": bool(row.get("stream_progress_complete")),
        })
    return rows


def stream_evidence_ready(stream: dict[str, Any], batch: dict[str, Any] | None = None) -> bool:
    if not isinstance(stream, dict) or not stream.get("stream_generation_ready"):
        return False
    progress = stream.get("progress") if isinstance(stream.get("progress"), dict) else {}
    expected_requests = safe_int(progress.get("expected_request_count") or (batch or {}).get("expected_request_count") or (batch or {}).get("request_count"), 1)
    if expected_requests > 1 or bool((batch or {}).get("enabled")):
        return bool(
            progress.get("per_request_progress")
            and progress.get("per_request_progress_complete") is True
            and progress.get("per_request_monotonic_progress") is True
        )
    return bool(progress.get("stream_progress_complete") is True and progress.get("monotonic_progress") is True)


STREAM_READY_CODES = {
    "public_swarm_generate_stream_ready",
    "public_swarm_v2_stream_generation_ready",
    "public_real_llm_swarm_beta_stream_ready",
    "public_real_llm_swarm_beta_p2p_stream_ready",
    "public_real_llm_swarm_beta_v2_stream_ready",
}

BATCH_READY_CODES = {
    "public_swarm_generate_batch_ready",
    "public_swarm_v2_batch_generation_ready",
    "public_real_llm_swarm_beta_batch_ready",
    "public_real_llm_swarm_beta_p2p_batch_ready",
    "public_real_llm_swarm_beta_v2_batch_ready",
}

PRODUCT_BATCH_READY_CODES = {
    "public_real_llm_swarm_beta_batch_ready",
}

P2P_BATCH_READY_CODES = {
    "p2p_candidate_batch_generation_ready",
    "public_real_llm_swarm_beta_p2p_batch_ready",
}

V2_BATCH_READY_CODES = {
    "public_swarm_v2_batch_generation_ready",
    "public_real_llm_swarm_beta_v2_batch_ready",
}
SUPERSEDED_PRODUCT_PATH_BLOCKED_CODES = {
    "p2p_lite_discovery_blocked",
    "p2p_lite_route_blocked",
    "public_swarm_inference_beta_blocked",
    "public_swarm_inference_beta_rc_blocked",
    "public_swarm_product_beta_blocked",
    "public_swarm_product_rc_blocked",
}


def drop_unproven_stream_codes(codes: set[str], *, stream_ready: bool) -> set[str]:
    if stream_ready:
        return codes
    return {code for code in codes if code not in STREAM_READY_CODES}


def drop_unproven_generation_ready_codes(
    codes: set[str],
    *,
    product_batch_ready: bool = False,
    p2p_batch_ready: bool = False,
    public_swarm_v2_batch_ready: bool = False,
    stream_ready: bool,
) -> set[str]:
    filtered = drop_unproven_stream_codes(codes, stream_ready=stream_ready)
    filtered.discard("public_swarm_generate_batch_ready")
    if not product_batch_ready:
        filtered -= PRODUCT_BATCH_READY_CODES
    if not p2p_batch_ready:
        filtered -= P2P_BATCH_READY_CODES
    if not public_swarm_v2_batch_ready:
        filtered -= V2_BATCH_READY_CODES
    return filtered


def drop_superseded_product_path_blockers(codes: set[str], *, product_ready: bool) -> set[str]:
    if not product_ready:
        return codes
    return set(codes) - SUPERSEDED_PRODUCT_PATH_BLOCKED_CODES


def batch_summary(payload: dict[str, Any]) -> dict[str, Any]:
    pending: list[Any] = [payload]
    seen: set[int] = set()
    while pending:
        item = pending.pop(0)
        if not isinstance(item, dict):
            continue
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        batch = item.get("batch") if isinstance(item.get("batch"), dict) else {}
        if batch:
            safe_results: list[dict[str, Any]] = []
            for result in batch.get("results") or []:
                if not isinstance(result, dict):
                    continue
                safe_results.append({
                    "request_id": result.get("request_id"),
                    "prompt_hash": result.get("prompt_hash"),
                    "generated_token_count": int(result.get("generated_token_count") or 0),
                    "max_new_tokens": result.get("max_new_tokens"),
                    "generated_text_hash": result.get("generated_text_hash"),
                    "decoded_tokens_match": result.get("decoded_tokens_match"),
                    "multi_token_generation_ready": bool(result.get("multi_token_generation_ready")),
                    "raw_generated_text_public": False,
                    "generated_token_ids_public": False,
                })
            expected_request_count = int(batch.get("expected_request_count") or batch.get("request_count") or 0)
            result_identity_keys = [
                str(result.get("request_id") or result.get("prompt_hash") or "")
                for result in safe_results[:expected_request_count]
            ]
            batch_identity_ready = bool(
                expected_request_count > 0
                and (
                    expected_request_count <= 1
                    or (
                        len(result_identity_keys) >= expected_request_count
                        and all(result_identity_keys)
                        and len(set(result_identity_keys)) == expected_request_count
                    )
                )
            )
            return {
                "enabled": bool(batch.get("enabled")),
                "request_count": int(batch.get("request_count") or 0),
                "expected_request_count": expected_request_count,
                "observed_request_count": int(batch.get("observed_request_count") or batch.get("request_count") or 0),
                "max_request_count": batch.get("max_request_count"),
                "prompt_hashes": list(batch.get("prompt_hashes") or []),
                "prompt_char_counts": list(batch.get("prompt_char_counts") or []),
                "result_count": int(batch.get("result_count") or len(safe_results)),
                "results": safe_results,
                "batch_identity_ready": batch_identity_ready,
                "batch_generation_ready": bool(batch.get("batch_generation_ready") and batch_identity_ready),
                "raw_prompts_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
            }
        for value in item.values():
            if isinstance(value, dict):
                pending.append(value)
            elif isinstance(value, list):
                pending.extend(entry for entry in value if isinstance(entry, dict))
    return {"enabled": False, "batch_generation_ready": False}


def safe_batch_from_batch(batch: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(batch, dict) or not batch:
        return {"enabled": False, "batch_generation_ready": False}
    safe_results: list[dict[str, Any]] = []
    raw_results = batch.get("results") if isinstance(batch.get("results"), list) else []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        safe_results.append({
            "request_id": item.get("request_id"),
            "prompt_hash": item.get("prompt_hash"),
            "generated_token_count": safe_int(item.get("generated_token_count")),
            "max_new_tokens": item.get("max_new_tokens"),
            "generated_text_hash": item.get("generated_text_hash"),
            "decoded_tokens_match": item.get("decoded_tokens_match"),
            "multi_token_generation_ready": bool(item.get("multi_token_generation_ready")),
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        })
    expected_request_count = safe_int(batch.get("expected_request_count") or batch.get("request_count"))
    result_identity_keys = [
        str(item.get("request_id") or item.get("prompt_hash") or "")
        for item in safe_results[:expected_request_count]
    ]
    batch_identity_ready = bool(
        expected_request_count > 0
        and (
            expected_request_count <= 1
            or (
                len(result_identity_keys) >= expected_request_count
                and all(result_identity_keys)
                and len(set(result_identity_keys)) == expected_request_count
            )
        )
    )
    return {
        "enabled": bool(batch.get("enabled")),
        "request_count": safe_int(batch.get("request_count")),
        "expected_request_count": expected_request_count,
        "observed_request_count": safe_int(batch.get("observed_request_count") or batch.get("request_count")),
        "max_request_count": batch.get("max_request_count"),
        "prompt_hashes": list(batch.get("prompt_hashes") or []),
        "prompt_char_counts": list(batch.get("prompt_char_counts") or []),
        "result_count": safe_int(batch.get("result_count") or len(safe_results)),
        "results": safe_results,
        "batch_identity_ready": batch_identity_ready,
        "batch_generation_ready": bool(batch.get("batch_generation_ready") and batch_identity_ready),
        "raw_prompts_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }


def safe_stream_from_stream(stream: dict[str, Any] | None, batch: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(stream, dict) or not stream:
        return {"enabled": False, "stream_generation_ready": False}
    progress = stream.get("progress") if isinstance(stream.get("progress"), dict) else {}
    raw_events = stream.get("events") if isinstance(stream.get("events"), list) else []
    safe_events: list[dict[str, Any]] = []
    for event in raw_events:
        if not isinstance(event, dict):
            continue
        safe_events.append({
            "schema": event.get("schema"),
            "session_id": event.get("session_id"),
            "task_id": event.get("task_id"),
            "miner_id": event.get("miner_id"),
            "stage_id": event.get("stage_id"),
            "request_id": event.get("request_id"),
            "prompt_hash": event.get("prompt_hash"),
            "generated_token_count": safe_int(event.get("generated_token_count")),
            "max_new_tokens": event.get("max_new_tokens"),
            "generation_step": event.get("generation_step"),
            "generated_text_hash": event.get("generated_text_hash"),
            "decoded_tokens_match": event.get("decoded_tokens_match"),
            "observed_at": event.get("observed_at"),
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        })
    safe_stream = {
        "enabled": bool(stream.get("enabled")),
        "requested": bool(stream.get("requested") or stream.get("enabled")),
        "event_count": safe_int(stream.get("event_count") or len(safe_events)),
        "source": stream.get("source"),
        "endpoint_ready": bool(stream.get("endpoint_ready")),
        "progress": {
            "stream_progress_complete": bool(progress.get("stream_progress_complete")),
            "all_token_events_ready": bool(progress.get("all_token_events_ready")),
            "monotonic_progress": bool(progress.get("monotonic_progress")),
            "expected_request_count": safe_int(progress.get("expected_request_count")),
            "per_request_progress": safe_per_request_progress(progress),
            "per_request_progress_complete": bool(progress.get("per_request_progress_complete")),
            "per_request_monotonic_progress": bool(progress.get("per_request_monotonic_progress")),
            "observed_token_counts": list(progress.get("observed_token_counts") or []),
            "max_observed_token_count": safe_int(progress.get("max_observed_token_count")),
            "max_new_tokens": progress.get("max_new_tokens"),
            "source": progress.get("source") or stream.get("source") or "",
        },
        "events": safe_events,
        "stream_generation_ready": bool(stream.get("stream_generation_ready")),
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }
    safe_stream["stream_generation_ready"] = stream_evidence_ready(safe_stream, batch)
    return safe_stream


def stream_summary(payload: dict[str, Any]) -> dict[str, Any]:
    pending: list[Any] = [payload]
    seen: set[int] = set()
    while pending:
        item = pending.pop(0)
        if not isinstance(item, dict):
            continue
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        stream = item.get("stream") if isinstance(item.get("stream"), dict) else {}
        if stream:
            progress = stream.get("progress") if isinstance(stream.get("progress"), dict) else {}
            events = stream.get("events") if isinstance(stream.get("events"), list) else []
            safe_events: list[dict[str, Any]] = []
            for event in events:
                if not isinstance(event, dict):
                    continue
                safe_events.append({
                    "schema": event.get("schema"),
                    "session_id": event.get("session_id"),
                    "task_id": event.get("task_id"),
                    "miner_id": event.get("miner_id"),
                    "stage_id": event.get("stage_id"),
                    "request_id": event.get("request_id"),
                    "prompt_hash": event.get("prompt_hash"),
                    "generated_token_count": int(event.get("generated_token_count") or 0),
                    "max_new_tokens": event.get("max_new_tokens"),
                    "generation_step": event.get("generation_step"),
                    "generated_text_hash": event.get("generated_text_hash"),
                    "decoded_tokens_match": event.get("decoded_tokens_match"),
                    "observed_at": event.get("observed_at"),
                    "raw_generated_text_public": False,
                    "generated_token_ids_public": False,
                })
            return {
                "enabled": bool(stream.get("enabled")),
                "requested": bool(stream.get("requested") or stream.get("enabled")),
                "event_count": int(stream.get("event_count") or len(safe_events)),
                "source": stream.get("source"),
                "endpoint_ready": bool(stream.get("endpoint_ready")),
                "progress": {
                    "stream_progress_complete": bool(progress.get("stream_progress_complete")),
                    "all_token_events_ready": bool(progress.get("all_token_events_ready")),
                    "monotonic_progress": bool(progress.get("monotonic_progress")),
                    "expected_request_count": safe_int(progress.get("expected_request_count"), 1),
                    "per_request_progress": safe_per_request_progress(progress),
                    "per_request_progress_complete": bool(progress.get("per_request_progress_complete")),
                    "per_request_monotonic_progress": bool(progress.get("per_request_monotonic_progress")),
                    "observed_token_counts": list(progress.get("observed_token_counts") or []),
                    "max_observed_token_count": int(progress.get("max_observed_token_count") or 0),
                    "max_new_tokens": progress.get("max_new_tokens"),
                    "source": progress.get("source") or stream.get("source") or "",
                },
                "events": safe_events,
                "stream_generation_ready": bool(stream.get("stream_generation_ready")),
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
            }
        for value in item.values():
            if isinstance(value, dict):
                pending.append(value)
            elif isinstance(value, list):
                pending.extend(entry for entry in value if isinstance(entry, dict))
    return {"enabled": False, "stream_generation_ready": False}


def _safe_kv_stage_summary(stage: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in stage.get("rows") or []:
        if not isinstance(row, dict):
            continue
        rows.append({
            "generation_step": safe_int(row.get("generation_step"), -1),
            "stage_id": safe_int(row.get("stage_id"), -1),
            "miner_id": row.get("miner_id"),
            "cache_schema": row.get("cache_schema") or row.get("kv_cache_schema"),
            "cache_stage": row.get("cache_stage") or row.get("kv_cache_stage"),
            "cache_ready": bool(row.get("cache_ready") or row.get("kv_cache_ready")),
            "cache_hit": bool(row.get("cache_hit") or row.get("kv_cache_hit")),
        })
    return {
        "schema": stage.get("schema"),
        "stage": stage.get("stage"),
        "ready": bool(stage.get("ready")),
        "row_count": safe_int(stage.get("row_count")),
        "ready_count": safe_int(stage.get("ready_count")),
        "hit_count": safe_int(stage.get("hit_count")),
        "expected_hit_count": safe_int(stage.get("expected_hit_count")),
        "rows": rows,
    }


def kv_cache_summary(
    payload: dict[str, Any],
    *,
    min_generated_tokens: int = 1,
    expected_model_id: str = DEFAULT_HF_MODEL_ID,
) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    model = model_compatibility(payload, expected_model_id)
    readiness = payload.get("readiness") if isinstance(payload.get("readiness"), dict) else {}
    p2p = readiness.get("p2p_product_path") if isinstance(readiness.get("p2p_product_path"), dict) else {}
    if not p2p and isinstance(readiness.get("local_p2p_generate"), dict):
        p2p = readiness["local_p2p_generate"]
    kv_cache = p2p.get("kv_cache") if isinstance(p2p.get("kv_cache"), dict) else {}
    if not kv_cache:
        pending: list[Any] = [payload]
        seen: set[int] = set()
        while pending and not kv_cache:
            item = pending.pop(0)
            if not isinstance(item, dict):
                continue
            marker = id(item)
            if marker in seen:
                continue
            seen.add(marker)
            candidate = item.get("kv_cache") if isinstance(item.get("kv_cache"), dict) else {}
            if candidate.get("stage0") and candidate.get("stage1"):
                kv_cache = candidate
                break
            for value in item.values():
                if isinstance(value, dict):
                    pending.append(value)
                elif isinstance(value, list):
                    pending.extend(entry for entry in value if isinstance(entry, dict))
    stage0 = kv_cache.get("stage0") if isinstance(kv_cache.get("stage0"), dict) else {}
    stage1 = kv_cache.get("stage1") if isinstance(kv_cache.get("stage1"), dict) else {}
    stage0_summary = _safe_kv_stage_summary(stage0)
    stage1_summary = _safe_kv_stage_summary(stage1)
    expected_hits = max(1, int(min_generated_tokens) - 1)
    generated_token_count = max(
        safe_int(p2p.get("generated_token_count")),
        safe_int((p2p.get("generation") or {}).get("generated_token_count") if isinstance(p2p.get("generation"), dict) else 0),
    )
    token_target_ready = generated_token_count >= int(min_generated_tokens)
    stage0_ready = bool(stage0_summary["ready"] and stage0_summary["hit_count"] >= expected_hits)
    stage1_ready = bool(stage1_summary["ready"] and stage1_summary["hit_count"] >= expected_hits)
    cache_ready = bool(
        payload.get("ok") is True
        and (kv_cache.get("ready") is True or "usable_real_llm_kv_cache_ready" in codes or "p2p_real_generate_kv_cache_ready" in codes)
        and stage0_ready
        and stage1_ready
        and token_target_ready
    )
    ready = bool(cache_ready and model["compatible"])
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "ready": ready,
        "cache_ready": cache_ready,
        "source": "usable_swarm_inference_v1",
        "model": model,
        "generated_token_count": generated_token_count,
        "required_generated_token_count": int(min_generated_tokens),
        "token_target_ready": token_target_ready,
        "expected_hit_count_per_stage": expected_hits,
        "process_scope": kv_cache.get("process_scope") or "single_miner_process_per_stage",
        "stage0": stage0_summary,
        "stage1": stage1_summary,
        "raw_activations_public": False,
        "raw_generated_values_public": False,
        "raw_token_inputs_public": False,
        "diagnosis_codes": sorted(codes),
    }


def safe_live_requeue_summary(payload: dict[str, Any]) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    best: dict[str, Any] = {}
    best_score = -1
    pending: list[Any] = [payload]
    seen: set[int] = set()
    while pending:
        item = pending.pop(0)
        if not isinstance(item, dict):
            continue
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        summary = item.get("live_requeue_summary") if isinstance(item.get("live_requeue_summary"), dict) else {}
        if summary:
            score = sum(1 for field in LIVE_REQUEUE_SUMMARY_FIELDS if field in summary)
            if score > best_score:
                best = summary
                best_score = score
        for value in item.values():
            if isinstance(value, dict):
                pending.append(value)
            elif isinstance(value, list):
                pending.extend(entry for entry in value if isinstance(entry, dict))
    safe = {field: best.get(field) for field in LIVE_REQUEUE_SUMMARY_FIELDS if field in best}
    if safe.get("lease_expired") == "<redacted>" and "live_requeue_lease_timeout_observed" in codes:
        safe["lease_expired"] = True
    return safe


def live_requeue_detail_ready(summary: dict[str, Any]) -> bool:
    return bool(
        summary.get("claim_observed") is True
        and summary.get("victim_kernel_deleted") is True
        and summary.get("lease_expired") is True
        and summary.get("rescue_miner_used") is True
        and summary.get("rescued_result") is True
        and summary.get("accepted_result_after_requeue") is True
        and summary.get("victim_result_accepted") is False
    )


def external_requeue_detail_ready(summary: dict[str, Any]) -> bool:
    return bool(
        summary.get("claim_observed") is True
        and summary.get("victim_kernel_deleted") is True
        and summary.get("lease_expired") is True
        and summary.get("rescued_result") is True
        and summary.get("victim_result_accepted") is False
    )


def first_string_value(payload: dict[str, Any], key: str) -> str:
    pending: list[Any] = [payload]
    seen: set[int] = set()
    while pending:
        item = pending.pop(0)
        if not isinstance(item, dict):
            continue
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
        for nested in item.values():
            if isinstance(nested, dict):
                pending.append(nested)
            elif isinstance(nested, list):
                pending.extend(entry for entry in nested if isinstance(entry, dict))
    return ""


def model_compatibility(payload: dict[str, Any], expected_model_id: str) -> dict[str, Any]:
    observed = first_string_value(payload, "hf_model_id") or first_string_value(payload, "observed_hf_model_id")
    return {
        "expected_hf_model_id": expected_model_id,
        "observed_hf_model_id": observed,
        "model_id_present": bool(observed),
        "model_id_match": bool(observed and observed == expected_model_id),
        "compatible": bool(observed and observed == expected_model_id),
        "default_model_retained_evidence": False,
    }


def run_product_local(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "public_swarm_product_beta_pack.py"),
        "local-loopback",
        "--output-dir",
        str(output_dir),
        "--base-port",
        str(args.base_port),
        "--port",
        str(args.port),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--hf-model-id",
        args.hf_model_id,
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--cpu-request-count",
        str(args.cpu_request_count),
        "--external-llm-request-count",
        str(args.external_llm_request_count),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--cpu-timeout-seconds",
        str(args.cpu_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--json",
    ]
    if args.prompt_texts_file:
        command.extend(["--prompt-texts-file", args.prompt_texts_file])
    elif args.prompt_texts:
        command.extend(["--prompt-texts", args.prompt_texts])
    else:
        command.extend(["--prompt-text", args.prompt_text])
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.stream_generation:
        command.append("--stream-generation")
    return run_json_step(
        "public_swarm_product_beta_local_loopback",
        command,
        runner=runner,
        timeout_seconds=max(args.timeout_seconds, args.remote_timeout_seconds, args.cpu_timeout_seconds, 60.0) + 420.0,
    )


def run_gpu_smoke(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "public_swarm_gpu_inference_beta_pack.py"),
        "local-smoke",
        "--output-dir",
        str(output_dir),
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(max(1, min(args.max_new_tokens, 32))),
        "--hf-model-id",
        args.hf_model_id,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    return run_json_step(
        "public_swarm_gpu_beta_local_smoke",
        command,
        runner=runner,
        timeout_seconds=max(args.timeout_seconds, 60.0) + 120.0,
    )


def run_public_swarm_v2_local(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    v2_tokens = max(16, int(args.max_new_tokens))
    command = [
        sys.executable,
        str(ROOT / "scripts" / "public_swarm_inference_v2_pack.py"),
        MODE_LOCAL_MODEL_VARIANT if args.mode == MODE_LOCAL_MODEL_VARIANT else "local",
        "--output-dir",
        str(output_dir),
        "--usable-report",
        args.usable_report,
        "--preview-report",
        args.public_swarm_v2_preview_report,
        "--real-p2p-report",
        args.public_swarm_v2_real_p2p_report,
        "--gpu-report",
        args.gpu_report,
        "--public-host",
        args.public_host,
        "--p2p-port",
        str(args.public_swarm_v2_p2p_port),
        "--coordinator-port",
        str(args.public_swarm_v2_coordinator_port),
        "--real-p2p-port",
        str(args.public_swarm_v2_real_p2p_port),
        "--real-p2p-coordinator-port",
        str(args.public_swarm_v2_real_p2p_coordinator_port),
        "--real-p2p-libp2p-port",
        str(args.public_swarm_v2_real_p2p_libp2p_port),
        "--real-p2p-discovery-backend",
        args.public_swarm_v2_real_p2p_discovery_backend,
        "--backend",
        args.public_swarm_v2_backend,
        "--hf-model-id",
        args.hf_model_id,
        "--max-new-tokens",
        str(v2_tokens),
        "--startup-timeout",
        str(args.startup_timeout),
        "--timeout-seconds",
        str(args.public_swarm_v2_timeout_seconds),
        "--http-timeout",
        str(max(args.http_timeout, 30.0)),
        "--stream-generation",
        "--json",
    ]
    if args.prompt_texts_file:
        command.extend(["--prompt-texts-file", args.prompt_texts_file])
    else:
        command.extend(["--prompt-texts", args.prompt_texts or f"{args.prompt_text},CrowdTensor v2 batch proof"])
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    return run_json_step(
        "public_swarm_v2_local_p2p_generate",
        command,
        runner=runner,
        timeout_seconds=max(args.public_swarm_v2_timeout_seconds, args.timeout_seconds, args.startup_timeout, 60.0) + 1800.0,
    )


def run_p2p_candidate_local(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    p2p_tokens = max(16, int(args.max_new_tokens))
    p2p_timeout = max(args.public_swarm_v2_timeout_seconds, args.timeout_seconds, args.startup_timeout, 60.0) + 1800.0
    command = [
        sys.executable,
        str(ROOT / "scripts" / "petals_class_p2p_candidate_pack.py"),
        "local-smoke",
        "--output-dir",
        str(output_dir),
        "--runtime-smoke-report",
        args.p2p_runtime_smoke_report,
        "--external-report",
        args.p2p_external_report,
        "--requeue-report",
        args.p2p_requeue_report,
        "--batch-stream-report",
        args.p2p_batch_stream_report,
        "--public-host",
        args.public_host,
        "--p2p-port",
        str(args.p2p_port),
        "--coordinator-port",
        str(args.port),
        "--libp2p-port",
        str(args.p2p_libp2p_port),
        "--max-new-tokens",
        str(p2p_tokens),
        "--timeout-seconds",
        str(p2p_timeout),
        "--startup-timeout",
        str(args.startup_timeout),
        "--session-queue-timeout",
        str(args.public_swarm_v2_timeout_seconds),
        "--miner-timeout",
        str(args.public_swarm_v2_timeout_seconds),
        "--generate-timeout",
        str(args.public_swarm_v2_timeout_seconds),
        "--http-timeout",
        str(max(args.http_timeout, 30.0)),
        "--json",
    ]
    return run_json_step(
        "petals_class_p2p_candidate_local_smoke",
        command,
        runner=runner,
        timeout_seconds=p2p_timeout + 300.0,
    )


def product_summary(payload: dict[str, Any], *, expected_model_id: str = DEFAULT_HF_MODEL_ID) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    product = payload.get("product_beta") if isinstance(payload.get("product_beta"), dict) else {}
    model = model_compatibility(payload, expected_model_id)
    batch = batch_summary(payload)
    stream = stream_summary(payload)
    required = {
        "public_swarm_product_beta_ready",
        "public_swarm_product_beta_user_path_ready",
        "serve_ready",
        "stage0_join_ready",
        "stage1_join_ready",
        "generate_ready",
        "serve_join_generate_loop_ready",
        "public_swarm_generate_ready",
        "decoded_tokens_match",
        "distinct_stage_miners",
        "stage_assignment_valid",
        "cpu_fallback_ready",
        "local_cpu_inference_ready",
    }
    if batch.get("enabled"):
        required.add("public_swarm_generate_batch_ready")
    if stream.get("requested") or stream.get("enabled"):
        required.add("public_swarm_generate_stream_ready")
    stream_ready = stream_evidence_ready(stream, batch)
    path_ready = bool(
        payload.get("ok") is True
        and required <= codes
        and (not batch.get("enabled") or batch.get("batch_generation_ready") is True)
        and (not (stream.get("requested") or stream.get("enabled")) or stream_ready)
    )
    ready = bool(path_ready and model["compatible"])
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "ready": ready,
        "path_ready": path_ready,
        "mode": payload.get("mode"),
        "diagnosis_codes": sorted(codes),
        "missing_codes": sorted(required - codes),
        "workload_type": product.get("workload_type"),
        "hf_model_id": product.get("hf_model_id"),
        "model": model,
        "max_new_tokens": product.get("max_new_tokens"),
        "batch": batch,
        "stream_ready": stream_ready,
        "stream": stream,
        "user_surface": product.get("user_surface") or ["serve", "join", "generate"],
    }


def external_summary(
    payload: dict[str, Any],
    *,
    expected_model_id: str = DEFAULT_HF_MODEL_ID,
    min_generated_tokens: int = 1,
) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    requeue = safe_live_requeue_summary(payload)
    model = model_compatibility(payload, expected_model_id)
    generation = generation_summary(payload)
    generated_token_count = safe_int(generation.get("generated_token_count"))
    token_target_ready = generated_token_count >= min_generated_tokens
    required = {
        "external_runtime_verified",
        "decoded_tokens_match",
        "distinct_stage_miners",
        "stage_assignment_valid",
        "kaggle_kernels_deleted",
    }
    requeue_ready = bool(
        "external_stage_requeue_ready" in codes
        and (
            "live_stage0_requeue_ready" in codes
            or "live_stage1_requeue_ready" in codes
            or requeue.get("rescued_result") is True
        )
        and external_requeue_detail_ready(requeue)
    )
    generate_ready = bool(
        "real_llm_internet_beta_ready" in codes
        or "generation_complete" in codes
        or "remote_real_llm_sharded_ready" in codes
    )
    ready = bool(
        payload.get("ok") is True
        and required <= codes
        and generate_ready
        and token_target_ready
        and requeue_ready
        and model["compatible"]
    )
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "ready": ready,
        "mode": payload.get("mode"),
        "diagnosis_codes": sorted(codes),
        "missing_codes": sorted(required - codes),
        "generate_ready": generate_ready,
        "generated_token_count": generated_token_count,
        "required_generated_token_count": min_generated_tokens,
        "token_target_ready": token_target_ready,
        "external_runtime_verified": "external_runtime_verified" in codes,
        "stage_requeue_ready": requeue_ready,
        "claim_observed": requeue.get("claim_observed"),
        "victim_kernel_deleted": requeue.get("victim_kernel_deleted"),
        "lease_expired": requeue.get("lease_expired"),
        "rescued_result": requeue.get("rescued_result"),
        "victim_result_not_accepted": requeue.get("victim_result_accepted") is False,
        "live_requeue_summary": requeue,
        "model": model,
        "generation": generation,
    }


def p2p_summary(
    payload: dict[str, Any],
    *,
    expected_model_id: str = DEFAULT_HF_MODEL_ID,
    min_generated_tokens: int = 1,
) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
    requeue = safe_live_requeue_summary(payload)
    model = model_compatibility(payload, expected_model_id)
    generation = generation_summary(payload)
    generated_token_count = max(
        safe_int(generation.get("generated_token_count")),
        safe_int(candidate.get("external_generated_token_count")),
    )
    token_target_ready = generated_token_count >= min_generated_tokens
    batch = batch_summary(payload)
    stream = stream_summary(payload)
    batch_ready = bool(batch.get("enabled") and batch.get("batch_generation_ready") is True)
    stream_ready = stream_evidence_ready(stream, batch)
    live_requeue_ready = bool(
        live_requeue_detail_ready(requeue)
        and (
            "p2p_live_requeue_rescue_ready" in codes
            or (
                "external_stage_requeue_ready" in codes
                and "rescue_miner_used" in codes
                and "accepted_result_after_requeue" in codes
            )
        )
    )
    required = {
        "petals_class_p2p_candidate_ready",
        "peer_scoring_ready",
        "external_multi_node_generation_ready",
        "external_stage_requeue_ready",
        "p2p_live_requeue_rescue_ready",
        "p2p_victim_result_not_accepted",
    }
    ready = bool(
        payload.get("ok") is True
        and required <= codes
        and live_requeue_ready
        and token_target_ready
        and model["compatible"]
    )
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "ready": ready,
        "mode": payload.get("mode"),
        "diagnosis_codes": sorted(codes),
        "missing_codes": sorted(required - codes),
        "real_p2p_discovery_ready": "libp2p_discovery_backend_ready" in codes or "real_p2p_provider_catalog_ready" in codes,
        "peer_scoring_ready": "peer_scoring_ready" in codes or candidate.get("peer_scoring_ready") is True,
        "p2p_live_requeue_ready": live_requeue_ready,
        "victim_result_not_accepted": requeue.get("victim_result_accepted") is False,
        "live_requeue_summary": requeue,
        "model": model,
        "generated_token_count": generated_token_count,
        "required_generated_token_count": min_generated_tokens,
        "token_target_ready": token_target_ready,
        "batch_ready": batch_ready,
        "batch": batch,
        "stream_ready": stream_ready,
        "stream": stream,
        "accepted_rows": candidate.get("accepted_rows"),
    }


def gpu_summary(payload: dict[str, Any]) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    beta = payload.get("beta") if isinstance(payload.get("beta"), dict) else {}
    cuda_unavailable = "cuda_runtime_unavailable" in codes
    cuda_ready = "public_swarm_gpu_beta_ready" in codes or "gpu_runtime_ready" in codes
    fail_closed = bool(payload.get("ok") is True and (cuda_unavailable or cuda_ready))
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "ready": bool(payload.get("ok") is True and "public_swarm_gpu_beta_ready" in codes),
        "fail_closed_ready": fail_closed,
        "cuda_available": beta.get("cuda_available"),
        "backend": beta.get("backend"),
        "diagnosis_codes": sorted(codes),
        "optional": True,
        "required_for_beta": False,
    }


def public_swarm_v2_summary(
    payload: dict[str, Any],
    *,
    expected_model_id: str = DEFAULT_HF_MODEL_ID,
    min_generated_tokens: int = 1,
    local_model_variant: bool = False,
) -> dict[str, Any]:
    codes = local_model_variant_codes(payload) if local_model_variant else set(diagnosis_codes(payload))
    v2 = payload.get("public_swarm_v2") if isinstance(payload.get("public_swarm_v2"), dict) else {}
    readiness = payload.get("readiness") if isinstance(payload.get("readiness"), dict) else {}
    local = readiness.get("local_p2p_generate") if isinstance(readiness.get("local_p2p_generate"), dict) else {}
    external = readiness.get("external_validation") if isinstance(readiness.get("external_validation"), dict) else {}
    route = readiness.get("p2p_route_hardening") if isinstance(readiness.get("p2p_route_hardening"), dict) else {}
    real_p2p_local = readiness.get("real_p2p_local_route_hardening") if isinstance(readiness.get("real_p2p_local_route_hardening"), dict) else {}
    performance = readiness.get("performance") if isinstance(readiness.get("performance"), dict) else {}
    source_verdict = payload.get("inference_verdict") if isinstance(payload.get("inference_verdict"), dict) else {}
    local_model = local.get("model") if isinstance(local.get("model"), dict) else model_compatibility(local, expected_model_id)
    external_model = external.get("model") if isinstance(external.get("model"), dict) else model_compatibility(external, expected_model_id)
    route_model = route.get("model") if isinstance(route.get("model"), dict) else model_compatibility(route, expected_model_id)
    models_compatible = bool(
        local_model.get("compatible")
        and (local_model_variant or external_model.get("compatible"))
        and route_model.get("compatible")
        and local_model.get("observed_hf_model_id") == expected_model_id
        and (local_model_variant or external_model.get("observed_hf_model_id") == expected_model_id)
        and route_model.get("observed_hf_model_id") == expected_model_id
    )
    generated_token_count = safe_int(local.get("generated_token_count"))
    external_generated_token_count = safe_int(external.get("generated_token_count"))
    accepted_rows = safe_int(local.get("accepted_rows"))
    external_accepted_rows = safe_int(external.get("accepted_rows"))
    required_stage_rows = int(min_generated_tokens) * 2
    kv_cache = local.get("kv_cache") if isinstance(local.get("kv_cache"), dict) else {}
    stage0 = kv_cache.get("stage0") if isinstance(kv_cache.get("stage0"), dict) else {}
    stage1 = kv_cache.get("stage1") if isinstance(kv_cache.get("stage1"), dict) else {}
    expected_hits = max(1, int(min_generated_tokens) - 1)
    kv_cache_ready = bool(
        local.get("kv_cache_ready") is True
        and kv_cache.get("ready") is True
        and safe_int(stage0.get("hit_count")) >= expected_hits
        and safe_int(stage1.get("hit_count")) >= expected_hits
    )
    batch = safe_batch_from_batch(local.get("batch") if isinstance(local.get("batch"), dict) else {})
    stream = safe_stream_from_stream(local.get("stream") if isinstance(local.get("stream"), dict) else {}, batch)
    batch_ready = bool(batch.get("enabled") and batch.get("batch_generation_ready") is True)
    stream_ready = bool(local.get("stream_ready") is not False and stream_evidence_ready(stream, batch))
    real_p2p_local_route_ready = real_p2p_local.get("ready") is True
    real_p2p_local_stage_requeue_ready = bool(
        real_p2p_local_route_ready and real_p2p_local.get("stage_requeue_ready") is True
    )
    performance_ready = bool(
        performance.get("stage_latency_ready") is True
        and performance.get("throughput_summary_ready") is True
        and performance.get("memory_or_vram_summary_ready") is True
    )
    ready = bool(
        payload.get("schema") == PUBLIC_SWARM_V2_SCHEMA
        and payload.get("ok") is True
        and v2.get("ready") is True
        and (
            "public_swarm_inference_v2_ready" in codes
            or (local_model_variant and "public_swarm_inference_v2_local_model_variant_ready" in codes)
        )
        and local.get("ready") is True
        and local.get("route_source") == "p2p-discovery"
        and generated_token_count >= int(min_generated_tokens)
        and accepted_rows >= required_stage_rows
        and kv_cache_ready
        and local.get("stage_requeue_rescue_ready") is True
        and route.get("ready") is True
        and (local_model_variant or external.get("ready") is True)
        and (local_model_variant or external_generated_token_count >= int(min_generated_tokens))
        and (local_model_variant or external_accepted_rows >= required_stage_rows)
        and batch_ready
        and stream_ready
        and performance_ready
        and models_compatible
    )
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "ready": ready,
        "mode": payload.get("mode"),
        "model": {
            "expected_hf_model_id": expected_model_id,
            "local": local_model,
            "external": external_model,
            "p2p_route": route_model,
            "compatible": models_compatible,
        },
        "user_surface": v2.get("user_surface") or ["p2pd", "serve", "join", "generate"],
        "route_source": local.get("route_source"),
        "local_p2p_generate_ready": local.get("ready") is True,
        "generated_token_count": generated_token_count,
        "required_generated_token_count": int(min_generated_tokens),
        "generation_target_ready": generated_token_count >= int(min_generated_tokens),
        "accepted_rows": accepted_rows,
        "external_accepted_rows": external_accepted_rows,
        "required_stage_rows": required_stage_rows,
        "accepted_rows_ready": accepted_rows >= required_stage_rows,
        "external_accepted_rows_ready": external_accepted_rows >= required_stage_rows,
        "batch_ready": batch_ready,
        "batch": batch,
        "stream_ready": stream_ready,
        "stream": stream,
        "kv_cache_ready": kv_cache_ready,
        "expected_hit_count_per_stage": expected_hits,
        "stage0_kv_cache_hits": safe_int(stage0.get("hit_count")),
        "stage1_kv_cache_hits": safe_int(stage1.get("hit_count")),
        "stage_requeue_rescue_ready": local.get("stage_requeue_rescue_ready") is True or "public_swarm_v2_stage_requeue_rescue_ready" in codes,
        "external_validation_ready": external.get("ready") is True,
        "external_generation_target_ready": external_generated_token_count >= int(min_generated_tokens),
        "signed_or_real_p2p_ready": route.get("ready") is True or "public_swarm_v2_signed_or_real_p2p_ready" in codes,
        "real_p2p_local_route_hardening_ready": real_p2p_local_route_ready,
        "real_p2p_local_stage_requeue_ready": real_p2p_local_stage_requeue_ready,
        "real_p2p_local_stage_requeue_target": real_p2p_local.get("stage_requeue_target") or "",
        "real_p2p_local_generated_token_count": safe_int(real_p2p_local.get("generated_token_count")),
        "real_p2p_local_discovery_backend": real_p2p_local.get("discovery_backend") or "",
        "performance_ready": performance_ready,
        "fresh_external_runtime_verified": external.get("fresh_external_runtime_verified") is True or "public_swarm_v2_fresh_external_runtime_verified" in codes,
        "retained_external_evidence_ready": (not local_model_variant) and (external.get("retained_external_evidence_ready") is True or "public_swarm_v2_retained_external_evidence_ready" in codes),
        "source_inference_verdict": {
            "schema": source_verdict.get("schema") or INFERENCE_VERDICT_SCHEMA,
            "kind": source_verdict.get("kind") or "Public Swarm v2",
            "state": source_verdict.get("state") or ("ready" if payload.get("ok") is True else "missing"),
            "completed": bool(source_verdict.get("completed", payload.get("ok") is True)),
            "preflight_only": bool(source_verdict.get("preflight_only")),
            "blocked": bool(source_verdict.get("blocked", payload.get("ok") is not True)),
            "evidence_level": source_verdict.get("evidence_level") or "",
            "gpu_state": source_verdict.get("gpu_state") or "unknown",
            "fresh_kaggle_gpu_verified": bool(source_verdict.get("fresh_kaggle_gpu_verified")),
            "next_step": source_verdict.get("next_step") or "",
            "recommended_label": source_verdict.get("recommended_label") or "",
            "public_artifact_safe": bool(source_verdict.get("public_artifact_safe", True)),
        },
        "local_model_variant_only": local_model_variant or v2.get("local_model_variant_only") is True or payload.get("mode") == MODE_LOCAL_MODEL_VARIANT,
        "external_validation_claimed": not local_model_variant,
        "raw_prompts_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "diagnosis_codes": sorted(codes),
    }


def write_runbook(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    path = output_dir / "PUBLIC_REAL_LLM_SWARM_BETA.md"
    coordinator = f"http://{args.public_host}:{args.port}"
    lines = [
        "# CrowdTensor Public Real-LLM Swarm Inference Beta v1",
        "",
        "This Beta is a Coordinator-backed, real-weight tiny/small LLM split inference workflow.",
        "It is the user-facing path for ordinary operators: `serve`, `join`, and `generate`.",
        "",
        "## Install",
        "",
        "```bash",
        "python -m pip install -e '.[hf]'",
        "export CROWDTENSOR_ADMIN_TOKEN='<admin-token>'",
        "export CROWDTENSOR_MINER_TOKEN='<miner-token>'",
        "```",
        "",
        "## Minimal Two-Stage CPU Run",
        "",
        "Coordinator host:",
        "",
        "```bash",
        f"crowdtensor serve --public-host {args.public_host} --bind-host 0.0.0.0 --port {args.port} --admin-token \"$CROWDTENSOR_ADMIN_TOKEN\" --miner-token \"$CROWDTENSOR_MINER_TOKEN\" --run",
        "```",
        "",
        "Stage 0 miner:",
        "",
        "```bash",
        f"crowdtensor join --coordinator-url {coordinator} --stage stage0 --miner-id beta-stage0 --miner-token \"$CROWDTENSOR_MINER_TOKEN\" --run",
        "```",
        "",
        "Stage 1 miner:",
        "",
        "```bash",
        f"crowdtensor join --coordinator-url {coordinator} --stage stage1 --miner-id beta-stage1 --miner-token \"$CROWDTENSOR_MINER_TOKEN\" --run",
        "```",
        "",
        "Generate from the operator machine:",
        "",
        "```bash",
        f"crowdtensor generate --coordinator-url {coordinator} --admin-token \"$CROWDTENSOR_ADMIN_TOKEN\" --prompt \"{args.prompt_text}\" --max-new-tokens {args.max_new_tokens} --include-output",
        "```",
        "",
        "## Verify The Full Beta Contract",
        "",
        "Run the aggregate check before sharing the result:",
        "",
        "```bash",
        shell_command([
            "crowdtensor",
            "public-real-llm-swarm-beta",
            "release",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--http-timeout",
            "30",
            "--json",
        ]),
        shell_command([
            "crowdtensor",
            "public-real-llm-swarm-beta",
            "check",
            "--beta-report",
            str(output_dir / "public_real_llm_swarm_beta.json"),
            "--output-dir",
            f"{output_dir}-check",
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--json",
        ]),
        "```",
        "",
        "A ready release reports `public_real_llm_swarm_beta_ready`, `public_swarm_v2_16_token_generation_ready`, `public_real_llm_swarm_beta_kv_cache_ready`, `public_real_llm_swarm_beta_v2_batch_ready`, and `public_real_llm_swarm_beta_v2_stream_ready`. A ready check reports `public_real_llm_swarm_beta_check_ready`, records `check_source: beta-report`, and writes `public_real_llm_swarm_beta_check.json` with safe artifact paths.",
        "",
        "## Review The Result",
        "",
        f"- Open `{output_dir / 'public_real_llm_swarm_beta.md'}` for the human summary.",
        "- Start with the Markdown `Review`, `Operator Action`, and `Not Completed` sections.",
        f"- Open `{output_dir / 'support_bundle.json'}` for diagnostics.",
        f"- Open `{output_dir / 'public_real_llm_swarm_beta.json'}` for machine-readable evidence.",
        "- If `ok` is false, read the `Not Completed` section first; each item maps to a missing readiness proof or artifact.",
        "",
        "## Share Safely",
        "",
        "- Share `public_real_llm_swarm_beta.json`, `public_real_llm_swarm_beta.md`, and `support_bundle.json`.",
        "- Do not share `operator.private.env`, `miner.private.env`, `miner_registry.json`, runtime `state/`, or raw task logs.",
        "- Public artifacts are hash/count summaries: raw prompts, generated text, generated token ids, credentials, activations, and lease tokens are excluded.",
        "",
        "## Troubleshooting",
        "",
        "- If the generate step cannot route, run `crowdtensor generate --dry-run` with the same coordinator or P2P options and check stage0/stage1 visibility.",
        "- If `not_completed` mentions Public Swarm v2 token targets, rerun with `--max-new-tokens 16` and inspect the v2 report path printed in artifacts.",
        "- If `not_completed` mentions KV cache, inspect the usable swarm KV-cache report and require stage0/stage1 hit counts of at least `max_new_tokens - 1`.",
        "- If external or P2P requeue evidence is missing, import retained evidence with `evidence-import` or rerun the controlled external proof; rotate temporary public demo tokens afterward.",
        "",
        "## Optional Real-P2P Discovery",
        "",
        "```bash",
        "export CROWDTENSOR_P2P_PEER_SECRET='<shared-discovery-secret>'",
        f"crowdtensor p2p-daemon --host 0.0.0.0 --public-host {args.public_host} --port {args.p2p_port} --discovery-backend libp2p-kad --require-signed --record-secret \"$CROWDTENSOR_P2P_PEER_SECRET\" --run",
        f"crowdtensor serve --p2p --p2p-backend real --peer-bootstrap http://{args.public_host}:{args.p2p_port} --peer-secret \"$CROWDTENSOR_P2P_PEER_SECRET\" --public-host {args.public_host} --port {args.port} --run",
        f"crowdtensor join --p2p --p2p-backend real --peer-bootstrap http://{args.public_host}:{args.p2p_port} --peer-secret \"$CROWDTENSOR_P2P_PEER_SECRET\" --stage stage0 --run",
        f"crowdtensor join --p2p --p2p-backend real --peer-bootstrap http://{args.public_host}:{args.p2p_port} --peer-secret \"$CROWDTENSOR_P2P_PEER_SECRET\" --stage stage1 --run",
        f"crowdtensor generate --p2p --p2p-backend real --peer-bootstrap http://{args.public_host}:{args.p2p_port} --prompt \"{args.prompt_text}\" --max-new-tokens {args.max_new_tokens}",
        "```",
        "",
        "## Optional CUDA",
        "",
        "CPU is the default. CUDA miners must explicitly use `--backend cuda`; hosts without CUDA should fail closed and report `cuda_runtime_unavailable` rather than pretending GPU readiness.",
        "",
        "## Boundaries",
        "",
        "- Coordinator remains the lease, session, and result authority.",
        "- Not full Hivemind/Petals production parity.",
        "- Not Coordinator-free execution.",
        "- Not complete DHT/NAT/relay production networking.",
        "- Not large-model throughput or public arbitrary prompt serving.",
        "- Not an economic, staking, billing, or anti-Sybil system.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return artifact_entry(path, output_dir, kind="public_real_llm_swarm_beta_runbook")


def safety_block() -> dict[str, Any]:
    return {
        "coordinator_backed_task_execution": True,
        "read_only_workload": WORKLOAD_TYPE,
        "serve_join_generate_product_loop": True,
        "cpu_default": True,
        "cuda_optional": True,
        "cuda_fail_closed_expected": True,
        "real_p2p_discovery_optional": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "activation_payloads_redacted": True,
        "not_production": True,
        "not_coordinator_free": True,
        "not_hivemind_petals_production": True,
        "not_complete_nat_traversal": True,
        "not_large_model_serving": True,
        "not_public_prompt_serving": True,
        "not_training": True,
        "not_economic_system": True,
    }


def output_request_summary() -> dict[str, Any]:
    return {
        "include_output": False,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Public Real-LLM Swarm Beta artifacts summarize release readiness with counts, "
            "hashes, route evidence, and child report references only. Run `crowdtensor generate` "
            "in human mode to see a local answer."
        ),
    }


def prompt_scope_summary(args: argparse.Namespace) -> dict[str, Any]:
    prompt_count = len(prompt_list_from_args(args))
    prompt_texts_file = str(getattr(args, "prompt_texts_file", "") or "")
    if prompt_texts_file:
        source = "prompt-texts-file"
    elif str(getattr(args, "prompt_texts", "") or ""):
        source = "prompt-texts"
    else:
        source = "prompt-text"
    inline_prompt_text = source in {"prompt-text", "prompt-texts"}
    return {
        "source": source,
        "prompt_count": prompt_count,
        "inline_prompt_text": inline_prompt_text,
        "terminal_next_commands_local_private": inline_prompt_text,
        "terminal_logs_local_private": inline_prompt_text,
        "saved_artifacts_prompt_placeholders": True,
        "saved_artifacts_public_safe": True,
        "prefer_prompt_file_or_stdin_for_shareable_logs": source in {"prompt-text", "prompt-texts", "prompt-texts-file"},
        "prompt_file_path_public": False,
        "raw_prompt_public": False,
        "public_artifact_safe": True,
        "summary": (
            "This aggregate records prompt source/count and placeholder safety only; "
            "raw prompt text is excluded from public JSON, Markdown, and support bundles."
        ),
    }


def prompt_list_from_args(args: argparse.Namespace) -> list[str]:
    prompt_list = getattr(args, "prompt_texts_list", None)
    if isinstance(prompt_list, list) and prompt_list:
        return [str(prompt) for prompt in prompt_list]
    prompt_texts_file = str(getattr(args, "prompt_texts_file", "") or "")
    if prompt_texts_file:
        return read_prompt_texts_file(prompt_texts_file)
    return parse_prompt_texts_arg(args.prompt_text, args.prompt_texts)


def answer_scope_summary() -> dict[str, Any]:
    return {
        "scope_state": "no-local-answer",
        "terminal_only": False,
        "visible_in_terminal": False,
        "saved_json_display": "hash-only",
        "saved_markdown_display": "hash-only",
        "json_stdout_display": "hash-only-json",
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "public_artifact_safe": True,
        "summary": (
            "This Public Real-LLM Swarm Beta report is shareable aggregate evidence, "
            "not a local answer transcript; raw prompts, generated text, generated token ids, "
            "activations, leases, and credentials are excluded."
        ),
    }


def shareable_summary() -> dict[str, Any]:
    return {
        "saved_artifacts_public_safe": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "answer_scope_state": "no-local-answer",
        "local_answer_terminal_only": False,
        "public_artifact_safe": True,
        "summary": "Share public_real_llm_swarm_beta.json/md and support_bundle.json; they contain hashes/counts and readiness evidence, not raw prompts or answers.",
    }


def runtime_provenance_summary(report: dict[str, Any]) -> dict[str, Any]:
    mode = str(report.get("mode") or "")
    readiness = report.get("readiness") if isinstance(report.get("readiness"), dict) else {}
    product = readiness.get("product_path") if isinstance(readiness.get("product_path"), dict) else {}
    external = readiness.get("external_kaggle") if isinstance(readiness.get("external_kaggle"), dict) else {}
    p2p = readiness.get("p2p_candidate") if isinstance(readiness.get("p2p_candidate"), dict) else {}
    v2 = readiness.get("public_swarm_v2") if isinstance(readiness.get("public_swarm_v2"), dict) else {}
    kv_cache = readiness.get("usable_p2p_kv_cache") if isinstance(readiness.get("usable_p2p_kv_cache"), dict) else {}
    cuda = readiness.get("cuda_optional") if isinstance(readiness.get("cuda_optional"), dict) else {}
    source_reports = report.get("source_reports") if isinstance(report.get("source_reports"), dict) else {}
    steps = report.get("steps") if isinstance(report.get("steps"), list) else []
    step_names = [
        str(step.get("name"))
        for step in steps
        if isinstance(step, dict) and step.get("name")
    ]
    cuda_codes = set(str(code) for code in (cuda.get("diagnosis_codes") or []))
    gpu_report = str(source_reports.get("gpu_report") or "")
    gpu_mode = str(cuda.get("mode") or "")
    if not gpu_mode:
        artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
        gpu_artifact = artifacts.get("gpu_optional_report") if isinstance(artifacts.get("gpu_optional_report"), dict) else {}
        artifact_path = str(gpu_artifact.get("path") or "")
        if "kaggle_auto" in gpu_report or "kaggle-auto" in gpu_report or "kaggle_auto" in artifact_path:
            gpu_mode = "kaggle-auto"
        elif gpu_report:
            gpu_mode = "evidence-import"
        elif "public_swarm_gpu_beta_local_smoke" in step_names:
            gpu_mode = "local-smoke"
    fresh_kaggle_gpu_attempted = any("kaggle" in name and "gpu" in name for name in step_names)
    fresh_kaggle_gpu_verified = bool(
        fresh_kaggle_gpu_attempted
        and (
            cuda.get("ready") is True
            or "public_swarm_gpu_beta_kaggle_auto_ready" in cuda_codes
            or "public_swarm_gpu_beta_ready" in cuda_codes
        )
    )
    retained_gpu_evidence_imported = bool(mode == MODE_EVIDENCE_IMPORT and gpu_report)
    local_gpu_smoke_ran = "public_swarm_gpu_beta_local_smoke" in step_names
    if mode == MODE_RELEASE:
        proof_level = "release-local-cpu-with-retained-external-and-local-gpu-smoke"
    elif mode == MODE_LOCAL_SMOKE:
        proof_level = "local-cpu-product-smoke"
    elif mode == MODE_LOCAL_MODEL_VARIANT:
        proof_level = "local-model-variant-cpu"
    elif mode == MODE_EVIDENCE_IMPORT:
        proof_level = "retained-evidence-import"
    elif mode == MODE_PACKAGE:
        proof_level = "package-only"
    else:
        proof_level = mode or "unknown"
    return {
        "schema": RUNTIME_PROVENANCE_SCHEMA,
        "proof_level": proof_level,
        "mode": mode,
        "local_cpu_product_path_ready": product.get("ready") is True or product.get("path_ready") is True,
        "local_public_swarm_v2_ready": v2.get("ready") is True,
        "local_kv_cache_ready": kv_cache.get("ready") is True,
        "external_kaggle_cpu_evidence_ready": external.get("ready") is True,
        "external_kaggle_cpu_evidence_claimed": bool(mode in {MODE_RELEASE, MODE_EVIDENCE_IMPORT} and external),
        "p2p_candidate_evidence_ready": p2p.get("ready") is True,
        "p2p_candidate_evidence_claimed": bool(mode in {MODE_RELEASE, MODE_EVIDENCE_IMPORT} and p2p),
        "cuda_optional_fail_closed_ready": cuda.get("fail_closed_ready") is True,
        "gpu_backend": cuda.get("backend") or "",
        "gpu_report_mode": gpu_mode or "none",
        "local_gpu_smoke_ran": local_gpu_smoke_ran,
        "retained_gpu_evidence_imported": retained_gpu_evidence_imported,
        "fresh_kaggle_gpu_attempted": fresh_kaggle_gpu_attempted,
        "fresh_kaggle_gpu_verified": fresh_kaggle_gpu_verified,
        "public_artifact_safe": True,
        "summary": (
            "This aggregate distinguishes local CPU proof, retained external/Kaggle CPU "
            "evidence, optional CUDA fail-closed smoke/import, and any fresh Kaggle GPU proof. "
            "A fresh Kaggle GPU run is only claimed when fresh_kaggle_gpu_verified is true."
        ),
    }


def runtime_provenance_text(provenance: dict[str, Any]) -> str:
    return (
        f"proof={provenance.get('proof_level') or 'unknown'} "
        f"local_cpu_product={bool(provenance.get('local_cpu_product_path_ready'))} "
        f"external_kaggle_cpu={bool(provenance.get('external_kaggle_cpu_evidence_ready'))} "
        f"p2p_candidate={bool(provenance.get('p2p_candidate_evidence_ready'))} "
        f"gpu_mode={provenance.get('gpu_report_mode') or 'none'} "
        f"local_gpu_smoke={bool(provenance.get('local_gpu_smoke_ran'))} "
        f"retained_gpu_import={bool(provenance.get('retained_gpu_evidence_imported'))} "
        f"fresh_kaggle_gpu_attempted={bool(provenance.get('fresh_kaggle_gpu_attempted'))} "
        f"fresh_kaggle_gpu_verified={bool(provenance.get('fresh_kaggle_gpu_verified'))}"
    )


def evidence_scope_summary(report: dict[str, Any]) -> dict[str, Any]:
    provenance = report.get("runtime_provenance") if isinstance(report.get("runtime_provenance"), dict) else runtime_provenance_summary(report)
    mode = str(report.get("mode") or provenance.get("mode") or "")
    local_cpu = bool(
        provenance.get("local_cpu_product_path_ready")
        or provenance.get("local_public_swarm_v2_ready")
        or provenance.get("local_kv_cache_ready")
    )
    retained_external = bool(
        provenance.get("external_kaggle_cpu_evidence_claimed")
        or provenance.get("p2p_candidate_evidence_claimed")
    )
    retained_external_ready = bool(
        provenance.get("external_kaggle_cpu_evidence_ready")
        or provenance.get("p2p_candidate_evidence_ready")
    )
    retained_gpu = bool(provenance.get("retained_gpu_evidence_imported"))
    fresh_kaggle_gpu_attempted = bool(provenance.get("fresh_kaggle_gpu_attempted"))
    fresh_kaggle_gpu_verified = bool(provenance.get("fresh_kaggle_gpu_verified"))
    local_gpu_smoke = bool(provenance.get("local_gpu_smoke_ran"))
    if fresh_kaggle_gpu_verified:
        level = "fresh-kaggle-gpu"
        executed_where = "fresh-kaggle-gpu-plus-aggregate"
        expectation = "This aggregate verified a fresh Kaggle GPU proof in the source report."
    elif mode == MODE_PACKAGE:
        level = "package-only"
        executed_where = "not-run"
        expectation = "Package mode writes runbooks and artifacts only; no inference proof ran."
    elif mode == MODE_EVIDENCE_IMPORT:
        level = "retained-evidence-import"
        executed_where = "retained-evidence"
        expectation = "This report imports retained evidence; retained GPU or external evidence is not a fresh run."
    elif mode == MODE_LOCAL_MODEL_VARIANT:
        level = "local-model-variant"
        executed_where = "local-cpu"
        expectation = "This report proves a local CPU model variant and does not claim external or Kaggle validation."
    elif local_cpu and retained_external:
        level = "release-local-cpu-with-retained-external"
        executed_where = "local-cpu-plus-retained-evidence"
        expectation = "This release mixes fresh local CPU checks with retained external/P2P evidence; fresh Kaggle GPU is false unless explicitly verified."
    elif local_cpu:
        level = "local-cpu"
        executed_where = "local-cpu"
        expectation = "This report ran local CPU product or smoke evidence."
    else:
        level = "blocked-or-mixed"
        executed_where = "mixed-or-not-proven"
        expectation = "Inspect not_completed and runtime_provenance for missing evidence."
    return {
        "schema": EVIDENCE_SCOPE_SCHEMA,
        "level": level,
        "executed_where": executed_where,
        "source": SCHEMA,
        "mode": mode,
        "proof_level": provenance.get("proof_level") or "",
        "local_cpu": local_cpu,
        "local_cpu_product_path_ready": bool(provenance.get("local_cpu_product_path_ready")),
        "local_public_swarm_v2_ready": bool(provenance.get("local_public_swarm_v2_ready")),
        "local_kv_cache_ready": bool(provenance.get("local_kv_cache_ready")),
        "retained_external_evidence_imported": retained_external,
        "retained_external_evidence_ready": retained_external_ready,
        "external_kaggle_cpu_evidence_ready": bool(provenance.get("external_kaggle_cpu_evidence_ready")),
        "p2p_candidate_evidence_ready": bool(provenance.get("p2p_candidate_evidence_ready")),
        "local_gpu_smoke_ran": local_gpu_smoke,
        "retained_gpu_evidence_imported": retained_gpu,
        "fresh_kaggle_gpu_attempted": fresh_kaggle_gpu_attempted,
        "fresh_kaggle_gpu_verified": fresh_kaggle_gpu_verified,
        "public_artifact_safe": True,
        "user_expectation": expectation,
    }


def evidence_scope_text(scope: dict[str, Any]) -> str:
    return (
        f"level={scope.get('level') or 'unknown'} "
        f"executed={scope.get('executed_where') or 'unknown'} "
        f"source={scope.get('source') or 'unknown'} "
        f"local_cpu={bool(scope.get('local_cpu'))} "
        f"retained_external={bool(scope.get('retained_external_evidence_imported'))} "
        f"local_gpu_smoke={bool(scope.get('local_gpu_smoke_ran'))} "
        f"retained_gpu={bool(scope.get('retained_gpu_evidence_imported'))} "
        f"fresh_kaggle_gpu_attempted={bool(scope.get('fresh_kaggle_gpu_attempted'))} "
        f"fresh_kaggle_gpu={bool(scope.get('fresh_kaggle_gpu_verified'))} "
        f"user_expectation={scope.get('user_expectation') or 'unknown'}"
    )


def gpu_status_summary(provenance: dict[str, Any], scope: dict[str, Any] | None = None) -> dict[str, Any]:
    provenance = provenance if isinstance(provenance, dict) else {}
    scope = scope if isinstance(scope, dict) else {}
    fresh_verified = bool(provenance.get("fresh_kaggle_gpu_verified") or scope.get("fresh_kaggle_gpu_verified"))
    fresh_attempted = bool(provenance.get("fresh_kaggle_gpu_attempted") or scope.get("fresh_kaggle_gpu_attempted"))
    retained_gpu = bool(provenance.get("retained_gpu_evidence_imported") or scope.get("retained_gpu_evidence_imported"))
    local_gpu_smoke = bool(provenance.get("local_gpu_smoke_ran") or scope.get("local_gpu_smoke_ran"))
    local_cpu = bool(
        scope.get("local_cpu")
        or provenance.get("local_cpu_product_path_ready")
        or provenance.get("local_public_swarm_v2_ready")
        or provenance.get("local_kv_cache_ready")
    )
    if fresh_verified:
        state = "fresh-kaggle-gpu-verified"
        note = "this report verified a fresh Kaggle GPU proof"
    elif fresh_attempted:
        state = "fresh-kaggle-gpu-attempted-unverified"
        note = "a Kaggle GPU path was attempted but not verified"
    elif retained_gpu:
        state = "retained-gpu-evidence"
        note = "GPU evidence is imported or retained, not fresh from this command"
    elif local_gpu_smoke:
        state = "local-gpu-smoke-only"
        note = "only a local or CI GPU smoke path is represented"
    elif local_cpu:
        state = "local-cpu-only"
        note = "this report proves local CPU inference and does not claim GPU execution"
    else:
        state = "no-gpu-evidence"
        note = "no GPU execution evidence is claimed by this report"
    return {
        "schema": "crowdtensor_gpu_status_v1",
        "state": state,
        "local_cpu": local_cpu,
        "local_gpu_smoke": local_gpu_smoke,
        "retained_gpu": retained_gpu,
        "fresh_kaggle_gpu_attempted": fresh_attempted,
        "fresh_kaggle_gpu_verified": fresh_verified,
        "note": note,
        "public_artifact_safe": True,
    }


def gpu_status_text(provenance: dict[str, Any], scope: dict[str, Any] | None = None) -> str:
    status = (
        provenance
        if scope is None and isinstance(provenance, dict) and isinstance(provenance.get("state"), str)
        else gpu_status_summary(provenance, scope)
    )
    return (
        f"state={status.get('state')} "
        f"local_cpu={bool(status.get('local_cpu'))} "
        f"local_gpu_smoke={bool(status.get('local_gpu_smoke'))} "
        f"retained_gpu={bool(status.get('retained_gpu'))} "
        f"fresh_kaggle_gpu_attempted={bool(status.get('fresh_kaggle_gpu_attempted'))} "
        f"fresh_kaggle_gpu_verified={bool(status.get('fresh_kaggle_gpu_verified'))} "
        f"note={status.get('note') or ''}"
    )


def gpu_proof_next_step_summary(gpu_status: dict[str, Any]) -> dict[str, Any]:
    status = gpu_status if isinstance(gpu_status, dict) else {}
    state = str(status.get("state") or "no-gpu-evidence")
    verified = bool(status.get("fresh_kaggle_gpu_verified"))
    note = (
        "Fresh Kaggle GPU proof is already verified in this report."
        if verified
        else "This report does not prove a fresh Kaggle GPU run; use the explicit GPU proof commands when you need that evidence."
    )
    commands = [
        {
            **command_entry(
                "run local GPU smoke",
                [
                    "crowdtensor",
                    "public-swarm-gpu-beta",
                    "local-smoke",
                    "--output-dir",
                    "dist/public-swarm-gpu-beta",
                    "--json",
                ],
            ),
            "reason": "check_optional_cuda_locally",
            "side_effectful": False,
            "requires_kaggle": False,
            "cleanup_required": False,
            "public_artifact_safe": True,
        },
        {
            **command_entry(
                "package Kaggle GPU proof",
                [
                    "crowdtensor",
                    "public-swarm-gpu-beta",
                    "kaggle-package",
                    "--output-dir",
                    "dist/public-swarm-gpu-beta-kaggle",
                    "--json",
                ],
            ),
            "reason": "prepare_private_kaggle_gpu_templates",
            "side_effectful": False,
            "requires_kaggle": False,
            "cleanup_required": False,
            "public_artifact_safe": True,
        },
        {
            **command_entry(
                "run fresh Kaggle GPU proof",
                [
                    "crowdtensor",
                    "public-swarm-gpu-beta",
                    "kaggle-auto",
                    "--public-host",
                    "24.199.118.54",
                    "--port",
                    "9320",
                    "--base-port",
                    "9321",
                    "--kaggle-owner",
                    "YOUR_KAGGLE_USERNAME",
                    "--request-count",
                    "1",
                    "--json",
                ],
            ),
            "reason": "verify_fresh_kaggle_gpu",
            "side_effectful": True,
            "requires_kaggle": True,
            "cleanup_required": True,
            "token_rotation_required": True,
            "public_artifact_safe": True,
        },
        {
            **command_entry(
                "run multi-token Kaggle GPU generation",
                [
                    "crowdtensor",
                    "gpu-generate",
                    "kaggle-auto",
                    "--kaggle-owner",
                    "YOUR_KAGGLE_USERNAME",
                    "--max-new-tokens",
                    "16",
                    "--json",
                ],
            ),
            "reason": "verify_multi_token_gpu_generation",
            "side_effectful": True,
            "requires_kaggle": True,
            "cleanup_required": True,
            "token_rotation_required": True,
            "public_artifact_safe": True,
        },
    ]
    return {
        "schema": "crowdtensor_gpu_proof_next_step_v1",
        "state": "fresh-kaggle-gpu-verified" if verified else "fresh-kaggle-gpu-not-verified",
        "current_gpu_status_state": state,
        "fresh_kaggle_gpu_verified": verified,
        "recommended_label": "review verified GPU proof" if verified else "run fresh Kaggle GPU proof",
        "recommended_reason": "already_verified" if verified else "verify_fresh_kaggle_gpu",
        "note": note,
        "requires_explicit_user_action": not verified,
        "side_effectful": not verified,
        "requires_kaggle": not verified,
        "cleanup_required": not verified,
        "token_rotation_required": not verified,
        "commands": [] if verified else commands,
        "public_artifact_safe": True,
    }


def gpu_proof_next_step_text(summary: dict[str, Any]) -> str:
    if not isinstance(summary, dict):
        return ""
    command_items = summary.get("commands") if isinstance(summary.get("commands"), list) else []
    command_text = ""
    for item in command_items:
        if isinstance(item, dict) and item.get("reason") == "verify_fresh_kaggle_gpu":
            command_text = str(item.get("command_line") or "")
            break
    if not command_text and command_items:
        first = command_items[0]
        command_text = str(first.get("command_line") or "") if isinstance(first, dict) else ""
    suffix = f" command={command_text}" if command_text else ""
    return (
        f"state={summary.get('state') or 'unknown'} "
        f"recommended={summary.get('recommended_label') or 'none'} "
        f"reason={summary.get('recommended_reason') or 'none'} "
        f"requires_kaggle={bool(summary.get('requires_kaggle'))} "
        f"side_effectful={bool(summary.get('side_effectful'))} "
        f"cleanup_required={bool(summary.get('cleanup_required'))} "
        f"token_rotation_required={bool(summary.get('token_rotation_required'))}"
        f"{suffix}"
    )


def source_inference_verdict_summary(report: dict[str, Any]) -> dict[str, Any]:
    readiness = report.get("readiness") if isinstance(report.get("readiness"), dict) else {}
    public_swarm_v2 = readiness.get("public_swarm_v2") if isinstance(readiness.get("public_swarm_v2"), dict) else {}
    verdict = public_swarm_v2.get("source_inference_verdict")
    if isinstance(verdict, dict):
        return {
            "schema": verdict.get("schema") or INFERENCE_VERDICT_SCHEMA,
            "kind": verdict.get("kind") or "Public Swarm v2",
            "state": verdict.get("state") or "unknown",
            "completed": bool(verdict.get("completed")),
            "preflight_only": bool(verdict.get("preflight_only")),
            "blocked": bool(verdict.get("blocked")),
            "evidence_level": verdict.get("evidence_level") or "",
            "gpu_state": verdict.get("gpu_state") or "unknown",
            "fresh_kaggle_gpu_verified": bool(verdict.get("fresh_kaggle_gpu_verified")),
            "next_step": verdict.get("next_step") or "",
            "recommended_label": verdict.get("recommended_label") or "",
            "public_artifact_safe": bool(verdict.get("public_artifact_safe", True)),
        }
    return {
        "schema": INFERENCE_VERDICT_SCHEMA,
        "kind": "Public Swarm v2",
        "state": "ready" if public_swarm_v2.get("ready") is True else "missing",
        "completed": bool(public_swarm_v2.get("ready") is True),
        "preflight_only": False,
        "blocked": public_swarm_v2.get("ready") is not True,
        "evidence_level": "",
        "gpu_state": "unknown",
        "fresh_kaggle_gpu_verified": False,
        "next_step": "",
        "recommended_label": "",
        "public_artifact_safe": True,
    }


def inference_verdict_summary(report: dict[str, Any]) -> dict[str, Any]:
    beta = report.get("beta") if isinstance(report.get("beta"), dict) else {}
    readiness = report.get("readiness") if isinstance(report.get("readiness"), dict) else {}
    public_swarm_v2 = readiness.get("public_swarm_v2") if isinstance(readiness.get("public_swarm_v2"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    evidence_scope = report.get("evidence_scope") if isinstance(report.get("evidence_scope"), dict) else {}
    gpu_status = report.get("gpu_status") if isinstance(report.get("gpu_status"), dict) else {}
    review = report.get("review_summary") if isinstance(report.get("review_summary"), dict) else {}
    user = report.get("user_status") if isinstance(report.get("user_status"), dict) else {}
    recommended = report.get("recommended_next_command") if isinstance(report.get("recommended_next_command"), dict) else {}
    artifacts = report.get("artifact_summary") if isinstance(report.get("artifact_summary"), dict) else {}
    source_verdict = source_inference_verdict_summary(report)
    ready = report_ready(report)
    mode = str(report.get("mode") or "")
    if ready and mode == MODE_LOCAL_SMOKE:
        state = "local-smoke-ready"
    elif ready and mode == MODE_LOCAL_MODEL_VARIANT:
        state = "local-model-ready"
    elif ready and mode == MODE_PACKAGE:
        state = "package-ready"
    else:
        state = str(review.get("state") or user.get("state") or ("ready" if ready else "blocked"))
    preflight_only = mode == MODE_PACKAGE
    if preflight_only and ready:
        result_status = "package-only"
    elif ready:
        result_status = "ready"
    else:
        result_status = "blocked"
    if ready:
        message = (
            "Public Real-LLM Swarm Beta evidence is ready. This top-level report proves "
            "the shareable inference evidence path; raw answers are not saved."
        )
    elif preflight_only:
        message = "Public Real-LLM Swarm Beta package artifacts are ready, but no inference proof ran in package mode."
    else:
        message = "Public Real-LLM Swarm Beta evidence is blocked; review Not Completed and the recommended command."
    if ready and mode == MODE_LOCAL_SMOKE:
        primary_code = "public_real_llm_swarm_beta_local_smoke_ready"
        message = (
            "Public Real-LLM Swarm Beta local smoke evidence is ready. This scoped "
            "report proves the local CPU serve/join/generate path and CUDA fail-closed "
            "behavior; it does not claim the full release aggregate or fresh Kaggle GPU."
        )
    elif ready and mode == MODE_LOCAL_MODEL_VARIANT:
        primary_code = "public_real_llm_swarm_beta_local_model_variant_ready"
    elif ready and mode == MODE_PACKAGE:
        primary_code = "public_real_llm_swarm_beta_package_ready"
    elif ready:
        primary_code = "public_real_llm_swarm_beta_ready"
    else:
        primary_code = "public_real_llm_swarm_beta_blocked"
    return {
        "schema": INFERENCE_VERDICT_SCHEMA,
        "kind": "Public Real-LLM Swarm Beta",
        "state": state,
        "completed": ready,
        "preflight_only": preflight_only,
        "blocked": not ready,
        "result_status": result_status,
        "generated_token_count": int(public_swarm_v2.get("generated_token_count") or 0),
        "max_new_tokens": int(beta.get("max_new_tokens") or public_swarm_v2.get("required_generated_token_count") or 0),
        "output_count": int(public_swarm_v2.get("accepted_rows") or 0),
        "answer_scope_state": answer_scope.get("scope_state") or shareable.get("answer_scope_state") or "unknown",
        "answer_visible_in_terminal": bool(answer_scope.get("visible_in_terminal")),
        "saved_artifacts_public_safe": bool(shareable.get("saved_artifacts_public_safe")),
        "evidence_level": evidence_scope.get("level") or "",
        "executed_where": evidence_scope.get("executed_where") or "",
        "gpu_state": gpu_status.get("state") or "unknown",
        "retained_gpu_evidence_imported": bool(gpu_status.get("retained_gpu")),
        "fresh_kaggle_gpu_verified": bool(gpu_status.get("fresh_kaggle_gpu_verified")),
        "source_inference_verdict": source_verdict,
        "source_inference_verdict_state": source_verdict.get("state") or "unknown",
        "source_inference_verdict_completed": bool(source_verdict.get("completed")),
        "recommended_label": recommended.get("label") or review.get("recommended_label") or "",
        "recommended_reason": recommended.get("reason") or review.get("recommended_reason") or "",
        "next_step": review.get("next_step") or user.get("next_step") or "",
        "primary_code": primary_code,
        "inspect_first": artifacts.get("inspect_first") or review.get("inspect_first") or "",
        "public_artifact_safe": True,
        "message": message,
    }


def inference_verdict_text(verdict: dict[str, Any]) -> str:
    return (
        f"state={verdict.get('state') or 'unknown'} "
        f"completed={bool(verdict.get('completed'))} "
        f"preflight_only={bool(verdict.get('preflight_only'))} "
        f"answer={verdict.get('answer_scope_state') or 'unknown'} "
        f"answer_visible={bool(verdict.get('answer_visible_in_terminal'))} "
        f"artifacts_public={bool(verdict.get('saved_artifacts_public_safe'))} "
        f"evidence={verdict.get('evidence_level') or 'unknown'} "
        f"executed={verdict.get('executed_where') or 'unknown'} "
        f"gpu={verdict.get('gpu_state') or 'unknown'} "
        f"fresh_kaggle_gpu={bool(verdict.get('fresh_kaggle_gpu_verified'))} "
        f"source={verdict.get('source_inference_verdict_state') or 'unknown'} "
        f"next={verdict.get('next_step') or 'none'} "
        f"recommended={verdict.get('recommended_label') or 'none'} "
        f"public_artifact_safe={bool(verdict.get('public_artifact_safe'))}"
    )


def artifact_path_summary(report: dict[str, Any], name: str, fallback: str) -> str:
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    artifact = artifacts.get(name) if isinstance(artifacts.get(name), dict) else {}
    return str(artifact.get("path") or fallback)


def artifact_summary(report: dict[str, Any]) -> dict[str, Any]:
    shareable_names = [
        "public_real_llm_swarm_beta_json",
        "public_real_llm_swarm_beta_markdown",
        "support_bundle_json",
    ]
    shareable_paths = []
    for name in shareable_names:
        path = artifact_path_summary(report, name, "")
        if path:
            shareable_paths.append(path)
    return {
        "schema": ARTIFACT_SUMMARY_SCHEMA,
        "inspect_first": artifact_path_summary(report, "public_real_llm_swarm_beta_markdown", "public_real_llm_swarm_beta.md"),
        "machine_readable": artifact_path_summary(report, "public_real_llm_swarm_beta_json", "public_real_llm_swarm_beta.json"),
        "support_bundle": artifact_path_summary(report, "support_bundle_json", "support_bundle.json"),
        "runbook": artifact_path_summary(report, "runbook", "PUBLIC_REAL_LLM_SWARM_BETA.md"),
        "shareable_artifacts": shareable_names,
        "shareable_paths": shareable_paths,
        "public_artifact_safe": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "summary": "Review the Markdown first, use the JSON for automation, and attach the support bundle for diagnostics.",
    }


def recommended_check_command(report: dict[str, Any]) -> dict[str, Any]:
    mode = str(report.get("mode") or "")
    if mode not in {MODE_RELEASE, MODE_LOCAL_SMOKE, MODE_LOCAL_MODEL_VARIANT}:
        return {}
    output_dir = str(report.get("output_dir") or "").strip()
    if not output_dir:
        return {}
    max_new_tokens = int((report.get("beta") if isinstance(report.get("beta"), dict) else {}).get("max_new_tokens") or 16)
    model_id = str((report.get("beta") if isinstance(report.get("beta"), dict) else {}).get("hf_model_id") or DEFAULT_HF_MODEL_ID)
    beta_report = Path(output_dir) / "public_real_llm_swarm_beta.json"
    check_dir = Path(f"{output_dir}-check")
    command = [
        "crowdtensor",
        "public-real-llm-swarm-beta",
        "check",
    ]
    if model_id != DEFAULT_HF_MODEL_ID:
        command.extend(["--hf-model-id", model_id])
    command.extend([
        "--beta-report",
        str(beta_report),
        "--output-dir",
        str(check_dir),
        "--max-new-tokens",
        str(max_new_tokens),
        "--json",
    ])
    return {
        "label": "validate beta report",
        "reason": "check_current_beta_report",
        "command_line": shell_command(command),
        "beta_report": str(beta_report),
        "output_dir": str(check_dir),
        "check_source": "beta-report",
        "public_artifact_safe": True,
    }


def command_entry(
    label: str,
    command: list[str],
    *,
    reason: str = "",
    side_effectful: bool = False,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "label": label,
        "command": [str(part) for part in command],
        "command_line": shell_command(command),
        "public_artifact_safe": True,
    }
    if reason:
        entry["reason"] = reason
    if side_effectful:
        entry["side_effectful"] = True
    return entry


def inspect_artifact_command(report: dict[str, Any], filename: str) -> list[str]:
    output_dir = Path(str(report.get("output_dir") or "."))
    return ["sed", "-n", "1,220p", str(output_dir / filename)]


def release_command(report: dict[str, Any], mode: str) -> dict[str, Any]:
    beta = report.get("beta") if isinstance(report.get("beta"), dict) else {}
    output_dir = str(report.get("output_dir") or DEFAULT_OUTPUT_DIR)
    command = [
        "crowdtensor",
        "public-real-llm-swarm-beta",
        mode,
        "--output-dir",
        output_dir,
        "--hf-model-id",
        str(beta.get("hf_model_id") or DEFAULT_HF_MODEL_ID),
        "--max-new-tokens",
        str(beta.get("max_new_tokens") or 16),
        "--json",
    ]
    return command_entry(
        "run Public Real-LLM Swarm Beta release",
        command,
        reason="run_full_beta_release",
    )


def report_ready(report: dict[str, Any]) -> bool:
    beta = report.get("beta") if isinstance(report.get("beta"), dict) else {}
    not_completed = report.get("not_completed") if isinstance(report.get("not_completed"), list) else []
    return bool(report.get("ok") is True and beta.get("ready") is True and not not_completed)


def recommended_next_command(report: dict[str, Any]) -> dict[str, Any]:
    mode = str(report.get("mode") or "")
    ready = report_ready(report)
    check_command = report.get("recommended_check_command") if isinstance(report.get("recommended_check_command"), dict) else {}
    if ready and check_command:
        return dict(check_command)
    if ready:
        return command_entry(
            "inspect shareable summary",
            inspect_artifact_command(report, "public_real_llm_swarm_beta.md"),
            reason="review_artifacts",
        )
    if mode == MODE_PACKAGE:
        return release_command(report, MODE_RELEASE)
    return command_entry(
        "inspect Not Completed section",
        inspect_artifact_command(report, "public_real_llm_swarm_beta.md"),
        reason="review_not_completed",
    )


def next_commands(report: dict[str, Any]) -> list[dict[str, Any]]:
    commands = [
        command_entry(
            "inspect shareable summary",
            inspect_artifact_command(report, "public_real_llm_swarm_beta.md"),
            reason="review_artifacts",
        ),
        command_entry(
            "inspect support bundle",
            inspect_artifact_command(report, "support_bundle.json"),
            reason="inspect_diagnostics",
        ),
    ]
    recommended = report.get("recommended_next_command") if isinstance(report.get("recommended_next_command"), dict) else {}
    if recommended and all(item.get("command_line") != recommended.get("command_line") for item in commands):
        commands.append(dict(recommended))
    check_command = report.get("recommended_check_command") if isinstance(report.get("recommended_check_command"), dict) else {}
    if check_command and all(item.get("command_line") != check_command.get("command_line") for item in commands):
        commands.append(dict(check_command))
    if str(report.get("mode") or "") == MODE_PACKAGE:
        release = release_command(report, MODE_RELEASE)
        if all(item.get("command_line") != release.get("command_line") for item in commands):
            commands.append(release)
    return commands


def user_status(report: dict[str, Any]) -> dict[str, Any]:
    mode = str(report.get("mode") or "")
    ready = report_ready(report)
    not_completed = report.get("not_completed") if isinstance(report.get("not_completed"), list) else []
    recommended = report.get("recommended_next_command") if isinstance(report.get("recommended_next_command"), dict) else {}
    if ready and mode == MODE_LOCAL_SMOKE:
        state = "local-smoke-ready"
        headline = "Local CPU smoke evidence is ready; full release and fresh Kaggle GPU are not claimed."
        next_step = "run_beta_report_check"
    elif ready and mode == MODE_LOCAL_MODEL_VARIANT:
        state = "local-model-ready"
        headline = "Local model variant evidence is ready; external validation is not claimed."
        next_step = "run_beta_report_check"
    elif ready and mode == MODE_PACKAGE:
        state = "package-ready"
        headline = "Public Real-LLM Swarm Beta package is ready; run release next."
        next_step = "run_release"
    elif ready:
        state = "ready"
        headline = "Public Real-LLM Swarm Beta evidence is ready."
        next_step = "run_beta_report_check"
    else:
        state = "blocked"
        headline = "Public Real-LLM Swarm Beta evidence needs attention."
        next_step = "review_not_completed" if not_completed else "review_diagnostics"
    return {
        "state": state,
        "headline": headline,
        "next_step": next_step,
        "recommended_label": recommended.get("label") or "none",
        "recommended_reason": recommended.get("reason") or "none",
        "not_completed_count": len(not_completed),
        "public_artifact_safe": True,
    }


def review_summary(report: dict[str, Any]) -> dict[str, Any]:
    beta = report.get("beta") if isinstance(report.get("beta"), dict) else {}
    not_completed = [str(item) for item in (report.get("not_completed") if isinstance(report.get("not_completed"), list) else [])]
    artifacts = report.get("artifact_summary") if isinstance(report.get("artifact_summary"), dict) else artifact_summary(report)
    ready = bool(report.get("ok") is True and beta.get("ready") is True and not not_completed)
    check_command = (
        report.get("recommended_check_command")
        if isinstance(report.get("recommended_check_command"), dict)
        else recommended_check_command(report)
    )
    next_command = (
        report.get("recommended_next_command")
        if isinstance(report.get("recommended_next_command"), dict)
        else recommended_next_command(report)
    )
    next_step = "run_beta_report_check" if ready and check_command else ("share_public_artifacts" if ready else ("review_not_completed" if not_completed else "review_diagnostics"))
    operator_action = [
        str(item)
        for item in (report.get("operator_action") if isinstance(report.get("operator_action"), list) else [])
    ]
    if ready and check_command:
        summary_text = "Ready: run the beta-report check, then share the public Markdown, public JSON, and support bundle."
    elif ready:
        summary_text = "Ready: share the public Markdown, public JSON, and support bundle."
    elif not_completed:
        summary_text = "Blocked: review the Not Completed section, then rerun the Beta check after fixing the missing evidence."
    else:
        summary_text = "Blocked: review diagnosis codes and safety errors, then rerun the Beta check after fixing the issue."
    return {
        "schema": REVIEW_SUMMARY_SCHEMA,
        "state": "ready" if ready else "blocked",
        "ready": ready,
        "next_step": next_step,
        "inspect_first": artifacts.get("inspect_first"),
        "machine_readable": artifacts.get("machine_readable"),
        "support_bundle": artifacts.get("support_bundle"),
        "recommended_check_command": check_command,
        "recommended_next_command": next_command,
        "recommended_label": next_command.get("label") or "none",
        "recommended_reason": next_command.get("reason") or "none",
        "next_command": next_command.get("command_line") or "",
        "shareable_paths": artifacts.get("shareable_paths") or [],
        "not_completed_count": len(not_completed),
        "not_completed_preview": not_completed[:8],
        "operator_action_preview": operator_action[:3],
        "public_artifact_safe": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "summary": summary_text,
    }


def attach_user_guidance(report: dict[str, Any]) -> dict[str, Any]:
    report["artifact_summary"] = artifact_summary(report)
    report["recommended_check_command"] = recommended_check_command(report)
    report["recommended_next_command"] = recommended_next_command(report)
    report["next_commands"] = next_commands(report)
    report["user_status"] = user_status(report)
    report["review_summary"] = review_summary(report)
    report["inference_verdict"] = inference_verdict_summary(report)
    return report


def output_request_text(summary: dict[str, Any]) -> str:
    return (
        f"include_output={bool(summary.get('include_output'))} "
        f"raw_generated_text_public={bool(summary.get('raw_generated_text_public'))} "
        f"public_artifact_safe={bool(summary.get('public_artifact_safe'))}"
    )


def output_request_note(output_request: dict[str, Any]) -> str:
    return str(
        output_request.get("summary")
        or "Public artifacts summarize inference evidence only and do not include answer text."
    )


def prompt_scope_text(prompt_scope: dict[str, Any]) -> str:
    return (
        f"source={prompt_scope.get('source') or 'unknown'} "
        f"count={prompt_scope.get('prompt_count')} "
        f"inline_prompt_text={bool(prompt_scope.get('inline_prompt_text'))} "
        f"terminal_next_commands_local_private={bool(prompt_scope.get('terminal_next_commands_local_private'))} "
        f"saved_artifacts_prompt_placeholders={bool(prompt_scope.get('saved_artifacts_prompt_placeholders'))} "
        f"prompt_file_path_public={bool(prompt_scope.get('prompt_file_path_public'))} "
        f"raw_prompt_public={bool(prompt_scope.get('raw_prompt_public'))} "
        f"public_artifact_safe={bool(prompt_scope.get('public_artifact_safe'))}"
    )


def prompt_scope_note(prompt_scope: dict[str, Any]) -> str:
    return str(
        prompt_scope.get("summary")
        or "Public artifacts record prompt source/count only and exclude raw prompt text."
    )


def answer_scope_text(answer_scope: dict[str, Any]) -> str:
    return (
        f"state={answer_scope.get('scope_state') or 'unknown'} "
        f"terminal_only={bool(answer_scope.get('terminal_only'))} "
        f"visible_in_terminal={bool(answer_scope.get('visible_in_terminal'))} "
        f"saved_json={answer_scope.get('saved_json_display')} "
        f"saved_markdown={answer_scope.get('saved_markdown_display')} "
        f"public_artifact_safe={bool(answer_scope.get('public_artifact_safe'))}"
    )


def answer_scope_note(answer_scope: dict[str, Any]) -> str:
    return str(
        answer_scope.get("summary")
        or "Public artifacts contain no local answer transcript or raw generated text."
    )


def shareable_summary_text(summary: dict[str, Any]) -> str:
    return (
        f"saved_artifacts={bool(summary.get('saved_artifacts_public_safe'))} "
        f"raw_prompt_public={bool(summary.get('raw_prompt_public'))} "
        f"raw_generated_text_public={bool(summary.get('raw_generated_text_public'))} "
        f"generated_token_ids_public={bool(summary.get('generated_token_ids_public'))} "
        f"local_output_display_only={bool(summary.get('local_output_display_only'))} "
        f"answer_scope_state={summary.get('answer_scope_state') or 'unknown'} "
        f"local_answer_terminal_only={bool(summary.get('local_answer_terminal_only'))}"
    )


def limitations() -> list[str]:
    return [
        "Public Real-LLM Swarm Inference Beta v1 is Coordinator-backed and read-only.",
        "The default model path is tiny/small Hugging Face real weights; it is not large-model throughput serving.",
        "Real-P2P discovery is optional provider discovery; the Coordinator remains the session and result authority.",
        "CUDA is optional and fail-closed; a CPU-only host can validate the Beta without claiming GPU readiness.",
        "No payments, incentives, staking, anti-Sybil network, or production NAT/relay fabric are included.",
    ]


def local_model_variant_codes(*payloads: dict[str, Any]) -> set[str]:
    blocked_prefixes = (
        "external_",
        "kaggle_",
        "live_stage",
        "public_swarm_live",
        "real_llm_internet",
        "real_llm_live",
        "remote_",
        "swarm_inference_beta_live",
        "token_rotation_required",
    )
    blocked_codes = {
        "accepted_result_after_requeue",
        "cuda_runtime_available",
        "gpu_generation_evidence_import_ready",
        "gpu_multi_machine_generation_ready",
        "gpu_runtime_ready",
        "gpu_sharded_generation_ready",
        "gpu_stage0_ready",
        "gpu_stage1_ready",
        "hf_transformers_cuda_ready",
        "partition_parameter_split_valid",
        "public_swarm_gpu_beta_kaggle_auto_ready",
        "public_swarm_gpu_beta_ready",
        "public_swarm_live_kaggle_ready",
        "public_swarm_v2_external_stage_rows_ready",
        "real_p2p_core_rc_evidence_import_ready",
        "real_p2p_kaggle_auto_ready",
        "real_p2p_kaggle_private_artifacts_cleaned",
        "rescue_miner_used",
        "stage0_partition_loaded",
        "stage1_partition_loaded",
        "stage_gpu_memory_reduced",
        "stage_local_partition_ready",
        "swarm_inference_beta_live_ready",
    }
    codes: set[str] = set()
    for code in diagnosis_codes(*payloads):
        if code in blocked_codes:
            continue
        if any(code.startswith(prefix) for prefix in blocked_prefixes):
            continue
        codes.add(code)
    return codes


def build_package(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    runbook = write_runbook(args, output_dir)
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "mode": MODE_PACKAGE,
        "output_dir": str(output_dir),
        "beta": {
            "ready": True,
            "package_only": True,
            "user_surface": ["serve", "join", "generate"],
            "workload_type": WORKLOAD_TYPE,
            "hf_model_id": args.hf_model_id,
            "max_new_tokens": args.max_new_tokens,
            "batch": {
                "enabled": bool(args.prompt_texts or args.prompt_texts_file),
                "request_count": len(prompt_list_from_args(args)),
                "batch_generation_ready": False,
                "documented_only": True,
            },
        },
        "diagnosis_codes": [
            "public_real_llm_swarm_beta_package_ready",
            "serve_join_generate_runbook_ready",
            "cpu_default_path_documented",
            "optional_cuda_fail_closed_documented",
            "p2p_ready_product_beta_documented",
            "read_only_workload",
            "not_production",
            "not_coordinator_free",
            "not_large_model_serving",
        ],
        "artifacts": {"runbook": runbook},
        "safety": safety_block(),
        "prompt_scope": prompt_scope_summary(args),
        "operator_action": [
            "Run release mode to generate fresh local product evidence and aggregate retained external evidence.",
            "Use the runbook for ordinary two-stage CPU operator trials.",
        ],
        "limitations": limitations(),
    }
    return persist_report(report, output_dir=output_dir)


def build_aggregate(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    product_payload: dict[str, Any],
    p2p_payload: dict[str, Any] | None = None,
    p2p_report_path: Path | None = None,
    public_swarm_v2_payload: dict[str, Any] | None = None,
    public_swarm_v2_report_path: Path | None = None,
    usable_payload: dict[str, Any] | None = None,
    usable_report_path: Path | None = None,
    gpu_payload: dict[str, Any],
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    external_payload = load_json(args.external_report)
    if p2p_payload is None:
        p2p_payload = load_json(args.p2p_report)
    if p2p_report_path is None:
        p2p_report_path = Path(args.p2p_report)
    if usable_payload is None:
        usable_payload = load_json(args.usable_report) if getattr(args, "usable_report", "") else {}
    if usable_report_path is None:
        usable_report_path = Path(args.usable_report) if getattr(args, "usable_report", "") else None
    if public_swarm_v2_payload is None:
        public_swarm_v2_payload = load_json(args.public_swarm_v2_report) if getattr(args, "public_swarm_v2_report", "") else {}
    if public_swarm_v2_report_path is None:
        public_swarm_v2_report_path = Path(args.public_swarm_v2_report)
    imported_gpu_payload = load_json(args.gpu_report) if args.gpu_report else {}
    if not gpu_payload and imported_gpu_payload:
        gpu_payload = imported_gpu_payload
    runbook = write_runbook(args, output_dir)

    product = product_summary(product_payload, expected_model_id=args.hf_model_id)
    external = external_summary(
        external_payload,
        expected_model_id=args.hf_model_id,
        min_generated_tokens=args.max_new_tokens,
    )
    p2p = p2p_summary(
        p2p_payload,
        expected_model_id=args.hf_model_id,
        min_generated_tokens=args.max_new_tokens,
    )
    gpu = gpu_summary(gpu_payload)
    public_swarm_v2 = public_swarm_v2_summary(
        public_swarm_v2_payload,
        expected_model_id=args.hf_model_id,
        min_generated_tokens=args.max_new_tokens,
    )
    kv_cache = kv_cache_summary(
        usable_payload,
        min_generated_tokens=args.max_new_tokens,
        expected_model_id=args.hf_model_id,
    )

    release_ready = bool(
        product["ready"]
        and external["ready"]
        and p2p["ready"]
        and public_swarm_v2["ready"]
        and public_swarm_v2["real_p2p_local_route_hardening_ready"]
        and public_swarm_v2["real_p2p_local_stage_requeue_ready"]
        and kv_cache["ready"]
        and gpu["fail_closed_ready"]
        and runbook.get("present")
    )
    product_batch = product.get("batch") if isinstance(product.get("batch"), dict) else {}
    product_batch_ready = bool(product_batch.get("enabled") and product_batch.get("batch_generation_ready") is True)
    p2p_batch_ready = bool(p2p.get("batch_ready"))
    public_swarm_v2_batch_ready = bool(public_swarm_v2.get("batch_ready"))
    aggregate_stream_ready = bool(product.get("stream_ready") or p2p.get("stream_ready") or public_swarm_v2.get("stream_ready"))
    codes = drop_unproven_generation_ready_codes(
        set(diagnosis_codes(product_payload, external_payload, p2p_payload, public_swarm_v2_payload, usable_payload, gpu_payload)),
        product_batch_ready=product_batch_ready,
        p2p_batch_ready=p2p_batch_ready,
        public_swarm_v2_batch_ready=public_swarm_v2_batch_ready,
        stream_ready=aggregate_stream_ready,
    )
    codes = drop_superseded_product_path_blockers(codes, product_ready=product["ready"])
    if product["ready"]:
        codes.update({
            "user_facing_serve_join_generate_ready",
            "public_real_llm_product_path_ready",
            "cpu_real_llm_default_ready",
            "public_real_llm_swarm_beta_product_model_match_ready",
        })
        if product_batch_ready:
            codes.update({
                "public_real_llm_swarm_beta_batch_ready",
                "public_swarm_generate_batch_ready",
            })
        product_stream = product.get("stream") if isinstance(product.get("stream"), dict) else {}
        if product.get("stream_ready"):
            codes.update({
                "public_real_llm_swarm_beta_stream_ready",
                "public_swarm_generate_stream_ready",
            })
            if product_stream.get("endpoint_ready"):
                codes.add("public_swarm_generate_stream_endpoint_ready")
    if external["ready"]:
        codes.update({
            "external_kaggle_two_stage_ready",
            "external_real_llm_generate_ready",
            "external_stage_requeue_ready",
            "external_generated_token_target_ready",
        })
    if p2p["ready"]:
        codes.update({
            "p2p_ready_product_beta",
            "real_p2p_discovery_candidate_ready",
            "peer_scoring_ready",
            "p2p_live_requeue_rescue_ready",
            "p2p_victim_result_not_accepted",
            "p2p_generated_token_target_ready",
        })
        if p2p.get("batch_ready"):
            codes.update({
                "public_real_llm_swarm_beta_p2p_batch_ready",
                "public_swarm_generate_batch_ready",
            })
        if p2p.get("stream_ready"):
            codes.update({
                "public_real_llm_swarm_beta_p2p_stream_ready",
                "public_swarm_generate_stream_ready",
            })
            stream = p2p.get("stream") if isinstance(p2p.get("stream"), dict) else {}
            if stream.get("endpoint_ready"):
                codes.add("public_swarm_generate_stream_endpoint_ready")
    if public_swarm_v2["ready"]:
        codes.update({
            "public_real_llm_swarm_beta_public_swarm_v2_ready",
            "public_real_llm_swarm_beta_p2p_user_path_ready",
            "public_swarm_inference_v2_ready",
            "public_swarm_v2_local_p2p_generate_ready",
            "public_swarm_v2_16_token_generation_ready",
            "public_swarm_v2_external_stage_rows_ready",
            "public_swarm_v2_dual_stage_kv_cache_ready",
            "public_swarm_v2_model_match_ready",
            "public_swarm_v2_signed_or_real_p2p_ready",
            "public_swarm_v2_stage_requeue_rescue_ready",
        })
        if public_swarm_v2.get("real_p2p_local_route_hardening_ready"):
            codes.update({
                "public_real_llm_swarm_beta_v2_real_p2p_local_ready",
                "public_swarm_v2_real_p2p_local_ready",
            })
        if public_swarm_v2.get("real_p2p_local_stage_requeue_ready"):
            codes.update({
                "public_real_llm_swarm_beta_v2_real_p2p_local_requeue_ready",
                "public_swarm_v2_real_p2p_local_requeue_ready",
            })
        if public_swarm_v2.get("batch_ready"):
            codes.update({
                "public_real_llm_swarm_beta_v2_batch_ready",
                "public_swarm_v2_batch_generation_ready",
                "public_swarm_generate_batch_ready",
            })
        if public_swarm_v2.get("stream_ready"):
            codes.update({
                "public_real_llm_swarm_beta_v2_stream_ready",
                "public_swarm_v2_stream_generation_ready",
                "public_swarm_generate_stream_ready",
            })
    if kv_cache["ready"]:
        codes.update({
            "public_real_llm_swarm_beta_kv_cache_ready",
            "public_real_llm_swarm_beta_kv_cache_model_match_ready",
            "usable_real_llm_kv_cache_ready",
            "p2p_real_generate_kv_cache_ready",
            "real_llm_stage0_kv_cache_v1_ready",
            "real_llm_stage1_kv_cache_v1_ready",
            "stage0_kv_cache_hits_ready",
            "stage1_kv_cache_hits_ready",
        })
    if gpu["fail_closed_ready"]:
        codes.add("optional_cuda_fail_closed_ready")
    if runbook.get("present"):
        codes.add("serve_join_generate_runbook_ready")
    codes.update({
        "read_only_workload",
        "not_production",
        "not_coordinator_free",
        "not_hivemind_petals_production",
        "not_large_model_serving",
    })
    if release_ready:
        codes.update({
            "public_real_llm_swarm_beta_ready",
            "release_evidence_ready",
        })
    else:
        codes.add("public_real_llm_swarm_beta_blocked")
        if not product.get("model", {}).get("compatible"):
            codes.add("product_model_mismatch")
        if not external.get("model", {}).get("compatible"):
            codes.add("external_model_mismatch")
        if not p2p.get("model", {}).get("compatible"):
            codes.add("p2p_model_mismatch")
        if not public_swarm_v2.get("model", {}).get("compatible"):
            codes.add("public_swarm_v2_model_mismatch")
        if not external.get("token_target_ready"):
            codes.add("external_generated_token_target_missing")
        if not p2p.get("token_target_ready"):
            codes.add("p2p_generated_token_target_missing")
        if not public_swarm_v2.get("generation_target_ready") or not public_swarm_v2.get("external_generation_target_ready"):
            codes.add("public_swarm_v2_token_target_missing")
        if not public_swarm_v2.get("accepted_rows_ready") or not public_swarm_v2.get("external_accepted_rows_ready"):
            codes.add("public_swarm_v2_stage_rows_missing")
        if not public_swarm_v2.get("batch_ready"):
            codes.add("public_swarm_v2_batch_generation_missing")
        if not public_swarm_v2.get("stream_ready"):
            codes.add("public_swarm_v2_stream_generation_missing")
        if not public_swarm_v2.get("real_p2p_local_route_hardening_ready"):
            codes.add("public_swarm_v2_real_p2p_local_missing")
        if not public_swarm_v2.get("real_p2p_local_stage_requeue_ready"):
            codes.add("public_swarm_v2_real_p2p_local_requeue_missing")
        if not kv_cache.get("cache_ready"):
            codes.add("public_real_llm_swarm_beta_kv_cache_missing")
        if not kv_cache.get("model", {}).get("compatible"):
            codes.add("kv_cache_model_mismatch")

    operator_actions = [
        "Use `crowdtensor serve`, `crowdtensor join --stage stage0`, `crowdtensor join --stage stage1`, and `crowdtensor generate` as the primary user path.",
        "Release mode also fresh-runs the Petals-class P2P candidate local-smoke and the local Public Swarm v2 `p2pd` / `serve --p2p` / `join --p2p` / `generate --p2p` proof; evidence-import mode uses `--p2p-report` and `--public-swarm-v2-report`.",
        "Share this top-level JSON/Markdown artifact; raw prompts, generated text, token ids, activations, and credentials are excluded.",
        "Rotate tokens after temporary public HTTP/Kaggle proofs.",
    ]
    check_preview = recommended_check_command({
        "mode": args.mode,
        "output_dir": str(output_dir),
        "beta": {"hf_model_id": args.hf_model_id, "max_new_tokens": args.max_new_tokens},
    })
    if check_preview:
        operator_actions.insert(0, f"Validate this report before sharing: {check_preview.get('command_line')}")

    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": release_ready,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "beta": {
            "ready": release_ready,
            "workload_type": WORKLOAD_TYPE,
            "hf_model_id": args.hf_model_id,
            "max_new_tokens": args.max_new_tokens,
            "batch": product.get("batch") or {"enabled": False, "batch_generation_ready": False},
            "stream": product.get("stream") or {"enabled": False, "stream_generation_ready": False},
            "user_surface": ["serve", "join", "generate"],
            "cpu_default_ready": product["ready"],
            "cuda_optional_fail_closed_ready": gpu["fail_closed_ready"],
            "external_two_stage_ready": external["ready"],
            "external_stage_requeue_ready": external["stage_requeue_ready"],
            "p2p_ready_product_beta": p2p["ready"],
            "p2p_live_requeue_ready": p2p["p2p_live_requeue_ready"],
            "p2p_victim_result_not_accepted": p2p["victim_result_not_accepted"],
            "p2p_batch_ready": p2p.get("batch_ready"),
            "p2p_stream_ready": p2p.get("stream_ready"),
            "public_swarm_v2_ready": public_swarm_v2.get("ready"),
            "public_swarm_v2_batch_ready": public_swarm_v2.get("batch_ready"),
            "public_swarm_v2_stream_ready": public_swarm_v2.get("stream_ready"),
            "public_swarm_v2_real_p2p_local_ready": public_swarm_v2.get("real_p2p_local_route_hardening_ready"),
            "public_swarm_v2_real_p2p_local_requeue_ready": public_swarm_v2.get("real_p2p_local_stage_requeue_ready"),
            "kv_cache_ready": kv_cache.get("ready"),
            "release_evidence_ready": release_ready,
        },
        "readiness": {
            "product_path": product,
            "external_kaggle": external,
            "p2p_candidate": p2p,
            "public_swarm_v2": public_swarm_v2,
            "usable_p2p_kv_cache": kv_cache,
            "cuda_optional": gpu,
        },
        "steps": steps,
        "diagnosis_codes": sorted(codes),
        "artifacts": {
            "runbook": runbook,
            "product_report": artifact_entry(
                Path(args.product_report) if args.mode == MODE_EVIDENCE_IMPORT else output_dir / "product-beta" / "public_swarm_product_beta.json",
                output_dir,
                kind="public_swarm_product_beta",
                schema=PRODUCT_SCHEMA,
                ok=product_payload.get("ok") if product_payload else None,
            ),
            "external_real_llm_report": artifact_entry(
                Path(args.external_report),
                output_dir,
                kind="real_llm_internet_beta",
                schema=REAL_LLM_INTERNET_BETA_SCHEMA,
                ok=external_payload.get("ok") if external_payload else None,
            ),
            "p2p_candidate_report": artifact_entry(
                p2p_report_path,
                output_dir,
                kind="petals_class_p2p_candidate",
                schema=P2P_SCHEMA,
                ok=p2p_payload.get("ok") if p2p_payload else None,
            ),
            "usable_swarm_kv_cache_report": artifact_entry(
                usable_report_path,
                output_dir,
                kind="usable_swarm_inference_kv_cache",
                schema="usable_swarm_inference_v1",
                ok=usable_payload.get("ok") if usable_payload else None,
            ),
            "public_swarm_v2_report": artifact_entry(
                public_swarm_v2_report_path,
                output_dir,
                kind="public_swarm_inference_v2",
                schema=PUBLIC_SWARM_V2_SCHEMA,
                ok=public_swarm_v2_payload.get("ok") if public_swarm_v2_payload else None,
            ),
            "gpu_optional_report": artifact_entry(
                Path(args.gpu_report) if args.mode == MODE_EVIDENCE_IMPORT and args.gpu_report else output_dir / "gpu-smoke" / "public_swarm_gpu_inference_beta_local_smoke.json",
                output_dir,
                kind="public_swarm_gpu_inference_beta",
                schema=GPU_SCHEMA,
                ok=gpu_payload.get("ok") if gpu_payload else None,
            ),
        },
        "source_reports": {
            "product_report": str(Path(args.product_report).resolve()) if args.mode == MODE_EVIDENCE_IMPORT else str((output_dir / "product-beta" / "public_swarm_product_beta.json").resolve()),
            "external_report": str(Path(args.external_report).resolve()),
            "p2p_report": str(p2p_report_path.resolve()) if p2p_report_path else "",
            "public_swarm_v2_report": str(public_swarm_v2_report_path.resolve()) if public_swarm_v2_report_path else "",
            "usable_report": str(usable_report_path.resolve()) if usable_report_path else "",
            "gpu_report": str(Path(args.gpu_report).resolve()) if args.gpu_report else "",
        },
        "safety": safety_block(),
        "prompt_scope": prompt_scope_summary(args),
        "output_request": output_request_summary(),
        "answer_scope": answer_scope_summary(),
        "shareable_summary": shareable_summary(),
        "operator_action": operator_actions,
        "limitations": limitations(),
        "not_completed": [] if release_ready else [
            item for item, ok in [
                ("local serve/join/generate product path", product["path_ready"]),
                ("local product evidence model match", (product.get("model") or {}).get("compatible")),
                ("external Kaggle two-stage real LLM proof", external["ready"]),
                ("external generated token target", external["token_target_ready"]),
                ("external stage requeue/rescue proof", external["stage_requeue_ready"]),
                ("external evidence model match", (external.get("model") or {}).get("compatible")),
                ("real-P2P discovery candidate with live requeue rescue", p2p["ready"]),
                ("real-P2P generated token target", p2p["token_target_ready"]),
                ("real-P2P evidence model match", (p2p.get("model") or {}).get("compatible")),
                ("Public Swarm v2 ordinary P2P user path", public_swarm_v2["ready"]),
                ("Public Swarm v2 generated token target", public_swarm_v2["generation_target_ready"] and public_swarm_v2["external_generation_target_ready"]),
                ("Public Swarm v2 accepted stage rows", public_swarm_v2["accepted_rows_ready"] and public_swarm_v2["external_accepted_rows_ready"]),
                ("Public Swarm v2 evidence model match", (public_swarm_v2.get("model") or {}).get("compatible")),
                ("Public Swarm v2 batch generation", public_swarm_v2["batch_ready"]),
                ("Public Swarm v2 stream generation", public_swarm_v2["stream_ready"]),
                ("Public Swarm v2 dual-stage KV-cache reuse", public_swarm_v2["kv_cache_ready"]),
                ("Public Swarm v2 stage requeue/rescue", public_swarm_v2["stage_requeue_rescue_ready"]),
                ("Public Swarm v2 fresh real-P2P local route hardening", public_swarm_v2["real_p2p_local_route_hardening_ready"]),
                ("Public Swarm v2 fresh real-P2P local stage requeue", public_swarm_v2["real_p2p_local_stage_requeue_ready"]),
                ("persistent dual-stage KV-cache reuse", kv_cache["cache_ready"]),
                ("KV-cache evidence model match", (kv_cache.get("model") or {}).get("compatible")),
                ("optional CUDA fail-closed smoke/import", gpu["fail_closed_ready"]),
                ("runbook", bool(runbook.get("present"))),
            ]
            if not ok
        ],
    }
    return persist_report(report, output_dir=output_dir)


def build_release(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    product_step, product_payload = run_product_local(args, output_dir=output_dir / "product-beta", runner=runner)
    p2p_step, p2p_payload = run_p2p_candidate_local(args, output_dir=output_dir / "p2p-candidate", runner=runner)
    public_swarm_v2_step, public_swarm_v2_payload = run_public_swarm_v2_local(args, output_dir=output_dir / "public-swarm-v2", runner=runner)
    gpu_step, gpu_payload = run_gpu_smoke(args, output_dir=output_dir / "gpu-smoke", runner=runner)
    fresh_usable_report_path = output_dir / "public-swarm-v2" / "usable-v1-local" / "usable_swarm_inference.json"
    usable_payload = load_json(fresh_usable_report_path) or public_swarm_v2_payload
    fresh_p2p_report_path = output_dir / "p2p-candidate" / "petals_class_p2p_candidate.json"
    p2p_payload = load_json(fresh_p2p_report_path) or p2p_payload
    return build_aggregate(
        args,
        output_dir=output_dir,
        product_payload=product_payload,
        p2p_payload=p2p_payload,
        p2p_report_path=fresh_p2p_report_path if fresh_p2p_report_path.is_file() else Path(args.p2p_report),
        public_swarm_v2_payload=public_swarm_v2_payload,
        public_swarm_v2_report_path=output_dir / "public-swarm-v2" / "public_swarm_inference_v2.json",
        usable_payload=usable_payload,
        usable_report_path=fresh_usable_report_path if fresh_usable_report_path.is_file() else output_dir / "public-swarm-v2" / "public_swarm_inference_v2.json",
        gpu_payload=gpu_payload,
        steps=[product_step, p2p_step, public_swarm_v2_step, gpu_step],
    )


def build_local_smoke(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    product_step, product_payload = run_product_local(args, output_dir=output_dir / "product-beta", runner=runner)
    gpu_step, gpu_payload = run_gpu_smoke(args, output_dir=output_dir / "gpu-smoke", runner=runner)
    runbook = write_runbook(args, output_dir)
    product = product_summary(product_payload, expected_model_id=args.hf_model_id)
    gpu = gpu_summary(gpu_payload)
    product_batch = product.get("batch") if isinstance(product.get("batch"), dict) else {}
    product_stream = product.get("stream") if isinstance(product.get("stream"), dict) else {}
    ready = bool(product["ready"] and gpu["fail_closed_ready"] and runbook.get("present"))
    product_batch_ready = bool(product_batch.get("enabled") and product_batch.get("batch_generation_ready") is True)
    codes = drop_unproven_generation_ready_codes(
        set(diagnosis_codes(product_payload, gpu_payload)),
        product_batch_ready=product_batch_ready,
        stream_ready=product.get("stream_ready") is True,
    )
    if product["ready"]:
        codes.update({"user_facing_serve_join_generate_ready", "public_real_llm_product_path_ready", "cpu_real_llm_default_ready"})
        if product_batch_ready:
            codes.update({"public_real_llm_swarm_beta_batch_ready", "public_swarm_generate_batch_ready"})
        if product.get("stream_ready"):
            codes.update({"public_real_llm_swarm_beta_stream_ready", "public_swarm_generate_stream_ready"})
            if product_stream.get("endpoint_ready"):
                codes.add("public_swarm_generate_stream_endpoint_ready")
    if gpu["fail_closed_ready"]:
        codes.add("optional_cuda_fail_closed_ready")
    if ready:
        codes.update({"public_real_llm_swarm_beta_local_smoke_ready", "read_only_workload", "not_production", "not_coordinator_free", "not_large_model_serving"})
    else:
        codes.add("public_real_llm_swarm_beta_local_smoke_blocked")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": MODE_LOCAL_SMOKE,
        "output_dir": str(output_dir),
        "beta": {
            "ready": ready,
            "local_smoke_only": True,
            "workload_type": WORKLOAD_TYPE,
            "hf_model_id": args.hf_model_id,
            "max_new_tokens": args.max_new_tokens,
            "user_surface": ["serve", "join", "generate"],
            "batch": product_batch if product_batch else {"enabled": False, "batch_generation_ready": False},
            "stream": product_stream if product_stream else {"enabled": False, "stream_generation_ready": False},
            "cpu_default_ready": product["ready"],
            "cuda_optional_fail_closed_ready": gpu["fail_closed_ready"],
        },
        "readiness": {"product_path": product, "cuda_optional": gpu},
        "steps": [product_step, gpu_step],
        "diagnosis_codes": sorted(codes),
        "artifacts": {
            "runbook": runbook,
            "product_report": artifact_entry(output_dir / "product-beta" / "public_swarm_product_beta.json", output_dir, kind="public_swarm_product_beta", schema=PRODUCT_SCHEMA, ok=product_payload.get("ok") if product_payload else None),
            "gpu_optional_report": artifact_entry(output_dir / "gpu-smoke" / "public_swarm_gpu_inference_beta_local_smoke.json", output_dir, kind="public_swarm_gpu_inference_beta", schema=GPU_SCHEMA, ok=gpu_payload.get("ok") if gpu_payload else None),
        },
        "safety": safety_block(),
        "prompt_scope": prompt_scope_summary(args),
        "operator_action": ["Run release or evidence-import mode to include retained external Kaggle and real-P2P evidence."],
        "limitations": limitations(),
    }
    return persist_report(report, output_dir=output_dir)


def build_local_model_variant(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    product_step, product_payload = run_product_local(args, output_dir=output_dir / "product-beta", runner=runner)
    public_swarm_v2_step, public_swarm_v2_payload = run_public_swarm_v2_local(
        args,
        output_dir=output_dir / "public-swarm-v2",
        runner=runner,
    )
    gpu_step, gpu_payload = run_gpu_smoke(args, output_dir=output_dir / "gpu-smoke", runner=runner)
    fresh_usable_report_path = output_dir / "public-swarm-v2" / "usable-v1-local" / "usable_swarm_inference.json"
    usable_payload = load_json(fresh_usable_report_path) or public_swarm_v2_payload
    product = product_summary(product_payload, expected_model_id=args.hf_model_id)
    public_swarm_v2 = public_swarm_v2_summary(
        public_swarm_v2_payload,
        expected_model_id=args.hf_model_id,
        min_generated_tokens=args.max_new_tokens,
        local_model_variant=True,
    )
    kv_cache = kv_cache_summary(
        usable_payload,
        min_generated_tokens=args.max_new_tokens,
        expected_model_id=args.hf_model_id,
    )
    gpu = gpu_summary(gpu_payload)
    runbook = write_runbook(args, output_dir)
    ready = bool(
        product["ready"]
        and public_swarm_v2["ready"]
        and public_swarm_v2["real_p2p_local_route_hardening_ready"]
        and public_swarm_v2["real_p2p_local_stage_requeue_ready"]
        and kv_cache["ready"]
        and gpu["fail_closed_ready"]
        and runbook.get("present")
    )
    product_batch = product.get("batch") if isinstance(product.get("batch"), dict) else {}
    product_batch_ready = bool(product_batch.get("enabled") and product_batch.get("batch_generation_ready") is True)
    codes = drop_unproven_generation_ready_codes(
        local_model_variant_codes(product_payload, public_swarm_v2_payload, usable_payload, gpu_payload),
        product_batch_ready=product_batch_ready,
        public_swarm_v2_batch_ready=bool(public_swarm_v2.get("batch_ready")),
        stream_ready=bool(product.get("stream_ready") or public_swarm_v2.get("stream_ready")),
    )
    codes.update({
        "read_only_workload",
        "not_production",
        "not_coordinator_free",
        "not_hivemind_petals_production",
        "not_large_model_serving",
        "public_real_llm_swarm_beta_local_model_variant_only",
        "external_validation_not_claimed",
    })
    if product["ready"]:
        codes.update({
            "user_facing_serve_join_generate_ready",
            "public_real_llm_product_path_ready",
            "public_real_llm_swarm_beta_product_model_match_ready",
        })
    if public_swarm_v2["ready"]:
        codes.update({
            "public_real_llm_swarm_beta_local_model_variant_v2_ready",
            "public_swarm_inference_v2_local_model_variant_ready",
            "public_swarm_v2_local_model_variant_ready",
            "public_swarm_v2_local_p2p_generate_ready",
            "public_swarm_v2_16_token_generation_ready",
            "public_swarm_v2_dual_stage_kv_cache_ready",
            "public_swarm_v2_model_match_ready",
            "public_swarm_v2_local_model_variant_model_match_ready",
            "public_swarm_v2_signed_or_real_p2p_ready",
            "public_swarm_v2_stage_requeue_rescue_ready",
        })
        if public_swarm_v2.get("real_p2p_local_route_hardening_ready"):
            codes.update({
                "public_real_llm_swarm_beta_v2_real_p2p_local_ready",
                "public_swarm_v2_real_p2p_local_ready",
            })
        if public_swarm_v2.get("real_p2p_local_stage_requeue_ready"):
            codes.update({
                "public_real_llm_swarm_beta_v2_real_p2p_local_requeue_ready",
                "public_swarm_v2_real_p2p_local_requeue_ready",
            })
        if public_swarm_v2.get("batch_ready"):
            codes.update({
                "public_real_llm_swarm_beta_v2_batch_ready",
                "public_swarm_v2_batch_generation_ready",
                "public_swarm_generate_batch_ready",
            })
        if public_swarm_v2.get("stream_ready"):
            codes.update({
                "public_real_llm_swarm_beta_v2_stream_ready",
                "public_swarm_v2_stream_generation_ready",
                "public_swarm_generate_stream_ready",
            })
    if kv_cache["ready"]:
        codes.update({
            "public_real_llm_swarm_beta_kv_cache_ready",
            "public_real_llm_swarm_beta_kv_cache_model_match_ready",
            "usable_real_llm_kv_cache_ready",
            "p2p_real_generate_kv_cache_ready",
            "real_llm_stage0_kv_cache_v1_ready",
            "real_llm_stage1_kv_cache_v1_ready",
            "stage0_kv_cache_hits_ready",
            "stage1_kv_cache_hits_ready",
        })
    if gpu["fail_closed_ready"]:
        codes.add("optional_cuda_fail_closed_ready")
    if runbook.get("present"):
        codes.add("serve_join_generate_runbook_ready")
    if ready:
        codes.add("public_real_llm_swarm_beta_local_model_variant_ready")
    else:
        codes.add("public_real_llm_swarm_beta_local_model_variant_blocked")
        if not public_swarm_v2.get("real_p2p_local_route_hardening_ready"):
            codes.add("public_swarm_v2_real_p2p_local_missing")
        if not public_swarm_v2.get("real_p2p_local_stage_requeue_ready"):
            codes.add("public_swarm_v2_real_p2p_local_requeue_missing")
    product_batch = product.get("batch") if isinstance(product.get("batch"), dict) else {}
    product_stream = product.get("stream") if isinstance(product.get("stream"), dict) else {}
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": MODE_LOCAL_MODEL_VARIANT,
        "output_dir": str(output_dir),
        "beta": {
            "ready": ready,
            "local_model_variant_only": True,
            "release_evidence_ready": False,
            "external_two_stage_ready": False,
            "external_stage_requeue_ready": False,
            "p2p_ready_product_beta": False,
            "workload_type": WORKLOAD_TYPE,
            "hf_model_id": args.hf_model_id,
            "max_new_tokens": args.max_new_tokens,
            "user_surface": ["serve", "join", "generate"],
            "batch": product_batch if product_batch else {"enabled": False, "batch_generation_ready": False},
            "stream": product_stream if product_stream else {"enabled": False, "stream_generation_ready": False},
            "cpu_default_ready": product["ready"],
            "public_swarm_v2_ready": public_swarm_v2["ready"],
            "public_swarm_v2_local_model_variant_ready": public_swarm_v2["ready"],
            "public_swarm_v2_batch_ready": public_swarm_v2.get("batch_ready"),
            "public_swarm_v2_stream_ready": public_swarm_v2.get("stream_ready"),
            "public_swarm_v2_real_p2p_local_ready": public_swarm_v2.get("real_p2p_local_route_hardening_ready"),
            "public_swarm_v2_real_p2p_local_requeue_ready": public_swarm_v2.get("real_p2p_local_stage_requeue_ready"),
            "kv_cache_ready": kv_cache.get("ready"),
            "cuda_optional_fail_closed_ready": gpu["fail_closed_ready"],
        },
        "readiness": {
            "product_path": product,
            "public_swarm_v2": public_swarm_v2,
            "usable_p2p_kv_cache": kv_cache,
            "cuda_optional": gpu,
            "external_kaggle": {
                "ready": False,
                "claimed": False,
                "reason": "local_model_variant_mode",
            },
            "p2p_candidate": {
                "ready": False,
                "claimed": False,
                "reason": "local_model_variant_mode",
            },
        },
        "steps": [product_step, public_swarm_v2_step, gpu_step],
        "diagnosis_codes": sorted(codes),
        "artifacts": {
            "runbook": runbook,
            "product_report": artifact_entry(output_dir / "product-beta" / "public_swarm_product_beta.json", output_dir, kind="public_swarm_product_beta", schema=PRODUCT_SCHEMA, ok=product_payload.get("ok") if product_payload else None),
            "public_swarm_v2_report": artifact_entry(output_dir / "public-swarm-v2" / "public_swarm_inference_v2.json", output_dir, kind="public_swarm_inference_v2", schema=PUBLIC_SWARM_V2_SCHEMA, ok=public_swarm_v2_payload.get("ok") if public_swarm_v2_payload else None),
            "usable_swarm_kv_cache_report": artifact_entry(fresh_usable_report_path, output_dir, kind="usable_swarm_inference_kv_cache", schema="usable_swarm_inference_v1", ok=usable_payload.get("ok") if usable_payload else None),
            "gpu_optional_report": artifact_entry(output_dir / "gpu-smoke" / "public_swarm_gpu_inference_beta_local_smoke.json", output_dir, kind="public_swarm_gpu_inference_beta", schema=GPU_SCHEMA, ok=gpu_payload.get("ok") if gpu_payload else None),
        },
        "source_reports": {
            "product_report": str((output_dir / "product-beta" / "public_swarm_product_beta.json").resolve()),
            "public_swarm_v2_report": str((output_dir / "public-swarm-v2" / "public_swarm_inference_v2.json").resolve()),
            "usable_report": str(fresh_usable_report_path.resolve()),
            "gpu_report": str((output_dir / "gpu-smoke" / "public_swarm_gpu_inference_beta_local_smoke.json").resolve()),
        },
        "safety": safety_block(),
        "prompt_scope": prompt_scope_summary(args),
        "operator_action": [
            f"Validate this local model variant report before sharing: {recommended_check_command({'mode': MODE_LOCAL_MODEL_VARIANT, 'output_dir': str(output_dir), 'beta': {'hf_model_id': args.hf_model_id, 'max_new_tokens': args.max_new_tokens}}).get('command_line')}",
            "Use release or evidence-import mode for retained external Kaggle and Petals-class P2P candidate claims.",
            "Use this mode only to prove a non-default small Hugging Face model across the local Coordinator-backed product, v2 P2P, real-P2P local requeue, and KV-cache paths.",
        ],
        "limitations": limitations(),
        "not_completed": [] if ready else [
            item for item, ok in [
                ("local serve/join/generate product path", product["ready"]),
                ("local product evidence model match", (product.get("model") or {}).get("compatible")),
                ("Public Swarm v2 local model variant path", public_swarm_v2["ready"]),
                ("Public Swarm v2 fresh real-P2P local route hardening", public_swarm_v2["real_p2p_local_route_hardening_ready"]),
                ("Public Swarm v2 fresh real-P2P local stage requeue", public_swarm_v2["real_p2p_local_stage_requeue_ready"]),
                ("persistent dual-stage KV-cache reuse", kv_cache["cache_ready"]),
                ("KV-cache evidence model match", (kv_cache.get("model") or {}).get("compatible")),
                ("optional CUDA fail-closed smoke/import", gpu["fail_closed_ready"]),
                ("runbook", bool(runbook.get("present"))),
            ]
            if not ok
        ],
    }
    return persist_report(report, output_dir=output_dir)


def build_evidence_import(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    product_payload = load_json(args.product_report)
    gpu_payload = load_json(args.gpu_report) if args.gpu_report else {}
    return build_aggregate(args, output_dir=output_dir, product_payload=product_payload, gpu_payload=gpu_payload, steps=[])


def render_markdown(report: dict[str, Any]) -> str:
    beta = report.get("beta") if isinstance(report.get("beta"), dict) else {}
    readiness = report.get("readiness") if isinstance(report.get("readiness"), dict) else {}
    review = report.get("review_summary") if isinstance(report.get("review_summary"), dict) else {}
    artifact_overview = report.get("artifact_summary") if isinstance(report.get("artifact_summary"), dict) else {}
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    provenance = report.get("runtime_provenance") if isinstance(report.get("runtime_provenance"), dict) else {}
    evidence_scope = report.get("evidence_scope") if isinstance(report.get("evidence_scope"), dict) else {}
    verdict = report.get("inference_verdict") if isinstance(report.get("inference_verdict"), dict) else {}
    user = report.get("user_status") if isinstance(report.get("user_status"), dict) else {}
    shareable_paths = artifact_overview.get("shareable_paths") if isinstance(artifact_overview.get("shareable_paths"), list) else []
    recommended = (
        report.get("recommended_next_command")
        if isinstance(report.get("recommended_next_command"), dict)
        else review.get("recommended_next_command") if isinstance(review.get("recommended_next_command"), dict) else {}
    )
    recommended_check = (
        report.get("recommended_check_command")
        if isinstance(report.get("recommended_check_command"), dict)
        else review.get("recommended_check_command") if isinstance(review.get("recommended_check_command"), dict) else {}
    )
    raw_operator_actions = report.get("operator_action")
    if isinstance(raw_operator_actions, list):
        operator_actions = [str(item) for item in raw_operator_actions]
    elif raw_operator_actions:
        operator_actions = [str(raw_operator_actions)]
    else:
        operator_actions = []
    lines = [
        "# CrowdTensor Public Real-LLM Swarm Inference Beta v1",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- ready: `{beta.get('ready')}`",
        f"- workload: `{beta.get('workload_type')}`",
        f"- model: `{beta.get('hf_model_id')}`",
        f"- output_dir: `{report.get('output_dir')}`",
        "",
        "## Review",
        "",
        f"- status: `{user.get('state')}` {user.get('headline') or ''}",
        f"- state: `{review.get('state')}`",
        f"- next step: `{review.get('next_step')}`",
        f"- inspect first: `{review.get('inspect_first')}`",
        f"- support bundle: `{review.get('support_bundle')}`",
        f"- not completed count: `{review.get('not_completed_count')}`",
        f"- recommended next: `{recommended.get('command_line') or 'none'}`",
        f"- recommended check: `{recommended_check.get('command_line') or 'none'}`",
        f"- verdict: `{inference_verdict_text(verdict)}`",
        f"- verdict note: {verdict.get('message') or ''}",
        "",
        "## What To Do Next",
        "",
    ]
    next_items = report.get("next_commands") if isinstance(report.get("next_commands"), list) else []
    if next_items:
        lines.extend(
            f"- {item.get('label')}: `{item.get('command_line')}`"
            for item in next_items
            if isinstance(item, dict)
        )
    else:
        lines.append("- none")
    gpu_status = report.get("gpu_status") if isinstance(report.get("gpu_status"), dict) else {}
    gpu_proof_next_step = (
        report.get("gpu_proof_next_step")
        if isinstance(report.get("gpu_proof_next_step"), dict)
        else gpu_proof_next_step_summary(gpu_status or gpu_status_summary(provenance, evidence_scope))
    )
    lines.extend([
        "",
        "## Runtime Provenance",
        "",
        f"- proof: `{provenance.get('proof_level')}`",
        f"- local CPU product path ready: `{provenance.get('local_cpu_product_path_ready')}`",
        f"- external Kaggle CPU evidence ready: `{provenance.get('external_kaggle_cpu_evidence_ready')}`",
        f"- P2P candidate evidence ready: `{provenance.get('p2p_candidate_evidence_ready')}`",
        f"- GPU report mode: `{provenance.get('gpu_report_mode')}`",
        f"- local GPU smoke ran: `{provenance.get('local_gpu_smoke_ran')}`",
        f"- retained GPU evidence imported: `{provenance.get('retained_gpu_evidence_imported')}`",
        f"- fresh Kaggle GPU attempted: `{provenance.get('fresh_kaggle_gpu_attempted')}`",
        f"- fresh Kaggle GPU verified: `{provenance.get('fresh_kaggle_gpu_verified')}`",
        f"- GPU status: `{gpu_status_text(gpu_status or gpu_status_summary(provenance, evidence_scope))}`",
        f"- GPU proof next: `{gpu_proof_next_step_text(gpu_proof_next_step)}`",
        f"- note: {provenance.get('summary') or ''}",
        "",
        "## Evidence Scope",
        "",
        f"- level: `{evidence_scope.get('level')}`",
        f"- executed: `{evidence_scope.get('executed_where')}`",
        f"- local CPU: `{evidence_scope.get('local_cpu')}`",
        f"- retained external evidence: `{evidence_scope.get('retained_external_evidence_imported')}`",
        f"- local GPU smoke ran: `{evidence_scope.get('local_gpu_smoke_ran')}`",
        f"- retained GPU evidence imported: `{evidence_scope.get('retained_gpu_evidence_imported')}`",
        f"- fresh Kaggle GPU verified: `{evidence_scope.get('fresh_kaggle_gpu_verified')}`",
        f"- note: {evidence_scope.get('user_expectation') or ''}",
        "",
        "## Operator Action",
        "",
    ])
    lines.extend(f"- {item}" for item in operator_actions) if operator_actions else lines.append("- none")
    lines.extend([
        "",
        "## Readiness",
        "",
    ])
    for name in ["product_path", "external_kaggle", "p2p_candidate", "public_swarm_v2", "usable_p2p_kv_cache", "cuda_optional"]:
        item = readiness.get(name) if isinstance(readiness.get(name), dict) else {}
        if item:
            lines.append(f"- {name}: ready=`{item.get('ready', item.get('fail_closed_ready'))}` schema=`{item.get('schema')}`")
    lines.extend([
        "",
        "## Output Scope",
        "",
        f"- include output: `{output_request.get('include_output')}`",
        f"- output request note: {output_request_note(output_request)}",
        f"- prompt scope: `{prompt_scope_text(prompt_scope)}`",
        f"- prompt scope note: {prompt_scope_note(prompt_scope)}",
        f"- answer scope: `{answer_scope.get('scope_state')}`",
        f"- answer scope note: {answer_scope_note(answer_scope)}",
        f"- saved JSON display: `{answer_scope.get('saved_json_display')}`",
        f"- saved Markdown display: `{answer_scope.get('saved_markdown_display')}`",
        f"- shareable: `saved_artifacts={shareable.get('saved_artifacts_public_safe')} raw_prompt_public={shareable.get('raw_prompt_public')} raw_generated_text_public={shareable.get('raw_generated_text_public')} generated_token_ids_public={shareable.get('generated_token_ids_public')} answer_scope_state={shareable.get('answer_scope_state')} local_answer_terminal_only={shareable.get('local_answer_terminal_only')}`",
        "",
        "## GPU Proof Next",
        "",
        f"- state: `{gpu_proof_next_step.get('state')}`",
        f"- note: {gpu_proof_next_step.get('note') or ''}",
        f"- requires Kaggle: `{gpu_proof_next_step.get('requires_kaggle')}`",
        f"- side effectful: `{gpu_proof_next_step.get('side_effectful')}`",
        f"- cleanup required: `{gpu_proof_next_step.get('cleanup_required')}`",
        f"- token rotation required: `{gpu_proof_next_step.get('token_rotation_required')}`",
    ])
    for item in gpu_proof_next_step.get("commands") or []:
        if isinstance(item, dict):
            lines.append(
                "- GPU proof command: "
                f"reason=`{item.get('reason')}` "
                f"requires_kaggle=`{bool(item.get('requires_kaggle'))}` "
                f"side_effectful=`{bool(item.get('side_effectful'))}` "
                f"cleanup_required=`{bool(item.get('cleanup_required'))}` "
                f"command=`{item.get('command_line')}`"
            )
    lines.extend([
        "",
        "## Artifacts",
        "",
        f"- inspect first: `{artifact_overview.get('inspect_first')}`",
        f"- machine readable: `{artifact_overview.get('machine_readable')}`",
        f"- support bundle: `{artifact_overview.get('support_bundle')}`",
        f"- runbook: `{artifact_overview.get('runbook')}`",
    ])
    if shareable_paths:
        lines.extend(f"- shareable path: `{path}`" for path in shareable_paths)
    else:
        lines.append("- shareable path: `none`")
    lines.extend([
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`",
        "",
        "## Not Completed",
        "",
    ])
    not_completed = report.get("not_completed") or []
    lines.extend(f"- {item}" for item in not_completed) if not_completed else lines.append("- none")
    lines.extend([
        "",
        "## Boundaries",
        "",
    ])
    for item in report.get("limitations") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def print_human_summary(report: dict[str, Any]) -> None:
    beta = report.get("beta") if isinstance(report.get("beta"), dict) else {}
    readiness = report.get("readiness") if isinstance(report.get("readiness"), dict) else {}
    external = readiness.get("external_kaggle") if isinstance(readiness.get("external_kaggle"), dict) else {}
    p2p = readiness.get("p2p_candidate") if isinstance(readiness.get("p2p_candidate"), dict) else {}
    public_swarm_v2 = readiness.get("public_swarm_v2") if isinstance(readiness.get("public_swarm_v2"), dict) else {}
    kv_cache = readiness.get("usable_p2p_kv_cache") if isinstance(readiness.get("usable_p2p_kv_cache"), dict) else {}
    stage0 = kv_cache.get("stage0") if isinstance(kv_cache.get("stage0"), dict) else {}
    stage1 = kv_cache.get("stage1") if isinstance(kv_cache.get("stage1"), dict) else {}
    product_batch = beta.get("batch") if isinstance(beta.get("batch"), dict) else {}
    product_stream = beta.get("stream") if isinstance(beta.get("stream"), dict) else {}
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    provenance = report.get("runtime_provenance") if isinstance(report.get("runtime_provenance"), dict) else {}
    evidence_scope = report.get("evidence_scope") if isinstance(report.get("evidence_scope"), dict) else {}
    verdict = report.get("inference_verdict") if isinstance(report.get("inference_verdict"), dict) else {}
    user = report.get("user_status") if isinstance(report.get("user_status"), dict) else {}
    review = report.get("review_summary") if isinstance(report.get("review_summary"), dict) else {}
    recommended = (
        report.get("recommended_next_command")
        if isinstance(report.get("recommended_next_command"), dict)
        else review.get("recommended_next_command") if isinstance(review.get("recommended_next_command"), dict) else {}
    )
    check_command = report.get("recommended_check_command") if isinstance(report.get("recommended_check_command"), dict) else {}
    raw_operator_actions = report.get("operator_action")
    if isinstance(raw_operator_actions, list):
        operator_actions = [str(item) for item in raw_operator_actions]
    elif raw_operator_actions:
        operator_actions = [str(raw_operator_actions)]
    else:
        operator_actions = []
    not_completed = report.get("not_completed") if isinstance(report.get("not_completed"), list) else []
    print("CrowdTensor Public Real-LLM Swarm Inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {beta.get('ready')}")
    print(f"  model: {beta.get('hf_model_id')} tokens={beta.get('max_new_tokens')}")
    print(f"  external tokens: {external.get('generated_token_count')}/{external.get('required_generated_token_count')}")
    print(f"  p2p tokens: {p2p.get('generated_token_count')}/{p2p.get('required_generated_token_count')}")
    print(
        "  public_swarm_v2 tokens: "
        f"{public_swarm_v2.get('generated_token_count')}/{public_swarm_v2.get('required_generated_token_count')} "
        f"accepted_rows={public_swarm_v2.get('accepted_rows')}/{public_swarm_v2.get('required_stage_rows')}"
    )
    print(f"  public_swarm_v2 real_p2p_local: route={beta.get('public_swarm_v2_real_p2p_local_ready')} requeue={beta.get('public_swarm_v2_real_p2p_local_requeue_ready')}")
    print(f"  batch ready: product={product_batch.get('batch_generation_ready')} p2p={beta.get('p2p_batch_ready')} v2={beta.get('public_swarm_v2_batch_ready')}")
    print(f"  stream ready: product={product_stream.get('stream_generation_ready')} p2p={beta.get('p2p_stream_ready')} v2={beta.get('public_swarm_v2_stream_ready')}")
    print(f"  kv_cache_ready: {beta.get('kv_cache_ready')}")
    print(f"  kv_cache hits: stage0={stage0.get('hit_count')} stage1={stage1.get('hit_count')}")
    if provenance:
        print(f"  runtime_provenance: {runtime_provenance_text(provenance)}")
    if evidence_scope:
        print(f"  evidence_scope: {evidence_scope_text(evidence_scope)}")
        print(f"  evidence_scope_note: {evidence_scope.get('user_expectation') or ''}")
    if provenance or evidence_scope:
        gpu_status = report.get("gpu_status") if isinstance(report.get("gpu_status"), dict) else {}
        effective_gpu_status = gpu_status or gpu_status_summary(provenance, evidence_scope)
        gpu_proof_next_step = (
            report.get("gpu_proof_next_step")
            if isinstance(report.get("gpu_proof_next_step"), dict)
            else gpu_proof_next_step_summary(effective_gpu_status)
        )
        print(f"  gpu_status: {gpu_status_text(effective_gpu_status)}")
        print(f"  gpu_proof_next: {gpu_proof_next_step_text(gpu_proof_next_step)}")
    if verdict:
        print(f"  verdict: {inference_verdict_text(verdict)}")
        print(f"  verdict_note: {verdict.get('message')}")
    if user:
        print(
            "  status: "
            f"{user.get('state')}: {user.get('headline')} next={user.get('next_step')} "
            f"recommendation={user.get('recommended_label')} public_artifact_safe={bool(user.get('public_artifact_safe'))}"
        )
    if review:
        print(f"  review: state={review.get('state')} next_step={review.get('next_step')} inspect_first={review.get('inspect_first')}")
    if recommended:
        print(f"  recommended_next: {recommended.get('command_line')}")
    if check_command:
        print(f"  recommended_check: {check_command.get('command_line')}")
    for index, item in enumerate((report.get("next_commands") or []), start=1):
        if isinstance(item, dict):
            print(f"  next[{index}] {item.get('label')}: {item.get('command_line')}")
    if operator_actions:
        print("  operator_action:")
        for item in operator_actions[:4]:
            print(f"    - {item}")
        if len(operator_actions) > 4:
            print(f"    - ... {len(operator_actions) - 4} more")
    if output_request:
        print(f"  output_request: {output_request_text(output_request)}")
        print(f"  output_request_note: {output_request_note(output_request)}")
    if prompt_scope:
        print(f"  prompt_scope: {prompt_scope_text(prompt_scope)}")
        print(f"  prompt_scope_note: {prompt_scope_note(prompt_scope)}")
    if answer_scope:
        print(f"  answer_scope: {answer_scope_text(answer_scope)}")
        print(f"  answer_scope_note: {answer_scope_note(answer_scope)}")
    if shareable:
        print(f"  shareable: {shareable_summary_text(shareable)}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    if not_completed:
        print("  not_completed:")
        for item in not_completed[:8]:
            print(f"    - {item}")
        if len(not_completed) > 8:
            print(f"    - ... {len(not_completed) - 8} more")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        if isinstance(artifact, dict):
            print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def validate_public_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    provenance = report.get("runtime_provenance") if isinstance(report.get("runtime_provenance"), dict) else {}
    evidence_scope = report.get("evidence_scope") if isinstance(report.get("evidence_scope"), dict) else {}
    verdict = report.get("inference_verdict") if isinstance(report.get("inference_verdict"), dict) else {}
    if prompt_scope.get("source") not in {"prompt-text", "prompt-texts", "prompt-texts-file"}:
        errors.append("prompt_scope_source_mismatch")
    if not isinstance(prompt_scope.get("prompt_count"), int) or prompt_scope.get("prompt_count") < 1:
        errors.append("prompt_scope_count_mismatch")
    expected_inline = prompt_scope.get("source") in {"prompt-text", "prompt-texts"}
    if prompt_scope.get("inline_prompt_text") is not expected_inline:
        errors.append("prompt_scope_inline_prompt_text_mismatch")
    if prompt_scope.get("terminal_next_commands_local_private") is not expected_inline:
        errors.append("prompt_scope_terminal_next_commands_mismatch")
    if prompt_scope.get("terminal_logs_local_private") is not expected_inline:
        errors.append("prompt_scope_terminal_logs_mismatch")
    if prompt_scope.get("saved_artifacts_prompt_placeholders") is not True:
        errors.append("prompt_scope_saved_placeholders_mismatch")
    if prompt_scope.get("saved_artifacts_public_safe") is not True:
        errors.append("prompt_scope_saved_artifacts_public_safe_mismatch")
    if prompt_scope.get("prefer_prompt_file_or_stdin_for_shareable_logs") is not True:
        errors.append("prompt_scope_shareable_log_guidance_mismatch")
    if prompt_scope.get("prompt_file_path_public") is not False:
        errors.append("prompt_scope_prompt_file_path_public_mismatch")
    if prompt_scope.get("raw_prompt_public") is not False:
        errors.append("prompt_scope_raw_prompt_public_mismatch")
    if prompt_scope.get("public_artifact_safe") is not True:
        errors.append("prompt_scope_public_artifact_safe_mismatch")
    if evidence_scope.get("schema") != EVIDENCE_SCOPE_SCHEMA:
        errors.append("evidence_scope_schema_mismatch")
    if evidence_scope.get("public_artifact_safe") is not True:
        errors.append("evidence_scope_public_artifact_safe_mismatch")
    allowed_levels = {
        "fresh-kaggle-gpu",
        "package-only",
        "retained-evidence-import",
        "local-model-variant",
        "release-local-cpu-with-retained-external",
        "local-cpu",
        "blocked-or-mixed",
    }
    if evidence_scope.get("level") not in allowed_levels:
        errors.append("evidence_scope_level_mismatch")
    expected_local_cpu = bool(
        provenance.get("local_cpu_product_path_ready")
        or provenance.get("local_public_swarm_v2_ready")
        or provenance.get("local_kv_cache_ready")
    )
    expected_retained_external = bool(
        provenance.get("external_kaggle_cpu_evidence_claimed")
        or provenance.get("p2p_candidate_evidence_claimed")
    )
    if bool(evidence_scope.get("local_cpu")) != expected_local_cpu:
        errors.append("evidence_scope_local_cpu_mismatch")
    if bool(evidence_scope.get("retained_external_evidence_imported")) != expected_retained_external:
        errors.append("evidence_scope_retained_external_mismatch")
    for key in [
        "local_gpu_smoke_ran",
        "retained_gpu_evidence_imported",
        "fresh_kaggle_gpu_attempted",
        "fresh_kaggle_gpu_verified",
    ]:
        if bool(evidence_scope.get(key)) != bool(provenance.get(key)):
            errors.append(f"evidence_scope_{key}_mismatch")
    if verdict.get("schema") != INFERENCE_VERDICT_SCHEMA:
        errors.append("inference_verdict_schema_mismatch")
    if verdict.get("kind") != "Public Real-LLM Swarm Beta":
        errors.append("inference_verdict_kind_mismatch")
    if verdict.get("state") not in {"ready", "blocked", "package-ready", "local-smoke-ready", "local-model-ready"}:
        errors.append("inference_verdict_state_mismatch")
    if verdict.get("completed") is not report_ready(report):
        errors.append("inference_verdict_completed_mismatch")
    if verdict.get("answer_scope_state") != "no-local-answer":
        errors.append("inference_verdict_answer_scope_state_mismatch")
    if verdict.get("saved_artifacts_public_safe") is not True:
        errors.append("inference_verdict_saved_artifacts_public_safe_mismatch")
    gpu_status = report.get("gpu_status") if isinstance(report.get("gpu_status"), dict) else {}
    if verdict.get("gpu_state") != gpu_status.get("state"):
        errors.append("inference_verdict_gpu_state_mismatch")
    if bool(verdict.get("fresh_kaggle_gpu_verified")) != bool(gpu_status.get("fresh_kaggle_gpu_verified")):
        errors.append("inference_verdict_fresh_kaggle_gpu_mismatch")
    source_verdict = verdict.get("source_inference_verdict") if isinstance(verdict.get("source_inference_verdict"), dict) else {}
    if source_verdict.get("kind") != "Public Swarm v2":
        errors.append("inference_verdict_source_kind_mismatch")
    if verdict.get("public_artifact_safe") is not True:
        errors.append("inference_verdict_public_artifact_safe_mismatch")
    encoded = json.dumps(report, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment and fragment in encoded:
            errors.append(f"sensitive_fragment:{fragment}")
    for path in public_leak_paths(report):
        if path.endswith(".prompt_hash") or ".safety." in path:
            continue
        errors.append(f"public_leak:{path}")
    return sorted(set(errors))


def cleanup_release_private_artifacts(output_dir: Path) -> dict[str, Any]:
    removed: list[str] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        if (
            path.name == "tasks.jsonl"
            or path.name == "miner_registry.json"
            or path.name.endswith(".private.env")
            or "peer-key" in path.name
        ):
            try:
                path.unlink()
                removed.append(path.relative_to(output_dir).as_posix())
            except OSError:
                pass
    for state_dir in sorted(output_dir.rglob("state"), reverse=True):
        if not state_dir.is_dir():
            continue
        try:
            shutil.rmtree(state_dir)
            removed.append(state_dir.relative_to(output_dir).as_posix() + "/")
        except OSError:
            pass
    remaining = [
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and (
            path.name == "tasks.jsonl"
            or path.name == "miner_registry.json"
            or path.name.endswith(".private.env")
            or "peer-key" in path.name
        )
    ]
    return {
        "schema": "public_real_llm_swarm_beta_private_artifact_cleanup_v1",
        "private_artifacts_cleaned": not remaining and not any(path.is_dir() for path in output_dir.rglob("state")),
        "removed_private_artifact_count": len(removed),
    }


def persist_report(report: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    report.setdefault("output_request", output_request_summary())
    report.setdefault("prompt_scope", prompt_scope_summary(argparse.Namespace(prompt_text=DEFAULT_PROMPT, prompt_texts="", prompt_texts_file="", prompt_texts_list=[])))
    report.setdefault("answer_scope", answer_scope_summary())
    report.setdefault("shareable_summary", shareable_summary())
    report["runtime_provenance"] = runtime_provenance_summary(report)
    report["evidence_scope"] = evidence_scope_summary(report)
    report["gpu_status"] = gpu_status_summary(report["runtime_provenance"], report["evidence_scope"])
    report["gpu_proof_next_step"] = gpu_proof_next_step_summary(report["gpu_status"])
    report["recommended_check_command"] = recommended_check_command(report)
    report["inference_verdict"] = inference_verdict_summary(report)
    cleanup = cleanup_release_private_artifacts(output_dir)
    report["release_private_artifact_cleanup"] = cleanup
    if cleanup["private_artifacts_cleaned"]:
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"public_real_llm_swarm_beta_private_artifacts_cleaned"})
    else:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"public_real_llm_swarm_beta_private_artifact_cleanup_blocked"})
        if isinstance(report.get("beta"), dict):
            report["beta"]["ready"] = False
        report["not_completed"] = list(report.get("not_completed") or []) + ["release private artifact cleanup"]
    report = support_bundle.sanitize(redact_values(report))
    errors = validate_public_report(report)
    if errors:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"public_report_safety_failed"})
        report["safety_errors"] = errors
        if isinstance(report.get("beta"), dict):
            report["beta"]["ready"] = False
    artifacts = report.setdefault("artifacts", {})
    artifacts.setdefault("public_real_llm_swarm_beta_json", {
        "kind": "public_real_llm_swarm_beta",
        "path": "public_real_llm_swarm_beta.json",
        "present": True,
        "schema": SCHEMA,
        "ok": report.get("ok"),
    })
    artifacts.setdefault("public_real_llm_swarm_beta_markdown", {
        "kind": "public_real_llm_swarm_beta_markdown",
        "path": "public_real_llm_swarm_beta.md",
        "present": True,
    })
    artifacts.setdefault("support_bundle_json", {
        "kind": "public_real_llm_swarm_beta_support_bundle",
        "path": "support_bundle.json",
        "present": True,
        "schema": SUPPORT_SCHEMA,
        "ok": report.get("ok"),
    })
    report = attach_user_guidance(report)
    errors = validate_public_report(report)
    if errors:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"public_report_safety_failed"})
        report["safety_errors"] = errors
        if isinstance(report.get("beta"), dict):
            report["beta"]["ready"] = False
        report["artifacts"]["public_real_llm_swarm_beta_json"]["ok"] = False
        report["artifacts"]["support_bundle_json"]["ok"] = False
        report = attach_user_guidance(report)
    write_json(output_dir / "public_real_llm_swarm_beta.json", report)
    (output_dir / "public_real_llm_swarm_beta.md").write_text(render_markdown(report), encoding="utf-8")
    bundle = support_bundle.sanitize({
        "schema": SUPPORT_SCHEMA,
        "ok": report.get("ok"),
        "diagnosis_codes": report.get("diagnosis_codes"),
        "beta": report.get("beta"),
        "readiness": report.get("readiness"),
        "artifact_summary": report.get("artifact_summary"),
        "review_summary": report.get("review_summary"),
        "recommended_check_command": report.get("recommended_check_command"),
        "recommended_next_command": report.get("recommended_next_command"),
        "next_commands": report.get("next_commands"),
        "user_status": report.get("user_status"),
        "operator_action": report.get("operator_action"),
        "not_completed": report.get("not_completed"),
        "release_private_artifact_cleanup": report.get("release_private_artifact_cleanup"),
        "safety_errors": report.get("safety_errors"),
        "output_request": report.get("output_request"),
        "prompt_scope": report.get("prompt_scope"),
        "answer_scope": report.get("answer_scope"),
        "shareable_summary": report.get("shareable_summary"),
        "runtime_provenance": report.get("runtime_provenance"),
        "evidence_scope": report.get("evidence_scope"),
        "gpu_status": report.get("gpu_status"),
        "gpu_proof_next_step": report.get("gpu_proof_next_step"),
        "inference_verdict": report.get("inference_verdict"),
        "safety": report.get("safety"),
        "limitations": report.get("limitations"),
    })
    write_json(output_dir / "support_bundle.json", bundle)
    report["artifacts"]["support_bundle_json"]["ok"] = bundle.get("ok")
    write_json(output_dir / "public_real_llm_swarm_beta.json", report)
    return report


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == MODE_PACKAGE:
        return build_package(args, output_dir=output_dir)
    if args.mode == MODE_LOCAL_SMOKE:
        return build_local_smoke(args, output_dir=output_dir, runner=runner)
    if args.mode == MODE_LOCAL_MODEL_VARIANT:
        return build_local_model_variant(args, output_dir=output_dir, runner=runner)
    if args.mode == MODE_EVIDENCE_IMPORT:
        return build_evidence_import(args, output_dir=output_dir)
    return build_release(args, output_dir=output_dir, runner=runner)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Public Real-LLM Swarm Inference Beta v1 evidence.")
    parser.add_argument("mode", choices=MODES, nargs="?", default=MODE_RELEASE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--product-report", default=DEFAULT_PRODUCT_REPORT)
    parser.add_argument("--external-report", default=DEFAULT_EXTERNAL_REPORT)
    parser.add_argument("--p2p-report", default=DEFAULT_P2P_REPORT)
    parser.add_argument("--usable-report", default=DEFAULT_USABLE_REPORT)
    parser.add_argument("--public-swarm-v2-report", default=DEFAULT_PUBLIC_SWARM_V2_REPORT)
    parser.add_argument("--public-swarm-v2-preview-report", default=DEFAULT_PUBLIC_SWARM_V2_PREVIEW_REPORT)
    parser.add_argument("--public-swarm-v2-real-p2p-report", default=DEFAULT_PUBLIC_SWARM_V2_REAL_P2P_REPORT)
    parser.add_argument("--p2p-runtime-smoke-report", default=DEFAULT_P2P_RUNTIME_SMOKE_REPORT)
    parser.add_argument("--p2p-external-report", default=DEFAULT_P2P_EXTERNAL_REPORT)
    parser.add_argument("--p2p-requeue-report", default=DEFAULT_P2P_REQUEUE_REPORT)
    parser.add_argument("--p2p-batch-stream-report", default=DEFAULT_P2P_BATCH_STREAM_REPORT)
    parser.add_argument("--gpu-report", default=DEFAULT_GPU_REPORT)
    parser.add_argument("--public-host", default="127.0.0.1")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9340)
    parser.add_argument("--base-port", type=int, default=9340)
    parser.add_argument("--p2p-port", type=int, default=9860)
    parser.add_argument("--p2p-libp2p-port", type=int, default=10860)
    parser.add_argument("--public-swarm-v2-p2p-port", type=int, default=9888)
    parser.add_argument("--public-swarm-v2-coordinator-port", type=int, default=9889)
    parser.add_argument("--public-swarm-v2-real-p2p-port", type=int, default=DEFAULT_PUBLIC_SWARM_V2_REAL_P2P_PORT)
    parser.add_argument("--public-swarm-v2-real-p2p-coordinator-port", type=int, default=DEFAULT_PUBLIC_SWARM_V2_REAL_P2P_COORDINATOR_PORT)
    parser.add_argument("--public-swarm-v2-real-p2p-libp2p-port", type=int, default=DEFAULT_PUBLIC_SWARM_V2_REAL_P2P_LIBP2P_PORT)
    parser.add_argument("--public-swarm-v2-real-p2p-discovery-backend", default=DEFAULT_PUBLIC_SWARM_V2_REAL_P2P_DISCOVERY_BACKEND)
    parser.add_argument("--public-swarm-v2-backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--hf-model-id", default=DEFAULT_HF_MODEL_ID)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-texts", default="", help="comma-separated bounded batch of up to 4 prompts")
    parser.add_argument("--prompt-texts-file", default="", help="newline-delimited bounded batch of up to 4 prompts")
    parser.add_argument("--stream-generation", action="store_true")
    parser.add_argument("--request-count", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--cpu-request-count", type=int, default=1)
    parser.add_argument("--external-llm-request-count", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--public-swarm-v2-timeout-seconds", type=float, default=420.0)
    parser.add_argument("--remote-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--cpu-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--process-exit-timeout", type=float, default=20.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--http-timeout", type=float, default=10.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.request_count < 1 or args.request_count > 4:
        raise SystemExit("--request-count must be between 1 and 4")
    if args.cpu_request_count < 1 or args.cpu_request_count > 4:
        raise SystemExit("--cpu-request-count must be between 1 and 4")
    if args.external_llm_request_count < 1 or args.external_llm_request_count > 4:
        raise SystemExit("--external-llm-request-count must be between 1 and 4")
    if args.max_new_tokens < 2 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 2 and 32")
    if args.prompt_texts and args.prompt_texts_file:
        raise SystemExit("public_real_llm_swarm_beta accepts either --prompt-texts or --prompt-texts-file, not both")
    try:
        if args.prompt_texts_file:
            args.prompt_texts_list = read_prompt_texts_file(args.prompt_texts_file)
        else:
            args.prompt_texts_list = parse_prompt_texts_arg(args.prompt_text, args.prompt_texts)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for name in [
        "timeout_seconds",
        "remote_timeout_seconds",
        "cpu_timeout_seconds",
        "startup_timeout",
        "process_exit_timeout",
        "poll_interval",
        "http_timeout",
    ]:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if (
        args.port < 1
        or args.base_port < 1
        or args.p2p_port < 1
        or args.p2p_libp2p_port < 1
        or args.public_swarm_v2_p2p_port < 1
        or args.public_swarm_v2_coordinator_port < 1
        or args.public_swarm_v2_real_p2p_port < 1
        or args.public_swarm_v2_real_p2p_coordinator_port < 1
        or args.public_swarm_v2_real_p2p_libp2p_port < 0
    ):
        raise SystemExit("public Real-LLM Beta ports must be positive, except --public-swarm-v2-real-p2p-libp2p-port may be 0")
    return args


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print_human_summary(report)
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
