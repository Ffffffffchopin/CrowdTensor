#!/usr/bin/env python3
"""Build the Public Swarm Inference v2 preview artifact.

This is the v2 product gate over the ordinary p2pd -> serve --p2p ->
join stage0/stage1 -> generate --p2p path. It keeps Coordinator authority for
sessions, leases, validation, and result ledgers while preferring signed/real
P2P evidence when available.
"""

from __future__ import annotations

import argparse
import json
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
from product_swarm_mvp_check import parse_prompt_texts_arg, read_prompt_texts_file  # noqa: E402
from crowdtensor.real_p2p import DISCOVERY_BACKENDS  # noqa: E402
from crowdtensor.session_protocol import public_leak_paths, safe_generation_summary  # noqa: E402


SCHEMA = "public_swarm_inference_v2"
SUPPORT_SCHEMA = "public_swarm_inference_v2_support_bundle_v1"
USABLE_SCHEMA = "usable_swarm_inference_v1"
PREVIEW_V04_SCHEMA = "public_swarm_preview_v04_v1"
REAL_P2P_SCHEMA = "real_p2p_swarm_inference_core_rc_v1"
GPU_SCHEMA = "public_swarm_gpu_inference_beta_v1"
WORKLOAD_TYPE = "real_llm_sharded_infer"

MODE_LOCAL = "local"
MODE_LOCAL_MODEL_VARIANT = "local-model-variant"
MODE_PACKAGE = "package"
MODE_EVIDENCE_IMPORT = "evidence-import"
MODES = [MODE_LOCAL, MODE_LOCAL_MODEL_VARIANT, MODE_PACKAGE, MODE_EVIDENCE_IMPORT]

DEFAULT_OUTPUT_DIR = "dist/public-swarm-inference-v2"
DEFAULT_USABLE_REPORT = "dist/goal-final-infer-usable-swarm-16tok-kv-cache-20260601/usable_swarm_inference.json"
DEFAULT_PREVIEW_REPORT = "dist/public-swarm-preview-v04-final/public_swarm_preview_v04.json"
DEFAULT_REAL_P2P_REPORT = "dist/goal-final-infer-real-p2p-core-fresh-16tok-import-strict-20260601/real_p2p_swarm_inference_core_rc.json"
DEFAULT_GPU_REPORT = "dist/public-swarm-gpu-beta-live-20260528-runtimepin/public_swarm_gpu_inference_beta_kaggle_auto.json"
DEFAULT_PROMPT = "CrowdTensor Public Swarm Inference v2"
DEFAULT_HF_MODEL_ID = "sshleifer/tiny-gpt2"
DEFAULT_REAL_P2P_LOCAL_P2P_PORT = 9890
DEFAULT_REAL_P2P_LOCAL_COORDINATOR_PORT = 9891
DEFAULT_REAL_P2P_DISCOVERY_BACKEND = "http-provider-store"

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
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
)
V2_OWNED_CODES = {
    "public_swarm_inference_v2_ready",
    "public_swarm_inference_v2_preview_ready",
    "public_swarm_inference_v2_blocked",
    "public_swarm_inference_v2_local_model_variant_ready",
    "public_swarm_v2_16_token_generation_ready",
    "public_swarm_v2_cuda_optional_fail_closed_ready",
    "public_swarm_v2_dual_stage_kv_cache_ready",
    "public_swarm_v2_external_evidence_ready",
    "public_swarm_v2_external_fresh_run_action_required",
    "public_swarm_v2_fresh_external_attempted",
    "public_swarm_v2_fresh_external_blocked_actionable",
    "public_swarm_v2_fresh_external_runtime_verified",
    "public_swarm_v2_external_stage_rows_ready",
    "public_swarm_v2_local_p2p_generate_ready",
    "public_swarm_v2_model_match_ready",
    "public_swarm_v2_local_model_mismatch",
    "public_swarm_v2_local_model_variant_blocked",
    "public_swarm_v2_local_model_variant_model_match_ready",
    "public_swarm_v2_local_model_variant_ready",
    "public_swarm_v2_external_model_mismatch",
    "public_swarm_v2_external_validation_not_claimed",
    "public_swarm_v2_p2p_model_mismatch",
    "public_swarm_v2_p2p_lite_fallback_ready",
    "public_swarm_v2_real_p2p_local_requeue_ready",
    "public_swarm_v2_real_p2p_local_ready",
    "public_swarm_v2_retained_external_evidence_ready",
    "public_swarm_v2_runbook_ready",
    "public_swarm_v2_signed_or_real_p2p_ready",
    "public_swarm_v2_stage_requeue_rescue_ready",
    "public_swarm_v2_stream_generation_ready",
    "public_swarm_v2_batch_generation_ready",
}

BATCH_READY_CODES = {
    "public_swarm_generate_batch_ready",
    "public_swarm_v2_batch_generation_ready",
}

STREAM_READY_CODES = {
    "public_swarm_generate_stream_ready",
    "public_swarm_generate_stream_endpoint_ready",
    "public_swarm_v2_stream_generation_ready",
}

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


def drop_unproven_generation_ready_codes(
    codes: set[str],
    *,
    batch_ready: bool,
    stream_ready: bool,
) -> set[str]:
    filtered = set(codes)
    if not batch_ready:
        filtered -= BATCH_READY_CODES
    if not stream_ready:
        filtered -= STREAM_READY_CODES
    return filtered


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def safe_batch_summary(batch: dict[str, Any] | None) -> dict[str, Any]:
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
    identity_keys = [
        str(item.get("request_id") or item.get("prompt_hash") or "")
        for item in safe_results[:expected_request_count]
    ]
    batch_identity_ready = bool(
        expected_request_count > 0
        and (
            expected_request_count <= 1
            or (
                len(identity_keys) >= expected_request_count
                and all(identity_keys)
                and len(set(identity_keys)) == expected_request_count
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


def nested_dicts(value: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    pending: list[Any] = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop(0)
        if not isinstance(item, dict):
            continue
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        items.append(item)
        for child in item.values():
            if isinstance(child, dict):
                pending.append(child)
            elif isinstance(child, list):
                pending.extend(entry for entry in child if isinstance(entry, dict))
    return items


def first_int(value: Any, key: str) -> int:
    for item in nested_dicts(value):
        if key not in item:
            continue
        try:
            return int(item.get(key) or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def first_string_value(payload: dict[str, Any], key: str) -> str:
    for item in nested_dicts(payload):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def observed_model_id(payload: dict[str, Any]) -> str:
    imported = payload.get("imported") if isinstance(payload.get("imported"), dict) else {}
    if imported:
        imported_model_id = imported.get("hf_model_id")
        return imported_model_id if isinstance(imported_model_id, str) and imported_model_id else ""
    return first_string_value(payload, "hf_model_id")


def model_compatibility(payload: dict[str, Any], expected_model_id: str) -> dict[str, Any]:
    observed = observed_model_id(payload)
    return {
        "expected_hf_model_id": expected_model_id,
        "observed_hf_model_id": observed,
        "model_id_present": bool(observed),
        "model_id_match": bool(observed and observed == expected_model_id),
        "compatible": bool(observed and observed == expected_model_id),
        "default_model_retained_evidence": False,
    }


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


def local_p2p_summary(payload: dict[str, Any], *, required_tokens: int, expected_model_id: str = DEFAULT_HF_MODEL_ID) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    model = model_compatibility(payload, expected_model_id)
    readiness = payload.get("readiness") if isinstance(payload.get("readiness"), dict) else {}
    p2p = readiness.get("p2p_product_path") if isinstance(readiness.get("p2p_product_path"), dict) else {}
    generation = p2p.get("generation") if isinstance(p2p.get("generation"), dict) else {}
    batch = safe_batch_summary(p2p.get("batch") if isinstance(p2p.get("batch"), dict) else {})
    stream = p2p.get("stream") if isinstance(p2p.get("stream"), dict) else {}
    kv_cache = p2p.get("kv_cache") if isinstance(p2p.get("kv_cache"), dict) else {}
    safe_generation = safe_generation_summary({"generation": generation or p2p}, max_new_tokens=required_tokens)
    generated = int(p2p.get("generated_token_count") or safe_generation.get("generated_token_count") or 0)
    accepted_rows = int(p2p.get("accepted_rows") or 0)
    route = p2p.get("route") if isinstance(p2p.get("route"), dict) else {}
    top_route_source = str(p2p.get("route_source") or "")
    nested_route_source = str(route.get("route_source") or "")
    route_source = top_route_source or nested_route_source
    route_source_consistent = not (
        top_route_source
        and nested_route_source
        and top_route_source != nested_route_source
    )
    stream_ready = stream_evidence_ready(stream, batch)
    ready = bool(
        payload.get("schema") == USABLE_SCHEMA
        and payload.get("ok") is True
        and p2p.get("route_ready") is True
        and route_source == "p2p-discovery"
        and route_source_consistent
        and p2p.get("p2p_counts_ready") is True
        and p2p.get("real_generate_ready") is True
        and p2p.get("kv_cache_ready") is True
        and generated >= required_tokens
        and accepted_rows >= required_tokens * 2
        and p2p.get("distinct_stage_miners") is True
        and p2p.get("stage_rescue_ready") is True
        and p2p.get("real_stage_rescue_ready") is True
        and model["compatible"]
    )
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "ready": ready,
        "model": model,
        "mode": payload.get("mode"),
        "route_source": route_source,
        "nested_route_source": nested_route_source,
        "route_source_consistent": route_source_consistent,
        "route_ready": p2p.get("route_ready") is True,
        "p2p_counts_ready": p2p.get("p2p_counts_ready") is True,
        "real_generate_ready": p2p.get("real_generate_ready") is True,
        "batch_ready": bool(batch.get("enabled") and batch.get("batch_generation_ready") is True),
        "stream_ready": stream_ready,
        "kv_cache_ready": p2p.get("kv_cache_ready") is True,
        "kv_cache": kv_cache,
        "generated_token_count": generated,
        "max_new_tokens": int(p2p.get("max_new_tokens") or safe_generation.get("max_new_tokens") or required_tokens),
        "generation_target_ready": generated >= required_tokens,
        "accepted_rows": accepted_rows,
        "accepted_rows_ready": accepted_rows >= required_tokens * 2,
        "distinct_stage_miners": p2p.get("distinct_stage_miners") is True,
        "stage_requeue_rescue_ready": bool(p2p.get("stage_rescue_ready") and p2p.get("real_stage_rescue_ready")),
        "stage_assignment": p2p.get("stage_assignment") if isinstance(p2p.get("stage_assignment"), dict) else {},
        "generation": safe_generation,
        "batch": batch,
        "stream": stream if stream else {"enabled": False, "stream_generation_ready": False},
        "usable_evidence_source": p2p.get("usable_evidence_source", ""),
        "diagnosis_codes": sorted(codes),
    }


def external_summary(
    payload: dict[str, Any],
    *,
    required_tokens: int,
    fresh_external_report: bool = False,
    expected_model_id: str = DEFAULT_HF_MODEL_ID,
) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    model = model_compatibility(payload, expected_model_id)
    generation = safe_generation_summary(payload, max_new_tokens=required_tokens)
    generated = int(generation.get("generated_token_count") or 0)
    generation_target_ready = generated >= required_tokens
    stage_assignment = payload.get("stage_assignment") if isinstance(payload.get("stage_assignment"), dict) else {}
    ledger = payload.get("ledger") if isinstance(payload.get("ledger"), dict) else {}
    accepted_rows = int(stage_assignment.get("completed_rows") or ledger.get("accepted_rows") or first_int(payload, "accepted_rows") or 0)
    accepted_rows_ready = accepted_rows >= required_tokens * 2
    external_verified = "external_runtime_verified" in codes or bool(payload.get("external", {}).get("external_runtime_verified") if isinstance(payload.get("external"), dict) else False)
    stage_requeue = bool(
        "external_stage_requeue_ready" in codes
        and (
            "live_stage0_requeue_ready" in codes
            or "live_stage1_requeue_ready" in codes
            or "accepted_result_after_requeue" in codes
            or "rescue_miner_used" in codes
        )
    )
    cleanup_ready = "kaggle_kernels_deleted" in codes or "real_p2p_kaggle_private_artifacts_cleaned" in codes
    token_rotation = "token_rotation_required" in codes
    route_source = ""
    for item in nested_dicts(payload):
        route = item.get("route") if isinstance(item.get("route"), dict) else {}
        if route.get("route_source"):
            route_source = str(route.get("route_source"))
            break
    ready = bool(
        payload.get("ok") is True
        and external_verified
        and generation_target_ready
        and accepted_rows_ready
        and cleanup_ready
        and token_rotation
        and "distinct_stage_miners" in codes
        and "stage_assignment_valid" in codes
        and model["compatible"]
    )
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "ready": ready,
        "model": model,
        "retained_external_evidence_ready": ready,
        "generation_target_ready": generation_target_ready,
        "accepted_rows": accepted_rows,
        "accepted_rows_ready": accepted_rows_ready,
        "fresh_external_runtime_verified": bool(fresh_external_report and ready and generation_target_ready),
        "fresh_external_run_required": not bool(fresh_external_report and ready and generation_target_ready),
        "actionable_external_runbook_ready": True,
        "mode": payload.get("mode"),
        "external_runtime_verified": external_verified,
        "generated_token_count": generated,
        "max_new_tokens": generation.get("max_new_tokens"),
        "generation": generation,
        "stage_requeue_ready": stage_requeue,
        "stage_assignment": stage_assignment,
        "kaggle_kernels_deleted": cleanup_ready,
        "token_rotation_required": token_rotation,
        "distinct_stage_miners": "distinct_stage_miners" in codes,
        "stage_assignment_valid": "stage_assignment_valid" in codes,
        "route_source": route_source,
        "real_or_signed_p2p_ready": "libp2p_or_real_p2p_discovery_ready" in codes or "real_p2p_signed_provider_records_ready" in codes,
        "diagnosis_codes": sorted(codes),
    }


def p2p_summary(payload: dict[str, Any], *, expected_model_id: str = DEFAULT_HF_MODEL_ID) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    model = model_compatibility(payload, expected_model_id)
    p2p = payload.get("p2p") if isinstance(payload.get("p2p"), dict) else {}
    route = p2p.get("route") if isinstance(p2p.get("route"), dict) else {}
    route_source = str(route.get("route_source") or "")
    for item in nested_dicts(payload):
        if route_source:
            break
        route = item.get("route") if isinstance(item.get("route"), dict) else {}
        if route.get("route_source"):
            route_source = str(route.get("route_source"))
            break
    matched = route.get("matched_capabilities") if isinstance(route.get("matched_capabilities"), dict) else {}
    real_route_evidence_ready = bool(
        p2p.get("backend") == "real"
        and route_source == "real-p2p-discovery"
        and route.get("usable_now") is True
        and route.get("coordinator_url_present") is True
        and matched.get("real_llm_sharded_stage0")
        and matched.get("real_llm_sharded_stage1")
        and int(p2p.get("coordinator_provider_count") or 0) >= 1
        and int(p2p.get("stage0_provider_count") or 0) >= 1
        and int(p2p.get("stage1_provider_count") or 0) >= 1
        and int(p2p.get("signed_provider_record_count") or 0) >= 1
        and not p2p.get("catalog_error")
    )
    route_ready = bool(
        "libp2p_or_real_p2p_discovery_ready" in codes
        or "libp2p_discovery_backend_ready" in codes
        or "real_p2p_signed_provider_records_ready" in codes
        or real_route_evidence_ready
    )
    local_stage_requeue_ready = bool(
        "real_p2p_local_stage_requeue_ready" in codes
        and (
            "local_stage_requeue_ready" in codes
            or "stage_requeue_ready" in codes
        )
        and (
            "live_stage0_requeue_ready" in codes
            or "live_stage1_requeue_ready" in codes
        )
        and "accepted_result_after_requeue" in codes
        and "rescue_miner_used" in codes
    )
    local_route_hardening_ready = bool(
        payload.get("mode") == "local-smoke"
        and route_ready
        and local_stage_requeue_ready
        and route_source == "real-p2p-discovery"
        and model["compatible"]
    )
    ready = bool((payload.get("ok") is True or local_route_hardening_ready) and route_ready and model["compatible"])
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "ready": ready,
        "model": model,
        "mode": payload.get("mode"),
        "preferred_route": "real-p2p" if ready else "p2p-lite-fallback",
        "route_source": route_source,
        "route_ready": route_ready,
        "real_route_evidence_ready": real_route_evidence_ready,
        "local_route_hardening_ready": local_route_hardening_ready,
        "signed_provider_records_ready": "real_p2p_signed_provider_records_ready" in codes or int(p2p.get("signed_provider_record_count") or 0) >= 1,
        "libp2p_discovery_backend_ready": "libp2p_discovery_backend_ready" in codes,
        "provider_store_ready": "real_p2p_provider_store_ready" in codes or bool(int(p2p.get("provider_count") or 0) >= 1 and not p2p.get("catalog_error")),
        "local_stage_requeue_ready": local_stage_requeue_ready,
        "local_stage_requeue_target": (payload.get("live_requeue_summary") or {}).get("target_stage") if isinstance(payload.get("live_requeue_summary"), dict) else "",
        "diagnosis_codes": sorted(codes),
    }


def gpu_summary(payload: dict[str, Any]) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    cuda_available = "cuda_runtime_available" in codes
    cuda_ready = "public_swarm_gpu_beta_ready" in codes or "gpu_runtime_ready" in codes
    cuda_unavailable = "cuda_runtime_unavailable" in codes
    fail_closed = bool(payload.get("ok") is True and (cuda_ready or cuda_unavailable))
    generated = first_int(payload, "generated_token_count")
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "ready": bool(payload.get("ok") is True and cuda_ready),
        "fail_closed_ready": fail_closed,
        "cuda_available": cuda_available,
        "cuda_unavailable": cuda_unavailable,
        "generated_token_count": generated,
        "external_gpu_runtime_verified": "external_gpu_runtime_verified" in codes,
        "kaggle_kernels_deleted": "kaggle_kernels_deleted" in codes,
        "diagnosis_codes": sorted(codes),
    }


def fresh_external_attempt_summary(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    if not payload:
        return {
            "schema": "public_swarm_v2_fresh_external_attempt_v1",
            "present": False,
            "attempted": False,
            "ready": False,
            "blocked": False,
            "diagnosis_codes": [],
        }
    codes = set(diagnosis_codes(payload))
    ready = bool(
        payload.get("ok") is True
        and (
            "public_swarm_v2_fresh_external_runtime_verified" in codes
            or "external_runtime_verified" in codes
            or payload.get("fresh_external_runtime_verified") is True
        )
    )
    cleanup_ready = bool(
        payload.get("kaggle_kernels_deleted") is True
        or "kaggle_kernels_deleted" in codes
        or "private_artifacts_cleaned" in codes
        or "real_p2p_kaggle_private_artifacts_cleaned" in codes
    )
    blocked = bool(payload.get("blocked") is True or (payload.get("attempted") is True and not ready))
    return {
        "schema": payload.get("schema", "public_swarm_v2_fresh_external_attempt_v1"),
        "present": True,
        "attempted": bool(payload.get("attempted", True)),
        "ready": ready,
        "blocked": blocked,
        "blocked_reason": payload.get("blocked_reason", ""),
        "actionable": bool(payload.get("actionable", blocked or ready)),
        "kaggle_kernels_deleted": cleanup_ready,
        "token_rotation_required": bool(payload.get("token_rotation_required") is True or "token_rotation_required" in codes),
        "diagnosis_codes": sorted(codes),
        "artifact_path": str(path),
    }


def performance_summary(preview_payload: dict[str, Any], usable_summary: dict[str, Any], gpu: dict[str, Any]) -> dict[str, Any]:
    preview = preview_payload.get("preview") if isinstance(preview_payload.get("preview"), dict) else {}
    performance = preview_payload.get("performance") if isinstance(preview_payload.get("performance"), dict) else {}
    stage_latency = performance.get("stage_latency") if isinstance(performance.get("stage_latency"), dict) else {}
    throughput = performance.get("throughput") if isinstance(performance.get("throughput"), dict) else {}
    memory = performance.get("memory_or_vram") if isinstance(performance.get("memory_or_vram"), dict) else {}
    generated = max(int(usable_summary.get("generated_token_count") or 0), int(gpu.get("generated_token_count") or 0), first_int(preview_payload, "generated_token_count"))
    return {
        "stage_latency_ready": bool(preview.get("stage_latency_ready") or stage_latency.get("stage_latency_ready")),
        "throughput_summary_ready": bool(preview.get("throughput_summary_ready") or throughput.get("throughput_summary_ready")),
        "memory_or_vram_summary_ready": bool(preview.get("memory_or_vram_summary_ready") or memory.get("memory_or_vram_summary_ready")),
        "stage_latency": stage_latency,
        "throughput": throughput,
        "memory_or_vram": memory,
        "generated_token_count_observed": generated,
    }


def safety_summary(*, external_ready: bool, cuda_optional_ready: bool) -> dict[str, Any]:
    return {
        "coordinator_backed_task_execution": True,
        "p2p_discovery_primary_path": True,
        "persistent_dual_stage_kv_cache_required": True,
        "signed_or_real_p2p_preferred_when_available": True,
        "p2p_lite_fallback_explicit": True,
        "read_only_workload": WORKLOAD_TYPE,
        "cpu_default": True,
        "cuda_optional": True,
        "cuda_optional_fail_closed_ready": cuda_optional_ready,
        "external_runtime_verified": external_ready,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "activation_payloads_redacted": True,
        "tokens_gossiped": False,
        "raw_prompts_gossiped": False,
        "activations_gossiped": False,
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
            "Public Swarm v2 artifacts summarize generation readiness with counts, hashes, "
            "stream milestones, and route evidence only. Run `crowdtensor generate --p2p` "
            "in human mode to see a local answer."
        ),
    }


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
            "This Public Swarm v2 report is shareable aggregate evidence, not a local answer transcript; "
            "raw prompts, generated text, generated token ids, activations, leases, and credentials are excluded."
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
        "summary": "Share public_swarm_inference_v2.json/md and support_bundle.json; they contain hashes/counts and readiness evidence, not raw prompts or answers.",
    }


def output_request_text(summary: dict[str, Any]) -> str:
    return (
        f"include_output={bool(summary.get('include_output'))} "
        f"raw_generated_text_public={bool(summary.get('raw_generated_text_public'))} "
        f"public_artifact_safe={bool(summary.get('public_artifact_safe'))}"
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
        "Public Swarm Inference v2 is a Coordinator-backed public preview, not full Hivemind/Petals production parity.",
        "Signed/real P2P discovery is preferred when evidence is available; P2P-lite remains an explicit fallback.",
        "The default model path is tiny/small Hugging Face real weights; this is not large-model throughput serving.",
        "CPU is the default; CUDA is optional and must fail closed when unavailable.",
        "No production NAT/relay fabric, decentralized security, payments, staking, billing, or anti-Sybil network is included.",
    ]


def local_model_variant_codes(*payloads: dict[str, Any]) -> set[str]:
    blocked_prefixes = (
        "external_",
        "kaggle_",
        "public_swarm_live",
        "real_llm_internet",
        "real_llm_live",
        "remote_",
        "swarm_inference_beta_live",
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
        "token_rotation_required",
    }
    codes: set[str] = set()
    for code in diagnosis_codes(*payloads):
        if code in blocked_codes:
            continue
        if any(code.startswith(prefix) for prefix in blocked_prefixes):
            continue
        codes.add(code)
    return codes


def write_runbook(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    path = output_dir / "PUBLIC_SWARM_INFERENCE_V2.md"
    lines = [
        "# CrowdTensor Public Swarm Inference v2",
        "",
        "Run this as a public preview over the ordinary product path. Keep Coordinator authority for sessions, leases, validation, and result ledgers.",
        "",
        "## Local 16-Token Proof",
        "",
        "```bash",
        "python -m pip install -e '.[hf]'",
        "read -r -s -p 'Admin token: ' CROWDTENSOR_ADMIN_TOKEN; echo",
        "read -r -s -p 'Miner token: ' CROWDTENSOR_MINER_TOKEN; echo",
        "export CROWDTENSOR_ADMIN_TOKEN CROWDTENSOR_MINER_TOKEN",
        f"crowdtensor public-swarm-v2 local --max-new-tokens {args.max_new_tokens} --http-timeout 30 --json",
        "```",
        "",
        "The local gate also fresh-runs the real-P2P route-hardening child proof with provider records and a local stage1 victim/rescue requeue:",
        "",
        "```bash",
        f"python scripts/real_p2p_swarm_inference_core_rc_pack.py local-smoke --swarm-id public-swarm-v2-real-p2p --p2p-port {args.real_p2p_port} --coordinator-port {args.real_p2p_coordinator_port} --discovery-backend {args.real_p2p_discovery_backend} --failure-mode kill-stage1-after-claim --lease-seconds 2 --victim-compute-seconds 8 --requeue-timeout 90 --max-new-tokens {args.max_new_tokens} --json",
        "```",
        "",
        "## Two-Machine or Kaggle Rehearsal",
        "",
        "```bash",
        "export COORDINATOR_PUBLIC_HOST='<public-host-or-vpn-hostname>'",
        f"crowdtensor p2pd --host 0.0.0.0 --port {args.p2p_port} --swarm-id public-swarm-v2 --run",
        f"crowdtensor serve --p2p --peer-bootstrap \"http://$COORDINATOR_PUBLIC_HOST:{args.p2p_port}\" --swarm-id public-swarm-v2 --bind-host 0.0.0.0 --public-host \"$COORDINATOR_PUBLIC_HOST\" --port {args.coordinator_port} --i-understand-public-bind --run",
        "```",
        "",
        "Run stage0 and stage1 on distinct hosts or notebooks:",
        "",
        "```bash",
        f"crowdtensor join --stage stage0 --p2p --peer-bootstrap \"http://$COORDINATOR_PUBLIC_HOST:{args.p2p_port}\" --swarm-id public-swarm-v2 --miner-id \"$(hostname)-stage0\" --run",
        f"crowdtensor join --stage stage1 --p2p --peer-bootstrap \"http://$COORDINATOR_PUBLIC_HOST:{args.p2p_port}\" --swarm-id public-swarm-v2 --miner-id \"$(hostname)-stage1\" --run",
        f"crowdtensor generate --p2p --peer-bootstrap \"http://$COORDINATOR_PUBLIC_HOST:{args.p2p_port}\" --prompt \"{args.prompt_text}\" --max-new-tokens {args.max_new_tokens}",
        "```",
        "",
        "For Kaggle, use two private notebooks as external Miner hosts. Put only the Miner token in notebooks; keep the admin token on the operator host. Rotate tokens after public HTTP tests.",
        "",
        "Maintainers can automate a fresh external real-P2P Kaggle proof and then import it as the v2 external report:",
        "",
        "```bash",
        f"python scripts/real_p2p_swarm_inference_core_rc_pack.py kaggle-auto --discovery-backend libp2p-kad --public-host \"$COORDINATOR_PUBLIC_HOST\" --p2p-port {args.p2p_port} --coordinator-port {args.coordinator_port} --max-new-tokens {args.max_new_tokens} --timeout-seconds 900 --generate-timeout 900 --http-timeout 30 --json",
        "crowdtensor public-swarm-v2 evidence-import --fresh-external-report --real-p2p-report dist/<fresh-real-p2p-run>/real_p2p_swarm_inference_core_rc.json --max-new-tokens 16 --json",
        "```",
        "",
        "If the fresh external run fails before generation completes, delete temporary Kaggle kernels, rotate tokens, write a redacted `public_swarm_v2_fresh_external_attempt_v1` summary, and import it instead of claiming fresh success:",
        "",
        "```bash",
        "crowdtensor public-swarm-v2 evidence-import --fresh-external-attempt-report dist/<fresh-run>/fresh_external_attempt.json --max-new-tokens 16 --json",
        "```",
        "",
        "## Optional CUDA",
        "",
        "```bash",
        f"crowdtensor public-swarm-v2 evidence-import --gpu-report {args.gpu_report} --max-new-tokens {args.max_new_tokens} --json",
        "```",
        "",
        "CUDA is optional. A CPU-only host must report fail-closed diagnostics rather than pretending GPU pooling is available.",
        "",
        "## Boundaries",
        "",
    ]
    lines.extend(f"- {item}" for item in limitations())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return artifact_entry(path, output_dir, kind="public_swarm_inference_v2_runbook")


def build_common_report(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    mode: str,
    steps: list[dict[str, Any]],
    usable_payload: dict[str, Any],
    preview_payload: dict[str, Any],
    external_real_p2p_payload: dict[str, Any],
    p2p_route_payload: dict[str, Any],
    gpu_payload: dict[str, Any],
    real_p2p_artifact_path: Path | None = None,
) -> dict[str, Any]:
    local = local_p2p_summary(usable_payload, required_tokens=args.max_new_tokens, expected_model_id=args.hf_model_id)
    external = external_summary(
        external_real_p2p_payload,
        required_tokens=args.max_new_tokens,
        fresh_external_report=args.fresh_external_report,
        expected_model_id=args.hf_model_id,
    )
    p2p = p2p_summary(p2p_route_payload, expected_model_id=args.hf_model_id)
    local_route_mode = mode in {MODE_LOCAL, MODE_LOCAL_MODEL_VARIANT}
    local_model_variant = mode == MODE_LOCAL_MODEL_VARIANT
    real_p2p_local_ready = bool(local_route_mode and p2p["ready"] and p2p_route_payload.get("mode") == "local-smoke")
    real_p2p_local_requeue_ready = bool(real_p2p_local_ready and p2p.get("local_stage_requeue_ready") is True)
    gpu = gpu_summary(gpu_payload)
    fresh_attempt = fresh_external_attempt_summary(args.fresh_external_attempt_report)
    performance = performance_summary(preview_payload, local, gpu)
    runbook = write_runbook(args, output_dir)
    batch_requested = len(getattr(args, "prompt_texts_list", []) or parse_prompt_texts_arg(args.prompt_text, args.prompt_texts)) > 1
    ready = bool(
        local["ready"]
        and local["generated_token_count"] >= args.max_new_tokens
        and local["route_source"] == "p2p-discovery"
        and local["kv_cache_ready"]
        and ((not batch_requested) or local["batch_ready"])
        and ((not args.stream_generation) or local["stream_ready"])
        and p2p["ready"]
        and (local_model_variant or external["ready"])
        and local["stage_requeue_rescue_ready"]
        and gpu["fail_closed_ready"]
        and performance["stage_latency_ready"]
        and performance["throughput_summary_ready"]
        and performance["memory_or_vram_summary_ready"]
        and runbook.get("present")
        and ((not local_route_mode) or real_p2p_local_requeue_ready)
    )
    inherited_payloads = [usable_payload, preview_payload, p2p_route_payload, gpu_payload]
    if not local_model_variant:
        inherited_payloads.append(external_real_p2p_payload)
    if local_model_variant:
        codes = local_model_variant_codes(*inherited_payloads) - V2_OWNED_CODES
    else:
        codes = set(diagnosis_codes(*inherited_payloads)) - V2_OWNED_CODES
    codes = drop_unproven_generation_ready_codes(
        codes,
        batch_ready=local["batch_ready"],
        stream_ready=local["stream_ready"],
    )
    codes.update({
        "serve_join_generate_p2p_primary_path",
        "read_only_workload",
        "not_production",
        "not_coordinator_free",
        "not_hivemind_petals_production",
        "not_large_model_serving",
    })
    if local["ready"]:
        codes.add("public_swarm_v2_local_p2p_generate_ready")
    model_match_ready = bool(
        local.get("model", {}).get("compatible")
        and p2p.get("model", {}).get("compatible")
        and (local_model_variant or external.get("model", {}).get("compatible"))
    )
    if model_match_ready:
        codes.add("public_swarm_v2_model_match_ready")
        if local_model_variant:
            codes.add("public_swarm_v2_local_model_variant_model_match_ready")
    else:
        if not local.get("model", {}).get("compatible"):
            codes.add("public_swarm_v2_local_model_mismatch")
        if not local_model_variant and not external.get("model", {}).get("compatible"):
            codes.add("public_swarm_v2_external_model_mismatch")
        if not p2p.get("model", {}).get("compatible"):
            codes.add("public_swarm_v2_p2p_model_mismatch")
    if local["generated_token_count"] >= args.max_new_tokens:
        codes.add("public_swarm_v2_16_token_generation_ready")
    if local["kv_cache_ready"]:
        codes.add("public_swarm_v2_dual_stage_kv_cache_ready")
    if local["batch_ready"]:
        codes.add("public_swarm_v2_batch_generation_ready")
        codes.add("public_swarm_generate_batch_ready")
    if local["stream_ready"]:
        codes.add("public_swarm_v2_stream_generation_ready")
        codes.add("public_swarm_generate_stream_ready")
        if local["stream"].get("endpoint_ready"):
            codes.add("public_swarm_generate_stream_endpoint_ready")
    if p2p["ready"]:
        codes.add("public_swarm_v2_signed_or_real_p2p_ready")
        if local_route_mode and p2p_route_payload.get("mode") == "local-smoke":
            codes.add("public_swarm_v2_real_p2p_local_ready")
            if real_p2p_local_requeue_ready:
                codes.add("public_swarm_v2_real_p2p_local_requeue_ready")
    else:
        codes.add("public_swarm_v2_p2p_lite_fallback_ready")
    if external["ready"]:
        codes.add("public_swarm_v2_external_evidence_ready")
        codes.add("public_swarm_v2_retained_external_evidence_ready")
    if external["accepted_rows_ready"] and not local_model_variant:
        codes.add("public_swarm_v2_external_stage_rows_ready")
    if external["fresh_external_runtime_verified"]:
        codes.add("public_swarm_v2_fresh_external_runtime_verified")
    elif local_model_variant:
        codes.add("public_swarm_v2_external_validation_not_claimed")
    else:
        codes.add("public_swarm_v2_external_fresh_run_action_required")
    if fresh_attempt["attempted"]:
        codes.add("public_swarm_v2_fresh_external_attempted")
    if fresh_attempt["ready"]:
        codes.add("public_swarm_v2_fresh_external_runtime_verified")
    elif fresh_attempt["blocked"]:
        codes.add("public_swarm_v2_fresh_external_blocked_actionable")
    if local["stage_requeue_rescue_ready"] or external["stage_requeue_ready"]:
        codes.add("public_swarm_v2_stage_requeue_rescue_ready")
    if gpu["fail_closed_ready"]:
        codes.add("public_swarm_v2_cuda_optional_fail_closed_ready")
    if performance["stage_latency_ready"]:
        codes.add("stage_latency_ready")
    if performance["throughput_summary_ready"]:
        codes.add("throughput_summary_ready")
    if performance["memory_or_vram_summary_ready"]:
        codes.add("memory_or_vram_summary_ready")
    if runbook.get("present"):
        codes.add("public_swarm_v2_runbook_ready")
    if ready:
        if local_model_variant:
            codes.update({
                "public_swarm_inference_v2_local_model_variant_ready",
                "public_swarm_v2_local_model_variant_ready",
            })
        else:
            codes.update({"public_swarm_inference_v2_ready", "public_swarm_inference_v2_preview_ready"})
    else:
        codes.add("public_swarm_v2_local_model_variant_blocked" if local_model_variant else "public_swarm_inference_v2_blocked")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": mode,
        "output_dir": str(output_dir),
        "public_swarm_v2": {
            "ready": ready,
            "workload_type": WORKLOAD_TYPE,
            "hf_model_id": args.hf_model_id,
            "max_new_tokens": args.max_new_tokens,
            "user_surface": ["p2pd", "serve", "join", "generate"],
            "p2p_discovery_primary_path": True,
            "dual_stage_kv_cache_required": True,
            "signed_or_real_p2p_preferred": True,
            "p2p_lite_fallback_explicit": True,
            "coordinator_authority": True,
            "local_model_variant_only": local_model_variant,
            "external_validation_claimed": not local_model_variant,
            "batch_requested": batch_requested,
        },
        "readiness": {
            "local_p2p_generate": local,
            "external_validation": external,
            "fresh_external_attempt": fresh_attempt,
            "p2p_route_hardening": p2p,
            "real_p2p_local_route_hardening": {
                "ready": real_p2p_local_ready,
                "required": local_route_mode,
                "mode": p2p_route_payload.get("mode"),
                "discovery_backend": (p2p_route_payload.get("p2p") or {}).get("discovery_backend") if isinstance(p2p_route_payload.get("p2p"), dict) else "",
                "generated_token_count": first_int(p2p_route_payload, "generated_token_count"),
                "stage_requeue_ready": real_p2p_local_requeue_ready,
                "stage_requeue_target": p2p.get("local_stage_requeue_target", ""),
            },
            "cuda_optional": gpu,
            "performance": performance,
        },
        "steps": steps,
        "diagnosis_codes": sorted(codes),
        "artifacts": {
            "runbook": runbook,
            "usable_swarm_v1": artifact_entry(
                output_dir / "usable-v1-local" / "usable_swarm_inference.json" if local_route_mode else Path(args.usable_report),
                output_dir,
                kind="usable_swarm_inference_v1",
                schema=USABLE_SCHEMA,
                ok=usable_payload.get("ok") if usable_payload else None,
            ),
            "preview_v04": artifact_entry(Path(args.preview_report), output_dir, kind="public_swarm_preview_v04", schema=PREVIEW_V04_SCHEMA, ok=preview_payload.get("ok") if preview_payload else None),
            "real_p2p": artifact_entry(real_p2p_artifact_path or Path(args.real_p2p_report), output_dir, kind="real_p2p_swarm_inference_core_rc", schema=REAL_P2P_SCHEMA, ok=p2p_route_payload.get("ok") if p2p_route_payload else None),
            "external_real_p2p": artifact_entry(Path(args.real_p2p_report), output_dir, kind="real_p2p_external_validation", schema=REAL_P2P_SCHEMA, ok=external_real_p2p_payload.get("ok") if external_real_p2p_payload else None),
            "gpu_optional": artifact_entry(Path(args.gpu_report), output_dir, kind="public_swarm_gpu_inference_beta", schema=GPU_SCHEMA, ok=gpu_payload.get("ok") if gpu_payload else None),
            "fresh_external_attempt": artifact_entry(Path(args.fresh_external_attempt_report), output_dir, kind="public_swarm_v2_fresh_external_attempt", ok=fresh_attempt.get("ready")) if args.fresh_external_attempt_report else {"kind": "public_swarm_v2_fresh_external_attempt", "present": False},
        },
        "output_request": output_request_summary(),
        "answer_scope": answer_scope_summary(),
        "shareable_summary": shareable_summary(),
        "safety": safety_summary(external_ready=external["ready"], cuda_optional_ready=gpu["fail_closed_ready"]),
        "operator_action": [
            "Run `crowdtensor public-swarm-v2 local --max-new-tokens 16 --json` for the local 16-token gate.",
            "Run or inspect the two-machine/Kaggle commands in PUBLIC_SWARM_INFERENCE_V2.md for fresh external proof diagnostics.",
            "If CUDA is unavailable, preserve the fail-closed diagnostic instead of claiming GPU pooling.",
        ],
        "not_completed": [] if ready else [
            item for item, ok in [
                ("local p2pd/serve/join/generate 16-token proof", local["ready"] and local["generated_token_count"] >= args.max_new_tokens),
                ("local batch generation", (not batch_requested) or local["batch_ready"]),
                ("local safe stream progress", (not args.stream_generation) or local["stream_ready"]),
                ("local persistent dual-stage KV cache reuse", local["kv_cache_ready"]),
                ("local evidence model match", local.get("model", {}).get("compatible")),
                ("fresh local real-P2P route hardening proof", p2p["ready"] and ((not local_route_mode) or p2p_route_payload.get("mode") == "local-smoke")),
                ("fresh local real-P2P stage requeue proof", (not local_route_mode) or real_p2p_local_requeue_ready),
                ("signed or real P2P preferred route evidence", p2p["ready"]),
                ("signed or real P2P evidence model match", p2p.get("model", {}).get("compatible")),
                ("external signed/real P2P validation at token target", local_model_variant or external["ready"]),
                ("external signed/real P2P accepted stage rows", local_model_variant or external["accepted_rows_ready"]),
                ("external signed/real P2P evidence model match", local_model_variant or external.get("model", {}).get("compatible")),
                ("stage requeue/rescue", local["stage_requeue_rescue_ready"] or external["stage_requeue_ready"]),
                ("CUDA optional fail-closed", gpu["fail_closed_ready"]),
                ("latency/throughput/memory evidence", performance["stage_latency_ready"] and performance["throughput_summary_ready"] and performance["memory_or_vram_summary_ready"]),
            ]
            if not ok
        ],
        "limitations": limitations(),
    }
    return persist_report(report, output_dir=output_dir)


def run_usable_local(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "usable_swarm_inference_pack.py"),
        "local",
        "--output-dir",
        str(output_dir),
        "--swarm-id",
        "public-swarm-v2",
        "--public-host",
        args.public_host,
        "--p2p-port",
        str(args.p2p_port),
        "--coordinator-port",
        str(args.coordinator_port),
        "--backend",
        args.backend,
        "--hf-model-id",
        args.hf_model_id,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--startup-timeout",
        str(args.startup_timeout),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--http-timeout",
        str(args.http_timeout),
        "--preview-v04-report",
        args.preview_report,
        "--json",
    ]
    if getattr(args, "prompt_texts_file", ""):
        command.extend(["--prompt-texts-file", args.prompt_texts_file])
    elif args.prompt_texts:
        command.extend(["--prompt-texts", args.prompt_texts])
    else:
        command.extend(["--prompt-text", args.prompt_text])
    if args.stream_generation:
        command.append("--stream-generation")
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    return run_json_step(
        "usable_swarm_v1_local_16_token",
        command,
        runner=runner,
        timeout_seconds=max(args.timeout_seconds, args.startup_timeout, 60.0) + 1500.0,
    )


def run_real_p2p_local(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "real_p2p_swarm_inference_core_rc_pack.py"),
        "local-smoke",
        "--output-dir",
        str(output_dir),
        "--swarm-id",
        "public-swarm-v2-real-p2p",
        "--public-host",
        args.public_host,
        "--p2p-port",
        str(args.real_p2p_port),
        "--coordinator-port",
        str(args.real_p2p_coordinator_port),
        "--libp2p-port",
        str(args.real_p2p_libp2p_port),
        "--backend",
        args.backend,
        "--hf-model-id",
        args.hf_model_id,
        "--prompt-text",
        args.prompt_text,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--startup-timeout",
        str(args.startup_timeout),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--session-queue-timeout",
        str(args.timeout_seconds),
        "--miner-timeout",
        str(args.timeout_seconds),
        "--generate-timeout",
        str(max(args.timeout_seconds, 180.0)),
        "--http-timeout",
        str(args.http_timeout),
        "--discovery-backend",
        args.real_p2p_discovery_backend,
        "--failure-mode",
        "kill-stage1-after-claim",
        "--lease-seconds",
        "2",
        "--victim-compute-seconds",
        "8",
        "--claim-observe-timeout",
        str(max(60.0, args.timeout_seconds)),
        "--requeue-timeout",
        str(max(90.0, args.timeout_seconds)),
        "--kaggle-status-poll-seconds",
        "1",
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    return run_json_step(
        "real_p2p_core_local_route_hardening",
        command,
        runner=runner,
        timeout_seconds=max(args.timeout_seconds, args.startup_timeout, 60.0) + 1800.0,
    )


def build_local(args: argparse.Namespace, *, output_dir: Path, runner: Runner, mode: str = MODE_LOCAL) -> dict[str, Any]:
    step, usable_payload = run_usable_local(args, output_dir=output_dir / "usable-v1-local", runner=runner)
    real_p2p_step, p2p_route_payload = run_real_p2p_local(
        args,
        output_dir=output_dir / "real-p2p-local",
        runner=runner,
    )
    preview_payload = load_json(args.preview_report)
    external_real_p2p_payload = load_json(args.real_p2p_report)
    gpu_payload = load_json(args.gpu_report)
    return build_common_report(
        args,
        output_dir=output_dir,
        mode=mode,
        steps=[step, real_p2p_step],
        usable_payload=usable_payload,
        preview_payload=preview_payload,
        external_real_p2p_payload=external_real_p2p_payload,
        p2p_route_payload=p2p_route_payload,
        gpu_payload=gpu_payload,
        real_p2p_artifact_path=output_dir / "real-p2p-local" / "real_p2p_swarm_inference_core_rc.json",
    )


def build_package(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    usable_payload = load_json(args.usable_report)
    preview_payload = load_json(args.preview_report)
    real_p2p_payload = load_json(args.real_p2p_report)
    gpu_payload = load_json(args.gpu_report)
    report = build_common_report(
        args,
        output_dir=output_dir,
        mode=MODE_PACKAGE,
        steps=[],
        usable_payload=usable_payload,
        preview_payload=preview_payload,
        external_real_p2p_payload=real_p2p_payload,
        p2p_route_payload=real_p2p_payload,
        gpu_payload=gpu_payload,
    )
    report["public_swarm_v2"]["package_only"] = True
    return persist_report(report, output_dir=output_dir)


def build_evidence_import(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    usable_payload = load_json(args.usable_report)
    preview_payload = load_json(args.preview_report)
    real_p2p_payload = load_json(args.real_p2p_report)
    gpu_payload = load_json(args.gpu_report)
    return build_common_report(
        args,
        output_dir=output_dir,
        mode=MODE_EVIDENCE_IMPORT,
        steps=[],
        usable_payload=usable_payload,
        preview_payload=preview_payload,
        external_real_p2p_payload=real_p2p_payload,
        p2p_route_payload=real_p2p_payload,
        gpu_payload=gpu_payload,
    )


def render_markdown(report: dict[str, Any]) -> str:
    v2 = report.get("public_swarm_v2") if isinstance(report.get("public_swarm_v2"), dict) else {}
    readiness = report.get("readiness") if isinstance(report.get("readiness"), dict) else {}
    local = readiness.get("local_p2p_generate") if isinstance(readiness.get("local_p2p_generate"), dict) else {}
    external = readiness.get("external_validation") if isinstance(readiness.get("external_validation"), dict) else {}
    p2p = readiness.get("p2p_route_hardening") if isinstance(readiness.get("p2p_route_hardening"), dict) else {}
    perf = readiness.get("performance") if isinstance(readiness.get("performance"), dict) else {}
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    lines = [
        "# CrowdTensor Public Swarm Inference v2",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- ready: `{v2.get('ready')}`",
        f"- model: `{v2.get('hf_model_id')}`",
        f"- max_new_tokens: `{v2.get('max_new_tokens')}`",
        f"- output_dir: `{report.get('output_dir')}`",
        "",
        "## Readiness",
        "",
        f"- local_p2p_generate_ready: `{local.get('ready')}` tokens=`{local.get('generated_token_count')}` route=`{local.get('route_source')}`",
        f"- dual_stage_kv_cache_ready: `{local.get('kv_cache_ready')}`",
        f"- external_validation_ready: `{external.get('ready')}` generated_tokens=`{external.get('generated_token_count')}` accepted_rows=`{external.get('accepted_rows')}`",
        f"- p2p_route_hardening_ready: `{p2p.get('ready')}` preferred=`{p2p.get('preferred_route')}` route=`{p2p.get('route_source')}`",
        f"- cuda_fail_closed_ready: `{(readiness.get('cuda_optional') or {}).get('fail_closed_ready')}`",
        f"- stage_latency_ready: `{perf.get('stage_latency_ready')}`",
        f"- throughput_summary_ready: `{perf.get('throughput_summary_ready')}`",
        f"- memory_or_vram_summary_ready: `{perf.get('memory_or_vram_summary_ready')}`",
        "",
        "## Output Scope",
        "",
        f"- include output: `{output_request.get('include_output')}`",
        f"- answer scope: `{answer_scope.get('scope_state')}`",
        f"- saved JSON display: `{answer_scope.get('saved_json_display')}`",
        f"- saved Markdown display: `{answer_scope.get('saved_markdown_display')}`",
        f"- shareable: `saved_artifacts={shareable.get('saved_artifacts_public_safe')} raw_prompt_public={shareable.get('raw_prompt_public')} raw_generated_text_public={shareable.get('raw_generated_text_public')} generated_token_ids_public={shareable.get('generated_token_ids_public')} answer_scope_state={shareable.get('answer_scope_state')} local_answer_terminal_only={shareable.get('local_answer_terminal_only')}`",
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`",
        "",
        "## Not Completed",
        "",
    ]
    not_completed = report.get("not_completed") or []
    lines.extend(f"- {item}" for item in not_completed) if not_completed else lines.append("- none")
    lines.extend(["", "## Boundaries", ""])
    lines.extend(f"- {item}" for item in report.get("limitations") or [])
    return "\n".join(lines) + "\n"


def validate_public_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    encoded = json.dumps(report, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment and fragment in encoded:
            errors.append(f"sensitive_fragment:{fragment}")
    for path in public_leak_paths(report):
        if path.endswith(".prompt_hash") or ".safety." in path:
            continue
        errors.append(f"public_leak:{path}")
    return sorted(set(errors))


def persist_report(report: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    report = support_bundle.sanitize(redact_values(report))
    errors = validate_public_report(report)
    if errors:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"public_report_safety_failed"})
        report["safety_errors"] = errors
        if isinstance(report.get("public_swarm_v2"), dict):
            report["public_swarm_v2"]["ready"] = False
    artifacts = report.setdefault("artifacts", {})
    artifacts["public_swarm_inference_v2_json"] = {
        "kind": "public_swarm_inference_v2",
        "path": "public_swarm_inference_v2.json",
        "present": True,
        "schema": SCHEMA,
        "ok": report.get("ok"),
    }
    artifacts["public_swarm_inference_v2_markdown"] = {
        "kind": "public_swarm_inference_v2_markdown",
        "path": "public_swarm_inference_v2.md",
        "present": True,
    }
    write_json(output_dir / "public_swarm_inference_v2.json", report)
    (output_dir / "public_swarm_inference_v2.md").write_text(render_markdown(report), encoding="utf-8")
    bundle = support_bundle.sanitize({
        "schema": SUPPORT_SCHEMA,
        "ok": report.get("ok"),
        "diagnosis_codes": report.get("diagnosis_codes"),
        "public_swarm_v2": report.get("public_swarm_v2"),
        "readiness": report.get("readiness"),
        "output_request": report.get("output_request"),
        "answer_scope": report.get("answer_scope"),
        "shareable_summary": report.get("shareable_summary"),
        "safety": report.get("safety"),
        "limitations": report.get("limitations"),
        "not_completed": report.get("not_completed"),
    })
    write_json(output_dir / "support_bundle.json", bundle)
    report["artifacts"]["support_bundle_json"] = {
        "kind": "public_swarm_inference_v2_support_bundle",
        "path": "support_bundle.json",
        "present": True,
        "schema": SUPPORT_SCHEMA,
        "ok": bundle.get("ok"),
    }
    write_json(output_dir / "public_swarm_inference_v2.json", report)
    return report


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == MODE_PACKAGE:
        return build_package(args, output_dir=output_dir)
    if args.mode == MODE_EVIDENCE_IMPORT:
        return build_evidence_import(args, output_dir=output_dir)
    return build_local(args, output_dir=output_dir, runner=runner, mode=args.mode)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Public Swarm Inference v2 evidence.")
    parser.add_argument("mode", choices=MODES, nargs="?", default=MODE_LOCAL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--usable-report", default=DEFAULT_USABLE_REPORT)
    parser.add_argument("--preview-report", default=DEFAULT_PREVIEW_REPORT)
    parser.add_argument("--real-p2p-report", default=DEFAULT_REAL_P2P_REPORT)
    parser.add_argument("--gpu-report", default=DEFAULT_GPU_REPORT)
    parser.add_argument("--fresh-external-report", action="store_true")
    parser.add_argument("--fresh-external-attempt-report", default="")
    parser.add_argument("--public-host", default="127.0.0.1")
    parser.add_argument("--p2p-port", type=int, default=9888)
    parser.add_argument("--coordinator-port", type=int, default=9889)
    parser.add_argument("--real-p2p-port", type=int, default=DEFAULT_REAL_P2P_LOCAL_P2P_PORT)
    parser.add_argument("--real-p2p-coordinator-port", type=int, default=DEFAULT_REAL_P2P_LOCAL_COORDINATOR_PORT)
    parser.add_argument("--real-p2p-libp2p-port", type=int, default=0)
    parser.add_argument("--real-p2p-discovery-backend", choices=sorted(DISCOVERY_BACKENDS), default=DEFAULT_REAL_P2P_DISCOVERY_BACKEND)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--hf-model-id", default=DEFAULT_HF_MODEL_ID)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-texts", default="")
    parser.add_argument("--prompt-texts-file", default="")
    parser.add_argument("--stream-generation", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=420.0)
    parser.add_argument("--http-timeout", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 8 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 8 and 32")
    if args.prompt_texts and args.prompt_texts_file:
        raise SystemExit("public_swarm_inference_v2 accepts either --prompt-texts or --prompt-texts-file, not both")
    try:
        if args.prompt_texts_file:
            args.prompt_texts_list = read_prompt_texts_file(args.prompt_texts_file)
        else:
            args.prompt_texts_list = parse_prompt_texts_arg(args.prompt_text, args.prompt_texts)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.p2p_port < 1 or args.coordinator_port < 1:
        raise SystemExit("--p2p-port and --coordinator-port must be positive")
    if args.real_p2p_port < 1 or args.real_p2p_coordinator_port < 1:
        raise SystemExit("--real-p2p-port and --real-p2p-coordinator-port must be positive")
    if args.real_p2p_libp2p_port < 0:
        raise SystemExit("--real-p2p-libp2p-port must be non-negative")
    for name in ["startup_timeout", "timeout_seconds", "http_timeout"]:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        v2 = report.get("public_swarm_v2") if isinstance(report.get("public_swarm_v2"), dict) else {}
        output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
        answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
        shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
        print("CrowdTensor Public Swarm Inference v2")
        print(f"  ok: {report.get('ok')}")
        print(f"  mode: {report.get('mode')}")
        print(f"  ready: {v2.get('ready')}")
        if output_request:
            print(f"  output_request: {output_request_text(output_request)}")
        if answer_scope:
            print(f"  answer_scope: {answer_scope_text(answer_scope)}")
        if shareable:
            print(f"  shareable: {shareable_summary_text(shareable)}")
        print(f"  output: {report.get('output_dir')}")
        print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
