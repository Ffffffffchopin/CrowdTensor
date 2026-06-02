#!/usr/bin/env python3
"""Build the Usable Swarm Inference v1 product evidence artifact.

This is the user-path gate for CrowdTensor swarm inference.  It runs the
product P2P route instead of asking users to understand the lower-level proof
stack: p2pd, serve --p2p, join --p2p stage0/stage1, and generate --p2p.
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
from product_swarm_mvp_check import parse_prompt_texts_arg  # noqa: E402
from crowdtensor.session_protocol import public_leak_paths  # noqa: E402


SCHEMA = "usable_swarm_inference_v1"
SUPPORT_SCHEMA = "usable_swarm_inference_support_bundle_v1"
P2P_V06_SCHEMA = "p2p_swarm_inference_v06_v1"
WORKLOAD_TYPE = "real_llm_sharded_infer"

MODE_LOCAL = "local"
MODE_PACKAGE = "package"
MODE_EVIDENCE_IMPORT = "evidence-import"
MODES = [MODE_LOCAL, MODE_PACKAGE, MODE_EVIDENCE_IMPORT]

DEFAULT_OUTPUT_DIR = "dist/usable-swarm-inference-v1"
DEFAULT_P2P_REPORT = "dist/goal-final-infer-p2p-v06-16tok-kv-cache-20260601/p2p_swarm_inference_v06.json"
DEFAULT_PROMPT = "CrowdTensor usable swarm inference"
DEFAULT_HF_MODEL_ID = "sshleifer/tiny-gpt2"

INHERITED_CODE_ALLOWLIST = {
    "p2pd_daemon_ready",
    "local_three_process_p2p_discovery_ready",
    "p2p_stage_discovery_ready",
    "p2p_generate_route_ready",
    "p2p_stage_rescue_ready",
    "p2p_real_generate_ready",
    "p2p_real_generate_kv_cache_ready",
    "p2p_real_stage_rescue_ready",
    "p2p_rescue_generation_completed",
    "stage0_rescue_generation_completed",
    "stage1_rescue_generation_completed",
    "coordinator_to_p2p_transition_ready",
    "coordinator_result_fallback_ready",
    "real_llm_stage0_kv_cache_v1_ready",
    "real_llm_stage1_kv_cache_v1_ready",
    "stage0_kv_cache_hits_ready",
    "stage1_kv_cache_hits_ready",
    "tiny_gpt2_multi_token_ready",
    "distinct_stage_miners",
    "stage_assignment_valid",
    "hf_dependencies_missing",
    "p2p_real_generate_hf_runtime_missing",
    "p2p_real_stage_rescue_hf_runtime_missing",
}

BATCH_READY_CODES = {
    "p2p_real_generate_batch_ready",
    "public_swarm_generate_batch_ready",
}

STREAM_READY_CODES = {
    "p2p_real_generate_stream_ready",
    "public_swarm_generate_stream_ready",
    "public_swarm_generate_stream_endpoint_ready",
}

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
        for value in item.values():
            if isinstance(value, dict):
                pending.append(value)
            elif isinstance(value, list):
                pending.extend(entry for entry in value if isinstance(entry, dict))
    return sorted(codes)


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


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def model_compatibility(payload: dict[str, Any], expected_model_id: str) -> dict[str, Any]:
    observed = first_string_value(payload, "hf_model_id")
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


def generation_summary(payload: dict[str, Any]) -> dict[str, Any]:
    p2p = payload.get("p2p") if isinstance(payload.get("p2p"), dict) else {}
    generation = p2p.get("generation") if isinstance(p2p.get("generation"), dict) else {}
    if not generation:
        local = payload.get("payload_summaries", {}).get("local_p2p_discovery", {}) if isinstance(payload.get("payload_summaries"), dict) else {}
        real = local.get("real_generate_probe") if isinstance(local.get("real_generate_probe"), dict) else {}
        generation = real.get("generation") if isinstance(real.get("generation"), dict) else {}
    return {
        "generated_token_count": generation.get("generated_token_count"),
        "max_new_tokens": generation.get("max_new_tokens"),
        "generated_text_hash": generation.get("generated_text_hash"),
        "multi_token_generation_ready": generation.get("multi_token_generation_ready"),
        "decoded_tokens_match": generation.get("decoded_tokens_match"),
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }


def batch_summary(payload: dict[str, Any]) -> dict[str, Any]:
    p2p = payload.get("p2p") if isinstance(payload.get("p2p"), dict) else {}
    batch = p2p.get("batch") if isinstance(p2p.get("batch"), dict) else {}
    if not batch:
        local = payload.get("payload_summaries", {}).get("local_p2p_discovery", {}) if isinstance(payload.get("payload_summaries"), dict) else {}
        real = local.get("real_generate_probe") if isinstance(local.get("real_generate_probe"), dict) else {}
        batch = real.get("batch") if isinstance(real.get("batch"), dict) else {}
    if not batch:
        return {
            "enabled": False,
            "batch_generation_ready": False,
            "raw_prompts_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        }
    safe_results: list[dict[str, Any]] = []
    for item in batch.get("results") or []:
        if not isinstance(item, dict):
            continue
        safe_results.append({
            "request_id": item.get("request_id"),
            "prompt_hash": item.get("prompt_hash"),
            "generated_token_count": int(item.get("generated_token_count") or 0),
            "max_new_tokens": item.get("max_new_tokens"),
            "generated_text_hash": item.get("generated_text_hash"),
            "decoded_tokens_match": item.get("decoded_tokens_match"),
            "multi_token_generation_ready": bool(item.get("multi_token_generation_ready")),
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
        })
    expected_request_count = int(batch.get("expected_request_count") or batch.get("request_count") or 0)
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
        "expected_request_count": expected_request_count,
        "observed_request_count": int(batch.get("observed_request_count") or 0),
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


def safe_per_request_progress(progress: dict[str, Any]) -> list[dict[str, Any]]:
    raw_rows = progress.get("per_request_progress") if isinstance(progress.get("per_request_progress"), list) else []
    rows: list[dict[str, Any]] = []
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
    if not stream.get("stream_generation_ready"):
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


def inherited_ready_codes(codes: set[str], *, batch_ready: bool, stream_ready: bool) -> list[str]:
    allowed = set(INHERITED_CODE_ALLOWLIST)
    if batch_ready:
        allowed.update(BATCH_READY_CODES)
    if stream_ready:
        allowed.update(STREAM_READY_CODES)
    return sorted(code for code in codes if code in allowed)


def stream_summary(payload: dict[str, Any]) -> dict[str, Any]:
    p2p = payload.get("p2p") if isinstance(payload.get("p2p"), dict) else {}
    stream = p2p.get("stream") if isinstance(p2p.get("stream"), dict) else {}
    if not stream:
        local = payload.get("payload_summaries", {}).get("local_p2p_discovery", {}) if isinstance(payload.get("payload_summaries"), dict) else {}
        real = local.get("real_generate_probe") if isinstance(local.get("real_generate_probe"), dict) else {}
        stream = real.get("stream") if isinstance(real.get("stream"), dict) else {}
    if not stream:
        return {
            "enabled": False,
            "requested": False,
            "event_count": 0,
            "stream_generation_ready": False,
            "endpoint_ready": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
    }
    progress = stream.get("progress") if isinstance(stream.get("progress"), dict) else {}
    events = stream.get("events") if isinstance(stream.get("events"), list) else []
    observed_counts = [
        safe_int(value)
        for value in (progress.get("observed_token_counts") or [])
        if safe_int(value, -1) >= 0
    ] if isinstance(progress.get("observed_token_counts"), list) else []
    return {
        "enabled": bool(stream.get("enabled")),
        "requested": bool(stream.get("requested") or stream.get("enabled")),
        "event_count": safe_int(stream.get("event_count"), len(events)),
        "source": stream.get("source"),
        "endpoint_ready": bool(stream.get("endpoint_ready")),
        "stream_generation_ready": bool(stream.get("stream_generation_ready")),
        "progress": {
            "stream_progress_complete": bool(progress.get("stream_progress_complete")),
            "all_token_events_ready": bool(progress.get("all_token_events_ready")),
            "monotonic_progress": bool(progress.get("monotonic_progress")),
            "expected_request_count": safe_int(progress.get("expected_request_count"), 1),
            "per_request_progress": safe_per_request_progress(progress),
            "per_request_progress_complete": bool(progress.get("per_request_progress_complete")),
            "per_request_monotonic_progress": bool(progress.get("per_request_monotonic_progress")),
            "observed_token_counts": observed_counts,
            "max_observed_token_count": safe_int(progress.get("max_observed_token_count")),
            "max_new_tokens": progress.get("max_new_tokens"),
            "source": progress.get("source") or stream.get("source"),
        },
        "events": list(events),
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
    }


def real_generate_probe(payload: dict[str, Any]) -> dict[str, Any]:
    local = payload.get("payload_summaries", {}).get("local_p2p_discovery", {}) if isinstance(payload.get("payload_summaries"), dict) else {}
    real = local.get("real_generate_probe") if isinstance(local.get("real_generate_probe"), dict) else {}
    return real if isinstance(real, dict) else {}


def p2p_summary(payload: dict[str, Any], *, required_tokens: int, expected_model_id: str = DEFAULT_HF_MODEL_ID) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    model = model_compatibility(payload, expected_model_id)
    p2p = payload.get("p2p") if isinstance(payload.get("p2p"), dict) else {}
    route = p2p.get("generate_route") if isinstance(p2p.get("generate_route"), dict) else {}
    generation = generation_summary(payload)
    batch = batch_summary(payload)
    stream = stream_summary(payload)
    real_probe = real_generate_probe(payload)
    stage_assignment = p2p.get("stage_assignment") if isinstance(p2p.get("stage_assignment"), dict) else {}
    if not stage_assignment:
        stage_assignment = real_probe.get("stage_assignment") if isinstance(real_probe.get("stage_assignment"), dict) else {}
    ledger = p2p.get("ledger") if isinstance(p2p.get("ledger"), dict) else {}
    if not ledger:
        ledger = real_probe.get("ledger") if isinstance(real_probe.get("ledger"), dict) else {}
    kv_cache = p2p.get("kv_cache") if isinstance(p2p.get("kv_cache"), dict) else {}
    if not kv_cache:
        kv_cache = real_probe.get("kv_cache") if isinstance(real_probe.get("kv_cache"), dict) else {}
    generated_count = int(generation.get("generated_token_count") or 0)
    max_tokens = int(generation.get("max_new_tokens") or required_tokens)
    route_ready = bool(route.get("usable_now") and route.get("route_source") == "p2p-discovery")
    p2p_counts_ready = bool(
        int(p2p.get("coordinator_peer_count") or 0) >= 1
        and int(p2p.get("stage0_peer_count") or 0) >= 1
        and int(p2p.get("stage1_peer_count") or 0) >= 1
    )
    real_generate_ready = bool(p2p.get("real_generate_ready") or "p2p_real_generate_ready" in codes)
    batch_ready = bool(batch.get("enabled") and batch.get("batch_generation_ready") is True)
    stream_ready = stream_evidence_ready(stream, batch)
    kv_cache_ready = bool(kv_cache.get("ready") or "p2p_real_generate_kv_cache_ready" in codes)
    real_stage_rescue_ready = bool(p2p.get("real_stage_rescue_ready") or "p2p_real_stage_rescue_ready" in codes)
    stage_rescue_ready = bool(p2p.get("stage_rescue_ready") or "p2p_stage_rescue_ready" in codes)
    distinct_stage_miners = bool(
        stage_assignment.get("distinct_stage_miners")
        or "distinct_stage_miners" in codes
        or "stage_assignment_valid" in codes
    )
    token_ready = bool(generated_count >= required_tokens and max_tokens >= required_tokens)
    accepted_rows = int(ledger.get("accepted_rows") or 0)
    accepted_rows_ready = bool(accepted_rows >= required_tokens * 2)
    expected_codes = {
        "p2pd_daemon_ready",
        "local_three_process_p2p_discovery_ready",
        "p2p_stage_discovery_ready",
        "p2p_generate_route_ready",
        "coordinator_to_p2p_transition_ready",
        "coordinator_result_fallback_ready",
    }
    ready = bool(
        route_ready
        and p2p_counts_ready
        and real_generate_ready
        and kv_cache_ready
        and token_ready
        and accepted_rows_ready
        and distinct_stage_miners
        and stage_rescue_ready
        and real_stage_rescue_ready
        and model["compatible"]
    )
    source_gate_ok = payload.get("ok") is True
    usable_evidence_source = "source_gate" if source_gate_ok else "nested_goal_evidence"
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "ready": ready,
        "model": model,
        "source_gate_ok": source_gate_ok,
        "usable_evidence_source": usable_evidence_source,
        "mode": payload.get("mode"),
        "missing_codes": sorted(expected_codes - codes),
        "route_ready": route_ready,
        "route_source": route.get("route_source"),
        "p2p_counts_ready": p2p_counts_ready,
        "route": route,
        "p2p_url": p2p.get("p2p_url"),
        "catalog_peer_count": p2p.get("catalog_peer_count"),
        "coordinator_peer_count": p2p.get("coordinator_peer_count"),
        "stage0_peer_count": p2p.get("stage0_peer_count"),
        "stage1_peer_count": p2p.get("stage1_peer_count"),
        "real_generate_ready": real_generate_ready,
        "batch_ready": batch_ready,
        "stream_ready": stream_ready,
        "kv_cache_ready": kv_cache_ready,
        "kv_cache": kv_cache,
        "stage_rescue_ready": stage_rescue_ready,
        "real_stage_rescue_ready": real_stage_rescue_ready,
        "distinct_stage_miners": distinct_stage_miners,
        "generated_token_count": generated_count,
        "max_new_tokens": max_tokens,
        "generation_target_ready": token_ready,
        "accepted_rows": accepted_rows,
        "accepted_rows_ready": accepted_rows_ready,
        "stage_assignment": stage_assignment,
        "ledger": ledger,
        "generation": generation,
        "batch": batch,
        "stream": stream,
        "diagnosis_codes": inherited_ready_codes(codes, batch_ready=batch_ready, stream_ready=stream_ready),
    }


def run_p2p_local(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "p2p_swarm_inference_v06_pack.py"),
        "local-smoke",
        "--output-dir",
        str(output_dir),
        "--swarm-id",
        args.swarm_id,
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
        "--prompt-text",
        args.prompt_text,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--startup-timeout",
        str(args.startup_timeout),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--http-timeout",
        str(args.http_timeout),
        "--preview-v04-report",
        args.preview_v04_report,
        "--product-mvp-report",
        args.product_mvp_report,
        "--optional-model-report",
        args.optional_model_report,
        "--json",
    ]
    if args.prompt_texts:
        command.extend(["--prompt-texts", args.prompt_texts])
    if args.stream_generation:
        command.append("--stream-generation")
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    return run_json_step(
        "p2p_swarm_inference_v06_local_smoke",
        command,
        runner=runner,
        timeout_seconds=max(args.timeout_seconds, 60.0) + 900.0,
    )


def safety_block() -> dict[str, Any]:
    return {
        "coordinator_backed_task_execution": True,
        "p2p_discovery_primary_path": True,
        "read_only_workload": WORKLOAD_TYPE,
        "cpu_default": True,
        "cuda_optional": True,
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


def limitations() -> list[str]:
    return [
        "Usable Swarm Inference v1 is a user-facing, Coordinator-backed product path.",
        "p2pd discovery is primary for routing, but Coordinator still owns session creation, leases, validation, and result ledgers.",
        "The default model path is tiny/small Hugging Face real weights; it is not large-model throughput serving.",
        "CPU is the default; CUDA remains optional and fail-closed.",
        "No production NAT/relay fabric, decentralized security, payments, staking, billing, or anti-Sybil network is included.",
    ]


def write_runbook(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    path = output_dir / "USABLE_SWARM_INFERENCE.md"
    p2p = f"http://{args.public_host}:{args.p2p_port}"
    public_host_placeholder = "COORDINATOR_PUBLIC_HOST"
    public_p2p = f"http://${public_host_placeholder}:{args.p2p_port}"
    lines = [
        "# CrowdTensor Usable Swarm Inference v1",
        "",
        "This is the ordinary user path.  It runs P2P discovery for route lookup while the Coordinator remains the execution authority.",
        "",
        "## Install",
        "",
        "```bash",
        "python -m pip install -e '.[hf]'",
        "read -r -s -p 'Admin token: ' CROWDTENSOR_ADMIN_TOKEN; echo",
        "read -r -s -p 'Miner token: ' CROWDTENSOR_MINER_TOKEN; echo",
        "export CROWDTENSOR_ADMIN_TOKEN CROWDTENSOR_MINER_TOKEN",
        "```",
        "",
        "## Local Five-Process Smoke",
        "",
        "Use this on one machine first. It proves the exact product path before exposing any port.",
        "",
        "Terminal 1:",
        "",
        "```bash",
        f"crowdtensor p2pd --host 127.0.0.1 --port {args.p2p_port} --swarm-id {args.swarm_id} --run",
        "```",
        "",
        "Terminal 2:",
        "",
        "```bash",
        f"crowdtensor serve --p2p --peer-bootstrap {p2p} --swarm-id {args.swarm_id} --public-host {args.public_host} --port {args.coordinator_port} --admin-token \"$CROWDTENSOR_ADMIN_TOKEN\" --miner-token \"$CROWDTENSOR_MINER_TOKEN\" --run",
        "```",
        "",
        "Terminal 3:",
        "",
        "```bash",
        f"crowdtensor join --p2p --peer-bootstrap {p2p} --swarm-id {args.swarm_id} --stage stage0 --miner-id usable-stage0 --miner-token \"$CROWDTENSOR_MINER_TOKEN\" --run",
        "```",
        "",
        "Terminal 4:",
        "",
        "```bash",
        f"crowdtensor join --p2p --peer-bootstrap {p2p} --swarm-id {args.swarm_id} --stage stage1 --miner-id usable-stage1 --miner-token \"$CROWDTENSOR_MINER_TOKEN\" --run",
        "```",
        "",
        "Terminal 5:",
        "",
        "```bash",
        f"crowdtensor generate --p2p --peer-bootstrap {p2p} --prompt \"{args.prompt_text}\" --max-new-tokens {args.max_new_tokens}",
        "```",
        "",
        "## Two-Machine/Public Rehearsal",
        "",
        "Use this only on a trusted network, VPN, SSH tunnel, or temporary firewall allowlist. Rotate tokens after public HTTP tests.",
        "",
        "Coordinator host:",
        "",
        "```bash",
        f"export {public_host_placeholder}='<public-host-or-vpn-hostname>'",
        "read -r -s -p 'Admin token: ' CROWDTENSOR_ADMIN_TOKEN; echo",
        "read -r -s -p 'Miner token: ' CROWDTENSOR_MINER_TOKEN; echo",
        "export CROWDTENSOR_ADMIN_TOKEN CROWDTENSOR_MINER_TOKEN",
        f"crowdtensor p2pd --host 0.0.0.0 --port {args.p2p_port} --swarm-id {args.swarm_id} --run",
        "```",
        "",
        "Coordinator service, on the same host:",
        "",
        "```bash",
        f"crowdtensor serve --p2p --peer-bootstrap {public_p2p} --swarm-id {args.swarm_id} --bind-host 0.0.0.0 --public-host \"${public_host_placeholder}\" --port {args.coordinator_port} --admin-token \"$CROWDTENSOR_ADMIN_TOKEN\" --miner-token \"$CROWDTENSOR_MINER_TOKEN\" --i-understand-public-bind --run",
        "```",
        "",
        "Stage 0 Miner host:",
        "",
        "```bash",
        f"export {public_host_placeholder}='<public-host-or-vpn-hostname>'",
        "read -r -s -p 'Miner token: ' CROWDTENSOR_MINER_TOKEN; echo",
        "export CROWDTENSOR_MINER_TOKEN",
        f"crowdtensor join --p2p --peer-bootstrap {public_p2p} --swarm-id {args.swarm_id} --stage stage0 --miner-id \"$(hostname)-stage0\" --miner-token \"$CROWDTENSOR_MINER_TOKEN\" --run",
        "```",
        "",
        "Stage 1 Miner host:",
        "",
        "```bash",
        f"export {public_host_placeholder}='<public-host-or-vpn-hostname>'",
        "read -r -s -p 'Miner token: ' CROWDTENSOR_MINER_TOKEN; echo",
        "export CROWDTENSOR_MINER_TOKEN",
        f"crowdtensor join --p2p --peer-bootstrap {public_p2p} --swarm-id {args.swarm_id} --stage stage1 --miner-id \"$(hostname)-stage1\" --miner-token \"$CROWDTENSOR_MINER_TOKEN\" --run",
        "```",
        "",
        "Operator laptop:",
        "",
        "```bash",
        f"export {public_host_placeholder}='<public-host-or-vpn-hostname>'",
        "read -r -s -p 'Admin token: ' CROWDTENSOR_ADMIN_TOKEN; echo",
        "export CROWDTENSOR_ADMIN_TOKEN",
        f"crowdtensor generate --p2p --peer-bootstrap {public_p2p} --admin-token \"$CROWDTENSOR_ADMIN_TOKEN\" --prompt \"{args.prompt_text}\" --max-new-tokens {args.max_new_tokens}",
        "```",
        "",
        "The two Miner commands must use different `--miner-id` values. If either Miner dies after claiming a stage task, the Coordinator lease timeout should requeue that stage for a replacement Miner.",
        "",
        "## Maintainer Gate",
        "",
        "```bash",
        f"crowdtensor usable-swarm local --max-new-tokens {args.max_new_tokens} --json",
        "```",
        "",
        "## Boundaries",
        "",
    ]
    lines.extend(f"- {item}" for item in limitations())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return artifact_entry(path, output_dir, kind="usable_swarm_inference_runbook")


def build_common_report(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    p2p_payload: dict[str, Any],
    steps: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    runbook = write_runbook(args, output_dir)
    p2p = p2p_summary(p2p_payload, required_tokens=args.max_new_tokens, expected_model_id=args.hf_model_id)
    ready = bool(p2p["ready"] and runbook.get("present"))
    codes = set(p2p.get("diagnosis_codes") or [])
    if p2p["route_ready"]:
        codes.add("usable_p2p_route_ready")
    if p2p["real_generate_ready"]:
        codes.add("usable_real_llm_generate_ready")
    if p2p["batch_ready"]:
        codes.add("usable_real_llm_batch_ready")
        codes.add("public_swarm_generate_batch_ready")
    if p2p["stream_ready"]:
        codes.add("usable_real_llm_stream_ready")
        codes.add("public_swarm_generate_stream_ready")
        if p2p["stream"].get("endpoint_ready"):
            codes.add("public_swarm_generate_stream_endpoint_ready")
    if p2p["kv_cache_ready"]:
        codes.add("usable_real_llm_kv_cache_ready")
    if p2p["generation_target_ready"]:
        codes.add("usable_multi_token_generation_ready")
    if p2p["distinct_stage_miners"]:
        codes.add("usable_distinct_stage_miners_ready")
    if p2p["stage_rescue_ready"] and p2p["real_stage_rescue_ready"]:
        codes.add("usable_stage_requeue_rescue_ready")
    if p2p.get("model", {}).get("compatible"):
        codes.add("usable_swarm_model_match_ready")
    else:
        codes.add("usable_swarm_model_mismatch")
    if runbook.get("present"):
        codes.add("usable_swarm_runbook_ready")
    codes.update({
        "serve_join_generate_p2p_primary_path",
        "read_only_workload",
        "not_production",
        "not_coordinator_free",
        "not_hivemind_petals_production",
        "not_large_model_serving",
    })
    if ready:
        codes.update({
            "usable_swarm_inference_ready",
            "usable_swarm_inference_v1_ready",
        })
    else:
        codes.add("usable_swarm_inference_blocked")
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": mode,
        "output_dir": str(output_dir),
        "usable_swarm": {
            "ready": ready,
            "workload_type": WORKLOAD_TYPE,
            "hf_model_id": args.hf_model_id,
            "max_new_tokens": args.max_new_tokens,
            "user_surface": ["p2pd", "serve", "join", "generate"],
            "p2p_discovery_primary_path": True,
            "cpu_default_ready": True,
            "cuda_optional": True,
            "coordinator_authority": True,
        },
        "readiness": {
            "p2p_product_path": p2p,
        },
        "steps": steps,
        "diagnosis_codes": sorted(codes),
        "artifacts": {
            "runbook": runbook,
            "p2p_local_report": artifact_entry(
                Path(args.p2p_report) if mode == MODE_EVIDENCE_IMPORT else output_dir / "p2p-local" / "p2p_swarm_inference_v06.json",
                output_dir,
                kind="p2p_swarm_inference_v06",
                schema=P2P_V06_SCHEMA,
                ok=p2p_payload.get("ok") if p2p_payload else None,
            ),
        },
        "safety": safety_block(),
        "limitations": limitations(),
        "operator_action": [
            "Use `crowdtensor p2pd`, `serve --p2p`, `join --p2p --stage stage0`, `join --p2p --stage stage1`, and `generate --p2p` as the ordinary user path.",
            "Use JSON/Markdown artifacts for issue reports; raw prompts, generated text, token ids, activations, and credentials are excluded.",
            "When imported from an older aggregate report, Usable v1 readiness is based on direct nested goal evidence rather than requiring every legacy aggregate gate to be ready.",
        ],
        "not_completed": [] if ready else [
            item for item, ok in [
                ("local p2pd + serve + stage0 + stage1 + generate --p2p", p2p["route_ready"]),
                ("real small HF multi-token generation", p2p["real_generate_ready"] and p2p["generation_target_ready"]),
                ("safe stream progress", (not args.stream_generation) or p2p["stream_ready"]),
                ("real LLM persistent dual-stage KV cache reuse", p2p["kv_cache_ready"]),
                ("imported P2P evidence model match", p2p.get("model", {}).get("compatible")),
                ("distinct stage miners", p2p["distinct_stage_miners"]),
                ("stage requeue/rescue", p2p["stage_rescue_ready"] and p2p["real_stage_rescue_ready"]),
                ("user runbook", bool(runbook.get("present"))),
            ]
            if not ok
        ],
    }
    return persist_report(report, output_dir=output_dir)


def build_local(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    step, payload = run_p2p_local(args, output_dir=output_dir / "p2p-local", runner=runner)
    return build_common_report(args, output_dir=output_dir, p2p_payload=payload, steps=[step], mode=MODE_LOCAL)


def build_package(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    runbook = write_runbook(args, output_dir)
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "mode": MODE_PACKAGE,
        "output_dir": str(output_dir),
        "usable_swarm": {
            "ready": True,
            "package_only": True,
            "workload_type": WORKLOAD_TYPE,
            "hf_model_id": args.hf_model_id,
            "max_new_tokens": args.max_new_tokens,
            "user_surface": ["p2pd", "serve", "join", "generate"],
            "p2p_discovery_primary_path": True,
        },
        "diagnosis_codes": [
            "usable_swarm_package_ready",
            "usable_swarm_runbook_ready",
            "serve_join_generate_p2p_primary_path",
            "read_only_workload",
            "not_production",
            "not_coordinator_free",
            "not_large_model_serving",
        ],
        "artifacts": {"runbook": runbook},
        "safety": safety_block(),
        "limitations": limitations(),
        "operator_action": ["Run local mode to execute the real local P2P product path."],
        "not_completed": [],
    }
    return persist_report(report, output_dir=output_dir)


def build_evidence_import(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    payload = load_json(args.p2p_report)
    return build_common_report(args, output_dir=output_dir, p2p_payload=payload, steps=[], mode=MODE_EVIDENCE_IMPORT)


def render_markdown(report: dict[str, Any]) -> str:
    usable = report.get("usable_swarm") if isinstance(report.get("usable_swarm"), dict) else {}
    readiness = report.get("readiness") if isinstance(report.get("readiness"), dict) else {}
    p2p = readiness.get("p2p_product_path") if isinstance(readiness.get("p2p_product_path"), dict) else {}
    lines = [
        "# CrowdTensor Usable Swarm Inference v1",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- ready: `{usable.get('ready')}`",
        f"- model: `{usable.get('hf_model_id')}`",
        f"- max_new_tokens: `{usable.get('max_new_tokens')}`",
        f"- output_dir: `{report.get('output_dir')}`",
        "",
        "## User Path",
        "",
        "`p2pd` -> `serve --p2p` -> `join --p2p --stage stage0` -> `join --p2p --stage stage1` -> `generate --p2p`",
        "",
        "## Readiness",
        "",
        f"- p2p route ready: `{p2p.get('route_ready')}`",
        f"- real generate ready: `{p2p.get('real_generate_ready')}`",
        f"- generated tokens: `{p2p.get('generated_token_count')}/{p2p.get('max_new_tokens')}`",
        f"- distinct stage miners: `{p2p.get('distinct_stage_miners')}`",
        f"- stage rescue ready: `{p2p.get('stage_rescue_ready')}`",
        f"- real stage rescue ready: `{p2p.get('real_stage_rescue_ready')}`",
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`",
        "",
        "## Boundaries",
        "",
    ]
    for item in report.get("limitations") or []:
        lines.append(f"- {item}")
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
        if isinstance(report.get("usable_swarm"), dict):
            report["usable_swarm"]["ready"] = False
    artifacts = report.setdefault("artifacts", {})
    artifacts.setdefault("usable_swarm_inference_json", {
        "kind": "usable_swarm_inference",
        "path": "usable_swarm_inference.json",
        "present": True,
        "schema": SCHEMA,
        "ok": report.get("ok"),
    })
    artifacts.setdefault("usable_swarm_inference_markdown", {
        "kind": "usable_swarm_inference_markdown",
        "path": "usable_swarm_inference.md",
        "present": True,
    })
    write_json(output_dir / "usable_swarm_inference.json", report)
    (output_dir / "usable_swarm_inference.md").write_text(render_markdown(report), encoding="utf-8")
    bundle = support_bundle.sanitize({
        "schema": SUPPORT_SCHEMA,
        "ok": report.get("ok"),
        "diagnosis_codes": report.get("diagnosis_codes"),
        "usable_swarm": report.get("usable_swarm"),
        "readiness": report.get("readiness"),
        "safety": report.get("safety"),
        "limitations": report.get("limitations"),
    })
    write_json(output_dir / "support_bundle.json", bundle)
    report["artifacts"]["support_bundle_json"] = {
        "kind": "usable_swarm_inference_support_bundle",
        "path": "support_bundle.json",
        "present": True,
        "schema": SUPPORT_SCHEMA,
        "ok": bundle.get("ok"),
    }
    write_json(output_dir / "usable_swarm_inference.json", report)
    return report


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == MODE_PACKAGE:
        return build_package(args, output_dir=output_dir)
    if args.mode == MODE_EVIDENCE_IMPORT:
        return build_evidence_import(args, output_dir=output_dir)
    return build_local(args, output_dir=output_dir, runner=runner)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Usable Swarm Inference v1 evidence.")
    parser.add_argument("mode", choices=MODES, nargs="?", default=MODE_LOCAL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--p2p-report", default=DEFAULT_P2P_REPORT)
    parser.add_argument("--swarm-id", default="usable-swarm-v1")
    parser.add_argument("--public-host", default="127.0.0.1")
    parser.add_argument("--p2p-port", type=int, default=9788)
    parser.add_argument("--coordinator-port", type=int, default=9789)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--hf-model-id", default=DEFAULT_HF_MODEL_ID)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-texts", default="")
    parser.add_argument("--stream-generation", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--http-timeout", type=float, default=10.0)
    parser.add_argument("--preview-v04-report", default="dist/public-swarm-preview-v04-final/public_swarm_preview_v04.json")
    parser.add_argument("--product-mvp-report", default="dist/public-swarm-preview-v04-distilgpt2-strict/product-mvp/product_swarm_mvp_check.json")
    parser.add_argument("--optional-model-report", default="dist/public-swarm-preview-v04-distilgpt2-strict/optional-model-mvp/product_swarm_mvp_check.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 2 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 2 and 32")
    try:
        parse_prompt_texts_arg(args.prompt_text, args.prompt_texts)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.p2p_port < 1 or args.coordinator_port < 1:
        raise SystemExit("--p2p-port and --coordinator-port must be positive")
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
        usable = report.get("usable_swarm") if isinstance(report.get("usable_swarm"), dict) else {}
        print("CrowdTensor Usable Swarm Inference v1")
        print(f"  ok: {report.get('ok')}")
        print(f"  mode: {report.get('mode')}")
        print(f"  ready: {usable.get('ready')}")
        print(f"  output: {report.get('output_dir')}")
        print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
