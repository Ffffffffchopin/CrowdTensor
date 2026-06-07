#!/usr/bin/env python3
"""Build the Petals-class P2P Inference Production Candidate artifact."""

from __future__ import annotations

import argparse
import json
import os
import signal
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


SCHEMA = "petals_class_p2p_candidate_v1"
SUPPORT_SCHEMA = "petals_class_p2p_candidate_support_bundle_v1"
MODE_LOCAL_SMOKE = "local-smoke"
MODE_PACKAGE = "package"
MODE_EVIDENCE_IMPORT = "evidence-import"
MODES = [MODE_LOCAL_SMOKE, MODE_PACKAGE, MODE_EVIDENCE_IMPORT]
DEFAULT_OUTPUT_DIR = "dist/petals-class-p2p-candidate"
DEFAULT_LOCAL_REPORT = "dist/real-p2p-libp2p-local-smoke-ready/real_p2p_swarm_inference_core_rc.json"
DEFAULT_RUNTIME_SMOKE_REPORT = "dist/real-p2p-libp2p-kaggle-runtime-smoke-20260531-r6/real_p2p_swarm_inference_core_rc.json"
DEFAULT_EXTERNAL_REPORT = "dist/goal-final-infer-fresh-real-p2p-kaggle-16tok-20260601/real_p2p_swarm_inference_core_rc.json"
DEFAULT_REQUEUE_REPORT = "dist/petals-p2p-candidate-live-stage0-20260531-r6/real_p2p_swarm_inference_core_rc.json"
DEFAULT_BATCH_STREAM_REPORT = ""
Runner = Callable[..., subprocess.CompletedProcess[str]]

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
    '"generated_text":',
    '"generated_token_ids":',
    '"prompt_text":',
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
    "kernel.py",
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

BATCH_READY_CODES = {
    "external_real_p2p_generate_batch_ready",
    "p2p_candidate_batch_generation_ready",
    "p2p_real_generate_batch_ready",
    "public_swarm_generate_batch_ready",
    "public_swarm_v2_batch_generation_ready",
    "usable_real_llm_batch_ready",
}

STREAM_READY_CODES = {
    "external_real_p2p_generate_stream_ready",
    "p2p_candidate_stream_generation_ready",
    "p2p_real_generate_stream_ready",
    "public_swarm_generate_stream_ready",
    "public_swarm_generate_stream_endpoint_ready",
    "public_swarm_v2_stream_generation_ready",
    "usable_real_llm_stream_ready",
}


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


def redact_text(value: str) -> str:
    result = str(value)
    for fragment in SECRET_FRAGMENTS:
        result = result.replace(fragment, "<redacted>")
    return result


def redact_values(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_values(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_values(item) for key, item in value.items()}
    return value


def diagnosis_codes(*payloads: dict[str, Any], extra: list[str] | None = None) -> list[str]:
    codes: set[str] = set(extra or [])
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for code in payload.get("diagnosis_codes") or []:
            if isinstance(code, str):
                codes.add(code)
        summaries = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
        for item in summaries.values():
            if isinstance(item, dict):
                for code in item.get("diagnosis_codes") or []:
                    if isinstance(code, str):
                        codes.add(code)
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


def output_request_summary() -> dict[str, Any]:
    return {
        "include_output": False,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Petals-class P2P Candidate public artifacts summarize local, runtime-smoke, "
            "external generation, requeue, peer scoring, safe batch, and safe stream "
            "evidence only. Run the product generate command in local human mode to "
            "display answer text."
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
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "public_artifact_safe": True,
        "summary": (
            "This Petals-class P2P Candidate report is shareable readiness evidence, "
            "not an answer transcript. Raw prompts, generated text, generated token ids, "
            "activations, lease tokens, peer secrets, private runtime payloads, and raw "
            "runtime state are excluded."
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
        "summary": (
            "Share `petals_class_p2p_candidate.json`, `petals_class_p2p_candidate.md`, "
            "and `support_bundle.json`; they contain candidate readiness evidence, "
            "hashes, counts, and safe summaries, not raw prompts or answers."
        ),
    }


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


def safe_live_requeue_summary(payload: dict[str, Any]) -> dict[str, Any]:
    codes = set(diagnosis_codes(payload))
    summary = payload.get("live_requeue_summary") if isinstance(payload.get("live_requeue_summary"), dict) else {}
    if not summary and isinstance(payload.get("candidate"), dict):
        candidate = payload["candidate"]
        summary = candidate.get("live_requeue_summary") if isinstance(candidate.get("live_requeue_summary"), dict) else {}
    if not summary and isinstance(payload.get("summaries"), dict):
        summaries = payload["summaries"]
        requeue = summaries.get("requeue") if isinstance(summaries.get("requeue"), dict) else {}
        summary = requeue.get("live_requeue_summary") if isinstance(requeue.get("live_requeue_summary"), dict) else {}
    safe = {field: summary.get(field) for field in LIVE_REQUEUE_SUMMARY_FIELDS if field in summary}
    if safe.get("lease_expired") == "<redacted>" and "live_requeue_lease_timeout_observed" in codes:
        safe["lease_expired"] = True
    return safe


def safe_batch_summary(payload: dict[str, Any]) -> dict[str, Any]:
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


def safe_stream_summary(payload: dict[str, Any]) -> dict[str, Any]:
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
            safe_stream = {
                "enabled": bool(stream.get("enabled")),
                "requested": bool(stream.get("requested") or stream.get("enabled")),
                "event_count": int(stream.get("event_count") or len(safe_events)),
                "source": stream.get("source"),
                "endpoint_ready": bool(stream.get("endpoint_ready")),
                "progress": {
                    "stream_progress_complete": bool(progress.get("stream_progress_complete")),
                    "all_token_events_ready": bool(progress.get("all_token_events_ready")),
                    "monotonic_progress": bool(progress.get("monotonic_progress")),
                    "expected_request_count": int(progress.get("expected_request_count") or 0),
                    "per_request_progress": [
                        {
                            "request_key": row.get("request_key"),
                            "request_id": row.get("request_id"),
                            "prompt_hash": row.get("prompt_hash"),
                            "event_count": int(row.get("event_count") or 0),
                            "observed_token_counts": list(row.get("observed_token_counts") or []),
                            "max_observed_token_count": int(row.get("max_observed_token_count") or 0),
                            "target_token_count": int(row.get("target_token_count") or 0),
                            "monotonic_progress": bool(row.get("monotonic_progress")),
                            "stream_progress_complete": bool(row.get("stream_progress_complete")),
                        }
                        for row in progress.get("per_request_progress") or []
                        if isinstance(row, dict)
                    ],
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
            safe_stream["stream_generation_ready"] = stream_evidence_ready(safe_stream, safe_batch_summary(item))
            return safe_stream
        for value in item.values():
            if isinstance(value, dict):
                pending.append(value)
            elif isinstance(value, list):
                pending.extend(entry for entry in value if isinstance(entry, dict))
    return {"enabled": False, "stream_generation_ready": False}


def stream_evidence_ready(stream: dict[str, Any], batch: dict[str, Any] | None = None) -> bool:
    if not isinstance(stream, dict) or stream.get("stream_generation_ready") is not True:
        return False
    progress = stream.get("progress") if isinstance(stream.get("progress"), dict) else {}
    expected_requests = int(
        progress.get("expected_request_count")
        or (batch or {}).get("expected_request_count")
        or (batch or {}).get("request_count")
        or 1
    )
    if expected_requests > 1 or bool((batch or {}).get("enabled")):
        return bool(
            progress.get("per_request_progress")
            and progress.get("per_request_progress_complete") is True
            and progress.get("per_request_monotonic_progress") is True
        )
    return bool(progress.get("stream_progress_complete") is True and progress.get("monotonic_progress") is True)


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


def report_summary(payload: dict[str, Any]) -> dict[str, Any]:
    p2p = payload.get("p2p") if isinstance(payload.get("p2p"), dict) else {}
    generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else {}
    stage_assignment = payload.get("stage_assignment") if isinstance(payload.get("stage_assignment"), dict) else {}
    lifecycle = payload.get("kaggle_lifecycle") if isinstance(payload.get("kaggle_lifecycle"), dict) else {}
    peer_scoring = p2p.get("peer_scoring") if isinstance(p2p.get("peer_scoring"), dict) else {}
    if not peer_scoring and isinstance(p2p.get("catalog"), dict):
        catalog = p2p["catalog"]
        peer_scoring = catalog.get("peer_scoring") if isinstance(catalog.get("peer_scoring"), dict) else {}
    codes = set(diagnosis_codes(payload))
    batch = safe_batch_summary(payload)
    stream = safe_stream_summary(payload)
    return {
        "schema": payload.get("schema"),
        "ok": bool(payload.get("ok")),
        "mode": payload.get("mode"),
        "diagnosis_codes": sorted(codes),
        "discovery_backend": p2p.get("discovery_backend"),
        "provider_count": p2p.get("provider_count"),
        "stage0_provider_count": p2p.get("stage0_provider_count"),
        "stage1_provider_count": p2p.get("stage1_provider_count"),
        "generated_token_count": generation.get("generated_token_count"),
        "decoded_tokens_match": generation.get("decoded_tokens_match"),
        "distinct_stage_miners": stage_assignment.get("distinct_stage_miners"),
        "accepted_rows": (payload.get("ledger") or {}).get("accepted_rows") if isinstance(payload.get("ledger"), dict) else None,
        "kaggle_kernels_deleted": lifecycle.get("kernels_deleted"),
        "private_artifacts_cleaned": lifecycle.get("local_private_artifacts_cleaned"),
        "peer_scoring_ready": bool("peer_scoring_ready" in codes or peer_scoring.get("schema") == "real_p2p_peer_scoring_v1"),
        "hf_model_id": first_string_value(payload, "hf_model_id"),
        "batch": batch,
        "batch_ready": bool(batch.get("enabled") and batch.get("batch_generation_ready") is True),
        "stream": stream,
        "stream_ready": stream_evidence_ready(stream, batch),
        "live_requeue_summary": safe_live_requeue_summary(payload),
    }


def run_json_step(
    name: str,
    command: list[str],
    *,
    runner: Runner,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    try:
        completed = run_subprocess_step(command, runner=runner, timeout_seconds=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "ok": False,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": "timeout",
            "stdout_tail": redact_text((exc.stdout or "")[-1200:] if isinstance(exc.stdout, str) else ""),
            "stderr_tail": redact_text((exc.stderr or "")[-1200:] if isinstance(exc.stderr, str) else ""),
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
        step["stdout_tail"] = redact_text((completed.stdout or "")[-1200:])
        step["stderr_tail"] = redact_text((completed.stderr or "")[-1200:])
    return step, redact_values(payload)


def run_subprocess_step(command: list[str], *, runner: Runner, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    if runner is not subprocess.run:
        return runner(
            command,
            cwd=str(ROOT),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )

    proc = subprocess.Popen(
        command,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        cleanup_timeout_processes(command)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            cleanup_timeout_processes(command, sig=signal.SIGKILL)
            stdout, stderr = proc.communicate(timeout=5)
        raise subprocess.TimeoutExpired(
            command,
            timeout_seconds,
            output=stdout or (exc.stdout if isinstance(exc.stdout, str) else ""),
            stderr=stderr or (exc.stderr if isinstance(exc.stderr, str) else ""),
        )
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


def _command_value(command: list[str], flag: str) -> str:
    try:
        return str(command[command.index(flag) + 1])
    except (ValueError, IndexError):
        return ""


def cleanup_timeout_processes(command: list[str], *, sig: signal.Signals = signal.SIGTERM) -> None:
    output_dir = _command_value(command, "--output-dir")
    p2p_port = _command_value(command, "--p2p-port")
    peer_bootstrap = f"--peer-bootstrap http://127.0.0.1:{p2p_port}" if p2p_port else ""
    fragments = [
        fragment
        for fragment in [
            output_dir,
            peer_bootstrap,
            f"real_p2p_daemon.py --host 127.0.0.1 --port {p2p_port}" if p2p_port else "",
            f"libp2p_kad_daemon.mjs --host 127.0.0.1 --port {p2p_port}" if p2p_port else "",
        ]
        if fragment
    ]
    if not fragments:
        return
    try:
        ps_output = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    except (OSError, subprocess.SubprocessError):
        return
    current_pid = os.getpid()
    for line in ps_output.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        pid_text, _sep, args = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        if any(fragment in args for fragment in fragments):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass


def cleanup_local_child_private_artifacts(child_dir: Path) -> dict[str, Any]:
    removed: list[str] = []
    for path in sorted(child_dir.glob("*peer-key.json")):
        try:
            path.unlink()
            removed.append(path.relative_to(child_dir).as_posix())
        except OSError:
            pass
    state_dir = child_dir / "state"
    if state_dir.exists():
        try:
            shutil.rmtree(state_dir)
            removed.append("state/")
        except OSError:
            pass
    return {
        "schema": "petals_class_p2p_candidate_local_cleanup_v1",
        "local_private_artifacts_cleaned": not any(path.exists() for path in child_dir.glob("*peer-key.json")) and not state_dir.exists(),
        "removed_paths": removed,
    }


def write_runbook(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    path = output_dir / "PETALS_CLASS_P2P_CANDIDATE.md"
    lines = [
        "# Petals-Class P2P Inference Production Candidate v0.1",
        "",
        "This candidate uses real libp2p discovery while keeping the Coordinator as the lease and result authority.",
        "",
        "## Product Path",
        "",
        "```bash",
        f"crowdtensor p2p-daemon --host 0.0.0.0 --public-host {args.public_host} --port {args.p2p_port} --discovery-backend libp2p-kad --libp2p-host 0.0.0.0 --libp2p-port {args.libp2p_port} --libp2p-public-host {args.public_host} --record-secret \"$CROWDTENSOR_P2P_PEER_SECRET\" --require-signed --run",
        f"crowdtensor serve --p2p --p2p-backend real --peer-bootstrap http://{args.public_host}:{args.p2p_port} --peer-secret \"$CROWDTENSOR_P2P_PEER_SECRET\" --public-host {args.public_host} --port {args.coordinator_port} --run",
        f"crowdtensor join --p2p --p2p-backend real --peer-bootstrap http://{args.public_host}:{args.p2p_port} --peer-secret \"$CROWDTENSOR_P2P_PEER_SECRET\" --stage stage0 --miner-id petals-stage0 --run",
        f"crowdtensor join --p2p --p2p-backend real --peer-bootstrap http://{args.public_host}:{args.p2p_port} --peer-secret \"$CROWDTENSOR_P2P_PEER_SECRET\" --stage stage1 --miner-id petals-stage1 --run",
        f"crowdtensor generate --p2p --p2p-backend real --peer-bootstrap http://{args.public_host}:{args.p2p_port} --prompt 'CrowdTensor Petals candidate' --max-new-tokens {args.max_new_tokens} --json",
        "```",
        "",
        "## Boundaries",
        "",
        "- Not full Hivemind/Petals production parity.",
        "- Not large-model throughput.",
        "- Not complete NAT traversal or relay.",
        "- Not Coordinator-free execution.",
        "- Not an economic or full anti-Sybil system.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return artifact_entry(path, output_dir, kind="petals_class_p2p_candidate_runbook")


def build_package(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    runbook = write_runbook(args, output_dir)
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": True,
        "mode": MODE_PACKAGE,
        "output_dir": str(output_dir),
        "diagnosis_codes": ["petals_class_p2p_candidate_runbook_ready"],
        "artifacts": {"runbook": runbook},
        "safety": safety_block(),
        "not_completed": [
            "external 3-node rescue proof",
            "8-token live generation",
            "production Hivemind/Petals parity",
        ],
    }
    return sanitize_report(report, output_dir=output_dir)


def safety_block() -> dict[str, Any]:
    return {
        "real_libp2p_discovery": True,
        "coordinator_result_fallback": True,
        "tokens_gossiped": False,
        "peer_secret_gossiped": False,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "activations_gossiped": False,
        "not_production_parity": True,
        "not_large_model_throughput": True,
        "not_complete_nat_traversal": True,
        "not_coordinator_free": True,
        "not_economic_system": True,
        "not_full_anti_sybil": True,
    }


def build_from_reports(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    local_report = load_json(args.local_report)
    runtime_smoke_report = load_json(args.runtime_smoke_report)
    external_report = load_json(args.external_report)
    requeue_report = load_json(args.requeue_report) if args.requeue_report else {}
    peer_scoring_report = load_json(args.peer_scoring_report) if args.peer_scoring_report else {}
    batch_stream_report = load_json(args.batch_stream_report) if args.batch_stream_report else {}
    runbook = write_runbook(args, output_dir)
    local_summary = report_summary(local_report)
    runtime_summary = report_summary(runtime_smoke_report)
    external_summary = report_summary(external_report)
    requeue_summary = report_summary(requeue_report)
    batch_stream_summary = report_summary(batch_stream_report) if batch_stream_report else {}
    batch_stream_source = batch_stream_summary if batch_stream_report else external_summary
    batch_stream_source_ok = bool(not batch_stream_report or batch_stream_summary.get("ok"))
    peer_scoring_codes = set(diagnosis_codes(peer_scoring_report))

    external_codes = set(external_summary.get("diagnosis_codes") or [])
    requeue_codes = set(requeue_summary.get("diagnosis_codes") or [])
    live_requeue = safe_live_requeue_summary(requeue_report)
    live_requeue_ready = live_requeue_detail_ready(live_requeue)
    victim_result_not_accepted = live_requeue.get("victim_result_accepted") is False
    observed_model_ids = sorted({
        str(value)
        for value in [
            local_summary.get("hf_model_id"),
            runtime_summary.get("hf_model_id"),
            external_summary.get("hf_model_id"),
            requeue_summary.get("hf_model_id"),
            batch_stream_summary.get("hf_model_id") if batch_stream_report else "",
        ]
        if isinstance(value, str) and value
    })
    model_id_consistent = len(observed_model_ids) <= 1
    generated_tokens = int(external_summary.get("generated_token_count") or 0)
    external_generate_ready = "external_libp2p_generate_ready" in external_codes and generated_tokens >= args.max_new_tokens
    local_ready = bool(local_summary.get("ok") and "hivemind_petals_class_alpha_local_ready" in set(local_summary.get("diagnosis_codes") or []))
    runtime_ready = bool(runtime_summary.get("ok") and "real_p2p_kaggle_runtime_smoke_ready" in set(runtime_summary.get("diagnosis_codes") or []))
    external_ready = bool(external_summary.get("ok") and "external_libp2p_stage_discovery_ready" in external_codes and external_generate_ready)
    cleanup_ready = bool(external_summary.get("kaggle_kernels_deleted") and external_summary.get("private_artifacts_cleaned"))
    stage_ready = bool(external_summary.get("distinct_stage_miners") and int(external_summary.get("accepted_rows") or 0) >= args.max_new_tokens * 2)
    batch = batch_stream_source.get("batch") if isinstance(batch_stream_source.get("batch"), dict) else {"enabled": False, "batch_generation_ready": False}
    batch_ready = bool(batch_stream_source_ok and batch.get("enabled") and batch.get("batch_generation_ready") is True)
    stream = batch_stream_source.get("stream") if isinstance(batch_stream_source.get("stream"), dict) else {"enabled": False, "stream_generation_ready": False}
    stream_ready = bool(batch_stream_source_ok and stream_evidence_ready(stream, batch))
    peer_scoring_ready = bool(
        args.allow_retained_alpha_without_requeue
        or external_summary.get("peer_scoring_ready")
        or "peer_scoring_ready" in external_codes
        or "peer_scoring_ready" in set(local_summary.get("diagnosis_codes") or [])
        or "peer_scoring_ready" in peer_scoring_codes
    )
    requeue_ready = bool(
        "external_stage_requeue_ready" in requeue_codes
        and ("live_stage0_requeue_ready" in requeue_codes or "live_stage1_requeue_ready" in requeue_codes)
        and "rescue_miner_used" in requeue_codes
        and "accepted_result_after_requeue" in requeue_codes
        and live_requeue_ready
    )
    if args.allow_retained_alpha_without_requeue:
        requeue_ready = True

    ready = bool(local_ready and runtime_ready and external_ready and cleanup_ready and stage_ready and peer_scoring_ready and requeue_ready and model_id_consistent and runbook.get("present"))
    codes = drop_unproven_generation_ready_codes(
        set(diagnosis_codes(local_report, runtime_smoke_report, external_report, requeue_report, peer_scoring_report, batch_stream_report)),
        batch_ready=batch_ready,
        stream_ready=stream_ready,
    )
    if local_ready:
        codes.add("local_libp2p_proof_ready")
    if runtime_ready:
        codes.add("kaggle_runtime_smoke_ready")
    if external_ready:
        codes.add("external_multi_node_generation_ready")
    if cleanup_ready:
        codes.add("private_artifacts_cleaned")
    if peer_scoring_ready:
        codes.add("peer_scoring_ready")
    if model_id_consistent:
        codes.add("p2p_candidate_model_id_consistent")
    else:
        codes.add("p2p_candidate_model_id_mismatch")
    if batch_ready:
        codes.update({
            "p2p_candidate_batch_generation_ready",
            "public_swarm_generate_batch_ready",
        })
    if stream_ready:
        codes.update({
            "p2p_candidate_stream_generation_ready",
            "public_swarm_generate_stream_ready",
        })
        if stream.get("endpoint_ready"):
            codes.add("public_swarm_generate_stream_endpoint_ready")
    if requeue_ready and not args.allow_retained_alpha_without_requeue:
        codes.update({
            "external_stage_requeue_ready",
            "rescue_miner_used",
            "accepted_result_after_requeue",
            "p2p_live_requeue_rescue_ready",
            "p2p_victim_result_not_accepted",
        })
    if runbook.get("present"):
        codes.add("petals_class_p2p_candidate_runbook_ready")
    if ready:
        codes.add("petals_class_p2p_candidate_ready")
    else:
        codes.add("petals_class_p2p_candidate_blocked")

    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "max_new_tokens": args.max_new_tokens,
        "candidate": {
            "local_libp2p_ready": local_ready,
            "kaggle_runtime_smoke_ready": runtime_ready,
            "external_libp2p_generate_ready": external_ready,
            "external_generated_token_count": generated_tokens,
            "distinct_stage_miners": external_summary.get("distinct_stage_miners"),
            "accepted_rows": external_summary.get("accepted_rows"),
            "hf_model_id": observed_model_ids[0] if len(observed_model_ids) == 1 else "",
            "observed_hf_model_ids": observed_model_ids,
            "model_id_consistent": model_id_consistent,
            "batch_stream_source": "batch_stream_report" if batch_stream_report else "external_report",
            "batch_ready": batch_ready,
            "batch": batch,
            "stream_ready": stream_ready,
            "stream": stream,
            "external_stage_requeue_ready": requeue_ready,
            "p2p_live_requeue_ready": live_requeue_ready,
            "victim_result_not_accepted": victim_result_not_accepted,
            "live_requeue_summary": live_requeue,
            "peer_scoring_ready": peer_scoring_ready,
            "kaggle_kernels_deleted": external_summary.get("kaggle_kernels_deleted"),
            "private_artifacts_cleaned": external_summary.get("private_artifacts_cleaned"),
            "runbook_ready": bool(runbook.get("present")),
        },
        "summaries": {
            "local": local_summary,
            "runtime_smoke": runtime_summary,
            "external": external_summary,
            "requeue": requeue_summary,
            "batch_stream": batch_stream_summary,
        },
        "source_reports": {
            "local_report": str(Path(args.local_report).resolve()),
            "runtime_smoke_report": str(Path(args.runtime_smoke_report).resolve()),
            "external_report": str(Path(args.external_report).resolve()),
            "requeue_report": str(Path(args.requeue_report).resolve()) if args.requeue_report else "",
            "peer_scoring_report": str(Path(args.peer_scoring_report).resolve()) if args.peer_scoring_report else "",
            "batch_stream_report": str(Path(args.batch_stream_report).resolve()) if args.batch_stream_report else "",
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": {
            "runbook": runbook,
            "batch_stream_report": artifact_entry(
                Path(args.batch_stream_report),
                output_dir,
                kind="petals_class_p2p_candidate_batch_stream_source",
                schema=str(batch_stream_report.get("schema") or ""),
                ok=batch_stream_report.get("ok") if batch_stream_report else None,
            ) if args.batch_stream_report else {
                "kind": "petals_class_p2p_candidate_batch_stream_source",
                "path": "",
                "present": False,
            },
        },
        "safety": safety_block(),
        "completed": ["Petals-class P2P candidate evidence aggregate"] if ready else [],
        "not_completed": [] if ready else [
            item for item, ok in [
                ("local libp2p proof", local_ready),
                ("Kaggle runtime smoke", runtime_ready),
                ("external 8-token libp2p generation", external_ready),
                ("distinct stage miner ledger", stage_ready),
                ("external requeue rescue proof with victim result rejection", requeue_ready),
                ("P2P candidate model-id consistency", model_id_consistent),
                ("peer scoring evidence", peer_scoring_ready),
                ("Kaggle cleanup", cleanup_ready),
            ]
            if not ok
        ],
    }
    return sanitize_report(report, output_dir=output_dir)


def build_local_smoke(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    child_dir = output_dir / "real-p2p-local"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "real_p2p_swarm_inference_core_rc_pack.py"),
        "local-smoke",
        "--output-dir",
        str(child_dir),
        "--discovery-backend",
        "libp2p-kad",
        "--p2p-port",
        str(args.p2p_port),
        "--coordinator-port",
        str(args.coordinator_port),
        "--libp2p-port",
        str(args.libp2p_port),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--startup-timeout",
        str(args.startup_timeout),
        "--session-queue-timeout",
        str(args.session_queue_timeout),
        "--miner-timeout",
        str(args.miner_timeout),
        "--generate-timeout",
        str(args.generate_timeout),
        "--http-timeout",
        str(args.http_timeout),
        "--json",
    ]
    step, payload = run_json_step("real_p2p_local_smoke", command, runner=runner, timeout_seconds=args.timeout_seconds)
    payload_path = child_dir / "real_p2p_swarm_inference_core_rc.json"
    if not payload and payload_path.is_file():
        payload = load_json(payload_path)
    cleanup = cleanup_local_child_private_artifacts(child_dir)
    args.local_report = str(payload_path)
    args.runtime_smoke_report = args.runtime_smoke_report
    args.external_report = args.external_report
    report = build_from_reports(args, output_dir=output_dir)
    report["steps"] = [step]
    report["child_payload"] = report_summary(payload)
    report["local_child_cleanup"] = {
        "schema": cleanup["schema"],
        "local_private_artifacts_cleaned": cleanup["local_private_artifacts_cleaned"],
        "removed_private_artifact_count": len(cleanup.get("removed_paths") or []),
    }
    if cleanup["local_private_artifacts_cleaned"]:
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"local_child_private_artifacts_cleaned"})
    else:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"local_child_private_artifacts_cleanup_blocked"})
        report["not_completed"] = list(report.get("not_completed") or []) + ["local child private artifact cleanup"]
    return sanitize_report(report, output_dir=output_dir)


def render_markdown(report: dict[str, Any]) -> str:
    candidate = report.get("candidate") if isinstance(report.get("candidate"), dict) else {}
    lines = [
        "# Petals-Class P2P Candidate",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- generated tokens: `{candidate.get('external_generated_token_count', 0)}`",
        f"- diagnosis: {', '.join(report.get('diagnosis_codes') or [])}",
        "",
        "## Readiness",
        "",
        f"- local libp2p: `{candidate.get('local_libp2p_ready')}`",
        f"- runtime smoke: `{candidate.get('kaggle_runtime_smoke_ready')}`",
        f"- external libp2p generate: `{candidate.get('external_libp2p_generate_ready')}`",
        f"- external requeue: `{candidate.get('external_stage_requeue_ready')}`",
        f"- peer scoring: `{candidate.get('peer_scoring_ready')}`",
        "",
        "## Output Scope",
        "",
        f"- output request: `include_output={bool((report.get('output_request') or {}).get('include_output'))} raw_prompt_public={bool((report.get('output_request') or {}).get('raw_prompt_public'))} raw_generated_text_public={bool((report.get('output_request') or {}).get('raw_generated_text_public'))} generated_token_ids_public={bool((report.get('output_request') or {}).get('generated_token_ids_public'))} public_artifact_safe={bool((report.get('output_request') or {}).get('public_artifact_safe'))}`",
        f"- answer scope: `state={(report.get('answer_scope') or {}).get('scope_state')} terminal_only={bool((report.get('answer_scope') or {}).get('terminal_only'))} visible_in_terminal={bool((report.get('answer_scope') or {}).get('visible_in_terminal'))} saved_json={(report.get('answer_scope') or {}).get('saved_json_display')} saved_markdown={(report.get('answer_scope') or {}).get('saved_markdown_display')} public_artifact_safe={bool((report.get('answer_scope') or {}).get('public_artifact_safe'))}`",
        f"- shareable: `saved_artifacts={bool((report.get('shareable_summary') or {}).get('saved_artifacts_public_safe'))} raw_prompt_public={bool((report.get('shareable_summary') or {}).get('raw_prompt_public'))} raw_generated_text_public={bool((report.get('shareable_summary') or {}).get('raw_generated_text_public'))} generated_token_ids_public={bool((report.get('shareable_summary') or {}).get('generated_token_ids_public'))} answer_scope_state={(report.get('shareable_summary') or {}).get('answer_scope_state')} local_answer_terminal_only={bool((report.get('shareable_summary') or {}).get('local_answer_terminal_only'))}`",
        f"- note: {(report.get('answer_scope') or {}).get('summary')}",
        "",
        "## Boundaries",
        "",
        "- Coordinator remains the lease/result authority.",
        "- This is not full Hivemind/Petals production parity.",
        "- This is not complete NAT traversal or relay.",
        "- This is not large-model throughput.",
    ]
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


def sanitize_report(report: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    report.setdefault("output_request", output_request_summary())
    report.setdefault("answer_scope", answer_scope_summary())
    report.setdefault("shareable_summary", shareable_summary())
    report = support_bundle.sanitize(redact_values(report))
    errors = validate_public_report(report)
    if errors:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"public_report_safety_failed"})
        report["safety_errors"] = errors
    artifacts = report.setdefault("artifacts", {})
    artifacts.setdefault("petals_class_p2p_candidate_json", {
        "kind": "petals_class_p2p_candidate",
        "path": "petals_class_p2p_candidate.json",
        "present": True,
        "schema": SCHEMA,
        "ok": report.get("ok"),
    })
    artifacts.setdefault("petals_class_p2p_candidate_markdown", {
        "kind": "petals_class_p2p_candidate_markdown",
        "path": "petals_class_p2p_candidate.md",
        "present": True,
    })
    write_json(output_dir / "petals_class_p2p_candidate.json", report)
    (output_dir / "petals_class_p2p_candidate.md").write_text(render_markdown(report), encoding="utf-8")
    bundle = support_bundle.sanitize({
        "schema": SUPPORT_SCHEMA,
        "ok": report.get("ok"),
        "diagnosis_codes": report.get("diagnosis_codes"),
        "candidate": report.get("candidate"),
        "summaries": report.get("summaries"),
        "output_request": report.get("output_request"),
        "answer_scope": report.get("answer_scope"),
        "shareable_summary": report.get("shareable_summary"),
        "safety": report.get("safety"),
    })
    write_json(output_dir / "support_bundle.json", bundle)
    report["artifacts"]["support_bundle_json"] = {
        "kind": "petals_class_p2p_candidate_support_bundle",
        "path": "support_bundle.json",
        "present": True,
        "schema": SUPPORT_SCHEMA,
        "ok": bundle.get("ok"),
    }
    write_json(output_dir / "petals_class_p2p_candidate.json", report)
    return report


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == MODE_PACKAGE:
        return build_package(args, output_dir=output_dir)
    if args.mode == MODE_LOCAL_SMOKE:
        return build_local_smoke(args, output_dir=output_dir, runner=runner)
    return build_from_reports(args, output_dir=output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Petals-class P2P candidate evidence.")
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--local-report", default=DEFAULT_LOCAL_REPORT)
    parser.add_argument("--runtime-smoke-report", default=DEFAULT_RUNTIME_SMOKE_REPORT)
    parser.add_argument("--external-report", default=DEFAULT_EXTERNAL_REPORT)
    parser.add_argument("--requeue-report", default=DEFAULT_REQUEUE_REPORT)
    parser.add_argument("--peer-scoring-report", default="")
    parser.add_argument("--batch-stream-report", default=DEFAULT_BATCH_STREAM_REPORT)
    parser.add_argument("--allow-retained-alpha-without-requeue", action="store_true")
    parser.add_argument("--public-host", default="24.199.118.54")
    parser.add_argument("--p2p-port", type=int, default=9860)
    parser.add_argument("--coordinator-port", type=int, default=9861)
    parser.add_argument("--libp2p-port", type=int, default=10860)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--session-queue-timeout", type=float, default=45.0)
    parser.add_argument("--miner-timeout", type=float, default=240.0)
    parser.add_argument("--generate-timeout", type=float, default=240.0)
    parser.add_argument("--http-timeout", type=float, default=10.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 2 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 2 and 32")
    for name in ["p2p_port", "coordinator_port", "libp2p_port"]:
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    for name in ["timeout_seconds", "startup_timeout", "session_queue_timeout", "miner_timeout", "generate_timeout", "http_timeout"]:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"Petals-class P2P candidate ready: {report.get('ok')}")
        print(f"Diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
