#!/usr/bin/env python3
"""Build the user-facing Public Swarm Product Beta artifact.

This layer turns the current release-candidate evidence into a product-shaped
operator flow: serve, join stage0/stage1, generate, diagnose, and package.  The
execution remains Coordinator-backed and read-only.
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

import public_swarm_inference_beta_rc_pack as rc_pack  # noqa: E402
import support_bundle  # noqa: E402
from crowdtensor.real_llm import missing_hf_dependencies  # noqa: E402
from product_swarm_mvp_check import parse_prompt_texts_arg, read_prompt_texts_file  # noqa: E402


SCHEMA = "public_swarm_product_beta_v1"
RC_SCHEMA = "public_swarm_inference_beta_rc_v1"
REMOTE_REAL_SCHEMA = "remote_real_llm_sharded_beta_v1"
DEFAULT_OUTPUT_DIR = "dist/public-swarm-product-beta"
DEFAULT_PROMPT = "CrowdTensor product beta"
WORKLOAD_TYPE = "real_llm_sharded_infer"
SECRET_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
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
)
PRIVATE_ARTIFACT_NAMES = {
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
}

Runner = Callable[..., subprocess.CompletedProcess[str]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def redact_text(text: str, secret_values: list[str] | None = None) -> str:
    redacted = str(text)
    for value in secret_values or []:
        if value:
            redacted = redacted.replace(value, "<redacted>")
    for fragment in SECRET_FRAGMENTS:
        redacted = redacted.replace(fragment, "<redacted>")
    return redacted


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


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


STREAM_READY_CODES = {
    "public_swarm_generate_stream_ready",
    "public_swarm_generate_stream_endpoint_ready",
}
BATCH_READY_CODES = {
    "public_swarm_generate_batch_ready",
}
LOCAL_LOOPBACK_SUPERSEDED_RC_BLOCKED_CODES = {
    "p2p_lite_discovery_blocked",
    "p2p_lite_route_blocked",
    "public_swarm_inference_beta_blocked",
    "public_swarm_inference_beta_rc_blocked",
    "public_swarm_product_beta_blocked",
    "public_swarm_product_rc_blocked",
}


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


def drop_superseded_local_loopback_blockers(codes: set[str]) -> set[str]:
    return set(codes) - LOCAL_LOOPBACK_SUPERSEDED_RC_BLOCKED_CODES


def drop_unproven_stream_codes(codes: set[str], *, stream_ready: bool) -> set[str]:
    if stream_ready:
        return codes
    return {code for code in codes if code not in STREAM_READY_CODES}


def drop_unproven_generation_ready_codes(
    codes: set[str],
    *,
    stream_ready: bool,
) -> set[str]:
    filtered = drop_unproven_stream_codes(codes, stream_ready=stream_ready)
    filtered -= BATCH_READY_CODES
    return filtered


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


def int_seconds(value: float | int | str) -> str:
    return str(max(1, int(float(value))))


def rc_args(args: argparse.Namespace, output_dir: Path) -> argparse.Namespace:
    argv = [
        args.mode,
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
        "--coordinator-url",
        args.coordinator_url,
        "--target",
        args.target,
        "--miner-id-prefix",
        args.miner_id_prefix,
        "--hf-model-id",
        args.hf_model_id,
        "--gpu-report",
        args.gpu_report,
        "--scenario-id",
        args.scenario_id,
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
        argv.extend(["--prompt-texts-file", args.prompt_texts_file])
    elif args.prompt_texts:
        argv.extend(["--prompt-texts", args.prompt_texts])
    else:
        argv.extend(["--prompt-text", args.prompt_text])
    if args.hf_cache_dir:
        argv.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.observer_token:
        argv.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        argv.extend(["--admin-token", args.admin_token])
    if args.stream_generation:
        argv.append("--stream-generation")
    return rc_pack.parse_args(argv)


def run_rc_core(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    payload = rc_pack.build_report(rc_args(args, output_dir), runner=runner)
    step = {
        "name": "public_swarm_beta_rc_core",
        "ok": bool(payload.get("ok")),
        "duration_seconds": round(time.monotonic() - started, 3),
        "payload_schema": payload.get("schema"),
        "payload_ok": payload.get("ok"),
    }
    return step, payload


def run_split_validation(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    missing = missing_hf_dependencies()
    if missing:
        return {
            "name": "real_llm_split_validation",
            "ok": False,
            "duration_seconds": 0.0,
            "error": "hf_dependencies_missing",
            "operator_action": "Install optional runtime dependencies with: python -m pip install -e '.[hf]'",
        }, {
            "schema": REMOTE_REAL_SCHEMA,
            "ok": False,
            "diagnosis_codes": ["hf_dependencies_missing"],
            "missing_dependencies": missing,
        }
    command = [
        sys.executable,
        str(ROOT / "scripts" / "remote_real_llm_sharded_beta_pack.py"),
        "--mode",
        "remote-loopback",
        "--output-dir",
        str(output_dir),
        "--base-port",
        str(args.base_port + 40),
        "--request-count",
        "1",
        "--max-new-tokens",
        "1",
        "--failure-mode",
        "none",
        "--stage-mode",
        "split",
        "--require-distinct-stage-miners",
        "--hf-model-id",
        args.hf_model_id,
        "--timeout-seconds",
        int_seconds(max(float(args.timeout_seconds), 240.0)),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    return rc_pack.run_json_step(
        "remote_real_llm_sharded_loopback",
        command,
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), 240.0) + 180.0,
    )


def find_private_artifacts(path: Path) -> list[str]:
    if not path.exists():
        return []
    private: list[str] = []
    for candidate in path.rglob("*"):
        if candidate.name in PRIVATE_ARTIFACT_NAMES:
            try:
                private.append(candidate.relative_to(path).as_posix())
            except ValueError:
                private.append(str(candidate))
    return sorted(private)


def summarize_rc(payload: dict[str, Any]) -> dict[str, Any]:
    rc = payload.get("rc") if isinstance(payload.get("rc"), dict) else {}
    batch = safe_batch_summary(rc.get("batch") if isinstance(rc.get("batch"), dict) else {})
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "ready": rc.get("ready"),
        "diagnosis_codes": diagnosis_codes(payload),
        "product_beta_ready": rc.get("product_beta_ready"),
        "p2p_lite_route_ready": rc.get("p2p_lite_route_ready"),
        "cpu_fallback_ready": rc.get("cpu_fallback_ready"),
        "mode_ready": rc.get("mode_ready"),
        "workload_type": rc.get("workload_type"),
        "max_new_tokens": rc.get("max_new_tokens"),
        "batch": batch,
    }


def summarize_split(payload: dict[str, Any]) -> dict[str, Any]:
    summaries = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
    inner = next((item for item in summaries.values() if isinstance(item, dict)), {})
    assignment = inner.get("stage_assignment") if isinstance(inner.get("stage_assignment"), dict) else {}
    session = inner.get("session") if isinstance(inner.get("session"), dict) else {}
    return {
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "mode": payload.get("mode"),
        "diagnosis_codes": diagnosis_codes(payload),
        "session": {
            "stage_count": session.get("stage_count"),
            "request_count": session.get("request_count"),
            "model_id": session.get("model_id"),
        },
        "stage_assignment": {
            "stage0_miner_id": assignment.get("stage0_miner_id"),
            "stage1_miner_id": assignment.get("stage1_miner_id"),
            "distinct_stage_miners": assignment.get("distinct_stage_miners"),
            "stage_assignment_valid": assignment.get("stage_assignment_valid"),
        },
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
            "Public Swarm Product Beta artifacts summarize serve/join/generate readiness "
            "with counts, hashes, and route evidence only. Run `crowdtensor generate` "
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
            "This Product Beta artifact records prompt source/count and placeholder safety only; "
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
            "This Public Swarm Product Beta report is shareable product-path evidence, "
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
        "summary": "Share public_swarm_product_beta.json/md and support_bundle.json; they contain hashes/counts and readiness evidence, not raw prompts or answers.",
    }


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
    return str(prompt_scope.get("summary") or "")


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


def write_support_bundle(output_dir: Path, report: dict[str, Any], *, secret_values: list[str] | None = None) -> dict[str, Any]:
    bundle = support_bundle.sanitize(redact_values({
        "schema": "public_swarm_product_beta_support_bundle_v1",
        "generated_at": utc_now(),
        "ok": bool(report.get("ok")),
        "product_beta": {
            "schema": report.get("schema"),
            "mode": report.get("mode"),
            "ok": report.get("ok"),
            "diagnosis_codes": report.get("diagnosis_codes") or [],
        },
        "artifacts": report.get("artifacts") or {},
        "output_request": report.get("output_request"),
        "prompt_scope": report.get("prompt_scope"),
        "answer_scope": report.get("answer_scope"),
        "shareable_summary": report.get("shareable_summary"),
        "safety": report.get("safety") or {},
        "limitations": report.get("limitations") or [],
    }, secret_values))
    path = output_dir / "support_bundle.json"
    write_json(path, bundle)
    return artifact_entry(path, output_dir, kind="public_swarm_product_beta_support_bundle", schema=str(bundle.get("schema")), ok=bundle.get("ok"))


def required_mode_ready(args: argparse.Namespace, rc_codes: set[str], split_codes: set[str]) -> tuple[bool, set[str]]:
    codes: set[str] = set()
    rc_ready = "public_swarm_inference_beta_rc_ready" in rc_codes
    if args.mode == "local-loopback":
        local_required = {
            "serve_join_generate_loop_ready",
            "remote_generate_session_ready",
            "public_swarm_generate_ready",
        }
        split_required = {
            "remote_real_llm_sharded_ready",
            "remote_real_llm_sharded_loopback_ready",
            "decoded_tokens_match",
            "distinct_stage_miners",
            "stage_assignment_valid",
        }
        local_ready = local_required <= rc_codes and split_required <= split_codes
        if local_ready:
            codes.update({
                "serve_ready",
                "stage0_join_ready",
                "stage1_join_ready",
                "generate_ready",
                "serve_join_generate_loop_ready",
                "remote_generate_session_ready",
                "public_swarm_generate_ready",
                "decoded_tokens_match",
                "distinct_stage_miners",
                "stage_assignment_valid",
            })
        return bool(local_ready), codes
    if args.mode == "package":
        package_required = {
            "public_swarm_beta_rc_package_ready",
            "miner_join_pack_ready",
            "private_artifacts_local_only",
        }
        package_ready = package_required <= rc_codes
        if package_ready:
            codes.update({
                "public_swarm_product_beta_package_ready",
                "miner_join_pack_ready",
                "private_artifacts_local_only",
            })
            if args.target == "kaggle":
                codes.add("kaggle_remote_miner_package_ready")
        return bool(rc_ready and package_ready), codes
    external_required = {
        "serve_join_generate_loop_ready",
        "remote_generate_session_ready",
        "public_swarm_generate_ready",
        "external_runtime_verified",
        "remote_real_llm_sharded_existing_ready",
    }
    external_ready = external_required <= rc_codes
    if external_ready:
        codes.update({
            "serve_ready",
            "stage0_join_ready",
            "stage1_join_ready",
            "generate_ready",
            "serve_join_generate_loop_ready",
            "remote_generate_session_ready",
            "public_swarm_generate_ready",
            "external_runtime_verified",
            "remote_real_llm_sharded_existing_ready",
        })
        for code in ["decoded_tokens_match", "distinct_stage_miners", "stage_assignment_valid"]:
            if code in rc_codes:
                codes.add(code)
    return bool(rc_ready and external_ready), codes


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    secret_values = [args.admin_token, args.observer_token]
    rc_output = output_dir / "rc"
    rc_step, rc_payload = run_rc_core(args, output_dir=rc_output, runner=runner)
    rc_codes = set(diagnosis_codes(rc_payload))
    rc = rc_payload.get("rc") if isinstance(rc_payload.get("rc"), dict) else {}
    rc_batch = safe_batch_summary(rc.get("batch") if isinstance(rc.get("batch"), dict) else {})
    rc_stream = rc.get("stream") if isinstance(rc.get("stream"), dict) else {}
    rc_batch_ready = bool(rc_batch.get("enabled") and rc_batch.get("batch_generation_ready") is True)
    rc_stream_ready = stream_evidence_ready(rc_stream, rc_batch)
    split_step: dict[str, Any] | None = None
    split_payload: dict[str, Any] = {}
    if args.mode == "local-loopback":
        split_step, split_payload = run_split_validation(args, output_dir=output_dir / "split-validation", runner=runner)
    split_codes = set(diagnosis_codes(split_payload))
    private_artifacts = find_private_artifacts(output_dir)
    private_clean = not private_artifacts
    mode_ready, mode_codes = required_mode_ready(args, rc_codes, split_codes)
    support_ready = True
    inherited_codes = set(rc_codes) | set(split_codes)
    # Reserve this code for the top-level Product Beta verdict.  The RC child
    # may report its own product aggregate as ready while the user-facing
    # serve/join/generate path is still blocked by missing runtime deps.
    inherited_codes.discard("public_swarm_product_beta_ready")
    codes = drop_unproven_generation_ready_codes(
        inherited_codes | mode_codes,
        stream_ready=rc_stream_ready,
    )
    common_ready = {
        "public_swarm_product_beta_ready",
        "public_swarm_product_beta_user_path_ready",
        "support_bundle_ready",
        "cpu_fallback_ready",
        "local_cpu_inference_ready",
        "read_only_workload",
        "not_production",
        "not_p2p",
        "not_large_model_serving",
    }
    legacy_p2p_ready_codes = {"p2p_lite_route_ready", "p2p_lite_discovery_ready"}
    if args.mode == "package":
        privacy_ready = "private_artifacts_local_only" in codes
    else:
        privacy_ready = private_clean
        if private_clean:
            codes.add("private_artifacts_cleaned")
    ready = bool(mode_ready and support_ready and privacy_ready)
    if ready:
        if args.mode == "local-loopback":
            codes = drop_superseded_local_loopback_blockers(codes)
        codes.update(common_ready)
        if legacy_p2p_ready_codes <= inherited_codes:
            codes.update(legacy_p2p_ready_codes)
        if rc_batch_ready:
            codes.add("public_swarm_generate_batch_ready")
        if rc_stream_ready:
            codes.add("public_swarm_generate_stream_ready")
            if rc_stream.get("endpoint_ready"):
                codes.add("public_swarm_generate_stream_endpoint_ready")
    else:
        codes.add("public_swarm_product_beta_blocked")
    artifacts = {
        "public_swarm_inference_beta_rc_json": artifact_entry(
            rc_output / "public_swarm_inference_beta_rc.json",
            output_dir,
            kind="public_swarm_inference_beta_rc",
            schema=RC_SCHEMA,
            ok=rc_payload.get("ok") if rc_payload else None,
        )
    }
    if args.mode == "local-loopback":
        artifacts["remote_real_llm_sharded_beta_json"] = artifact_entry(
            output_dir / "split-validation" / "remote_real_llm_sharded_beta.json",
            output_dir,
            kind="remote_real_llm_sharded_beta",
            schema=REMOTE_REAL_SCHEMA,
            ok=split_payload.get("ok") if split_payload else None,
        )
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": args.mode,
        "target": args.target,
        "output_dir": str(output_dir),
        "product_beta": {
            "ready": ready,
            "mode_ready": mode_ready,
            "support_bundle_ready": support_ready,
            "privacy_ready": privacy_ready,
            "workload_type": WORKLOAD_TYPE,
            "hf_model_id": args.hf_model_id,
            "max_new_tokens": args.max_new_tokens,
            "batch": rc_batch if rc_batch else {"enabled": False, "batch_generation_ready": False},
            "stream": rc_stream if rc_stream else {"enabled": False, "stream_generation_ready": False},
            "user_surface": ["serve", "join", "generate"],
        },
        "steps": [rc_step] + ([split_step] if split_step else []),
        "payload_summaries": {
            "public_swarm_beta_rc": summarize_rc(rc_payload),
            "real_llm_split_validation": summarize_split(split_payload) if split_payload else {},
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": artifacts,
        "private_artifacts": private_artifacts,
        "safety": {
            "coordinator_backed_task_execution": True,
            "serve_join_generate_product_loop": args.mode == "local-loopback",
            "p2p_lite_discovery_only": True,
            "tokens_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_payloads_redacted": True,
            "read_only_workload": WORKLOAD_TYPE,
            "not_production": True,
            "not_p2p": True,
            "not_libp2p": True,
            "not_dht": True,
            "not_nat_traversal": True,
            "not_gpu_pooling_marketplace": True,
            "not_large_model_serving": True,
            "not_public_prompt_serving": True,
        },
        "operator_action": [
            "Start the Coordinator with crowdtensor serve.",
            "Join two stage Miners with crowdtensor join --stage stage0 and --stage stage1.",
            "Create bounded read-only generation sessions with crowdtensor generate.",
            "Use this Product Beta artifact as the shareable redacted readiness report.",
        ],
        "limitations": [
            "Public Swarm Product Beta is Coordinator-backed and read-only; it is not production Swarm Inference.",
            "P2P-lite is route discovery only; not libp2p, DHT, NAT traversal, or decentralized execution.",
            "The default path uses tiny GPT / CPU evidence and safe summaries; not Hivemind/Petals-level serving or large-model public prompt serving.",
        ],
    }
    report["output_request"] = output_request_summary()
    report["prompt_scope"] = prompt_scope_summary(args)
    report["answer_scope"] = answer_scope_summary()
    report["shareable_summary"] = shareable_summary()
    report["artifacts"]["support_bundle_json"] = write_support_bundle(output_dir, report, secret_values=secret_values)
    return persist_report(report, output_dir=output_dir, secret_values=secret_values)


def render_markdown(report: dict[str, Any]) -> str:
    beta = report.get("product_beta") if isinstance(report.get("product_beta"), dict) else {}
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    lines = [
        "# CrowdTensor Public Swarm Product Beta",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- output_dir: `{report.get('output_dir')}`",
        f"- ready: `{beta.get('ready')}`",
        f"- max_new_tokens: `{beta.get('max_new_tokens')}`",
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
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`",
        "",
        "## Steps",
        "",
    ]
    for step in report.get("steps") or []:
        lines.append(f"- `{step.get('name')}`: `{step.get('ok')}` schema=`{step.get('payload_schema')}`")
    lines.extend(["", "## Boundaries", ""])
    for item in report.get("limitations") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def persist_report(report: dict[str, Any], *, output_dir: Path, secret_values: list[str] | None = None) -> dict[str, Any]:
    report.setdefault("output_request", output_request_summary())
    report.setdefault("prompt_scope", prompt_scope_summary(argparse.Namespace(prompt_text=DEFAULT_PROMPT, prompt_texts="", prompt_texts_file="", prompt_texts_list=[])))
    report.setdefault("answer_scope", answer_scope_summary())
    report.setdefault("shareable_summary", shareable_summary())
    report = support_bundle.sanitize(redact_values(report, secret_values))
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"sensitive_output_detected"})
        report["safety_error"] = "Public Swarm Product Beta report contained secret-like fragments"
    json_path = output_dir / "public_swarm_product_beta.json"
    markdown_path = output_dir / "public_swarm_product_beta.md"
    report.setdefault("artifacts", {})
    report["artifacts"]["public_swarm_product_beta_json"] = artifact_entry(
        json_path,
        output_dir,
        kind="public_swarm_product_beta",
        schema=SCHEMA,
        ok=report.get("ok"),
    )
    report["artifacts"]["public_swarm_product_beta_markdown"] = artifact_entry(
        markdown_path,
        output_dir,
        kind="public_swarm_product_beta_markdown",
    )
    write_json(json_path, report)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    report["artifacts"]["public_swarm_product_beta_json"]["present"] = True
    report["artifacts"]["public_swarm_product_beta_markdown"]["present"] = True
    write_json(json_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Public Swarm Product Beta evidence.")
    parser.add_argument("mode", choices=["local-loopback", "package", "external-existing"])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-port", type=int, default=9320)
    parser.add_argument("--port", type=int, default=9320)
    parser.add_argument("--public-host", default="127.0.0.1")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--target", choices=["local", "kaggle"], default="local")
    parser.add_argument("--miner-id-prefix", default="public-swarm-product-beta")
    parser.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--gpu-report", default=rc_pack.DEFAULT_GPU_REPORT)
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-texts", default="", help="comma-separated bounded batch of up to 4 prompts")
    parser.add_argument("--prompt-texts-file", default="", help="newline-delimited bounded batch of up to 4 prompts")
    parser.add_argument("--stream-generation", action="store_true")
    parser.add_argument("--scenario-id", default="route-baseline")
    parser.add_argument("--request-count", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--cpu-request-count", type=int, default=1)
    parser.add_argument("--external-llm-request-count", type=int, default=1)
    parser.add_argument("--admin-token", default="")
    parser.add_argument("--observer-token", default="")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--remote-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--cpu-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--process-exit-timeout", type=float, default=20.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--http-timeout", type=float, default=10.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.base_port < 1 or args.port < 1:
        raise SystemExit("--base-port and --port must be positive")
    if args.request_count < 1 or args.request_count > 4:
        raise SystemExit("--request-count must be between 1 and 4")
    if args.cpu_request_count < 1 or args.cpu_request_count > 4:
        raise SystemExit("--cpu-request-count must be between 1 and 4")
    if args.external_llm_request_count < 1 or args.external_llm_request_count > 4:
        raise SystemExit("--external-llm-request-count must be between 1 and 4")
    if args.max_new_tokens < 2 or args.max_new_tokens > 32:
        raise SystemExit("--max-new-tokens must be between 2 and 32")
    if args.prompt_texts and args.prompt_texts_file:
        raise SystemExit("public_swarm_product_beta accepts either --prompt-texts or --prompt-texts-file, not both")
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
    if args.mode == "external-existing":
        missing = [
            name
            for name in ["coordinator_url", "observer_token", "admin_token"]
            if not getattr(args, name)
        ]
        if missing:
            raise SystemExit(f"external-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
    return args


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
        prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
        answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
        shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
        print(f"Public Swarm Product Beta ready: {report.get('ok')}")
        if output_request:
            print(f"  output_request: {output_request_text(output_request)}")
            print(f"  output_request_note: {output_request_note(output_request)}")
        if prompt_scope:
            print(f"  prompt_scope: {prompt_scope_text(prompt_scope)}")
            if prompt_scope_note(prompt_scope):
                print(f"  prompt_scope_note: {prompt_scope_note(prompt_scope)}")
        if answer_scope:
            print(f"  answer_scope: {answer_scope_text(answer_scope)}")
            print(f"  answer_scope_note: {answer_scope_note(answer_scope)}")
        if shareable:
            print(f"  shareable: {shareable_summary_text(shareable)}")
        print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
