#!/usr/bin/env python3
"""Public Swarm Inference Alpha evidence wrapper.

This is the product-shaped aggregation layer above the existing real tiny GPT
split proofs.  It deliberately keeps the workload CPU-only and read-only while
combining the live Kaggle proof with the mandatory local stage requeue proof.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import support_bundle  # noqa: E402
from crowdtensor.real_llm import DEFAULT_MODEL_ID, DEFAULT_PROMPTS  # noqa: E402


SCHEMA = "public_swarm_inference_alpha_v1"
MODE_LOCAL_GENERATED = "local-generated"
MODE_LIVE_KAGGLE = "live-kaggle"
DEFAULT_OUTPUT_DIR = "dist/public-swarm-inference-alpha"
DEFAULT_PUBLIC_HOST = "24.199.118.54"
DEFAULT_PORT = 9220
DEFAULT_BASE_PORT = 9221
WORKLOAD_TYPE = "real_llm_sharded_infer"
FAILURE_NONE = "none"
FAILURE_KILL_STAGE0_AFTER_CLAIM = "kill-stage0-after-claim"
FAILURE_KILL_STAGE1_AFTER_CLAIM = "kill-stage1-after-claim"
FAILURE_MODES = {
    FAILURE_NONE,
    FAILURE_KILL_STAGE0_AFTER_CLAIM,
    FAILURE_KILL_STAGE1_AFTER_CLAIM,
}

Runner = Callable[..., subprocess.CompletedProcess[str]]

SECRET_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "lease_token",
    "idempotency_key",
    "activation_results",
    "activation_result",
    "hidden_state",
    "input_ids",
    "logits",
    "real_llm_sharded_result",
    "generated_text",
    "generated_token_ids",
    "Bearer ",
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("command emitted no JSON object")


def redact_text(text: str) -> str:
    redacted = str(text)
    for fragment in SECRET_FRAGMENTS:
        redacted = redacted.replace(fragment, "<redacted>")
    return redacted


def redact_values(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_values(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_values(item) for key, item in value.items()}
    return value


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


def count_tree(path: Path) -> tuple[int, int]:
    files = 0
    total_bytes = 0
    if not path.exists():
        return files, total_bytes
    for child in path.rglob("*"):
        if child.is_file():
            files += 1
            try:
                total_bytes += child.stat().st_size
            except OSError:
                pass
    return files, total_bytes


def prune_child_artifacts(output_dir: Path) -> dict[str, Any]:
    """Keep the public Alpha report shareable by dropping child debug trees."""

    removed_dirs: list[str] = []
    removed_files = 0
    removed_bytes = 0
    for relative in ("local-generated", "live-kaggle"):
        child_dir = output_dir / relative
        if not child_dir.exists():
            continue
        files, total_bytes = count_tree(child_dir)
        shutil.rmtree(child_dir)
        removed_dirs.append(relative)
        removed_files += files
        removed_bytes += total_bytes
    return {
        "child_artifacts_pruned": True,
        "removed_child_dirs": removed_dirs,
        "removed_files": removed_files,
        "removed_bytes": removed_bytes,
        "retained_public_artifacts": [
            "public_swarm_inference_alpha.json",
            "public_swarm_inference_alpha.md",
        ],
    }


def refresh_artifacts(report: dict[str, Any], *, output_dir: Path, local_payload: dict[str, Any], live_payload: dict[str, Any]) -> None:
    report["artifacts"] = {
        "public_swarm_inference_alpha_json": artifact_entry(
            output_dir / "public_swarm_inference_alpha.json",
            output_dir,
            kind="public_swarm_inference_alpha",
            schema=SCHEMA,
            ok=report.get("ok"),
        ),
        "public_swarm_inference_alpha_markdown": artifact_entry(
            output_dir / "public_swarm_inference_alpha.md",
            output_dir,
            kind="public_swarm_inference_alpha_markdown",
        ),
        "local_requeue_json": artifact_entry(
            output_dir / "local-generated" / "real_llm_internet_alpha.json",
            output_dir,
            kind="real_llm_internet_alpha_local_generated",
            schema="real_llm_internet_alpha_v1",
            ok=local_payload.get("ok") if local_payload else None,
        ),
        "live_swarm_beta_json": artifact_entry(
            output_dir / "live-kaggle" / "swarm_inference_beta_live.json",
            output_dir,
            kind="swarm_inference_beta_live",
            schema="swarm_inference_beta_v1",
            ok=live_payload.get("ok") if live_payload else None,
        ),
        "live_support_bundle_json": artifact_entry(
            output_dir / "live-kaggle" / "support_bundle.json",
            output_dir,
            kind="support_bundle",
            schema="support_bundle_v1",
        ),
    }


def diagnosis_codes(*payloads: dict[str, Any], extra: list[str] | None = None) -> list[str]:
    codes: set[str] = set(extra or [])
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for code in payload.get("diagnosis_codes") or []:
            if isinstance(code, str):
                codes.add(code)
        summaries = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
        for summary in summaries.values():
            if isinstance(summary, dict):
                for code in summary.get("diagnosis_codes") or []:
                    if isinstance(code, str):
                        codes.add(code)
    return sorted(codes)


def run_json_step(
    name: str,
    command: list[str],
    *,
    runner: Runner,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
        return {"name": name, "ok": False, "returncode": None, "error": "timeout"}, {}
    step: dict[str, Any] = {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
    }
    payload: dict[str, Any] = {}
    try:
        payload = json_from_stdout(completed.stdout)
    except ValueError as exc:
        step["ok"] = False
        step["error"] = str(exc)
    else:
        step["payload_schema"] = payload.get("schema")
        step["payload_ok"] = payload.get("ok")
        step["ok"] = bool(step["ok"] and payload.get("ok") is not False)
    if not step["ok"]:
        if completed.stdout and not payload:
            step["stdout_tail"] = redact_text(completed.stdout[-1600:])
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1600:])
    return step, payload


def summarize_requeue(alpha_payload: dict[str, Any]) -> dict[str, Any]:
    runtime = alpha_payload.get("runtime_classification") if isinstance(alpha_payload.get("runtime_classification"), dict) else {}
    summaries = alpha_payload.get("payload_summaries") if isinstance(alpha_payload.get("payload_summaries"), dict) else {}
    codes = set(diagnosis_codes(alpha_payload))
    return {
        "schema": alpha_payload.get("schema"),
        "ok": alpha_payload.get("ok"),
        "mode": alpha_payload.get("mode"),
        "stage_requeue_verified": bool(runtime.get("stage_requeue_verified")),
        "stage0_requeue_ready": "stage0_requeue_ready" in codes or "stage_requeue_ready" in diagnosis_codes(summaries.get("stage0_requeue", {})),
        "stage1_requeue_ready": "stage1_requeue_ready" in codes or "stage_requeue_ready" in diagnosis_codes(summaries.get("stage1_requeue", {})),
        "diagnosis_codes": sorted(codes),
    }


def summarize_live(swarm_payload: dict[str, Any]) -> dict[str, Any]:
    runtime = swarm_payload.get("runtime_classification") if isinstance(swarm_payload.get("runtime_classification"), dict) else {}
    lifecycle = swarm_payload.get("kaggle_lifecycle") if isinstance(swarm_payload.get("kaggle_lifecycle"), dict) else {}
    workload = swarm_payload.get("workload") if isinstance(swarm_payload.get("workload"), dict) else {}
    beta = swarm_payload.get("real_llm_internet_beta_summary")
    if not isinstance(beta, dict):
        beta = {}
    live_requeue = swarm_payload.get("live_requeue_summary")
    if not isinstance(live_requeue, dict):
        live_requeue = beta.get("live_requeue_summary") if isinstance(beta.get("live_requeue_summary"), dict) else {}
    return {
        "schema": swarm_payload.get("schema"),
        "ok": swarm_payload.get("ok"),
        "mode": swarm_payload.get("mode"),
        "live_mode": swarm_payload.get("live_mode"),
        "coordinator_url": swarm_payload.get("coordinator_url"),
        "diagnosis_codes": diagnosis_codes(swarm_payload),
        "runtime_classification": {
            "kaggle_auto": runtime.get("kaggle_auto"),
            "external_runtime_verified": runtime.get("external_runtime_verified"),
            "stage_requeue_verified": runtime.get("stage_requeue_verified"),
        },
        "kaggle_lifecycle": {
            "kernels_deleted": lifecycle.get("kernels_deleted"),
            "cleanup_required": lifecycle.get("cleanup_required"),
        },
        "workload": {
            "workload_type": workload.get("workload_type"),
            "stage_mode": workload.get("stage_mode"),
            "request_count": workload.get("request_count"),
            "hf_model_id": workload.get("hf_model_id"),
        },
        "live_requeue_summary": {
            "enabled": live_requeue.get("enabled"),
            "failure_mode": live_requeue.get("failure_mode"),
            "target_stage": live_requeue.get("target_stage"),
            "claim_observed": live_requeue.get("claim_observed"),
            "victim_kernel_deleted": live_requeue.get("victim_kernel_deleted"),
            "lease_expired": live_requeue.get("lease_expired"),
            "rescued_result": live_requeue.get("rescued_result"),
            "victim_result_accepted": live_requeue.get("victim_result_accepted"),
        },
        "beta_summary": beta,
    }


def public_session_summary(args: argparse.Namespace, *, live_payload: dict[str, Any], local_payload: dict[str, Any]) -> dict[str, Any]:
    live_summary = summarize_live(live_payload) if live_payload else {}
    local_summary = summarize_requeue(local_payload) if local_payload else {}
    live_codes = set(diagnosis_codes(live_payload))
    local_codes = set(diagnosis_codes(local_payload))
    return {
        "model_id": args.hf_model_id,
        "request_count": args.request_count,
        "decode_steps": 1,
        "stage_count": 2,
        "prompt_count": len(DEFAULT_PROMPTS),
        "workload_type": WORKLOAD_TYPE,
        "live_external_runtime_verified": "external_runtime_verified" in live_codes,
        "decoded_tokens_match": "decoded_tokens_match" in (live_codes | local_codes),
        "baseline_match": "baseline_match" in (live_codes | local_codes),
        "distinct_stage_miners": "distinct_stage_miners" in (live_codes | local_codes),
        "stage_assignment_valid": "stage_assignment_valid" in (live_codes | local_codes),
        "local_stage_requeue_verified": bool(local_summary.get("stage_requeue_verified")),
        "live_stage_requeue_verified": "external_stage_requeue_ready" in live_codes,
        "live_kaggle_kernels_deleted": "kaggle_kernels_deleted" in live_codes,
        "live_summary": live_summary,
        "local_requeue_summary": local_summary,
    }


def run_local_requeue(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    child_dir = output_dir / "local-generated"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "real_llm_internet_alpha_pack.py"),
        "--mode",
        MODE_LOCAL_GENERATED,
        "--output-dir",
        str(child_dir),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--port",
        str(args.base_port),
        "--base-port",
        str(args.base_port + 1),
        "--miner-id",
        f"{args.miner_id_prefix}-local",
        "--request-count",
        str(args.request_count),
        "--hf-model-id",
        args.hf_model_id,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--lease-seconds",
        str(args.lease_seconds),
        "--compute-seconds",
        str(args.compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        str(args.idle_sleep),
        "--max-request-attempts",
        str(args.max_request_attempts),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    return run_json_step(
        "real_llm_internet_alpha_local_requeue",
        command,
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), 300.0) + 300.0,
    )


def run_live_kaggle(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[dict[str, Any], dict[str, Any]]:
    child_dir = output_dir / "live-kaggle"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "swarm_inference_beta_pack.py"),
        "live",
        "--output-dir",
        str(child_dir),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--miner-id-prefix",
        args.miner_id_prefix,
        "--request-count",
        str(args.request_count),
        "--hf-model-id",
        args.hf_model_id,
        "--dataset-title",
        args.dataset_title,
        "--kernel-title-prefix",
        args.kernel_title_prefix,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--kaggle-push-timeout-seconds",
        str(args.kaggle_push_timeout_seconds),
        "--kaggle-delete-timeout-seconds",
        str(args.kaggle_delete_timeout_seconds),
        "--kaggle-status-timeout-seconds",
        str(args.kaggle_status_timeout_seconds),
        "--kaggle-status-poll-interval",
        str(args.kaggle_status_poll_interval),
        "--lease-seconds",
        str(args.lease_seconds),
        "--compute-seconds",
        str(args.compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        str(args.idle_sleep),
        "--max-request-attempts",
        str(args.max_request_attempts),
        "--failure-mode",
        args.failure_mode,
        "--victim-compute-seconds",
        str(args.victim_compute_seconds),
        "--claim-observe-timeout",
        str(args.claim_observe_timeout),
        "--requeue-timeout",
        str(args.requeue_timeout),
        "--json",
    ]
    if args.ready_url:
        command.extend(["--ready-url", args.ready_url])
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.kaggle_owner:
        command.extend(["--kaggle-owner", args.kaggle_owner])
    if args.dataset_slug:
        command.extend(["--dataset-slug", args.dataset_slug])
    if args.kernel_slug_prefix:
        command.extend(["--kernel-slug-prefix", args.kernel_slug_prefix])
    if args.inline_kernel_payload:
        command.append("--inline-kernel-payload")
    else:
        command.append("--no-inline-kernel-payload")
    if args.skip_kaggle_cleanup:
        command.append("--skip-kaggle-cleanup")
    if args.keep_live_private_artifacts:
        command.append("--keep-live-private-artifacts")
    return run_json_step(
        "swarm_inference_beta_live",
        command,
        runner=runner,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), 300.0) + 600.0,
    )


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []
    local_payload: dict[str, Any] = {}
    live_payload: dict[str, Any] = {}

    if not args.skip_local_requeue:
        local_step, local_payload = run_local_requeue(args, output_dir=output_dir, runner=runner)
        steps.append(local_step)
    else:
        steps.append({"name": "real_llm_internet_alpha_local_requeue", "ok": False, "skipped": True})

    if args.mode == MODE_LIVE_KAGGLE:
        live_step, live_payload = run_live_kaggle(args, output_dir=output_dir, runner=runner)
        steps.append(live_step)

    local_codes = set(diagnosis_codes(local_payload))
    live_codes = set(diagnosis_codes(live_payload))
    codes = set(local_codes | live_codes)
    local_requeue_ready = bool(
        not args.skip_local_requeue
        and local_payload.get("ok")
        and "real_llm_stage_requeue_ready" in local_codes
    )
    live_ready = args.mode == MODE_LOCAL_GENERATED or bool(
        live_payload.get("ok")
        and "swarm_inference_beta_live_ready" in live_codes
        and "external_runtime_verified" in live_codes
        and "kaggle_kernels_deleted" in live_codes
    )
    live_requeue_ready = args.mode == MODE_LOCAL_GENERATED or args.failure_mode == FAILURE_NONE or bool(
        "external_stage_requeue_ready" in live_codes
    )
    if local_requeue_ready:
        codes.update({"stage_requeue_ready", "local_stage_requeue_ready"})
    elif args.skip_local_requeue:
        codes.add("local_stage_requeue_skipped")
    else:
        codes.add("local_stage_requeue_blocked")
    if args.mode == MODE_LIVE_KAGGLE and live_ready:
        codes.update({"public_swarm_live_kaggle_ready", "external_runtime_verified"})
    if args.mode == MODE_LIVE_KAGGLE and live_requeue_ready and args.failure_mode != FAILURE_NONE:
        codes.update({"public_swarm_live_requeue_ready", "external_stage_requeue_ready"})
    elif args.mode == MODE_LIVE_KAGGLE and args.failure_mode != FAILURE_NONE:
        codes.add("public_swarm_live_requeue_blocked")
    ok = bool(local_requeue_ready and live_ready and live_requeue_ready)
    if ok:
        codes.update({"public_swarm_inference_alpha_ready", "public_swarm_session_ready"})
    else:
        codes.add("public_swarm_inference_alpha_blocked")

    session = public_session_summary(args, live_payload=live_payload, local_payload=local_payload)
    cleanup = {"child_artifacts_pruned": False}
    if not args.keep_child_artifacts:
        cleanup = prune_child_artifacts(output_dir)

    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "coordinator_url": args.coordinator_url.rstrip("/") if args.coordinator_url else f"http://{args.public_host}:{args.port}",
        "failure_mode": args.failure_mode,
        "session": session,
        "steps": steps,
        "payload_summaries": {
            "local_requeue": summarize_requeue(local_payload),
            "live_kaggle": summarize_live(live_payload),
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": {},
        "artifact_cleanup": cleanup,
        "safety": {
            "cpu_only": True,
            "read_only_workload": WORKLOAD_TYPE,
            "activation_payloads_redacted": True,
            "token_values_redacted": True,
            "child_artifacts_pruned": bool(cleanup.get("child_artifacts_pruned")),
            "local_requeue_required": not args.skip_local_requeue,
            "local_requeue_verified": local_requeue_ready,
            "live_kaggle_required": args.mode == MODE_LIVE_KAGGLE,
            "live_kaggle_verified": live_ready if args.mode == MODE_LIVE_KAGGLE else False,
            "live_requeue_required": args.mode == MODE_LIVE_KAGGLE and args.failure_mode != FAILURE_NONE,
            "live_requeue_verified": live_requeue_ready if args.mode == MODE_LIVE_KAGGLE else False,
            "kaggle_cleanup_required": args.mode == MODE_LIVE_KAGGLE,
            "token_rotation_required": args.mode == MODE_LIVE_KAGGLE,
            "not_production": True,
            "not_p2p": True,
            "not_gpu_tpu_pooling": True,
            "not_large_model_serving": True,
            "not_public_prompt_serving": True,
        },
        "operator_action": [
            "Rotate generated Coordinator and Miner tokens after every temporary public HTTP/Kaggle run.",
            "Use this Alpha as controlled evidence only; do not expose it as arbitrary prompt serving.",
        ],
        "limitations": [
            "Public Swarm Inference Alpha aggregates controlled tiny GPT split proofs; it is not production Swarm Inference.",
            "CPU-only read-only Hugging Face tiny GPT evidence; not P2P routing, NAT traversal, GPU/TPU pooling, GGUF/llama.cpp serving, large-model serving, training, payments, or staking.",
            "When --failure-mode is not none, live-kaggle Alpha verifies external victim/rescue lease requeue in addition to the local-generated requeue control path.",
        ],
    }
    refresh_artifacts(report, output_dir=output_dir, local_payload=local_payload, live_payload=live_payload)
    return persist_report(report, output_dir=output_dir)


def persist_report(report: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    report = support_bundle.sanitize(redact_values(report))
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"sensitive_output_detected"})
        report["safety_error"] = "public Swarm Inference Alpha report contained secret-like fragments"
    json_path = output_dir / "public_swarm_inference_alpha.json"
    md_path = output_dir / "public_swarm_inference_alpha.md"
    write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    if "artifacts" in report:
        report["artifacts"]["public_swarm_inference_alpha_json"]["present"] = True
        report["artifacts"]["public_swarm_inference_alpha_markdown"]["present"] = True
        write_json(json_path, report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    session = report.get("session") if isinstance(report.get("session"), dict) else {}
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    lines = [
        "# CrowdTensor Public Swarm Inference Alpha",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- coordinator_url: `{report.get('coordinator_url')}`",
        f"- model_id: `{session.get('model_id')}`",
        f"- request_count: `{session.get('request_count')}`",
        f"- external_runtime_verified: `{session.get('live_external_runtime_verified')}`",
        f"- local_stage_requeue_verified: `{session.get('local_stage_requeue_verified')}`",
        f"- decoded_tokens_match: `{session.get('decoded_tokens_match')}`",
        f"- distinct_stage_miners: `{session.get('distinct_stage_miners')}`",
        f"- kaggle_cleanup_required: `{safety.get('kaggle_cleanup_required')}`",
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`",
        "",
        "## Artifacts",
        "",
    ]
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        lines.append(f"- `{name}`: `{artifact.get('path')}` present=`{artifact.get('present')}`")
    lines.extend(["", "## Boundaries", ""])
    for limitation in report.get("limitations") or []:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def default_kaggle_owner() -> str:
    import os

    if os.environ.get("KAGGLE_USERNAME"):
        return str(os.environ["KAGGLE_USERNAME"])
    config = Path.home() / ".kaggle" / "kaggle.json"
    if config.is_file():
        try:
            payload = json.loads(config.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return ""
        username = payload.get("username")
        return str(username) if username else ""
    return ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Public Swarm Inference Alpha evidence.")
    parser.add_argument("--mode", choices=[MODE_LOCAL_GENERATED, MODE_LIVE_KAGGLE], default=MODE_LIVE_KAGGLE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--public-host", default=DEFAULT_PUBLIC_HOST)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT)
    parser.add_argument("--ready-url", default="")
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--miner-id-prefix", default="public-swarm-alpha")
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument("--hf-model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--failure-mode", choices=sorted(FAILURE_MODES), default=FAILURE_KILL_STAGE0_AFTER_CLAIM)
    parser.add_argument("--skip-local-requeue", action="store_true")
    parser.add_argument("--kaggle-owner", default=default_kaggle_owner())
    parser.add_argument("--dataset-slug", default="")
    parser.add_argument("--dataset-title", default="CrowdTensor Public Swarm Inference Alpha")
    parser.add_argument("--kernel-slug-prefix", default="crowdtensor-public-swarm-alpha")
    parser.add_argument("--kernel-title-prefix", default="CrowdTensor Public Swarm Inference Alpha")
    parser.add_argument("--inline-kernel-payload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-kaggle-cleanup", action="store_true")
    parser.add_argument("--keep-live-private-artifacts", action="store_true")
    parser.add_argument("--keep-child-artifacts", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--remote-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--process-exit-timeout", type=float, default=10.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--http-timeout", type=float, default=30.0)
    parser.add_argument("--kaggle-push-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--kaggle-status-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--kaggle-status-poll-interval", type=float, default=5.0)
    parser.add_argument("--lease-seconds", type=float, default=15.0)
    parser.add_argument("--compute-seconds", type=float, default=0.2)
    parser.add_argument("--victim-compute-seconds", type=float, default=45.0)
    parser.add_argument("--heartbeat-interval", type=float, default=0.1)
    parser.add_argument("--idle-sleep", type=float, default=0.2)
    parser.add_argument("--claim-observe-timeout", type=float, default=180.0)
    parser.add_argument("--requeue-timeout", type=float, default=120.0)
    parser.add_argument("--max-request-attempts", type=int, default=240)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.port < 1 or args.base_port < 1:
        raise SystemExit("--port and --base-port must be positive")
    if args.request_count < 1 or args.request_count > 4:
        raise SystemExit("--request-count must be between 1 and 4")
    for name in [
        "timeout_seconds",
        "remote_timeout_seconds",
        "startup_timeout",
        "process_exit_timeout",
        "poll_interval",
        "http_timeout",
        "kaggle_push_timeout_seconds",
        "kaggle_delete_timeout_seconds",
        "kaggle_status_timeout_seconds",
        "kaggle_status_poll_interval",
        "lease_seconds",
        "victim_compute_seconds",
        "heartbeat_interval",
        "idle_sleep",
        "claim_observe_timeout",
        "requeue_timeout",
    ]:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.compute_seconds < 0:
        raise SystemExit("--compute-seconds must be non-negative")
    if args.max_request_attempts < 1:
        raise SystemExit("--max-request-attempts must be at least 1")
    if args.failure_mode != FAILURE_NONE:
        if args.victim_compute_seconds <= args.lease_seconds:
            args.victim_compute_seconds = args.lease_seconds + 30.0
        if args.requeue_timeout <= args.lease_seconds:
            args.requeue_timeout = args.lease_seconds + 45.0
    if args.mode == MODE_LIVE_KAGGLE and not args.kaggle_owner:
        raise SystemExit("--kaggle-owner is required or KAGGLE_USERNAME/~/.kaggle/kaggle.json must be configured")
    return args


def print_human(report: dict[str, Any]) -> None:
    session = report.get("session") if isinstance(report.get("session"), dict) else {}
    print("CrowdTensor Public Swarm Inference Alpha")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  model: {session.get('model_id')}")
    print(f"  external runtime: {session.get('live_external_runtime_verified')}")
    print(f"  local requeue: {session.get('local_stage_requeue_verified')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print_human(report)
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
